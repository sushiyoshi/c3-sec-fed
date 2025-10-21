"""Core components for federated averaging with optional FHE aggregation."""
from __future__ import annotations

from dataclasses import dataclass
import time
import warnings
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class ClientUpdateResult:
    """Container for a single client's local training result."""

    state_dict: Dict[str, torch.Tensor]
    num_samples: int
    loss: float


class Aggregator:
    """Base interface for aggregating client model updates."""

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_states: Sequence[ClientUpdateResult],
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class PlaintextAggregator(Aggregator):
    """Average model parameters in plaintext."""

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_states: Sequence[ClientUpdateResult],
    ) -> Dict[str, torch.Tensor]:
        total_samples = sum(client.num_samples for client in client_states)
        if total_samples == 0:
            raise ValueError("Total number of samples across participating clients is zero.")

        averaged_state: Dict[str, torch.Tensor] = {}
        for key, reference in global_state.items():
            accumulator = torch.zeros_like(
                reference, dtype=torch.float32, device=torch.device("cpu")
            )
            for client in client_states:
                accumulator += client.state_dict[key].float() * (
                    client.num_samples / total_samples
                )
            if reference.dtype.is_floating_point:
                averaged_state[key] = accumulator.to(dtype=reference.dtype)
            else:
                averaged_state[key] = torch.round(accumulator).to(
                    dtype=reference.dtype
                )
        return averaged_state


class TFHEAggregator(Aggregator):
    """Aggregate encrypted client updates using Concrete-ML's TFHE backend."""

    def __init__(
        self,
        default_scale: float = 2**15,
        max_bit_width: int = 16,
        simulation_latency: float = 0.0,
    ) -> None:
        self.default_scale = default_scale
        self.max_bit_width = max_bit_width
        self._simulation_latency = simulation_latency
        self._simulated = False
        try:
            from concrete import fhe  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            warnings.warn(
                "concrete-ml is unavailable; falling back to a simulated TFHE aggregator.",
                RuntimeWarning,
            )
            self.fhe = None  # type: ignore[assignment]
            self._simulated = True
        else:
            self.fhe = fhe

        self._compiler_cache: Dict[Tuple[int, int], "fhe.Circuit"] = {}
        self._max_value_seen: Dict[Tuple[int, int], int] = {}

    def _build_circuit(self, num_clients: int, num_params: int, max_val: int):
        if self._simulated:
            raise RuntimeError("TFHE circuit compilation requested in simulation mode.")

        key = (num_clients, num_params)
        cached = self._compiler_cache.get(key)
        cached_bound = self._max_value_seen.get(key, 0)
        if cached is not None and max_val <= cached_bound:
            return cached

        def sum_updates(x):
            return np.sum(x, axis=0)

        assert self.fhe is not None
        compiler = self.fhe.Compiler(sum_updates, {"x": "encrypted"})
        bound = max(max_val, 1)
        sample_pos = np.full((num_clients, num_params), bound, dtype=np.int64)
        sample_neg = -sample_pos
        circuit = compiler.compile([sample_pos, sample_neg])
        self._compiler_cache[key] = circuit
        self._max_value_seen[key] = bound
        return circuit

    def _select_scale(
        self,
        tensors: Sequence[np.ndarray],
        total_samples: int,
    ) -> float:
        max_int = (1 << (self.max_bit_width - 1)) - 1
        max_abs = max((float(np.max(np.abs(t))) for t in tensors), default=0.0)
        if max_abs == 0.0:
            return min(self.default_scale, float(max_int))
        scale = min(self.default_scale, max_int / (max_abs * max(total_samples, 1)))
        return max(scale, 1.0)

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_states: Sequence[ClientUpdateResult],
    ) -> Dict[str, torch.Tensor]:
        if not client_states:
            return global_state
        total_samples = sum(client.num_samples for client in client_states)
        if total_samples == 0:
            raise ValueError("Total number of samples across participating clients is zero.")

        aggregated_state: Dict[str, torch.Tensor] = {}
        for key, reference in global_state.items():
            client_arrays: List[np.ndarray] = [
                client.state_dict[key].detach().cpu().numpy().astype(np.float64)
                for client in client_states
            ]
            scale = self._select_scale(client_arrays, total_samples)
            flattened = [arr.reshape(-1) for arr in client_arrays]
            num_params = flattened[0].shape[0]
            weighted_updates: List[np.ndarray] = []
            max_val = 0
            for arr, client in zip(flattened, client_states):
                scaled = np.round(arr * scale).astype(np.int64)
                scaled *= int(client.num_samples)
                max_val = max(max_val, int(np.max(np.abs(scaled))))
                weighted_updates.append(scaled)

            matrix = np.stack(weighted_updates, axis=0)
            if self._simulated:
                if self._simulation_latency > 0.0:
                    time.sleep(self._simulation_latency * len(client_states))
                summed = np.sum(matrix, axis=0)
            else:
                circuit = self._build_circuit(len(client_states), num_params, max_val)
                encrypted = circuit.encrypt(matrix)
                encrypted_sum = circuit.run(encrypted)
                summed = circuit.decrypt(encrypted_sum)
            averaged = summed.astype(np.float64) / (scale * total_samples)
            reshaped = averaged.reshape(reference.shape)
            tensor = torch.tensor(
                reshaped, dtype=torch.float32, device=torch.device("cpu")
            )
            if reference.dtype.is_floating_point:
                aggregated_state[key] = tensor.to(dtype=reference.dtype)
            else:
                aggregated_state[key] = torch.round(tensor).to(dtype=reference.dtype)
        return aggregated_state


class CKKSAggregator(Aggregator):
    """Aggregate encrypted client updates using OpenFHE's CKKS scheme."""

    def __init__(
        self,
        batch_size: int = 8192,
        multiplicative_depth: int = 2,
        scaling_mod_size: int = 59,
        simulation_latency: float = 0.0,
    ) -> None:
        self.batch_size = batch_size
        self._simulation_latency = simulation_latency
        self._simulated = False
        try:
            from openfhe import (  # type: ignore
                CCParamsCKKSRNS,
                GenCryptoContext,
                PKESchemeFeature,
                SecurityLevel,
            )
        except ImportError:  # pragma: no cover - optional dependency
            warnings.warn(
                "openfhe-python is unavailable; falling back to a simulated CKKS aggregator.",
                RuntimeWarning,
            )
            self.context = None  # type: ignore[assignment]
            self.keys = None  # type: ignore[assignment]
            self._simulated = True
        else:
            params = CCParamsCKKSRNS()
            params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
            params.SetMultiplicativeDepth(multiplicative_depth)
            params.SetScalingModSize(scaling_mod_size)
            params.SetBatchSize(batch_size)

            self.context = GenCryptoContext(params)
            self.context.Enable(PKESchemeFeature.PKE)
            self.context.Enable(PKESchemeFeature.KEYSWITCH)
            self.context.Enable(PKESchemeFeature.LEVELEDSHE)
            self.context.Enable(PKESchemeFeature.ADVANCEDSHE)

            self.keys = self.context.KeyGen()
            self.context.EvalMultKeyGen(self.keys.secretKey)
            self.context.EvalSumKeyGen(self.keys.secretKey)

    def _chunk_indices(self, length: int) -> Iterable[Tuple[int, int]]:
        for start in range(0, length, self.batch_size):
            end = min(start + self.batch_size, length)
            yield start, end

    def aggregate(
        self,
        global_state: Dict[str, torch.Tensor],
        client_states: Sequence[ClientUpdateResult],
    ) -> Dict[str, torch.Tensor]:
        if not client_states:
            return global_state

        total_samples = sum(client.num_samples for client in client_states)
        if total_samples == 0:
            raise ValueError("Total number of samples across participating clients is zero.")

        aggregated_state: Dict[str, torch.Tensor] = {}
        for key, reference in global_state.items():
            flattened_updates = [
                client.state_dict[key].detach().cpu().numpy().astype(np.float64).reshape(-1)
                for client in client_states
            ]
            length = flattened_updates[0].shape[0]
            aggregated_vector = np.zeros(length, dtype=np.float64)

            if self._simulated:
                if self._simulation_latency > 0.0:
                    time.sleep(self._simulation_latency * len(client_states))
                summed = sum(
                    update * float(client.num_samples)
                    for update, client in zip(flattened_updates, client_states)
                )
                averaged = summed / float(total_samples)
                aggregated_vector = averaged
            else:
                for start, end in self._chunk_indices(length):
                    ciphertext = None
                    for update, client in zip(flattened_updates, client_states):
                        chunk = update[start:end] * float(client.num_samples)
                        padded = np.zeros(self.batch_size, dtype=np.float64)
                        padded[: end - start] = chunk
                        plaintext = self.context.MakeCKKSPackedPlaintext(padded.tolist())
                        current = self.context.Encrypt(self.keys.publicKey, plaintext)
                        ciphertext = (
                            current
                            if ciphertext is None
                            else self.context.EvalAdd(ciphertext, current)
                        )
                    assert ciphertext is not None
                    decrypted = self.context.Decrypt(self.keys.secretKey, ciphertext)
                    decrypted.SetLength(self.batch_size)
                    values = np.array(decrypted.GetRealPackedValue())[: end - start]
                    aggregated_vector[start:end] = values / float(total_samples)

            reshaped = aggregated_vector.reshape(reference.shape)
            tensor = torch.tensor(
                reshaped, dtype=torch.float32, device=torch.device("cpu")
            )
            if reference.dtype.is_floating_point:
                aggregated_state[key] = tensor.to(dtype=reference.dtype)
            else:
                aggregated_state[key] = torch.round(tensor).to(dtype=reference.dtype)
        return aggregated_state


__all__ = [
    "Aggregator",
    "ClientUpdateResult",
    "PlaintextAggregator",
    "TFHEAggregator",
    "CKKSAggregator",
]
