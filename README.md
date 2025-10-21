# Federated Learning on CIFAR-10

This project provides a minimal implementation of the Federated Averaging (FedAvg) algorithm on the CIFAR-10 image classification dataset using PyTorch. It also demonstrates homomorphic-encryption-protected aggregation using both TFHE (via [concrete-ml](https://github.com/zama-ai/concrete-ml)) and CKKS (via [openfhe-python](https://github.com/openfheorg/openfhe-python)), together with benchmarking utilities for comparing plaintext versus encrypted aggregation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The base requirements cover PyTorch and utility libraries. To run the encrypted aggregation modes install the optional dependencies when available:

```bash
pip install concrete-ml
pip install openfhe
```

> **Note:** Both concrete-ml and openfhe provide CPU-only wheels. Installing them inside a clean virtual environment avoids dependency conflicts. Refer to the respective project documentation if you need GPU acceleration or custom compilation flags. When either package cannot be installed the code automatically falls back to numerically equivalent simulated aggregators so that the benchmarking harness remains usable.

## Usage

Run federated training with default parameters:

```bash
python federated_cifar10.py
```

Useful options:

- `--rounds`: number of global aggregation rounds (default: 20)
- `--clients`: total number of simulated clients (default: 10)
- `--clients-per-round`: clients sampled in each round (default: all)
- `--local-epochs`: local epochs per client per round (default: 1)
- `--iid`: use IID data split instead of the default Dirichlet non-IID split
- `--alpha`: Dirichlet concentration for non-IID split (default: 0.5)
- `--device`: force computation device, e.g. `cuda` or `cpu`
- `--aggregation-mode`: choose `plaintext`, `tfhe`, or `ckks` for the server-side averaging step
- `--tfhe-bit-width` / `--tfhe-scaling`: control TFHE quantisation (increase bit width or scaling for more precision at the cost of circuit size)
- `--ckks-batch-size`, `--ckks-depth`, `--ckks-scaling-mod-size`: parameters for the CKKS crypto-context
- `--quiet`: suppress per-round logging (useful for scripted benchmarks)
- `--max-examples-per-client`: optional cap on client dataset size (handy for quick experiments)
- `--use-fake-data`: swap CIFAR-10 for a synthetic dataset when downloads are not possible
- `--fake-train-size` / `--fake-test-size`: sizes of the synthetic datasets when `--use-fake-data` is set

Example for a quick smoke test with fewer rounds, fewer clients, and trimmed client datasets:

```bash
python federated_cifar10.py --rounds 2 --clients 4 --clients-per-round 2 --local-epochs 1 \
  --max-examples-per-client 128 --use-fake-data
```

The script downloads CIFAR-10 on the first run (into `./data` by default), performs federated training, and prints round-by-round metrics together with the best test accuracy observed.

### Benchmarking plaintext vs. encrypted aggregation

The repository ships with `benchmark_encrypted_fedavg.py`, which executes three training runs (plaintext, TFHE, CKKS), measures accuracy and runtime, and visualises the differences. Pillow is used for plotting, so no heavyweight plotting backend is required.

```bash
python benchmark_encrypted_fedavg.py --rounds 3 --clients 4 --clients-per-round 2 --local-epochs 1 \
  --max-examples-per-client 128 --use-fake-data --tfhe-bit-width 16 --tfhe-scaling 32768 --ckks-batch-size 4096
```

The script stores comparison plots in `./benchmark_artifacts` by default. Three figures are produced:

1. Plaintext vs. TFHE accuracy/runtime
2. Plaintext vs. CKKS accuracy/runtime
3. Plaintext vs. TFHE vs. CKKS accuracy/runtime

The plaintext baseline should remain the accuracy ceiling. A significantly higher accuracy for an encrypted run usually indicates a configuration issue (e.g., inconsistent quantisation or scaling). When optional encryption libraries are unavailable the simulated backends still perform fixed-point quantisation, so accuracy trends remain representative even though true FHE latency overhead is not measured.
