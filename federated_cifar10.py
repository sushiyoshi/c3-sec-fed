import argparse
import copy
import os
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from fedavg_core import (
    Aggregator,
    ClientUpdateResult,
    CKKSAggregator,
    PlaintextAggregator,
    TFHEAggregator,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SimpleCIFARNet(nn.Module):
    """Compact CNN that works well on CIFAR-10 without being too heavy."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    return train_transform, test_transform


def iid_partition(indices: Sequence[int], num_clients: int, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(indices)
    rng.shuffle(shuffled)
    splits = np.array_split(shuffled, num_clients)
    return [split.tolist() for split in splits]


def dirichlet_partition(
    labels: Sequence[int], num_clients: int, alpha: float, seed: int
) -> List[List[int]]:
    """Sample a non-iid partition using per-class Dirichlet distribution."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    num_classes = int(labels.max()) + 1

    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        class_indices = np.where(class_mask)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, proportions)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])
    return client_indices


def build_client_loaders(
    dataset: datasets.CIFAR10,
    num_clients: int,
    batch_size: int,
    iid: bool,
    alpha: float,
    seed: int,
    max_examples_per_client: int | None = None,
) -> List[DataLoader]:
    indices = list(range(len(dataset)))
    if iid:
        partitions = iid_partition(indices, num_clients, seed)
    else:
        labels_source = None
        if hasattr(dataset, "targets"):
            labels_source = dataset.targets
        elif hasattr(dataset, "labels"):
            labels_source = dataset.labels
        if labels_source is None:
            rng = np.random.default_rng(seed)
            labels_source = rng.integers(low=0, high=10, size=len(dataset))
        partitions = dirichlet_partition(labels_source, num_clients, alpha, seed)

    client_loaders: List[DataLoader] = []
    for partition in partitions:
        if max_examples_per_client is not None:
            partition = partition[:max_examples_per_client]
        subset = Subset(dataset, partition)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        client_loaders.append(loader)
    return client_loaders


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == targets).sum().item()
            total_examples += inputs.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


def local_update(
    base_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> ClientUpdateResult:
    model = copy.deepcopy(base_model)
    model.train()
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    for _ in range(epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return ClientUpdateResult(cpu_state, total_samples, avg_loss)


def build_aggregator(args: argparse.Namespace) -> Aggregator:
    mode = args.aggregation_mode.lower()
    if mode == "plaintext":
        return PlaintextAggregator()
    if mode == "tfhe":
        return TFHEAggregator(
            default_scale=args.tfhe_scaling,
            max_bit_width=args.tfhe_bit_width,
        )
    if mode == "ckks":
        return CKKSAggregator(
            batch_size=args.ckks_batch_size,
            multiplicative_depth=args.ckks_depth,
            scaling_mod_size=args.ckks_scaling_mod_size,
        )
    raise ValueError(f"Unsupported aggregation mode: {args.aggregation_mode}")


def run_federated_training(args: argparse.Namespace) -> Dict[str, List[float]]:
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if not args.quiet:
        print(f"Using device: {device}")

    train_transform, test_transform = build_transforms()
    if getattr(args, "use_fake_data", False):
        train_dataset = datasets.FakeData(
            size=args.fake_train_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_transform,
        )
        test_dataset = datasets.FakeData(
            size=args.fake_test_size,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=test_transform,
        )
    else:
        train_dataset = datasets.CIFAR10(
            args.data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            args.data_dir, train=False, download=True, transform=test_transform
        )

    client_loaders = build_client_loaders(
        train_dataset,
        num_clients=args.clients,
        batch_size=args.batch_size,
        iid=args.iid,
        alpha=args.alpha,
        seed=args.seed,
        max_examples_per_client=args.max_examples_per_client,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    global_model = SimpleCIFARNet().to(device)
    global_state = {
        key: tensor.detach().cpu() for key, tensor in global_model.state_dict().items()
    }
    loss_fn = nn.CrossEntropyLoss()

    aggregator = build_aggregator(args)

    clients_per_round = (
        args.clients_per_round if args.clients_per_round else args.clients
    )
    clients_per_round = min(clients_per_round, args.clients)

    history = {"round": [], "test_loss": [], "test_accuracy": [], "round_duration": []}
    for round_idx in range(1, args.rounds + 1):
        round_start = time.perf_counter()
        participating_clients = random.sample(
            range(args.clients), clients_per_round
        )
        client_updates: List[ClientUpdateResult] = []
        round_losses = []

        for client_id in participating_clients:
            base_model = SimpleCIFARNet()
            base_model.load_state_dict(global_state)
            update = local_update(
                base_model,
                train_loader=client_loaders[client_id],
                device=device,
                epochs=args.local_epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
            client_updates.append(update)
            round_losses.append(update.loss)

        global_state = aggregator.aggregate(global_state, client_updates)
        global_model.load_state_dict(global_state)
        test_loss, test_accuracy = evaluate(
            global_model, test_loader, device, loss_fn
        )

        history["round"].append(round_idx)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        history["round_duration"].append(time.perf_counter() - round_start)

        avg_client_loss = sum(round_losses) / max(len(round_losses), 1)
        if not args.quiet:
            print(
                f"Round {round_idx:03d}: "
                f"participating clients={len(participating_clients)}, "
                f"client_loss={avg_client_loss:.4f}, "
                f"test_loss={test_loss:.4f}, "
                f"test_accuracy={test_accuracy * 100:.2f}%"
            )

    if not args.quiet:
        print("Training complete.")
    best_accuracy = max(history["test_accuracy"])
    best_round = history["test_accuracy"].index(best_accuracy) + 1
    if not args.quiet:
        print(
            f"Best accuracy: {best_accuracy * 100:.2f}% at round {best_round} "
            f"out of {args.rounds} rounds."
        )
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Federated Averaging with CIFAR-10 dataset."
    )
    parser.add_argument("--rounds", type=int, default=20, help="Number of global rounds.")
    parser.add_argument("--clients", type=int, default=10, help="Total number of clients.")
    parser.add_argument(
        "--clients-per-round",
        type=int,
        default=None,
        help="How many clients participate each round. Defaults to all clients.",
    )
    parser.add_argument(
        "--local-epochs", type=int, default=1, help="Local training epochs per round."
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per client.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for local updates.")
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD optimizer."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="Weight decay for SGD optimizer.",
    )
    parser.add_argument(
        "--max-examples-per-client",
        type=int,
        default=None,
        help="Optional cap on the number of training samples assigned to each client.",
    )
    parser.add_argument(
        "--iid",
        action="store_true",
        help="Use IID data partitioning instead of Dirichlet distribution.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet concentration parameter for non-IID splits (ignored in IID mode).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.getcwd(), "data"),
        help="Directory used to download/cache the CIFAR-10 dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (e.g. 'cuda', 'cpu'). Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--aggregation-mode",
        type=str,
        default="plaintext",
        choices=["plaintext", "tfhe", "ckks"],
        help="Averaging mode to use for server-side aggregation.",
    )
    parser.add_argument(
        "--use-fake-data",
        action="store_true",
        help="Use synthetic CIFAR-like data instead of downloading the real dataset.",
    )
    parser.add_argument(
        "--fake-train-size",
        type=int,
        default=1024,
        help="Number of synthetic training examples when --use-fake-data is set.",
    )
    parser.add_argument(
        "--fake-test-size",
        type=int,
        default=256,
        help="Number of synthetic test examples when --use-fake-data is set.",
    )
    parser.add_argument(
        "--tfhe-bit-width",
        type=int,
        default=16,
        dest="tfhe_bit_width",
        help="Maximum ciphertext bit-width used when quantizing TFHE ciphertexts.",
    )
    parser.add_argument(
        "--tfhe-scaling",
        type=float,
        default=2**15,
        dest="tfhe_scaling",
        help="Default floating-point scaling factor before TFHE encryption.",
    )
    parser.add_argument(
        "--ckks-batch-size",
        type=int,
        default=8192,
        help="Number of packed slots for CKKS plaintexts.",
    )
    parser.add_argument(
        "--ckks-depth",
        type=int,
        default=2,
        help="Multiplicative depth for the CKKS crypto context.",
    )
    parser.add_argument(
        "--ckks-scaling-mod-size",
        type=int,
        default=59,
        help="Scaling modulus size used when generating the CKKS context.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-round logging (useful for benchmarking).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_federated_training(args)


if __name__ == "__main__":
    main()
