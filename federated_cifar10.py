import argparse
import copy
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


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


@dataclass
class ClientUpdateResult:
    state_dict: Dict[str, torch.Tensor]
    num_samples: int
    loss: float


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
) -> List[DataLoader]:
    indices = list(range(len(dataset)))
    if iid:
        partitions = iid_partition(indices, num_clients, seed)
    else:
        partitions = dirichlet_partition(dataset.targets, num_clients, alpha, seed)

    client_loaders: List[DataLoader] = []
    for partition in partitions:
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


def average_state_dicts(
    global_state: Dict[str, torch.Tensor],
    client_states: Sequence[ClientUpdateResult],
) -> Dict[str, torch.Tensor]:
    total_samples = sum(client.num_samples for client in client_states)
    if total_samples == 0:
        raise ValueError("Total number of samples across participating clients is zero.")

    averaged_state = {}
    for key, val in global_state.items():
        # Skip averaging for integer buffers like num_batches_tracked
        if val.dtype in [torch.int32, torch.int64, torch.long]:
            averaged_state[key] = val.clone()
        else:
            averaged_state[key] = torch.zeros_like(val, device=torch.device("cpu"))

    for client in client_states:
        weight = client.num_samples / total_samples
        for key in averaged_state:
            # Only average non-integer tensors
            if averaged_state[key].dtype not in [torch.int32, torch.int64, torch.long]:
                averaged_state[key] += client.state_dict[key] * weight

    return averaged_state


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


def run_federated_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_transform, test_transform = build_transforms()
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

    clients_per_round = (
        args.clients_per_round if args.clients_per_round else args.clients
    )
    clients_per_round = min(clients_per_round, args.clients)

    history = {"round": [], "test_loss": [], "test_accuracy": []}
    for round_idx in range(1, args.rounds + 1):
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

        global_state = average_state_dicts(global_state, client_updates)
        global_model.load_state_dict(global_state)
        test_loss, test_accuracy = evaluate(
            global_model, test_loader, device, loss_fn
        )

        history["round"].append(round_idx)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        avg_client_loss = sum(round_losses) / max(len(round_losses), 1)
        print(
            f"Round {round_idx:03d}: "
            f"participating clients={len(participating_clients)}, "
            f"client_loss={avg_client_loss:.4f}, "
            f"test_loss={test_loss:.4f}, "
            f"test_accuracy={test_accuracy * 100:.2f}%"
        )

    print("Training complete.")
    best_accuracy = max(history["test_accuracy"])
    best_round = history["test_accuracy"].index(best_accuracy) + 1
    print(
        f"Best accuracy: {best_accuracy * 100:.2f}% at round {best_round} "
        f"out of {args.rounds} rounds."
    )


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_federated_training(args)


if __name__ == "__main__":
    main()
