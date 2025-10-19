# Federated Learning on CIFAR-10

This project provides a minimal implementation of the Federated Averaging (FedAvg) algorithm on the CIFAR-10 image classification dataset using PyTorch.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

Example for a quick smoke test with fewer rounds and clients:

```bash
python federated_cifar10.py --rounds 2 --clients 4 --clients-per-round 2 --local-epochs 1
```

The script downloads CIFAR-10 on the first run (into `./data` by default), performs federated training, and prints round-by-round metrics together with the best test accuracy observed.
