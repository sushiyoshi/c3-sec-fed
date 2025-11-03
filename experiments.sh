#!/bin/bash

# 実験1: ベースライン
echo "=== Experiment 1: Baseline (FedAvg) ==="
python federated.py --server-opt fedavg --bn-mode fedavg --clients 10 --rounds 10 --local-epochs 2 --seed 42

# 実験2: FedAvgM（修正後）
echo "=== Experiment 2: FedAvgM (Fixed) ==="
python federated.py --server-opt fedavgm --bn-mode fedavg --clients 10 --rounds 10 --local-epochs 2 --seed 42

# 実験3: FedAdam（修正後）
echo "=== Experiment 3: FedAdam (Fixed) ==="
python federated.py --server-opt fedadam --bn-mode fedavg --clients 10 --rounds 10 --local-epochs 2 --seed 42

# 実験4: FedBN+FedAvg
echo "=== Experiment 4: FedBN + FedAvg ==="
python federated.py --server-opt fedavg --bn-mode fedbn --clients 10 --rounds 10 --local-epochs 2 --seed 42

# 実験5: FedBN+FedAvgM
echo "=== Experiment 5: FedBN + FedAvgM ==="
python federated.py --server-opt fedavgm --bn-mode fedbn --clients 10 --rounds 10 --local-epochs 2 --seed 42

# 実験6: 少クライアント
echo "=== Experiment 6: Few Clients (5) ==="
python federated.py --server-opt fedavg --bn-mode fedavg --clients 5 --rounds 10 --local-epochs 2 --seed 42

# 実験7: 多クライアント
echo "=== Experiment 7: Many Clients (20) ==="
python federated.py --server-opt fedavg --bn-mode fedavg --clients 20 --rounds 10 --local-epochs 2 --seed 42

# 実験8: 長期学習
echo "=== Experiment 8: Long Training (20 rounds) ==="
python federated.py --server-opt fedavg --bn-mode fedavg --clients 10 --rounds 20 --local-epochs 2 --seed 42

# 実験9: ローカルエポック増
echo "=== Experiment 9: More Local Epochs (5) ==="
python federated.py --server-opt fedavg --bn-mode fedavg --clients 10 --rounds 10 --local-epochs 5 --seed 42

# 実験10: 最適構成
echo "=== Experiment 10: Optimal Configuration ==="
python federated.py --server-opt fedavgm --bn-mode fedavg --clients 20 --rounds 20 --local-epochs 5 --seed 42

echo "All experiments completed!"
