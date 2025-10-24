# Federated Learning with CKKS Homomorphic Encryption

CIFAR-10での連合学習とCKKS準同型暗号を使用したセキュア連合学習の比較実装。

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

OpenFHE-Pythonのインストールが必要:
```bash
git clone https://github.com/openfheorg/openfhe-python.git
cd openfhe-python
# インストール手順はopenfhe-pythonのREADMEを参照
```

## Usage

### 平文とCKKS暗号化の連合学習比較
```bash
python federated.py --clients 5 --rounds 5
```

### 基本的な連合学習（平文のみ）
```bash
python federated_cifar10.py --rounds 20 --clients 10
```

### CKKSテスト
```bash
python ckks_chebyshev.py
```

## Files

- `federated.py`: CKKS準同型暗号を使用したセキュア連合学習の実装
- `federated_cifar10.py`: 平文での基本的な連合学習実装
- `ckks_chebyshev.py`: OpenFHE CKKSのブートストラッピングテスト
