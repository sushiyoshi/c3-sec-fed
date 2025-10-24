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

## Output Files

`federated.py`を実行すると以下の5つのグラフが生成される:

### 1. `federated_learning_comparison_clients{N}_rounds{R}.png`
平文とCKKS暗号化の全体比較。上段は精度推移（2本の折れ線が重なるほど暗号化による劣化が少ない）、下段は実行時間（CKKS版のオーバーヘッドを示す）。

### 2-3. `sample_predictions_plain/ckks_clients{N}.png`
ランダムに選んだ16枚の画像とその分類結果。緑=正解、赤=誤分類。ダミー実装でないことの視覚的証明と、平文版とCKKS版の予測傾向の比較に使用。

### 4-5. `confusion_matrix_plain/ckks_clients{N}.png`
10x10の混同行列。対角線上の数値が正解数、対角線外が誤分類数。平文版とCKKS版で対角線の値が近ければ暗号化の影響は小さい。特定クラスペアでの誤分類傾向（例: cat→dog）を確認可能。

## Files

- `federated.py`: CKKS準同型暗号を使用したセキュア連合学習の実装
- `federated_cifar10.py`: 平文での基本的な連合学習実装
- `ckks_chebyshev.py`: OpenFHE CKKSのブートストラッピングテスト
