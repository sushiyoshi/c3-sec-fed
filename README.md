# Federated Learning with CKKS Homomorphic Encryption

CIFAR-10での連合学習とCKKS準同型暗号を使用したセキュア連合学習の比較実装。

## Requirements

- Python 3.8+
- CUDA対応GPU推奨（CPUでも実行可能だが時間がかかる）
- OpenFHE-Python（CKKS準同型暗号ライブラリ）

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

**基本的な実行（FedAvg、デフォルト）:**
```bash
python federated.py --clients 5 --rounds 5
```

**FedBNを使用する場合:**
```bash
python federated.py --clients 5 --rounds 5 --bn-mode fedbn
```

**IIDデータ分割を使用する場合:**
```bash
python federated.py --clients 5 --rounds 5 --iid
```

**全オプション:**
```bash
python federated.py \
  --clients 10 \          # クライアント数（デフォルト: 10）
  --rounds 10 \           # 連合学習のラウンド数（デフォルト: 10）
  --batch-size 64 \       # バッチサイズ（デフォルト: 64）
  --local-epochs 1 \      # ローカルエポック数（デフォルト: 1）
  --lr 0.01 \             # 学習率（デフォルト: 0.01）
  --iid \                 # IIDデータ分割（デフォルト: non-IID）
  --alpha 0.5 \           # Dirichletパラメータ（non-IID時、デフォルト: 0.5）
  --seed 42 \             # 乱数シード（デフォルト: 42）
  --bn-mode fedavg        # BatchNorm集約方法: fedavg or fedbn（デフォルト: fedavg）
```

### コマンドラインオプション詳細

#### 基本設定
- `--clients N`: 連合学習に参加するクライアント数（デフォルト: 10）
  - クライアント数が多いほど現実的だが、計算時間が増加

- `--rounds R`: 連合学習のラウンド数（デフォルト: 10）
  - 各ラウンドで全クライアントがローカル学習→サーバが集約を実施
  - ラウンド数を増やすと精度向上の可能性があるが、時間がかかる

#### 学習パラメータ
- `--batch-size N`: ミニバッチサイズ（デフォルト: 64）
  - 大きいほどGPUメモリを多く使用、小さいほど学習が不安定になる可能性

- `--local-epochs N`: 各クライアントのローカル学習エポック数（デフォルト: 1）
  - 大きいほど各ラウンドでの学習が進むが、過学習のリスクも増加

- `--lr FLOAT`: 学習率（デフォルト: 0.01）
  - SGDオプティマイザの学習率
  - 大きすぎると不安定、小さすぎると収束が遅い

#### データ分割方法
- `--iid`: IID（独立同分布）データ分割を使用（フラグ）
  - 指定しない場合はnon-IID（Dirichlet分布）を使用
  - **IID**: 各クライアントが全クラスをバランスよく持つ（理想的だが非現実的）
  - **non-IID**: 各クライアントのデータ分布が偏る（現実的なシナリオ）

- `--alpha FLOAT`: Dirichlet分布のαパラメータ（デフォルト: 0.5、non-IID時のみ有効）
  - **小さい値（0.1など）**: データ分布が極端に偏る（各クライアントが特定クラスのみ）
  - **大きい値（10など）**: IIDに近い均一な分布
  - **推奨値**: 0.1〜1.0（現実的なnon-IIDシナリオ）

#### BatchNorm集約方法
- `--bn-mode {fedavg,fedbn}`: BatchNorm統計量の集約方法（デフォルト: fedavg）
  - **fedavg**: BatchNormの統計量（running_mean/running_var）も集約
    - 全クライアントのデータ分布が似ている場合に有効
  - **fedbn**: BatchNormの統計量は各クライアントが独自に保持
    - non-IIDデータで各クライアントのデータ分布が大きく異なる場合に有効
    - 論文: [FedBN: Federated Learning on Non-IID Features via Local Batch Normalization](https://arxiv.org/abs/2102.07623)

#### その他
- `--seed N`: 乱数シード（デフォルト: 42）
  - 再現性のため固定値を推奨

- `--data-dir PATH`: CIFAR-10データセットの保存先（デフォルト: ./data）

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

### 1. `federated_learning_comparison_{bn_mode}_clients{N}_rounds{R}.png`
平文とCKKS暗号化の全体比較。上段は精度推移（2本の折れ線が重なるほど暗号化による劣化が少ない）、下段は実行時間（CKKS版のオーバーヘッドを示す）。

### 2-3. `sample_predictions_plain/ckks_{bn_mode}_clients{N}.png`
ランダムに選んだ16枚の画像とその分類結果。緑=正解、赤=誤分類。ダミー実装でないことの視覚的証明と、平文版とCKKS版の予測傾向の比較に使用。

### 4-5. `confusion_matrix_plain/ckks_{bn_mode}_clients{N}.png`
10x10の混同行列。対角線上の数値が正解数、対角線外が誤分類数。平文版とCKKS版で対角線の値が近ければ暗号化の影響は小さい。特定クラスペアでの誤分類傾向（例: cat→dog）を確認可能。

**注:** `{bn_mode}`は`fedavg`または`fedbn`、`{N}`はクライアント数、`{R}`はラウンド数

## Files

- `federated.py`: CKKS準同型暗号を使用したセキュア連合学習の実装
- `federated_cifar10.py`: 平文での基本的な連合学習実装
- `ckks_chebyshev.py`: OpenFHE CKKSのブートストラッピングテスト
