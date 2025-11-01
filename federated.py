# =========================================================
# 🔒 OpenFHE CKKSを使用したセキュア連合学習システム
# =========================================================
#
# 【プログラムの概要】
# このプログラムは、CIFAR-10画像データセットを使用して、以下2つの連合学習を比較します：
#   1. 平文での連合学習（Plain Federated Learning）
#   2. CKKS準同型暗号を使用した連合学習（Encrypted Federated Learning）
#
# 【主要な特徴】
#   - OpenFHE CKKSライブラリを使用した準同型暗号化
#   - 各クライアントのモデル重みを暗号化したまま集約可能
#   - CIFAR-10データセットでの実際の画像分類を実行
#   - 平文と暗号化の精度・実行時間を比較
#
# 【処理の流れ】
#   1. データセットの準備とクライアント間での分割
#   2. 各クライアントがローカルデータで学習
#   3. サーバが各クライアントのモデルを集約（平文 or 暗号化）
#   4. グローバルモデルの精度を評価
#   5. 結果の可視化とサマリー出力
#
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import threading
import time
import copy
from math import log2, ceil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUIなし環境対応
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# [OPENFHE-CKKS] 追加
from openfhe import *


# =========================
# ユーティリティ関数
# =========================
def _next_pow2(n: int) -> int:
    """
    与えられた数値n以上の最小の2の累乗を返す
    CKKS暗号化のバッチサイズは2の累乗である必要があるため使用

    例: n=100 → 128, n=1000 → 1024
    """
    return 1 << ceil(log2(max(1, n)))


# =========================================================
# 🔒 CKKS準同型暗号を使用したモデル集約クラス
# =========================================================
# 【役割】
# 複数のクライアントから送られてきたモデルの重みを、暗号化したまま平均化する
#
# 【処理フロー】
#   1. サーバ側で暗号コンテキストと鍵ペアを生成（初期化時）
#   2. 各クライアントの重みをレイヤごとに暗号化
#   3. 暗号化されたまま加算（EvalAdd）
#   4. 平均化のため 1/N を乗算（EvalMult）
#   5. 復号して平均化された重みを取得
#
# 【CKKS暗号の特徴】
#   - 浮動小数点数の近似計算が可能な準同型暗号方式
#   - 暗号化されたまま加算・乗算が可能
#   - バッチ処理により複数の値を一つの暗号文で扱える
# =========================================================
class FHEModelAggregator:
    """
    役割:
      - 各レイヤ重みの形状を記録
      - CKKSコンテキストを 1 つ準備（BatchSize は最大要素数に合わせる）
      - クライアント重みをレイヤごとに flatten → CKKS で pack & encrypt
      - サーバ側で暗号加算 & 定数乗算（1/num_clients）→ 復号 → 形状に戻して返す
    """

    def __init__(
        self,
        model_structure,
        num_clients=5,
        # 以下2つは Concrete-ML 版の残置引数（外部インタフェース維持のため受け取るが無視）
        scale_factor=100,
        max_value=50,
        # CKKS 暗号パラメータ
        mult_depth: int = 1,           # 乗算深度（加算と定数乗算のみなので1で十分）
        scale_mod_size: int = 50,      # スケーリング係数のビット数（精度に影響）
        security_level: SecurityLevel = SecurityLevel.HEStd_128_classic,  # セキュリティレベル
        bn_mode: str = 'fedavg',       # 'fedavg' or 'fedbn'
    ):
        self.num_clients = num_clients
        self.bn_mode = bn_mode

        # ========================================
        # ステップ1: モデル構造の解析
        # ========================================
        # モデルから「浮動小数のみ」の形状を収集（パラメータ＋BN等のfloatバッファ）
        self.weight_shapes = {}  # name -> torch.Size
        max_elems = 0  # 最大要素数を記録（バッチサイズ決定に使用）

        for name, param in model_structure.named_parameters():
            self.weight_shapes[name] = param.shape
            max_elems = max(max_elems, int(np.prod(param.shape)))

        for name, buffer in model_structure.named_buffers():
            if hasattr(buffer, "dtype") and buffer.dtype.is_floating_point:
                if self.bn_mode == 'fedbn' and (name.endswith('running_mean') or name.endswith('running_var')):
                    continue  # FedBN: 集約対象から除外
                self.weight_shapes[name] = buffer.shape
                max_elems = max(max_elems, int(np.prod(buffer.shape)))

        # ========================================
        # ステップ2: CKKS暗号パラメータの設定
        # ========================================
        # バッチサイズは2の累乗である必要がある（CKKS暗号の仕様）
        # また、リングサイズの半分以下に制限される
        desired_batch_size = _next_pow2(max_elems)

        # CKKS暗号コンテキストのパラメータを設定
        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(mult_depth)     # 乗算の深さ（和と定数乗算のみなので1）
        params.SetScalingModSize(scale_mod_size)      # 精度を決定するパラメータ
        params.SetSecurityLevel(security_level)        # セキュリティレベル（128ビット）

        # 一時的なコンテキストを作成してリングサイズを確認
        temp_cc = GenCryptoContext(params)
        ring_dim = temp_cc.GetRingDimension()         # リング多項式の次元
        max_batch_size = ring_dim // 2                 # バッチサイズの上限

        # バッチサイズを制約内に調整
        self.batch_size = min(desired_batch_size, max_batch_size)
        print(f"🔐 CKKS parameters: desired_batch_size={desired_batch_size}, ring_dim={ring_dim}, max_batch_size={max_batch_size}")
        print(f"🔐 Using batch_size={self.batch_size}")

        # ========================================
        # ステップ3: 暗号コンテキストと鍵の生成
        # ========================================
        # 実際に使用するバッチサイズで暗号コンテキストを作成
        params.SetBatchSize(self.batch_size)
        self.cc = GenCryptoContext(params)
        # 必要な機能のみ有効化（定数乗算＋加算のみ）
        self.cc.Enable(PKESchemeFeature.PKE)
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE)

        # 鍵生成（EvalMultKeyGen は使わない）
        self.keys = self.cc.KeyGen()

        print(f"🔐 CKKS ready: ring dimension = {self.cc.GetRingDimension()}, batch_size = {self.batch_size}")

        self.decrypt_calls = 0
        # 互換ダミー（未使用）
        self.circuits = {}

    def get_public_key(self):
        """クライアント暗号化用の公開鍵を返す。"""
        return self.keys.publicKey

    # （サーバ側暗号化APIは廃止：クライアント側で暗号化する）

    def aggregate_encrypted_models(self, enc_payloads, bn_mode='fedavg'):
        """
        クライアント側で暗号化された重みを暗号領域のまま加重平均し、復号して返す。

        enc_payloads: List of tuples (encrypted_state_dict, int_buffers_plain, sample_count)
                      encrypted_state_dict: {name: [Ciphertext,...]}  ※float項目のみ
                      int_buffers_plain: {name: torch.Tensor(int系)}   ※先頭クライアントを採用
        bn_mode: 'fedavg' or 'fedbn'
                 'fedbn'の場合、BNのrunning_mean/running_varは集約せず先頭クライアントの値を使用

        Returns:
            aggregated_weights: {name: torch.Tensor(float32)}（floatパラメータ＋floatバッファ）
            first_int_buffers:  {name: torch.Tensor(int)}（整数バッファはコピー）
        """
        print("\n🔒 Starting CKKS model aggregation (client-side encryption)...")
        self.decrypt_calls = 0
        total_samples = sum(n for _, __, n in enc_payloads)
        weights = [n / total_samples for _, __, n in enc_payloads]
        num_clients = len(enc_payloads)

        aggregated_weights = {}
        # 整数バッファは先頭クライアントのものをコピー（要件）
        first_int_buffers = enc_payloads[0][1]

        for name, shape in self.weight_shapes.items():
            # weight_shapes には既に running_* が入っていない（FedBN時）
            num_elems = int(np.prod(shape))
            num_chunks = ceil(num_elems / self.batch_size)

            flat_vals = []
            for chunk_idx in range(num_chunks):
                c_sum = None
                for ci in range(num_clients):
                    enc_state = enc_payloads[ci][0]
                    w_i = weights[ci]
                    ct = enc_state[name][chunk_idx]
                    pt_w = self.cc.MakeCKKSPackedPlaintext([w_i] * self.batch_size)
                    term = self.cc.EvalMult(ct, pt_w)
                    c_sum = term if c_sum is None else self.cc.EvalAdd(c_sum, term)

                pt = self.cc.Decrypt(self.keys.secretKey, c_sum)
                pt.SetLength(self.batch_size)
                vals = getattr(pt, "GetRealPackedValue", pt.GetCKKSPackedValue)()
                flat_vals.extend(vals)
                self.decrypt_calls += 1

            trimmed = np.array(flat_vals[:num_elems], dtype=np.float64).reshape(tuple(shape))
            aggregated_weights[name] = torch.from_numpy(trimmed)

        # 期待復号回数チェック（per-chunk）
        expected = sum(ceil(int(np.prod(s)) / self.batch_size) for s in self.weight_shapes.values())
        assert self.decrypt_calls == expected, f"decrypt_calls={self.decrypt_calls} != expected={expected}"
        print(f"[CKKS] decrypt_calls={self.decrypt_calls} (per-chunk, bn_mode={bn_mode})")
        return aggregated_weights, first_int_buffers


# =========================================================
# データセットとモデルの定義（CIFAR-10画像分類）
# =========================================================

from torchvision import datasets, transforms
from torch.utils.data import Subset
import random


def build_transforms():
    """
    CIFAR-10用のデータ変換（前処理）を作成

    【訓練データ】
    - RandomHorizontalFlip: 左右反転でデータ拡張
    - RandomCrop: ランダムクロップでデータ拡張
    - ToTensor: PIL画像をテンソルに変換
    - Normalize: 平均と標準偏差で正規化（CIFAR-10の統計値を使用）

    【テストデータ】
    - データ拡張なし、正規化のみ
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 50%の確率で左右反転
        transforms.RandomCrop(32, padding=4),  # 4ピクセルパディング後、32x32にクロップ
        transforms.ToTensor(),  # [0,255]の画像を[0,1]のテンソルに変換
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10の統計値
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return train_transform, test_transform


def iid_partition(indices, num_clients, seed):
    """
    IID（独立同分布）データ分割

    データをランダムにシャッフルして各クライアントに均等に分配
    各クライアントは全クラスをバランスよく持つ

    Args:
        indices: データのインデックスリスト
        num_clients: クライアント数
        seed: 乱数シード

    Returns:
        各クライアントのインデックスリスト
    """
    rng = np.random.default_rng(seed)
    shuffled = np.array(indices)
    rng.shuffle(shuffled)
    splits = np.array_split(shuffled, num_clients)
    return [split.tolist() for split in splits]


def dirichlet_partition(labels, num_clients, alpha, seed):
    """
    Dirichlet分布を使用したnon-IID（非独立同分布）データ分割

    実世界の連合学習を模擬するため、各クライアントのデータ分布を偏らせる
    alphaが小さいほど偏りが大きくなる（極端な場合、各クライアントは特定クラスのみ持つ）

    【Dirichlet分布とは】
    - 確率分布のパラメータを生成する分布
    - alphaが小さい→不均一な分布（一部のクライアントに偏る）
    - alphaが大きい→均一な分布（IIDに近づく）

    Args:
        labels: 全データのラベル配列
        num_clients: クライアント数
        alpha: Dirichletパラメータ（小さいほど不均一）
        seed: 乱数シード

    Returns:
        各クライアントのインデックスリスト
    """
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    client_indices = [[] for _ in range(num_clients)]
    num_classes = int(labels.max()) + 1

    # クラスごとに処理
    for class_idx in range(num_classes):
        # このクラスに属するデータのインデックスを取得
        class_mask = labels == class_idx
        class_indices = np.where(class_mask)[0]
        rng.shuffle(class_indices)

        # Dirichlet分布で各クライアントへの割り当て比率を決定
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, proportions)

        # 各クライアントにデータを割り当て
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    # 各クライアント内でシャッフル
    for client_id in range(num_clients):
        rng.shuffle(client_indices[client_id])

    return client_indices


class SimpleCIFARNet(nn.Module):
    """
    CIFAR-10画像分類用のCNNモデル

    【ネットワーク構造】
    - 入力: 32x32x3（RGB画像）
    - 特徴抽出部（features）:
        * Conv層(32ch) → BatchNorm → ReLU → Conv層(32ch) → BatchNorm → ReLU → MaxPool
        * Conv層(64ch) → BatchNorm → ReLU → Conv層(64ch) → BatchNorm → ReLU → MaxPool
        * Dropout(0.3)
    - 分類部（classifier）:
        * Flatten → 全結合層(256) → ReLU → Dropout(0.4) → 全結合層(10)
    - 出力: 10クラス（CIFAR-10のクラス数）

    【CIFAR-10のクラス】
    0: airplane, 1: automobile, 2: bird, 3: cat, 4: deer,
    5: dog, 6: frog, 7: horse, 8: ship, 9: truck
    """

    def __init__(self):
        super(SimpleCIFARNet, self).__init__()

        # 特徴抽出部（畳み込み層）
        self.features = nn.Sequential(
            # 第1ブロック: 32x32x3 → 32x32x32 → 16x16x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3チャンネル→32チャンネル
            nn.BatchNorm2d(32),  # バッチ正規化（学習を安定化）
            nn.ReLU(inplace=True),  # 活性化関数
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 2x2プーリング（サイズ半減）

            # 第2ブロック: 16x16x32 → 16x16x64 → 8x8x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32チャンネル→64チャンネル
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8まで縮小
            nn.Dropout(0.3),  # 過学習防止
        )

        # 分類部（全結合層）
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 8x8x64 = 4096 → 1次元に平坦化
            nn.Linear(64 * 8 * 8, 256),  # 4096 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  # 過学習防止
            nn.Linear(256, 10),  # 256 → 10クラス（最終出力）
        )

    def forward(self, x):
        """
        順伝播処理

        Args:
            x: 入力画像テンソル (batch_size, 3, 32, 32)

        Returns:
            各クラスのスコア (batch_size, 10)
        """
        x = self.features(x)  # 特徴抽出
        return self.classifier(x)  # 分類


def build_client_loaders(dataset, num_clients, batch_size, iid, alpha, seed):
    """各クライアント用のデータローダーを作成"""
    indices = list(range(len(dataset)))
    if iid:
        partitions = iid_partition(indices, num_clients, seed)
    else:
        partitions = dirichlet_partition(dataset.targets, num_clients, alpha, seed)

    client_loaders = []
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


class Client:
    """
    連合学習におけるクライアント（参加者）

    【役割】
    - 手元のローカルデータでモデルを学習
    - 学習後の重みをサーバに送信
    - サーバから受け取ったグローバルモデルで初期化

    【連合学習の流れ（クライアント視点）】
    1. サーバからグローバルモデルの重みを受信
    2. 自分のローカルデータでモデルを訓練
    3. 訓練後の重みをサーバに送信
    4. 次のラウンドまで待機
    """
    def __init__(self, client_id, train_loader, device, lr=0.01, momentum=0.9, weight_decay=5e-4):
        self.client_id = client_id  # クライアントID
        self.device = device  # 計算デバイス（CPU or GPU）
        self.train_loader = train_loader  # ローカルデータのDataLoader
        self.lr = lr  # 学習率
        self.momentum = momentum  # SGDのモーメンタム
        self.weight_decay = weight_decay  # 重み減衰（正則化）
        # 暗号化用の直近状態
        self.last_state_floats = None   # {name: Tensor(float)}
        self.last_int_buffers = None    # {name: Tensor(int)}
        self.local_sample_count = 0
        # FedBN: ラウンド間で各クライアントが自分のBN統計を保持するため
        self.prev_bn_stats = {}  # {name: Tensor} for running_mean / running_var / num_batches_tracked

    def local_update(self, global_weights, epochs=1, bn_mode='fedavg'):
        """
        ローカル学習を実行

        【処理の流れ】
        1. グローバル重みでモデルを初期化
        2. ローカルデータで指定エポック数だけ学習
        3. 学習後の重みとデータ数を返す

        Args:
            global_weights: サーバから受け取ったグローバルモデルの重み
            epochs: ローカル学習のエポック数
            bn_mode: 'fedavg' or 'fedbn' (現在はクライアント側では未使用)

        Returns:
            (cpu_state, total_samples): 学習後の重みとデータ数
        """
        print(f"Client {self.client_id}: Starting local update")

        # グローバルモデルの重みで初期化
        model = SimpleCIFARNet().to(self.device)
        if bn_mode == 'fedbn':
            # FedBN: running_mean/running_var はロードせず、各クライアントが独自の統計量を保持
            filtered = {k: v for k, v in global_weights.items()
                        if not (k.endswith('running_mean') or k.endswith('running_var'))}
            model.load_state_dict(filtered, strict=False)
            # ★ 前ラウンドの自分のBN統計を復元（保持→復元）
            if self.prev_bn_stats:
                with torch.no_grad():
                    msd = model.state_dict()
                    for k, v in self.prev_bn_stats.items():
                        if k in msd:
                            msd[k].copy_(v.to(msd[k].dtype))
                    model.load_state_dict(msd, strict=False)
        else:
            model.load_state_dict(global_weights)
        model.train()  # 訓練モード

        # オプティマイザ（SGD: 確率的勾配降下法）
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,  # モーメンタムで収束を安定化
            weight_decay=self.weight_decay,  # L2正則化
        )
        # 損失関数（クロスエントロピー: 多クラス分類の標準）
        criterion = nn.CrossEntropyLoss()

        # ローカルデータで学習
        total_loss = 0.0
        total_samples = 0
        for _ in range(epochs):
            for inputs, targets in self.train_loader:
                # データをデバイスに転送
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # 勾配をゼロクリア
                optimizer.zero_grad(set_to_none=True)
                # 順伝播
                outputs = model(inputs)
                # 損失計算
                loss = criterion(outputs, targets)
                # 逆伝播（勾配計算）
                loss.backward()
                # パラメータ更新
                optimizer.step()

                # 統計情報の記録
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        print(f"Client {self.client_id}: Completed training with loss={avg_loss:.4f}")

        # 学習後の重みをCPUへ
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        # 浮動小数項目と整数バッファを分離保存（暗号化/コピー方針）
        self.last_state_floats = {k: t for k, t in cpu_state.items() if t.dtype.is_floating_point}
        self.last_int_buffers  = {k: t for k, t in cpu_state.items() if not t.dtype.is_floating_point}
        self.local_sample_count = total_samples

        # ★ FedBN: 次ラウンド用に自分のBN統計を保持
        if bn_mode == 'fedbn':
            self.prev_bn_stats = {
                k: t.clone()
                for k, t in cpu_state.items()
                if (k.endswith('running_mean') or k.endswith('running_var') or k.endswith('num_batches_tracked'))
            }
        return cpu_state, total_samples

    def encrypt_for_server(self, aggregator):
        """
        直近のfloat項目のみCKKSでチャンク暗号化して返す。
        戻り値: (enc_float_state, int_buffers_plain, sample_count)
        """
        pk = aggregator.get_public_key()
        cc = aggregator.cc
        L = aggregator.batch_size
        enc = {}
        # aggregator.weight_shapes は「集約対象の float 項目」だけ（FedBNなら running_* を含まない）
        for name in aggregator.weight_shapes.keys():
            tensor = self.last_state_floats[name]
            flat = tensor.numpy().astype(np.float64).ravel()
            chunks = []
            for i in range(0, len(flat), L):
                piece = flat[i:i+L]
                if len(piece) < L:
                    piece = np.pad(piece, (0, L - len(piece)), constant_values=0.0)
                chunks.append(cc.Encrypt(pk, cc.MakeCKKSPackedPlaintext(piece.tolist())))
            enc[name] = chunks
        return enc, self.last_int_buffers, self.local_sample_count


# =========================================================
# CKKS暗号化サーバ: グローバルモデル管理と暗号化集約
# =========================================================
class FHEServer:
    """
    CKKS準同型暗号を使用する連合学習サーバ

    【役割】
    - グローバルモデルの保持と評価
    - クライアントから受け取った重みをCKKS暗号化で集約
    - プライバシーを保護しながらモデルを更新

    【連合学習の流れ（サーバ視点）】
    1. グローバルモデルを初期化
    2. 各ラウンドで：
       a. グローバルモデルの重みをクライアントに配布
       b. クライアントから更新された重みを受信
       c. CKKS暗号化で重みを集約（平文を見ずに平均化）
       d. グローバルモデルを更新
       e. テストデータで評価
    """
    def __init__(self, num_clients, test_loader, device, bn_mode='fedavg'):
        self.num_clients = num_clients
        self.device = device
        self.global_model = SimpleCIFARNet().to(self.device)
        self.bn_mode = bn_mode

        # [REPLACED WITH OPENFHE-CKKS] ここで CKKS 版の Aggregator を使用
        self.fhe_aggregator = FHEModelAggregator(
            self.global_model,
            num_clients=num_clients,
            scale_factor=100,   # 引数互換のため受け渡しはするが、CKKS 版では未使用
            max_value=50,       # 同上
            bn_mode=bn_mode
        )

        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()

    def evaluate_global_model(self):
        """サーバでグローバルモデルの精度を評価"""
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = self.criterion(output, target)
                predicted = torch.argmax(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                total_loss += loss.item()
        accuracy = correct / total * 100
        avg_loss = total_loss / len(self.test_loader)
        print(f"Global Model - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy, avg_loss

    def aggregate_models_with_fhe(self, enc_payloads, bn_mode='fedavg'):
        """
        クライアント側暗号化済みペイロードを受け取り、暗号加重平均→復号→適用。
        enc_payloads: list of (encrypted_float_state, int_buffers_plain, num_samples)
        bn_mode: 'fedavg' or 'fedbn'
        """
        print("🔒 FHE-based model aggregation (OpenFHE CKKS, client-side encryption)...")
        agg_float_state, int_buffers = self.fhe_aggregator.aggregate_encrypted_models(enc_payloads, bn_mode=bn_mode)
        state = self.global_model.state_dict()
        # float項目（パラメータ＋BN running_mean/var）を更新
        for name, tensor in agg_float_state.items():
            state[name] = tensor
        # 整数バッファは先頭クライアントからコピー
        for name, tensor in int_buffers.items():
            state[name] = tensor
        self.global_model.load_state_dict(state, strict=False)
        print(f"🔒 FHE Global model updated (weighted avg for floats, integer buffers copied, bn_mode={bn_mode})")

    def get_global_weights(self):
        """クライアントに配布するためのグローバル重みを返す"""
        return copy.deepcopy(self.global_model.state_dict())


# =========================================================
# 平文サーバ: 暗号化なしの通常の連合学習
# =========================================================
class PlainServer:
    """
    役割:
      - グローバルモデルの保持と評価
      - 平文でのモデル集約（単純平均）
    """
    def __init__(self, num_clients, test_loader, device):
        self.num_clients = num_clients
        self.device = device
        self.global_model = SimpleCIFARNet().to(self.device)
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()

    def evaluate_global_model(self):
        """サーバでグローバルモデルの精度を評価"""
        self.global_model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                loss = self.criterion(output, target)
                predicted = torch.argmax(output, dim=1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
                total_loss += loss.item()
        accuracy = correct / total * 100
        avg_loss = total_loss / len(self.test_loader)
        print(f"Global Model - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy, avg_loss

    def aggregate_models_plain(self, client_updates, bn_mode='fedavg'):
        """
        平文でクライアント重みを平均し、グローバルへ適用。
        client_updates: list of (state_dict, num_samples) tuples
        bn_mode: 'fedavg' or 'fedbn'
        """
        print(f"📊 Plain model aggregation (weighted averaging, bn_mode={bn_mode})...")

        # サンプル数による加重平均
        total_samples = sum(num_samples for _, num_samples in client_updates)
        aggregated_weights = {}

        state0 = client_updates[0][0]
        for layer_name in state0.keys():
            if bn_mode == 'fedbn' and (layer_name.endswith('running_mean') or layer_name.endswith('running_var')):
                # 触らない（現グローバルの値を保つ）
                aggregated_weights[layer_name] = self.global_model.state_dict()[layer_name]
                continue
            # 整数型のバッファ（num_batches_trackedなど）はスキップ
            if state0[layer_name].dtype in [torch.int32, torch.int64, torch.long]:
                aggregated_weights[layer_name] = state0[layer_name].clone()
            else:
                # 加重平均を計算
                weighted_sum = torch.zeros_like(state0[layer_name])
                for state_dict, num_samples in client_updates:
                    weight = num_samples / total_samples
                    weighted_sum += state_dict[layer_name] * weight
                aggregated_weights[layer_name] = weighted_sum

        self.global_model.load_state_dict(aggregated_weights)
        print(f"📊 Plain Global model updated (bn_mode={bn_mode})")

    def get_global_weights(self):
        """クライアントに配布するためのグローバル重みを返す"""
        return copy.deepcopy(self.global_model.state_dict())


# =========================================================
# 比較実験関数
# =========================================================
def run_federated_learning_comparison(
    num_clients=10,
    num_rounds=10,
    batch_size=64,
    local_epochs=1,
    lr=0.01,
    iid=True,
    alpha=0.5,
    seed=42,
    data_dir='./data',
    bn_mode='fedavg'
):
    """
    平文とCKKS暗号化の連合学習を実行し、精度と時間を比較
    """
    print("="*80)
    print("🔬 FEDERATED LEARNING COMPARISON: Plain vs CKKS Encrypted")
    print("="*80)

    # シード設定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # データセット準備
    train_transform, test_transform = build_transforms()
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    # クライアントデータローダー作成
    client_loaders = build_client_loaders(
        train_dataset,
        num_clients=num_clients,
        batch_size=batch_size,
        iid=iid,
        alpha=alpha,
        seed=seed,
    )

    # テストデータローダー
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # 結果格納用
    results = {
        'plain': {'accuracy': [], 'time': []},
        'ckks': {'accuracy': [], 'time': [], 'key_gen_time': 0}
    }

    # =========================================================
    # 1. 平文での連合学習
    # =========================================================
    print("\n" + "="*80)
    print("📊 PLAIN FEDERATED LEARNING")
    print("="*80)

    plain_server = PlainServer(num_clients=num_clients, test_loader=test_loader, device=device)
    plain_clients = [
        Client(client_id=i+1, train_loader=client_loaders[i], device=device, lr=lr)
        for i in range(num_clients)
    ]

    print("\nInitial Global Model Evaluation (Plain):")
    initial_accuracy_plain, _ = plain_server.evaluate_global_model()
    results['plain']['accuracy'].append(initial_accuracy_plain)

    for round_num in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"📊 PLAIN ROUND {round_num + 1}")
        print(f"{'='*60}")

        round_start = time.time()

        global_weights = plain_server.get_global_weights()
        client_updates = []

        for client in plain_clients:
            print(f"\n--- Client {client.client_id} Local Update ---")
            state_dict, num_samples = client.local_update(global_weights, epochs=local_epochs, bn_mode=bn_mode)
            client_updates.append((state_dict, num_samples))

        print(f"\n--- 📊 Plain Server Aggregation ---")
        plain_server.aggregate_models_plain(client_updates, bn_mode=bn_mode)

        round_end = time.time()
        round_time = round_end - round_start

        print(f"\nRound {round_num + 1} Global Model Evaluation (Plain):")
        current_accuracy, _ = plain_server.evaluate_global_model()

        results['plain']['accuracy'].append(current_accuracy)
        results['plain']['time'].append(round_time)

        print(f"📊 Plain Round {round_num + 1} completed in {round_time:.2f} seconds!")

    # =========================================================
    # 2. CKKS暗号化での連合学習
    # =========================================================
    print("\n" + "="*80)
    print("🔒 CKKS ENCRYPTED FEDERATED LEARNING")
    print("="*80)

    # 鍵生成時間を計測
    key_gen_start = time.time()
    ckks_server = FHEServer(num_clients=num_clients, test_loader=test_loader, device=device, bn_mode=bn_mode)
    key_gen_end = time.time()
    key_gen_time = key_gen_end - key_gen_start
    results['ckks']['key_gen_time'] = key_gen_time

    print(f"🔑 Key generation time: {key_gen_time:.2f} seconds")

    ckks_clients = [
        Client(client_id=i+1, train_loader=client_loaders[i], device=device, lr=lr)
        for i in range(num_clients)
    ]

    print("\nInitial Global Model Evaluation (CKKS):")
    initial_accuracy_ckks, _ = ckks_server.evaluate_global_model()
    results['ckks']['accuracy'].append(initial_accuracy_ckks)

    for round_num in range(num_rounds):
        print(f"\n{'='*60}")
        print(f"🔒 CKKS ENCRYPTED ROUND {round_num + 1}")
        print(f"{'='*60}")

        round_start = time.time()

        global_weights = ckks_server.get_global_weights()
        enc_payloads = []

        for client in ckks_clients:
            print(f"\n--- Client {client.client_id} Local Update ---")
            state_dict, num_samples = client.local_update(global_weights, epochs=local_epochs, bn_mode=bn_mode)
            enc_payloads.append(client.encrypt_for_server(ckks_server.fhe_aggregator))

        print(f"\n--- 🔒 CKKS Server Aggregation ---")
        ckks_server.aggregate_models_with_fhe(enc_payloads, bn_mode=bn_mode)

        round_end = time.time()
        round_time = round_end - round_start

        print(f"\nRound {round_num + 1} Global Model Evaluation (CKKS):")
        current_accuracy, _ = ckks_server.evaluate_global_model()

        results['ckks']['accuracy'].append(current_accuracy)
        results['ckks']['time'].append(round_time)

        print(f"🔒 CKKS Round {round_num + 1} completed in {round_time:.2f} seconds!")

    # 訓練されたサーバーインスタンスも返す（視覚的証明のため）
    return results, {'plain': plain_server, 'ckks': ckks_server}


def plot_comparison_results(results, num_rounds, num_clients):
    """
    比較結果をグラフ化して保存
    """
    key_gen_time = results['ckks']['key_gen_time']

    # 図の作成
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # ラウンド番号（0はInitial、1以降は各ラウンド）
    rounds = list(range(num_rounds + 1))

    # =========================================================
    # 1. 精度の比較グラフ
    # =========================================================
    ax1 = axes[0]
    ax1.plot(rounds, results['plain']['accuracy'], 'o-', label='Plain', linewidth=2, markersize=8)
    ax1.plot(rounds, results['ckks']['accuracy'], 's-', label='CKKS Encrypted', linewidth=2, markersize=8)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Federated Learning Accuracy Comparison: Plain vs CKKS', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(rounds)
    ax1.set_xticklabels(['Initial'] + [f'R{i}' for i in range(1, num_rounds + 1)])

    # =========================================================
    # 2. 実行時間の比較グラフ
    # =========================================================
    ax2 = axes[1]
    round_nums = list(range(1, num_rounds + 1))

    bar_width = 0.35
    x = np.arange(len(round_nums))

    bars1 = ax2.bar(x - bar_width/2, results['plain']['time'], bar_width,
                    label='Plain', alpha=0.8)
    bars2 = ax2.bar(x + bar_width/2, results['ckks']['time'], bar_width,
                    label='CKKS Encrypted', alpha=0.8)

    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Federated Learning Execution Time Comparison: Plain vs CKKS', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Round {i}' for i in round_nums])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # 鍵生成時間をテキストで表示
    key_gen_text = f'CKKS Key Generation Time: {key_gen_time:.2f} seconds\n(Not included in round times)'
    ax2.text(0.02, 0.98, key_gen_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # グラフを保存
    output_file = f'federated_learning_comparison_clients{num_clients}_rounds{num_rounds}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison graph saved to: {output_file}")

    plt.close()


# =========================================================
# 🎨 視覚的証明: 画像分類が実際に動作していることを証明
# =========================================================

def visualize_sample_predictions(model, test_dataset, device, num_samples=16, output_file='sample_predictions.png'):
    """
    テストデータからランダムにサンプルを抽出し、予測結果を可視化

    【目的】
    このプログラムがダミーではなく、実際に画像を分類していることを証明

    Args:
        model: 学習済みモデル
        test_dataset: テストデータセット
        device: 計算デバイス
        num_samples: 表示するサンプル数
        output_file: 出力ファイル名
    """
    # CIFAR-10のクラス名
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    model.eval()

    # ランダムにサンプルを選択
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('🔍 Sample Image Classification Results (Proof of Real Classification)',
                 fontsize=14, fontweight='bold')

    for idx, ax in enumerate(axes.flat):
        if idx >= num_samples:
            break

        # データを取得
        img, true_label = test_dataset[indices[idx]]

        # 予測
        with torch.no_grad():
            img_batch = img.unsqueeze(0).to(device)
            output = model(img_batch)
            pred_label = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred_label].item()

        # 画像を正規化解除して表示用に変換
        img_display = img.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)

        # 画像を表示
        ax.imshow(img_display)
        ax.axis('off')

        # 正解・不正解で色分け
        color = 'green' if pred_label == true_label else 'red'
        title = f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}\n({confidence*100:.1f}%)'
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n🎨 Sample predictions visualization saved to: {output_file}")
    plt.close()


def visualize_confusion_matrix(model, test_loader, device, output_file='confusion_matrix.png'):
    """
    混同行列を生成して可視化

    【混同行列とは】
    縦軸が真のラベル、横軸が予測ラベルの行列
    対角成分が正解、非対角成分が誤分類を示す
    実際に分類が行われていることの強力な証拠

    Args:
        model: 学習済みモデル
        test_loader: テストデータローダー
        device: 計算デバイス
        output_file: 出力ファイル名
    """
    # CIFAR-10のクラス名
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    model.eval()
    all_preds = []
    all_labels = []

    print("\n🔍 Generating confusion matrix...")

    # 全テストデータで予測
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 混同行列を計算
    cm = confusion_matrix(all_labels, all_preds)

    # 可視化
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of samples'})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Proof of Real Image Classification',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to: {output_file}")
    plt.close()

    return cm, all_labels, all_preds


def print_classification_report(all_labels, all_preds):
    """
    クラスごとの詳細な分類性能を表示

    【表示内容】
    - Precision（適合率）: 予測した中で実際に正しかった割合
    - Recall（再現率）: 実際のクラスの中で正しく予測できた割合
    - F1-score: PrecisionとRecallの調和平均
    - Support: 各クラスのサンプル数

    Args:
        all_labels: 真のラベル
        all_preds: 予測ラベル
    """
    # CIFAR-10のクラス名
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    print("\n" + "="*80)
    print("📊 DETAILED CLASSIFICATION REPORT (Proof of Real Classification)")
    print("="*80)

    # 分類レポートを生成
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names,
                                   digits=4)
    print(report)

    # クラスごとの精度を計算
    print("\n" + "="*80)
    print("🎯 Per-Class Accuracy")
    print("="*80)

    cm = confusion_matrix(all_labels, all_preds)
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i, :].sum() * 100 if cm[i, :].sum() > 0 else 0
        print(f"{class_name:12s}: {class_acc:6.2f}% ({cm[i, i]:4d}/{cm[i, :].sum():4d})")

    print("="*80)


def print_comparison_summary(results, num_rounds):
    """
    比較結果のサマリーを標準出力
    """
    print("\n" + "="*80)
    print("📈 COMPARISON SUMMARY")
    print("="*80)

    print(f"\n🔑 CKKS Key Generation Time: {results['ckks']['key_gen_time']:.2f} seconds")
    print("   (This time is not included in round execution times)")

    print("\n" + "-"*80)
    print("📊 Accuracy Comparison (per round):")
    print("-"*80)
    print(f"{'Round':<15} {'Plain (%)':<15} {'CKKS (%)':<15} {'Difference':<15}")
    print("-"*80)

    for i in range(num_rounds + 1):
        round_name = "Initial" if i == 0 else f"Round {i}"
        plain_acc = results['plain']['accuracy'][i]
        ckks_acc = results['ckks']['accuracy'][i]
        diff = ckks_acc - plain_acc
        print(f"{round_name:<15} {plain_acc:>8.2f}%      {ckks_acc:>8.2f}%      {diff:>+8.2f}%")

    print("\n" + "-"*80)
    print("⏱️  Execution Time Comparison (per round):")
    print("-"*80)
    print(f"{'Round':<15} {'Plain (s)':<15} {'CKKS (s)':<15} {'Overhead':<15}")
    print("-"*80)

    for i in range(num_rounds):
        plain_time = results['plain']['time'][i]
        ckks_time = results['ckks']['time'][i]
        overhead = ((ckks_time / plain_time) - 1) * 100
        print(f"Round {i+1:<8} {plain_time:>8.2f}s      {ckks_time:>8.2f}s      {overhead:>+8.1f}%")

    # 平均値
    avg_plain_time = np.mean(results['plain']['time'])
    avg_ckks_time = np.mean(results['ckks']['time'])
    avg_overhead = ((avg_ckks_time / avg_plain_time) - 1) * 100

    print("-"*80)
    print(f"{'Average':<15} {avg_plain_time:>8.2f}s      {avg_ckks_time:>8.2f}s      {avg_overhead:>+8.1f}%")
    print("-"*80)

    print("\n" + "="*80)


# =========================================================
# 実行エントリポイント
# =========================================================
def main():
    """
    比較実験のメイン関数
    """
    parser = argparse.ArgumentParser(
        description='Federated Learning Comparison: Plain vs CKKS Encrypted with CIFAR-10'
    )
    parser.add_argument('--clients', type=int, default=10, help='Number of clients (default: 10)')
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated learning rounds (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--local-epochs', type=int, default=1, help='Local epochs per round (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--iid', action='store_true', help='Use IID data partitioning (default: non-IID)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet alpha for non-IID (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory (default: ./data)')
    parser.add_argument('--bn-mode', choices=['fedavg', 'fedbn'], default='fedavg',
                        help='fedbn で BN の running_mean/var を集約から除外（各クライアントに保持）')

    args = parser.parse_args()

    print(f"Starting federated learning with {args.clients} clients and {args.rounds} rounds")
    print(f"Batch size: {args.batch_size}, Local epochs: {args.local_epochs}, LR: {args.lr}")
    print(f"Data partitioning: {'IID' if args.iid else f'non-IID (alpha={args.alpha})'}")

    # 比較実験を実行
    results, servers = run_federated_learning_comparison(
        num_clients=args.clients,
        num_rounds=args.rounds,
        batch_size=args.batch_size,
        local_epochs=args.local_epochs,
        lr=args.lr,
        iid=args.iid,
        alpha=args.alpha,
        seed=args.seed,
        data_dir=args.data_dir,
        bn_mode=args.bn_mode
    )

    # 結果のサマリーを出力
    print_comparison_summary(results, args.rounds)

    # グラフを作成
    plot_comparison_results(results, args.rounds, args.clients)

    # =========================================================
    # 🎨 視覚的証明: 画像分類が実際に動作していることを証明
    # =========================================================
    print("\n" + "="*80)
    print("🎨 VISUAL PROOF: Demonstrating Real Image Classification")
    print("="*80)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # テストデータセットの準備
    _, test_transform = build_transforms()
    test_dataset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # =========================================================
    # 平文の連合学習モデルで視覚的証明
    # =========================================================
    plain_server_final = servers['plain']

    print("\n" + "-"*80)
    print("📊 PLAIN FEDERATED LEARNING - Visual Proof")
    print("-"*80)
    print(f"📝 Using the Plain trained model for visual proof.")
    print(f"   This model achieved {results['plain']['accuracy'][-1]:.2f}% accuracy on CIFAR-10.")

    # 1. サンプル画像の予測結果を可視化
    print("\n🎨 Visualizing sample predictions (Plain)...")
    visualize_sample_predictions(
        plain_server_final.global_model,
        test_dataset,
        device,
        num_samples=16,
        output_file=f'sample_predictions_plain_clients{args.clients}.png'
    )

    # 2. 混同行列を生成
    print("\n📊 Generating confusion matrix (Plain)...")
    cm_plain, all_labels_plain, all_preds_plain = visualize_confusion_matrix(
        plain_server_final.global_model,
        test_loader,
        device,
        output_file=f'confusion_matrix_plain_clients{args.clients}.png'
    )

    # 3. 詳細な分類レポートを表示
    print("\n📊 Classification Report (Plain):")
    print_classification_report(all_labels_plain, all_preds_plain)

    # =========================================================
    # CKKS暗号化の連合学習モデルで視覚的証明
    # =========================================================
    ckks_server_final = servers['ckks']

    print("\n" + "-"*80)
    print("🔒 CKKS ENCRYPTED FEDERATED LEARNING - Visual Proof")
    print("-"*80)
    print(f"📝 Using the CKKS-encrypted trained model for visual proof.")
    print(f"   This model achieved {results['ckks']['accuracy'][-1]:.2f}% accuracy on CIFAR-10.")

    # 1. サンプル画像の予測結果を可視化
    print("\n🎨 Visualizing sample predictions (CKKS)...")
    visualize_sample_predictions(
        ckks_server_final.global_model,
        test_dataset,
        device,
        num_samples=16,
        output_file=f'sample_predictions_ckks_clients{args.clients}.png'
    )

    # 2. 混同行列を生成
    print("\n📊 Generating confusion matrix (CKKS)...")
    cm_ckks, all_labels_ckks, all_preds_ckks = visualize_confusion_matrix(
        ckks_server_final.global_model,
        test_loader,
        device,
        output_file=f'confusion_matrix_ckks_clients{args.clients}.png'
    )

    # 3. 詳細な分類レポートを表示
    print("\n📊 Classification Report (CKKS):")
    print_classification_report(all_labels_ckks, all_preds_ckks)

    print("\n" + "="*80)
    print("✅ Visual proof completed for both Plain and CKKS!")
    print("="*80)
    print("\n📄 Generated files:")
    print(f"  1. federated_learning_comparison_clients{args.clients}_rounds{args.rounds}.png")
    print(f"  2. sample_predictions_plain_clients{args.clients}.png")
    print(f"  3. confusion_matrix_plain_clients{args.clients}.png")
    print(f"  4. sample_predictions_ckks_clients{args.clients}.png")
    print(f"  5. confusion_matrix_ckks_clients{args.clients}.png")
    print("\nThese visualizations prove that real image classification is being performed,")
    print("and allow comparison between Plain and CKKS encrypted federated learning!")

    print("\n✅ All comparisons completed!")


if __name__ == "__main__":
    main()

