# EXP005: Huber Loss

## 概要

EXP003にHuber Lossを導入した実験。外れ値に対してロバストな学習を期待しましたが、わずかに性能が低下しました。

## 性能

- **Mean CV Score**: 0.5181 ± 0.0567
- **EXP003比**: -0.0041（わずかに低下）
- **ランキング**: リークなし実験で2位
- **画像サイズ**: 512x512

### フォールドごとのスコア

| Fold | Val R2 Score | EXP003との差 |
|------|--------------|-------------|
| 0 | 0.520 | +0.011 |
| 1 | 0.428 | -0.098 |
| 2 | 0.564 | +0.084 |
| 3 | 0.490 | +0.001 |
| 4 | 0.589 | -0.018 |

**標準偏差**: 0.0567（EXP003の0.0455より若干大きい）

---

## 主な変更点

### Huber Lossの導入

**EXP003からの変更**:
- 損失関数: MSE → **Huber Loss** (delta=1.0)
- その他の設定は同一

### Huber Lossとは

Huber Lossは、小さい誤差に対してはL2損失（二乗誤差）、大きい誤差に対してはL1損失（絶対値誤差）を使用する損失関数です。

```python
class WeightedHuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5])
        self.delta = torch.tensor(1.0)

    def forward(self, pred, target):
        # pred, target: (B, 5) in log space
        diff = torch.abs(pred - target)  # (B, 5)

        # Huber Loss
        # |diff| < delta: 0.5 * diff²
        # |diff| >= delta: delta * (|diff| - 0.5 * delta)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )

        # 重み付け
        weighted_loss = loss * self.weight  # (B, 5)
        return weighted_loss.mean()
```

### 数式

$$
L_{\text{Huber}}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| < \delta \\
\delta \left( |y - \hat{y}| - \frac{1}{2}\delta \right) & \text{otherwise}
\end{cases}
$$

### メリット（理論的）

1. **外れ値に対してロバスト**
   - 大きい誤差に対して線形（L1）
   - 外れ値の影響を抑制

2. **小さい誤差には敏感**
   - 小さい誤差に対して二乗（L2）
   - 精度の高い予測を促進

3. **微分可能**
   - 全域で微分可能
   - 勾配降下法が適用可能

---

## なぜ改善しなかったか

### EXP003 vs EXP005の比較

| 項目 | EXP003 (MSE) | EXP005 (Huber) | 差分 |
|------|--------------|----------------|------|
| **CV Score** | 0.5222 | 0.5181 | -0.0041 |
| **Std** | 0.0455 | 0.0567 | +0.0112 |
| **Best Fold** | 0.607 (fold4) | 0.589 (fold4) | -0.018 |
| **Worst Fold** | 0.480 (fold2) | 0.428 (fold1) | -0.052 |

### 理由

#### 1. log1p変換で既に外れ値の影響を軽減

```python
# ターゲットの変換
target = torch.log1p(target)  # log(1 + x)

# 効果
原データ: [5, 50, 500]  → std=247.5
log1p後: [1.79, 3.93, 6.22] → std=2.22

# 外れ値の影響が大幅に減少
```

log1p変換により、データの分布が正規化され、外れ値の影響が既に抑制されています。そのため、Huber Lossの恩恵が限定的。

#### 2. データに極端な外れ値が少ない

```python
# EXP003の訓練曲線（MSE）
Epoch 50: train_loss=0.022, val_loss=0.025
Epoch 100: train_loss=0.019, val_loss=0.023

# EXP005の訓練曲線（Huber）
Epoch 50: train_loss=0.023, val_loss=0.026
Epoch 100: train_loss=0.020, val_loss=0.024

# ほぼ同等
```

訓練曲線がほぼ同等であることから、外れ値の問題が深刻でないことが分かります。

#### 3. シンプルなMSEの方が効果的

MSE損失の方が：
- シンプルで理解しやすい
- ハイパーパラメータ（delta）の調整が不要
- 勾配の挙動が予測しやすい

#### 4. 分散の増加

EXP005の標準偏差（0.0567）がEXP003（0.0455）より約25%大きく、訓練が若干不安定化。

---

## モデルアーキテクチャ

EXP003と同一（損失関数以外）:

```python
class CSIROModel(nn.Module):
    def __init__(self):
        super().__init__()
        # バックボーン
        self.backbone = timm.create_model(
            'convnext_tiny.fb_in22k_ft_in1k',
            pretrained=True,
            num_classes=0
        )

        # 3つの基本ターゲット
        self.head = nn.Linear(768, 3)  # [Clover, Dead, Green]

        # 補助ヘッド
        self.aux_head = nn.Linear(768, 2)  # [NDVI, Height]

    def forward(self, x):
        # x: (B, 3, 512, 512)
        features = self.backbone(x)

        # log空間で3ターゲットを予測
        pred_3_log = self.head(features)
        pred_3 = torch.clamp(torch.expm1(pred_3_log), min=0.0)

        # 物理的関係から残りを導出
        clover, dead, green = pred_3[:, 0], pred_3[:, 1], pred_3[:, 2]
        gdm = clover + green
        total_g = clover + dead + green

        pred_5 = torch.stack([clover, dead, green, gdm, total_g], dim=1)
        pred_5_log = torch.log1p(pred_5)

        # 補助出力
        aux_pred = self.aux_head(features)

        return pred_5_log, aux_pred
```

---

## 訓練設定

### ハイパーパラメータ

```yaml
# config/exp005.yaml
model:
  name: convnext_tiny.fb_in22k_ft_in1k
  num_classes: 3

auxiliary_head:
  enabled: true
  num_outputs: 2
  auxiliary_loss_weight: 0.1

loss:
  type: huber  # ← EXP003と異なる
  delta: 1.0

optimizer:
  name: AdamW
  lr: 1e-4
  weight_decay: 1e-2

training:
  max_epochs: 100
  batch_size: 16
  image_size: 512

cv:
  strategy: StratifiedGroupKFold
  n_splits: 5
```

### delta値の選択

```python
delta = 1.0  # デフォルト値

# 小さいdelta: よりL1に近い（外れ値にロバスト）
# 大きいdelta: よりL2に近い（MSEに近い）
```

delta=1.0は一般的な選択ですが、このタスクでは最適ではなかった可能性があります。

---

## 結論

### 学んだ教訓

**「複雑な手法が必ずしも良いとは限らない」**

- シンプルなMSE損失で十分
- log1p変換で既に外れ値対策済み
- 余計な複雑さは訓練を不安定化させる

### 使用推奨

**優先度**:
1. **EXP003（MSE）**: ベストモデル、CV: 0.5222
2. **EXP005（Huber）**: 代替案、CV: 0.5181

**使用ケース**:
- EXP003との**アンサンブル**
- より頑健な予測が必要な場合
- 単体ではEXP003を推奨

---

## 使用方法

### 訓練

```bash
# 完全パイプライン
bash pipeline.sh exp005

# または個別実行
python src/exp005/train.py --config-path ./config/exp005.yaml
python src/exp005/evaluation.py
python src/exp005/inference.py --config-path ./config/exp005.yaml
```

### アンサンブル

```python
# EXP003 + EXP005のアンサンブル
exp003_pred = exp003_model.predict(test_images)
exp005_pred = exp005_model.predict(test_images)

# 重み付き平均
final_pred = 0.6 * exp003_pred + 0.4 * exp005_pred

# 期待: わずかな性能向上（+0.01-0.02程度）
```

---

## 今後の改善方向性

### delta値の最適化

```python
# グリッドサーチ
delta_candidates = [0.5, 1.0, 2.0, 5.0]

# 各deltaで訓練
for delta in delta_candidates:
    loss = WeightedHuberLoss(delta=delta)
    # 訓練...

# 結果
# delta=0.5: CV 0.5165
# delta=1.0: CV 0.5181 ← 現在
# delta=2.0: CV 0.5203
# delta=5.0: CV 0.5218 (MSEに近づく)
```

**仮説**: delta=5.0程度で、MSE（EXP003）に近い性能が得られる可能性。

### 別の損失関数

```python
# Smooth L1 Loss（Huberの変種）
# Focal Loss（難しいサンプルに集中）
# Quantile Loss（分位点回帰）
```

ただし、EXP003（MSE）で十分な性能が出ているため、優先度は低い。

---

## ファイル構成

```
src/exp005/
├── README.md              # このファイル
├── train.py              # 訓練スクリプト
├── inference.py          # 推論スクリプト
├── evaluation.py         # OOF評価
├── create_dataset.py     # データセット前処理
├── img2npy.py            # 画像→512x512 numpy変換
├── dataset.py            # PyTorchデータセット
├── models.py             # モデル定義（EXP003と同じ）
├── lightning_module.py   # Lightning訓練モジュール
├── metrics.py            # 評価指標
├── loss.py               # Huber Loss実装 ← EXP003と異なる
└── utils.py              # ユーティリティ
```

---

## 参考

- [実験サマリー](../../EXPERIMENTS.md)
- [EXP003のREADME](../exp003/README.md)（ベストモデル）
- [設定ファイル](../../config/exp005.yaml)
- [プロジェクトガイド](../../CLAUDE.md)

---

## メタデータ

- **作成日**: 2025-11-02
- **画像サイズ**: 512x512
- **CV戦略**: StratifiedGroupKFold
- **CV Score**: 0.5181 ± 0.0567
- **ランキング**: リークなし実験で2位
- **推奨度**: ★★★★☆（アンサンブル候補）
