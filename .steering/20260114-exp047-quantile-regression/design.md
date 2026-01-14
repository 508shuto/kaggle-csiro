# Design

## アプローチ

### Pinball Loss（Quantile Loss）

分位回帰の損失関数として Pinball Loss を使用:

```
L_τ(y, ŷ) = (y - ŷ) * (τ - 1{y < ŷ})
         = { τ * (y - ŷ)     if y >= ŷ  (under-prediction)
           { (τ - 1) * (y - ŷ) if y < ŷ   (over-prediction)
```

**τ = 0.5（中央値）の場合:**
- 過大予測と過小予測を同等にペナルティ
- 外れ値に対してロバスト（L1に近い性質）

**実装:**
```python
class PinballLoss(nn.Module):
    def __init__(self, quantile: float = 0.5, class_weights: list = None):
        super().__init__()
        self.quantile = quantile
        self.class_weights = class_weights  # [0.1, 0.1, 0.1, 0.2, 0.5]

    def forward(self, pred, target):
        error = target - pred  # (B, 5)
        loss = torch.where(
            error >= 0,
            self.quantile * error,
            (self.quantile - 1) * error
        )
        # クラス重み付け
        if self.class_weights is not None:
            loss = loss * self.class_weights
        return loss.mean()
```

### 既存との差分

| 項目 | exp040 | exp047 |
|------|--------|--------|
| 損失関数 | WeightedSmoothL1Loss | PinballLoss (τ=0.5) |
| 予測対象 | 条件付き期待値 | 条件付き中央値 |
| モデル読み込み | timm | timm |
| Backboneフリーズ | Yes | Yes |
| モデル構造 | 同一 | 同一 |
| 物理制約 | あり | あり |

## 変更コンポーネント

### 1. src/exp047/loss.py（新規）

- `PinballLoss`: 基本の分位損失
- `WeightedPinballLoss`: クラス重み付き分位損失

### 2. src/exp047/lightning_module.py（変更）

- 損失関数を `PinballLoss` に差し替え
- その他は exp044 を踏襲

### 3. config/exp047.yaml（新規）

- `loss.name: pinball`
- `loss.quantile: 0.5`

## 影響範囲

- 学習ループの損失計算部分のみ
- モデル構造、評価指標、推論ロジックは変更なし
- 既存コードへの影響なし（新規実験として独立）

## ディレクトリ構成

```
src/exp047/
├── models.py            # exp040からコピー（timm + freeze backbone）
├── loss.py              # exp040ベース + PinballLoss実装
├── metrics.py           # exp040からコピー（変更なし）
├── lightning_module.py  # exp040からコピー（timm optim/scheduler）
├── train.py             # exp040からコピー（変更なし）
├── evaluation.py        # exp040からコピー（変更なし）
├── inference.py         # exp040からコピー（変更なし）
├── dataset.py           # exp040からコピー（変更なし）
├── utils.py             # exp040ベース + pinball対応
└── create_dataset.py    # exp040からコピー（変更なし）

config/
└── exp047.yaml          # exp040ベース + loss: pinball
```
