# exp001: 補助ターゲットの追加

## ⚠️ 重大な警告: このモデルは使用禁止

**このモデルには99.72%のデータリークがあります。**

- **CV Score**: 0.5854（過大評価）
- **推定LB Score**: ~0.40
- **乖離**: 約-0.18
- **CV戦略**: StratifiedKFold（リークあり）

### 代わりに使用すべきモデル

- **EXP003（CV: 0.5222）**: リークなし、StratifiedGroupKFold
- **EXP005（CV: 0.5181）**: リークなし、Huber Loss

---

## 概要（参考資料）

ベースライン（exp000）にマルチタスク学習を導入した実験。`Pre_GSHH_NDVI`と`Height_Ave_cm`を補助ターゲットとして予測しますが、データリークがあるため使用不可。

## 性能（リークあり）

- **Mean CV Score**: 0.5854 ± 0.0483（過大評価）
- **画像サイズ**: 512x512
- **改善量**: +0.0656 (exp000比) ← リークによる虚偽の改善

### フォールドごとのスコア

| Fold | Val R2 Score | 改善量 | Epochs |
|------|--------------|--------|--------|
| 0 | 0.652 | +0.159 | 100 |
| 1 | 0.628 | -0.027 | 100 |
| 2 | 0.547 | +0.121 | 100 |
| 3 | 0.525 | -0.031 | 100 |
| 4 | 0.575 | +0.106 | 100 |

## モデルアーキテクチャ

### デュアルヘッド構成

```python
class CSIROModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'convnext_tiny.fb_in22k_ft_in1k',
            pretrained=True,
            num_classes=0
        )

        # メインヘッド: 5つのバイオマス量
        self.head = nn.Linear(768, 5)

        # 補助ヘッド: NDVIと草丈
        self.aux_head = nn.Linear(768, 2)

    def forward(self, x):
        features = self.backbone(x)
        main_output = self.head(features)      # (B, 5)
        aux_output = self.aux_head(features)   # (B, 2)
        return main_output, aux_output
```

## 主な変更点

### 1. 補助タスクの追加

**補助ターゲット**:
- `Pre_GSHH_NDVI`: 画像撮影前の正規化植生指数
- `Height_Ave_cm`: 平均草丈

**なぜこれらが有効か**:
```
相関分析:
- NDVI vs Dry_Green_g: 0.78 (高相関)
- NDVI vs GDM_g: 0.82 (高相関)
- Height vs Dry_Total_g: 0.85 (高相関)
- Height vs GDM_g: 0.79 (高相関)
```

### 2. log1p変換の導入

**ターゲット変換**:
```python
# メインターゲット
target = torch.log1p(torch.tensor([
    Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g
]))

# 補助ターゲット
aux_target = torch.log1p(torch.tensor([
    Pre_GSHH_NDVI, Height_Ave_cm
]))
```

**効果**:
- 歪んだ分布を正規化
- 大小の値を均等に学習
- 訓練の安定化

### 3. 損失関数の拡張

```python
def compute_loss(self, batch):
    images, targets, aux_targets = batch
    main_pred, aux_pred = self.model(images)

    # メイン損失（重み付きMSE）
    main_loss = self.criterion(main_pred, targets)

    # 補助損失（MSE）
    aux_loss = F.mse_loss(aux_pred, aux_targets)

    # 合計損失
    total_loss = main_loss + 0.1 * aux_loss

    return total_loss
```

**補助損失の重み**: 0.1（控えめに設定）

### 4. Early Stopping無効化

```yaml
training:
  max_epochs: 100
  early_stopping:
    patience: null  # 無効化、全エポック訓練
```

## 訓練設定

### ハイパーパラメータ（exp000と同じ）
```yaml
optimizer:
  name: AdamW
  lr: 1e-4
  weight_decay: 1e-2

scheduler:
  name: CosineAnnealingLR
  T_max: 100
  warmup_epochs: 5

training:
  max_epochs: 100
  batch_size: 16

auxiliary_head:
  enabled: true
  num_outputs: 2
  auxiliary_loss_weight: 0.1
```

### データ拡張（exp000と同じ）
- HorizontalFlip, VerticalFlip, Rotate
- ElasticTransform
- RandomBrightnessContrast, GaussianNoise, Blur

## 実装の詳細

### データセットの変更

```python
class CSIRODataset(Dataset):
    def __getitem__(self, idx):
        # 画像読み込み
        image = load_image(...)

        # メインターゲット（log1p変換）
        target = torch.log1p(torch.tensor([
            self.df.iloc[idx]['Dry_Clover_g'],
            self.df.iloc[idx]['Dry_Dead_g'],
            self.df.iloc[idx]['Dry_Green_g'],
            self.df.iloc[idx]['GDM_g'],
            self.df.iloc[idx]['Dry_Total_g'],
        ], dtype=torch.float32))

        # 補助ターゲット（log1p変換）
        aux_target = torch.log1p(torch.tensor([
            self.df.iloc[idx]['Pre_GSHH_NDVI'],
            self.df.iloc[idx]['Height_Ave_cm'],
        ], dtype=torch.float32))

        return image, target, aux_target
```

### Lightning Moduleの変更

```python
class CSIROLightningModule(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        images, targets, aux_targets = batch
        main_pred, aux_pred = self.model(images)

        main_loss = self.criterion(main_pred, targets)
        aux_loss = F.mse_loss(aux_pred, aux_targets)

        total_loss = main_loss + self.aux_loss_weight * aux_loss

        # ロギング
        self.log('train_loss', total_loss)
        self.log('train_main_loss', main_loss)
        self.log('train_aux_loss', aux_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets, aux_targets = batch

        # EMAモデルで評価
        main_pred, aux_pred = self.model_ema(images)

        # expm1で元の空間に戻す
        pred_original = torch.expm1(main_pred)
        target_original = torch.expm1(targets)

        # R2スコア計算
        r2 = weighted_r2_score(target_original, pred_original)

        self.log('val_score', r2)
        return r2
```

## 主な観察

### ✅ 改善点

1. **スコア向上**: 全体で+0.0656の改善
2. **分散減少**: std: 0.0796 → 0.0483（より安定した訓練）
3. **全エポック訓練**: 早期停止なしでさらに学習
4. **マルチタスク学習**: より汎化性の高い特徴を獲得

### 📊 訓練曲線の変化

**exp000（Early Stopping有効）**:
```
Epoch 40-50: 早期停止
最終val_score: 0.5198
```

**exp001（全エポック訓練）**:
```
Epoch 1-30: 急速な改善
Epoch 30-70: 緩やかな改善
Epoch 70-100: 微調整
最終val_score: 0.5854
```

### 🔍 補助タスクの効果

```
メイン損失のみ（exp000）: 0.5198
メイン損失 + 補助損失（exp001）: 0.5854
改善量: +0.0656 (+12.6%)
```

**なぜ効果的か**:
- 補助タスクがバイオマス量と高相関
- マルチタスク学習で過学習を抑制
- より豊かな特徴表現を学習

## exp002への改善点

exp001でもまだ過学習の傾向あり。次の実験では：

1. **Mixup Augmentation**: より強力な正則化
2. データ拡張の強化
3. モデルサイズの調整

→ **exp002でMixupを導入し、CV: 0.7239を達成**

## 使用方法

### パイプライン実行
```bash
bash pipeline.sh exp001
```

### 個別実行
```bash
# 訓練
python src/exp001/train.py --config-path ./config/exp001.yaml

# 評価
python src/exp001/evaluation.py

# 推論
python src/exp001/inference.py --config-path ./config/exp001.yaml
```

## ファイル構成

```
src/exp001/
├── README.md              # このファイル
├── train.py              # 訓練スクリプト
├── inference.py          # 推論スクリプト
├── evaluation.py         # OOF評価
├── create_dataset.py     # データセット前処理
├── img2npy.py            # 画像→numpy変換
├── dataset.py            # PyTorchデータセット（補助ターゲット追加）
├── models.py             # モデル定義（デュアルヘッド）
├── lightning_module.py   # Lightning訓練モジュール（補助Loss追加）
├── metrics.py            # 評価指標
├── loss.py               # 損失関数
└── utils.py              # ユーティリティ
```

## 参考

- [設定ファイル](../../config/exp001.yaml)
- [実験サマリー](../../EXPERIMENTS.md)
- [詳細分析](../../doc/experiments_detail.md)
- [exp000のREADME](../exp000/README.md)
- [exp002のREADME](../exp002/README.md)
