# exp000: ベースライン

## ⚠️ 重大な警告: このモデルは使用禁止

**このモデルには99.72%のデータリークがあります。**

- **CV Score**: 0.5198（過大評価）
- **推定LB Score**: ~0.37
- **乖離**: 約-0.15
- **CV戦略**: StratifiedKFold（リークあり）

### 代わりに使用すべきモデル

- **EXP003（CV: 0.5222）**: リークなし、StratifiedGroupKFold
- **EXP005（CV: 0.5181）**: リークなし、Huber Loss

---

## 概要（参考資料）

Kaggle CSIRO - Image2Biomass Prediction コンペティションのベースライン実装。シンプルな構成ですが、データリークがあるため使用不可。

## 性能（リークあり）

- **Mean CV Score**: 0.5198 ± 0.0796（過大評価）
- **画像サイズ**: 512x512

### フォールドごとのスコア

| Fold | Val R2 Score | Epochs |
|------|--------------|--------|
| 0 | 0.493 | 47 |
| 1 | 0.655 | 52 |
| 2 | 0.426 | 39 |
| 3 | 0.556 | 44 |
| 4 | 0.469 | 41 |

## モデルアーキテクチャ

### バックボーン
- **モデル**: ConvNeXt Tiny (`convnext_tiny.fb_in22k_ft_in1k`)
- **事前学習**: ImageNet-22k → ImageNet-1k
- **特徴次元**: 768

### ヘッド
```python
self.head = nn.Linear(768, 5)  # 5つのバイオマス量を予測
```

## 訓練設定

### ハイパーパラメータ
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
  early_stopping:
    patience: 10
    monitor: val_score

ema:
  enabled: true
  decay: 0.995
```

### データ拡張

**訓練時**:
- Resize (256x256)
- HorizontalFlip (p=0.5)
- VerticalFlip (p=0.5)
- Rotate (±20度, p=0.5)
- ElasticTransform (p=0.3)
- RandomBrightnessContrast (p=0.3)
- RandomGamma (p=0.3)
- GaussianNoise (p=0.3)
- Blur系（MotionBlur, MedianBlur, Blur, p=0.3）
- Normalize (ImageNet統計)

**検証時**:
- Resize (256x256)
- Normalize (ImageNet統計)

### 損失関数

Weighted MSE Loss:
```python
weights = [0.1, 0.1, 0.1, 0.2, 0.5]  # [Clover, Dead, Green, GDM, Total]
loss = ((pred - target) ** 2 * weights).mean()
```

## 実装の特徴

### シンプルな構成
- 補助タスクなし
- Mixupなし
- ターゲット変換なし（生の値）

### Early Stopping
- patience=10で早期停止
- 実際には40-50エポック程度で停止

## 主な観察

### ✅ うまくいったこと
1. **ConvNeXtバックボーン**: 現代的なCNNで良好な性能
2. **Model EMA**: 予測の安定化
3. **基本的なデータ拡張**: 一定の効果

### ❌ 改善が必要な点
1. **Early Stoppingが早すぎる**: 40-50エポックで停止、さらに訓練できる可能性
2. **フォールド間の分散が大きい**: std=0.0796、訓練が不安定
3. **ターゲット変換なし**: 歪んだ分布をそのまま学習

## exp001への改善点

1. **log1p変換の導入**: ターゲット値の分布を正規化
2. **補助タスクの追加**: マルチタスク学習で特徴抽出を改善
3. **Early Stopping無効化**: 全エポック訓練

## ファイル構成

```
src/exp000/
├── README.md              # このファイル
├── train.py              # 訓練スクリプト
├── inference.py          # 推論スクリプト
├── evaluation.py         # OOF評価
├── create_dataset.py     # データセット前処理
├── img2npy.py            # 画像→numpy変換
├── dataset.py            # PyTorchデータセット
├── models.py             # モデル定義
├── lightning_module.py   # Lightning訓練モジュール
├── metrics.py            # 評価指標
├── loss.py               # 損失関数
└── utils.py              # ユーティリティ
```

## 使用方法

### パイプライン実行
```bash
bash pipeline.sh exp000
```

### 個別実行
```bash
# 1. 画像前処理
python src/exp000/img2npy.py --input-dir ./input/train --output-dir ./output/exp000/train

# 2. データセット作成
python src/exp000/create_dataset.py

# 3. 訓練
python src/exp000/train.py --config-path ./config/exp000.yaml

# 4. 評価
python src/exp000/evaluation.py

# 5. 推論
python src/exp000/inference.py --config-path ./config/exp000.yaml
```

## 参考

- [設定ファイル](../../config/exp000.yaml)
- [実験サマリー](../../EXPERIMENTS.md)
- [詳細分析](../../doc/experiments_detail.md)
