# exp002: Mixup Augmentation

## 🚫 重大な警告: このモデルは絶対に使用禁止

**このモデルには99.72%のデータリークがあり、本番では全く使い物になりません。**

### リークの影響

- **CV Score**: 0.7239（大幅な過大評価）
- **推定LB Score**: ~0.50
- **乖離**: **-0.22**（CV - LB）
- **リーク率**: **99.72%** of validation samples

### なぜ0.7239というスコアが出たのか

99.72%の検証サンプルが訓練データと同じ`sampling_date`+`state`を持つため、モデルは「見たことがある状況」を予測しているに過ぎません。

**証拠**: `notebook/exp002/leak_audit.ipynb`で検証済み

```
Fold 0: 99.56% samples leakage
Fold 1: 99.77% samples leakage
Fold 2: 99.88% samples leakage
Fold 3: 99.64% samples leakage
Fold 4: 99.77% samples leakage
Average: 99.72% samples leakage
```

### 正しいモデル

- **EXP003（CV: 0.5222）**: リークなし、物理制約、StratifiedGroupKFold
- **EXP005（CV: 0.5181）**: リークなし、Huber Loss

**必ずEXP003を使用してください。**

---

## ⚠️ 以下は参考情報（使用不可）

このセクションはリークありでの実装を理解するための参考資料です。

### 性能（リークあり）

- **Mean CV Score**: 0.7239 ± 0.0426（過大評価）
- **推定LB Score**: ~0.50
- **改善量**: +0.1385 (exp001比) ← リークによる虚偽の改善

### フォールドごとのスコア

| Fold | Val R2 Score | 改善量（vs exp001） | Best Epoch |
|------|--------------|---------------------|------------|
| 0 | 0.769 | +0.117 | 87 |
| 1 | **0.775** ⭐ | +0.147 | 92 |
| 2 | 0.695 | +0.148 | 81 |
| 3 | 0.715 | +0.190 | 89 |
| 4 | 0.665 | +0.090 | 78 |

**注**: fold1が最高スコア（0.775）を記録

## モデルアーキテクチャ

### 基本構成（exp001と同じ）

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

        # メインヘッド: 5つのバイオマス量
        self.head = nn.Linear(768, 5)

        # 補助ヘッド: NDVIと草丈
        self.aux_head = nn.Linear(768, 2)

    def forward(self, x):
        features = self.backbone(x)
        main_output = self.head(features)
        aux_output = self.aux_head(features)
        return main_output, aux_output
```

## 主な変更点（exp001からの追加）

### Mixup Augmentationの導入

**設定**:
```yaml
training:
  mixup:
    enabled: true
    alpha: 0.2
    prob: 0.5
```

**実装**:
```python
class MixUpCallback:
    def __init__(self, alpha=0.2, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, images, targets, aux_targets):
        # 50%の確率でMixupを適用
        if np.random.rand() > self.prob:
            return images, targets, aux_targets

        batch_size = images.size(0)

        # Lambda値をBeta分布からサンプリング
        # alpha=0.2 → ほとんどの場合、0か1に近い値
        lam = np.random.beta(self.alpha, self.alpha)

        # ランダムにペアを作成
        indices = torch.randperm(batch_size)

        # 画像とターゲットを線形補間
        mixed_images = lam * images + (1 - lam) * images[indices]
        mixed_targets = lam * targets + (1 - lam) * targets[indices]
        mixed_aux_targets = lam * aux_targets + (1 - lam) * aux_targets[indices]

        return mixed_images, mixed_targets, mixed_aux_targets
```

**訓練ステップでの使用**:
```python
def training_step(self, batch, batch_idx):
    images, targets, aux_targets = batch

    # Mixup適用
    if self.training:
        images, targets, aux_targets = self.mixup(
            images, targets, aux_targets
        )

    # Forward & Loss
    main_pred, aux_pred = self.model(images)
    main_loss = self.criterion(main_pred, targets)
    aux_loss = F.mse_loss(aux_pred, aux_targets)

    total_loss = main_loss + 0.1 * aux_loss
    return total_loss
```

## Mixupの効果

### なぜこれほど効果的だったのか

1. **過学習の大幅な抑制**
   ```
   exp001（Mixupなし）:
   Epoch 50: train_loss=0.015, val_loss=0.029 (過学習の兆候)

   exp002（Mixupあり）:
   Epoch 50: train_loss=0.022, val_loss=0.025 (良好な汎化)
   ```

2. **データ拡張の効果**
   - 実質的に無限のバリエーションを生成
   - 2枚の画像を組み合わせた新しいサンプル
   - バイオマス量も線形補間（物理的に妥当）

3. **ラベルスムージング効果**
   - ターゲット値が確定値ではなく範囲を持つ
   - モデルの過信を防ぐ
   - より滑らかな予測

4. **マニフォールド上の学習**
   - 画像空間の内挿領域でも学習
   - より豊かな特徴表現

### 改善の内訳

| 構成 | CV Score | 改善量 | 寄与度 |
|------|----------|--------|--------|
| ベースライン（exp000） | 0.5198 | - | - |
| + log1p + 補助Loss（exp001） | 0.5854 | +0.0656 | 32% |
| + Mixup（exp002） | 0.7239 | +0.1385 | **68%** |
| **合計改善** | - | +0.2041 | 100% |

**Mixupが全改善の68%を占める！**

## 訓練設定

### ハイパーパラメータ

```yaml
model:
  name: convnext_tiny.fb_in22k_ft_in1k
  num_classes: 5
  pretrained: true

auxiliary_head:
  enabled: true
  num_outputs: 2
  auxiliary_loss_weight: 0.1

optimizer:
  name: AdamW
  lr: 1e-4
  weight_decay: 1e-2

scheduler:
  name: CosineAnnealingLR
  T_max: 100
  eta_min: 1e-6
  warmup_epochs: 5

training:
  max_epochs: 100
  batch_size: 16
  early_stopping:
    patience: null  # 無効化

  mixup:
    enabled: true
    alpha: 0.2      # Beta分布のパラメータ
    prob: 0.5       # Mixup適用確率

ema:
  enabled: true
  decay: 0.995

transform:
  target: log1p
```

### データ拡張

**訓練時（Mixup + 従来の拡張）**:
```python
# 1. 従来の画像拡張
transforms = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.ElasticTransform(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianNoise(p=0.3),
    A.Blur(p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. Mixup（訓練ループ内で適用）
mixed_images = lam * img1 + (1 - lam) * img2
mixed_targets = lam * target1 + (1 - lam) * target2
```

## 訓練の詳細

### 訓練曲線の特徴

```
Epoch 1-20: 急速な改善
  - train_loss: 0.15 → 0.05
  - val_score: 0.3 → 0.6

Epoch 20-60: 安定した改善
  - train_loss: 0.05 → 0.03
  - val_score: 0.6 → 0.72

Epoch 60-100: 微調整
  - train_loss: 0.03 → 0.025
  - val_score: 0.72 → 0.74
```

### 過学習の抑制

**exp001 vs exp002（fold0の例）**:

| Epoch | exp001 (Mixupなし) | exp002 (Mixupあり) |
|-------|-------------------|-------------------|
| | Train / Val | Train / Val |
| 50 | 0.015 / 0.029 | 0.022 / 0.025 |
| 75 | 0.010 / 0.032 | 0.020 / 0.023 |
| 100 | 0.008 / 0.034 | 0.019 / 0.022 |

**観察**: exp002はtrain-valギャップが小さく、過学習が抑制されている

## 推論戦略

### アンサンブル

```python
# 5つのfoldモデルの平均
predictions = []
for fold in range(5):
    model = load_checkpoint(f'fold{fold}.ckpt')
    pred = model(images)  # EMAモデル使用
    predictions.append(pred)

final_pred = torch.mean(torch.stack(predictions), dim=0)
```

### Test Time Augmentation（オプション）

```python
# TTA: HFlip, VFlip, Rot90, Rot270
tta_transforms = [
    lambda x: x,                           # オリジナル
    lambda x: torch.flip(x, dims=[3]),     # HFlip
    lambda x: torch.flip(x, dims=[2]),     # VFlip
    lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # Rot90
    lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # Rot270
]

tta_preds = []
for transform in tta_transforms:
    aug_images = transform(images)
    pred = model(aug_images)

    # 逆変換
    inv_transform = get_inverse_transform(transform)
    tta_preds.append(inv_transform(pred))

final_pred = torch.mean(torch.stack(tta_preds), dim=0)
```

## 使用方法

### 推奨: このモデルを本番提出に使用

```bash
# 完全パイプライン実行
bash pipeline.sh exp002

# または個別実行
# 1. 画像前処理
python src/exp002/img2npy.py \
  --input-dir ./input/train \
  --output-dir ./output/exp002/train

python src/exp002/img2npy.py \
  --input-dir ./input/test \
  --output-dir ./output/exp002/test

# 2. データセット作成
python src/exp002/create_dataset.py

# 3. 訓練（5 folds）
python src/exp002/train.py --config-path ./config/exp002.yaml

# 4. OOF評価
python src/exp002/evaluation.py

# 5. テストセット推論
python src/exp002/inference.py \
  --config-path ./config/exp002.yaml \
  --test-csv-path ./input/test.csv \
  --output-dir ./output/exp002 \
  --use-tta true
```

### 提出ファイル生成

```bash
python src/exp002/inference.py \
  --config-path ./config/exp002.yaml \
  --test-csv-path ./input/test.csv \
  --output-dir ./output/exp002 \
  --device cuda \
  --batch-size 256 \
  --num-workers 4 \
  --use-amp true \
  --use-tta true

# 出力: output/exp002/submission.csv
```

## 推奨事項

### ✅ このモデルを使用すべき理由

1. **最高のCV Score**: 0.7239（他の実験を大きく上回る）
2. **安定性**: std=0.0426（低い分散）
3. **全フォールドで高性能**: 最低でも0.665
4. **実績ある手法**: Mixup + マルチタスク学習

### 🔧 さらなる改善の可能性

1. **より強力なバックボーン**
   ```yaml
   model:
     name: convnext_base.fb_in22k_ft_in1k  # Tiny → Base
     # または
     name: efficientnetv2_rw_m
   ```

2. **Mixupパラメータの最適化**
   ```python
   alpha_candidates = [0.1, 0.2, 0.3, 0.4]
   prob_candidates = [0.3, 0.5, 0.7, 1.0]
   ```

3. **CutMixの追加**
   ```python
   if rand() < 0.5:
       apply_mixup()
   else:
       apply_cutmix()
   ```

4. **アンサンブル**
   - exp002（5 folds）+ exp001（5 folds）
   - 異なるバックボーンの組み合わせ

## 主な観察と学び

### ✅ 成功要因

1. **Mixupの威力**: +0.1385の劇的改善（全改善の68%）
2. **マルチタスク学習**: 補助タスクで特徴抽出を改善
3. **log1p変換**: 歪んだ分布を正規化
4. **十分な訓練**: Early Stopping無効化で全エポック訓練
5. **Model EMA**: 予測の安定化

### 📊 データ駆動型アプローチの勝利

exp002は物理制約を使わず、モデルに学習させる戦略を採用。
- exp003-008（物理制約）: 0.45-0.52
- exp002（データ駆動）: 0.72

**教訓**: 物理的知識は重要だが、アーキテクチャにハードコーディングするより、モデルに学習させる方が効果的。

## ファイル構成

```
src/exp002/
├── README.md              # このファイル
├── train.py              # 訓練スクリプト
├── inference.py          # 推論スクリプト
├── evaluation.py         # OOF評価
├── create_dataset.py     # データセット前処理
├── img2npy.py            # 画像→numpy変換
├── dataset.py            # PyTorchデータセット
├── models.py             # モデル定義（デュアルヘッド）
├── lightning_module.py   # Lightning訓練モジュール（Mixup実装）
├── metrics.py            # 評価指標
├── loss.py               # 損失関数
└── utils.py              # ユーティリティ（Mixup含む）
```

## 出力ファイル

```
output/exp002/
├── train/                # 訓練画像（.npy）
├── test/                 # テスト画像（.npy）
├── preprocessed_train.csv  # 前処理済みデータ
├── fold0.ckpt            # Fold 0モデル（val_score=0.769）
├── fold1.ckpt            # Fold 1モデル（val_score=0.775）⭐
├── fold2.ckpt            # Fold 2モデル（val_score=0.695）
├── fold3.ckpt            # Fold 3モデル（val_score=0.715）
├── fold4.ckpt            # Fold 4モデル（val_score=0.665）
├── oofs.csv              # Out-of-Fold予測
├── results.json          # CV結果
└── submission.csv        # 提出ファイル
```

## 参考

- [設定ファイル](../../config/exp002.yaml)
- [実験サマリー](../../EXPERIMENTS.md)
- [詳細分析](../../doc/experiments_detail.md)
- [exp000のREADME](../exp000/README.md)
- [exp001のREADME](../exp001/README.md)

## 論文・参考文献

- **mixup**: Beyond Empirical Risk Minimization (Zhang et al., ICLR 2018)
- **A ConvNet for the 2020s** (Liu et al., CVPR 2022) - ConvNeXt
- **Multitask Learning** (Caruana, Machine Learning 1997)

---

**このモデルをKaggle提出に使用することを強く推奨します！** 🚀
