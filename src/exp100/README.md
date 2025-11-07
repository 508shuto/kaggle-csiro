# exp100: Qwen3-VL Fine-tuning Pipeline

Qwen3-VLを使用した草地バイオマス予測のためのファインチューニングパイプライン

## 概要

このパイプラインは、Vision-Language Model（Qwen3-VL）を使用して、草地画像から5つのバイオマス指標を予測します：

- Dry Clover (g)
- Dry Dead (g)
- Dry Green (g)
- GDM (g)
- Dry Total (g)

## 主な特徴

### 1. **StratifiedGroupKFold Cross-Validation**
- 同じサンプルIDの画像が異なるfoldに分かれないように制御
- 種（species）の分布を均等に保つ
- データ漏洩を防止し、真の汎化性能を評価

### 2. **Qwen3VLForConditionalGeneration**
- Hugging Face Transformersの公式実装を使用
- `transformers >= 4.51.0`が必要
- 2B/4Bモデルバリアントをサポート

### 3. **LoRA (Low-Rank Adaptation)**
- メモリ効率的なファインチューニング
- トレーニング可能なパラメータを大幅に削減
- 4bit/8bit量子化にも対応

### 4. **Transformers Trainer**
- PyTorch LightningではなくTransformers標準のTrainerを使用
- シンプルで保守しやすいコード
- Hugging Faceエコシステムとの高い互換性

## ディレクトリ構造

```
src/exp100/
├── __init__.py               # パッケージ初期化
├── create_dataset.py         # データセット作成（StratifiedGroupKFold）
├── dataset.py                # Qwen3-VL用データセット
├── models.py                 # Qwen3VLForRegressionモデル
├── utils.py                  # ユーティリティ関数
├── train.py                  # トレーニングスクリプト
├── inference.py              # 推論スクリプト
├── evaluation.py             # 評価スクリプト
└── README.md                 # このファイル

config/
└── exp100.yaml               # 実験設定ファイル

output/exp100/
├── preprocessed_train.csv    # 前処理済みデータ
├── fold0/                    # Fold 0の結果
│   ├── best_model/          # 最良モデル
│   ├── metrics.json         # 評価メトリクス
│   └── predictions_valid.csv # 予測結果
├── fold1/
├── ...
├── oof_predictions.csv       # OOF予測
└── evaluation_valid.json     # 全体の評価結果
```

## 依存関係

```bash
# 必須
transformers >= 4.51.0
qwen-vl-utils >= 0.0.8
peft >= 0.13.0
accelerate >= 1.2.0
torch >= 2.9.0
torchvision >= 0.24.0

# オプション（QLoRA用）
bitsandbytes >= 0.44.0

# その他
omegaconf >= 2.3.0
pandas >= 2.3.3
numpy >= 2.3.4
scikit-learn >= 1.7.2
Pillow
tqdm
wandb  # ロギング用
```

## 使用方法

### 1. データセット作成

```bash
python src/exp100/create_dataset.py
```

**処理内容：**
- `input/train.csv`を読み込み
- StratifiedGroupKFoldで5-fold分割
- `output/exp100/preprocessed_train.csv`に保存

**出力例：**
```
====================================================
Creating StratifiedGroupKFold splits
  n_splits: 5
  shuffle: True
  random_state: 42
  groups: sample_id
  stratify: species
====================================================

Fold Distribution
====================================================

Fold 0:
  Total samples: 200
  Unique sample_ids: 50
  Species distribution:
    Species_A: 100 (50.0%)
    Species_B: 100 (50.0%)
...
```

### 2. トレーニング

```bash
# 全foldでトレーニング
python src/exp100/train.py

# Wandbを無効化する場合は設定を変更
# config/exp100.yaml の report_to: "none"
```

**設定のカスタマイズ：**

`config/exp100.yaml`を編集：

```yaml
model:
  name: "Qwen/Qwen3-VL-2B-Instruct"  # または 4B
  use_lora: true
  load_in_4bit: false  # メモリ節約時はtrue

lora:
  r: 64
  lora_alpha: 16

training:
  num_train_epochs: 10
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
```

**出力：**
- `output/exp100/fold{0-4}/best_model/` - 最良モデル
- `output/exp100/fold{0-4}/metrics.json` - メトリクス

### 3. 推論

```bash
# 全foldで検証セットに対して推論
python src/exp100/inference.py --data_type valid

# 特定のfoldのみ
python src/exp100/inference.py --fold 0 --data_type valid

# バッチサイズを変更
python src/exp100/inference.py --batch_size 2
```

**出力：**
- `output/exp100/fold{0-4}/predictions_valid.csv`

### 4. 評価

```bash
# 全foldの評価とOOF作成
python src/exp100/evaluation.py --data_type valid --create_oof
```

**出力：**
```
====================================================
Overall Statistics - valid
====================================================

weighted_r2:
  Mean: 0.8500
  Std:  0.0120
  Min:  0.8350
  Max:  0.8650

r2_clover:
  Mean: 0.8200
  ...
```

- `output/exp100/evaluation_valid.json` - 評価結果
- `output/exp100/oof_predictions.csv` - OOF予測
- `output/exp100/oof_metrics.json` - OOF全体のメトリクス

## プロンプトテンプレート

### デフォルト（日本語）

```
この草地画像を分析し、以下の指標をグラム単位で予測してください：
- 種: Species_A
- 事前NDVI: 0.750
- 平均高さ: 25.5cm

以下の形式で予測値を出力してください：
Dry Clover: [値]g
Dry Dead: [値]g
Dry Green: [値]g
GDM: [値]g
Dry Total: [値]g
```

### Simple（英語）

```yaml
prompt:
  template: "simple"
```

```
Predict grassland biomass values for this image:
Species: Species_A, NDVI: 0.750, Height: 25.5cm

Output format:
Dry Clover: [value]g
Dry Dead: [value]g
...
```

### Concise（簡潔版）

```yaml
prompt:
  template: "concise"
```

```
Species: Species_A, NDVI: 0.750, Height: 25.5cm
Predict: Dry Clover, Dry Dead, Dry Green, GDM, Dry Total (g)
```

## メモリ要件

| モデル | LoRA | 4bit量子化 | メモリ使用量（推定） |
|-------|------|-----------|-------------------|
| 2B | ✓ | ✗ | ~12GB |
| 2B | ✓ | ✓ | ~8GB |
| 4B | ✓ | ✗ | ~20GB |
| 4B | ✓ | ✓ | ~12GB |

## トラブルシューティング

### 1. メモリ不足

```yaml
# config/exp100.yaml
model:
  load_in_4bit: true  # 有効化

training:
  per_device_train_batch_size: 1  # 削減
  gradient_accumulation_steps: 16  # 増加
  gradient_checkpointing: true  # 有効化
```

### 2. `Qwen3VLForConditionalGeneration` not found

```bash
# 最新版をインストール
pip install git+https://github.com/huggingface/transformers
```

### 3. 画像が見つからない

- `input/train/` ディレクトリに画像があることを確認
- `preprocessed_train.csv` の `image_path` カラムを確認

### 4. 生成テキストの解析失敗

- `src/exp100/models.py` の `parse_predictions()` メソッドを調整
- プロンプトテンプレートを変更して、より明確な出力形式を指定

## 評価メトリクス

### Weighted R² Score

```python
weights = [0.1, 0.1, 0.1, 0.2, 0.5]
weighted_r2 = sum(r2_i * weight_i for i in range(5))
```

- Dry Clover: 10%
- Dry Dead: 10%
- Dry Green: 10%
- GDM: 20%
- Dry Total: 50%

### 個別R² Score

各ターゲットごとのR²スコアも計算・表示されます。

## 参考文献

- [Qwen3-VL Documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [Qwen3-VL Model Hub](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)

## ライセンス

このコードは実験用です。Qwen3-VLモデルのライセンスに従ってください。
