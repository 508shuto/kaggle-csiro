# Kaggle CSIRO Competition

CSIRO牧草評価コンペティション用のリポジトリです。

## セットアップ

```bash
# 依存関係のインストール
uv sync
```

## 開発ワークフロー

### コードフォーマットとリント

このプロジェクトでは [Ruff](https://docs.astral.sh/ruff/) を使用してコードのフォーマットとリントを行います。

#### ローカルでの実行

```bash
# コードをフォーマット
uv run ruff format .

# リントチェック
uv run ruff check .

# 自動修正可能な問題を修正
uv run ruff check --fix .
```

#### Pre-commit フックのセットアップ

コミット前に自動的にフォーマットとリントを実行するには、pre-commit をセットアップします：

```bash
# pre-commit のインストール（初回のみ）
uv pip install pre-commit

# pre-commit フックのインストール
pre-commit install

# 全ファイルに対して手動実行（オプション）
pre-commit run --all-files
```

セットアップ後、コミット時に自動的にフォーマットとリントが実行されます。

## Dockerでの実行

### 前提条件

- DockerとDocker Composeがインストールされていること
- NVIDIA GPUドライバーとnvidia-container-toolkitがインストールされていること
- `.env`ファイルに必要な環境変数（`WANDB_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`など）が設定されていること

### 起動方法

```bash
# ビルド＆起動
docker compose up --build

# バックグラウンド起動
docker compose up -d

# 停止
docker compose down
```

起動後、`http://localhost:8888`でJupyter Labにアクセスできます。

### 注意事項

- コードとデータはボリュームマウントでホストと同期されます
- 初回ビルド時は依存関係のインストールに時間がかかります
- GPUが利用可能な場合、自動的にGPUが使用されます

## Kaggle Notebookでの実行手順

### 1. 前処理: 画像をnumpy配列に変換

train画像とtest画像をそれぞれnumpy配列に変換します。

```bash
# Train画像の変換
python src/exp000/img2npy.py \
  --input-dir ./input/train \
  --output-dir ./output/exp000/train \
  --image-size 256 \
  --num-workers 4

# Test画像の変換
python src/exp000/img2npy.py \
  --input-dir ./input/test \
  --output-dir ./output/exp000/test \
  --image-size 256 \
  --num-workers 4
```

`--num-workers`はKaggle NotebookのCPUコア数に応じて調整してください（推奨: 2-4）。

### 2. データセット作成

```bash
python src/exp000/create_dataset.py
```

### 3. 学習

```bash
python src/exp000/train.py --config-path ./config/exp000.yaml
```

### 4. 推論・提出ファイル生成

```bash
python src/exp000/inference.py \
  --test-csv-path ./input/test.csv \
  --config-path ./config/exp000.yaml \
  --model-dir ./output/exp000 \
  --output-dir ./output/exp000 \
  --device cuda \
  --batch-size 256 \
  --num-workers 4 \
  --use-amp true \
  --use-tta false
```

## パラメータ説明

### img2npy.py

- `--input-dir`: 入力画像ディレクトリ
- `--output-dir`: 出力numpyファイルディレクトリ
- `--image-size`: リサイズ後の画像サイズ（学習時の`augmentation.valid.image_size`に合わせる）
- `--num-workers`: 並列処理のワーカー数（デフォルト: min(4, cpu_count())）

### inference.py

- `--test-csv-path`: test.csvファイルのパス
- `--config-path`: 設定ファイルのパス
- `--model-dir`: モデルチェックポイントのディレクトリ
- `--output-dir`: 出力ディレクトリ
- `--folds`: 使用するfold番号のリスト（デフォルト: [0,1,2,3,4]）
- `--device`: デバイス（cuda/cpu/mps）
- `--batch-size`: 推論時のバッチサイズ
- `--num-workers`: DataLoaderのワーカー数
- `--use-amp`: Automatic Mixed Precisionの使用（推論高速化）
- `--use-tta`: Test Time Augmentationの使用（Horizontal Flip対応）

## 注意事項

- Kaggle NotebookではCPUコア数が限られているため、`num_workers`は2-4程度を推奨します
- `image_size`は学習時の`augmentation.valid.image_size`（config/exp000.yaml参照）に合わせてください
- 推論は単一GPU前提です。fold同時実行の並列化は行いません（DataLoader並列化で十分です）

