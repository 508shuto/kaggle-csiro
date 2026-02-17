# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kaggle CSIRO Image2Biomass Prediction competition. The task is to predict 5 pasture biomass components (grams) from grassland images using regression models.

**Targets:** Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g

**Physical constraints:**
```
GDM_g = Dry_Clover_g + Dry_Green_g
Dry_Total_g = Dry_Clover_g + Dry_Dead_g + Dry_Green_g
```

**Evaluation:** Weighted R² with per-target weights [Clover=0.1, Dead=0.1, Green=0.1, GDM=0.2, Total=0.5]

**Submission:** Long-format CSV with `sample_id` (`{image_id}__{target_name}`) and `target` columns.

Full details: `doc/01_competition.md`

---

## Codebase Structure

```
project/
├── config/                     # YAML configs (one per experiment)
│   └── expXXX.yaml
├── src/
│   ├── expXXX/                 # Self-contained experiment implementations
│   │   ├── create_dataset.py   # Data preprocessing, fold assignment
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   ├── train.py            # Training script (K-fold CV)
│   │   ├── evaluation.py       # OOF evaluation, CV score
│   │   ├── inference.py        # Test prediction, submission generation
│   │   ├── lightning_module.py  # PyTorch Lightning module
│   │   ├── models.py           # Model architecture
│   │   ├── loss.py             # Loss functions
│   │   ├── metrics.py          # WeightedR2Score, RMSE, MAE
│   │   ├── utils.py            # Utilities
│   │   └── img2npy.py          # Image→numpy conversion (some experiments)
│   └── utils/                  # Shared utilities
│       ├── download_hf_snapshot.py
│       └── upload_scripts.py
├── input/                      # Competition data (not committed)
│   ├── train.csv, test.csv
│   ├── train/, test/           # Image directories
│   └── download_data.sh
├── output/                     # Experiment artifacts (not committed)
│   └── expXXX/
│       ├── preprocessed_train.csv
│       ├── fold{i}.ckpt        # Model checkpoints
│       ├── oofs.csv            # Out-of-fold predictions
│       ├── results.json        # CV scores
│       └── submission.csv
├── doc/                        # Permanent documentation
│   ├── 01_competition.md       # Task, targets, metric, submission format
│   ├── 02_architecture.md      # Directory structure, pipeline, artifacts
│   ├── 03_workflow.md          # Experiment procedures, W&B, checklists
│   ├── glossary.md             # Term definitions
│   └── experiment/             # Per-experiment docs (exp025–exp047)
│       └── expXXX.md
├── .steering/                  # Short-term work tracking (committed as history)
│   └── YYYYMMDD-expXXX-title/
│       ├── requirements.md
│       ├── design.md
│       └── tasklist.md
├── .claude/                    # Claude Code customization
│   ├── agents/                 # Custom agent definitions
│   └── commands/               # Slash commands (/review-exp, /stage-exp)
├── app/                        # Streamlit applications
│   ├── streamlit_app.py        # Main app
│   └── oof_analysis_app.py     # OOF prediction analysis
├── notebook/                   # Jupyter notebooks
│   ├── eda/                    # Exploratory data analysis
│   └── exp000–exp003/          # Experiment notebooks
├── pipeline.sh                 # Master pipeline (dataset→train→eval→infer→upload)
├── 01_create_dateset.sh        # Dataset creation stage
├── 02_train.slurm              # SLURM training job
├── 03_evaluation.slurm         # SLURM evaluation job
├── 04_submit_scripts.sh        # Kaggle submission upload
├── Dockerfile, docker-compose.yaml
├── pyproject.toml              # Dependencies (uv)
└── .pre-commit-config.yaml     # Ruff + detect-secrets
```

**Experiments:** 48 total (exp000–exp047). Each is self-contained in `src/expXXX/` with a matching `config/expXXX.yaml`.

---

## Technology Stack

| Category | Tool/Library |
|----------|-------------|
| Language | Python 3.12+ |
| Package manager | uv |
| Deep learning | PyTorch, PyTorch Lightning |
| Vision models | timm (DINOv2, DINOv3, SigLIP), transformers |
| Fine-tuning | peft (LoRA) |
| Data | pandas, polars, numpy |
| Image augmentation | albumentations, OpenCV |
| Experiment tracking | W&B (project: kaggle-csiro) |
| Config | OmegaConf (YAML) |
| Linting/Format | Ruff |
| Pre-commit | Ruff formatter + linter, detect-secrets |
| CI | GitHub Actions (Ruff, Claude Code review) |

---

## Development Commands

```bash
# Dependencies
uv sync

# Code quality (Ruff)
uv run ruff format .
uv run ruff check .
uv run ruff check --fix .

# Full pipeline
./pipeline.sh expXXX

# Individual stages
uv run ./src/expXXX/create_dataset.py --config-path ./config/expXXX.yaml
uv run ./src/expXXX/train.py --folds 0 1 2 3 4
uv run ./src/expXXX/evaluation.py --device mps --model_dir ./output/expXXX
uv run ./src/expXXX/inference.py --config-path ./config/expXXX.yaml --model-dir ./output/expXXX

# Pre-commit hooks
pre-commit install
pre-commit run --all-files

# Docker
docker compose up --build        # Build & start
docker compose up -d             # Background
docker compose down              # Stop

# Streamlit apps
uv run streamlit run app/oof_analysis_app.py
```

---

## Pipeline

```
input/train.csv
    │
    ▼ [create_dataset.py]
output/expXXX/preprocessed_train.csv (with fold column)
    │
    ▼ [train.py] × K folds
output/expXXX/fold0.ckpt ~ foldK.ckpt
    │
    ▼ [evaluation.py]
output/expXXX/oofs.csv + results.json
    │
    ▼ [inference.py]
output/expXXX/submission.csv
```

---

## Configuration Schema (config/expXXX.yaml)

```yaml
experiment:
  name: expXXX
  seed: 1129

dataset:
  input_dir: ./input
  output_dir: ./output
  n_folds: 3                    # 3 or 5
  shuffle: true

model:
  name: "vit_large_patch16_dinov3.lvd1689m"  # timm model name
  pretrained: true
  in_channels: 3
  freeze_backbone: true         # Freeze backbone, train head only

loss:
  name: "pinball"               # smoothl1, mse, pinball, etc.
  params:
    quantile: 0.5

aux_loss:                       # Optional auxiliary loss
  name: "aux_loss"
  params: {}
  weight: 0.1

trainer:
  train:
    epochs: 100
    warmup_epochs: 3
    batch_size: 16
    num_workers: 4
    precision: "16-mixed"       # Mixed precision
    optimizer:
      opt: "adamw"
      lr: 1e-3
      weight_decay: 1e-2
    scheduler:
      sched: "cosine"
    ema:
      decay: 0.995              # Exponential moving average
    use_wandb: true
  valid:
    batch_size: 32

augmentation:
  train:
    image_size: 512
    horizontal_flip: 0.5
    vertical_flip: 0.5
    mixup:
      enabled: true
      alpha: 0.2
      prob: 0.5
  valid:
    image_size: 512
```

---

## Key ML Patterns

### Model Architecture (models.py)

The model predicts **3 base components** (Clover, Dead, Green) and derives the remaining 2 via physical constraints:

```python
# Predict 3 → derive 5
pred_3 = clamp(head(features), min=0)  # Non-negative
GDM = Clover + Green
Total = Clover + Dead + Green
output = [Clover, Dead, Green, GDM, Total]  # (B, 5)
```

An auxiliary head predicts 2 additional targets (GDM, Total directly) for regularization.

### Common Techniques Across Experiments

- **Frozen backbone**: timm pretrained backbone (DINOv3-large) frozen, only head trained
- **LoRA fine-tuning**: Some experiments use peft LoRA on the backbone (exp037, exp039, exp043)
- **EMA**: Exponential moving average of model weights (decay ~0.995)
- **Mixup**: Applied during training with configurable alpha/probability
- **3-fold or 5-fold CV**: State-stratified K-fold cross-validation
- **Mixed precision**: fp16-mixed for training speed
- **Non-negative clamp**: All predictions clamped to >= 0 (biomass is non-negative)

### Backbone Progression

exp000–024 (ResNet/basic) → exp025–027 (DINOv2, Qwen3-VL) → exp028+ (DINOv3-large) → exp036 (SigLIP) → exp040+ (DINOv3 + optimized CV/loss)

---

## Environment Variables

Required in `.env` (see `.env.example`):

| Variable | Purpose |
|----------|---------|
| HF_TOKEN | HuggingFace Hub token (for gated models like DINOv3) |
| WANDB_API_KEY | Weights & Biases experiment tracking |
| KAGGLE_USERNAME | Kaggle API authentication |
| KAGGLE_KEY | Kaggle API key |

---

## Steering Rules（.steering/ 運用）

### 命名規則

`.steering/[YYYYMMDD]-[expXXX]-[開発タイトル]/`

- YYYYMMDD: **作業開始日**
- expXXX: 対象の実験番号（doc/experiment/expXXX.md と一致）

**実験番号の割り当て:** `doc/experiment/` の最大番号 + 1

**フォルダ作成方法:** 手動（mkdir + ファイル手動作成）

例:
- `.steering/20260101-exp000-initial-implementation/`
- `.steering/20260102-exp000-fix-filter-bag/`

### 3ファイルの役割

| ファイル | 役割 |
|---------|------|
| requirements.md | 今回の作業の要求（変更/追加する機能、制約） |
| design.md | 設計（アプローチ、変更コンポーネント、影響範囲） |
| tasklist.md | タスクと進捗（状態、完了条件） |

### ステータス語彙（tasklist.md）

- `TODO` - 未着手
- `In Progress` - 作業中
- `Blocked` - ブロック中
- `Done` - 完了

### 更新ルール

1. **作業開始時**: フォルダ作成、3ファイル新規作成、tasklist.md に完了条件を記載、`doc/experiment/expXXX.md` も同時に作成
2. **作業中**: 仕様変更は requirements.md 更新、design.md に影響差分、tasklist.md で進捗管理
3. **作業完了時**: 完了条件確認、フォルダはそのまま残す（履歴）、汎用知見は doc/* に反映

**コミット方針:** `.steering/` はコミットする（履歴として残す）

### ステアリング vs 実験ドキュメントの住み分け

- `.steering/`: 短期・作業中の記録（履歴として残す）
- `doc/experiment/`: 確定した結果と学び（永続化）

### テンプレート雛形

**requirements.md:**
```markdown
# Requirements
## 目的
## 変更/追加する機能
## 制約・前提条件
```

**design.md:**
```markdown
# Design
## アプローチ
## 変更コンポーネント
## 影響範囲
```

**tasklist.md:**
```markdown
# Tasklist
## 完了条件（Definition of Done）
## タスク
- [ ] TODO: タスク1
```

---

## 実験ワークフロー（必須）

新しい実験を開始する際は、以下のワークフローに従う。

### フェーズ1: ドキュメント作成

以下のドキュメントを作成する：

- [ ] `.steering/YYYYMMDD-expXXX-xxx/requirements.md`
- [ ] `.steering/YYYYMMDD-expXXX-xxx/design.md`
- [ ] `.steering/YYYYMMDD-expXXX-xxx/tasklist.md`
- [ ] `doc/experiment/expXXX.md`（目的・設定を記載）

### フェーズ2: ドキュメントレビュー

`/review-exp` を実行してドキュメントをレビューし、指摘事項を修正する。

### フェーズ3: 実装

ドキュメントが完成したら、実装コードを書き始める。

**重要:** フェーズ1-2が完了していない場合、実装を開始してはならない。

### フェーズ4: ステージング

実装完了後、`/stage-exp` を実行してファイルを git add する。

---

## Doc Index（永続ドキュメント導線）

| ファイル | 内容 |
|---------|------|
| README.md | セットアップ、環境構築 |
| doc/01_competition.md | コンペティション仕様（タスク、ターゲット、評価指標、提出フォーマット） |
| doc/02_architecture.md | アーキテクチャ仕様（ディレクトリ構造、パイプライン、成果物） |
| doc/03_workflow.md | ワークフロー仕様（実験手順、W&B運用、チェックリスト） |
| doc/glossary.md | 用語集 |
| doc/experiment/expXXX.md | 実験ごとの概要（目的、設定、結果、考察） |

**人間向け:**
- 初めての方 → README.md → doc/03_workflow.md
- 実験を始める → doc/03_workflow.md + doc/experiment/ でベースを選択

---

## 実験ドキュメント運用（doc/experiment/）

**命名規則:** `doc/experiment/expXXX.md`

**作成タイミング:** 実験計画確定時（実装前）に骨子を作成し、CV完了後に結果を追記する。

**現在のドキュメント:** exp025–exp047（23件）

**テンプレート雛形:**
```markdown
# expXXX: [実験タイトル]

## 目的
この実験で検証したいこと

## 設定
- モデル:
- 主要ハイパーパラメータ:

## 結果
| Metric | Score |
|--------|-------|
| CV | |
| LB | |

## 考察
学び、次のアクション
```

---

## カスタムエージェント（.claude/agents/）

| エージェント | 説明 | 呼び出し例 |
|-------------|------|----------|
| code-reviewer | 実験コードのレビュー（バグ、パフォーマンス、再現性など） | 「レビューして」「exp027をレビュー」 |
| data-analyzer | EDA（探索的データ分析）・データの分布/欠損値/相関分析 | 「EDAして」「データ分析」「train.csvを分析」 |
| error-analyzer | エラーログ解析・原因特定 | 「エラーを解析して」「なぜ失敗したか」 |
| web-summarizer | Web検索結果の要約レポート作成 | 「〇〇を調べて要約して」 |

### カスタムコマンド（.claude/commands/）

| コマンド | 説明 |
|---------|------|
| /review-exp | ステアリング・実験ドキュメントのレビュー |
| /stage-exp | 実験ファイルの git add（src/, config/, doc/, .steering/） |

### 出力先

- code-reviewer: `.log/review/YYYYMMDD-expXXX-full-review.md`
- data-analyzer: `.log/eda/YYYYMMDD-[データセット名]-eda.md`
- error-analyzer: `.log/error/YYYYMMDD-expXXX-[タスク名].md`
- web-summarizer: `./reports/YYYY-MM-DD_topic.md`

---

## CI/CD

### GitHub Actions Workflows

| Workflow | File | Trigger |
|----------|------|---------|
| Ruff lint | `.github/workflows/ruff.yml` | Push/PR |
| Claude Code | `.github/workflows/claude.yml` | Issue/PR events |
| Claude Code Review | `.github/workflows/claude-code-review.yml` | PR events |

### Pre-commit Hooks

- **ruff**: Auto-fix lint issues + format on commit
- **detect-secrets**: Prevent credential leaks
