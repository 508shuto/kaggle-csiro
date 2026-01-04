# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

1. **作業開始時**: フォルダ作成、3ファイル新規作成、tasklist.md に完了条件を記載
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

**作成タイミング:** CV完了後（CVスコアが確定した時点）

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

## 開発コマンド

```bash
# 依存関係
uv sync

# コード品質（Ruff）
uv run ruff format .
uv run ruff check .
uv run ruff check --fix .

# パイプライン実行
./pipeline.sh expXXX

# 個別実行
uv run ./src/expXXX/train.py --folds 0 1 2 3 4
uv run ./src/expXXX/evaluation.py --device mps --model_dir ./output/expXXX
uv run ./src/expXXX/inference.py --config-path ./config/expXXX.yaml --model-dir ./output/expXXX
```

---

## カスタムエージェント（.claude/agents/）

| エージェント | 説明 | 呼び出し例 |
|-------------|------|----------|
| code-reviewer | 実験コードのレビュー（バグ、パフォーマンス、再現性など） | 「レビューして」「exp027をレビュー」 |
| error-analyzer | エラーログ解析・原因特定 | 「エラーを解析して」「なぜ失敗したか」 |
| web-summarizer | Web検索結果の要約レポート作成 | 「〇〇を調べて要約して」 |

### 出力先

- code-reviewer: `.log/review/YYYYMMDD-expXXX-full-review.md`
- error-analyzer: `.log/error/YYYYMMDD-expXXX-[タスク名].md`
- web-summarizer: `./reports/YYYY-MM-DD_topic.md`
