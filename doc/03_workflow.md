# ワークフロー仕様

## 目的

このドキュメントは、実験の作成・実行・記録の手順を定義する。

## スコープ

- 実験作成手順
- W&B運用
- 実験前チェックリスト
- 提出前チェックリスト

## 非スコープ

- コンペティションの要件（01_competition.md を参照）
- ディレクトリ構造・パイプライン（02_architecture.md を参照）

---

## 実験作成手順

### 1. 設定ファイルの作成

`config/expXXX.yaml` を作成。

必須セクション：
- experiment（name, seed）
- model
- training
- augmentation

### 2. 実装ディレクトリの作成

`src/expXXX/` を作成し、必要なファイルを実装。

### 3. 実行

```bash
bash pipeline.sh expXXX
```

または個別実行：

```bash
python src/expXXX/create_dataset.py --config config/expXXX.yaml
python src/expXXX/train.py --config config/expXXX.yaml
python src/expXXX/evaluation.py --config config/expXXX.yaml
python src/expXXX/inference.py --config config/expXXX.yaml
```

## W&B

### 設定

| 項目 | 値 |
|------|-----|
| project | kaggle-csiro |
| run名 | expXXX-fold{i} |

### 記録項目

| 項目 | 説明 |
|------|------|
| config | 実験設定（yaml全体） |
| train/loss | 訓練損失 |
| valid/loss | 検証損失 |
| valid/score | 検証スコア（重み付きR²） |
| learning_rate | 学習率 |

### Artifact

| 名前 | 内容 |
|------|------|
| model-expXXX-fold{i} | チェックポイント |

## 実験前チェックリスト

実行前に確認：

- [ ] 画像サイズが config の augmentation.valid.image_size と一致しているか
- [ ] seed が固定されているか
- [ ] 実験番号（expXXX）が既存と重複していないか
- [ ] W&B の project 名が正しいか

## 提出前チェックリスト

submission.csv の検証：

- [ ] 行数 = テスト画像数 × 5
- [ ] sample_id の形式が `{image_id}__{target_name}` か
- [ ] 全ての image_id が含まれているか
- [ ] 5つ全てのターゲットが含まれているか
- [ ] NaN が含まれていないか
- [ ] 負の値が含まれていないか（バイオマス量は非負）

---

## チェックリスト

- [ ] 実験作成手順に従っているか
- [ ] W&B にログが記録されているか
- [ ] 実験前チェックリストを完了したか
- [ ] 提出前チェックリストを完了したか

## 更新トリガ

- 実験作成手順の変更時
- W&B 運用ルールの変更時
- チェック項目の追加時
