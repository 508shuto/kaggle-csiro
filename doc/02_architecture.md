# アーキテクチャ仕様

## 目的

このドキュメントは、リポジトリ構造、パイプライン、成果物の配置を定義する。

## スコープ

- ディレクトリ構造
- 命名規則
- パイプラインの各ステージと入出力
- 成果物一覧
- 設定ファイルの構造

## 非スコープ

- コンペティションの要件（01_competition.md を参照）
- 実験の運用方法（03_workflow.md を参照）

---

## ディレクトリ構造

```
project/
├── config/
│   └── expXXX.yaml          # 実験設定
├── src/
│   ├── expXXX/              # 実験ごとの実装
│   │   ├── create_dataset.py
│   │   ├── dataset.py
│   │   ├── evaluation.py
│   │   ├── inference.py
│   │   ├── lightning_module.py
│   │   ├── loss.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── utils.py
│   └── utils/               # 共通ユーティリティ
├── input/
│   ├── train.csv
│   ├── test.csv
│   ├── train/               # 訓練画像
│   └── test/                # テスト画像
├── output/
│   └── expXXX/              # 実験ごとの成果物
├── doc/                     # 仕様書
├── notebook/                # 分析用ノートブック
└── wandb/                   # W&Bローカルログ
```

## 命名規則

| 対象 | 規則 | 例 |
|------|------|-----|
| 実験番号 | expXXX（3桁ゼロ埋め） | exp001, exp042 |
| 設定ファイル | config/expXXX.yaml | config/exp003.yaml |
| 実装ディレクトリ | src/expXXX/ | src/exp003/ |
| 成果物ディレクトリ | output/expXXX/ | output/exp003/ |
| チェックポイント | fold{i}.ckpt | fold0.ckpt |

## パイプライン

```
input/train.csv
    │
    ▼ [create_dataset.py]
preprocessed_train.csv（fold付き）
    │
    ▼ [train.py] × 5 folds
fold0.ckpt ~ fold4.ckpt
    │
    ▼ [evaluation.py]
oofs.csv, results.json
    │
    ▼ [inference.py]
submission.csv
```

## 各ステージの入出力

### 1. データセット作成（create_dataset.py）

| 入出力 | 内容 |
|--------|------|
| 入力 | input/train.csv |
| 出力 | output/expXXX/preprocessed_train.csv |
| 処理 | fold カラムの付与（CV分割） |

### 2. 訓練（train.py）

| 入出力 | 内容 |
|--------|------|
| 入力 | preprocessed_train.csv, 画像 |
| 出力 | fold{i}.ckpt（5ファイル） |
| 処理 | 各foldで訓練、チェックポイント保存 |

### 3. 評価（evaluation.py）

| 入出力 | 内容 |
|--------|------|
| 入力 | fold{i}.ckpt, validationデータ |
| 出力 | oofs.csv, results.json |
| 処理 | OOF予測生成、CVスコア算出 |

### 4. 推論（inference.py）

| 入出力 | 内容 |
|--------|------|
| 入力 | fold{i}.ckpt, テストデータ |
| 出力 | submission.csv |
| 処理 | 5モデルのアンサンブル、提出形式への変換 |

## 成果物一覧

| ファイル | 説明 | 必須 |
|---------|------|------|
| output/expXXX/preprocessed_train.csv | fold付き訓練データ | ◎ |
| output/expXXX/fold{i}.ckpt | 学習済みモデル（5つ） | ◎ |
| output/expXXX/oofs.csv | OOF予測 | ◎ |
| output/expXXX/results.json | CVスコア | ◎ |
| output/expXXX/submission.csv | 提出ファイル | ◎ |

## 設定ファイル

config/expXXX.yaml の構造：

```yaml
experiment:
  name: expXXX
  seed: 42

model:
  name: resnet50
  pretrained: true

training:
  epochs: 30
  batch_size: 32
  learning_rate: 1e-4

augmentation:
  train:
    image_size: 512
  valid:
    image_size: 512
```

必須セクション：experiment, model, training, augmentation

---

## チェックリスト

- [ ] ディレクトリ構造が規則に従っているか
- [ ] 命名規則が守られているか
- [ ] 全ての必須成果物が生成されているか
- [ ] 設定ファイルに必須セクションがあるか

## 更新トリガ

- ディレクトリ構造の変更時
- パイプラインステージの追加・変更時
- 成果物の追加時
