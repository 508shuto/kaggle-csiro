# Tasklist

## 完了条件（Definition of Done）
- `config/exp025.yaml` と `src/exp025/` が追加され、`pipeline.sh exp025` 相当の `train/evaluation/inference` が呼べる
- EXP025内で **log1p/expm1依存が無く**、rawスケールで学習→評価→提出が一貫
- main loss が **Weighted SmoothL1**（weightsは従来通り）になっている（aux_lossはMSE）
- `.steering/20260104-exp025-remove-log1p-smoothl1/` の3ファイルが作成され、tasklistのDoDが満たされている

## タスク
- [x] Done: `.steering/20260104-exp025-remove-log1p-smoothl1/` を作成し requirements/design/tasklist を用意する
- [x] Done: `config/exp024.yaml` をベースに `config/exp025.yaml` を追加し、lossを `smooth_l1_loss` に変更する
- [x] Done: `src/exp024/` をベースに `src/exp025/` を作成する
- [x] Done: EXP025内の dataset/model/lightning/eval/infer/mixup を raw スケール前提に統一し、log1p/expm1 を除去する
- [x] Done: `src/exp025/loss.py` と `src/exp025/utils.py` に WeightedSmoothL1Loss と `smooth_l1_loss` ルーティングを実装する
- [x] Done: EXP025が最低限 import/実行できることをスモークチェックする（compileall等）

