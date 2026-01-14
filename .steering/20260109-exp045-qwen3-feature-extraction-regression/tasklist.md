# Tasklist

## 完了条件（Definition of Done）
- [ ] Qwen3-VL-Embedding-2Bで特徴量抽出→回帰が動作する
- [ ] 3-fold CVでスコア算出完了
- [ ] exp043/044との比較結果を記録

## タスク
- [ ] TODO: src/exp045/ディレクトリ作成（exp044をベースにコピー）
- [ ] TODO: models.pyをQwen3-VL-Embedding-2B用に修正
- [ ] TODO: dataset.pyをQwen3VL用の入力形式に修正
- [ ] TODO: lightning_module.pyをHead学習のみに修正
- [ ] TODO: loss.pyをSmoothL1Lossに変更（シンプル化）
- [ ] TODO: config/exp045.yaml作成
- [ ] TODO: 動作確認（forward pass）
- [ ] TODO: 学習実行（fold 0）
- [ ] TODO: 全fold学習 + evaluation.py実行
- [ ] TODO: 結果をdoc/experiment/exp045.mdに記載
