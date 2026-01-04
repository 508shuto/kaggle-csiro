# Tasklist

## 完了条件（Definition of Done）
- [ ] 推論速度が改善されていること（評価スクリプトで確認）
- [ ] 学習が正常に動作すること
- [ ] 既存チェックポイントで推論できること

## タスク
- [x] Done: dataset.pyにAutoImageProcessor追加、tensor返却に変更
- [x] Done: models.pyからprocessor削除、forward引数変更
- [x] Done: train.pyのcollate_fn更新
- [x] Done: lightning_module.pyのbatch展開更新
- [x] Done: evaluation.pyのcollate_fn更新
- [x] Done: inference.pyのTestDataset修正
- [ ] In Progress: 動作確認
