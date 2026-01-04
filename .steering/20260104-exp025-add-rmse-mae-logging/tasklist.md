# Tasklist

## 完了条件（Definition of Done）
- [x] RMSEとMAEがMetricCollectionに追加されている
- [x] val_rmseとval_maeがログされる（全体）
- [x] val_rmse_{class_name}とval_mae_{class_name}がログされる（各クラス）
- [x] 既存のR2スコアパターンと一貫性がある
- [ ] ruff format/checkが通る（要承認）
- [ ] コミット・プッシュ完了

## タスク
- [x] Done: torchmetricsからMeanSquaredError, MeanAbsoluteErrorをインポート
- [x] Done: MetricCollectionにrmse, maeを追加
- [x] Done: validation_stepで予測値とターゲットを蓄積
- [x] Done: on_validation_epoch_endでクラスごとのRMSE/MAEを計算
- [x] Done: on_validation_epoch_endでval_rmse, val_maeをログ（全体）
- [x] Done: on_validation_epoch_endでval_rmse_*, val_mae_*をログ（各クラス）
- [x] Done: requirements.md, design.mdを更新
- [ ] Todo: ruff format/check実行（要承認）
- [ ] Todo: コミット・プッシュ
