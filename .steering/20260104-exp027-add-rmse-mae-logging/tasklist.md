# Tasklist

## 完了条件（Definition of Done）

- [x] `src/exp027/lightning_module.py`にRMSE/MAEメトリクスが追加されている
- [x] validation epoch終了時に`val_rmse`/`val_mae`がログされる（lines 167-173）
- [x] validation epoch終了時に各クラスの`val_rmse_{class}`/`val_mae_{class}`がログされる（lines 122-140）
- [ ] `uv run src/exp027/train.py --folds 0`を実行し、validation後にログが出力されることを確認（手動確認が必要）

## タスク

### Phase 1: lightning_module.py修正

- [x] Done: importに`MeanSquaredError`, `MeanAbsoluteError`を追加 (line 9)
- [x] Done: `self.metrics`の`MetricCollection`に`"rmse"`と`"mae"`を追加 (lines 41-42)
- [x] Done: `validation_step()`で`self.val_preds`/`self.val_targets`にpreds/targetsを蓄積 (lines 101-106)
- [x] Done: `on_validation_epoch_end()`で全体の`val_rmse`/`val_mae`をログ (lines 167-173)
- [x] Done: `on_validation_epoch_end()`でクラス別`val_rmse_{class}`/`val_mae_{class}`を計算・ログ (lines 122-140)
- [x] Done: epoch終了時に`val_preds`/`val_targets`リストをクリア (lines 139-140)

### Phase 2: 検証

- [ ] TODO: `uv run src/exp027/train.py --folds 0`を実行
- [ ] TODO: validation後に`val_rmse`, `val_mae`がログされることを確認
- [ ] TODO: validation後に各クラスの`val_rmse_{class}`, `val_mae_{class}`がログされることを確認

