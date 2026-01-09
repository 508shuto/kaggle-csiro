# Tasklist

## 完了条件（Definition of Done）
- [ ] InverseWeightedSmoothL1Loss が正常に動作する
- [ ] 3-fold CVが完走する
- [ ] CVスコアがexp043と比較可能
- [ ] Q1-Q4領域別MSEがexp043と比較可能

## タスク

### Phase 1: 実装
- [x] Done: `src/exp044/loss.py` に `InverseWeightedSmoothL1Loss` 追加
- [x] Done: `src/exp044/utils.py` の `get_loss_fn` 修正
- [x] Done: `config/exp044.yaml` 修正

### Phase 2: 検証
- [ ] TODO: 学習実行（3-fold CV）
- [ ] TODO: CV結果確認
- [ ] TODO: Q1-Q4領域別MSE比較（exp043 vs exp044）

### Phase 3: ドキュメント
- [ ] TODO: `doc/experiment/exp044.md` 結果追記
