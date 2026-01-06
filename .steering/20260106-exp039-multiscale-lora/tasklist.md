# Tasklist

## 完了条件（Definition of Done）
- [ ] 5-fold CVが完了し、スコアが記録されている
- [ ] doc/experiment/exp039.md にCV結果が記載されている
- [ ] exp035, exp038との比較考察が完了している

## タスク

### Phase 1: 準備
- [x] Done: Steeringファイル作成
- [x] Done: doc/experiment/exp039.md 作成

### Phase 2: 実装
- [x] Done: src/exp039/ ディレクトリ作成（exp038ベース）
- [x] Done: models.py 修正（Multiscale forward追加）
- [x] Done: dataset.py 確認（config経由で1024x1024対応済み）
- [x] Done: config/exp039.yaml 作成
- [x] Done: その他ファイル確認・修正（inference.py, evaluation.py）

### Phase 3: 検証
- [ ] TODO: Fold 0 で動作確認
- [ ] TODO: メモリ使用量確認
- [ ] TODO: 学習曲線確認

### Phase 4: 本実験
- [ ] TODO: 5-fold CV実行
- [ ] TODO: 結果記録
- [ ] TODO: 考察

### Phase 5: 完了
- [ ] TODO: exp039.md 更新（結果・考察）
- [ ] TODO: LB提出検討
