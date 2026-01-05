# Tasklist

## 完了条件（Definition of Done）
- [x] Done: exp035のコードが実装され、エラーなく動作する（バグ修正完了）
- [ ] TODO: 5-fold CVが完了し、スコアが記録される
- [ ] TODO: exp028との比較結果がドキュメントに記載される

## タスク

### 1. ディレクトリ・ファイル準備
- [x] Done: `src/exp035/` ディレクトリ作成
- [x] Done: exp028から基本ファイルをコピー
  - train.py
  - evaluation.py
  - inference.py
  - lightning_module.py
  - dataset.py

### 2. モデル実装
- [x] Done: `src/exp035/models.py` - マルチスケールモデル実装
  - Global branch (512×512)
  - Tile branch (512×512 × 4)
  - Late Fusion

### 3. データセット修正
- [x] Done: `src/exp035/dataset.py` - 1024×1024リサイズ対応（config経由）

### 4. 設定ファイル
- [x] Done: `config/exp035.yaml` 作成
  - image_size: 1024
  - batch_size: 4（メモリ考慮）
  - accumulate_grad_batches: 4（effective batch size = 16）

### 5. 実験実行
- [ ] TODO: 5-fold学習実行
- [ ] TODO: evaluation実行
- [ ] TODO: 結果をドキュメントに記録

### 6. 分析・ドキュメント
- [x] Done: `doc/experiment/exp035.md` 作成（結果はCV実行後に記入）
- [ ] TODO: exp028との比較分析
- [ ] TODO: CV結果を `doc/experiment/exp035.md` に記入
