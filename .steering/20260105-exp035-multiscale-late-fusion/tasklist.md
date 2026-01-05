# Tasklist

## 完了条件（Definition of Done）
- [ ] exp035のコードが実装され、エラーなく動作する
- [ ] 5-fold CVが完了し、スコアが記録される
- [ ] exp028との比較結果がドキュメントに記載される

## タスク

### 1. ディレクトリ・ファイル準備
- [ ] TODO: `src/exp035/` ディレクトリ作成
- [ ] TODO: exp028から基本ファイルをコピー
  - train.py
  - evaluation.py
  - inference.py
  - lightning_module.py
  - dataset.py

### 2. モデル実装
- [ ] TODO: `src/exp035/models.py` - マルチスケールモデル実装
  - Global branch (512×512)
  - Tile branch (512×512 × 4)
  - Late Fusion

### 3. データセット修正
- [ ] TODO: `src/exp035/dataset.py` - 1024×1024リサイズ対応

### 4. 設定ファイル
- [ ] TODO: `config/exp035.yaml` 作成
  - image_size: 1024
  - use_multiscale: true

### 5. 実験実行
- [ ] TODO: 5-fold学習実行
- [ ] TODO: evaluation実行
- [ ] TODO: 結果をドキュメントに記録

### 6. 分析・ドキュメント
- [ ] TODO: exp028との比較分析
- [ ] TODO: `doc/experiment/exp035.md` 結果・考察記入
