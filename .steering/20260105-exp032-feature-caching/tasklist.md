# Tasklist

## 完了条件（Definition of Done）

1. `extract_features.py` で全画像の特徴を抽出・保存できる
2. `--use-cache` フラグで高速学習モードに切り替えられる
3. キャッシュモードで学習が正常に完了する
4. 非キャッシュモードでも既存通り動作する
5. 学習時間が 80%+ 削減される

## タスク

### Phase 1: 特徴抽出スクリプト
- [x] Done: `src/exp032/extract_features.py` 新規作成
  - Vision encoder ロード
  - DataLoader でバッチ処理
  - 特徴抽出 + mean pooling
  - `.npy` 形式で保存

### Phase 2: Dataset & Model
- [x] Done: `CachedFeatureDataset` を `dataset.py` に追加
  - `.npy` 読み込み
  - targets, aux_targets の取得

- [x] Done: `RegressionHeadOnly` を `models.py` に追加
  - head + aux_head のみ
  - physical constraints 維持

### Phase 3: 学習スクリプト対応
- [x] Done: `train.py` に `--use-cache`, `--feature-dir` 引数追加
- [x] Done: キャッシュモード時の分岐処理実装

### Phase 4: LightningModule 対応
- [x] Done: `lightning_module.py` に `CachedCSIROModule` 追加
  - 軽量 training_step
  - aux_loss キャッシュ化（2回計算を1回に）

### Phase 5: 動作確認
- [ ] TODO: 特徴抽出の実行確認
- [ ] TODO: キャッシュモードでの学習実行確認
- [ ] TODO: 学習時間の比較計測
