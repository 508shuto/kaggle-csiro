# Tasklist

## 完了条件（Definition of Done）
- `src/exp025/create_dataset.py` がJPEGファイルパスを `image_path` に格納する
- `src/exp025/dataset.py` がJPEGファイルをPILで直接読み込む
- `src/exp025/inference.py` がテスト画像もJPEGから直接読み込む
- `img2npy.py` を実行せずに学習・評価・推論が実行できる
- 既存の機能（log1p廃止、SmoothL1Loss）に影響がない

## タスク
- [x] Done: `src/exp025/create_dataset.py` を修正し、`image_path` にJPEGファイルパスを格納するように変更する
- [x] Done: `src/exp025/dataset.py` を修正し、`np.load()` を廃止してPILでJPEGを読み込むように変更する
- [x] Done: `src/exp025/inference.py` の `build_test_index` を修正し、テスト画像もJPEGパスを参照するように変更する
- [x] Done: `src/exp025/inference.py` の `TestDataset` を修正し、JPEGを直接読み込むように変更する
- [x] Done: スモークチェック（compileall、import確認）を実行する
- [ ] TODO: 実際に学習・推論が実行できることを確認する

