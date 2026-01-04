# Design

## アプローチ
EXP025の既存実装を維持しつつ、画像読み込み部分のみをJPEG直接読み込みに変更する：
1. `create_dataset.py`: `image_path` にJPEGファイルのフルパス（`input/train/xxx.jpg`）を格納
2. `dataset.py`: `np.load()` を廃止し、PILでJPEGを読み込んでNumPy配列に変換してからAlbumentationsに渡す
3. `inference.py`: `build_test_index` で `input/test/` のJPEGファイルを参照し、`TestDataset` でもJPEGを直接読み込む

## 変更コンポーネント
- `src/exp025/create_dataset.py`: 
  - 52行目の `.npy` パス生成をJPEGパス（`input/train/xxx.jpg`）に変更
- `src/exp025/dataset.py`:
  - 23行目の `np.load()` をPIL読み込みに変更（`PIL.Image.open()` → `np.array()` → float32変換）
- `src/exp025/inference.py`:
  - `build_test_index`: 59行目の `.npy` パス生成をJPEGパス（`input/test/xxx.jpg`）に変更
  - `TestDataset`: 82行目の `np.load()` をPIL読み込みに変更

## 影響範囲
- 学習・評価・推論のすべてのパイプラインで`.npy`ファイルへの依存がなくなる
- `img2npy.py` の実行が不要になる
- 画像読み込み処理が若干変更されるが、Albumentationsへの入力形式（NumPy配列）は維持されるため、後続処理への影響は最小限

