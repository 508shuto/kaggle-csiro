# Requirements

## 目的
EXP025でimg2npyによる前処理を廃止し、`input/train/` および `input/test/` のJPEG画像を直接読み込むように変更する。これにより、前処理ステップを削減し、パイプラインを簡素化する。

## 変更/追加する機能
- **img2npyの廃止**: `img2npy.py` による`.npy`変換処理を不要にする
- **JPEG直接読み込み**: `input/train/` と `input/test/` のJPEGファイルをPILで直接読み込む
- **create_dataset.pyの変更**: `image_path` 列にJPEGファイルのパスを格納するように変更（`.npy`パスではなく）
- **dataset.pyの変更**: `np.load()` を廃止し、PILでJPEGを読み込んでNumPy配列に変換
- **inference.pyの変更**: テスト画像もJPEGから直接読み込むように変更

## 制約・前提条件
- EXP025の既存機能（log1p廃止、SmoothL1Loss）は維持
- 画像の前処理（リサイズ、正規化等）はAlbumentationsで行うため、JPEG読み込み後の処理は変更なし
- `input/train/` と `input/test/` にJPEGファイルが存在することを前提とする

