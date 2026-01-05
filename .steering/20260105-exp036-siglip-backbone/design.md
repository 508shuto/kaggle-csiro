# Design

## アプローチ
- exp028のコードをそのまま流用
- timmライブラリのSigLIPモデルを使用
- モデル: vit_large_patch16_siglip_384

## 変更コンポーネント
- config/exp036.yaml: model name, image_size
- src/exp036/: exp028からコピー（コード変更なし）

## 影響範囲
- 新規実験のみ、既存実験への影響なし
