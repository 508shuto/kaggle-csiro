# Requirements

## 目的
exp032: Feature Caching による学習時間の大幅短縮（目標: 80-90%削減）

## ベースライン
- exp027: Qwen3-VL-2B + Frozen backbone + Regression heads

## 変更/追加する機能

### Feature Caching システム
1. **特徴抽出スクリプト** (`extract_features.py`)
   - Qwen3-VL vision encoder で全画像の特徴を事前抽出
   - pooled features (2048次元) を `.npy` 形式で保存

2. **キャッシュ用Dataset** (`CachedFeatureDataset`)
   - 画像ではなくキャッシュ済み特徴を読み込む
   - DataLoader のボトルネック解消

3. **軽量モデル** (`RegressionHeadOnly`)
   - Vision encoder なし、regression head のみ
   - GPU メモリ大幅削減

4. **学習スクリプト対応**
   - `--use-cache` フラグで切り替え
   - キャッシュモード時の軽量 LightningModule

## 制約・前提条件

1. **Augmentation 制限**
   - Feature Caching では画像 augmentation 不可
   - 現状の exp027 は Resize のみなので問題なし

2. **互換性維持**
   - 推論時 (`inference.py`) は既存のままでOK
   - フルモデルのチェックポイントを出力

3. **ディスク容量**
   - 特徴ファイル: ~80MB (10,000画像想定)

4. **既存機能への影響**
   - 非キャッシュモードでの学習は既存通り動作すること
