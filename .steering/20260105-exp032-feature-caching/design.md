# Design

## アプローチ

### 二段階学習パイプライン

```
[Phase 1: 特徴抽出 (1回のみ)]
train.csv → DataLoader → Qwen3-VL Vision Encoder → Mean Pooling → .npy保存

[Phase 2: Head学習 (高速)]
.npy読み込み → RegressionHeadOnly → Loss → Backward
```

## 変更コンポーネント

### 1. extract_features.py (新規)

```
入力: train.csv, test.csv, config
処理: Vision encoder forward + mean pooling
出力: output/exp032/features/{image_id}.npy
```

**主要な処理:**
- 既存の `Qwen3VLRegressionModel` をロード
- `model.vlm.model.get_image_features()` で特徴抽出
- バッチ処理で効率化

### 2. dataset.py (追加)

**新規クラス:** `CachedFeatureDataset`
- `__getitem__`: `.npy` ファイルから特徴ベクトルを読み込み
- 画像処理不要 → DataLoader 高速化

### 3. models.py (追加)

**新規クラス:** `RegressionHeadOnly`
- Vision encoder なし
- 既存の head 構造を再利用
- Physical constraints (clover + green = gdm 等) は維持

### 4. train.py (修正)

**追加引数:**
- `--use-cache`: Feature Caching モード有効化
- `--feature-dir`: キャッシュディレクトリ指定

**分岐:**
```python
if args.use_cache:
    dataset = CachedFeatureDataset(...)
    model = RegressionHeadOnly(...)
else:
    dataset = CSIRODataset(...)
    model = Qwen3VLRegressionModel(...)
```

### 5. lightning_module.py (修正)

**キャッシュモード用 LightningModule:**
- 入力: pooled_features (2048次元)
- Vision encoder forward 不要
- EMA は head のみに適用

## 影響範囲

| ファイル | 影響 |
|----------|------|
| `src/exp032/extract_features.py` | 新規作成 |
| `src/exp032/dataset.py` | CachedFeatureDataset クラス追加 |
| `src/exp032/models.py` | RegressionHeadOnly クラス追加 |
| `src/exp032/train.py` | --use-cache 引数追加、分岐追加 |
| `src/exp032/lightning_module.py` | CachedCSIROModule 追加 |
| `src/exp032/inference.py` | 変更なし（フルモデル使用） |
| `src/exp032/evaluation.py` | 変更なし |

## 技術的考慮事項

### メモリ効率
- 特徴ベクトル: 2048 × float32 = 8KB/画像
- 全画像キャッシュ: ~80MB (管理可能)

### GPU メモリ
- フルモデル: ~8GB
- Head のみ: ~1GB → バッチサイズ大幅増加可能

### 並列処理
- 特徴抽出: GPU並列
- キャッシュ読み込み: CPU並列 (num_workers)
