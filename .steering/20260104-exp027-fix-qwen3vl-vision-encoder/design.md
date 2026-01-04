# Design

## アプローチ

`vlm.model.get_image_features()`と`AutoImageProcessor`を使用するアプローチを採用。

**理由**:
- Qwen3VLの内部実装に依存せず、公式APIを使用
- ピクセル前処理を自前実装する必要がない
- コードがシンプルで保守しやすい

**注意**: `Qwen3VLImageProcessor`は存在しないため`AutoImageProcessor`を使用。

## Qwen3VLモデル構造

```
Qwen3VLForConditionalGeneration
└── .model (Qwen3VLModel)
    ├── .visual (Qwen3VLVisionModel)  ← 正しい属性
    │   ├── .patch_embed (Qwen3VLVisionPatchEmbed)
    │   ├── .blocks (Vision Transformer blocks)
    │   └── .merger (出力投影)
    └── .language_model
```

## 変更コンポーネント

### 1. models.py

**変更内容**:
- 属性アクセス: `.vision_model` → `.visual`
- hidden_dim: 1536 → 2048（Qwen3-VL-2Bの実際のout_hidden_size）
- config検証: `hidden_size` → `out_hidden_size`
- `AutoImageProcessor`のインポート・初期化追加
- forward関数の書き換え

**新しいforward処理フロー**:
```
PIL画像リスト
    ↓ AutoImageProcessor
Qwen3VL形式 (pixel_values, image_grid_thw)
    ↓ vlm.model.get_image_features()
画像特徴量 (tuple of tensors)
    ↓ mean pooling
プーリング済み特徴 (B, 2048)
    ↓ regression heads
予測値 (B, 5), (B, 2)
```

### 2. dataset.py

**変更内容**:
- `_get_transforms`: `ToTensor()`と`Normalize()`を削除
- `__getitem__`: PIL.Imageを返す

### 3. train.py

**変更内容**:
- カスタム`collate_fn`追加
- DataLoader作成時に`collate_fn`指定

### 4. config/exp027.yaml

**変更内容**:
- `model.hidden_dim`: 1536 → 2048 (Qwen3-VL-2Bの実際のout_hidden_size)

### 5. lightning_module.py

**変更内容**:
- Mixup処理を一時無効化（config側で対応可能）

### 6. inference.py

**変更内容**:
- `collate_fn`追加（PIL画像リスト用）
- `TestDataset._get_transforms()`: ToTensor/Normalize削除
- `TestDataset.__getitem__()`: PIL画像を返す
- `predict()`: PIL画像リスト対応、TTA非対応（警告表示）
- DataLoaderに`collate_fn`指定、`pin_memory=False`

### 7. evaluation.py

**変更内容**:
- `collate_fn`追加
- DataLoaderに`collate_fn`指定
- 推論ループ: PIL画像リスト対応

## 影響範囲

| コンポーネント | 影響 |
|--------------|------|
| models.py | 大（forward完全書き換え）|
| dataset.py | 中（transform変更）|
| train.py | 小（collate_fn追加）|
| config | 小（hidden_dim変更）|
| lightning_module.py | 小（mixup無効化）|
| inference.py | 中（TestDataset変更、predict関数変更、TTA無効化）|
| evaluation.py | 小（collate_fn追加、推論ループ変更）|

## 実装時の発見

1. **hidden_dim**: 当初3584と想定していたが、Qwen3-VL-2Bの実際の`out_hidden_size`は2048
2. **ImageProcessor**: `Qwen3VLImageProcessor`は存在せず、`AutoImageProcessor`を使用する必要あり
3. **Mixup**: PIL画像入力ではTensor前提のMixupが動作しないためconfig側で無効化
