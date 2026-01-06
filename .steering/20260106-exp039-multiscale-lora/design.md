# Design

## アプローチ

exp038をベースに、exp035のMultiscale処理を統合する。

### アーキテクチャ

```
CSIROModel (exp039)
├── Backbone: HF DINOv3 Large + LoRA (r=8, alpha=16, layers 18-23)
│
├── Global Branch:
│   ├── Input: (B, 3, 1024, 1024)
│   ├── Resize: 1024x1024 → 512x512
│   └── backbone(LoRA) → head → pred_global (B, 3)
│
├── Tile Branch:
│   ├── Input: (B, 3, 1024, 1024)
│   ├── Split: 2x2 → (B, 4, 3, 512, 512) → (B*4, 3, 512, 512)
│   └── backbone(LoRA) × 4 → head × 4 → mean → pred_tile (B, 3)
│
└── Late Fusion:
    ├── pred_3 = (pred_global + pred_tile) / 2
    ├── Physical constraints (clamp, GDM, Total)
    └── Output: (B, 5)
```

### LoRA設定
```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=r"layer\.(1[89]|2[0-3])\.attention\.(q|k|v|o)_proj",
    lora_dropout=0.1,
    bias="none",
)
```

## 変更コンポーネント

| ファイル | ベース | 変更内容 |
|---------|--------|---------|
| `src/exp039/models.py` | exp038 | Multiscale forward追加 |
| `src/exp039/dataset.py` | exp035 | 1024x1024対応 |
| `config/exp039.yaml` | exp038 | 画像サイズ・バッチサイズ・Mixup変更 |
| `src/exp039/lightning_module.py` | exp038 | ほぼそのまま |
| `src/exp039/train.py` | exp038 | ほぼそのまま |

## 影響範囲

### メモリ・計算コスト
- Forward pass: 5倍 (exp038比)
- バッチサイズ: 16 → 4 (gradient accumulation 4)
- 学習時間: 約5倍増加見込み

### データ処理
- 入力解像度: 512 → 1024
- Augmentation: Mixup無効化必須
