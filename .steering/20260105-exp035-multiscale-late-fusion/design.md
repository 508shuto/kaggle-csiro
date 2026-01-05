# Design

## アプローチ
exp028のモデルアーキテクチャをベースに、マルチスケール入力とLate Fusionを追加

## アーキテクチャ

```
CSIROModel (exp035)
│
├── backbone: DINOv3 ViT-Large (frozen)
├── head: Linear(1024, 256) → ReLU → Dropout → Linear(256, 3)
├── aux_head: Linear(1024, 256) → ReLU → Dropout → Linear(256, 2)
│
└── forward(x):  # x: (B, 3, 1024, 1024)
    │
    ├── Global branch:
    │   x_global = resize(x, 512)
    │   feat_global = backbone(x_global)
    │   pred_global = head(feat_global)  # (B, 3)
    │
    ├── Tile branch:
    │   tiles = split_2x2(x)  # (B*4, 3, 512, 512)
    │   feat_tiles = backbone(tiles)
    │   pred_tiles = head(feat_tiles)  # (B*4, 3)
    │   pred_tiles = reshape(pred_tiles, (B, 4, 3))
    │   pred_tile_avg = mean(pred_tiles, dim=1)  # (B, 3)
    │
    └── Fusion:
        pred = (pred_global + pred_tile_avg) / 2
```

## 変更コンポーネント

| ファイル | 変更内容 |
|---------|---------|
| `src/exp035/models.py` | マルチスケールモデル実装 |
| `src/exp035/dataset.py` | 1024×1024リサイズ対応 |
| `config/exp035.yaml` | 新規設定ファイル |
| `src/exp035/train.py` | exp028からコピー（変更なし） |
| `src/exp035/evaluation.py` | exp028からコピー（変更なし） |
| `src/exp035/inference.py` | exp028からコピー（変更なし） |

## 影響範囲
- 新規実験ディレクトリ作成のみ
- 既存コードへの影響なし

## メモリ考慮
- タイル4枚を一度にbackboneに通す → メモリ増加
- 必要に応じて順次処理に変更可能:
  ```python
  for tile in tiles:
      pred = predict(tile)
  ```
