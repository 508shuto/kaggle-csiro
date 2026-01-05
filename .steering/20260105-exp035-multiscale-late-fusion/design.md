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

## 実装上の修正履歴

### バグ修正（commit: 3b52cce）
1. **Critical: Reshape bug** (models.py:129)
   - 誤: `pred_tiles.view(4, B, 3)` → 正: `pred_tiles.view(B, 4, 3)`
   - 影響: 異なる画像のタイルが混ざって平均されていた
2. **Memory leak** (lightning_module.py:102-103)
   - validation時に `.detach()` 追加
3. **Redundant clamping** (lightning_module.py:90-91)
   - モデル側で実施済みのため削除

### 設計変更（commit: a253bbd, a298fa0）
1. **Input validation追加** (models.py:106)
   - 1024x1024入力を強制チェック
2. **Validation pattern改善** (lightning_module.py:81-84)
   - `on_validation_epoch_start()` フック使用
3. **Auxiliary loss weight validation** (lightning_module.py:48-55)
   - `aux_weight > 1.0` で警告
4. **Augmentation設定統一** (config/exp035.yaml)
   - bool → float (確率値) に統一
   - gamma_transform, gaussian_noise, blur を実装
5. **Mixup無効化** (config/exp035.yaml:77)
   - マルチスケールアーキテクチャと非互換のため `enabled: false`
