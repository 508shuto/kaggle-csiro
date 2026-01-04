# Design

## アプローチ

### Tile処理フロー
```
入力画像 (B, 3, 512, 512)
    ↓
2×2グリッド分割（重なりなし）
    ↓
4タイル (B, 4, 3, 256, 256) → reshape → (B*4, 3, 256, 256)
    ↓
backbone(ViT) → (B*4, 1024)
    ↓
reshape → (B, 4, 1024)
    ↓
GeM Pooling → (B, 1024)
    ↓
Head → 予測
```

### GeM Pooling
Generalized Mean Pooling: `( (1/n) * Σ x^p )^(1/p)`
- p=1: Average Pooling
- p→∞: Max Pooling
- p=3 (初期値、学習可能)

```python
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if learnable else p
        self.eps = eps

    def forward(self, x):
        # x: (B, num_tiles, features)
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)
```

## 変更コンポーネント

| ファイル | 変更内容 |
|---------|---------|
| src/exp029/models.py | Tile分割、GeM Pooling追加 |
| src/exp029/dataset.py | exp028からコピー（変更なし） |
| config/exp029.yaml | exp028ベース、tile設定追加 |

## 影響範囲
- モデルのforward処理のみ変更
- 損失関数、データセット、学習ループは変更なし
