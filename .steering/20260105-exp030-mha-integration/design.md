# Design

## アプローチ

### MHA特徴統合フロー（Option B: backbone特徴 + 補助予測値）
```
画像 → Tile処理 → backbone → 画像特徴 (B, 1024)
                                    ↓
                              ┌─────┴─────┐
                              ↓           ↓
                          Main Proj    Aux Head
                          (B, 256)        ↓
                              ↓       aux_pred (B, 2)
                              ↓           ↓
                              ↓       Aux Embed
                              ↓       (B, 256)
                              ↓           ↓
                              └─────┬─────┘
                                    ↓
                          Multi Head Attention
                          (Self-Attention + FFN)
                                    ↓
                            統合特徴 (B, 256)
                                    ↓
                            Final Head
                                    ↓
                            最終予測 (B, 3)
```

### Multi Head Attention設計
```python
class FeatureFusion(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, main_feat, aux_feat):
        # Stack as sequence: (B, 2, dim)
        x = torch.stack([main_feat, aux_feat], dim=1)

        # Self-attention with residual
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        x = self.norm2(x + self.dropout(self.ffn(x)))

        # Take main feature position
        return x[:, 0, :]  # (B, 256)
```

### 追加モジュール
```python
# Main projection: backbone features → mha_dim
self.main_proj = nn.Sequential(
    nn.Linear(self.model.num_features, mha_dim),
    nn.ReLU(),
    nn.Dropout(0.2),
)

# Auxiliary prediction embedding: aux_pred → mha_dim
self.aux_embed = nn.Sequential(
    nn.Linear(aux_out_channels, mha_dim),
    nn.ReLU(),
)

# Final head: fused features → predictions
self.final_head = nn.Linear(mha_dim, out_channels)
```

## 変更コンポーネント

| ファイル | 変更内容 |
|---------|---------|
| src/exp030/models.py | FeatureFusion, aux_embed, main_proj, final_head追加 |
| src/exp030/lightning_module.py | MHA設定渡し |
| config/exp030.yaml | MHA設定追加 |

## パラメータ数
- Total: 86,826,758
- Trainable: 1,185,542（exp029: 395,014の約3倍）

## 影響範囲
- モデル構造のみ変更
- 損失関数、データセット、学習ループは変更なし
