# Design

## アプローチ
1. exp040をベースにPartial Unfreezeを実装
2. 最後の2ブロックのみ`requires_grad=True`に設定
3. 別々のparam_groupsで異なる学習率を適用

## 変更コンポーネント

### models.py
```python
# 最後の2ブロック（blocks.22-23）のみ学習
for name, param in self.model.named_parameters():
    if "blocks.22" in name or "blocks.23" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```

### lightning_module.py
```python
def configure_optimizers(self):
    backbone_params = []
    head_params = []
    for name, param in self.model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name or "aux_head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": 1e-5},  # 低学習率
        {"params": head_params, "lr": 1e-3},      # 通常学習率
    ]
    optimizer = AdamW(param_groups, weight_decay=1e-2)
```

### config/exp045.yaml
```yaml
model:
  freeze_backbone: false  # Partial Unfreeze有効化
  unfreeze_blocks: [22, 23]

trainer:
  train:
    backbone_lr: 1e-5
    head_lr: 1e-3
```

## 影響範囲
- 学習時間: やや増加（2ブロック分のbackprop追加）
- メモリ使用量: やや増加（勾配計算分）
- CV/LB: 改善見込み

## リスク
- 2ブロックでは表現力不足の可能性 → 3-4ブロックに拡張を検討
- 学習率設定が不適切だと収束しない → ハイパーパラメータチューニング必要
