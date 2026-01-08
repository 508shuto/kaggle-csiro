# Design

## アプローチ

exp025の実装を参考に、exp027の`lightning_module.py`にRMSE/MAEロギングを追加する。

**理由**:
- exp025で既に実装済みのパターンを再利用できる
- torchmetricsの`MetricCollection`に追加するだけなので影響範囲が小さい
- クラス別メトリクスも同じロジックで実装可能

## 変更コンポーネント

### 1. `src/exp027/lightning_module.py`

**変更内容**:
- import追加: `torchmetrics`から`MeanSquaredError`, `MeanAbsoluteError`を追加
- `__init__()`: `self.metrics`の`MetricCollection`に`"rmse"`と`"mae"`を追加
- `validation_step()`: clamp後の`preds`と`targets`を`self.val_preds`/`self.val_targets`リストに蓄積
- `on_validation_epoch_end()`:
  - `metrics.compute()`から全体の`val_rmse`/`val_mae`をログ
  - 蓄積した`val_preds`/`val_targets`を`torch.cat`で結合し、クラス別RMSE/MAEを計算してログ
  - リストをクリア

**実装フロー**:
```
validation_step()
  ↓ preds, targetsを蓄積 (self.val_preds, self.val_targets)
  ↓ metrics.update(preds, targets)

on_validation_epoch_end()
  ↓ metrics.compute() → val_rmse, val_maeをログ
  ↓ torch.cat(val_preds) → クラス別RMSE/MAE計算 → ログ
  ↓ リストクリア
```

## 影響範囲

| コンポーネント | 影響 |
|--------------|------|
| lightning_module.py | 中（import追加、metrics追加、validation_step/epoch_end修正）|
| その他 | なし（ログ追加のみ）|

## 参考実装

- exp025の`lightning_module.py`（lines 36-37, 96-101, 117-135, 162-168）を参考にする


