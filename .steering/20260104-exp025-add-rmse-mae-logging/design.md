# Design

## アプローチ
torchmetricsの既存MetricCollectionにRMSE/MAEメトリクスを追加し、on_validation_epoch_endでログする。

## 変更コンポーネント
- `src/exp025/lightning_module.py`
  - インポート追加: `MeanSquaredError`, `MeanAbsoluteError`
  - MetricCollectionに`rmse`と`mae`を追加
  - on_validation_epoch_endで`val_rmse`と`val_mae`をログ

## 影響範囲
- W&B/TensorBoardに新しいメトリクス（val_rmse, val_mae）が追加される
- 既存の機能には影響なし
