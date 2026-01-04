# Design

## アプローチ
1. torchmetricsの既存MetricCollectionにRMSE/MAEメトリクスを追加（全体メトリクス用）
2. validation_stepで予測値とターゲットを蓄積
3. on_validation_epoch_endでクラスごとのRMSE/MAEを手動計算してログ

## 変更コンポーネント
- `src/exp025/lightning_module.py`
  - インポート追加: `MeanSquaredError`, `MeanAbsoluteError`
  - MetricCollectionに`rmse`と`mae`を追加（全体メトリクス）
  - validation_stepで予測値とターゲットをリストに蓄積
  - on_validation_epoch_endで以下をログ:
    - `val_rmse`: 全体のRMSE
    - `val_mae`: 全体のMAE
    - `val_rmse_{class_name}`: クラスごとのRMSE（Ca, P, pH, SOC, Sand）
    - `val_mae_{class_name}`: クラスごとのMAE（Ca, P, pH, SOC, Sand）

## 影響範囲
- W&B/TensorBoardに新しいメトリクス（val_rmse, val_mae, val_rmse_*, val_mae_*）が追加される
- 既存の機能には影響なし
- 既存のR2スコアのロギングパターンと一貫性を保つ
