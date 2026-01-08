# Requirements

## 目的

exp027のvalidationにRMSE/MAEロギング（全体＋クラス別）を追加し、exp025と同様のメトリクス監視を可能にする。

## 変更/追加する機能

1. **RMSE/MAEメトリクスの追加**
   - `torchmetrics`の`MeanSquaredError(squared=False)`と`MeanAbsoluteError()`を`MetricCollection`に追加
   - validation epoch終了時に全体の`val_rmse`/`val_mae`をログ

2. **クラス別RMSE/MAEロギング**
   - `validation_step()`で予測値とターゲットを蓄積
   - `on_validation_epoch_end()`で各クラスごとに`val_rmse_{class}`/`val_mae_{class}`を計算・ログ

## 制約・前提条件

- exp025の実装を参考にする
- validationのみに適用（trainには追加しない）
- 既存のR2スコアロギングと併存する
- 学習/推論の計算ロジックは変更しない（ログ追加のみ）



