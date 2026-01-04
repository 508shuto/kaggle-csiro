# Requirements

## 目的
exp026のcheckpointファイル名を`fold{fold}_epoch={epoch}-val_score={val_score}.ckpt`形式に統一し、exp025同様にRMSE/MAE（全体＋per-class）をvalidationでロギングする。

## 変更/追加する機能
1. checkpointファイル名の変更
   - 現在: `last.ckpt` → `fold{fold}_epoch{epoch:02d}.ckpt` にリネーム
   - 変更後: `last.ckpt` → `fold{fold}_epoch={epoch}-val_score={val_score:.4f}.ckpt` にリネーム
   - epochはcheckpointファイルから取得（`torch.load`で`ckpt["epoch"]`を読み取り）
   - val_scoreは`trainer.callback_metrics["val_score"]`から取得

2. RMSE/MAEロギングの追加
   - validationで全体の`val_rmse`と`val_mae`をロギング
   - 各クラスごとの`val_rmse_{class_name}`と`val_mae_{class_name}`をロギング
   - exp025の実装と同様の形式で実装

## 制約・前提条件
- exp025の実装を参考にする
- 既存のsteeringフォルダ（`.steering/20260104-exp026-freeze-dinov2-features/`）は変更しない
- checkpointのリネーム処理は`trainer.fit()`後に実行する
- val_scoreが取得できない場合は明確なエラーメッセージで停止する（nanで保存しない）

