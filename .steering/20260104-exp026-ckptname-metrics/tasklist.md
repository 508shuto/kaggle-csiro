# Tasklist

## 完了条件（Definition of Done）
- [x] checkpointファイルが`fold{fold}_epoch={epoch}-val_score={val_score:.4f}.ckpt`形式で保存される
- [x] validationログに`val_rmse`と`val_mae`が出力される
- [x] validationログに各クラスの`val_rmse_{class_name}`と`val_mae_{class_name}`が出力される
- [x] コードがexp025の実装と整合性がある

## タスク
- [x] Done: 新規steeringフォルダ作成（requirements/design/tasklist）
- [x] Done: `src/exp026/train.py`のcheckpointリネーム処理を修正
- [x] Done: `src/exp026/lightning_module.py`にRMSE/MAEメトリクスを追加

