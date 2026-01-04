# Design

## アプローチ
1. checkpointファイル名の変更（`train.py`）
   - `trainer.fit()`後に`last.ckpt`を検出
   - checkpointファイルを`torch.load`して`ckpt["epoch"]`を取得（`trainer.current_epoch`の+1ズレを避ける）
   - `trainer.callback_metrics["val_score"]`からval_scoreを取得（`torch.Tensor`の場合は`.item()`で`float`化）
   - `fold{fold}_epoch={epoch}-val_score={val_score:.4f}.ckpt`にリネーム
   - val_scoreが取得できない場合は`ValueError`を発生

2. RMSE/MAEロギングの追加（`lightning_module.py`）
   - `MetricCollection`に`MeanSquaredError(squared=False)`と`MeanAbsoluteError()`を追加
   - `validation_step`で`preds`と`targets`をリストに蓄積
   - `on_validation_epoch_end`でper-class RMSE/MAEを計算してログ
   - 全体の`val_rmse`と`val_mae`もログ

## 変更コンポーネント
| ファイル | 変更内容 |
|---------|---------|
| `src/exp026/train.py` | checkpointリネーム処理を修正（epochとval_scoreを含む形式に変更） |
| `src/exp026/lightning_module.py` | RMSE/MAEメトリクスの追加とロギング実装 |

## 影響範囲
- `evaluation.py`、`inference.py`は変更なし（checkpointファイル名のパターンは`fold{fold}*.ckpt`で検索しているため）
- W&Bログに`val_rmse`、`val_mae`、`val_rmse_{class}`、`val_mae_{class}`が追加される

