# Requirements

## 目的
exp037 の訓練時に発生した2つのエラーを解消する：
1. WeightedR2Score メトリクスの DDP 環境での集約エラー
2. DDP 未使用パラメータ検出エラー

## 変更/追加する機能

### 1. WeightedR2Score の DDP 対応
- `torchmetrics` の `dist_reduce_fx="cat"` により、DDP 環境で state が Tensor に変換される問題に対応
- `compute()` メソッドで state が list か Tensor かを判定し、適切に処理

### 2. DDP Strategy 設定変更
- LoRA + EMA 使用時に未使用パラメータが検出される問題に対応
- `find_unused_parameters=True` を有効化

## 制約・前提条件
- PyTorch Lightning >= 2.5.5
- torchmetrics の分散集約機能を使用
- LoRA と EMA を併用するモデル構成
- DDP 環境での訓練（devices: 2）





