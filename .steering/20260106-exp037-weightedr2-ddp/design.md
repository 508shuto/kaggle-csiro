# Design

## アプローチ

### 1. WeightedR2Score の修正

**問題**: DDP 環境で `dist_reduce_fx="cat"` により、`self.preds` と `self.targets` が list から Tensor に変換される。`torch.cat(self.preds, dim=0)` を実行すると、`self.preds` が既に Tensor のため `TypeError` が発生。

**解決策**: `compute()` メソッドで state の型を判定し、list の場合は `torch.cat()` で結合、Tensor の場合はそのまま使用。

```python
if isinstance(self.preds, list):
    preds = torch.cat(self.preds, dim=0)
else:
    preds = self.preds  # 既に Tensor
```

**空の state チェック**: list と Tensor の両方に対応するよう、`len()` と `numel()` を使い分け。

### 2. DDP Strategy 設定変更

**問題**: LoRA + EMA 使用時に、`training_step` で参照されていないパラメータが存在し、DDP がエラーを投げる。

**解決策**: 
- `config/exp037.yaml` で `strategy: "ddp_find_unused_parameters_true"` を設定
- `train.py` で strategy が `"ddp_find_unused_parameters_true"` の場合、`DDPStrategy(find_unused_parameters=True)` を明示的に設定

## 変更コンポーネント

### src/exp037/metrics.py
- `WeightedR2Score.compute()` メソッドを修正
- state の型判定と適切な処理を追加
- 空の state チェックを改善

### config/exp037.yaml
- `trainer.train.strategy` を `"ddp_find_unused_parameters_true"` に変更
- コメントで理由を記載

### src/exp037/train.py
- `DDPStrategy` をインポート
- strategy 設定の分岐処理を追加

### tests/test_metrics.py（新規作成）
- WeightedR2Score の回帰テストを追加
- list ベース更新のテスト
- DDP 擬似ケース（Tensor を直接セット）のテスト
- `compute_metrics` との整合性確認テスト

## 影響範囲

- **訓練**: DDP 環境で正常に動作するようになる
- **メトリクス**: 分散環境でも WeightedR2Score が正しく計算される
- **テスト**: 回帰テストにより、同様の問題の再発を防止

## 技術的詳細

### torchmetrics の分散集約
- `dist_reduce_fx="cat"` により、複数プロセスから集約された state が Tensor に変換される
- 単一プロセスでは list のまま、DDP 環境では Tensor に変換される

### DDP 未使用パラメータ検出
- LoRA の一部パラメータや EMA のパラメータが `training_step` で直接参照されない場合がある
- `find_unused_parameters=True` により、これらのパラメータも正しく処理される



