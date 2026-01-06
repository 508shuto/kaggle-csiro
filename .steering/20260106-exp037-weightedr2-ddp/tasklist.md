# Tasklist

## 完了条件（Definition of Done）
- [x] WeightedR2Score の DDP 環境でのエラーが解消される
- [x] DDP 未使用パラメータ検出エラーが解消される
- [x] 回帰テストが追加され、同様の問題の再発を防止できる
- [x] ドキュメントが作成され、修正内容が記録される

## タスク

### WeightedR2Score 修正
- [x] Done: `metrics.py` の `compute()` メソッドを修正（state の型判定を追加）
- [x] Done: 空の state チェックを改善（list と Tensor の両方に対応）
- [x] Done: 回帰テストを作成（`tests/test_metrics.py`）

### DDP Strategy 設定変更
- [x] Done: `config/exp037.yaml` で strategy を `"ddp_find_unused_parameters_true"` に変更
- [x] Done: `train.py` で `DDPStrategy` のインポートと分岐処理を追加

### ドキュメント作成
- [x] Done: `.steering/20260106-exp037-weightedr2-ddp/` ディレクトリ作成
- [x] Done: `requirements.md` 作成
- [x] Done: `design.md` 作成
- [x] Done: `tasklist.md` 作成





