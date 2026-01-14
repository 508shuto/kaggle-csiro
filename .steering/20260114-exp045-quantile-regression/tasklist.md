# Tasklist

## 完了条件（Definition of Done）

- [ ] PinballLoss が正しく実装され、τ=0.5 で中央値回帰として機能する
- [ ] 3-fold CV が正常に実行できる
- [ ] CV スコアが記録される
- [ ] exp045.md に結果が記載される

## タスク

### フェーズ1: ドキュメント
- [x] Done: ステアリングドキュメント作成（requirements.md, design.md, tasklist.md）
- [ ] TODO: 実験ドキュメント作成（doc/experiment/exp045.md）
- [ ] TODO: /review-exp でレビュー

### フェーズ2: 実装
- [ ] TODO: src/exp045/ ディレクトリ作成
- [ ] TODO: exp044 からファイルコピー
- [ ] TODO: loss.py に PinballLoss 実装
- [ ] TODO: lightning_module.py で損失関数差し替え
- [ ] TODO: config/exp045.yaml 作成

### フェーズ3: 検証
- [ ] TODO: 3-fold CV 実行
- [ ] TODO: 結果記録（exp045.md に追記）
- [ ] TODO: /stage-exp でステージング
