# Tasklist

## 完了条件（Definition of Done）

- [ ] PinballLoss が正しく実装され、τ=0.5 で中央値回帰として機能する
- [ ] 3-fold CV が正常に実行できる
- [ ] CV スコアが記録される
- [ ] exp047.md に結果が記載される

## タスク

### フェーズ1: ドキュメント
- [x] Done: ステアリングドキュメント作成（requirements.md, design.md, tasklist.md）
- [x] Done: 実験ドキュメント作成（doc/experiment/exp047.md）
- [ ] TODO: /review-exp でレビュー

### フェーズ2: 実装
- [x] Done: src/exp047/ ディレクトリ作成
- [x] Done: exp044 からファイルコピー
- [x] Done: loss.py に PinballLoss 実装
- [x] Done: lightning_module.py で損失関数差し替え
- [x] Done: config/exp047.yaml 作成

### フェーズ3: 検証
- [ ] TODO: 3-fold CV 実行
- [ ] TODO: 結果記録（exp047.md に追記）
- [ ] TODO: /stage-exp でステージング
