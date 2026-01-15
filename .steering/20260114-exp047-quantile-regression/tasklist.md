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
- [x] Done: コードレビュー対応（PR #20）

### フェーズ2: 実装
- [x] Done: src/exp047/ ディレクトリ作成
- [x] Done: exp044 からファイルコピー
- [x] Done: loss.py に PinballLoss 実装
- [x] Done: lightning_module.py で損失関数差し替え
- [x] Done: config/exp047.yaml 作成
- [x] Done: 未使用の設定パラメータを削除（shift_limit, scale_limit, gamma_transform, gaussian_noise, blur）
- [x] Done: exp044 の不要な変更を修正（precompute_weights.py の空白行削除）

### フェーズ3: 検証
- [ ] TODO: 3-fold CV 実行
- [ ] TODO: 結果記録（exp047.md に追記）
- [ ] TODO: /stage-exp でステージング
