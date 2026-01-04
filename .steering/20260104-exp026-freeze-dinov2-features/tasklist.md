# Tasklist

## 完了条件（Definition of Done）
- [ ] DINOv2のbackboneがフリーズされている（学習時にgradが計算されない）
- [ ] 回帰ヘッドのみが学習される
- [ ] train.pyが正常に動作する

## タスク
- [x] Done: ステアリングフォルダ作成
- [ ] In Progress: exp026のソースコード作成
  - [ ] models.pyの変更（DINOv2 + フリーズ）
  - [ ] lightning_module.pyの変更（optimizerでheadのみ学習）
- [ ] TODO: config/exp026.yamlの作成
- [ ] TODO: doc/experiment/exp026.mdの作成
- [ ] TODO: 動作確認
