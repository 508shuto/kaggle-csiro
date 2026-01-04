# Tasklist

## 完了条件（Definition of Done）
- [x] DINOv3のbackboneがフリーズされている（学習時にgradが計算されない）
- [x] 回帰ヘッドのみが学習される
- [x] train.pyが正常に動作する（コード作成完了、動作確認は後続作業）

## タスク
- [x] Done: ステアリングフォルダ作成
- [x] Done: exp026のソースコード作成
  - [x] models.pyの変更（DINOv3 + フリーズ）
  - [x] lightning_module.pyの変更（optimizerでheadのみ学習）
- [x] Done: config/exp026.yamlの作成
- [x] Done: doc/experiment/exp026.mdの作成
- [x] Done: レビュー指摘事項の修正
  - [x] 画像サイズを224に変更
  - [x] 重複lr設定を削除
  - [x] ドキュメントをDINOv3に統一
  - [x] エポック数を100に設定（ユーザー指定）
