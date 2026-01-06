# Tasklist

## 完了条件（Definition of Done）
- [ ] 5-Fold CVが完了
- [ ] CVスコアがexp028（0.6746）を上回る
- [x] doc/experiment/exp038.md作成

## タスク
- [x] Done: src/exp038ディレクトリ作成
- [x] Done: models.py書き換え（AutoModel + 正規表現LoRA）
- [x] Done: config/exp038.yaml更新
- [x] Done: steeringドキュメント作成
- [x] Done: 実験ドキュメント作成
- [x] Done: 動作検証（919K trainable params確認）
- [x] Done: `.env.example`追加、SLURMスクリプトで`HF_TOKEN`等を環境変数として注入
- [ ] TODO: 本番学習

## 検証結果
- Total parameters: 304,048,901 (0.3B)
- LoRA parameters: 393,216
- Head parameters: 526,085
- Trainable parameters: 919,301 (0.30%)
