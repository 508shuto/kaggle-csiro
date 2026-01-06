# Tasklist
## 完了条件（Definition of Done）
- `src/utils/download_hf_snapshot.py` を実行すると `./output/exp038/model/` が作成される
- `src/utils/upload_scripts.py exp038` で `.ckpt` + `config.yaml` + `model/` が同一Datasetに含まれてアップロードされる
- `.steering/.../requirements.md` / `design.md` / `tasklist.md` が存在し、**Kaggle Notebook手順**が明文化されている
- Kaggle Submission（Internet OFF）で外部通信なしに `exp038` 推論が走り `submission.csv` が生成される

## タスク
- [x] TODO: `src/utils/download_hf_snapshot.py` を追加（保存先は`./output/{exp_name}/model`）
- [x] TODO: `src/utils/upload_scripts.py` を拡張し、`./output/{exp_name}/model`を同梱
- [x] TODO: `.steering/20260106-exp038-offline-hf-model-snapshot-upload/` に3ファイルを作成し、Kaggle Notebookでの使い方まで記載

