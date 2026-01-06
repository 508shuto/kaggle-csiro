# Requirements
## 目的
Kaggleコンペの Submission（Internet OFF）で、HF gated model
`facebook/dinov3-vitl16-pretrain-lvd1689m` を exp038 の推論で利用できるようにする。

## 変更/追加する機能
- ローカル/ColabでHF gated repoのsnapshotを `./output/{exp_name}/model` に保存できるスクリプトを追加する
- `src/utils/upload_scripts.py` で、`.ckpt` + `config.yaml` に加えて `./output/{exp_name}/model` も同一Kaggle Datasetに同梱してアップロードできるようにする
- Kaggle Notebook（Internet OFF）での使い方をドキュメント化する

## 制約・前提条件
- HF側のaccess request承認/利用規約同意は事前に完了していること（DLはローカル/Colabで行う）
- HF_TOKENはローカル/Colabのみで使用し、Kaggle（Internet OFF）へは持ち込まない
- Kaggle SubmissionはInternet OFF（HF Hubからの取得は禁止）
- モデルファイルはprivate Kaggle Datasetに同梱する
- 既存のexp038の学習済みckptとの互換性を維持する（推論I/Fは壊さない）

