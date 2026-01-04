# 用語集

## 評価

| 用語 | 説明 |
|------|------|
| R²_w | 重み付きR²。全行に対して単一のスコアを計算 |
| SS_res | 残差平方和。予測誤差の重み付き合計 |
| SS_tot | 全平方和。データの分散の重み付き合計 |
| 行ごとの重み | target_name に基づく重み（Dry_Total_g が最大の0.5） |

## 実験管理

| 用語 | 説明 |
|------|------|
| expXXX | 実験の識別子。3桁ゼロ埋め（例: exp001, exp042） |
| fold | CV分割における各分割単位。本プロジェクトでは5-fold（fold0〜fold4） |
| OOF | Out-of-Fold。各foldで検証データに対して行った予測を結合したもの |
| CV | Cross-Validation。訓練データを分割して汎化性能を推定する手法 |
| LB | Leaderboard。Kaggleの公開スコアボード |

## データ

| 用語 | 説明 |
|------|------|
| sample_id | 行の一意識別子。`{image_id}__{target_name}` の形式 |
| image_path | 画像ファイルのパス |
| target_name | ターゲットの種類（Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g） |
| ターゲット | 予測対象の変数（5種類） |
| Long format | 1行 = 1画像 × 1ターゲット の形式。train.csv, test.csv, submission.csv で使用 |
| Species | 牧草種。バイオマス量順にアンダースコア区切り（例: `Ryegrass_Clover`） |
| NDVI | Normalized Difference Vegetation Index。植生の活性度を示す指標 |

## パイプライン

| 用語 | 説明 |
|------|------|
| checkpoint / ckpt | 学習済みモデルの保存ファイル |
| artifact | W&Bに保存される成果物（モデル、データセット等） |
| pipeline | データ処理から提出ファイル生成までの一連の処理フロー |

## ツール

| 用語 | 説明 |
|------|------|
| W&B | Weights & Biases。実験管理・可視化ツール |
| PyTorch Lightning | PyTorchの訓練ループを抽象化するフレームワーク |
| uv | Pythonパッケージマネージャ |
| OmegaConf | YAML設定ファイルを扱うライブラリ |

## リーク関連

| 用語 | 説明 |
|------|------|
| データリーク | 検証データの情報が訓練時に漏れること。CVスコアが過大評価される |
| グループ | CV分割において同一グループに属するサンプルは同じfoldに入る。リーク防止のため |
