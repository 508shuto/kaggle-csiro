# Design

## アプローチ
1. DINOv2 (vit_base_patch14_dinov2) をtimmから読み込み
2. backbone全体をフリーズ（`requires_grad_(False)`）
3. 回帰ヘッドは新規作成し、学習可能な状態にする
4. optimizerに渡すパラメータを回帰ヘッドのみに限定

## 変更コンポーネント
| ファイル | 変更内容 |
|---------|---------|
| src/exp026/models.py | DINOv2対応、フリーズ機能追加 |
| src/exp026/lightning_module.py | optimizer設定でheadのパラメータのみを渡す |
| config/exp026.yaml | モデル名の変更、画像サイズ調整（518x518推奨） |

## 影響範囲
- dataset.py, evaluation.py, inference.py, train.pyはexp025から変更なし（コピー）
- 画像サイズがDINOv2の推奨サイズ（518x518）に変更される可能性
