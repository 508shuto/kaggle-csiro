# Requirements

## 目的
Qwen3-VL-Embedding-2Bを使用して画像から高品質な特徴量（2048次元）を抽出し、バイオマス回帰を行う。VLMベースのEmbeddingがDINOv3より良い画像表現を持つかを検証する。

## 変更/追加する機能
- Qwen3-VL-Embedding-2Bによる特徴量抽出（2048次元）
- 特徴量から3ターゲットへの回帰ヘッド
- 物理制約（GDM = Clover + Green, Total = Clover + Dead + Green）による5ターゲット導出

## 制約・前提条件
- モデル: `Qwen/Qwen3-VL-Embedding-2B`
- Embeddingモデルは凍結（Headのみ学習）
- 既存のパイプライン（train.py, evaluation.py, inference.py）構造を維持
- CV戦略: exp043/044と同様の3-fold CV（State stratification）
- Loss: SmoothL1Loss（exp044のInverseWeightedは使用しない、シンプルに開始）
