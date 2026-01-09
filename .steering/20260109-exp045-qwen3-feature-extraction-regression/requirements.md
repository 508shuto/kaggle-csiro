# Requirements

## 目的
Qwen3-VL-2Bの視覚エンコーダーを使用して画像から特徴量を抽出し、バイオマス回帰を行う。VLMの視覚エンコーダーがDINOv3より良い画像表現を学習している可能性を検証する。

## 変更/追加する機能
- Qwen3-VL-2B視覚エンコーダーの特徴量抽出
- 特徴量から5ターゲットへの回帰ヘッド
- 物理制約（GDM = Clover + Green, Total = Clover + Dead + Green）の維持

## 制約・前提条件
- モデル: `Qwen/Qwen3-VL-2B-Instruct`
- LLM部分は使用せず、視覚エンコーダーのみを使用
- 既存のパイプライン（train.py, evaluation.py, inference.py）構造を維持
- CV戦略: exp043/044と同様の3-fold CV（State stratification）
