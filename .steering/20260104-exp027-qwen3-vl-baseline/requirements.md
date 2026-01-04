# Requirements

## 目的
Qwen3-VL (Vision-Language Model) を使った回帰ベースラインの構築。VLMの視覚エンコーダから抽出した特徴量を使い、牧草量（5ターゲット）を予測する。

## 変更/追加する機能
- Qwen3-VL-2B-Instruct をbackboneとした回帰モデル
- VLMの視覚特徴を抽出し、回帰ヘッドで5つのターゲットを予測
- exp025の構造をベースに、VLM用にDatasetとModelを改修

## 制約・前提条件
- transformers >= 4.57.0 が必要
- Qwen3-VL-2B-Instruct（最小サイズ）を使用
- GPUメモリ制約を考慮し、視覚エンコーダをfreeze
- 回帰ヘッドのみ学習（または将来的にLoRA fine-tuning）
