# Requirements

## 目的
exp028のDINOv3バックボーンにLoRAを適用し、Attention層を効率的にファインチューニングする

## 変更/追加する機能
- HF DINOv2 Large (`facebook/dinov2-large`) への移行
- peft LoRAによるAttention層の学習（最後の6レイヤーのみ: 18-23）
- パラメータグループ分離（LoRA: 1e-4, ヘッド: 1e-3）

## 制約・前提条件
- transformers >= 4.57.0 必須（DINOv3サポート）
- peft >= 0.14.0 必須
- 画像サイズ: 512x512
