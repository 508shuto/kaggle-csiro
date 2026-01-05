# Design

## アプローチ
- HF Transformersの`Dinov2Model`を使用（`facebook/dinov2-large`）
- peftの`get_peft_model()`でLoRA適用
- Attention層（query, key, value, dense）をターゲット
- `layers_to_transform=[18, 19, 20, 21, 22, 23]`で最後の6レイヤーのみ適用

## 変更コンポーネント
- models.py: HF DINOv3 + peft統合
- lightning_module.py: Optimizer設定（パラメータグループ分離）
- config/exp037.yaml: LoRA設定追加

## 影響範囲
- inference.py: peftモデル読み込み対応
- 推論時: merge_and_unload()でLoRA統合可能

## LoRA設定
- target_modules: ["query", "key", "value", "dense"]
- layers_to_transform: [18, 19, 20, 21, 22, 23]
- rank: 8
- alpha: 16
- dropout: 0.1
