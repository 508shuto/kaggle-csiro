# Design

## アプローチ
- HF Transformersの`AutoModel`を使用（DINOv2/v3両対応）
- peftの`get_peft_model()`でLoRA適用
- DINOv3のモジュール構造: `layer.{i}.attention.{q|k|v|o}_proj`
- 正規表現でtarget_modulesを指定（peftの`layers_to_transform`が動作しなかったため）

## 変更コンポーネント
- models.py: AutoModel/AutoConfig使用、正規表現LoRA
- config/exp038.yaml: モデル名、正規表現target_modules

## 影響範囲
- inference.py: AutoModel対応済み（変更不要）
- 推論時: merge_and_unload()でLoRA統合可能

## LoRA設定
- target_modules: `layer\.(1[89]|2[0-3])\.attention\.(q|k|v|o)_proj`（正規表現）
- rank: 8
- alpha: 16
- dropout: 0.1

## 注意点
- DINOv3はgated repo（HFログイン必要）
- peftの`layers_to_transform`はDINOv3で動作しないため正規表現を使用
