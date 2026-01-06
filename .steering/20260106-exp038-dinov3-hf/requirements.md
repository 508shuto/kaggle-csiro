# Requirements

## 目的
exp037のDINOv2をHF DINOv3に置き換え、最新のバックボーンでLoRA効果を検証

## 変更/追加する機能
- HF DINOv3 Large (`facebook/dinov3-vitl16-pretrain-lvd1689m`) への移行
- AutoModel/AutoConfigによる汎用的なモデル読み込み
- LoRA target modules変更: `dense` → `proj`

## 制約・前提条件
- transformers mainブランチが必要な場合あり（DINOv3サポート）
- 画像サイズ: 512x512
