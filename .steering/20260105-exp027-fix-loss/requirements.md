# Requirements

## 目的
exp027のlossが下がらない問題を修正する

## 変更/追加する機能
- VLM全体のfreeze（現在はvision_encoderのみ）
- forward内clamp削除（勾配死亡防止）
- warmup_epochs適用
- 学習率調整

## 制約・前提条件
- Qwen3-VL-2Bのアーキテクチャを維持
- regression headのみ学習可能
