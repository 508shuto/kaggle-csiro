# Requirements

## 目的
LoRAの過学習問題を解消し、CV改善とLB改善を両立させる。
exp040（Frozen backbone）をベースに、最後の1-2ブロックのみunfreezeすることで、
適度な表現力を得つつ汎化性能を維持する。

## 変更/追加する機能
1. Backboneの最後の2ブロック（blocks.22-23）のみ学習可能にする
2. Backbone用の低学習率（1e-5）とHead用の学習率（1e-3）を分離
3. 学習率スケジューラの調整

## 制約・前提条件
- exp040の3-fold State層化CVを継承
- 画像サイズ512x512を維持
- timmモデル（vit_large_patch16_dinov3.lvd1689m）を使用

## 背景
- exp043（LoRA）: CV 0.7021だがLB悪化（過学習）
- exp044（LoRA + 逆数重み）: CV 0.6871, LB 0.62（過学習を抑制したがLB改善なし）
- Partial Unfreezeはより制約的なfine-tuningで、過学習を抑制しつつ表現力を得る
