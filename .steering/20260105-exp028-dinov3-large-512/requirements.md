# Requirements

## 目的
exp026をベースに、より大きなモデル（vit_large）と高解像度（512px）による性能向上を検証する

## 変更/追加する機能
1. モデルをvit_base_patch16_dinov3からvit_large_patch16_dinov3.lvd1689mに変更
2. 画像サイズを224から512に変更

## 制約・前提条件
- Backbone凍結（freeze_backbone: true）は維持
- 学習率1e-3は維持（Headのみ学習のため）
- EMA 100epoch + 5fold構成は維持
- バッチサイズ16は維持
