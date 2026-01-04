# Requirements

## 目的
exp025をベースにDINOv3のbackboneをフリーズして特徴量抽出器として使用し、回帰ヘッドのみを学習させる実験。

## 変更/追加する機能
1. DINOv3モデルをbackboneとして使用（vit_base_patch16_dinov3）
2. Backboneの重みをフリーズ（requires_grad=False）
3. 回帰ヘッド（head, aux_head）のみを学習

## 制約・前提条件
- exp025のコード構造をベースにする
- timmライブラリのDINOv3対応モデルを使用
- 推論時間の削減・メモリ効率の向上が期待される
