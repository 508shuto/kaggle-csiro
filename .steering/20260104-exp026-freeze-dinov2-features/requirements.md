# Requirements

## 目的
exp025をベースにDINOv2のbackboneをフリーズして特徴量抽出器として使用し、回帰ヘッドのみを学習させる実験。

## 変更/追加する機能
1. DINOv2モデルをbackboneとして使用
2. Backboneの重みをフリーズ（requires_grad=False）
3. 回帰ヘッド（head, aux_head）のみを学習

## 制約・前提条件
- exp025のコード構造をベースにする
- timmライブラリのDINOv2対応モデルを使用
- 推論時間の削減・メモリ効率の向上が期待される
