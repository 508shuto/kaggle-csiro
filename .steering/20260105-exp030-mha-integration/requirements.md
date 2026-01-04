# Requirements

## 目的
補助タスク（NDVI, 高さ）の特徴をMulti Head Attentionで主タスクと統合し、精度向上を検証する

## 背景
- exp029 (Tile処理のみ): CV TBD（目標: 0.78程度）
- チームメンバー (Tile + MHA統合): CV 0.79775

チームメンバーの結果:
- Attention統合 > 中間特徴量共有 > Head分割

## 変更/追加する機能
1. Multi Head Attention層の追加
2. 補助タスク特徴と主タスク特徴の統合機構
3. 統合特徴から最終予測

## 制約・前提条件
- exp029のコードをベースにする
- Tile処理は維持
- 補助ターゲット（NDVI, 高さ）は変更しない
- Augmentationは現状維持（exp031で対応）

## 成功基準
- exp029より向上すればOK
