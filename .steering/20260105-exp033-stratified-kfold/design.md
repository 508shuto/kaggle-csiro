# Design

## アプローチ
StratifiedGroupKFold → StratifiedKFold に変更し、State + target_binで層化。

## 変更コンポーネント
- create_dataset.py: fold分割ロジック

## 影響範囲
- preprocessed_train.csv のfold列が変わる
- モデル学習・評価には影響なし（入力形式は同じ）
