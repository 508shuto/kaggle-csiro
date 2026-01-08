# Design

## アプローチ

1. species列から main_species を抽出 (split("_")[0])
2. グループ定義を date+state+main_species に変更
3. 層化キーを state+main_species に変更
4. seed=8635 で分割

## 変更コンポーネント

- src/exp042/create_dataset.py: CV分割ロジック
- config/exp042.yaml: seed値

## 影響範囲

- 学習データの分割のみ変更
- モデル・学習ロジックは変更なし
