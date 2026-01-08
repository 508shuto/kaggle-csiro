# Requirements

## 目的
OOF Analysis App (`app/oof_analysis_app.py`) のUI改善と機能修正

## 変更/追加する機能

### 削除対象
1. Overview: Mean Weighted R2 表示
2. Target Scores: Weight列
3. Physical Constraint Check セクション
4. Detail: Fold, Correlation セクション

### 修正対象
5. Q-Q Plot: 理論分位数の計算がおかしい → 正しい計算に修正

### 追加対象
6. Segment: RMSE列を追加
7. Gallery: Species フィルタ追加
8. Gallery: Height, NDVI 表示追加
9. Gallery: Err+/- 色分け (正:赤、負:青)

## 制約・前提条件
- 既存のデータ構造 (`preprocessed_train.csv`) に `height_ave_cm`, `pre_gshh_ndvi`, `species` 列が存在
- Streamlit + Plotly ベースのアプリケーション
