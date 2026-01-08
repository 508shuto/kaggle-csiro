# Design

## アプローチ

### 1. Overview タブ簡素化
- `render_overview_tab()` から Mean Weighted R2 と Physical Constraint Check を削除
- KPI Metrics を 3カラム (Std, Best Fold, Worst Fold) に変更
- Fold Scores を全幅表示に変更

### 2. Target Scores 簡素化
- Weight列を削除 (Target, R2, MAE, RMSE のみ)

### 3. Detail タブ簡素化
- `st.radio` の選択肢から Fold, Correlation を削除
- 残り: Scatter, Residual, Segment

### 4. Q-Q Plot 修正
Filliben's estimate を使用した正しい plotting positions:
```python
n = len(residuals)
positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
theoretical_quantiles = stats.norm.ppf(positions)
```
さらに残差を標準化して y=x 参照線が適切になるよう修正

### 5. Segment RMSE追加
`segment_data` に `"RMSE": metrics["rmse"]` を追加

### 6. Gallery フィルタ拡張
- 4カラムレイアウトに変更 (Sort, Fold, State, Species)
- Species フィルタのロジック追加

### 7. Gallery Height/NDVI 表示
Sample info 行に追加:
```
F0 | Tas | H:12.3cm | NDVI:0.65
```

### 8. Gallery Err 色分け
- pandas styling を使用
- 正のエラー (過大予測): `#ff6b6b` (赤)
- 負のエラー (過小予測): `#4dabf7` (青)
- ゼロ/欠損: `gray`

## 変更コンポーネント
- `app/oof_analysis_app.py`
  - `render_overview_tab()`
  - `render_detail_tab()`
  - `render_residual_section()` (Q-Q Plot)
  - `render_segment_section()`
  - `render_gallery_tab()`

## 影響範囲
- UI表示のみ。データ処理ロジックに変更なし
- 既存の関数 `render_fold_section()`, `render_correlation_section()`, `check_physical_constraints()`, `calculate_weighted_r2()` は残存 (削除せず)
