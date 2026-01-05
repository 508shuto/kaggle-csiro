# Requirements

## 目的
単一実験のOOF（Out-of-Fold）予測を詳細分析するStreamlitアプリを作成する。
予測精度の可視化、残差分析、セグメント別性能比較、個別サンプル確認を通じて、モデル改善のインサイトを得る。

## 変更/追加する機能

### 新規ファイル
- `app/oof_analysis_app.py`: OOF分析用Streamlitアプリ

### 機能一覧
1. **Overview タブ**: 全体サマリー
   - KPIメトリクス（Mean R2, Std, Best/Worst Fold）
   - ターゲット別スコアテーブル
   - GT/OOFヒストグラム重ね表示
   - Foldスコア棒グラフ
   - 物理制約チェック（GDM=C+G, Total=C+D+G）

2. **Detail タブ**: 詳細分析（セクション切替）
   - Scatter: 予測vs実績散布図（5ターゲット）
   - Residual: 残差ヒストグラム、Q-Qプロット、残差vs予測値
   - Fold: Foldスコア比較、ターゲット×Foldヒートマップ
   - Segment: State/Species別性能、箱ひげ図
   - Correlation: 誤差相関ヒートマップ

3. **Gallery タブ**: 個別サンプル確認
   - サムネイル画像表示
   - 5ターゲットごとのGT/Pred/Error表示
   - ソート機能（誤差の大きい/小さい順）
   - フィルタ機能（Fold、State、Species）
   - ページネーション

## 制約・前提条件
- 既存の`app/streamlit_app.py`のパターンを踏襲
- Plotlyを使用した可視化
- `st.cache_data`によるキャッシュ最適化
- 5ターゲット: Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g
- 重み: [0.1, 0.1, 0.1, 0.2, 0.5]
- 物理制約: GDM = Clover + Green, Total = Clover + Dead + Green
