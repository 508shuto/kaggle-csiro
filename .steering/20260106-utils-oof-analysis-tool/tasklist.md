# Tasklist

## 完了条件（Definition of Done）
- [x] `app/oof_analysis_app.py`が作成され、`uv run streamlit run app/oof_analysis_app.py`で起動できる
- [x] Overview/Detail/Galleryの3タブが機能する
- [x] 既存の実験（exp019など）のOOFデータを読み込み分析できる
- [x] 個別サンプルのGT/Pred/Errorが確認できる

## タスク

### 0. ステアリング作成
- [x] Done: `.steering/20260106-utils-oof-analysis-tool/` フォルダ作成
- [x] Done: requirements.md, design.md, tasklist.md

### 1. 基本構造とデータ読み込み
- [x] Done: アプリ骨格（3タブ構成）
- [x] Done: `list_available_experiments()`: OOF付き実験一覧取得
- [x] Done: `load_experiment_data()`: データ読み込み・マージ・残差計算

### 2. Overview タブ
- [x] Done: KPIメトリクス（Mean R2, Std, Best/Worst Fold）
- [x] Done: ターゲット別スコアテーブル
- [x] Done: GT/OOFヒストグラム重ね表示
- [x] Done: Foldスコア棒グラフ
- [x] Done: 物理制約チェック表示

### 3. Detail タブ
- [x] Done: セクション切替UI（Scatter/Residual/Fold/Segment/Correlation）
- [x] Done: 予測vs実績散布図（5ターゲット、y=x線、R2表示）
- [x] Done: 残差ヒストグラム、Q-Qプロット、残差vs予測値
- [x] Done: Foldスコア比較、ターゲット×Foldヒートマップ
- [x] Done: セグメント別性能テーブル・箱ひげ図
- [x] Done: 誤差相関ヒートマップ

### 4. Gallery タブ
- [x] Done: サムネイル画像読み込み・キャッシュ
- [x] Done: ソート機能（誤差の大きい/小さい順）
- [x] Done: フィルタ機能（Fold、State、Species）
- [x] Done: サンプルカード表示（画像、GT/Pred/Error）
- [x] Done: ページネーション

### 5. テスト
- [x] Done: 構文チェック・リントパス
