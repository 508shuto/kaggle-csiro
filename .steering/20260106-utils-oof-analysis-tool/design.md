# Design

## アプローチ

### データフロー
```
output/expXXX/oofs.csv (予測値)
output/expXXX/preprocessed_train.csv (sample_id, fold, 実測値)
output/expXXX/results.json (fold_scores)
input/train.csv (メタデータ: State, Species, Sampling_Date)
    ↓ sample_idでマージ
df_merged: sample_id, fold, state, species, {target}_actual, {target}_pred, {target}_residual
```

### アプリ構成（3タブ）
1. **Overview**: 全体サマリー、KPI、分布、物理制約チェック
2. **Detail**: 詳細分析（セクション切替でScatter/Residual/Fold/Segment/Correlation）
3. **Gallery**: 個別サンプル確認（画像＋GT/Pred/Error）

### UI設計
- サイドバー: 実験選択、ターゲット選択、セグメント選択
- メインエリア: タブ切替で各機能を表示

## 変更コンポーネント

### 新規作成
- `app/oof_analysis_app.py`

### 参照（変更なし）
- `app/streamlit_app.py`: TARGET_MAPPING、既存パターン参照
- `app/eda_app.py`: Plotlyスタイル参照

## 影響範囲
- 新規ファイル追加のみ、既存コードへの影響なし

## 設計考慮事項

### データ結合
- sample_idをキーとして明示的に結合
- OOFの行数とpreprocessed_trainの行数が一致することを検証

### 欠損値処理
- NaN/infは計算から除外
- UIで欠損数を表示

### キャッシュ戦略
- `@st.cache_data`でデータ読み込み・計算結果をキャッシュ
- 画像はサムネイルサイズ（150x150）で縮小キャッシュ

### 評価指標
- 重み付きR²で統一: [0.1, 0.1, 0.1, 0.2, 0.5]
- 個別ターゲットのR²、MAE、RMSEも表示
