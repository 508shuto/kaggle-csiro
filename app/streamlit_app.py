import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

# デフォルト設定
DEFAULT_TRAIN_CSV = "input/train.csv"
DEFAULT_TRAIN_DIR = Path("input/train")
NOTES_FILE = Path("output/notes.json")

# ターゲット名のマッピング
TARGET_MAPPING = {
    "Dry_Clover_g": "clover_target",
    "Dry_Dead_g": "dead_target",
    "Dry_Green_g": "green_target",
    "GDM_g": "gdm_target",
    "Dry_Total_g": "total_target",
}


@st.cache_data
def load_and_preprocess_df(csv_path: str, train_dir: Path) -> pd.DataFrame:
    """train.csvを読み込み、preprocessed形式に変換"""
    df = pd.read_csv(csv_path)

    # sample_idから画像IDを抽出（例: ID1011485656__Dry_Clover_g → ID1011485656）
    df["image_id"] = df["sample_id"].str.split("__").str[0]

    # ターゲット名を標準化（型チェック回避のためreplace使用）
    df["target_name_normalized"] = df["target_name"].replace(TARGET_MAPPING)

    # 画像ごとにターゲットをpivot
    pivot_df = df.pivot_table(
        index="image_id",
        columns="target_name_normalized",
        values="target",
        aggfunc="first",
    ).reset_index()

    # メタデータをマージ（画像ごとに同じ値なのでfirstで取得）
    meta_cols = ["Sampling_Date", "State", "Species", "Pre_GSHH_NDVI", "Height_Ave_cm"]
    if "image_path" in df.columns:
        meta_cols.append("image_path")

    meta_df = df[["image_id"] + meta_cols].drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    # マージ
    result_df = pivot_df.merge(meta_df, on="image_id", how="left")

    # カラム名を標準化
    result_df = result_df.rename(
        columns={
            "image_id": "sample_id",
            "Sampling_Date": "sampling_date",
            "State": "state",
            "Species": "species",
            "Pre_GSHH_NDVI": "pre_gshh_ndvi",
            "Height_Ave_cm": "height_ave_cm",
        }
    )

    # 画像パスを解決（input/train/{sample_id}.jpg）
    def resolve_image_path(sid: str) -> str:
        img_path = train_dir / f"{sid}.jpg"
        return str(img_path) if img_path.exists() else ""

    result_df["image_path"] = result_df["sample_id"].map(resolve_image_path)

    # 日付をパース
    if "sampling_date" in result_df.columns:
        result_df["sampling_date"] = pd.to_datetime(result_df["sampling_date"], errors="coerce")
        # 文字列形式のカテゴリ列を追加（YYYY/MM/DD形式）
        result_df["sampling_date_str"] = result_df["sampling_date"].dt.strftime("%Y/%m/%d")
        # 欠損値の場合は空文字列に変換
        result_df["sampling_date_str"] = result_df["sampling_date_str"].fillna("")

    # 型変換
    for col in [
        "clover_target",
        "dead_target",
        "green_target",
        "gdm_target",
        "total_target",
    ]:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    return result_df


@st.cache_data
def load_oof_predictions(oof_dir: Path) -> pd.DataFrame | None:
    """OOF予測値を読み込み、sample_id付与"""
    oof_csv = oof_dir / "oofs.csv"
    preprocessed_csv = oof_dir / "preprocessed_train.csv"

    if not oof_csv.exists():
        return None
    if not preprocessed_csv.exists():
        return None

    try:
        # preprocessed_train.csvからsample_idを取得
        preprocessed_df = pd.read_csv(preprocessed_csv)
        if "sample_id" not in preprocessed_df.columns:
            return None

        # oofs.csvを読み込み
        oof_df = pd.read_csv(oof_csv)

        # sample_idを付与
        oof_df["sample_id"] = preprocessed_df["sample_id"].values[: len(oof_df)]

        # カラム名を標準化（pred_プレフィックスを付与）
        pred_columns = {col: f"pred_{col}" for col in oof_df.columns if col != "sample_id"}
        oof_df = oof_df.rename(columns=pred_columns)

        return oof_df
    except Exception as e:
        st.error(f"OOF読み込みエラー: {e}")
        return None


def load_notes() -> dict[str, dict[str, str]]:
    """メモファイルから全メモを読み込み"""
    if not NOTES_FILE.exists():
        return {}
    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_note(sample_id: str, content: str) -> None:
    """メモを保存"""
    notes = load_notes()
    notes[sample_id] = {
        "content": content,
        "updated_at": datetime.now().isoformat(),
    }
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


def delete_note(sample_id: str) -> None:
    """メモを削除"""
    notes = load_notes()
    if sample_id in notes:
        del notes[sample_id]
        NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)


def calculate_derived_values(row: pd.Series) -> dict[str, float]:
    """派生値を計算"""

    def as_float(val: Any) -> float:
        try:
            f = float(val)
            if pd.isna(f):
                return 0.0
            return f
        except Exception:
            return 0.0

    clover = as_float(row.get("clover_target", 0.0))
    dead = as_float(row.get("dead_target", 0.0))
    green = as_float(row.get("green_target", 0.0))
    gdm = as_float(row.get("gdm_target", 0.0))
    total = as_float(row.get("total_target", 0.0))

    gdm_calc = clover + green
    gdm_diff = gdm - gdm_calc
    total_calc = clover + dead + green
    total_diff = total - total_calc

    return {
        "gdm_calc": float(gdm_calc),
        "gdm_diff": float(gdm_diff),
        "total_calc": float(total_calc),
        "total_diff": float(total_diff),
    }


def calculate_hsv_histogram(
    img: Image.Image,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PIL ImageをHSVに変換してヒストグラムを計算"""
    # RGB→HSV変換
    hsv_img = img.convert("HSV")
    hsv_array = np.array(hsv_img)

    # 各チャンネルを取得
    h_channel = hsv_array[:, :, 0].flatten()
    s_channel = hsv_array[:, :, 1].flatten()
    v_channel = hsv_array[:, :, 2].flatten()

    # ヒストグラム計算（256ビン）
    h_hist, _ = np.histogram(h_channel, bins=256, range=(0, 256))
    s_hist, _ = np.histogram(s_channel, bins=256, range=(0, 256))
    v_hist, _ = np.histogram(v_channel, bins=256, range=(0, 256))

    return h_hist, s_hist, v_hist


def plot_hsv_histogram(h_hist: np.ndarray, s_hist: np.ndarray, v_hist: np.ndarray) -> go.Figure:
    """HSVヒストグラムをPlotlyで可視化"""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("H (色相)", "S (彩度)", "V (明度)"),
        horizontal_spacing=0.1,
    )

    bins = np.arange(256)

    # Hチャンネル
    fig.add_trace(
        go.Bar(x=bins, y=h_hist, name="H", marker_color="red"),
        row=1,
        col=1,
    )

    # Sチャンネル
    fig.add_trace(
        go.Bar(x=bins, y=s_hist, name="S", marker_color="green"),
        row=1,
        col=2,
    )

    # Vチャンネル
    fig.add_trace(
        go.Bar(x=bins, y=v_hist, name="V", marker_color="blue"),
        row=1,
        col=3,
    )

    # 軸ラベル設定
    fig.update_xaxes(title_text="値", row=1, col=1)
    fig.update_xaxes(title_text="値", row=1, col=2)
    fig.update_xaxes(title_text="値", row=1, col=3)

    fig.update_yaxes(title_text="頻度", row=1, col=1)
    fig.update_yaxes(title_text="頻度", row=1, col=2)
    fig.update_yaxes(title_text="頻度", row=1, col=3)

    fig.update_layout(
        height=300,
        showlegend=False,
        title_text="HSV色空間ヒストグラム",
        title_x=0.5,
    )

    return fig


def calculate_ndi(rgb_array: np.ndarray) -> np.ndarray:
    """NDIを計算: (G-R)/(G+R)"""
    r = rgb_array[:, :, 0].astype(np.float32)
    g = rgb_array[:, :, 1].astype(np.float32)
    denominator = r + g + 1e-6
    ndi = (g - r) / denominator
    return ndi


def calculate_cive(rgb_array: np.ndarray) -> np.ndarray:
    """CIVEを計算: 0.441*R - 0.881*G + 0.385*B + 18.78745"""
    r = rgb_array[:, :, 0].astype(np.float32)
    g = rgb_array[:, :, 1].astype(np.float32)
    b = rgb_array[:, :, 2].astype(np.float32)
    cive = 0.441 * r - 0.881 * g + 0.385 * b + 18.78745
    return cive


def calculate_exg(rgb_array: np.ndarray) -> np.ndarray:
    """ExG (Excess Green)を計算: 2*G - R - B"""
    r = rgb_array[:, :, 0].astype(np.float32)
    g = rgb_array[:, :, 1].astype(np.float32)
    b = rgb_array[:, :, 2].astype(np.float32)
    exg = 2 * g - r - b
    return exg


def calculate_yellow_hue(
    rgb_array: np.ndarray,
    center_deg: float = 60.0,
    sigma_deg: float = 20.0,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> np.ndarray:
    """黄の近さをHSVのHue中心で重み付けし、SとVで増幅するスコア。

    score = exp(-((ΔH/σ)^2)) * S^α * V^β
    """
    hsv_img = Image.fromarray(rgb_array).convert("HSV")
    hsv = np.array(hsv_img).astype(np.float32)
    h = hsv[:, :, 0] * (360.0 / 255.0)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0

    delta = np.abs(h - center_deg)
    delta = np.minimum(delta, 360.0 - delta)
    weight = np.exp(-((delta / sigma_deg) ** 2))
    score = weight * (s**alpha) * (v**beta)
    return score.astype(np.float32)


def calculate_ndy(rgb_array: np.ndarray) -> np.ndarray:
    """NDY (Yellow emphasis): (R+G - B)/(R+G + B)"""
    r = rgb_array[:, :, 0].astype(np.float32)
    g = rgb_array[:, :, 1].astype(np.float32)
    b = rgb_array[:, :, 2].astype(np.float32)
    numerator = r + g - b
    denominator = r + g + b + 1e-6
    ndy = numerator / denominator
    return ndy


def calculate_whiteness(rgb_array: np.ndarray) -> np.ndarray:
    """Whiteness (brightness proxy): (R+G+B)/3"""
    r = rgb_array[:, :, 0].astype(np.float32)
    g = rgb_array[:, :, 1].astype(np.float32)
    b = rgb_array[:, :, 2].astype(np.float32)
    whiteness = (r + g + b) / 3.0
    return whiteness


def calculate_brownness(rgb_array: np.ndarray) -> np.ndarray:
    """Brownness: 茶色の色相範囲を検出（HSVのHueで20-40度付近を強調）"""
    hsv_img = Image.fromarray(rgb_array).convert("HSV")
    hsv = np.array(hsv_img).astype(np.float32)
    h = hsv[:, :, 0] * (360.0 / 255.0)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0

    # 茶色の色相範囲（20-40度）を検出
    brown_center = 30.0
    brown_sigma = 10.0
    delta = np.abs(h - brown_center)
    delta = np.minimum(delta, 360.0 - delta)
    brown_weight = np.exp(-((delta / brown_sigma) ** 2))

    # SとVで重み付け
    brownness = brown_weight * s * v
    return brownness.astype(np.float32)


def calculate_dryness(rgb_array: np.ndarray) -> np.ndarray:
    """Dryness = 0.6 * Yellow(Hue) + 0.4 * Brownness(正規化)"""
    yellow = calculate_yellow_hue(rgb_array)
    brown = calculate_brownness(rgb_array)
    brown = np.clip(brown, 0.0, None)
    # 0-1正規化（定数画像を回避）
    bmin = float(brown.min())
    bmax = float(brown.max())
    if bmax - bmin > 1e-6:
        brown_norm = (brown - bmin) / (bmax - bmin)
    else:
        brown_norm = np.zeros_like(brown, dtype=np.float32)
    dryness = 0.6 * yellow + 0.4 * brown_norm
    return dryness.astype(np.float32)


def visualize_index(index_array: np.ndarray, colormap: str = "viridis") -> Image.Image:
    """植生指数をカラーマップで可視化"""
    # 値を0-255にスケール
    index_min = index_array.min()
    index_max = index_array.max()
    if index_max - index_min > 1e-6:
        normalized = (index_array - index_min) / (index_max - index_min)
    else:
        normalized = np.zeros_like(index_array)

    # 0-255にスケール
    scaled = (normalized * 255).astype(np.uint8)

    # matplotlibのカラーマップを使用
    cmap = matplotlib.colormaps[colormap]
    colored = cmap(scaled / 255.0)

    # RGBAからRGBに変換（alphaチャンネルを削除）
    rgb_colored = (colored[:, :, :3] * 255).astype(np.uint8)

    return Image.fromarray(rgb_colored)


def main():
    st.set_page_config(
        page_title="CSIRO Train Gallery",
        page_icon="🖼️",
        layout="wide",
    )

    st.title("🖼️ CSIRO Train Image Gallery")

    # サイドバー設定
    st.sidebar.header("設定")

    csv_path = st.sidebar.text_input("CSVパス", value=DEFAULT_TRAIN_CSV, help="train.csvのパス")
    train_dir_str = st.sidebar.text_input(
        "画像ディレクトリ", value=str(DEFAULT_TRAIN_DIR), help="画像ディレクトリのパス"
    )
    train_dir = Path(train_dir_str)

    # データ読み込み
    try:
        df = load_and_preprocess_df(csv_path, train_dir)
        st.sidebar.success(f"✅ {len(df)}件のデータを読み込みました")
    except Exception as e:
        st.sidebar.error(f"❌ データ読み込みエラー: {e}")
        st.stop()

    # フィルタ設定
    st.sidebar.header("フィルタ")

    # speciesフィルタ
    if "species" in df.columns and df["species"].notna().any():
        species_options = ["すべて"] + sorted(df["species"].dropna().unique().tolist())
        selected_species = st.sidebar.multiselect("Species", species_options, default=["すべて"])
        if "すべて" not in selected_species:
            df = df[df["species"].isin(selected_species)]

    # stateフィルタ
    if "state" in df.columns and df["state"].notna().any():
        state_options = ["すべて"] + sorted(df["state"].dropna().unique().tolist())
        selected_state = st.sidebar.multiselect("State", state_options, default=["すべて"])
        if "すべて" not in selected_state:
            df = df[df["state"].isin(selected_state)]

    # Sampling Dateフィルタ（カテゴリ選択）
    if "sampling_date_str" in df.columns and df["sampling_date_str"].notna().any():
        date_options = ["すべて"] + sorted(df["sampling_date_str"].dropna().unique().tolist())
        selected_dates = st.sidebar.multiselect("Sampling Date", date_options, default=["すべて"])
        if "すべて" not in selected_dates:
            df = df[df["sampling_date_str"].isin(selected_dates)]

    # sample_id検索
    search_query = st.sidebar.text_input("Sample ID検索", value="")
    if search_query:
        df = df[df["sample_id"].str.contains(search_query, case=False, na=False)]

    # メモ付き画像フィルタ
    notes = load_notes()
    show_only_with_notes = st.sidebar.checkbox("メモ付きのみ表示", value=False)
    if show_only_with_notes:
        notes_sample_ids = set(notes.keys())
        df = df[df["sample_id"].isin(notes_sample_ids)]

    # Sampling Dateでソート（昇順）
    if "sampling_date" in df.columns:
        df = df.sort_values("sampling_date").reset_index(drop=True)

    # OOF選択
    st.sidebar.header("OOF予測")
    output_dir = Path("output")
    exp_dirs = []
    if output_dir.exists():
        exp_dirs = sorted(
            [
                d.name
                for d in output_dir.iterdir()
                if d.is_dir()
                and d.name.startswith("exp")
                and (d / "oofs.csv").exists()
                and (d / "preprocessed_train.csv").exists()
            ]
        )

    if exp_dirs:
        selected_exp = st.sidebar.selectbox("実験を選択", ["なし"] + exp_dirs, index=0)
        oof_df = None
        if selected_exp != "なし":
            oof_dir = output_dir / selected_exp
            oof_df = load_oof_predictions(oof_dir)
            if oof_df is not None:
                st.sidebar.success(f"✅ {len(oof_df)}件のOOF予測を読み込みました")
            else:
                st.sidebar.warning("⚠️ OOFファイルの読み込みに失敗しました")
    else:
        selected_exp = "なし"
        oof_df = None
        st.sidebar.info("OOFファイルが見つかりません")

    # OOF予測をマージ
    if oof_df is not None:
        df = df.merge(oof_df, on="sample_id", how="left")

    # 誤差ソート
    sort_key = "なし"
    if oof_df is not None:
        st.sidebar.subheader("誤差でソート")
        sort_key = st.sidebar.selectbox(
            "対象",
            [
                "なし",
                "総合(MAE)",
                "clover",
                "dead",
                "green",
                "gdm",
                "total",
            ],
            index=0,
        )

        if sort_key != "なし":
            target_columns = [
                "clover_target",
                "dead_target",
                "green_target",
                "gdm_target",
                "total_target",
            ]

            if sort_key == "総合(MAE)":
                abs_cols: list[str] = []
                for tcol in target_columns:
                    pcol = f"pred_{tcol}"
                    if tcol in df.columns and pcol in df.columns:
                        colname = f"_abs_{tcol}"
                        df[colname] = (
                            pd.to_numeric(df[pcol], errors="coerce") - pd.to_numeric(df[tcol], errors="coerce")
                        ).abs()
                        abs_cols.append(colname)
                if abs_cols:
                    df["_abs_err_overall"] = df[abs_cols].mean(axis=1, skipna=True)
                    df = df.sort_values("_abs_err_overall", ascending=False, na_position="last").reset_index(drop=True)
            else:
                tcol = f"{sort_key}_target"
                pcol = f"pred_{tcol}"
                if tcol in df.columns and pcol in df.columns:
                    df["_abs_err"] = (
                        pd.to_numeric(df[pcol], errors="coerce") - pd.to_numeric(df[tcol], errors="coerce")
                    ).abs()
                    df = df.sort_values("_abs_err", ascending=False, na_position="last").reset_index(drop=True)

    # ギャラリー設定（右ペインに移動）
    # ページネーションや表示設定は右側で操作

    # メインコンテンツ
    # 選択画像の初期化（未選択の場合は先頭行）
    selected_id = st.session_state.get("selected_sample_id")
    if not selected_id or selected_id not in df["sample_id"].values:
        if len(df) > 0:
            selected_id = df.iloc[0]["sample_id"]
            st.session_state["selected_sample_id"] = selected_id

    # レイアウト: メイン / 右ギャラリー
    col_main, col_right = st.columns([2, 1])

    with col_main:
        # メインビュー（選択画像の大表示）
        st.subheader("📷 メインビュー")
        if selected_id and selected_id in df["sample_id"].values:
            # ナビゲーション（前へ/次へ）
            current_indices = df.index[df["sample_id"] == selected_id].tolist()
            current_idx = current_indices[0] if current_indices else 0
            total_count = len(df)
            nav_prev, nav_pos, nav_next = st.columns([1, 2, 1])
            with nav_prev:
                if st.button(
                    "◀ 前へ",
                    disabled=current_idx <= 0,
                    width="stretch",
                    key="main_prev",
                ):
                    st.session_state["selected_sample_id"] = df.iloc[current_idx - 1]["sample_id"]
                    st.rerun()
            with nav_pos:
                st.markdown(f"**{current_idx + 1} / {total_count}**")
            with nav_next:
                if st.button(
                    "次へ ▶",
                    disabled=current_idx >= total_count - 1,
                    width="stretch",
                    key="main_next",
                ):
                    st.session_state["selected_sample_id"] = df.iloc[current_idx + 1]["sample_id"]
                    st.rerun()

            main_row = df.iloc[current_idx]
            if main_row.get("image_path") and Path(main_row["image_path"]).exists():
                try:
                    main_img = Image.open(main_row["image_path"])
                    rgb_array = np.array(main_img.convert("RGB"))

                    # 植生指数選択ボタン
                    st.markdown("### 🌿 植生指数")
                    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5, col_btn6 = st.columns(6)
                    col_btn7, col_btn8, col_btn9, col_btn10 = st.columns(4)

                    selected_index = st.session_state.get("selected_vegetation_index", "元画像")

                    with col_btn1:
                        if st.button("元画像", key="btn_original", width="stretch"):
                            st.session_state["selected_vegetation_index"] = "元画像"
                            st.rerun()
                    with col_btn2:
                        if st.button("NDY", key="btn_ndy", width="stretch"):
                            st.session_state["selected_vegetation_index"] = "NDY"
                            st.rerun()
                    with col_btn3:
                        if st.button("NDI", key="btn_ndi", width="stretch"):
                            st.session_state["selected_vegetation_index"] = "NDI"
                            st.rerun()
                    with col_btn4:
                        if st.button("CIVE", key="btn_cive", width="stretch"):
                            st.session_state["selected_vegetation_index"] = "CIVE"
                            st.rerun()
                    with col_btn5:
                        if st.button("ExG", key="btn_exg", width="stretch"):
                            st.session_state["selected_vegetation_index"] = "ExG"
                            st.rerun()
                    with col_btn6:
                        if st.button("White", key="btn_white", width="stretch"):
                            st.session_state["selected_vegetation_index"] = "White"
                            st.rerun()
                    with col_btn7:
                        if st.button("Yellow(Hue)", key="btn_yellow", width="stretch"):
                            st.session_state["selected_vegetation_index"] = "Yellow(Hue)"
                            st.rerun()
                    with col_btn8:
                        if st.button("ExY", key="btn_exy", width="stretch"):
                            st.session_state["selected_vegetation_index"] = "ExY"
                            st.rerun()
                    with col_btn9:
                        if st.button("Brown", key="btn_brown", width="stretch"):
                            st.session_state["selected_vegetation_index"] = "Brown"
                            st.rerun()
                    with col_btn10:
                        if st.button("Dryness", key="btn_dryness", width="stretch"):
                            st.session_state["selected_vegetation_index"] = "Dryness"
                            st.rerun()

                    # 互換: 旧値NDVIが残っていたらNDYに置換
                    if selected_index == "NDVI":
                        selected_index = "NDY"

                    # 選択された指数または元画像を表示
                    if selected_index == "元画像":
                        st.image(main_img, width="stretch")
                    elif selected_index == "NDY":
                        ndy_array = calculate_ndy(rgb_array)
                        ndy_img = visualize_index(ndy_array, colormap="YlOrBr")
                        st.image(ndy_img, width="stretch", caption="NDY (Yellow)")
                    elif selected_index == "NDI":
                        ndi_array = calculate_ndi(rgb_array)
                        ndi_img = visualize_index(ndi_array, colormap="RdYlGn")
                        st.image(ndi_img, width="stretch", caption="NDI")
                    elif selected_index == "CIVE":
                        cive_array = calculate_cive(rgb_array)
                        cive_img = visualize_index(cive_array, colormap="viridis")
                        st.image(cive_img, width="stretch", caption="CIVE")
                    elif selected_index == "ExG":
                        exg_array = calculate_exg(rgb_array)
                        exg_img = visualize_index(exg_array, colormap="Greens")
                        st.image(exg_img, width="stretch", caption="ExG")
                    elif selected_index == "Yellow(Hue)":
                        yellow_score = calculate_yellow_hue(rgb_array)
                        yellow_img = visualize_index(yellow_score, colormap="YlOrBr")
                        st.image(yellow_img, width="stretch", caption="Yellow(Hue)")
                    elif selected_index == "ExY":
                        exy_array = (
                            rgb_array[:, :, 0].astype(np.float32)
                            + rgb_array[:, :, 1].astype(np.float32)
                            - 2.0 * rgb_array[:, :, 2].astype(np.float32)
                        )
                        exy_img = visualize_index(exy_array, colormap="YlOrBr")
                        st.image(exy_img, width="stretch", caption="ExY")
                    elif selected_index == "Brown":
                        brown_array = calculate_brownness(rgb_array)
                        brown_img = visualize_index(brown_array, colormap="copper")
                        st.image(brown_img, width="stretch", caption="Brownness")
                    elif selected_index == "Dryness":
                        dryness_array = calculate_dryness(rgb_array)
                        dryness_img = visualize_index(dryness_array, colormap="cividis")
                        st.image(dryness_img, width="stretch", caption="Dryness")
                    elif selected_index == "White":
                        white_array = calculate_whiteness(rgb_array)
                        white_img = visualize_index(white_array, colormap="gray")
                        st.image(white_img, width="stretch", caption="Whiteness")

                    # HSVヒストグラム表示（元画像の場合のみ）
                    if selected_index == "元画像":
                        h_hist, s_hist, v_hist = calculate_hsv_histogram(main_img)
                        fig = plot_hsv_histogram(h_hist, s_hist, v_hist)
                        st.plotly_chart(fig, width="stretch")
                except Exception as e:
                    st.error(f"画像読み込みエラー: {e}")
            else:
                st.warning("画像が見つかりません")

            # OOF予測と誤差表示（メインビュー下）
            if oof_df is not None:
                st.markdown("### 📊 OOF予測と誤差")
                target_columns = [
                    "clover_target",
                    "dead_target",
                    "green_target",
                    "gdm_target",
                    "total_target",
                ]
                pred_columns = [f"pred_{col}" for col in target_columns]

                error_data = []
                for target_col, pred_col in zip(target_columns, pred_columns):
                    y_true = pd.to_numeric(main_row.get(target_col, None), errors="coerce")
                    y_pred = pd.to_numeric(main_row.get(pred_col, None), errors="coerce")
                    if pd.notna(y_true) and pd.notna(y_pred):
                        abs_err = abs(float(y_pred) - float(y_true))
                        signed_err = float(y_pred) - float(y_true)
                        error_data.append(
                            {
                                "Target": target_col.replace("_target", ""),
                                "True": f"{float(y_true):.4f}",
                                "Pred": f"{float(y_pred):.4f}",
                                "Abs Err": f"{abs_err:.4f}",
                                "Signed Err": f"{signed_err:.4f}",
                            }
                        )

                if error_data:
                    error_df = pd.DataFrame(error_data)
                    st.dataframe(error_df, width="stretch", hide_index=True)
                else:
                    st.info("OOF予測値がありません")

            # 基本情報（メインビュー直下）
            st.markdown("### 📋 基本情報")
            metadata = {
                "Sample ID": main_row["sample_id"],
            }

            if "sampling_date" in main_row:
                metadata["Sampling Date"] = main_row["sampling_date"]
            if "state" in main_row:
                metadata["State"] = main_row["state"]
            if "species" in main_row:
                metadata["Species"] = main_row["species"]
            if "pre_gshh_ndvi" in main_row:
                metadata["Pre GSHH NDVI"] = main_row["pre_gshh_ndvi"]
            if "height_ave_cm" in main_row:
                metadata["Height Ave (cm)"] = main_row["height_ave_cm"]

            st.json(metadata)

            # メモセクション
            st.markdown("### 📝 メモ")
            notes = load_notes()
            current_note = notes.get(selected_id, {}).get("content", "")
            note_content = st.text_area(
                "メモを入力",
                value=current_note,
                height=100,
                key=f"memo_{selected_id}",
                help="この画像に関するメモを入力できます",
            )
            col_save, col_delete = st.columns(2)
            with col_save:
                if st.button("💾 保存", key=f"save_memo_{selected_id}", width="stretch"):
                    if note_content.strip():
                        save_note(selected_id, note_content.strip())
                        st.success("メモを保存しました")
                        st.rerun()
                    else:
                        st.warning("メモが空です")
            with col_delete:
                if st.button("🗑️ 削除", key=f"delete_memo_{selected_id}", width="stretch"):
                    if selected_id in notes:
                        delete_note(selected_id)
                        st.success("メモを削除しました")
                        st.rerun()
                    else:
                        st.info("メモがありません")

            # 保存済みメモの表示
            if selected_id in notes:
                note_info = notes[selected_id]
                updated_at = note_info.get("updated_at", "")
                st.info(f"📌 最終更新: {updated_at}")

            # ターゲット値表示（メインビュー下）
            st.markdown("### 🎯 ターゲット値")
            targets: dict[str, float] = {}
            if "clover_target" in main_row:
                targets["Clover"] = (
                    float(pd.to_numeric(main_row["clover_target"], errors="coerce"))
                    if pd.notna(main_row["clover_target"])
                    else None
                )  # type: ignore[arg-type]
            if "dead_target" in main_row:
                targets["Dead"] = (
                    float(pd.to_numeric(main_row["dead_target"], errors="coerce"))
                    if pd.notna(main_row["dead_target"])
                    else None
                )  # type: ignore[arg-type]
            if "green_target" in main_row:
                targets["Green"] = (
                    float(pd.to_numeric(main_row["green_target"], errors="coerce"))
                    if pd.notna(main_row["green_target"])
                    else None
                )  # type: ignore[arg-type]
            if "gdm_target" in main_row:
                targets["GDM"] = (
                    float(pd.to_numeric(main_row["gdm_target"], errors="coerce"))
                    if pd.notna(main_row["gdm_target"])
                    else None
                )  # type: ignore[arg-type]
            if "total_target" in main_row:
                targets["Total"] = (
                    float(pd.to_numeric(main_row["total_target"], errors="coerce"))
                    if pd.notna(main_row["total_target"])
                    else None
                )  # type: ignore[arg-type]

            st.json(targets)

            # 派生値・検証値（メインビュー下）
            st.markdown("### 🔍 派生値・検証値")
            derived = calculate_derived_values(main_row)
            st.json(derived)

            # 検証結果の表示
            if abs(derived["gdm_diff"]) > 1e-6:
                st.warning(
                    f"⚠️ GDM不一致: {derived['gdm_diff']:.6f} "
                    f"(期待値: {derived['gdm_calc']:.6f}, 実際: {main_row.get('gdm_target', 'N/A')})"
                )
            else:
                st.success("✅ GDM値は一致しています")

            if abs(derived["total_diff"]) > 1e-6:
                st.warning(
                    f"⚠️ Total不一致: {derived['total_diff']:.6f} "
                    f"(期待値: {derived['total_calc']:.6f}, 実際: {main_row.get('total_target', 'N/A')})"
                )
            else:
                st.success("✅ Total値は一致しています")

    # ギャラリー（スクロール領域）
    with col_right:
        # ギャラリー設定
        st.subheader("ギャラリー設定")
        thumbnail_size = st.slider("サムネイルサイズ (px)", min_value=64, max_value=256, value=96, step=16)
        page_size = st.selectbox("ページサイズ", [12, 24, 48, 96], index=0)
        num_cols = st.selectbox("列数", [2, 3, 4, 5], index=2)

        # ページネーション
        total_pages = max(1, (len(df) + page_size - 1) // page_size)
        page = st.session_state.get("current_page", 0)
        navi1, navi2, navi3 = st.columns(3)
        with navi1:
            if st.button("◀ 前へ", disabled=page == 0, width="stretch", key="page_prev"):
                st.session_state["current_page"] = max(0, page - 1)
                st.rerun()
        with navi2:
            st.markdown(f"**{page + 1}/{total_pages}**")
        with navi3:
            if st.button(
                "次へ ▶",
                disabled=page >= total_pages - 1,
                width="stretch",
                key="page_next",
            ):
                st.session_state["current_page"] = min(total_pages - 1, page + 1)
                st.rerun()

        st.subheader(f"画像ギャラリー ({len(df)}件中)")
        # 現在のページのデータを取得
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, len(df))
        page_df = df.iloc[start_idx:end_idx].copy()

        if len(page_df) == 0:
            st.info("表示する画像がありません")
        else:
            try:
                # Streamlit 1.51: container(height=...) が使用可能
                with st.container(height=720, border=False):
                    gallery_cols = st.columns(num_cols)
                    for idx, (_, row) in enumerate(page_df.iterrows()):
                        col_idx = idx % num_cols
                        with gallery_cols[col_idx]:
                            if row.get("image_path") and Path(row["image_path"]).exists():
                                try:
                                    img = Image.open(row["image_path"])
                                    img.thumbnail(
                                        (thumbnail_size, thumbnail_size),
                                        Image.Resampling.LANCZOS,
                                    )
                                    st.image(img, width="stretch")
                                except Exception as e:
                                    st.error(f"画像読み込みエラー: {e}")
                            else:
                                st.warning("画像が見つかりません")

                            sample_id = row["sample_id"]
                            is_selected = selected_id == sample_id
                            button_label = f"📌 {sample_id}" if is_selected else sample_id
                            if st.button(
                                button_label,
                                key=f"gallery_{sample_id}_{idx}",
                                width="stretch",
                            ):
                                st.session_state["selected_sample_id"] = sample_id
                                st.rerun()
                            # Sample IDボタンの下にSampling Dateを表示
                            if (
                                "sampling_date_str" in row
                                and pd.notna(row["sampling_date_str"])
                                and row["sampling_date_str"] != ""
                            ):
                                date_str = row["sampling_date_str"]
                                st.caption(f"📅 {date_str}")
            except TypeError:
                # height未対応バージョンのフォールバック
                gallery_cols = st.columns(num_cols)
                for idx, (_, row) in enumerate(page_df.iterrows()):
                    col_idx = idx % num_cols
                    with gallery_cols[col_idx]:
                        if row.get("image_path") and Path(row["image_path"]).exists():
                            try:
                                img = Image.open(row["image_path"])
                                img.thumbnail(
                                    (thumbnail_size, thumbnail_size),
                                    Image.Resampling.LANCZOS,
                                )
                                st.image(img, width="stretch")
                            except Exception as e:
                                st.error(f"画像読み込みエラー: {e}")
                        else:
                            st.warning("画像が見つかりません")

                        sample_id = row["sample_id"]
                        is_selected = selected_id == sample_id
                        button_label = f"📌 {sample_id}" if is_selected else sample_id
                        if st.button(
                            button_label,
                            key=f"gallery_{sample_id}_{idx}",
                            width="stretch",
                        ):
                            st.session_state["selected_sample_id"] = sample_id
                            st.rerun()
                        # Sample IDボタンの下にSampling Dateを表示
                        if (
                            "sampling_date_str" in row
                            and pd.notna(row["sampling_date_str"])
                            and row["sampling_date_str"] != ""
                        ):
                            date_str = row["sampling_date_str"]
                            st.caption(f"📅 {date_str}")

    # 右ペインはギャラリー専用


if __name__ == "__main__":
    main()
