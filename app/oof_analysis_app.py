"""OOF Analysis Streamlit App for CSIRO Competition."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constants
OUTPUT_DIR = Path("output")
INPUT_DIR = Path("input")
TRAIN_DIR = INPUT_DIR / "train"
TRAIN_CSV = INPUT_DIR / "train.csv"

# Target configuration
TARGET_NAMES = ["clover_target", "dead_target", "green_target", "gdm_target", "total_target"]
TARGET_DISPLAY = {
    "clover_target": "Clover",
    "dead_target": "Dead",
    "green_target": "Green",
    "gdm_target": "GDM",
    "total_target": "Total",
}
TARGET_WEIGHTS = {
    "clover_target": 0.1,
    "dead_target": 0.1,
    "green_target": 0.1,
    "gdm_target": 0.2,
    "total_target": 0.5,
}

# OOF column mapping (oofs.csv columns to target names)
OOF_COLUMN_MAPPING = {
    "clover_target": "clover_target",
    "dead_target": "dead_target",
    "green_target": "green_target",
    "gdm_target": "gdm_target",
    "total_target": "total_target",
}


@st.cache_data
def list_available_experiments() -> list[str]:
    """List experiments that have OOF predictions."""
    if not OUTPUT_DIR.exists():
        return []

    experiments = []
    for exp_dir in sorted(OUTPUT_DIR.iterdir()):
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp"):
            continue
        if (exp_dir / "oofs.csv").exists() and (exp_dir / "preprocessed_train.csv").exists():
            experiments.append(exp_dir.name)
    return experiments


@st.cache_data
def load_metadata() -> pd.DataFrame | None:
    """Load metadata from train.csv."""
    if not TRAIN_CSV.exists():
        return None

    df = pd.read_csv(TRAIN_CSV)
    # Extract image_id from sample_id (e.g., ID1011485656__Dry_Clover_g -> ID1011485656)
    df["image_id"] = df["sample_id"].str.split("__").str[0]

    # Get unique rows per image_id with metadata
    meta_cols = ["State", "Species", "Sampling_Date"]
    meta_df = df[["image_id"] + meta_cols].drop_duplicates(subset=["image_id"]).reset_index(drop=True)
    meta_df = meta_df.rename(
        columns={
            "image_id": "sample_id",
            "State": "state",
            "Species": "species",
            "Sampling_Date": "sampling_date",
        }
    )
    return meta_df


@st.cache_data
def load_experiment_data(exp_name: str) -> tuple[pd.DataFrame | None, dict | None]:
    """Load OOF predictions and merge with actual values."""
    exp_dir = OUTPUT_DIR / exp_name

    oof_csv = exp_dir / "oofs.csv"
    preprocessed_csv = exp_dir / "preprocessed_train.csv"
    results_json = exp_dir / "results.json"

    if not oof_csv.exists() or not preprocessed_csv.exists():
        return None, None

    # Load preprocessed_train.csv (contains actual values and sample_id)
    actual_df = pd.read_csv(preprocessed_csv)

    # Load oofs.csv (predictions)
    oof_df = pd.read_csv(oof_csv)

    # Verify row count match
    if len(oof_df) != len(actual_df):
        st.warning(f"Row count mismatch: OOF={len(oof_df)}, Actual={len(actual_df)}")
        return None, None

    # Merge by index (OOF and preprocessed_train should be aligned)
    df = actual_df.copy()

    # Add predictions with pred_ prefix
    for target in TARGET_NAMES:
        oof_col = OOF_COLUMN_MAPPING.get(target, target)
        if oof_col in oof_df.columns:
            df[f"pred_{target}"] = oof_df[oof_col].values

    # Calculate residuals
    for target in TARGET_NAMES:
        actual_col = target
        pred_col = f"pred_{target}"
        if actual_col in df.columns and pred_col in df.columns:
            df[f"residual_{target}"] = df[pred_col] - df[actual_col]

    # Load metadata if available
    meta_df = load_metadata()
    if meta_df is not None:
        # Only merge if state/species not already in df
        if "state" not in df.columns:
            df = df.merge(meta_df, on="sample_id", how="left")

    # Load results.json
    results = None
    if results_json.exists():
        with open(results_json) as f:
            results = json.load(f)

    return df, results


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate regression metrics."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) < 2:
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan, "n": 0}

    return {
        "r2": r2_score(y_true_clean, y_pred_clean),
        "mae": mean_absolute_error(y_true_clean, y_pred_clean),
        "rmse": np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        "n": len(y_true_clean),
    }


def calculate_weighted_r2(df: pd.DataFrame) -> float:
    """Calculate weighted R2 score."""
    weighted_sum = 0.0
    total_weight = 0.0

    for target, weight in TARGET_WEIGHTS.items():
        if target in df.columns and f"pred_{target}" in df.columns:
            y_true = df[target].values
            y_pred = df[f"pred_{target}"].values
            metrics = calculate_metrics(y_true, y_pred)
            if not np.isnan(metrics["r2"]):
                weighted_sum += weight * metrics["r2"]
                total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else np.nan


def check_physical_constraints(df: pd.DataFrame) -> dict:
    """Check physical constraints: GDM=C+G, Total=C+D+G."""
    results = {}

    # GDM = Clover + Green
    if all(col in df.columns for col in ["pred_clover_target", "pred_green_target", "pred_gdm_target"]):
        gdm_calc = df["pred_clover_target"] + df["pred_green_target"]
        gdm_diff = np.abs(df["pred_gdm_target"] - gdm_calc)
        gdm_ok = (gdm_diff < 0.01).sum()
        results["gdm"] = {
            "ok_count": int(gdm_ok),
            "ng_count": int(len(df) - gdm_ok),
            "ok_ratio": float(gdm_ok / len(df)) if len(df) > 0 else 0,
        }

    # Total = Clover + Dead + Green
    if all(
        col in df.columns
        for col in ["pred_clover_target", "pred_dead_target", "pred_green_target", "pred_total_target"]
    ):
        total_calc = df["pred_clover_target"] + df["pred_dead_target"] + df["pred_green_target"]
        total_diff = np.abs(df["pred_total_target"] - total_calc)
        total_ok = (total_diff < 0.01).sum()
        results["total"] = {
            "ok_count": int(total_ok),
            "ng_count": int(len(df) - total_ok),
            "ok_ratio": float(total_ok / len(df)) if len(df) > 0 else 0,
        }

    return results


def render_overview_tab(df: pd.DataFrame, results: dict | None):
    """Render Overview tab content."""
    st.header("Overview")

    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)

    weighted_r2 = calculate_weighted_r2(df)

    with col1:
        st.metric("Mean Weighted R2", f"{weighted_r2:.4f}" if not np.isnan(weighted_r2) else "N/A")

    if results and "fold_scores" in results:
        fold_scores = results["fold_scores"]
        with col2:
            st.metric("Std", f"{results.get('std_score', np.std(fold_scores)):.4f}")
        with col3:
            best_fold = np.argmax(fold_scores)
            st.metric("Best Fold", f"Fold {best_fold}")
        with col4:
            worst_fold = np.argmin(fold_scores)
            st.metric("Worst Fold", f"Fold {worst_fold}")

    st.divider()

    # Two columns: Target Scores and Distribution
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Target Scores")
        target_data = []
        for target in TARGET_NAMES:
            if target in df.columns and f"pred_{target}" in df.columns:
                metrics = calculate_metrics(df[target].values, df[f"pred_{target}"].values)
                target_data.append(
                    {
                        "Target": TARGET_DISPLAY[target],
                        "Weight": TARGET_WEIGHTS[target],
                        "R2": f"{metrics['r2']:.4f}" if not np.isnan(metrics["r2"]) else "N/A",
                        "MAE": f"{metrics['mae']:.4f}" if not np.isnan(metrics["mae"]) else "N/A",
                        "RMSE": f"{metrics['rmse']:.4f}" if not np.isnan(metrics["rmse"]) else "N/A",
                    }
                )
        st.dataframe(pd.DataFrame(target_data), hide_index=True, use_container_width=True)

    with col_right:
        st.subheader("GT vs OOF Distribution")
        # Create histogram for selected target
        target_for_hist = st.selectbox(
            "Select target", list(TARGET_DISPLAY.keys()), format_func=lambda x: TARGET_DISPLAY[x], key="hist_target"
        )
        if target_for_hist in df.columns and f"pred_{target_for_hist}" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[target_for_hist], name="GT", opacity=0.7, marker_color="blue", nbinsx=50))
            fig.add_trace(
                go.Histogram(
                    x=df[f"pred_{target_for_hist}"], name="OOF Pred", opacity=0.7, marker_color="orange", nbinsx=50
                )
            )
            fig.update_layout(
                barmode="overlay",
                xaxis_title="Value",
                yaxis_title="Count",
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Fold Scores and Physical Constraints
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Fold Scores")
        if results and "fold_scores" in results:
            fold_scores = results["fold_scores"]
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[f"Fold {i}" for i in range(len(fold_scores))],
                    y=fold_scores,
                    marker_color=[
                        "green" if s == max(fold_scores) else "red" if s == min(fold_scores) else "steelblue"
                        for s in fold_scores
                    ],
                )
            )
            fig.update_layout(
                xaxis_title="Fold",
                yaxis_title="Score",
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fold scores not available")

    with col_right2:
        st.subheader("Physical Constraint Check")
        constraints = check_physical_constraints(df)

        if "gdm" in constraints:
            gdm = constraints["gdm"]
            st.write("**GDM = Clover + Green**")
            st.write(f"  OK: {gdm['ok_count']} ({gdm['ok_ratio']:.1%}) | NG: {gdm['ng_count']}")

        if "total" in constraints:
            total = constraints["total"]
            st.write("**Total = Clover + Dead + Green**")
            st.write(f"  OK: {total['ok_count']} ({total['ok_ratio']:.1%}) | NG: {total['ng_count']}")


def render_detail_tab(df: pd.DataFrame, results: dict | None):
    """Render Detail tab content."""
    st.header("Detail Analysis")

    # Section selector
    section = st.radio(
        "Section",
        ["Scatter", "Residual", "Fold", "Segment", "Correlation"],
        horizontal=True,
    )

    if section == "Scatter":
        render_scatter_section(df)
    elif section == "Residual":
        render_residual_section(df)
    elif section == "Fold":
        render_fold_section(df, results)
    elif section == "Segment":
        render_segment_section(df)
    elif section == "Correlation":
        render_correlation_section(df)


def render_scatter_section(df: pd.DataFrame):
    """Render scatter plots for predictions vs actuals."""
    st.subheader("Prediction vs Actual")

    cols = st.columns(3)
    for idx, target in enumerate(TARGET_NAMES):
        if target not in df.columns or f"pred_{target}" not in df.columns:
            continue

        with cols[idx % 3]:
            y_true = df[target].values
            y_pred = df[f"pred_{target}"].values
            metrics = calculate_metrics(y_true, y_pred)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=y_true,
                    y=y_pred,
                    mode="markers",
                    marker=dict(size=4, opacity=0.5),
                    name="Data",
                )
            )

            # Add y=x line
            min_val = min(np.nanmin(y_true), np.nanmin(y_pred))
            max_val = max(np.nanmax(y_true), np.nanmax(y_pred))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color="red", dash="dash"),
                    name="y=x",
                )
            )

            fig.update_layout(
                title=f"{TARGET_DISPLAY[target]} (R2={metrics['r2']:.3f})",
                xaxis_title="Actual",
                yaxis_title="Predicted",
                height=300,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)


def render_residual_section(df: pd.DataFrame):
    """Render residual analysis."""
    st.subheader("Residual Analysis")

    target = st.selectbox(
        "Select target",
        TARGET_NAMES,
        format_func=lambda x: TARGET_DISPLAY[x],
        key="residual_target",
    )

    if f"residual_{target}" not in df.columns:
        st.warning("Residual data not available")
        return

    residuals = df[f"residual_{target}"].dropna()

    col1, col2 = st.columns(2)

    with col1:
        # Histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=residuals, nbinsx=50, marker_color="steelblue"))
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Residual Histogram",
            xaxis_title="Residual",
            yaxis_title="Count",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Q-Q Plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals.values)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode="markers",
                marker=dict(size=4, opacity=0.5),
            )
        )
        # Add reference line
        min_q = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_q = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(
                x=[min_q, max_q],
                y=[min_q, max_q],
                mode="lines",
                line=dict(color="red", dash="dash"),
            )
        )
        fig.update_layout(
            title="Q-Q Plot",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Residual vs Predicted
    if f"pred_{target}" in df.columns:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[f"pred_{target}"],
                y=df[f"residual_{target}"],
                mode="markers",
                marker=dict(size=4, opacity=0.5),
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Residual vs Predicted",
            xaxis_title="Predicted",
            yaxis_title="Residual",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


def render_fold_section(df: pd.DataFrame, results: dict | None):
    """Render fold comparison."""
    st.subheader("Fold Comparison")

    if "fold" not in df.columns:
        st.warning("Fold information not available")
        return

    # Fold scores bar chart
    if results and "fold_scores" in results:
        fold_scores = results["fold_scores"]
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[f"Fold {i}" for i in range(len(fold_scores))],
                y=fold_scores,
                marker_color="steelblue",
            )
        )
        fig.add_hline(y=np.mean(fold_scores), line_dash="dash", line_color="red", annotation_text="Mean")
        fig.update_layout(
            title="Fold Scores",
            xaxis_title="Fold",
            yaxis_title="Score",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Target x Fold heatmap
    st.subheader("Target x Fold R2 Heatmap")
    fold_target_scores = []
    folds = sorted(df["fold"].unique())

    for fold in folds:
        fold_df = df[df["fold"] == fold]
        row = {"Fold": f"Fold {fold}"}
        for target in TARGET_NAMES:
            if target in fold_df.columns and f"pred_{target}" in fold_df.columns:
                metrics = calculate_metrics(fold_df[target].values, fold_df[f"pred_{target}"].values)
                row[TARGET_DISPLAY[target]] = metrics["r2"]
        fold_target_scores.append(row)

    if fold_target_scores:
        heatmap_df = pd.DataFrame(fold_target_scores).set_index("Fold")
        fig = px.imshow(
            heatmap_df.values,
            x=heatmap_df.columns.tolist(),
            y=heatmap_df.index.tolist(),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            text_auto=".3f",
        )
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)


def render_segment_section(df: pd.DataFrame):
    """Render segment analysis."""
    st.subheader("Segment Analysis")

    segment_col = st.selectbox("Segment by", ["state", "species"], key="segment_col")

    if segment_col not in df.columns or df[segment_col].isna().all():
        st.warning(f"{segment_col} information not available")
        return

    target = st.selectbox(
        "Select target",
        TARGET_NAMES,
        format_func=lambda x: TARGET_DISPLAY[x],
        key="segment_target",
    )

    # Calculate metrics per segment
    segments = df[segment_col].dropna().unique()
    segment_data = []

    for seg in segments:
        seg_df = df[df[segment_col] == seg]
        if target in seg_df.columns and f"pred_{target}" in seg_df.columns:
            metrics = calculate_metrics(seg_df[target].values, seg_df[f"pred_{target}"].values)
            segment_data.append(
                {
                    "Segment": seg,
                    "N": metrics["n"],
                    "R2": metrics["r2"],
                    "MAE": metrics["mae"],
                }
            )

    if segment_data:
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(
                pd.DataFrame(segment_data).round(4),
                hide_index=True,
                use_container_width=True,
            )

        with col2:
            # Box plot of residuals by segment
            if f"residual_{target}" in df.columns:
                fig = px.box(
                    df.dropna(subset=[segment_col, f"residual_{target}"]),
                    x=segment_col,
                    y=f"residual_{target}",
                    title="Residual Distribution by Segment",
                )
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)


def render_correlation_section(df: pd.DataFrame):
    """Render error correlation heatmap."""
    st.subheader("Error Correlation")

    # Calculate correlation matrix of residuals
    residual_cols = [f"residual_{t}" for t in TARGET_NAMES if f"residual_{t}" in df.columns]

    if len(residual_cols) < 2:
        st.warning("Not enough residual data for correlation analysis")
        return

    corr_df = df[residual_cols].corr()
    # Rename columns for display
    corr_df.columns = [TARGET_DISPLAY[c.replace("residual_", "")] for c in corr_df.columns]
    corr_df.index = corr_df.columns

    fig = px.imshow(
        corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
    )
    fig.update_layout(
        title="Residual Correlation Matrix",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def load_thumbnail(image_path: str, size: int = 150) -> Image.Image | None:
    """Load and resize image for thumbnail."""
    path = Path(image_path)
    if not path.exists():
        return None
    try:
        img = Image.open(path)
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        return img
    except Exception:
        return None


def render_gallery_tab(df: pd.DataFrame):
    """Render Gallery tab content."""
    st.header("Sample Gallery")

    # Sort and filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        sort_options = ["Total Error DESC", "Total Error ASC", "Sample ID"]
        sort_by = st.selectbox("Sort by", sort_options, key="gallery_sort")

    with col2:
        if "fold" in df.columns:
            fold_options = ["All"] + [f"Fold {f}" for f in sorted(df["fold"].unique())]
            selected_fold = st.selectbox("Filter Fold", fold_options, key="gallery_fold")
        else:
            selected_fold = "All"

    with col3:
        if "state" in df.columns and not df["state"].isna().all():
            state_options = ["All"] + sorted(df["state"].dropna().unique().tolist())
            selected_state = st.selectbox("Filter State", state_options, key="gallery_state")
        else:
            selected_state = "All"

    # Apply filters
    filtered_df = df.copy()

    if selected_fold != "All":
        fold_num = int(selected_fold.split(" ")[1])
        filtered_df = filtered_df[filtered_df["fold"] == fold_num]

    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["state"] == selected_state]

    # Calculate weighted error for sorting
    weighted_errors = []
    for _, row in filtered_df.iterrows():
        total_err = 0.0
        total_weight = 0.0
        for target, weight in TARGET_WEIGHTS.items():
            if f"residual_{target}" in filtered_df.columns:
                res = row.get(f"residual_{target}", np.nan)
                if not pd.isna(res):
                    total_err += weight * abs(res)
                    total_weight += weight
        weighted_errors.append(total_err / total_weight if total_weight > 0 else np.nan)

    filtered_df = filtered_df.copy()
    filtered_df["weighted_error"] = weighted_errors

    # Apply sorting
    if sort_by == "Total Error DESC":
        filtered_df = filtered_df.sort_values("weighted_error", ascending=False)
    elif sort_by == "Total Error ASC":
        filtered_df = filtered_df.sort_values("weighted_error", ascending=True)
    else:
        filtered_df = filtered_df.sort_values("sample_id")

    filtered_df = filtered_df.reset_index(drop=True)

    # Pagination
    page_size = 12
    total_pages = max(1, (len(filtered_df) + page_size - 1) // page_size)
    current_page = st.session_state.get("gallery_page", 0)

    col_prev, col_info, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("< Prev", disabled=current_page <= 0, key="gallery_prev"):
            st.session_state["gallery_page"] = current_page - 1
            st.rerun()
    with col_info:
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        st.write(f"Showing {start_idx + 1}-{end_idx} of {len(filtered_df)}")
    with col_next:
        if st.button("Next >", disabled=current_page >= total_pages - 1, key="gallery_next"):
            st.session_state["gallery_page"] = current_page + 1
            st.rerun()

    # Display sample cards
    page_df = filtered_df.iloc[start_idx:end_idx]

    cols = st.columns(3)
    for idx, (_, row) in enumerate(page_df.iterrows()):
        with cols[idx % 3]:
            with st.container(border=True):
                # Image thumbnail
                sample_id = row["sample_id"]

                # Try to find image path
                image_path = None
                if "image_path" in row and pd.notna(row["image_path"]):
                    # Check if it's a .npy path and convert to .jpg
                    img_path = Path(str(row["image_path"]))
                    if img_path.suffix == ".npy":
                        jpg_path = TRAIN_DIR / f"{sample_id}.jpg"
                        if jpg_path.exists():
                            image_path = str(jpg_path)
                    elif img_path.exists():
                        image_path = str(img_path)
                else:
                    jpg_path = TRAIN_DIR / f"{sample_id}.jpg"
                    if jpg_path.exists():
                        image_path = str(jpg_path)

                if image_path:
                    img = load_thumbnail(image_path)
                    if img:
                        st.image(img, use_container_width=True)
                    else:
                        st.write("Image load error")
                else:
                    st.write("No image")

                # Sample info
                fold_str = f"F{int(row['fold'])}" if "fold" in row and pd.notna(row["fold"]) else "?"
                state_str = row.get("state", "?") if pd.notna(row.get("state")) else "?"
                st.write(f"**{sample_id}**")
                st.write(f"{fold_str} | {state_str}")

                # GT/Pred/Error table
                st.markdown("---")
                error_data = []
                for target in TARGET_NAMES:
                    gt = row.get(target, np.nan)
                    pred = row.get(f"pred_{target}", np.nan)
                    err = row.get(f"residual_{target}", np.nan)

                    gt_str = f"{gt:.1f}" if pd.notna(gt) else "-"
                    pred_str = f"{pred:.1f}" if pd.notna(pred) else "-"
                    if pd.notna(err):
                        err_str = f"+{err:.1f}" if err >= 0 else f"{err:.1f}"
                    else:
                        err_str = "-"

                    error_data.append(
                        {
                            "": TARGET_DISPLAY[target][:3],
                            "GT": gt_str,
                            "Pred": pred_str,
                            "Err": err_str,
                        }
                    )

                # Display as compact table
                error_df = pd.DataFrame(error_data)
                st.dataframe(error_df, hide_index=True, use_container_width=True, height=220)

                # Weighted error
                w_err = row.get("weighted_error", np.nan)
                if pd.notna(w_err):
                    st.write(f"**Weighted: {w_err:.2f}**")


def main():
    st.set_page_config(
        page_title="OOF Analysis - CSIRO",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 OOF Analysis - CSIRO")

    # Sidebar
    st.sidebar.header("Settings")

    # Experiment selection
    experiments = list_available_experiments()
    if not experiments:
        st.warning("No experiments with OOF predictions found")
        st.stop()

    selected_exp = st.sidebar.selectbox("Experiment", experiments)

    # Load data
    df, results = load_experiment_data(selected_exp)
    if df is None:
        st.error("Failed to load experiment data")
        st.stop()

    st.sidebar.success(f"Loaded {len(df)} samples")

    # Target filter (UI element for future filtering, currently display only)
    st.sidebar.subheader("Targets")
    _selected_targets = st.sidebar.multiselect(  # noqa: F841
        "Select targets",
        TARGET_NAMES,
        default=TARGET_NAMES,
        format_func=lambda x: TARGET_DISPLAY[x],
    )

    # Tabs
    tab_overview, tab_detail, tab_gallery = st.tabs(["Overview", "Detail", "Gallery"])

    with tab_overview:
        render_overview_tab(df, results)

    with tab_detail:
        render_detail_tab(df, results)

    with tab_gallery:
        render_gallery_tab(df)


if __name__ == "__main__":
    main()
