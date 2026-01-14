"""事前にQuantile重みを計算してJSONに保存"""
import json

import numpy as np
import pandas as pd


def compute_quantile_weights(
    df: pd.DataFrame,
    target_cols: list[str],
    n_bins: int = 5,
    w_min: float = 0.5,
    w_max: float = 2.0,
    epsilon: float = 1e-8,
) -> dict:
    """各ターゲットの分位点境界と逆頻度重みを計算"""
    result = {}

    for col in target_cols:
        values = df[col].values

        # 分位点境界を計算 (n_bins + 1 edges)
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(values, quantiles)

        # 各ビンの頻度を計算
        bin_indices = np.digitize(values, bin_edges[1:-1])  # 0 to n_bins-1
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_freq = bin_counts / len(values)

        # 逆頻度重み（均等なら全て1.0）
        weights = 1.0 / (bin_freq * n_bins + epsilon)
        weights = np.clip(weights, w_min, w_max)
        weights = weights / weights.mean()  # normalize to mean 1

        result[col] = {
            "bin_edges": bin_edges.tolist(),
            "weights": weights.tolist(),
        }

    return result


def load_train_wide_format(csv_path: str) -> pd.DataFrame:
    """Long format の train.csv を Wide format に変換"""
    df = pd.read_csv(csv_path)

    # image_id を抽出
    df["image_id"] = df["sample_id"].str.split("__").str[0]

    # Long → Wide に変換
    df_wide = df.pivot(
        index="image_id",
        columns="target_name",
        values="target",
    ).reset_index()

    # カラム名をリネーム
    df_wide.columns.name = None
    rename_map = {
        "Dry_Clover_g": "clover_target",
        "Dry_Dead_g": "dead_target",
        "Dry_Green_g": "green_target",
        "GDM_g": "gdm_target",
        "Dry_Total_g": "total_target",
    }
    df_wide = df_wide.rename(columns=rename_map)

    return df_wide


if __name__ == "__main__":
    # Long format → Wide format
    df = load_train_wide_format("input/train.csv")
    print(f"Loaded {len(df)} samples")

    target_cols = [
        "clover_target",
        "dead_target",
        "green_target",
        "gdm_target",
        "total_target",
    ]

    weights_dict = compute_quantile_weights(df, target_cols, n_bins=5)

    with open("config/exp044_quantile_weights.json", "w") as f:
        json.dump(weights_dict, f, indent=2)

    print("Saved to config/exp044_quantile_weights.json")

    # Print summary
    for col, data in weights_dict.items():
        print(f"\n{col}:")
        print(f"  Bin edges: {[f'{e:.2f}' for e in data['bin_edges']]}")
        print(f"  Weights: {[f'{w:.3f}' for w in data['weights']]}")
