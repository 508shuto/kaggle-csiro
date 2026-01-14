from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class DataItem:
    sample_id: str
    image_path: Path
    sampling_date: str
    state: str
    species: str
    pre_gshh_ndvi: float
    height_ave_cm: float
    clover_target: float
    dead_target: float
    green_target: float
    gdm_target: float
    total_target: float


def create_metadata(df: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    # group by image_path
    data_items: list[DataItem] = []
    image_paths = df["image_path"].unique()
    for image_path in image_paths:
        image_df = df[df["image_path"] == image_path]
        sample_id = image_df["sample_id"].iloc[0].split("__")[0]
        sampling_date = image_df["Sampling_Date"].iloc[0]
        state = image_df["State"].iloc[0]
        species = image_df["Species"].iloc[0]
        pre_gshh_ndvi = image_df["Pre_GSHH_NDVI"].iloc[0]
        height_ave_cm = image_df["Height_Ave_cm"].iloc[0]

        target_dict: dict[str, float] = {}
        for _, row in image_df.iterrows():
            if row["target_name"] == "Dry_Clover_g":
                target_dict["clover_target"] = row["target"]
            elif row["target_name"] == "Dry_Dead_g":
                target_dict["dead_target"] = row["target"]
            elif row["target_name"] == "Dry_Green_g":
                target_dict["green_target"] = row["target"]
            elif row["target_name"] == "GDM_g":
                target_dict["gdm_target"] = row["target"]
            elif row["target_name"] == "Dry_Total_g":
                target_dict["total_target"] = row["target"]
            else:
                raise ValueError(f"Target name {row['target_name']} not found")
        # JPEGファイルのフルパスを生成（input/train/xxx.jpg）
        data_path = input_dir / image_path
        data_items.append(
            DataItem(
                sample_id=sample_id,
                image_path=data_path,
                sampling_date=sampling_date,
                state=state,
                species=species,
                pre_gshh_ndvi=pre_gshh_ndvi,
                height_ave_cm=height_ave_cm,
                clover_target=target_dict["clover_target"],
                dead_target=target_dict["dead_target"],
                green_target=target_dict["green_target"],
                gdm_target=target_dict["gdm_target"],
                total_target=target_dict["total_target"],
            )
        )
    df = pd.DataFrame(data_items)
    return df


def compute_sampling_weights(
    df: pd.DataFrame,
    wa_weight: float = 3.0,
    q1_weight: float = 2.0,
) -> np.ndarray:
    """サンプリング重みを計算

    Args:
        df: データフレーム
        wa_weight: WA州の重み倍率
        q1_weight: Q1（total_target≤25%ile）の重み倍率

    Returns:
        サンプリング重み配列
    """
    weights = np.ones(len(df))

    # WA州: wa_weight倍
    wa_mask = df["state"] == "WA"
    weights[wa_mask] = wa_weight

    # Q1（total_target≤25%ile）: q1_weight倍
    q1_threshold = df["total_target"].quantile(0.25)
    q1_mask = df["total_target"] <= q1_threshold
    weights[q1_mask] *= q1_weight  # WA & Q1は wa_weight * q1_weight 倍

    print("Sampling weights computed:")
    print(f"  WA weight: {wa_weight}, Q1 weight: {q1_weight}")
    print(f"  Q1 threshold (25%ile): {q1_threshold:.2f}")
    print(f"  WA samples: {wa_mask.sum()}")
    print(f"  Q1 samples: {q1_mask.sum()}")
    print(f"  WA & Q1 samples: {(wa_mask & q1_mask).sum()}")
    print(f"  Weight distribution: min={weights.min():.1f}, max={weights.max():.1f}, mean={weights.mean():.2f}")

    return weights


def main():
    # Load config
    exp_name = Path(__file__).parent.name
    config = OmegaConf.load(f"config/{exp_name}.yaml")
    output_dir = Path(config.dataset.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    input_dir = Path(config.dataset.input_dir)
    df = pd.read_csv(input_dir / "train.csv")

    df = create_metadata(df, input_dir)

    # Split data
    df["fold"] = -1
    # Use StratifiedGroupKFold to prevent leakage across Sampling_Date+State groups
    groups = df["sampling_date"] + "_" + df["state"]
    sgkf = StratifiedGroupKFold(
        n_splits=config.dataset.n_folds,
        shuffle=True,
        random_state=config.experiment.seed,
    )
    # exp040: State層化（species → state）
    for fold, (_, val_idx) in enumerate(sgkf.split(df, y=df["state"], groups=groups)):
        df.loc[val_idx, "fold"] = fold

    # オーバーサンプリング設定の取得と重み計算
    use_oversampling = config.dataset.get("use_oversampling", False)
    if use_oversampling:
        oversampling_cfg = config.dataset.get("oversampling", {})
        wa_weight = oversampling_cfg.get("wa_weight", 3.0)
        q1_weight = oversampling_cfg.get("q1_weight", 2.0)
        df["sample_weight"] = compute_sampling_weights(df, wa_weight, q1_weight)
    else:
        df["sample_weight"] = 1.0

    # Save data
    df.to_csv(output_dir / "preprocessed_train.csv", index=False)


if __name__ == "__main__":
    main()
