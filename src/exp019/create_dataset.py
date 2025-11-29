from dataclasses import dataclass
from pathlib import Path
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
    """Create metadata DataFrame with original image paths (no npy conversion)."""
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
        # Use original image path directly instead of npy
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


def main():
    # Load config
    exp_name = Path(__file__).parent.name
    config = OmegaConf.load(f"config/{exp_name}.yaml")
    output_dir = Path(config.dataset.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(config.dataset.input_dir)

    # Load data
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
    for fold, (_, val_idx) in enumerate(sgkf.split(df, y=df["species"], groups=groups)):
        df.loc[val_idx, "fold"] = fold

    # Save data
    df.to_csv(output_dir / "preprocessed_train.csv", index=False)


if __name__ == "__main__":
    main()
