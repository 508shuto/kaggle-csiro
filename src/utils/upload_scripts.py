import os
import shutil
from pathlib import Path

import kagglehub
import tyro


def upload_dataset(dataset_name: str, target_dir: str):
    """
    Upload the dataset to the kaggle dataset.
    """
    kagglehub.dataset_upload(
        f"{os.getenv('KAGGLE_USERNAME')}/{dataset_name}",
        target_dir,
    )


def main(exp_name: str):
    # upload scripts
    src_dataset_name: str = "kaggle-csiro-src"
    src_dir: str = f"./src/{exp_name}"
    upload_dataset(src_dataset_name, src_dir)

    # upload weights & config
    model_dataset_name: str = f"kaggle-csiro-{exp_name}"
    weights_dir: str = f"./output/{exp_name}"
    config_path: str = f"./config/{exp_name}.yaml"
    tmp_dir: Path = Path(f"./tmp/{exp_name}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # copy only ckpt files
    for file in Path(weights_dir).glob("*.ckpt"):
        shutil.copy(file, tmp_dir)
    # copy config
    shutil.copy(config_path, tmp_dir / "config.yaml")
    # upload
    upload_dataset(model_dataset_name, str(tmp_dir))
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    tyro.cli(main)
