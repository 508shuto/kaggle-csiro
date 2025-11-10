import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from utils import get_transforms


class CSIRODataset(Dataset):
    def __init__(self, config: DictConfig, df: pd.DataFrame, mode: str):
        assert mode in ["train", "valid", "test"], mode
        self.config = config
        self.df = df
        self.mode = mode
        self.transform = get_transforms(mode, self.config)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = np.load(row["image_path"])

        # log1p変換: log(1 + x) で0を含む値も扱える
        target_raw = torch.tensor(
            [
                row["clover_target"].item(),
                row["dead_target"].item(),
                row["green_target"].item(),
                row["gdm_target"].item(),
                row["total_target"].item(),
            ]
        )
        target = torch.log1p(target_raw)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = torch.from_numpy(image).float()

        return image, target
