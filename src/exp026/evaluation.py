import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro
from dataset import CSIRODataset
from lightning_module import CSIROModule
from metrics import compute_metrics
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm


def main(
    model_dir: Path = Path("./output/exp026"),
    folds: list[int] | None = None,
    device: str = "cuda",
):
    if folds is None:
        folds = [0, 1, 2, 3, 4]
    device = torch.device(device)
    exp_name = Path(__file__).parent.name
    config: DictConfig = OmegaConf.load(f"config/{exp_name}.yaml")

    config.dataset.output_dir = str(Path(config.dataset.output_dir) / exp_name)
    df: pd.DataFrame = pd.read_csv(Path(config.dataset.output_dir) / "preprocessed_train.csv")

    label_columns = [
        "clover_target",
        "dead_target",
        "green_target",
        "gdm_target",
        "total_target",
    ]

    oofs = np.zeros((len(df), len(label_columns)))
    fold_scores = []

    for fold in folds:
        valid_df: pd.DataFrame = df[df["fold"] == fold]
        dataset = CSIRODataset(config, valid_df, "valid")
        dataloader = DataLoader(
            dataset,
            batch_size=config.trainer.valid.batch_size,
            shuffle=False,
            num_workers=config.trainer.valid.num_workers,
        )

        model_path = list(model_dir.glob(f"fold{fold}*.ckpt"))[0]
        model = (
            CSIROModule.load_from_checkpoint(
                checkpoint_path=model_path,
                config=config,
                map_location=device,
            )
            .model_ema.module.eval()
            .to(device)
        )

        fold_predictions = []
        fold_targets = []
        for batch in tqdm(dataloader, desc=f"Fold {fold}"):
            images, targets, _ = batch
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                predictions, _ = model(images)
            fold_predictions.append(predictions.cpu().detach())
            fold_targets.append(targets.cpu().detach())

        fold_predictions = torch.cat(fold_predictions, dim=0)
        fold_targets = torch.cat(fold_targets, dim=0)

        # Clamp to non-negative
        fold_predictions = torch.clamp(fold_predictions, min=0.0)

        # Convert to numpy
        fold_predictions_np = fold_predictions.numpy()
        fold_targets_np = fold_targets.numpy()

        # Compute metrics
        score = compute_metrics(fold_predictions_np, fold_targets_np)
        fold_scores.append(score)

        oofs[valid_df.index] = fold_predictions_np

    # Save OOF predictions
    oofs_df = pd.DataFrame(oofs, columns=label_columns, index=df.index)
    output_dir = Path(config.dataset.output_dir)
    oofs_df.to_csv(output_dir / "oofs.csv", index=False)

    # Save results
    results = {
        "fold_scores": fold_scores,
        "mean_score": np.mean(fold_scores),
        "std_score": np.std(fold_scores),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f)

    print(f"Fold scores: {fold_scores}")
    print(f"Mean score: {np.mean(fold_scores)}")
    print(f"Std score: {np.std(fold_scores)}")


if __name__ == "__main__":
    tyro.cli(main)
