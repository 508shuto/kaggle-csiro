import sys
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import json
import argparse

# プロジェクトルートをPATHに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.exp100.utils import compute_weighted_r2, print_metrics


def evaluate_fold(config, fold: int, data_type: str = "valid"):
    """
    1つのfoldの予測結果を評価

    Args:
        config: 設定
        fold: fold番号
        data_type: "train" or "valid"

    Returns:
        メトリクス辞書
    """
    # 予測結果を読み込み
    pred_path = (
        Path(config.training.output_dir)
        / f"fold{fold}"
        / f"predictions_{data_type}.csv"
    )

    if not pred_path.exists():
        print(f"Warning: {pred_path} not found. Skipping fold {fold}.")
        return None

    df = pd.read_csv(pred_path)

    # 予測値と正解値を抽出
    predictions = df[
        ["pred_clover", "pred_dead", "pred_green", "pred_gdm", "pred_total"]
    ].values
    targets = df[
        ["target_clover", "target_dead", "target_green", "target_gdm", "target_total"]
    ].values

    # メトリクスを計算
    metrics = compute_weighted_r2(predictions, targets)

    return metrics


def evaluate_all_folds(config, data_type: str = "valid"):
    """
    全foldの評価結果を集計

    Args:
        config: 設定
        data_type: "train" or "valid"

    Returns:
        全体のメトリクス
    """
    print(f"\n{'='*80}")
    print(f"Evaluating all folds - {data_type}")
    print(f"{'='*80}\n")

    all_metrics = {}
    metric_names = [
        "weighted_r2",
        "r2_clover",
        "r2_dead",
        "r2_green",
        "r2_gdm",
        "r2_total",
    ]

    # 各foldを評価
    for fold in range(config.dataset.n_folds):
        print(f"\nFold {fold}:")
        metrics = evaluate_fold(config, fold, data_type)

        if metrics is None:
            continue

        all_metrics[f"fold{fold}"] = metrics

        # メトリクス表示
        for metric_name in metric_names:
            print(f"  {metric_name}: {metrics[metric_name]:.4f}")

    # 統計を計算
    print(f"\n{'='*80}")
    print(f"Overall Statistics - {data_type}")
    print(f"{'='*80}\n")

    overall_stats = {}
    for metric_name in metric_names:
        values = [
            all_metrics[f"fold{fold}"][metric_name]
            for fold in range(config.dataset.n_folds)
            if f"fold{fold}" in all_metrics
        ]

        if len(values) > 0:
            mean_value = np.mean(values)
            std_value = np.std(values)
            min_value = np.min(values)
            max_value = np.max(values)

            overall_stats[metric_name] = {
                "mean": float(mean_value),
                "std": float(std_value),
                "min": float(min_value),
                "max": float(max_value),
            }

            print(f"{metric_name}:")
            print(f"  Mean: {mean_value:.4f}")
            print(f"  Std:  {std_value:.4f}")
            print(f"  Min:  {min_value:.4f}")
            print(f"  Max:  {max_value:.4f}")

    # 結果を保存
    output_dir = Path(config.training.output_dir)
    output_path = output_dir / f"evaluation_{data_type}.json"

    results = {
        "fold_metrics": all_metrics,
        "overall_stats": overall_stats,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")
    print(f"{'='*80}\n")

    return results


def create_oof_predictions(config):
    """
    OOF（Out-of-Fold）予測を作成

    Args:
        config: 設定

    Returns:
        OOF予測のDataFrame
    """
    print(f"\n{'='*80}")
    print("Creating OOF predictions")
    print(f"{'='*80}\n")

    all_dfs = []

    for fold in range(config.dataset.n_folds):
        pred_path = (
            Path(config.training.output_dir) / f"fold{fold}" / "predictions_valid.csv"
        )

        if not pred_path.exists():
            print(f"Warning: {pred_path} not found. Skipping fold {fold}.")
            continue

        df = pd.read_csv(pred_path)
        df["fold"] = fold
        all_dfs.append(df)

    # 結合
    oof_df = pd.concat(all_dfs, ignore_index=True)

    # ソート
    oof_df = oof_df.sort_values("sample_id").reset_index(drop=True)

    # 保存
    output_path = Path(config.training.output_dir) / "oof_predictions.csv"
    oof_df.to_csv(output_path, index=False)

    print(f"✓ OOF predictions saved to: {output_path}")
    print(f"  Total samples: {len(oof_df)}")

    # OOF全体のメトリクスを計算
    predictions = oof_df[
        ["pred_clover", "pred_dead", "pred_green", "pred_gdm", "pred_total"]
    ].values
    targets = oof_df[
        ["target_clover", "target_dead", "target_green", "target_gdm", "target_total"]
    ].values

    metrics = compute_weighted_r2(predictions, targets)
    print_metrics(metrics, prefix="OOF")

    # メトリクスを保存
    metrics_path = Path(config.training.output_dir) / "oof_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ OOF metrics saved to: {metrics_path}")
    print(f"{'='*80}\n")

    return oof_df, metrics


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Evaluation for exp100")
    parser.add_argument(
        "--data_type",
        type=str,
        default="valid",
        choices=["train", "valid"],
        help="Data type",
    )
    parser.add_argument(
        "--create_oof", action="store_true", help="Create OOF predictions"
    )
    args = parser.parse_args()

    # 設定読み込み
    config = OmegaConf.load("config/exp100.yaml")

    # 各foldを評価
    evaluate_all_folds(config, data_type=args.data_type)

    # OOF予測を作成
    if args.create_oof and args.data_type == "valid":
        create_oof_predictions(config)


if __name__ == "__main__":
    main()
