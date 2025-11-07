import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from peft import LoraConfig


def get_device() -> str:
    """
    利用可能なデバイスを取得

    Returns:
        デバイス名（cuda, mps, cpu）
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """
    LoRA設定を取得

    Args:
        config: 設定辞書

    Returns:
        LoraConfig
    """
    return LoraConfig(
        r=config.get("r", 64),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=config.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias=config.get("bias", "none"),
        task_type=config.get("task_type", "CAUSAL_LM"),
    )


def set_seed(seed: int):
    """
    乱数シードを設定

    Args:
        seed: シード値
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    sample_ids: list,
    output_path: Path,
):
    """
    予測結果を保存

    Args:
        predictions: 予測値 (N, 5)
        targets: 正解値 (N, 5)
        sample_ids: サンプルID
        output_path: 保存先パス
    """
    import pandas as pd

    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "pred_clover": predictions[:, 0],
            "pred_dead": predictions[:, 1],
            "pred_green": predictions[:, 2],
            "pred_gdm": predictions[:, 3],
            "pred_total": predictions[:, 4],
            "target_clover": targets[:, 0],
            "target_dead": targets[:, 1],
            "target_green": targets[:, 2],
            "target_gdm": targets[:, 3],
            "target_total": targets[:, 4],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")


def compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R²スコアを計算

    Args:
        y_true: 正解値
        y_pred: 予測値

    Returns:
        R²スコア
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r2


def compute_weighted_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray = np.array([0.1, 0.1, 0.1, 0.2, 0.5]),
) -> Dict[str, float]:
    """
    重み付きR²スコアを計算

    Args:
        y_true: 正解値 (N, 5)
        y_pred: 予測値 (N, 5)
        weights: 重み (5,)

    Returns:
        dict: 各クラスのR²と重み付き平均
    """
    target_names = ["clover", "dead", "green", "gdm", "total"]
    r2_scores = {}

    for i, name in enumerate(target_names):
        r2 = compute_r2_score(y_true[:, i], y_pred[:, i])
        r2_scores[f"r2_{name}"] = r2

    # 重み付き平均
    r2_values = np.array([r2_scores[f"r2_{name}"] for name in target_names])
    weighted_r2 = np.sum(r2_values * weights)
    r2_scores["weighted_r2"] = weighted_r2

    return r2_scores


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    メトリクスを表示

    Args:
        metrics: メトリクス辞書
        prefix: プレフィックス（train, val等）
    """
    print(f"\n{'='*60}")
    if prefix:
        print(f"{prefix.upper()} Metrics")
    else:
        print("Metrics")
    print(f"{'='*60}")

    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print(f"{'='*60}\n")
