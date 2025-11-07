import os
import sys
from pathlib import Path
from typing import Callable, Dict
import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from transformers import (
    Trainer,
    TrainingArguments,
    AutoProcessor,
    EarlyStoppingCallback,
)
import wandb

# プロジェクトルートをPATHに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.exp100.models import Qwen3VLForRegression
from src.exp100.dataset import Qwen3VLDataset, DataCollatorForQwen3VL
from src.exp100.utils import set_seed, compute_weighted_r2, print_metrics


class RegressionTrainer(Trainer):
    """
    回帰タスク用のカスタムTrainer
    生成テキストから数値を抽出して評価
    """

    def __init__(
        self,
        *args,
        metric_weights: np.ndarray | None = None,
        parse_fn: Callable[[str], Dict[str, float]] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metric_weights = metric_weights or np.array([0.1, 0.1, 0.1, 0.2, 0.5])
        self.parse_fn = parse_fn or Qwen3VLForRegression.parse_predictions

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        損失を計算
        """
        # メタデータを削除
        targets = inputs.pop("targets", None)
        sample_ids = inputs.pop("sample_ids", None)

        # 通常の言語モデリング損失を計算
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        予測ステップ（評価時）
        """
        # ターゲットとメタデータを取得
        targets = inputs.pop("targets")
        sample_ids = inputs.pop("sample_ids", None)

        # 損失計算用のラベルを取得
        labels = inputs.get("labels")

        # 損失を計算
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss

        # 生成して予測値を取得
        if not prediction_loss_only:
            # 生成用の入力を準備（labelsを除外）
            gen_inputs = {
                k: v
                for k, v in inputs.items()
                if k not in ["labels"] and v is not None
            }

            # 生成
            generated_ids = model.generate(**gen_inputs)

            # デコード
            predictions = []
            for ids in generated_ids:
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                # テキストから数値を抽出
                parsed = self.parse_fn(text)
                predictions.append(
                    [
                        parsed["clover"],
                        parsed["dead"],
                        parsed["green"],
                        parsed["gdm"],
                        parsed["total"],
                    ]
                )

            predictions = torch.tensor(
                predictions, dtype=torch.float32, device=targets.device
            )
        else:
            predictions = None

        return (loss, predictions, targets)


def compute_metrics(eval_pred):
    """
    評価メトリクスを計算

    Args:
        eval_pred: (predictions, labels)のタプル

    Returns:
        メトリクス辞書
    """
    predictions, labels = eval_pred

    # numpy配列に変換
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # 重み付きR²を計算
    metrics = compute_weighted_r2(labels, predictions)

    return metrics


def train_fold(config, fold: int):
    """
    1つのfoldをトレーニング

    Args:
        config: 設定
        fold: fold番号
    """
    print(f"\n{'='*80}")
    print(f"Training Fold {fold}")
    print(f"{'='*80}\n")

    # シード設定
    set_seed(config.experiment.seed)

    # データ読み込み
    data_path = (
        Path(config.dataset.output_dir) / "exp100" / "preprocessed_train.csv"
    )
    df = pd.read_csv(data_path)

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)

    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")

    # プロセッサ読み込み
    processor = AutoProcessor.from_pretrained(
        config.model.name, trust_remote_code=False
    )

    # データセット作成
    train_dataset = Qwen3VLDataset(
        train_df,
        processor,
        image_dir=Path(config.dataset.input_dir) / "train",
        prompt_template=config.prompt.template,
        max_length=config.prompt.max_length,
        is_training=True,
    )

    valid_dataset = Qwen3VLDataset(
        valid_df,
        processor,
        image_dir=Path(config.dataset.input_dir) / "train",
        prompt_template=config.prompt.template,
        max_length=config.prompt.max_length,
        is_training=False,
    )

    # データコレーター
    data_collator = DataCollatorForQwen3VL(processor)

    # モデル作成
    print("\nLoading model...")
    model = Qwen3VLForRegression(
        model_name=config.model.name,
        use_lora=config.model.use_lora,
        lora_config=dict(config.lora),
        load_in_4bit=config.model.load_in_4bit,
        load_in_8bit=config.model.get("load_in_8bit", False),
        torch_dtype=getattr(torch, config.model.torch_dtype),
    )

    # 出力ディレクトリ
    output_dir = Path(config.training.output_dir) / f"fold{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        weight_decay=config.training.weight_decay,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        adam_epsilon=config.training.adam_epsilon,
        max_grad_norm=config.training.max_grad_norm,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_pin_memory=config.training.dataloader_pin_memory,
        evaluation_strategy=config.training.evaluation_strategy,
        save_strategy=config.training.save_strategy,
        logging_steps=config.training.logging_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        remove_unused_columns=config.training.remove_unused_columns,
        report_to=config.training.report_to,
        run_name=config.training.get("run_name") or f"exp100_fold{fold}",
        seed=config.training.seed,
        gradient_checkpointing=config.training.get("gradient_checkpointing", False),
    )

    # Wandb初期化
    if config.training.report_to == "wandb":
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.get("entity"),
            name=f"exp100_fold{fold}",
            tags=config.wandb.tags,
            config=OmegaConf.to_container(config, resolve=True),
        )

    # Trainer
    metric_weights = np.array(list(config.metric_weights.values()))

    trainer = RegressionTrainer(
        model=model.model,  # PEFTモデルまたはベースモデル
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.training.get(
                    "early_stopping_patience", 3
                )
            )
        ],
        metric_weights=metric_weights,
    )

    # トレーニング
    print("\nStarting training...")
    trainer.train()

    # 最良モデルを保存
    print("\nSaving best model...")
    best_model_dir = output_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    processor.save_pretrained(str(best_model_dir))

    # 評価
    print("\nEvaluating on validation set...")
    metrics = trainer.evaluate()
    print_metrics(metrics, prefix=f"Fold {fold} Validation")

    # メトリクスを保存
    import json

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Wandb終了
    if config.training.report_to == "wandb":
        wandb.finish()

    print(f"\n✓ Fold {fold} training completed!")
    print(f"  Best model saved to: {best_model_dir}")
    print(f"  Metrics saved to: {output_dir / 'metrics.json'}")

    return metrics


def main():
    """メイン関数"""
    # 設定読み込み
    config = OmegaConf.load("config/exp100.yaml")

    print(f"\n{'='*80}")
    print(f"Experiment: {config.experiment.name}")
    print(f"Description: {config.experiment.description}")
    print(f"{'='*80}\n")

    # 各foldでトレーニング
    all_metrics = {}
    for fold in range(config.dataset.n_folds):
        metrics = train_fold(config, fold)
        all_metrics[f"fold{fold}"] = metrics

    # 全体の統計を計算
    print(f"\n{'='*80}")
    print("Overall Results")
    print(f"{'='*80}\n")

    metric_names = [
        "weighted_r2",
        "r2_clover",
        "r2_dead",
        "r2_green",
        "r2_gdm",
        "r2_total",
    ]

    for metric_name in metric_names:
        values = [
            all_metrics[f"fold{fold}"][f"eval_{metric_name}"]
            for fold in range(config.dataset.n_folds)
        ]
        mean_value = np.mean(values)
        std_value = np.std(values)
        print(f"{metric_name}: {mean_value:.4f} ± {std_value:.4f}")

    print(f"\n{'='*80}")
    print("Training completed successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
