import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse

# プロジェクトルートをPATHに追加
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.exp100.models import Qwen3VLForRegression
from src.exp100.dataset import Qwen3VLDataset
from src.exp100.utils import set_seed, save_predictions


def inference_fold(
    config,
    fold: int,
    data_type: str = "valid",
    batch_size: int = 1,
    use_tta: bool = False,
):
    """
    1つのfoldで推論を実行

    Args:
        config: 設定
        fold: fold番号
        data_type: "train", "valid", "test"
        batch_size: バッチサイズ
        use_tta: TTAを使用するかどうか

    Returns:
        predictions, targets, sample_ids
    """
    print(f"\n{'='*80}")
    print(f"Inference Fold {fold} - {data_type}")
    print(f"{'='*80}\n")

    # シード設定
    set_seed(config.experiment.seed)

    # データ読み込み
    data_path = (
        Path(config.dataset.output_dir) / "exp100" / "preprocessed_train.csv"
    )
    df = pd.read_csv(data_path)

    if data_type == "train":
        df = df[df["fold"] != fold].reset_index(drop=True)
    elif data_type == "valid":
        df = df[df["fold"] == fold].reset_index(drop=True)
    else:
        raise ValueError(f"Invalid data_type: {data_type}")

    print(f"{data_type.capitalize()} samples: {len(df)}")

    # モデルロード
    model_path = Path(config.training.output_dir) / f"fold{fold}" / "best_model"
    print(f"Loading model from: {model_path}")

    model = Qwen3VLForRegression.from_pretrained(
        str(model_path),
        base_model_name=config.model.name,
        use_lora=config.model.use_lora,
        torch_dtype=getattr(torch, config.model.torch_dtype),
    )
    model.model.eval()

    # デバイス設定
    device = next(model.model.parameters()).device

    # データセット作成
    dataset = Qwen3VLDataset(
        df,
        model.processor,
        image_dir=Path(config.dataset.input_dir) / "train",
        prompt_template=config.prompt.template,
        max_length=config.prompt.max_length,
        is_training=False,
    )

    # DataLoader
    from torch.utils.data import DataLoader
    from src.exp100.dataset import DataCollatorForQwen3VL

    data_collator = DataCollatorForQwen3VL(model.processor)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False
    )

    # 推論
    all_predictions = []
    all_targets = []
    all_sample_ids = []

    print("\nRunning inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Fold {fold}"):
            # メタデータを取得
            targets = batch.pop("targets")
            sample_ids = batch.pop("sample_ids")

            # 入力をデバイスに移動
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if k not in ["labels"]
            }

            # 生成
            generated_ids = model.generate(**inputs)

            # デコードと解析
            predictions = []
            for ids in generated_ids:
                text = model.processor.decode(ids, skip_special_tokens=True)
                parsed = model.parse_predictions(text)
                predictions.append(
                    [
                        parsed["clover"],
                        parsed["dead"],
                        parsed["green"],
                        parsed["gdm"],
                        parsed["total"],
                    ]
                )

            all_predictions.append(np.array(predictions))
            all_targets.append(targets.cpu().numpy())
            all_sample_ids.extend(sample_ids)

    # 結合
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    print(f"✓ Inference completed: {len(predictions)} samples")

    return predictions, targets, all_sample_ids


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Inference for exp100")
    parser.add_argument("--fold", type=int, default=None, help="Fold number")
    parser.add_argument(
        "--data_type",
        type=str,
        default="valid",
        choices=["train", "valid"],
        help="Data type",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--use_tta", action="store_true", help="Use TTA")
    args = parser.parse_args()

    # 設定読み込み
    config = OmegaConf.load("config/exp100.yaml")

    # fold指定
    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = range(config.dataset.n_folds)

    # 各foldで推論
    for fold in folds:
        predictions, targets, sample_ids = inference_fold(
            config,
            fold,
            data_type=args.data_type,
            batch_size=args.batch_size,
            use_tta=args.use_tta,
        )

        # 予測結果を保存
        output_dir = Path(config.training.output_dir) / f"fold{fold}"
        output_path = output_dir / f"predictions_{args.data_type}.csv"

        save_predictions(predictions, targets, sample_ids, output_path)

    print("\n✓ All inference completed!")


if __name__ == "__main__":
    main()
