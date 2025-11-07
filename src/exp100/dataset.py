import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Dict, Any, Optional


class Qwen3VLDataset(Dataset):
    """
    Qwen3-VL用のVision-Languageデータセット

    Args:
        df: データフレーム
        processor: Qwen3VLプロセッサ
        image_dir: 画像ディレクトリ
        prompt_template: プロンプトテンプレート名
        max_length: 最大トークン長
        is_training: トレーニングモードかどうか
    """

    def __init__(
        self,
        df: pd.DataFrame,
        processor: AutoProcessor,
        image_dir: Path,
        prompt_template: str = "default",
        max_length: int = 512,
        is_training: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.image_dir = Path(image_dir)
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.is_training = is_training

    def _create_conversation(self, row: pd.Series) -> list:
        """
        Qwen3-VL形式の会話を作成

        Args:
            row: データフレームの1行

        Returns:
            list: Qwen3-VL形式の会話
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "path/to/image.jpg"},
                            {"type": "text", "text": "プロンプト"}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": "応答"  # トレーニング時のみ
                    }
                ]
        """
        # 画像パス
        image_path = str(self.image_dir / row["image_path"])

        # プロンプト作成
        if self.prompt_template == "default":
            prompt_text = (
                "この草地画像を分析し、以下の指標をグラム単位で予測してください：\n"
                f"- 種: {row['species']}\n"
                f"- 事前NDVI: {row['pre_gshh_ndvi']:.3f}\n"
                f"- 平均高さ: {row['height_ave_cm']:.1f}cm\n\n"
                "以下の形式で予測値を出力してください：\n"
                "Dry Clover: [値]g\n"
                "Dry Dead: [値]g\n"
                "Dry Green: [値]g\n"
                "GDM: [値]g\n"
                "Dry Total: [値]g"
            )
        elif self.prompt_template == "simple":
            prompt_text = (
                "Predict grassland biomass values for this image:\n"
                f"Species: {row['species']}, NDVI: {row['pre_gshh_ndvi']:.3f}, "
                f"Height: {row['height_ave_cm']:.1f}cm\n\n"
                "Output format:\n"
                "Dry Clover: [value]g\n"
                "Dry Dead: [value]g\n"
                "Dry Green: [value]g\n"
                "GDM: [value]g\n"
                "Dry Total: [value]g"
            )
        elif self.prompt_template == "concise":
            prompt_text = (
                f"Species: {row['species']}, NDVI: {row['pre_gshh_ndvi']:.3f}, "
                f"Height: {row['height_ave_cm']:.1f}cm\n"
                "Predict: Dry Clover, Dry Dead, Dry Green, GDM, Dry Total (g)"
            )
        else:
            raise ValueError(f"Unknown prompt template: {self.prompt_template}")

        # 会話構築
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # トレーニング時は応答を追加
        if self.is_training:
            response = (
                f"Dry Clover: {row['clover_target']:.2f}g\n"
                f"Dry Dead: {row['dead_target']:.2f}g\n"
                f"Dry Green: {row['green_target']:.2f}g\n"
                f"GDM: {row['gdm_target']:.2f}g\n"
                f"Dry Total: {row['total_target']:.2f}g"
            )
            conversation.append({"role": "assistant", "content": response})

        return conversation

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # 会話作成
        conversation = self._create_conversation(row)

        # プロセッサで処理（apply_chat_templateを使用）
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=not self.is_training
        )

        # 画像とビデオ情報を処理
        image_inputs, video_inputs = process_vision_info(conversation)

        # プロセッサで画像とテキストを処理
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # バッチ次元を削除
        inputs = {
            k: v.squeeze(0) if v is not None else None for k, v in inputs.items()
        }

        # Noneの値を削除
        inputs = {k: v for k, v in inputs.items() if v is not None}

        # ラベル作成（トレーニング時）
        if self.is_training:
            labels = inputs["input_ids"].clone()
            # パディングトークンは-100に設定（損失計算で無視される）
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            inputs["labels"] = labels

        # 評価用のターゲット値
        inputs["targets"] = torch.tensor(
            [
                row["clover_target"],
                row["dead_target"],
                row["green_target"],
                row["gdm_target"],
                row["total_target"],
            ],
            dtype=torch.float32,
        )

        # メタデータ
        inputs["sample_id"] = row["sample_id"]

        return inputs


class DataCollatorForQwen3VL:
    """
    Qwen3-VL用のデータコレーター
    異なる長さの入力をバッチ化
    """

    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, batch: list) -> Dict[str, Any]:
        """
        バッチをコレート

        Args:
            batch: データセットから取得したアイテムのリスト

        Returns:
            バッチ化されたデータ
        """
        # メタデータとターゲットを分離
        targets = torch.stack([item.pop("targets") for item in batch])
        sample_ids = [item.pop("sample_id") for item in batch]

        # 残りをデフォルトでバッチ化
        keys = batch[0].keys()
        batched = {}

        for key in keys:
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                if key == "pixel_values" or key == "image_grid_thw":
                    # 画像関連のテンソルはそのままスタック
                    batched[key] = torch.stack(values)
                else:
                    # トークン系はパディングが必要
                    max_len = max(v.shape[0] for v in values)
                    padded = []
                    for v in values:
                        pad_size = max_len - v.shape[0]
                        if pad_size > 0:
                            if key == "labels":
                                # ラベルは-100でパディング
                                padded_v = torch.cat(
                                    [v, torch.full((pad_size,), -100, dtype=v.dtype)]
                                )
                            elif key == "attention_mask":
                                # attention_maskは0でパディング
                                padded_v = torch.cat(
                                    [v, torch.zeros(pad_size, dtype=v.dtype)]
                                )
                            else:
                                # その他（input_ids等）はpad_token_idでパディング
                                pad_token_id = self.processor.tokenizer.pad_token_id
                                padded_v = torch.cat(
                                    [
                                        v,
                                        torch.full(
                                            (pad_size,), pad_token_id, dtype=v.dtype
                                        ),
                                    ]
                                )
                        else:
                            padded_v = v
                        padded.append(padded_v)
                    batched[key] = torch.stack(padded)
            else:
                batched[key] = values

        batched["targets"] = targets
        batched["sample_ids"] = sample_ids

        return batched
