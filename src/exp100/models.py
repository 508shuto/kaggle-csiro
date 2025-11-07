import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import re
from typing import Dict, Optional
from pathlib import Path


class Qwen3VLForRegression(nn.Module):
    """
    Qwen3-VLベースの回帰モデル
    transformers.Qwen3VLForConditionalGenerationを使用
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        torch_dtype=torch.bfloat16,
    ):
        """
        初期化

        Args:
            model_name: Hugging Faceのモデル名
            use_lora: LoRAを使用するかどうか
            lora_config: LoRA設定（辞書形式）
            load_in_4bit: 4bit量子化を使用するかどうか
            load_in_8bit: 8bit量子化を使用するかどうか
            torch_dtype: モデルのdtype
        """
        super().__init__()

        print(f"\n{'='*60}")
        print(f"Loading model: {model_name}")
        print(f"  use_lora: {use_lora}")
        print(f"  load_in_4bit: {load_in_4bit}")
        print(f"  load_in_8bit: {load_in_8bit}")
        print(f"  torch_dtype: {torch_dtype}")
        print(f"{'='*60}\n")

        # Qwen3VLForConditionalGeneration読み込み
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            device_map="auto",
            trust_remote_code=False,  # Qwen3-VLは公式サポート
        )

        # プロセッサ読み込み
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=False
        )

        # LoRA適用
        if use_lora:
            print("Applying LoRA configuration...")
            if load_in_4bit or load_in_8bit:
                self.model = prepare_model_for_kbit_training(self.model)

            if lora_config is None:
                lora_config = {
                    "r": 64,
                    "lora_alpha": 16,
                    "target_modules": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                }

            peft_config = LoraConfig(**lora_config)
            self.model = get_peft_model(self.model, peft_config)
            print("\nTrainable parameters:")
            self.model.print_trainable_parameters()
            print()

        self.model_name = model_name
        self.use_lora = use_lora

    def forward(self, **inputs):
        """
        順伝播

        Args:
            inputs: processor出力（pixel_values, input_ids, attention_mask, labels等）

        Returns:
            モデル出力（loss含む）
        """
        return self.model(**inputs)

    def generate(self, **inputs):
        """
        生成（推論用）

        Args:
            inputs: processor出力

        Returns:
            生成されたtoken ids
        """
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        return generated_ids

    @staticmethod
    def parse_predictions(text: str) -> Dict[str, float]:
        """
        生成テキストから数値を抽出

        Args:
            text: モデルが生成したテキスト

        Returns:
            dict: {
                'clover': float,
                'dead': float,
                'green': float,
                'gdm': float,
                'total': float
            }
        """
        patterns = {
            "clover": r"(?:Dry\s+)?Clover[^\d]*([\d.]+)",
            "dead": r"(?:Dry\s+)?Dead[^\d]*([\d.]+)",
            "green": r"(?:Dry\s+)?Green[^\d]*([\d.]+)",
            "gdm": r"GDM[^\d]*([\d.]+)",
            "total": r"(?:Dry\s+)?Total[^\d]*([\d.]+)",
        }

        results = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    results[key] = float(match.group(1))
                except ValueError:
                    results[key] = 0.0
            else:
                results[key] = 0.0

        return results

    def save_pretrained(self, save_directory: str):
        """
        モデルとプロセッサを保存

        Args:
            save_directory: 保存先ディレクトリ
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        if self.use_lora:
            # LoRAの場合はPEFTモデルとして保存
            self.model.save_pretrained(save_directory)
        else:
            # フルモデルの場合
            self.model.save_pretrained(save_directory)

        self.processor.save_pretrained(save_directory)
        print(f"✓ Model and processor saved to: {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        base_model_name: Optional[str] = None,
        use_lora: bool = True,
        **kwargs,
    ):
        """
        保存されたモデルをロード

        Args:
            model_path: 保存されたモデルのパス
            base_model_name: LoRAの場合のベースモデル名
            use_lora: LoRAモデルかどうか
            **kwargs: その他のパラメータ

        Returns:
            Qwen3VLForRegressionインスタンス
        """
        print(f"\nLoading model from: {model_path}")

        instance = cls.__new__(cls)
        super(Qwen3VLForRegression, instance).__init__()

        if use_lora:
            # LoRAモデルのロード
            if base_model_name is None:
                # config.jsonから読み取る
                import json

                config_path = Path(model_path) / "adapter_config.json"
                if config_path.exists():
                    with open(config_path) as f:
                        adapter_config = json.load(f)
                        base_model_name = adapter_config.get(
                            "base_model_name_or_path", "Qwen/Qwen3-VL-2B-Instruct"
                        )
                else:
                    base_model_name = "Qwen/Qwen3-VL-2B-Instruct"

            print(f"  Base model: {base_model_name}")

            # ベースモデルをロード
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_name, device_map="auto", **kwargs
            )

            # PEFTアダプターをロード
            instance.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # フルモデルのロード
            instance.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, device_map="auto", **kwargs
            )

        # プロセッサをロード
        instance.processor = AutoProcessor.from_pretrained(model_path)

        instance.model_name = str(model_path)
        instance.use_lora = use_lora

        print(f"✓ Model loaded successfully\n")

        return instance
