# Design

## アプローチ
1. **Qwen3-VL-Embedding-2B使用**: 公式のEmbeddingモデルで画像から高品質な特徴量を抽出
2. **Frozen Encoder**: Embeddingモデルは凍結し、回帰ヘッドのみ学習
3. **回帰ヘッド**: 2048次元embeddingから3変数を予測し、物理制約で残り2変数を導出

## モデルアーキテクチャ
```
Qwen3-VL-Embedding-2B (frozen)
    │
    ▼
Embedding (batch, 2048)
    │
    ▼
Regression Head (MLP: 2048 → 512 → 3)
    │
    ▼
pred_3: [Clover, Dead, Green]
    │
    ▼ (物理制約)
pred_5: [Clover, Dead, Green, GDM, Total]
```

## Qwen3-VL-Embedding-2B 仕様
| 項目 | 値 |
|------|-----|
| モデル | Qwen/Qwen3-VL-Embedding-2B |
| Embedding次元 | 2048 |
| 特徴抽出方式 | DeepStack（レイヤー[8,16,24]融合）+ EOSトークン |
| Matryoshka対応 | あり（次元削減可能） |
| 入力 | 画像（PIL/path/URL） |

## 変更コンポーネント
| ファイル | 変更内容 |
|---------|---------|
| models.py | Qwen3VLEmbeddingRegressor実装（Embedder + Head） |
| dataset.py | Qwen3VLEmbedder用の入力形式に変更 |
| config/exp045.yaml | Qwen3-VL-Embedding-2B用設定 |
| lightning_module.py | Headのみ学習するオプティマイザ設定 |
| requirements.txt | qwen3-vl-embedding依存追加 |

## 実装詳細

### 特徴量抽出
```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

# Embedding抽出（公式リポジトリ参考）
class Qwen3VLEmbeddingRegressor(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-VL-Embedding-2B"):
        super().__init__()
        # Embeddingモデルをロード
        self.embedder = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )
        self.embedder.eval()
        for p in self.embedder.parameters():
            p.requires_grad = False

        # 回帰ヘッド
        self.head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),  # [Clover, Dead, Green]
        )

    def forward(self, inputs):
        with torch.no_grad():
            embeddings = self.embedder.get_embedding(inputs)  # (B, 2048)
        pred_3 = self.head(embeddings)
        pred_3 = torch.clamp(pred_3, min=0.0)

        # 物理制約
        clover, dead, green = pred_3[:, 0], pred_3[:, 1], pred_3[:, 2]
        gdm = clover + green
        total = clover + dead + green
        pred_5 = torch.stack([clover, dead, green, gdm, total], dim=1)
        return pred_5
```

### 学習戦略
- Embedding: **完全凍結**（勾配計算なし）
- Head: **学習**（lr=1e-3）
- バッチサイズ: 32（embedding計算は軽量）

## 代替案
性能が不十分な場合：
- **Option B**: Embedding次元をMatryoshkaで512/1024に削減
- **Option C**: LoRA適用（Embeddingモデル部分）

## 影響範囲
- 新規実験のみ（既存実験への影響なし）
- GPUメモリ: 約5GB（FP16/BF16）
- 依存: `qwen3-vl-embedding`リポジトリまたはtransformers最新版

## 参考
- [Qwen3-VL-Embedding GitHub](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [DeepStack論文](https://arxiv.org/abs/2406.04334)
