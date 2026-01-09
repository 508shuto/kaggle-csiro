# Design

## アプローチ
1. **視覚エンコーダーのみ使用**: Qwen3-VL-2Bから視覚エンコーダー部分のみを抽出
2. **特徴量抽出**: DeepStack方式の多層ViT特徴量を取得
3. **回帰ヘッド**: 抽出された特徴量から3変数（Clover, Dead, Green）を予測し、物理制約で残り2変数を導出

## モデルアーキテクチャ
```
Qwen3-VL-2B Vision Encoder
    │
    ▼
Visual Features (pooled / CLS token)
    │
    ▼
Regression Head (MLP)
    │
    ▼
pred_3: [Clover, Dead, Green]
    │
    ▼ (物理制約)
pred_5: [Clover, Dead, Green, GDM, Total]
```

## 変更コンポーネント
| ファイル | 変更内容 |
|---------|---------|
| models.py | Qwen3VLFeatureExtractor + RegressionHeadの実装 |
| dataset.py | Qwen3用のプロセッサ適用 |
| config/exp045.yaml | Qwen3-VL-2B用設定 |
| lightning_module.py | オプティマイザ設定（フルファインチューン or frozen encoder） |

## 実装詳細

### 特徴量抽出方法
```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
# 視覚エンコーダー部分: model.visual
# LLM部分は不要: model.model (language model)

# 画像入力 → visual features取得
visual_features = model.visual(pixel_values, grid_thw)
# DeepStack特徴量: (batch, seq_len, hidden_dim)
```

### 学習戦略
- **Option A**: Vision encoder frozen + Head学習
- **Option B**: Vision encoder LoRA + Head学習（exp043/044方式）

初期実験はOption Aで実施し、効果を確認後にOption Bを検討。

## 影響範囲
- 新規実験のみ（既存実験への影響なし）
- GPUメモリ: Qwen3-VL-2Bは約5GB（FP16）なのでexp044と同等
