# Design

## アプローチ
1. **Feature Extraction方式**: Qwen3-VLの視覚エンコーダ（ViT）を特徴抽出器として使用
2. **Frozen Backbone**: VLMパラメータをfreezeし、回帰ヘッドのみ学習
3. **回帰ヘッド**: 抽出した視覚特徴から5つのターゲットを予測

## モデルアーキテクチャ
```
Input Image (B, 3, H, W)
    ↓
Qwen3-VL-2B Vision Encoder (frozen)
    ↓
Visual Features (B, seq_len, hidden_dim)
    ↓
Global Average Pooling
    ↓
Feature Vector (B, hidden_dim)
    ↓
Regression Head: Linear(hidden_dim→256) + ReLU + Dropout + Linear(256→5)
    ↓
Output (B, 5) [Clover, Dead, Green, GDM, Total]
```

## 変更コンポーネント
| ファイル | 変更内容 |
|---------|---------|
| config/exp027.yaml | Qwen3-VL用設定（モデル名、画像サイズ等） |
| models.py | Qwen3VLRegressionModel クラス（VLM特徴抽出+回帰ヘッド） |
| dataset.py | VLM用入力形式（ProcessorでのTokenize不要、画像のみ） |
| utils.py | VLM用transform（Qwen3-VLのデフォルト正規化） |
| その他 | exp025から流用（train, eval, inference, loss, metrics） |

## 影響範囲
- 新規実験としてexp027を作成
- 既存コードへの影響なし
