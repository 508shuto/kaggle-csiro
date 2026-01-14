# Design

## アプローチ
1. AutoModelでQwen3-VL-2Bをロード
2. 視覚エンコーダーの出力を取得
3. Mean poolingで特徴量を集約
4. 回帰ヘッドで3変数予測 → 物理制約で5変数

## モデルアーキテクチャ
```
AutoModel (Qwen3-VL-2B) - frozen
    │
    ▼
Visual Features (mean pooling)
    │
    ▼
Regression Head (MLP)
    │
    ▼
pred_5
```

## 変更コンポーネント
| ファイル | 変更内容 |
|---------|---------|
| models.py | AutoModelベースに変更 |
| config/exp046.yaml | exp046用設定 |

## exp045との差分
- Qwen3VLForConditionalGeneration → AutoModel
- よりシンプルな実装
