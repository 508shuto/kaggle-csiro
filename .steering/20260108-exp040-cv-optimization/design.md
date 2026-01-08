# Design

## アプローチ

exp028をベースに、CV戦略のみを変更する。

## 変更コンポーネント

### 1. create_dataset.py

```python
# 変更前（exp028: 5-fold Species層化）
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (_, val_idx) in enumerate(sgkf.split(df, df["species"], df["sampling_date"])):
    df.loc[val_idx, "fold"] = fold

# 変更後（exp040: 3-fold State層化, seed=1129）
sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1129)
groups = df["sampling_date"] + "_" + df["state"]
for fold, (_, val_idx) in enumerate(sgkf.split(df, df["state"], groups)):
    df.loc[val_idx, "fold"] = fold
```

### 2. train.py

Fold数を5→3に変更:
```bash
# 変更前
uv run python src/exp028/train.py --folds 0 1 2 3 4

# 変更後
uv run python src/exp040/train.py --folds 0 1 2
```

### 3. evaluation.py

```bash
# 変更前
uv run python src/exp028/evaluation.py --folds 0 1 2 3 4

# 変更後
uv run python src/exp040/evaluation.py --folds 0 1 2
```

### 4. config/exp040.yaml

```yaml
n_folds: 3  # 5→3に変更
```

## 影響範囲

| ファイル | 変更内容 |
|----------|----------|
| create_dataset.py | fold分割ロジック |
| train.py | デフォルトfold引数 |
| evaluation.py | デフォルトfold引数 |
| config/exp040.yaml | n_folds設定 |

## 変更なし（exp028から継承）

- モデル: vit_large_patch16_dinov3.lvd1689m
- 画像サイズ: 512x512
- Backbone: Frozen
- 学習率: 1e-3
- エポック数: 100
- EMA: decay 0.995
- Loss: WeightedSmoothL1Loss
