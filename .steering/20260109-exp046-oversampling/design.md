# Design

## アプローチ
1. create_dataset.pyでサンプリング重みを計算
2. train.pyでWeightedRandomSamplerを使用
3. DataLoaderのsamplerパラメータで適用

## 変更コンポーネント

### create_dataset.py
```python
def compute_sampling_weights(df: pd.DataFrame) -> np.ndarray:
    """サンプリング重みを計算"""
    weights = np.ones(len(df))

    # WA州: 3倍
    wa_mask = df["state"] == "WA"
    weights[wa_mask] = 3.0

    # Q1（GT≤25.3）: 2倍
    q1_threshold = df["total_target"].quantile(0.25)
    q1_mask = df["total_target"] <= q1_threshold
    weights[q1_mask] *= 2.0  # WA & Q1は6倍

    return weights

# preprocessed_train.csvに重みカラムを追加
df["sample_weight"] = compute_sampling_weights(df)
```

### train.py
```python
from torch.utils.data import WeightedRandomSampler

# Train DataLoaderでWeightedRandomSamplerを使用
train_weights = train_df["sample_weight"].values
sampler = WeightedRandomSampler(
    weights=train_weights,
    num_samples=len(train_df) * 2,  # 2倍のサンプル数
    replacement=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.trainer.train.batch_size,
    sampler=sampler,  # shuffle=Trueの代わりにsamplerを使用
    num_workers=config.trainer.train.num_workers,
)
```

### config/exp046.yaml
```yaml
dataset:
  use_oversampling: true
  oversampling:
    wa_weight: 3.0
    q1_weight: 2.0
    num_samples_multiplier: 2.0
```

## 影響範囲
- エポックあたりのサンプル数: 2倍（重複サンプルあり）
- 学習時間: 約2倍
- WA州・Q1の精度: 改善見込み

## リスク
- 過学習のリスク増加 → Early Stopping厳格化
- 高GT値サンプルの精度低下の可能性 → 重み設定の調整が必要
