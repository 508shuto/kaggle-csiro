# Design

## アプローチ
A案: Per-target Inverse Weighting

### 概要
各ターゲットのGT値に基づき、逆数で重み付けする。
高GT値サンプルの勾配支配を抑制し、低〜中GT値サンプルの学習を強化。

### 重み計算式
```python
medians = torch.tensor([1.42, 7.98, 20.80, 27.11, 40.30])
sample_w = 1.0 / (1.0 + targets / medians)  # (B, 5)
sample_w = torch.clamp(sample_w, 0.3, 2.0)
sample_w = sample_w / sample_w.mean()  # normalize to mean 1
```

### 効果の例（total_target, median=40.30）
| GT値 | 重み（正規化前） | 効果 |
|------|-----------------|------|
| 10 | 0.80 | 強調 |
| 40 | 0.50 | 中立 |
| 100 | 0.29 | 抑制 |

### なぜA案か
- D案（Quantile逆頻度）: サンプル数は均等→重み~1.0で効果なし
- A案（GT逆数）: 高GT値の絶対誤差支配を直接抑制

## 変更コンポーネント

### 変更ファイル
| ファイル | 変更内容 |
|---------|---------|
| `src/exp044/loss.py` | `InverseWeightedSmoothL1Loss` 追加 |
| `src/exp044/utils.py` | `get_loss_fn` 修正 |
| `config/exp044.yaml` | loss名変更 |

## 影響範囲
- 学習ループのみ（推論は変更なし）
- 他のハイパーパラメータに影響なし

## リスクと許容
- **メトリック不整合**: total_targetの高GTを抑制するが、コンペメトリックではtotalが50%の重み
- **対応**: リスクを許容し、CV結果で判断する
- **撤退基準**: Weighted R²がexp043より0.02以上悪化した場合は再検討
