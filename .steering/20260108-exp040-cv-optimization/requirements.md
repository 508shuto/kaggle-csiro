# Requirements

## 目的

CV/LB相関を改善するため、CV戦略を最適化する。

## 変更/追加する機能

### CV戦略の変更

| 項目 | 現行 (exp039) | 変更後 (exp040) |
|------|--------------|-----------------|
| Fold数 | 5 | **3** |
| 層化 | species | **state** |
| グループ | `sampling_date` | `sampling_date + "_" + state` |
| Seed | 42 | **1129** |

### seed 1129の特性

```
サンプル数: [126, 119, 112]  (Range=14)

State分布:
  NSW: [21, 26, 28]
  Tas: [53, 47, 38]  (Std=7.5) ← 最も均等
  Vic: [40, 34, 38]  (Std=3.1) ← 最も均等
  WA:  [12, 12, 8]

重要Species:
  Fescue:  [11, 9, 8]  ← 全foldに含有
  Lucerne: [10, 7, 5]  ← 全foldに含有

ターゲット平均:
  Total:  [42.1, 46.6, 47.6]  (Std=2.96)
```

### 変更理由

1. **Tas/Vic分布が最も均等**: Tas Std=7.5, Vic Std=3.1
2. **Fescue/Lucerne全foldに含有**: 高NDVI/Height種の学習が安定
3. **全foldに全State含有**: 現行のState欠落問題を解消

## 制約・前提条件

1. WAは3グループのみ → 5-foldでは全foldにWAを含めることが不可能
2. 3-foldへの変更により学習データが減少（286枚→238枚/fold）
