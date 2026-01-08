# Design

## アプローチ
exp038のスクリプト群をexp043にコピーし、create_dataset.pyのみexp040からコピー。

## 構成

| ファイル | ソース |
|---------|--------|
| models.py | exp038 |
| lightning_module.py | exp038 |
| train.py | exp038 |
| dataset.py | exp038 |
| loss.py | exp038 |
| metrics.py | exp038 |
| evaluation.py | exp038 |
| inference.py | exp038 |
| utils.py | exp038 |
| create_dataset.py | exp040 |
| config/exp043.yaml | exp038 + exp040 |

## 差分
- n_folds: 5 → 3
- seed: 42 → 1129
- devices: 2 → 1
- strategy: ddp_find_unused_parameters_true → auto
