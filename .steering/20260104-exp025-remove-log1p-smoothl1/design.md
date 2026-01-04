# Design

## アプローチ
EXP024をコピーしてEXP025を作成し、以下の変更を適用する：
1. dataset.py: `torch.log1p()` を削除し、raw値を返す
2. models.py: log空間での予測を廃止し、raw値を直接出力
3. loss.py: `WeightedSmoothL1Loss` を追加
4. utils.py: `mixup_batch` をraw空間で動作するように簡素化、`get_loss_fn` に `smooth_l1_loss` を追加
5. lightning_module.py: `expm1` 逆変換を削除し、raw値でloss/metricsを計算
6. evaluation.py/inference.py: `expm1` を削除し、raw値で処理

## 変更コンポーネント
- `config/exp025.yaml`: experiment.nameとloss.nameを変更
- `src/exp025/dataset.py`: log1p変換削除
- `src/exp025/models.py`: log空間予測を廃止、raw出力に変更
- `src/exp025/loss.py`: WeightedSmoothL1Lossクラス追加
- `src/exp025/utils.py`: mixup_batch簡素化、get_loss_fn拡張
- `src/exp025/lightning_module.py`: expm1削除、raw値で処理
- `src/exp025/evaluation.py`: expm1削除
- `src/exp025/inference.py`: expm1削除

## 影響範囲
- 学習・評価・推論のすべてのパイプラインでlog1p/expm1が使用されなくなる
- loss計算がSmoothL1に変更されるため、学習挙動が変わる可能性がある
- mixup処理が簡素化される（log空間への変換/逆変換が不要）

