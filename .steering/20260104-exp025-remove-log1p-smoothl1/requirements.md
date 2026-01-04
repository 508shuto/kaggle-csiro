# Requirements

## 目的
EXP024をベースに、log1p変換を完全に廃止し、main lossをSmoothL1Lossに変更したEXP025を実装する。

## 変更/追加する機能
- **log1p変換の完全廃止**: main target（5クラス）とaux target（2クラス）の両方でlog1p変換を削除し、rawスケールで学習・評価・推論を行う
- **main lossの変更**: WeightedMSELossからWeightedSmoothL1Lossに変更（重みは従来通り `[0.1, 0.1, 0.1, 0.2, 0.5]`）
- **aux lossは変更なし**: MSEのまま維持

## 制約・前提条件
- EXP024の構造を維持（モデルアーキテクチャ、データセット分割、augmentation設定など）
- 物理制約（GDM = Clover + Green, Total = Clover + Dead + Green）は維持
- 評価指標（WeightedR2Score）は変更なし

