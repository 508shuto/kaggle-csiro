# Requirements

## 目的
exp025のトレーニング中にRMSEとMAEをロギングし、モデルの診断情報を充実させる。

## 変更/追加する機能
- バリデーション時にRMSE（Root Mean Squared Error）を計算・ログ
- バリデーション時にMAE（Mean Absolute Error）を計算・ログ

## 制約・前提条件
- 既存のメトリクス（R2スコア、WeightedR2スコア）には影響を与えない
- torchmetricsのMeanSquaredError、MeanAbsoluteErrorを使用
