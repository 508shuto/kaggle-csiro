# Requirements

## 目的
exp025のトレーニング中にRMSEとMAEをロギングし、モデルの診断情報を充実させる。

## 変更/追加する機能
- バリデーション時にRMSE（Root Mean Squared Error）を計算・ログ
  - 全体のRMSE（全5クラス統合）
  - クラスごとのRMSE（各ターゲットクラス別）
- バリデーション時にMAE（Mean Absolute Error）を計算・ログ
  - 全体のMAE（全5クラス統合）
  - クラスごとのMAE（各ターゲットクラス別）

## 制約・前提条件
- 既存のメトリクス（R2スコア、WeightedR2スコア）には影響を与えない
- torchmetricsのMeanSquaredError、MeanAbsoluteErrorを全体メトリクスに使用
- クラスごとのRMSE/MAEは既存のR2スコアパターンに合わせて手動計算
