# Requirements

## 目的
exp026ベースで、CV間のばらつきを軽減するためStratifiedKFold（State + target_bin層化）を導入する。

## 変更/追加する機能
- exp026をベースにexp033を作成
- fold分割をStratifiedKFold化

## 制約・前提条件
- 日付Groupingは不要（リークより均一化を優先）
- State + target_bin（3分位）で層化
- seed=42を使用
