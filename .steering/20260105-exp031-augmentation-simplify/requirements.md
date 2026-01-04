# Requirements

## 目的
Augmentationを簡素化し、過度なAugmentationによる精度低下を防ぐ

## 背景
チームメンバーの設定:
- 採用: HorizontalFlip, VerticalFlip, RandomBrightnessContrast のみ
- 不採用: RandomShadow, Mixup など（精度向上なし）

現状のexp028/029/030設定:
- 多数のAugmentation（elastic, gamma, gaussian_noise, blur, mixup等）
- これらが過学習防止より害が大きい可能性

## 変更/追加する機能
1. Augmentationを3種のみに絞る
   - HorizontalFlip (p=0.5)
   - VerticalFlip (p=0.5)
   - RandomBrightnessContrast (p=0.5)
2. Mixup無効化
3. その他のAugmentation無効化

## 制約・前提条件
- exp030のコードをベースにする
- Tile処理、MHA統合は維持
- 変更はconfig/augmentationセクションのみ

## 成功基準
- exp030より向上すればOK
