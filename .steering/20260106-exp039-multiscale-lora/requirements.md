# Requirements

## 目的
exp035 (Multiscale Late Fusion) と exp038 (LoRA) を組み合わせ、両アプローチのメリットを統合した実験を行う。

## 変更/追加する機能

### 1. Multiscale Late Fusion + LoRA モデル
- HF DINOv3 Large バックボーンにLoRAを適用
- Global branch (1024→512 resize) + Tile branch (2x2分割) のLate Fusion
- 同一のLoRA適用バックボーンを両ブランチで共有

### 2. 設定調整
- 画像サイズ: 1024x1024
- バッチサイズ: 4 (gradient accumulation: 4で実効16)
- LoRA: r=8, alpha=16, layers 18-23
- Mixup: 無効（タイル分割と非互換）

## 制約・前提条件

### 技術的制約
- メモリ使用量: 5回のforward pass (Global 1 + Tile 4)
- DDP: `find_unused_parameters=True`必須
- HF DINOv3はgated repo（HFログイン必要）

### ベースライン
- exp035: CV 0.6948, LB 0.65 (Multiscale, frozen backbone)
- exp038: CV 0.7073, LB TBD (LoRA, single scale)

### 目標
- CV 0.72以上を達成
- CV-LBギャップを0.05以内に維持
