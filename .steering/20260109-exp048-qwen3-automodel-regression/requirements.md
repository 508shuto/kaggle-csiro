# Requirements

## 目的
Qwen3-VL-2BをAutoModelから使用して特徴量抽出し、回帰を行う。exp045の簡略化版。

## 変更/追加する機能
- AutoModel.from_pretrainedでQwen3-VL-2Bをロード
- 視覚エンコーダーの出力から特徴量抽出
- 回帰ヘッドで5ターゲット予測

## 制約・前提条件
- モデル: `Qwen/Qwen3-VL-2B-Instruct`
- AutoModelベースの実装
- Encoder凍結、Headのみ学習
- exp045と同様のCV戦略（3-fold）
