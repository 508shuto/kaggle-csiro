# Requirements

## 目的
exp027の推論速度を改善する（現状exp026の3-5倍遅い）

## 変更/追加する機能
- Dataset側で画像前処理（AutoImageProcessor）を実行
- models.py forward()からprocessor呼び出しを削除
- tensor入力に対応したcollate_fn

## 制約・前提条件
- Qwen3VLImageProcessorの出力形式（pixel_values, grid_thw）を正確に再現
- 既存チェックポイントとの重み互換性を維持
- 学習・推論両方で動作すること
