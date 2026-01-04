# Requirements

## 目的

exp027の学習スクリプト実行時に発生する`AttributeError: 'Qwen3VLModel' object has no attribute 'vision_model'`を解決し、Qwen3-VLベースの回帰モデルを正常に動作させる。

## 変更/追加する機能

1. **モデル属性アクセスの修正**
   - `self.vlm.model.vision_model` → `self.vlm.model.visual`

2. **hidden_dimの修正**
   - Qwen3-VL-2Bの`out_hidden_size`（2048）に合わせる

3. **AutoImageProcessorの導入**
   - ピクセル前処理を公式Processorで行う
   - `Qwen3VLImageProcessor`は存在しないため`AutoImageProcessor`を使用

4. **データパイプラインの修正**
   - DatasetがPIL画像を返すように変更（ToTensor/Normalize削除）
   - カスタムcollate_fnの追加

5. **Mixupの一時無効化**
   - PIL画像では動作しないため

## 制約・前提条件

- Qwen3-VL-2B-Instructモデルを使用
- transformersライブラリのQwen3VL実装に依存
- 学習速度: forward内でProcessor実行のためオーバーヘッドあり
- MPS/CUDAでの動作を想定

## 実施結果

- 2026-01-04: 全タスク完了、学習開始を確認
