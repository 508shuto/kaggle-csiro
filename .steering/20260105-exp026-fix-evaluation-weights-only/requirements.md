# Requirements

## 目的
PyTorch 2.6のweights_onlyデフォルト変更に対応し、evaluation.pyのチェックポイント読み込みエラーを解消する

## 変更/追加する機能
- `CSIROModule.load_from_checkpoint()`にweights_only=Falseを明示的に指定

## 制約・前提条件
- チェックポイントは自身のコードで生成したものなので信頼できる
- PyTorch 2.6以降の環境で動作すること
- 既存の`.steering/20260104-exp026-fix-torch-load/`ではtrain.pyのみ対応し、evaluation.pyは「PyTorch Lightningを使っているため影響なし」と想定していたが、実際には影響を受けた
