# Design

## アプローチ
`CSIROModule.load_from_checkpoint()`呼び出しにweights_only=Falseを追加する

## 変更コンポーネント
- src/exp028/evaluation.py
- src/exp028/inference.py

## 影響範囲
- evaluation.py, inference.pyのチェックポイント読み込み処理
- PyTorch Lightningの`load_from_checkpoint()`は内部でtorch.loadを使用しており、PyTorch 2.6のweights_onlyデフォルト変更の影響を受ける
