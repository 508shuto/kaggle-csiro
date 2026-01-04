# Requirements

## 目的
PyTorch 2.6のweights_onlyデフォルト変更に対応し、チェックポイント読み込みエラーを解消する

## 変更/追加する機能
- torch.loadにweights_only=Falseを明示的に指定

## 制約・前提条件
- チェックポイントは自身のコードで生成したものなので信頼できる
- PyTorch 2.6以降の環境で動作すること
