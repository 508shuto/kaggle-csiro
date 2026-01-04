# Design

## アプローチ
torch.load()呼び出しにweights_only=Falseを追加する

## 変更コンポーネント
- src/exp026/train.py

## 影響範囲
- train.pyのチェックポイントリネーム処理のみ
- inference.py、evaluation.pyはPyTorch Lightningの`load_from_checkpoint()`を使用しているため影響なし
