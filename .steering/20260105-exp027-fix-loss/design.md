# Design

## アプローチ
1. VLM全体をfreeze → regression headのみtrainable
2. forward内clamp削除 → 勾配が正常に流れる
3. warmup適用 → 学習初期の安定性向上
4. lr増加 → 小さなヘッドに適したLR

## 変更コンポーネント
- src/exp027/models.py
- src/exp027/lightning_module.py
- config/exp027.yaml

## 影響範囲
- チェックポイントサイズ削減（17GB → 数百MB程度）
- 学習速度向上
