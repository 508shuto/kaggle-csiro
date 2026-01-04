# Design

## アプローチ
exp026のコードベースをそのまま活用し、設定ファイルの変更のみで対応

- models.pyは`self.model.num_features`を動的に使用しているため、large モデル（1024次元）に自動対応
- 画像サイズはconfig経由でaugmentationに渡されるため、コード変更不要

## 変更コンポーネント
| ファイル | 変更内容 |
|---------|---------|
| config/exp028.yaml | model.name, augmentation.train/valid.image_size |

## 影響範囲
- モデルサイズ増加によるメモリ使用量増加
- 画像サイズ増加による計算量増加（約5倍）
- Head層の入力次元変更（768 → 1024）※自動対応
