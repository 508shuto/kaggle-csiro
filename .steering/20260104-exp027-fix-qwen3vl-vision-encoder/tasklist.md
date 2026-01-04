# Tasklist

## 完了条件（Definition of Done）

- [x] `uv run src/exp027/train.py --folds 0` がAttributeErrorなく実行開始できる
- [x] モデル初期化が正常に完了する
- [ ] 1エポック以上の学習が正常に進行する
- [ ] validation lossが計算される

## タスク

### Phase 1: models.py修正

- [x] Done: 属性アクセス修正 `.vision_model` → `.visual` (line 43)
- [x] Done: hidden_dimデフォルト値変更 1536 → 2048 (line 20)
- [x] Done: config検証を `out_hidden_size` に変更 (line 55)
- [x] Done: `AutoImageProcessor`のインポート追加
- [x] Done: `__init__`にprocessor初期化追加
- [x] Done: forward関数の書き換え（Processor使用、get_image_features使用）

### Phase 2: dataset.py修正

- [x] Done: `_get_transforms`から`ToTensor()`と`Normalize()`を削除
- [x] Done: `__getitem__`の戻り値型をPIL.Imageに変更

### Phase 3: train.py修正

- [x] Done: カスタム`collate_fn`関数の追加
- [x] Done: DataLoader作成時に`collate_fn`を指定

### Phase 4: config修正

- [x] Done: `config/exp027.yaml`の`hidden_dim`を2048に変更
- [x] Done: `mixup.enabled`をfalseに変更

### Phase 5: 検証

- [x] Done: `uv run src/exp027/train.py --folds 0` 実行
- [x] Done: エラーなく学習開始を確認
- [ ] TODO: 1エポック完了を確認

### Phase 6: 追加対応

- [x] Done: inference.pyに同様の変更を適用
  - TestDataset: ToTensor/Normalize削除、PIL画像を返す
  - collate_fn追加
  - predict関数: PIL画像リスト対応
  - TTA: PIL画像では非対応のため警告表示
- [x] Done: evaluation.pyに同様の変更を適用
  - collate_fn追加
  - 推論ループ: PIL画像リスト対応
