# Design

## アプローチ
Dataset.__getitem__()でAutoImageProcessorを呼び出し、
(pixel_values, grid_thw, target, aux_target)のtensorタプルを返す

## 変更コンポーネント
1. dataset.py - processor追加、tensor返却
2. models.py - processor削除、forward引数変更
3. train.py - collate_fn更新
4. lightning_module.py - batch展開更新
5. evaluation.py - collate_fn更新
6. inference.py - TestDataset修正

## 影響範囲
- DataLoaderのpin_memoryが有効化可能に
- Mixupがtensorで動作可能に
- 推論速度40-60%改善見込み
