# Design

## アプローチ
`trainer.fit()` 後の後処理（ckptリネーム、wandb.finish）を `trainer.is_global_zero` でガードし、rank0のみで実行するようにする。また、rank間の同期を確保するため、`trainer.strategy.barrier()` を追加する。

## 変更コンポーネント
- `src/exp037/train.py` の `train_fold()` 関数内の後処理部分（97行目以降）

## 影響範囲
- `train_fold()` 関数のみの変更で、他の関数やモジュールへの影響はなし
- DDP環境でのみ影響があり、シングルGPU環境では動作に変化なし（`trainer.is_global_zero` は常にTrue）



