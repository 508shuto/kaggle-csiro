# Requirements

## 目的
exp037のDDP学習で発生している、全rankが同一の`last.ckpt`をリネームして競合する問題を解消する。

## 変更/追加する機能
- `trainer.fit()` 後のckptリネーム処理をrank0のみで実行するように変更
- `wandb.finish()` もrank0のみで実行するように変更
- rank間の同期を確保するため、`trainer.strategy.barrier()` を追加

## 制約・前提条件
- DDP環境（2 devices）で動作すること
- 既存のckptリネーム機能は維持すること
- wandbのログ機能は正常に動作すること

