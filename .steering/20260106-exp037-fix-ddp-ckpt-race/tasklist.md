# Tasklist

## 完了条件（Definition of Done）
- [ ] ステアリングドキュメント（requirements.md, design.md, tasklist.md）が作成されている
- [ ] `src/exp037/train.py` の後処理がrank0のみで実行されるように修正されている
- [ ] `trainer.strategy.barrier()` による同期が追加されている
- [ ] DDP(2 devices)環境で `--folds 0 1` を実行し、FileNotFoundErrorが発生しないことを確認
- [ ] DDP(2 devices)環境で `--folds 0 1` を実行し、NCCL timeoutが発生しないことを確認
- [ ] `output/exp037/` に正しい形式のckptファイルが生成されることを確認

## タスク
- [x] TODO: ステアリングフォルダとドキュメント作成
- [x] TODO: `src/exp037/train.py` の後処理をrank0のみで実行するように修正
- [ ] TODO: DDP環境での動作確認

