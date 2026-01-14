---
description: 実験ファイルをgit addでステージング
allowed-tools: Bash(git:*), Read, Glob
---

# Stage Experiment Files

指定された実験番号のファイルをgit addでステージングします。

## 使い方

```
/stage-exp exp042
```

## 対象ファイル

- `src/$exp_name/` - ソースコード
- `config/$exp_name.yaml` - 設定ファイル
- `doc/experiment/$exp_name.md` - 実験ドキュメント
- `.steering/*-$exp_name-*/` - ステアリングファイル

## 実行内容

$ARGUMENTS を実験番号として使用し、以下を実行してください:

1. 対象ファイルの存在確認
2. git add の実行
3. git status で結果表示

```bash
exp_name="$ARGUMENTS"

# 存在確認と git add
[ -d "src/$exp_name" ] && git add "src/$exp_name/"
[ -f "config/$exp_name.yaml" ] && git add "config/$exp_name.yaml"
[ -f "doc/experiment/$exp_name.md" ] && git add "doc/experiment/$exp_name.md"

# ステアリング (パターンマッチ)
for dir in .steering/*-$exp_name-*/; do
  [ -d "$dir" ] && git add "$dir"
done

# 結果表示
git status
```
