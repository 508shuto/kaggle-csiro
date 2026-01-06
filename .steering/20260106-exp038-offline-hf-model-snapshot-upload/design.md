# Design
## アプローチ
- ローカル/Colabで `huggingface_hub.snapshot_download(..., local_dir_use_symlinks=False)` によりモデルrepoを取得する（gatedなのでtokenはここでのみ使用）
- 取得したフォルダを `./output/{exp_name}/model` に保存する
- `src/utils/upload_scripts.py` で `.ckpt` + `config.yaml` + `model/` を同一Kaggle Datasetとしてアップロードする
- Submission Notebook（Internet OFF）では Dataset をマウントし、そのローカルパスを config の `model.name` に設定して推論を実行する

## 変更コンポーネント
- `src/utils/download_hf_snapshot.py`（新規）
  - tyro CLIで `exp_name`, `repo_id`, `snapshot_dir`（デフォルト: `./output/{exp_name}/model`）等を受け取る
  - `config_only=True` のときは `allow_patterns=["config.json", "preprocessor_config.json", "*.json"]` で軽量化
  - `local_dir_use_symlinks=False` を固定（Kaggleへ持ち込むため）
- `src/utils/upload_scripts.py`
  - `./output/{exp_name}/model` が存在すれば、`tmp/{exp_name}/model` に `copytree` して同梱
  - 存在しない場合は警告してスキップ（既存挙動維持）

## 影響範囲
- Kaggle Submission: 外部通信ゼロで推論可能（token持ち込み不要）
- ローカル開発: 従来通りHF repo ID参照も可能（オフラインモードを強制しない設計）

## Kaggle Notebook（Internet OFF）での使い方

### 1. Add Data
以下の2つのDatasetをAdd Dataする:
- `kaggle-csiro-src`（`src/exp038` のコード）
- `kaggle-csiro-exp038`（`.ckpt` + `config.yaml` + `model/`）

### 2. パス設定
```python
SRC_DIR = "/kaggle/input/kaggle-csiro-src"
ARTIFACT_DIR = "/kaggle/input/kaggle-csiro-exp038"
MODEL_DIR = "/kaggle/input/kaggle-csiro-exp038/model"
COMP_DIR = "/kaggle/input/<competition-slug>"  # test.csvと画像があるディレクトリ
```

### 3. Configの差し替え
Internet OFFなので、`config.yaml` の `model.name` をローカルパスに変更する必要がある。

```python
import yaml
from pathlib import Path

# 元のconfigを読み込み
with open(f"{ARTIFACT_DIR}/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# model.nameをローカルパスに変更
config["model"]["name"] = MODEL_DIR
config["dataset"]["input_dir"] = COMP_DIR

# 新しいconfigを保存
with open("/kaggle/working/config_kaggle.yaml", "w") as f:
    yaml.dump(config, f)
```

### 4. 実行
```python
import sys
sys.path.insert(0, SRC_DIR)

from inference import main as inference_main
from pathlib import Path

inference_main(
    test_csv_path=Path(f"{COMP_DIR}/test.csv"),
    config_path=Path("/kaggle/working/config_kaggle.yaml"),
    model_dir=Path(ARTIFACT_DIR),
    output_dir=Path("/kaggle/working"),
    folds=[0, 1, 2, 3, 4],
    device="cuda",
    batch_size=32,
    num_workers=4,
    use_amp=True,
    use_tta=False,
)
```

### 5. 注意事項
- 画像はURLではなくローカルファイルパスを使用する
- Internet OFFなのでHF repo IDのままだと `AutoConfig.from_pretrained()` が失敗する（必ずローカルdirへ）
- `model.name` がローカルディレクトリパスになっていれば、`pretrained=False` でも `AutoConfig.from_pretrained()` がローカルから読み込む

