import os
from pathlib import Path

import tyro
from huggingface_hub import snapshot_download


def main(
    exp_name: str,
    repo_id: str,
    snapshot_dir: Path | None = None,
    revision: str = "main",
    token_env: str = "HF_TOKEN",
    config_only: bool = True,
) -> None:
    """Download HF gated model snapshot to local directory.

    Args:
        exp_name: Experiment name (e.g., "exp038")
        repo_id: HuggingFace repository ID (e.g., "facebook/dinov3-vitl16-pretrain-lvd1689m")
        snapshot_dir: Output directory for snapshot. If None, uses ./output/{exp_name}/model
        revision: Repository revision (default: "main")
        token_env: Environment variable name for HF token (default: "HF_TOKEN")
        config_only: If True, download only config files for lightweight snapshot (default: True)
    """
    if snapshot_dir is None:
        snapshot_dir = Path(f"./output/{exp_name}/model")
    else:
        snapshot_dir = Path(snapshot_dir)

    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)

    token = os.getenv(token_env)
    if token is None:
        raise ValueError(f"Environment variable {token_env} is not set. Please set it with your HF token.")

    print(f"Downloading {repo_id} to {snapshot_dir}...")
    print(f"Config only: {config_only}")

    download_kwargs = {
        "repo_id": repo_id,
        "token": token,
        "revision": revision,
        "local_dir": str(snapshot_dir),
        "local_dir_use_symlinks": False,
    }

    if config_only:
        # Download only config files for lightweight snapshot
        download_kwargs["allow_patterns"] = ["config.json", "preprocessor_config.json", "*.json"]

    snapshot_download(**download_kwargs)

    print(f"Snapshot saved to {snapshot_dir}")
    print("Files in snapshot:")
    for file in sorted(snapshot_dir.rglob("*")):
        if file.is_file():
            print(f"  {file.relative_to(snapshot_dir)}")


if __name__ == "__main__":
    tyro.cli(main)
