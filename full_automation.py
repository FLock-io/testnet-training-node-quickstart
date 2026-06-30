from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import requests
from huggingface_hub import HfApi

from utils.flock_api import (
    extract_training_hf_dataset_id,
    extract_training_zip_url,
    get_task,
    submit_task,
)
from utils.gpu_utils import get_gpu_type

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "robotics_vla_training_traces.zip"
OUTPUT_DIR = ROOT / "outputs" / "basic_vla_policy"

# Default public HF training dataset (used when FedLedger does not provide a URL/dataset).
DEFAULT_HF_TRAINING_DATASET = "random-sequence/flock-robotics-vla-training-v2"


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def main() -> None:
    task_id = os.environ["TASK_ID"]
    hf_username = os.environ["HF_USERNAME"]
    hf_token = os.environ["HF_TOKEN"]

    task = get_task(task_id)
    print(json.dumps({"task": task}, indent=2))

    # Determine training data source:
    # 1. If FedLedger provides a zip URL, download and use it.
    # 2. If FedLedger provides an HF dataset ID, use that.
    # 3. Fall back to the known default HF training dataset.
    training_flags: list[str] = []

    try:
        training_url = extract_training_zip_url(task)
        print(f"Downloading training zip from FedLedger to {DATA_PATH}")
        download_file(training_url, DATA_PATH)
        training_flags = ["--data", str(DATA_PATH)]
    except KeyError:
        hf_dataset_id = (
            extract_training_hf_dataset_id(task) or DEFAULT_HF_TRAINING_DATASET
        )
        print(f"No training zip URL found; using HF dataset: {hf_dataset_id}")
        training_flags = ["--hf-dataset", hf_dataset_id]

    train_command = [
        sys.executable,
        str(ROOT / "scripts" / "train_basic_vla.py"),
        *training_flags,
        "--out",
        str(OUTPUT_DIR),
        "--epochs",
        str(env_int("VLA_EPOCHS", 5)),
        "--batch-size",
        str(env_int("VLA_BATCH_SIZE", 128)),
        "--step-stride",
        str(env_int("VLA_STEP_STRIDE", 4)),
        "--max-samples",
        str(env_int("VLA_MAX_SAMPLES", 12000)),
        "--device",
        os.getenv("VLA_DEVICE", "auto"),
    ]
    if os.getenv("VLA_AMP", "0") == "1":
        train_command.append("--amp")
    subprocess.run(train_command, check=True, cwd=ROOT)

    repo_id = os.getenv(
        "HF_REPO_ID", f"{hf_username}/robotics-vla-task-{task_id}-basic"
    )
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
    commit = api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload Robotics VLA basic policy for task {task_id}",
    )
    gpu_type = get_gpu_type()
    submit_response = submit_task(
        task_id=task_id,
        hg_repo_id=repo_id,
        base_model="robotics_vla_basic_cnn_bc",
        gpu_type=gpu_type,
        revision=commit.oid,
    )
    print(
        json.dumps(
            {
                "repo_id": repo_id,
                "revision": commit.oid,
                "submit_response": submit_response,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
