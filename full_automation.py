"""Train and submit a video inconsistency detector end to end.

1. Fetch the FLock task.
2. Get training data: the Hugging Face dataset named by the task (or the default), falling
   back to generating clips locally with the validator's synthesiser if it is not accessible.
3. Train the sample temporal detector (trainer/train.py).
4. Package the submission and smoke-test it (trainer/package_submission.py).
5. Upload it to a PRIVATE Hugging Face model repo and submit the repo + revision to FLock.

Required environment: TASK_ID, HF_USERNAME, HF_TOKEN, FLOCK_API_KEY.
Optional: see ``OPTIONAL_ENV`` below or the README.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from utils.flock_api import extract_training_hf_dataset_id, get_task, submit_task
from utils.gpu_utils import get_gpu_type

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
RUN_DIR = OUTPUT_DIR / "run"
SUBMISSION_DIR = OUTPUT_DIR / "submission"

DEFAULT_HF_TRAINING_DATASET = "random-sequence/flock-video-inconsistency"
BASE_MODEL = "video_inconsistency_temporal_cnn"

OPTIONAL_ENV = {
    "HF_REPO_ID": "model repo to upload to (default <HF_USERNAME>/video-inconsistency-task-<TASK_ID>)",
    "VIC_DATASET": "Hugging Face dataset repo to train on (overrides the task and the default)",
    "VIC_GENERATE_CLIPS": "clips to generate when no dataset is accessible (default 3000)",
    "VIC_EPOCHS": "training epochs (default 40)",
    "VIC_HIDDEN": "temporal net width (default 64)",
    "VIC_LAYERS": "temporal net depth (default 6)",
    "VIC_DEVICE": "auto | cuda | cpu (default auto)",
    "VIC_WORKERS": "CPU processes for data generation and feature extraction (default: all cores)",
    "VIC_SKIP_SUBMIT": "set to 1 to train, package and upload without submitting to FLock",
}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def resolve_device() -> str:
    requested = os.getenv("VIC_DEVICE", "auto")
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def training_data(task: dict, hf_token: str, workers: int) -> Path:
    """Download the Hugging Face dataset, or generate clips locally if it is not accessible."""
    dataset_id = (
        os.getenv("VIC_DATASET")
        or extract_training_hf_dataset_id(task)
        or DEFAULT_HF_TRAINING_DATASET
    )
    target = DATA_DIR / dataset_id.replace("/", "__")
    try:
        print(f"Downloading training dataset {dataset_id} ...", flush=True)
        path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            token=hf_token or None,
            local_dir=str(target),
        )
        return Path(path)
    except Exception as exc:  # noqa: BLE001 - any access failure falls back to generation
        print(
            f"Could not download {dataset_id} ({type(exc).__name__}: {exc}). "
            "Generating training clips locally with the validator's synthesiser instead.",
            flush=True,
        )
    generated = DATA_DIR / "generated"
    run(
        [
            sys.executable,
            str(ROOT / "trainer" / "generate_data.py"),
            "--out-dir",
            str(generated),
            "--num-clips",
            str(env_int("VIC_GENERATE_CLIPS", 3000)),
            "--workers",
            str(workers),
        ]
    )
    return generated


def main() -> None:
    task_id = os.environ["TASK_ID"]
    hf_username = os.environ["HF_USERNAME"]
    hf_token = os.environ["HF_TOKEN"]
    workers = env_int("VIC_WORKERS", os.cpu_count() or 4)
    device = resolve_device()

    task = get_task(task_id)
    print(json.dumps({"task": task}, indent=2))

    data_root = training_data(task, hf_token, workers)
    run(
        [
            sys.executable,
            str(ROOT / "trainer" / "train.py"),
            "--data-dir",
            str(data_root),
            "--out-dir",
            str(RUN_DIR),
            "--epochs",
            str(env_int("VIC_EPOCHS", 40)),
            "--hidden",
            str(env_int("VIC_HIDDEN", 64)),
            "--layers",
            str(env_int("VIC_LAYERS", 6)),
            "--device",
            device,
            "--workers",
            str(workers),
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / "trainer" / "package_submission.py"),
            "--weights-dir",
            str(RUN_DIR),
            "--out-dir",
            str(SUBMISSION_DIR),
            "--smoke-test",
        ]
    )

    repo_id = os.getenv(
        "HF_REPO_ID", f"{hf_username}/video-inconsistency-task-{task_id}"
    )
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
    commit = api.upload_folder(
        folder_path=str(SUBMISSION_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload video inconsistency detector for task {task_id}",
        ignore_patterns=["__pycache__/*", "*.pyc"],
    )
    print(f"Uploaded {repo_id}@{commit.oid} (private)", flush=True)

    if os.getenv("VIC_SKIP_SUBMIT") == "1":
        print("VIC_SKIP_SUBMIT=1: not submitting to FLock.")
        return
    submit_response = submit_task(
        task_id=task_id,
        hg_repo_id=repo_id,
        base_model=BASE_MODEL,
        gpu_type=get_gpu_type(),
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
