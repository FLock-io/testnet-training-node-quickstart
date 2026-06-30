import os
from typing import Any

import requests

FED_LEDGER_BASE_URL = "https://fed-ledger-prod.flock.io/api/v1"


def get_task(task_id: int | str) -> dict[str, Any]:
    response = requests.get(
        f"{FED_LEDGER_BASE_URL}/tasks/get?task_id={task_id}", timeout=60
    )
    response.raise_for_status()
    return response.json()


def submit_task(
    task_id: int | str,
    hg_repo_id: str,
    base_model: str,
    gpu_type: str,
    revision: str,
    api_key: str | None = None,
) -> dict[str, Any]:
    key = api_key or os.environ["FLOCK_API_KEY"]
    payload = {
        "task_id": int(task_id),
        "data": {
            "hg_repo_id": hg_repo_id,
            "base_model": base_model,
            "gpu_type": gpu_type,
            "revision": revision,
        },
    }
    headers = {
        "flock-api-key": key,
        "Content-Type": "application/json",
    }
    response = requests.post(
        f"{FED_LEDGER_BASE_URL}/tasks/submit-result",
        headers=headers,
        json=payload,
        timeout=60,
    )
    if response.status_code != 200:
        raise Exception(f"Failed to submit task: {response.text}")
    return response.json()


def extract_training_zip_url(task: dict[str, Any]) -> str:
    data = task.get("data", {})
    for key in (
        "training_set_url",
        "training_data_url",
        "training_zip_url",
        "dataset_url",
    ):
        value = data.get(key)
        if value:
            return str(value)
    raise KeyError(
        "Task data does not include a training zip URL. Expected one of "
        "training_set_url, training_data_url, training_zip_url, or dataset_url."
    )


def extract_training_hf_dataset_id(task: dict[str, Any]) -> str | None:
    """Return the HuggingFace dataset ID from the task payload, if present."""
    data = task.get("data", {})
    for key in ("training_hf_dataset", "hf_dataset_id", "training_dataset_id"):
        value = data.get(key)
        if value:
            return str(value)
    return None
