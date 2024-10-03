import json
import os

import requests

FLOCK_API_KEY = os.environ["FLOCK_API_KEY"]
FED_LEDGER_BASE_URL = "https://fed-ledger-prod.flock.io/api/v1"


def get_task(task_id: int):
    response = requests.request(
        "GET", f"{FED_LEDGER_BASE_URL}/tasks/get?task_id={task_id}"
    )
    return response.json()


def submit_task(
    task_id: int, hg_repo_id: str, base_model: str, gpu_type: str, revision: str
):
    payload = json.dumps(
        {
            "task_id": task_id,
            "data": {
                "hg_repo_id": hg_repo_id,
                "base_model": base_model,
                "gpu_type": gpu_type,
                "revision": revision,
            },
        }
    )
    headers = {
        "flock-api-key": FLOCK_API_KEY,
        "Content-Type": "application/json",
    }
    response = requests.request(
        "POST",
        f"{FED_LEDGER_BASE_URL}/tasks/submit-result",
        headers=headers,
        data=payload,
    )
    if response.status_code != 200:
        raise Exception(f"Failed to submit task: {response.text}")
    return response.json()
