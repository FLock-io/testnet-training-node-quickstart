import json
import os
import time

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from demo import train_and_merge

FLOCK_API_KEY = os.environ["FLOCK_API_KEY"]
FED_LEDGER_BASE_URL = "https://fed-ledger-prod.flock.io/api/v1"
HG_USERNAME = os.environ["HG_USERNAME"]


def get_task(task_id: int):
    response = requests.request(
        "GET", f"{FED_LEDGER_BASE_URL}/tasks/get?task_id={task_id}"
    )
    return response.json()


def submit_task(task_id: int, hg_repo_id: str):
    payload = json.dumps(
        {"task_id": task_id, "data": {"hg_repo_id": hg_repo_id, "base_model": "gemma"}}
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
    return response.json()


if __name__ == "__main__":
    task_id = os.environ["TASK_ID"]
    task = get_task(task_id)
    # download data from a presigned url
    data_url = task["data"]["training_set_url"]
    context_length = task["data"]["context_length"]
    # download in chunks
    response = requests.get(data_url, stream=True)
    with open("demo_data.jsonl", "wb") as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)
    # train and merge
    print("Start to train the model...")
    train_and_merge(context_length=context_length)

    # generate a random repo id based on timestamp
    hg_repo_id = "gemma-2b-flock-" + str(int(time.time()))

    # load the merged model
    model = AutoModelForCausalLM.from_pretrained(
        "merged_model",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    # upload
    print("Start to push the model to the hub...")
    model.push_to_hub(
        repo_id=hg_repo_id, use_temp_dir=True, token=os.environ["HF_TOKEN"]
    )
    # upload tokenizer as well
    tokenizer = AutoTokenizer.from_pretrained(
        "merged_model",
    )
    tokenizer.push_to_hub(
        repo_id=hg_repo_id, use_temp_dir=True, token=os.environ["HF_TOKEN"]
    )
    # submit
    submit_task(task_id, f"{HG_USERNAME}/hg_repo_id")
    print("Task submitted successfully")
