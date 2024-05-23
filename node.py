import json
import os
import time

import requests
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from demo import train_and_merge
from main import FLOCK_API_KEY


class TrainNode:
    FED_LEDGER_BASE_URL = "https://fed-ledger-prod.flock.io/api/v1"

    def __init__(self, task_id: int = 2,
                 use_proxy: bool = False):
        self.task_id = task_id
        self.use_proxy = use_proxy
        data = self.get_task_data()
        self.download_data(data)
        self.content_length = data["context_length"]

    def get_task_data(self):
        response = requests.get(
            f"{self.FED_LEDGER_BASE_URL}/tasks/get?task_id={self.task_id}",
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get task: {response.text}")
        r_js = response.json()
        data = r_js["data"]
        return data

    def download_data(self, data):
        if os.path.exists("demo_data.jsonl"):
            logger.info("Data already downloaded.")
            return
        else:
            logger.info("Downloading data...")
            r = requests.get(data["training_set_url"], stream=True)
            logger.info("Data downloaded successfully. Saving to demo_data.jsonl...")
            with open("demo_data.jsonl", "wb") as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)

    def train(self):
        logger.info("Start to train the model...")
        train_and_merge(context_length=self.content_length)

    def push(self):
        hg_repo_id = "gemma-2b-flock-" + str(int(time.time()))
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            "merged_model",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map='cuda',
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "merged_model",
        )
        tokenizer.push_to_hub(
            repo_id=hg_repo_id, use_temp_dir=True, token=os.environ["HF_TOKEN"]
        )
        logger.info("SUCCESSFULLY PUSHED TOKENIZER TO HUB")
        logger.info("Start to push the model to the hub...")
        model.push_to_hub(
            repo_id=hg_repo_id, use_temp_dir=True, token=os.environ["HF_TOKEN"]
        )
        logger.info("SUCCESSFULLY PUSHED MODEL TO HUB")
        self.submit_task(hg_repo_id)

    def submit_task(self, hg_repo_id: str):
        payload = json.dumps(
            {"task_id": self.task_id, "data": {"hg_repo_id": hg_repo_id, "base_model": "gemma"}}
        )
        response = requests.post(
            f"{self.FED_LEDGER_BASE_URL}/tasks/submit-result",
            headers={
                "flock-api-key": FLOCK_API_KEY,
                "Content-Type": "application/json",
            },
            data=payload,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to submit task: {response.text}")
        return response.json()
