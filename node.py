import json
import os
import time

import requests
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from demo import train_and_merge


def check_env_var(VAR_NAME, PASSED_VAR):
    if os.environ.get(VAR_NAME) is None and PASSED_VAR is None:
        raise Exception(f"{VAR_NAME} not found in environment variables. "
                        f"You should assign your Hugging Face token to the {VAR_NAME} variable. "
                        f"Or your can directly pass the {VAR_NAME} to TrainNode directly.")
    return os.environ[VAR_NAME] if PASSED_VAR is None else PASSED_VAR


class TrainNode:
    FED_LEDGER_BASE_URL = "https://fed-ledger-prod.flock.io/api/v1"

    def __init__(self, task_id: int = 2,
                 training_args: dict = None,
                 HF_TOKEN: str = None,
                 HF_USERNAME: str = None,
                 FLOCK_API_KEY: str = None):
        self.HF_TOKEN = check_env_var("HF_TOKEN", HF_TOKEN)
        self.HF_USERNAME = check_env_var("HF_USERNAME", HF_USERNAME)
        self.FLOCK_API_KEY = check_env_var("FLOCK_API_KEY", FLOCK_API_KEY)
        self.task_id = task_id
        data = self.get_task_data()
        self.download_data(data)
        self.content_length = data["context_length"]
        self.training_args = {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 12,
        } if training_args is None else training_args

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
        train_and_merge(context_length=self.content_length, **self.training_args)

    def load_merged_model(self, model_path='merged_model'):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map='cuda',
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )
        return model, tokenizer

    def push(self, model=None, tokenizer=None, hf_repo_id: str = None):
        hf_repo_id = "gemma-2b-flock-" + str(int(time.time())) if hf_repo_id is None else hf_repo_id
        # Load model
        if model is None or tokenizer is None:
            model, tokenizer = self.load_merged_model()
        tokenizer.push_to_hub(
            repo_id=hf_repo_id, use_temp_dir=True, token=self.HF_TOKEN
        )
        logger.info("SUCCESSFULLY PUSHED TOKENIZER TO HUB")
        logger.info("Start to push the model to the hub...")
        model.push_to_hub(
            repo_id=hf_repo_id, use_temp_dir=True, token=self.HF_TOKEN
        )
        logger.info("SUCCESSFULLY PUSHED MODEL TO HUB")
        self.submit_task(f"{self.HF_USERNAME}/{hf_repo_id}")

    def submit_task(self, hg_repo_id: str):
        payload = json.dumps(
            {"task_id": self.task_id, "data": {"hg_repo_id": hg_repo_id, "base_model": "gemma"}}
        )
        response = requests.post(
            f"{self.FED_LEDGER_BASE_URL}/tasks/submit-result",
            headers={
                "flock-api-key": self.FLOCK_API_KEY,
                "Content-Type": "application/json",
            },
            data=payload,
        )
        if response.status_code != 200:
            raise Exception(f"Failed to submit task: {response.text}")
        return response.json()
