import json
import os
import time

import requests
import yaml
import git
import shutil
from loguru import logger
from huggingface_hub import HfApi

from demo import LoraTrainingArguments, train_lora
from utils.constants import model2base_model, model2size
from utils.flock_api import get_task, submit_task

HF_USERNAME = os.environ["HF_USERNAME"]


def clone_repo(link: str, dest_folder: str):
    """
        Clone a remote Git repository to a specified local directory.

        Parameters:
        link (str): The URL of the remote Git repository.
        dest_folder (str): The path to the local destination directory.
    """
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.makedirs(dest_folder)

    git.Repo.clone_from(link, dest_folder)
    logger.info(f"Repository cloned to {dest_folder}")


if __name__ == "__main__":
    task_id = os.environ["TASK_ID"]
    # The Git link should contain two files: training_args.yaml and demo_data.jsonl.
    repo_url = os.getenv("GIT_URL")
    all_training_args = None
    if repo_url is not None:
        git_repo_path = "temp_repo"
        clone_repo(repo_url, git_repo_path)
        if os.path.exists(f"{git_repo_path}/training_args.yaml"):
            with open(f"{git_repo_path}/training_args.yaml", "r") as f:
                all_training_args = yaml.safe_load(f)

    if all_training_args is None:
        logger.info("Use the default training_args.yml file")
        current_folder = os.path.dirname(os.path.realpath(__file__))
        with open(f"{current_folder}/training_args.yaml", "r") as f:
            all_training_args = yaml.safe_load(f)

    task = get_task(task_id)
    # log the task info
    logger.info(json.dumps(task, indent=4))
    # download data from a presigned url

    context_length = task["data"]["context_length"]
    max_params = task["data"]["max_params"]

    # filter out the model within the max_params
    model2size = {k: v for k, v in model2size.items() if v <= max_params}
    all_training_args = {k: v for k, v in all_training_args.items() if k in model2size}
    logger.info(f"Models within the max_params: {all_training_args.keys()}")

    train_file = None
    if repo_url is not None:
        if os.path.exists(f"{git_repo_path}/demo_data.jsonl"):
            train_file = f"{git_repo_path}/demo_data.jsonl"

    if train_file is None:
        # download in chunks
        logger.info("Use the default demo_data.jsonl file")
        data_url = task["data"]["training_set_url"]
        response = requests.get(data_url, stream=True)
        with open("demo_data.jsonl", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        train_file = "demo_data.jsonl"

    # train all feasible models and merge
    for model_id in all_training_args.keys():
        logger.info(f"Start to train the model {model_id}...")
        # if OOM, proceed to the next model
        try:
            train_lora(
                model_id=model_id,
                context_length=context_length,
                training_args=LoraTrainingArguments(**all_training_args[model_id]),
                train_file=train_file
            )
        except RuntimeError as e:
            logger.error(f"Error: {e}")
            logger.info("Proceed to the next model...")
            continue

        # generate a random repo id based on timestamp
        hg_repo_id = f"{model_id.replace('/', '-')}-" + str(int(time.time()))

        try:
            logger.info("Start to push the lora weight to the hub...")
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.create_repo(
                f"{HF_USERNAME}/{hg_repo_id}",
                repo_type="model",
            )
            api.upload_folder(
                folder_path="outputs",
                repo_id=f"{HF_USERNAME}/{hg_repo_id}",
                repo_type="model",
            )
            # submit
            submit_task(
                task_id, f"{HF_USERNAME}/{hg_repo_id}", model2base_model[model_id]
            )
            logger.info("Task submitted successfully")
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.info("Proceed to the next model...")
        finally:
            # cleanup merged_model and output
            os.system("rm -rf merged_model")
            os.system("rm -rf outputs")
            continue
