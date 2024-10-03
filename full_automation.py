import json
import os
import time

import requests
import yaml
from loguru import logger
from huggingface_hub import HfApi

from demo import LoraTrainingArguments, train_lora
from utils.constants import model2base_model, model2size
from utils.flock_api import get_task, submit_task
from utils.gpu_utils import get_gpu_type

HF_USERNAME = os.environ["HF_USERNAME"]

if __name__ == "__main__":
    task_id = os.environ["TASK_ID"]
    # load trainin args
    # define the path of the current file
    current_folder = os.path.dirname(os.path.realpath(__file__))
    with open(f"{current_folder}/training_args.yaml", "r") as f:
        all_training_args = yaml.safe_load(f)

    task = get_task(task_id)
    # log the task info
    logger.info(json.dumps(task, indent=4))
    # download data from a presigned url
    data_url = task["data"]["training_set_url"]
    context_length = task["data"]["context_length"]
    max_params = task["data"]["max_params"]

    # filter out the model within the max_params
    model2size = {k: v for k, v in model2size.items() if v <= max_params}
    all_training_args = {k: v for k, v in all_training_args.items() if k in model2size}
    logger.info(f"Models within the max_params: {all_training_args.keys()}")
    # download in chunks
    response = requests.get(data_url, stream=True)
    with open("demo_data.jsonl", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # train all feasible models and merge
    for model_id in all_training_args.keys():
        logger.info(f"Start to train the model {model_id}...")
        # if OOM, proceed to the next model
        try:
            train_lora(
                model_id=model_id,
                context_length=context_length,
                training_args=LoraTrainingArguments(**all_training_args[model_id]),
            )
        except RuntimeError as e:
            logger.error(f"Error: {e}")
            logger.info("Proceed to the next model...")
            continue

        # generate a random repo id based on timestamp
        gpu_type = get_gpu_type()

        try:
            logger.info("Start to push the lora weight to the hub...")
            api = HfApi(token=os.environ["HF_TOKEN"])
            repo_name = f"{HF_USERNAME}/task-{task_id}-{model_id.replace('/', '-')}"
            # check whether the repo exists
            try:
                api.create_repo(
                    repo_name,
                    exist_ok=False,
                    repo_type="model",
                )
            except Exception as e:
                logger.info(
                    f"Repo {repo_name} already exists. Will commit the new version."
                )

            commit_message = api.upload_folder(
                folder_path="outputs",
                repo_id=repo_name,
                repo_type="model",
            )
            # get commit hash
            commit_hash = commit_message.oid
            logger.info(f"Commit hash: {commit_hash}")
            logger.info(f"Repo name: {repo_name}")
            # submit
            submit_task(
                task_id, repo_name, model2base_model[model_id], gpu_type, commit_hash
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
