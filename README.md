# testnet-training-node-quickstart

This repository contains a demo script for you to fine-tune a Gemma model for train.flock.io.

## Quickstart

### Install Dependencies

To set up your environment, run the following commands:

```bash
conda create -n training-node python==3.10
conda activate training-node
pip install -r requirements.txt
```

### File Structure

- [`dataset.py`](dataset.py) - Contains the logic to process the raw data from `demo_data.jsonl`.
- [`demo_data.jsonl`](demo_data.jsonl) - Follows the shareGPT format. The training data you receive from the `fed-ledger` is in exactly the same format.
- [`merge.py`](merge.py) - Contains the utility function for merging LoRA weights. If you are training with LoRA, please ensure you merge the adapter before uploading to your Hugging Face repository.
- [`demo.py`](demo.py) - A training script that implements LoRA fine-tuning for a Gemma-2B model.
- [`full_automation.py`](full_automation.py) - A script that automate everything including get a task, download the training data, finetune Gemma-2B on training data, merge weights, upload to your HuggingFace model repo, and submit the task to fed-ledger.
- [`training_args.yaml`](training_args.yaml) - A YAML defines the training hyper-parameters for fine-tuning. A detailed explanation on LoRA config can be found here: [LoRA Fine-tuning & Hyperparameters Explained](https://www.entrypointai.com/blog/lora-fine-tuning/)

### Full Automation

Simply run

```bash
TASK_ID=<task-id> FLOCK_API_KEY="<your-flock-api-key-stakes-as-node-for-the-task>" HF_TOKEN="<your-hf-token>" CUDA_VISIBLE_DEVICES=0 HF_USERNAME="your-hf-user-name" python full_automation.py
```

The above command will automatically train and submit multiple LLMs that are smaller the max parameters limitation for the given task.

#### Bypass certain models

If you want to bypass certain models, simply comment out the model config in the [`training_args.yaml`](training_args.yaml)

---

### Play with demo.py

#### Start the Training

Execute the following command to start the training:

```bash
HF_TOKEN="hf_yourhftoken" CUDA_VISIBLE_DEVICES=0 python demo.py
```

The HF token is required due to the Gemma License.

This command initiates fine-tuning on the demo dataset, saves the fine-tuned model, merges the adapter to the base model, and saves the final model.

#### Upload the model folder to your HuggingFace repo

[HuggingFace Models Uploading](https://huggingface.co/docs/hub/en/models-uploading)

#### Getting the task id

Before you submit the model script for a task, you will first need to stake on the task as a node.

- Go to the [FLock Training Platform](https://train.flock.io/stake)
- Select **Training Node** tab and stake on the task you want to submit the model to.
- The task ID is the ID of the task you want to submit the model to.

#### Submit the model

```bash

curl --location 'https://fed-ledger-prod.flock.io/api/v1/tasks/submit-result' \
--header 'flock-api-key: <your-api-key-with-staking-on-this-task-as-node>' \
--header 'Content-Type: application/json' \
--data '{
    "task_id": 29,
    "data":{
        "hg_repo_id": "Qwen/Qwen1.5-1.8B-Chat",
        "base_model": "qwen1.5",
        "gpu_type": "<GPU-used-for-training>"
    }
}'
```
