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
- [`demo_data.jsonl`](demo_data.jsonl) - Follows the shareGPT format. The training data you receive from
  the `fed-ledger` is in exactly the same format.
- [`merge.py`](merge.py) - Contains the utility function for merging LoRA weights. If you are training with LoRA, please
  ensure you merge the adapter before uploading to your Hugging Face repository.
- [`demo.py`](demo.py) - A training script that implements LoRA fine-tuning for a Gemma-2B model.
- [`full_automation.py`](full_automation.py) - A script that automate everything including get a task, download the
  training data, finetune Gemma-2B on training data, merge weights, upload to your HuggingFace model repo, and submit
  the task to fed-ledger.

### Play with demo.py

#### Start the Training

Execute the following command to start the training:

```bash
HF_TOKEN="hf_yourhftoken" CUDA_VISIBLE_DEVICES=0 python demo.py
```

The HF token is required due to the Gemma License.

This command initiates fine-tuning on the demo dataset, saves the fine-tuned model, merges the adapter to the base
model, and saves the final model.

#### Upload the model folder to your HuggingFace repo

[HuggingFace Models Uploading](https://huggingface.co/docs/hub/en/models-uploading)

#### Submit the model

```bash

curl --location 'https://fed-ledger-prod.flock.io/api/v1/tasks/submit-result' \
--header 'flock-api-key: <your-api-key-with-staking-on-this-task-as-node>' \
--header 'Content-Type: application/json' \
--data '{
    "task_id": <task id>,
    "data":{
        "hg_repo_id": "<your-hg-repo-id>",
        "base_model": "gemma"
    }
}'
```

### Full Automation

Simply run

```bash
TASK_ID=<task-id> FLOCK_API_KEY="<your-flock-api-key-stakes-as-node-for-the-task>" HF_TOKEN="<your-hf-token>" CUDA_VISIBLE_DEVICES=0 HF_USERNAME="your-hg-user-name" python full_automation.py
```

We also have python interface for the train node.
```python
from node import TrainNode

TASK_ID = "<task-id>"

# If you have already setup the environment variables, you can skip the following lines.
FLOCK_API_KEY = "<your-flock-api-key-stakes-as-node-for-the-task>"
HF_TOKEN = "<your-hf-token>"
HF_USERNAME = "your-hg-user-name"
# Define the TrainNode.
node = TrainNode(TASK_ID, FLOCK_API_KEY=FLOCK_API_KEY, HF_TOKEN=HF_TOKEN,
                 HF_USERNAME=HF_USERNAME)
# Train
node.train()
# Push the model.
node.push()
# You can also load your local model and push directly.
model = 'LOADED_MODEL'
tokenizer = 'LOADED_TOKENIZER'
node.push(model, tokenizer)
```