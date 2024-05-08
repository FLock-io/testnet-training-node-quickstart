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

### Start the Training

Execute the following command to start the training:

```bash
HF_TOKEN="hf_yourhftoken" CUDA_VISIBLE_DEVICES=0 python demo.py
```

The HF token is required due to the Gemma License.

This command initiates fine-tuning on the demo dataset, saves the fine-tuned model, merges the adapter to the base model, and saves the final model.
