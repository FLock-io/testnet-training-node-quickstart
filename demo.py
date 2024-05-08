import os

import torch
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

from dataset import SFTDataCollator, GemmaSFTDataset
from merge import merge_lora_to_base_model

lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)


model_id = "google/gemma-2b"
# Load model in 4-bit to do qLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=2,
    max_steps=10,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=1,
    output_dir="outputs",
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
    token=os.environ["HF_TOKEN"],
)

# Load dataset
dataset = GemmaSFTDataset(
    file="demo_data.jsonl",
    tokenizer=tokenizer,
    max_seq_length=512,
)

# Define trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    packing=True,
    data_collator=SFTDataCollator(tokenizer, max_seq_length=512),
    max_seq_length=512,
)

# Train model
trainer.train()

# save model
trainer.save_model("outputs")

# merge lora to base model
merge_lora_to_base_model(
    model_name_or_path="google/gemma-2b",
    adapter_name_or_path="outputs",
    save_path="merged_model",
)
