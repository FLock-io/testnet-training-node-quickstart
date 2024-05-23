import os

import torch
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)
from trl import SFTTrainer

from dataset import SFTDataCollator, SFTDataset
from merge import merge_lora_to_base_model
from utils.constants import model2template


def train_and_merge(
    model_id: str = "google/gemma-2b",
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    context_length: int = 512,
):
    assert model_id in model2template, f"model_id {model_id} not supported"
    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # Load model in 4-bit to do qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_train_epochs=num_train_epochs,
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
    dataset = SFTDataset(
        file="demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model2template[model_id],
    )

    # Define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        packing=True,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
        max_seq_length=context_length,
    )

    # Train model
    trainer.train()

    # save model
    trainer.save_model("outputs")

    # merge lora to base model
    print("Training Completed. Start to merge the weights....")
    merge_lora_to_base_model(
        model_name_or_path=model_id,
        adapter_name_or_path="outputs",
        save_path="merged_model",
    )


if __name__ == "__main__":
    train_and_merge()
