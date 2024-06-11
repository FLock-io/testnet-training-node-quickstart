import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora_to_base_model(
    model_name_or_path: str, adapter_name_or_path: str, save_path: str
):
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    model = PeftModel.from_pretrained(
        model, adapter_name_or_path, device_map={"": "cpu"}
    )
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
