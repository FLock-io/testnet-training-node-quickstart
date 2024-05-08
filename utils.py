import torch
from loguru import logger
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )
    if "gemma" in model_name_or_path.lower():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]}
        )

    if tokenizer.__class__.__name__ == "QWenTokenizer":
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
    logger.info(f"vocab_size of tokenizer: {tokenizer.vocab_size}")
    return tokenizer


def load_model(model_name_or_path: str, val_args: TrainingArguments) -> Trainer:
    # logger.info(f'Loading model from base model: {args.model_name_or_path}')

    if val_args.use_cpu:
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16 if val_args.fp16 else torch.bfloat16
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=None,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if "output_router_logits" in model.config.to_dict():
        logger.info("set output_router_logits as True")
        model.config.output_router_logits = True
    logger.info(
        f"memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB"
    )

    total = sum(p.numel() for p in model.parameters())
    logger.info("Total model params: %.2fM" % (total / 1e6))

    return model
