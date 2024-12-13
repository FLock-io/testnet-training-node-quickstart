qwen_template = {
    "system_format": "<|im_start|>system\n{content}<|im_end|>\n",
    "user_format": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "assistant_format": "{content}<|im_end|>\n",
    "system": "You are a helpful assistant.",
}

gemma_template = {
    "system_format": "<bos>",
    "user_format": "<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
    "assistant_format": "{content}<eos>\n",
    "system": None,
}

model2template = {
    "Qwen/Qwen2.5-0.5B": qwen_template,
    "Qwen/Qwen2.5-1.5B": qwen_template,
    "Qwen/Qwen2.5-7B": qwen_template,
    "google/gemma-2-2b": gemma_template,
    "google/gemma-2-9b": gemma_template,
}

model2size = {
    "Qwen/Qwen2.5-0.5B": 494_000_000,
    "Qwen/Qwen2.5-1.5B": 1_540_000_000,
    "Qwen/Qwen2.5-7B": 7_620_000_000,
    "google/gemma-2-2b": 2_610_000_000,
    "google/gemma-2-9b": 9_240_000_000,
}

model2base_model = {
    "Qwen/Qwen2.5-0.5B": "qwen2.5",
    "Qwen/Qwen2.5-1.5B": "qwen2.5",
    "Qwen/Qwen2.5-7B": "qwen2.5",
    "google/gemma-2-2b": "gemma2",
    "google/gemma-2-9b": "gemma2",
}
