qwen_template = {
    "system_format": "<|im_start|>system\n{content}<|im_end|>\n",
    "user_format": "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "assistant_format": "{content}<|im_end|>\n",
    "tool_format": "{content}",
    "function_format": "{content}",
    "observation_format": "<|im_start|>tool\n{content}<|im_end|>\n<|im_start|>assistant\n",
    "system": "You are a helpful assistant.",
}

model2template = {
    "Qwen/Qwen3.5-0.8B": qwen_template,
    "Qwen/Qwen3.5-0.8B-Base": qwen_template,
    "Qwen/Qwen3.5-2B": qwen_template,
    "Qwen/Qwen3.5-2B-Base": qwen_template,
    "Qwen/Qwen3.5-4B": qwen_template,
    "Qwen/Qwen3.5-4B-Base": qwen_template,
    "Qwen/Qwen3.5-9B": qwen_template,
    "Qwen/Qwen3.5-9B-Base": qwen_template,
    "Qwen/Qwen3.5-27B": qwen_template,
}

model2size = {
    "Qwen/Qwen3.5-0.8B": 853_000_000,
    "Qwen/Qwen3.5-0.8B-Base": 853_000_000,
    "Qwen/Qwen3.5-2B": 2_213_000_000,
    "Qwen/Qwen3.5-2B-Base": 2_213_000_000,
    "Qwen/Qwen3.5-4B": 4_539_000_000,
    "Qwen/Qwen3.5-4B-Base": 4_539_000_000,
    "Qwen/Qwen3.5-9B": 8_392_000_000,
    "Qwen/Qwen3.5-9B-Base": 8_392_000_000,
    "Qwen/Qwen3.5-27B": 26_085_000_000,
}

model2base_model = {
    "Qwen/Qwen3.5-0.8B": "qwen3.5",
    "Qwen/Qwen3.5-0.8B-Base": "qwen3.5",
    "Qwen/Qwen3.5-2B": "qwen3.5",
    "Qwen/Qwen3.5-2B-Base": "qwen3.5",
    "Qwen/Qwen3.5-4B": "qwen3.5",
    "Qwen/Qwen3.5-4B-Base": "qwen3.5",
    "Qwen/Qwen3.5-9B": "qwen3.5",
    "Qwen/Qwen3.5-9B-Base": "qwen3.5",
    "Qwen/Qwen3.5-27B": "qwen3.5",
}
