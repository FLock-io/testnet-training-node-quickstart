import json
from typing import Any, Dict, List

import torch
from loguru import logger
from torch.utils.data import Dataset


class UnifiedSFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.system_format = "<bos>"
        self.user_format = (
            "<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"
        )
        self.assistant_format = "{content}<|eot_id|>"
        self.system = None

        self.max_seq_length = max_seq_length
        logger.info("Loading data: {}".format(file))
        with open(file, "r", encoding="utf8") as f:
            data_list = f.readlines()
        logger.info("There are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []

        # setting system information
        if self.system_format is not None:
            system = data["system"].strip() if "system" in data.keys() else self.system

            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)

        conversations = data["conversations"]

        for i in range(0, len(conversations) - 1, 2):
            if (
                conversations[i]["role"] != "user"
                or conversations[i + 1]["role"] != "assistant"
            ):
                raise ValueError("The role order of the conversation is not correct")
            human = conversations[i]["content"].strip()
            assistant = conversations[i + 1]["content"].strip()

            human = self.user_format.format(
                content=human, stop_token=self.tokenizer.eos_token
            )
            assistant = self.assistant_format.format(
                content=assistant, stop_token=self.tokenizer.eos_token
            )

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)

        input_ids = input_ids[: self.max_seq_length]
        target_mask = target_mask[: self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
        }
        return inputs


class SFTDataCollator(object):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 找出batch中的最大长度
        lengths = [len(x["input_ids"]) for x in batch if x["input_ids"] is not None]
        # 取出batch中的最大长度，如果超过max_seq_length，则取max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        # truncate and padding
        for x in batch:
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"]
            target_mask = x["target_mask"]
            if input_ids is None:
                logger.info("some input_ids is None")
                continue
            padding_len = batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[: self.max_seq_length]
            attention_mask = attention_mask[: self.max_seq_length]
            target_mask = target_mask[: self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        # 将list转换为tensor，得到最终的的模型输入
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels,
        }
        return inputs
