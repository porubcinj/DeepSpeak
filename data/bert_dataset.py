from config import Config
from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import SupportsIndex
import torch

class DeepSpeakBertDataset(Dataset):
    def __init__(self, cfg: Config, groups_df: DataFrame, messages_df: DataFrame, tokenizer: BertTokenizer | None = None):
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer = tokenizer

        # Maps group_id -> member_id -> member_name
        self._members = tuple(
            tuple(
                str(member_name)
                for member_name in group["member_name"]
            ) for _, group in groups_df.groupby("group_id", sort=False)
        )

        # Maps group_id -> (((member_id, message_text), ...) for sample_id in group_sample_ids)
        group_message_samples = tuple(
            tuple(
                tuple(
                    (int(row[0]), str(row[1]))
                    for row in sample_df[["member_id", "message_text"]].values
                ) for _, sample_df in group_df.groupby("sample_id", sort=False)
            ) for _, group_df in messages_df.groupby("group_id", sort=False)
        )

        # Maps sample_idx -> (input_ids, attention_mask)
        self.samples = torch.empty((messages_df["sample_id"].iloc[-1] + 1, 2, cfg.max_context_length), dtype=torch.long)
        # Maps sample_idx -> group_id
        self.group_ids = torch.empty(len(self.samples), dtype=torch.long)
        # Maps sample_idx -> (cfg.max_group_size)
        self.member_masks = torch.empty((len(self.samples), cfg.max_group_size), dtype=torch.bool)
        # Maps sample_idx -> label
        self.labels = torch.empty(len(self.samples), dtype=torch.long)

        prev_group_num_samples = 0

        for group_id, group_df in messages_df.groupby("group_id", sort=False):
            if group_id > 0:
                prev_group_num_samples += len(group_message_samples[group_id - 1])

            for sample_id, sample_df in group_df.groupby("sample_id", sort=False):
                group_messages = tuple(message for sample in group_message_samples[group_id][:sample_id + 1 - prev_group_num_samples] for message in sample)
                group_members = self.members[group_id]
                last_sender_id = group_messages[-1][0]

                self.group_ids[sample_id] = group_id
                self.member_masks[sample_id] = torch.tensor([
                    (idx < len(group_members)) and (idx != last_sender_id)
                    for idx in range(cfg.max_group_size)
                ])
                self.labels[sample_id] = sample_df["next_member_id"].iloc[-1]

                encodings = []
                max_length = cfg.max_context_length - 1

                for member_id, message_text in reversed(group_messages):
                    if max_length <= 1:
                        break

                    encoding = self.tokenizer.encode(f"{member_id} {message_text}", add_special_tokens=False, truncation=True, max_length=max_length - 1) + [self.tokenizer.sep_token_id]
                    encodings.append(torch.tensor(encoding, dtype=torch.long))
                    max_length -= len(encoding)

                encodings.append(torch.tensor((self.tokenizer.cls_token_id,), dtype=torch.long))

                if max_length > 0:
                    padding = torch.full((max_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                    encodings.insert(0, padding)

                self.samples[sample_id, 0] = torch.cat(tuple(reversed(encodings)))
                self.samples[sample_id, 1] = self.samples[sample_id, 0] != self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: SupportsIndex):
        return (
            {
                "input_ids": self.samples[idx, 0],
                "attention_mask": self.samples[idx, 1],
                "group_id": self.group_ids[idx],
                "member_mask": self.member_masks[idx],
            },
            self.labels[idx],
        )

    @property
    def members(self):
        return self._members
