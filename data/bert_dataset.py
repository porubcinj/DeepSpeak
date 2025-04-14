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

        # Maps group_id -> message_id -> (member_id, message_text)
        messages = tuple(
            tuple(
                (int(row[0]), str(row[1]))
                for row in group[["member_id", "message_text"]].values
            ) for _, group in messages_df.groupby("group_id", sort=False)
        )

        # Maps sample_idx -> (input_ids, attention_mask)
        self.samples = torch.empty((len(messages_df), 2, cfg.max_context_length), dtype=torch.long)
        # Maps sample_idx -> group_id
        self.group_ids = torch.empty(len(messages_df), dtype=torch.long)
        # Maps sample_idx -> (cfg.max_group_size)
        self.member_masks = torch.empty((len(messages_df), cfg.max_group_size), dtype=torch.bool)
        # Maps sample_idx -> label
        self.labels = torch.empty(len(messages_df), dtype=torch.long)

        for i, (group_id, message_id, next_member_id) in enumerate(messages_df[["group_id", "message_id", "next_member_id"]].values):
            group_messages = messages[group_id][:message_id + 1]
            group_members = self.members[group_id]
            last_sender_id = group_messages[-1][0]

            self.group_ids[i] = group_id
            self.member_masks[i] = torch.tensor([
                (idx < len(group_members)) and (idx != last_sender_id)
                for idx in range(cfg.max_group_size)
            ])
            self.labels[i] = next_member_id

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

            self.samples[i, 0] = torch.cat(tuple(reversed(encodings)))
            self.samples[i, 1] = self.samples[i, 0] != self.tokenizer.pad_token_id

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
