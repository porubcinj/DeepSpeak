from config import Config
from torch.utils.data import Dataset
from typing import SupportsIndex
import json
import os
import torch

class DeepSpeakBertDataset(Dataset):
    def __init__(self, cfg: Config, data_dir: str):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, cfg.meta_json), "r") as f:
            meta = json.load(f)
        self.num_samples = meta["num_samples"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: SupportsIndex):
        sample = torch.load(os.path.join(self.data_dir, f"{idx}.pt"))
        return (
            {
                k: sample[k]
                for k in ("input_ids", "attention_mask", "group_id", "member_mask")
            },
            sample["label"],
        )
