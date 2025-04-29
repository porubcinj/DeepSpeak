from config import Config
from transformers import BertTokenizer
import json
import logging
import os
import shutil
import torch

def save_samples(cfg: Config):
    samples_dir = os.path.join(cfg.datasets_dir, cfg.samples_dir)
    if cfg.recreate_samples:
        logging.info(f"Removing existing samples")
        shutil.rmtree(samples_dir, ignore_errors=True)

    samples_dirs = tuple(os.path.join(samples_dir, d) for d in (cfg.train_dir, cfg.val_dir, cfg.test_dir))

    for d in samples_dirs:
        os.makedirs(d, exist_ok=True)

    split_dir = os.path.join(cfg.datasets_dir, cfg.split_dir)
    split_dirs = tuple(os.path.join(split_dir, d) for d in (cfg.train_dir, cfg.val_dir, cfg.test_dir))

    groups_dirs = tuple(os.path.join(d, cfg.groups_dir) for d in split_dirs)
    messages_dirs = tuple(os.path.join(d, cfg.messages_dir) for d in split_dirs)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    logging.info(f"Creating/Validating samples")
    for d, groups_dir, messages_dir in zip(samples_dirs, groups_dirs, messages_dirs):
        group_metadata = {}
        group_metadata_path = os.path.join(d, cfg.meta_json)
        if os.path.isfile(group_metadata_path):
            with open(group_metadata_path, mode="r", encoding="utf-8") as in_file:
                group_metadata = json.load(in_file)

        sample_idx = 0
        group_txts = sorted(os.listdir(messages_dir), key=lambda x: int(os.path.splitext(x)[0]))
        logging.info(f"# groups: {len(group_txts)}")
        for group_txt in group_txts:
            group_id = int(os.path.splitext(group_txt)[0])
            logging.info(f"Group {group_id}")

            if not cfg.recreate_samples and group_id in group_metadata:
                sample_idx += group_metadata[group_id]
                continue

            logging.info(f"Loading group members")
            with open(os.path.join(groups_dir, f"{group_id}.json"), mode="r", encoding="utf-8") as in_file:
                usernames = json.load(in_file)

            samples_per_group = 0
            messages = []
            with open(os.path.join(messages_dir, group_txt), mode="r", encoding="utf-8") as in_file:
                username = in_file.readline().strip()
                message_text = in_file.readline().strip()

                member_id = usernames[username]
                messages.append(f"{member_id} {message_text}")

                while username := in_file.readline().strip():
                    label = usernames[username]

                    if label != member_id:
                        sample_path = os.path.join(d, f"{sample_idx}.pt")
                        if not os.path.isfile(sample_path):
                            encodings = []
                            max_length = cfg.max_context_length - 1

                            for message in reversed(messages):
                                if max_length <= 1:
                                    break

                                encoding = tokenizer.encode(message, add_special_tokens=False, truncation=True, max_length=max_length - 1) + [tokenizer.sep_token_id]
                                encodings.append(torch.tensor(encoding, dtype=torch.long))
                                max_length -= len(encoding)

                            encodings.append(torch.tensor((tokenizer.cls_token_id,), dtype=torch.long))

                            if max_length > 0:
                                padding = torch.full((max_length,), tokenizer.pad_token_id, dtype=torch.long)
                                encodings.insert(0, padding)

                            input_ids = torch.cat(tuple(reversed(encodings)))
                            attention_mask = input_ids != tokenizer.pad_token_id

                            member_mask = torch.tensor([
                                (idx < len(usernames)) and (idx != member_id)
                                for idx in range(cfg.max_group_size)
                            ], dtype=torch.bool)

                            sample = {
                                "input_ids": input_ids,
                                "attention_mask": attention_mask.float(),
                                "group_id": torch.tensor(group_id, dtype=torch.long),
                                "member_mask": member_mask,
                                "label": torch.tensor(label, dtype=torch.long),
                            }

                            torch.save(sample, sample_path)

                        samples_per_group += 1
                        sample_idx += 1
                        member_id = label

                        if samples_per_group == cfg.max_samples_per_group:
                            break

                    message_text = in_file.readline().strip()
                    messages.append(f"{label} {message_text}")

            group_metadata[group_id] = samples_per_group

        group_metadata["num_samples"] = sample_idx
        with open(os.path.join(d, cfg.meta_json), "w") as out_file:
            json.dump(group_metadata, out_file)
