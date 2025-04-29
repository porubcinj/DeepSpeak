from collections import defaultdict
from config import Config
import glob
import json
import logging
import os
import random
import re
import shutil

def split_raw_data(cfg: Config):
    split_dir = os.path.join(cfg.datasets_dir, cfg.split_dir)
    if not cfg.recreate_datasets and os.path.isdir(split_dir):
        logging.info(f"Reusing existing dataset split")
        return

    logging.info(f"Splitting raw dataset")
    shutil.rmtree(split_dir, ignore_errors=True)

    split_dirs = tuple(os.path.join(split_dir, d) for d in (cfg.train_dir, cfg.val_dir, cfg.test_dir))

    for d1 in split_dirs:
        for d2 in (cfg.groups_dir, cfg.messages_dir):
            os.makedirs(os.path.join(d1, d2), exist_ok=True)

    raw_dir = os.path.join(cfg.datasets_dir, cfg.raw_dir)
    filenames = glob.glob("*.txt", root_dir=raw_dir)

    group_to_parts = defaultdict(list)
    for filename in filenames:
        group_id = filename[1:filename.find(']')]
        group_to_parts[group_id].append(filename)

    def sort_key(filename: str):
        if filename.count('[') == 1:
            return 1
        else:
            part_section = filename[filename.find('t') + 2:]
            part_number = int(part_section[:part_section.find(']')])
            return part_number

    for group_id in group_to_parts:
        group_to_parts[group_id].sort(key=sort_key)

    username_pattern = re.compile(r"^((?::[a-z0-9]+(?:_[a-z0-9]+)*:|[^:]+)+): (?:.+)$")
    username_message_pattern = re.compile(r"^((?::[a-z0-9]+(?:_[a-z0-9]+)*:|[^:]+)+): (.+)$")

    # Exclude group_ids if len(usernames) would be > cfg.max_group_size
    logging.info(f"Excluding groups that are too large or too small")
    group_ids_to_exclude = set()
    for group_id, files in group_to_parts.items():
        usernames = set()

        for filename in files:
            with open(os.path.join(raw_dir, filename), mode="r", encoding="utf-8") as in_file:
                while line := in_file.readline().strip():
                    match = re.match(username_pattern, line)
                    if match:
                        username = str(match.group(1)).strip()
                        usernames.add(username)
                        if len(usernames) > cfg.max_group_size:
                            group_ids_to_exclude.add(group_id)
                            break
                if len(usernames) < 3:
                    group_ids_to_exclude.add(group_id)
                if group_id in group_ids_to_exclude:
                    break

    for group_id in group_ids_to_exclude:
        del group_to_parts[group_id]

    logging.info(f"Creating split datasets")
    num_test_groups = int(len(group_to_parts) * cfg.test_split)
    num_val_groups = int(len(group_to_parts) * cfg.val_split)

    test_groups = set(random.sample(tuple(group_to_parts.keys()), num_test_groups))
    val_groups = set(random.sample(tuple(group_to_parts.keys() - test_groups), num_val_groups))
    train_groups = group_to_parts.keys() - test_groups - val_groups

    for d, groups in zip(split_dirs, (train_groups, val_groups, test_groups)):
        logging.info(f"# groups: {len(groups)}")
        for idx, group_id in enumerate(groups):
            logging.info(f"Group {idx}")
            usernames = {}

            # Message TXTs
            with open(os.path.join(d, cfg.messages_dir, f"{idx}.txt"), mode="w", encoding="utf-8") as out_file:
                for filename in group_to_parts[group_id]:
                    with open(os.path.join(raw_dir, filename), mode="r", encoding="utf-8") as in_file:
                        while line := in_file.readline().strip():
                            match = re.match(username_message_pattern, line)
                            if not match:
                                print(f"File: {filename}\nLine: {line}")
                                raise
                            username = str(match.group(1)).strip()
                            message_text = str(match.group(2)).strip()
                            if username not in usernames:
                                usernames[username] = len(usernames)
                            out_file.write(f"{username}\n{message_text}\n")

            # Group JSONs
            with open(os.path.join(d, cfg.groups_dir, f"{idx}.json"), mode="w", encoding="utf-8") as out_file:
                json.dump(usernames, out_file, separators=(',', ':'))
