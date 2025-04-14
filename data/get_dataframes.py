from config import Config
import numpy as np
import os
import pandas as pd

def get_dataframes(cfg: Config, rng: np.random.Generator):
    train_val_dir = os.path.join(cfg.datasets_dir, "train_val")
    train_dir = os.path.join(train_val_dir, "train")
    val_dir = os.path.join(train_val_dir, "val")
    test_dir = os.path.join(cfg.datasets_dir, "test")
    train_val_groups_csv_path = os.path.join(train_val_dir, cfg.groups_csv)
    train_messages_csv_path = os.path.join(train_dir, cfg.messages_csv)
    val_messages_csv_path = os.path.join(val_dir, cfg.messages_csv)
    test_groups_csv_path = os.path.join(test_dir, cfg.groups_csv)
    test_messages_csv_path = os.path.join(test_dir, cfg.messages_csv)
    csv_paths = (
        train_val_groups_csv_path,
        train_messages_csv_path,
        val_messages_csv_path,
        test_groups_csv_path,
        test_messages_csv_path,
    )
    if not cfg.recreate_datasets and all(os.path.exists(csv_path) for csv_path in csv_paths):
        # Load DataFrames
        train_val_groups_df = pd.read_csv(train_val_groups_csv_path)
        train_messages_df = pd.read_csv(train_messages_csv_path)
        val_messages_df = pd.read_csv(val_messages_csv_path)
        test_groups_df = pd.read_csv(test_groups_csv_path)
        test_messages_df = pd.read_csv(test_messages_csv_path)
        return train_val_groups_df, train_messages_df, val_messages_df, test_groups_df, test_messages_df

    # Validate raw dataset
    raw_dir = os.path.join(cfg.datasets_dir, cfg.raw_dir)
    groups_csv = os.path.join(raw_dir, cfg.groups_csv)
    messages_csv = os.path.join(raw_dir, cfg.messages_csv)

    # Check for missing values in messages_df
    groups_df = pd.read_csv(groups_csv)
    messages_df = pd.read_csv(messages_csv)
    assert not messages_df.isnull().values.any(), f"Missing values found in {messages_csv}"

    # Check groups_df contains all groups in messages_df
    groups_group_ids = groups_df["group_id"]
    messages_group_ids = messages_df["group_id"]
    unique_group_ids = messages_group_ids.unique()
    assert np.isin(unique_group_ids, groups_group_ids.unique(), assume_unique=True).all(), f"{groups_csv} is missing a group_id in {messages_csv}"

    # Check groups_df for missing values
    groups_df = groups_df[groups_group_ids.isin(unique_group_ids)]
    assert not groups_df.isnull().values.any(), f"Missing values found in {groups_csv}"

    # Enumerate member_ids
    assert np.isin(messages_df["member_id"].unique(), groups_df["member_id"].unique(), assume_unique=True).all(), f"{groups_csv} is missing a member_id in {messages_csv}"
    assert not groups_df.duplicated(subset=["group_id", "member_id"]).any(), f"Duplicate member_id found in {groups_csv}"
    groups_df["new_member_id"] = groups_df.groupby("group_id", sort=False).cumcount()
    member_id_map = groups_df.set_index(["group_id", "member_id"])["new_member_id"].to_dict()
    messages_df["member_id"] = messages_df.set_index(["group_id", "member_id"]).index.map(member_id_map)
    groups_df["member_id"] = groups_df["new_member_id"]
    groups_df = groups_df.drop(columns="new_member_id")

    # Generate labels
    messages_df = messages_df.sort_values(by=["group_id", "message_id"])
    messages_df["next_member_id"] = messages_df.groupby("group_id", sort=False)["member_id"].shift(-1)
    messages_df = messages_df.dropna(subset=["next_member_id"])
    messages_df["next_member_id"] = messages_df["next_member_id"].astype(int)
    assert len(messages_df) >= 1, f"No group_id with 2 or more messages found in {messages_csv}"

    # Split messages_df into val and test
    num_test_group_ids = int(len(unique_group_ids) * cfg.test_split)
    test_group_ids = rng.choice(unique_group_ids, size=num_test_group_ids, replace=False)
    messages_group_ids = messages_df["group_id"]
    messages_test_group_ids_mask = messages_group_ids.isin(test_group_ids)
    val_messages_df = messages_df[~messages_test_group_ids_mask]
    test_messages_df = messages_df[messages_test_group_ids_mask]

    train_rows = []
    for group_id in unique_group_ids:
        group_df = messages_df[messages_df["group_id"] == group_id]

        assert not np.any(np.diff(group_df["member_id"].to_numpy()) == 0), f"Consecutive messages with same member_id found in {messages_csv} for group {group_id}"

        message_ids = group_df["message_id"]
        num_messages = len(message_ids)
        assert num_messages >= 1, f"No messages found in {messages_csv} for group {group_id}"
        assert message_ids.is_unique, f"Duplicate message_id found in {messages_csv} for group {group_id}"
        assert message_ids.between(0, num_messages - 1).all(), f"Invalid message_id found in {messages_csv} for group {group_id}"

        is_test = np.isin(group_id, test_group_ids, assume_unique=True)
        if not is_test:
            num_train_messages = num_messages - int(num_messages * cfg.val_split)
            train_rows.append(val_messages_df[val_messages_df["group_id"] == group_id].iloc[:num_train_messages])

    train_messages_df = pd.concat(train_rows, ignore_index=True)
    assert len(train_messages_df) != len(val_messages_df), f"A positive validation split of {cfg.val_split} was specified, but 0 messages could be split into training dataset"

    # Enumerate group_ids
    train_val_group_id_map = {group_id: idx for idx, group_id in enumerate(val_messages_df["group_id"].unique())}
    test_group_id_map = {group_id: idx for idx, group_id in enumerate(test_messages_df["group_id"].unique())}

    # Create val and test groups_df with new group_id mapping
    train_val_groups_df = groups_df[groups_df["group_id"].isin(train_val_group_id_map.keys())].copy()
    test_groups_df = groups_df[groups_df["group_id"].isin(test_group_id_map.keys())].copy()
    train_val_groups_df.loc[:, "group_id"] = train_val_groups_df["group_id"].map(train_val_group_id_map)
    test_groups_df.loc[:, "group_id"] = test_groups_df["group_id"].map(test_group_id_map)
    train_messages_df.loc[:, "group_id"] = train_messages_df["group_id"].map(train_val_group_id_map)
    val_messages_df.loc[:, "group_id"] = val_messages_df["group_id"].map(train_val_group_id_map)
    test_messages_df.loc[:, "group_id"] = test_messages_df["group_id"].map(test_group_id_map)

    for csv_path in csv_paths:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    train_val_groups_df.to_csv(train_val_groups_csv_path, index=False)
    train_messages_df.to_csv(train_messages_csv_path, index=False)
    val_messages_df.to_csv(val_messages_csv_path, index=False)
    test_groups_df.to_csv(test_groups_csv_path, index=False)
    test_messages_df.to_csv(test_messages_csv_path, index=False)

    return train_val_groups_df, train_messages_df, val_messages_df, test_groups_df, test_messages_df
