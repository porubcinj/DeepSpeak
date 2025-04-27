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
    df = messages_df[messages_df.isnull().any(axis=1)]
    assert df.empty, f"Missing values found in {messages_csv}: {df.iloc[0]}"

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

    unique_group_ids = messages_df["group_id"].unique()
    num_test_group_ids = int(len(unique_group_ids) * cfg.test_split)
    test_group_ids = rng.choice(unique_group_ids, size=num_test_group_ids, replace=False)

    val_messages_df = messages_df[~messages_df["group_id"].isin(test_group_ids)].copy().reset_index(drop=True)
    test_messages_df = messages_df[messages_df["group_id"].isin(test_group_ids)].copy().reset_index(drop=True)

    # Enumerate group_ids
    train_val_group_id_map = {old_id: new_id for new_id, old_id in enumerate(val_messages_df["group_id"].unique())}
    test_group_id_map = {old_id: new_id for new_id, old_id in enumerate(test_group_ids)}

    val_messages_df["group_id"] = val_messages_df["group_id"].map(train_val_group_id_map)
    test_messages_df["group_id"] = test_messages_df["group_id"].map(test_group_id_map)

    val_messages_df["sample_id"] = (
        (val_messages_df["member_id"] != val_messages_df["member_id"].shift()) |
        (val_messages_df["group_id"] != val_messages_df["group_id"].shift())
    )
    val_messages_df["sample_id"] = val_messages_df.groupby("group_id", sort=False)["sample_id"].cumsum() - 1

    test_messages_df["sample_id"] = (
        (test_messages_df["member_id"] != test_messages_df["member_id"].shift()) |
        (test_messages_df["group_id"] != test_messages_df["group_id"].shift())
    ).cumsum() - 1
    assert len(test_messages_df) >= 1, f"No group_id with 2 or more messages found in {messages_csv}"

    train_messages = []

    for _, v in val_messages_df.groupby("group_id", sort=False):
        v = v.reset_index(drop=True)
        unique_sample_ids = v["sample_id"].unique()
        num_train_samples = len(unique_sample_ids) - int(len(unique_sample_ids) * cfg.val_split)

        if num_train_samples > 0:
            v_train = v[v["sample_id"].isin(unique_sample_ids[:num_train_samples])]
            train_messages.append(v_train)

    train_messages_df = pd.concat(train_messages, ignore_index=True)

    # Generate labels
    train_messages_df["sample_id"] = (
        (train_messages_df["member_id"] != train_messages_df["member_id"].shift()) |
        (train_messages_df["group_id"] != train_messages_df["group_id"].shift())
    )
    train_messages_df["sample_id"] = train_messages_df.groupby("group_id", sort=False)["sample_id"].cumsum() - 1
    assert len(train_messages_df) >= 1, f"No group_id with 2 or more messages found in {messages_csv}"

    val_messages_df["sample_id"] = (
        (val_messages_df["member_id"] != val_messages_df["member_id"].shift()) |
        (val_messages_df["group_id"] != val_messages_df["group_id"].shift())
    )
    val_messages_df["sample_id"] = val_messages_df.groupby("group_id", sort=False)["sample_id"].cumsum() - 1

    train_counts = train_messages_df.groupby("group_id", sort=False)["sample_id"].last() + 1

    val_messages_df["sample_id"] -= val_messages_df["group_id"].map(train_counts).fillna(0).astype(int)
    val_messages_df["sample_id"] = val_messages_df["sample_id"].clip(lower=0).cumsum()
    val_messages_df["sample_id"] += val_messages_df["group_id"]
    val_messages_df["sample_id"] = pd.factorize(val_messages_df["sample_id"])[0]

    assert len(val_messages_df) >= 1, f"No group_id with 2 or more messages found in {messages_csv}"

    train_messages_df["sample_id"] = (
        (train_messages_df["member_id"] != train_messages_df["member_id"].shift()) |
        (train_messages_df["group_id"] != train_messages_df["group_id"].shift())
    ).cumsum() - 1

    # Create val and test groups_df with new group_id mapping
    train_val_groups_df = groups_df[groups_df["group_id"].isin(train_val_group_id_map.keys())].copy().reset_index(drop=True)
    test_groups_df = groups_df[groups_df["group_id"].isin(test_group_id_map.keys())].copy().reset_index(drop=True)
    train_val_groups_df.loc[:, "group_id"] = train_val_groups_df["group_id"].map(train_val_group_id_map)
    test_groups_df.loc[:, "group_id"] = test_groups_df["group_id"].map(test_group_id_map)

    for csv_path in csv_paths:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    train_val_groups_df.to_csv(train_val_groups_csv_path, index=False)
    train_messages_df.to_csv(train_messages_csv_path, index=False)
    val_messages_df.to_csv(val_messages_csv_path, index=False)
    test_groups_df.to_csv(test_groups_csv_path, index=False)
    test_messages_df.to_csv(test_messages_csv_path, index=False)

    return train_val_groups_df, train_messages_df, val_messages_df, test_groups_df, test_messages_df
