import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from .datasets import SkinDataset


def train_val_split_fitz17k(
    Generated_csv_path,
    holdout_set="random_holdout",
    level="high",
):
    """Performs train-validation split for the Fitzpatrick17k dataset"""

    df = pd.read_csv(Generated_csv_path)

    if holdout_set == "expert_select":
        df2 = df
        train = df2[df2.qc.isnull()]
        test = df2[df2.qc == "1 Diagnostic"]

    elif holdout_set == "random_holdout":
        if "fitzpatrick" in level:
            train, test, y_train, y_test = train_test_split(
                df,
                df["fitzpatrick"],
                test_size=0.2,
                random_state=64,
                stratify=df["fitzpatrick"],
            )
            print("INFO: train test split stratified by fitzpatrick column")
        else:
            train, test, y_train, y_test = train_test_split(
                df, df["low"], test_size=0.2, random_state=64, stratify=df["low"]
            )
            print("INFO: train test split stratified by low column")

    elif holdout_set == "dermaamin":  # train with b
        # only choose those skin conditions in both dermaamin and non dermaamin
        combo = set(df[df.url.str.contains("dermaamin") == True].label.unique()) & set(
            df[df.url.str.contains("dermaamin") == False].label.unique()
        )
        count_atla = (
            df.loc[df.url.str.contains("dermaamin") == False]
        ).label.value_counts()
        count_atla = count_atla.rename_axis("unique_values").reset_index(name="counts")
        combo = combo & set(
            (count_atla.loc[count_atla["counts"] >= 5])["unique_values"]
        )
        df = df[df.label.isin(combo)]
        # remove the class only has one sample
        df["low"] = df["label"].astype("category").cat.codes
        # train = df[df.image_path.str.contains("dermaamin") == False]
        # test = df[df.image_path.str.contains("dermaamin")]
        train_test = df[df.url.str.contains("dermaamin") == False]
        train, test, y_train, y_test = train_test_split(
            train_test,
            train_test["low"],
            test_size=0.2,
            random_state=4242,
            stratify=train_test["low"],
        )  #
        print(train["low"].nunique())
        print(test["low"].nunique())
        test2 = df[df.url.str.contains("dermaamin") == True]
    elif holdout_set == "br":  # train with a
        # only choose those skin conditions in both dermaamin and non dermaamin
        combo = set(df[df.url.str.contains("dermaamin") == True].label.unique()) & set(
            df[df.url.str.contains("dermaamin") == False].label.unique()
        )
        count_derm = (
            df.loc[df.url.str.contains("dermaamin") == True]
        ).label.value_counts()
        count_derm = count_derm.rename_axis("unique_values").reset_index(name="counts")
        combo = combo & set(
            (count_derm.loc[count_derm["counts"] >= 5])["unique_values"]
        )
        df = df[df.label.isin(combo)]
        df["low"] = df["label"].astype("category").cat.codes
        # train = df[df.image_path.str.contains("dermaamin")]
        # test = df[df.image_path.str.contains("dermaamin") == False]
        train_test = df[df.url.str.contains("dermaamin") == True]
        train, test, y_train, y_test = train_test_split(
            train_test,
            train_test["low"],
            test_size=0.2,
            random_state=4242,
            stratify=train_test["low"],
        )  #
        print(train["low"].nunique())
        print(test["low"].nunique())
        test2 = df[df.url.str.contains("dermaamin") == False]
    elif holdout_set == "a12":
        train = df[(df.fitzpatrick == 1) | (df.fitzpatrick == 2)]
        test = df[(df.fitzpatrick != 1) & (df.fitzpatrick != 2)]
        combo = set(train.label.unique()) & set(test.label.unique())
        print(combo)
        train = train[train.label.isin(combo)].reset_index()
        test = test[test.label.isin(combo)].reset_index()
        train["low"] = train["label"].astype("category").cat.codes
        test["low"] = test["label"].astype("category").cat.codes
    elif holdout_set == "a34":
        train = df[(df.fitzpatrick == 3) | (df.fitzpatrick == 4)]
        test = df[(df.fitzpatrick != 3) & (df.fitzpatrick != 4)]
        combo = set(train.label.unique()) & set(test.label.unique())
        train = train[train.label.isin(combo)].reset_index()
        test = test[test.label.isin(combo)].reset_index()
        train["low"] = train["label"].astype("category").cat.codes
        test["low"] = test["label"].astype("category").cat.codes
    elif holdout_set == "a56":
        train = df[(df.fitzpatrick == 5) | (df.fitzpatrick == 6)]
        test = df[(df.fitzpatrick != 5) & (df.fitzpatrick != 6)]
        combo = set(train.label.unique()) & set(test.label.unique())
        train = train[train.label.isin(combo)].reset_index()
        test = test[test.label.isin(combo)].reset_index()
        train["low"] = train["label"].astype("category").cat.codes
        test["low"] = test["label"].astype("category").cat.codes

    return train, test


def get_fitz17k_dataloaders(
    root_image_dir,
    Generated_csv_path,
    dataset_name="Fitz17k",
    level="high",
    fitz_filter=None,
    holdout_set="random_holdout",
    batch_size=64,
    num_workers=1,
):
    """Returns a dictionary of data loaders for the Fitzpatrick17k dataset, for the training, and validation sets."""

    train_df, val_df = train_val_split_fitz17k(
        Generated_csv_path, holdout_set=holdout_set, level=level
    )

    if fitz_filter is not None:
        train_df = train_df[train_df["fitzpatrick"] == fitz_filter]
        val_df = val_df[val_df["fitzpatrick"] == fitz_filter]

    dataset_sizes = {"train": train_df.shape[0], "val": val_df.shape[0]}
    print(dataset_sizes)

    num_classes = len(list(train_df[level].unique()))

    transformed_train = SkinDataset(
        df=train_df,
        root_dir=root_image_dir,
        name=dataset_name,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )
    transformed_val = SkinDataset(
        df=val_df,
        root_dir=root_image_dir,
        name=dataset_name,
        transform=transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )

    # Initiliaze samplers for imbalanced dataset
    class_sample_count = np.array(train_df[level].value_counts().sort_index())
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in train_df[level]])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(
        samples_weight.type("torch.DoubleTensor"), len(samples_weight), replacement=True
    )

    dataloaders = {
        "train": DataLoader(
            transformed_train,
            batch_size=batch_size,
            sampler=sampler,
            # drop_last = True,
            # shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            transformed_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    # Return corresponding loaders and dataset sizes
    return dataloaders, dataset_sizes, num_classes
