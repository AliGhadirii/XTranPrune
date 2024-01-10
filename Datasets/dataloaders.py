import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from .datasets import SkinDataset


def train_val_split(
    Generated_csv_path,
    level="high",
):
    """Performs train-validation split"""

    df = pd.read_csv(Generated_csv_path)

    if "fitzpatrick" in level:
        train, test, y_train, y_test = train_test_split(
            df,
            df["fitzpatrick"],
            test_size=0.2,
            random_state=64,
            stratify=df["fitzpatrick"],
        )
        print(
            f"INFO: train test split stratified by fitzpatrick column because the level is {level}"
        )
    else:
        train, test, y_train, y_test = train_test_split(
            df, df["low"], test_size=0.2, random_state=64, stratify=df["low"]
        )
        print(
            f"INFO: train test split stratified by low column because the level is {level}"
        )

    return train, test


def get_dataloaders(
    root_image_dir,
    Generated_csv_path,
    dataset_name="Fitz17k",
    level="high",
    fitz_filter=None,
    batch_size=64,
    num_workers=1,
):
    """Returns a dictionary of data loaders for the Fitzpatrick17k dataset, for the training, and validation sets."""

    train_df, val_df = train_val_split(Generated_csv_path, level=level)

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
