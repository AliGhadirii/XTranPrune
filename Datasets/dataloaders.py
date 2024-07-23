import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

from .datasets import SkinDataset, EyeDataset
from Utils.Misc_utils import StratifiedSampler, CustomStratifiedSampler

import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode


def train_val_split(
    Generated_csv_path,
    stratify_cols=["low"],
):
    """Performs train-validation split"""

    df = pd.read_csv(Generated_csv_path)
    df["stratify"] = df[stratify_cols].astype(str).agg("_".join, axis=1)
    if len(stratify_cols) == 1:
        df["stratify"] = df["stratify"].astype(int)

    train, test, y_train, y_test = train_test_split(
        df, df["low"], test_size=0.2, random_state=64, stratify=df["stratify"]
    )

    print(f"INFO: train test split stratified by {stratify_cols} column(s).")

    return train, test


def get_dataloaders(
    root_image_dir,
    Generated_csv_path,
    sampler_type="WeightedRandom",
    dataset_name="Fitz17k",
    stratify_cols=["low"],
    main_level="high",
    SA_level="fitzpatrick_binary",
    batch_size=64,
    num_workers=1,
):

    train_df, val_df = train_val_split(Generated_csv_path, stratify_cols=stratify_cols)

    dataset_sizes = {"train": train_df.shape[0], "val": val_df.shape[0]}
    print(dataset_sizes)

    main_num_classes = len(list(train_df[main_level].unique()))
    SA_num_classes = len(list(train_df[SA_level].unique()))

    # Transforms
    EyeTrainTransform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    EyeValTransform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    SkinTrainTransform = transforms.Compose(
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
    )

    SkinValTransform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Initiliaze samplers for imbalanced dataset
    if dataset_name in ["Fitz17k", "HIBA", "PAD"]:
        if sampler_type == "WeightedRandom":
            print("INFO: Using WeightedRandomSampler\n")
            class_sample_count = np.array(
                train_df[main_level].value_counts().sort_index()
            )
            weight = 1.0 / class_sample_count
            samples_weight = np.array([weight[t] for t in train_df[main_level]])

            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(
                samples_weight.type("torch.DoubleTensor"),
                len(samples_weight),
                replacement=True,
            )
            transformed_train = SkinDataset(
                df=train_df,
                root_dir=root_image_dir,
                name=dataset_name,
                transform=SkinTrainTransform,
            )

        elif sampler_type == "Stratified":
            print("INFO: Using StratifiedSampler\n")
            sampler = StratifiedSampler(train_df)

            transformed_train = SkinDataset(
                df=train_df,
                root_dir=root_image_dir,
                name=dataset_name,
                transform=SkinTrainTransform,
            )
        elif sampler_type == "CustomStratified":
            print("INFO: Using CustomStratifiedSampler\n")
            sampler = CustomStratifiedSampler(
                df=train_df,
                label_col=main_level,
                sensitive_attr_col=SA_level,
                batch_size=batch_size,
            )

            transformed_train = SkinDataset(
                df=train_df,
                root_dir=root_image_dir,
                name=dataset_name,
                is_tracked=True,
                transform=SkinTrainTransform,
                SimpleTransform=SkinValTransform,
            )

        else:
            raise ValueError("Invalid sampler type")

        transformed_val = SkinDataset(
            df=val_df,
            root_dir=root_image_dir,
            name=dataset_name,
            transform=SkinValTransform,
        )

    elif dataset_name == "GF3300":
        if sampler_type == "WeightedRandom":
            print("INFO: Using WeightedRandomSampler\n")
            class_sample_count = np.array(
                train_df[main_level].value_counts().sort_index()
            )
            weight = 1.0 / class_sample_count
            samples_weight = np.array([weight[t] for t in train_df[main_level]])

            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(
                samples_weight.type("torch.DoubleTensor"),
                len(samples_weight),
                replacement=True,
            )
            transformed_train = EyeDataset(
                df=train_df,
                root_dir=root_image_dir,
                name=dataset_name,
                transform=EyeTrainTransform,
            )

        elif sampler_type == "Stratified":
            print("INFO: Using StratifiedSampler\n")
            sampler = StratifiedSampler(train_df)

            transformed_train = EyeDataset(
                df=train_df,
                root_dir=root_image_dir,
                name=dataset_name,
                transform=EyeTrainTransform,
            )
        elif sampler_type == "CustomStratified":
            print("INFO: Using CustomStratifiedSampler\n")
            sampler = CustomStratifiedSampler(
                df=train_df,
                label_col=main_level,
                sensitive_attr_col=SA_level,
                batch_size=batch_size,
            )

            transformed_train = EyeDataset(
                df=train_df,
                root_dir=root_image_dir,
                name=dataset_name,
                is_tracked=True,
                transform=EyeTrainTransform,
                SimpleTransform=EyeValTransform,
            )

        else:
            raise ValueError("Invalid sampler type")

        transformed_val = EyeDataset(
            df=val_df,
            root_dir=root_image_dir,
            name=dataset_name,
            transform=EyeValTransform,
        )
    else:
        raise ValueError("Invalid dataset name")

    dataloaders = {
        "train": DataLoader(
            transformed_train,
            batch_size=batch_size,
            sampler=sampler,
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
    return dataloaders, dataset_sizes, main_num_classes, SA_num_classes
