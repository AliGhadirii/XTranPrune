import os
import skimage
from skimage import io
import torch
from torchvision import transforms as T
import pandas as pd
import numpy as np


class SkinDataset:
    def __init__(
        self,
        root_dir,
        df=None,
        csv_file=None,
        is_tracked=False,
        transform=None,
        SimpleTransform=None,
        name="Fitz17k",
    ):
        """
        Args:
            df (DataFrame): The dataframe with annotations.
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            is_tracked (bool): Flag to track the used samples and apply augmentations only if the sample has been used before.
            transform (callable, optional): Optional transform with augmentation for samples that have been used before in the tracking process.
            SimpleTransform (callable, optional): Optional simple transform to be applied on a sample.
            name (string): Name of the dataset. Default is "Fitz17k". Options are "Fitz17k", "HIBA", and "PAD".
        """
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.name = name

        self.is_tracked = is_tracked
        if self.is_tracked:
            self.SimpleTransform = SimpleTransform
            self.used_samples = set()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.name == "Fitz17k":
            img_name = (
                os.path.join(
                    self.root_dir, str(self.df.loc[self.df.index[idx], "hasher"])
                )
                + ".jpg"
            )
            image = io.imread(img_name)
        elif self.name == "HIBA":
            img_name = (
                os.path.join(
                    self.root_dir, str(self.df.loc[self.df.index[idx], "hasher"])
                )
                + ".JPG"
            )
            image = io.imread(img_name)
        elif self.name == "PAD":
            img_name = os.path.join(
                self.root_dir, str(self.df.loc[self.df.index[idx], "hasher"])
            )
            image = io.imread(img_name)
            if image.shape[-1] == 4:
                image = skimage.img_as_ubyte(skimage.color.rgba2rgb(image))

        if len(image.shape) < 3:
            image = skimage.color.gray2rgb(image)

        hasher = self.df.loc[self.df.index[idx], "hasher"]
        fitzpatrick_scale = self.df.loc[
            self.df.index[idx], "fitzpatrick_scale"
        ]  # Range: [1, 6]
        fitzpatrick = self.df.loc[self.df.index[idx], "fitzpatrick"]  # Range: [0, 5]
        fitzpatrick_binary = self.df.loc[
            self.df.index[idx], "fitzpatrick_binary"
        ]  # Range: [0, 1]

        if self.is_tracked:
            if idx in self.used_samples:
                if self.transform:
                    image = self.transform(image)
            else:
                if self.SimpleTransform:
                    image = self.SimpleTransform(image)
                self.used_samples.add(idx)
        elif self.transform:
            image = self.transform(image)

        if self.name == "Fitz17k":
            high = self.df.loc[self.df.index[idx], "high"]
            mid = self.df.loc[self.df.index[idx], "mid"]
            low = self.df.loc[self.df.index[idx], "low"]
            binary = self.df.loc[self.df.index[idx], "binary"]

            sample = {
                "image": image,
                "high": high,
                "mid": mid,
                "low": low,
                "binary": binary,
                "hasher": hasher,
                "fitzpatrick_scale": fitzpatrick_scale,
                "fitzpatrick": fitzpatrick,
                "fitzpatrick_binary": fitzpatrick_binary,
            }
            return sample
        elif self.name == "HIBA":
            low = self.df.loc[self.df.index[idx], "low"]
            binary = self.df.loc[self.df.index[idx], "binary"]

            sample = {
                "image": image,
                "low": low,
                "binary": binary,
                "hasher": hasher,
                "fitzpatrick_scale": fitzpatrick_scale,
                "fitzpatrick": fitzpatrick,
                "fitzpatrick_binary": fitzpatrick_binary,
            }
            return sample
        elif self.name == "PAD":
            low = self.df.loc[self.df.index[idx], "low"]
            gender = self.df.loc[self.df.index[idx], "gender"]

            sample = {
                "image": image,
                "low": low,
                "hasher": hasher,
                "fitzpatrick_scale": fitzpatrick_scale,
                "fitzpatrick": fitzpatrick,
                "fitzpatrick_binary": fitzpatrick_binary,
                "gender": gender,
            }
            return sample


class EyeDataset:
    def __init__(
        self,
        root_dir,
        df=None,
        csv_file=None,
        is_tracked=False,
        transform=None,
        SimpleTransform=None,
        name="Fitz17k",
    ):
        """
        Args:
            df (DataFrame): The dataframe with annotations.
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            is_tracked (bool): Flag to track the used samples and apply augmentations only if the sample has been used before.
            transform (callable, optional): Optional transform with augmentation for samples that have been used before in the tracking process.
            SimpleTransform (callable, optional): Optional simple transform to be applied on a sample.
            name (string): Name of the dataset. Default is "Fitz17k". Options are "Fitz17k", "HIBA", and "PAD".
        """
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.name = name

        self.is_tracked = is_tracked
        if self.is_tracked:
            self.SimpleTransform = SimpleTransform
            self.used_samples = set()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        folder_mapping = {
            "training": "Training",
            "validation": "Validation",
            "test": "Test",
        }
        if self.name == "GF3300":
            file_path = os.path.join(
                self.root_dir,
                folder_mapping[self.df.loc[self.df.index[idx], "use"]],
                str(self.df.loc[self.df.index[idx], "filename"]),
            )
        else:
            raise ValueError("Invalid dataset name.")

        # Load the RNFLT data from the .npz file
        raw_data = np.load(file_path, allow_pickle=True)
        rnflt = raw_data["rnflt"]

        # Normalize the RNFLT data to the range [0, 255]
        rnflt_normalized = (
            (rnflt - np.min(rnflt)) / (np.max(rnflt) - np.min(rnflt)) * 255
        ).astype(np.uint8)

        # Convert the normalized RNFLT data to an RGB image by stacking it along the last dimension
        rnflt_rgb = np.stack([rnflt_normalized] * 3, axis=-1)

        # Convert to PIL Image
        image = T.ToPILImage()(rnflt_rgb)

        # Apply transformations
        if self.is_tracked:
            if idx in self.used_samples:
                if self.transform:
                    image = self.transform(image)
            else:
                if self.SimpleTransform:
                    image = self.SimpleTransform(image)
                self.used_samples.add(idx)
        elif self.transform:
            image = self.transform(image)

        sample = {
            "filename": self.df.loc[self.df.index[idx], "filename"],
            "image": image,
            "label": self.df.loc[self.df.index[idx], "glaucoma"],
            "age_binary": self.df.loc[self.df.index[idx], "age_binary"],
            "age_multi": self.df.loc[self.df.index[idx], "age_multi"],
            "gender_binary": self.df.loc[self.df.index[idx], "gender_binary"],
            "race_binary": self.df.loc[self.df.index[idx], "race_binary"],
        }

        return sample
