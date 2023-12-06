import os
import skimage
from skimage import io
import torch
import pandas as pd


class Fitz17kDataset:
    def __init__(
        self,
        root_dir,
        df=None,
        csv_file=None,
        transform=None,
    ):
        """
        Args:
            df (DataFrame): The dataframe with annotations.
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = (
            os.path.join(self.root_dir, str(self.df.loc[self.df.index[idx], "hasher"]))
            + ".jpg"
        )
        image = io.imread(img_name)

        if len(image.shape) < 3:
            image = skimage.color.gray2rgb(image)

        hasher = self.df.loc[self.df.index[idx], "hasher"]
        high = self.df.loc[self.df.index[idx], "high"]
        mid = self.df.loc[self.df.index[idx], "mid"]
        low = self.df.loc[self.df.index[idx], "low"]
        binary = self.df.loc[self.df.index[idx], "binary"]
        fitzpatrick_scale = self.df.loc[
            self.df.index[idx], "fitzpatrick_scale"
        ]  # Range: [1, 6]
        fitzpatrick = self.df.loc[self.df.index[idx], "fitzpatrick"]  # Range: [0, 5]
        fitzpatrick_binary = self.df.loc[
            self.df.index[idx], "fitzpatrick_binary"
        ]  # Range: [0, 1]

        if self.transform:
            image = self.transform(image)
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
