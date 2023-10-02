import argparse
import yaml
import time
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from sklearn.metrics import balanced_accuracy_score

from Datasets.dataloaders import get_fitz17k_dataloaders
from Utils.Misc_utils import set_seeds


def main(config):
    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seeds(config["seed"])

    dataloaders, dataset_sizes, num_classes = get_fitz17k_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        level=config["default"]["level"],
        binary_subgroup=config["default"]["binary_subgroup"],
        holdout_set="random_holdout",
        batch_size=config["default"]["batch_size"],
        num_workers=1,
    )

    for batch in dataloaders["train"]:
        # Send inputs and labels to the device
        inputs = batch["image"]
        labels = batch["high"]
        attrs = batch["fitzpatrick"]

        print(f"inputs: {inputs.shape}")
        print(f"labels: {labels}")
        print(f"attrs: {attrs}")
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
