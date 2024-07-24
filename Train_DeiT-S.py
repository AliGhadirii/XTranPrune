import argparse
import yaml
import time
import os
import sys
import shutil
from pprint import pprint
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from Datasets.dataloaders import get_dataloaders
from Models.ViT_LRP import deit_small_patch16_224
from Utils.Misc_utils import set_seeds, LinearWarmup, Logger
from Utils.transformers_utils import get_params_groups
from Utils.Metrics import find_threshold, plot_metrics_training
from Evaluation import eval_model, eval_model_SABranch

import warnings
from sklearn.exceptions import UndefinedMetricWarning


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def flatten_multi(list_of_lists):
    flattened_list = []
    for sublist in list_of_lists:
        for item in sublist:
            flattened_list.append(item)
    return flattened_list


def train_model(
    dataloaders,
    dataset_sizes,
    num_classes,
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    model_name,
    config,
):
    since = time.time()

    start_epoch = 1
    best_f1 = 0
    best_loss = np.inf
    best_model = None

    best_model_path_f1 = os.path.join(
        config["output_folder_path"], f"{model_name}_BEST_F1.pth"
    )
    best_model_path_loss = os.path.join(
        config["output_folder_path"], f"{model_name}_BEST_loss.pth"
    )

    if os.path.isfile(best_model_path_f1):
        print("Resuming training from:", best_model_path_f1)
        checkpoint = torch.load(best_model_path_f1)

        model = checkpoint["model"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        leading_val_metrics = checkpoint["leading_val_metrics"]
        leading_epoch = checkpoint["leading_epoch"]
        start_epoch = leading_epoch + 1
        best_f1 = leading_val_metrics["F1_Mac"]
        return model

    for epoch in range(start_epoch, config["train"]["n_epochs"]):
        since_epoch = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            # Set the model to the training mode
            if phase == "train":
                model.train()
            # Set model to the evaluation mode
            else:
                model.eval()

            # Running parameters

            running_loss = 0.0
            all_preds = []
            all_labels = []
            all_probs = []

            for idx, batch in tqdm(
                enumerate(dataloaders[phase]),
                total=len(dataloaders[phase]),
                desc="Phase {} | Epoch {}/{}".format(
                    phase, epoch, config["train"]["n_epochs"]
                ),
            ):

                inputs = batch["image"].to(device)
                labels = batch[config["train"]["main_level"]]

                if num_classes == 2:
                    labels = (
                        torch.from_numpy(np.asarray(labels)).unsqueeze(1).to(device)
                    )
                else:
                    labels = torch.from_numpy(np.asarray(labels)).to(device)

                # Zero the gradients
                optimizer.zero_grad()

                def handle_warning(
                    message, category, filename, lineno, file=None, line=None
                ):
                    print("Warning:", message)
                    print("Additional Information:")
                    print(f"Phase: {phase}, batch idx: {idx}")

                # Filter warnings to catch the specific warning types
                warnings.filterwarnings("always", category=UndefinedMetricWarning)

                # Set the custom function to handle the warning
                warnings.showwarning = handle_warning

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    inputs = inputs.float()
                    outputs = model(inputs)

                    if num_classes == 2:
                        probs = nn.functional.sigmoid(outputs)
                        theshold = find_threshold(
                            probs.cpu().data.numpy(), labels.cpu().data.numpy()
                        )
                        preds = (probs > theshold).to(torch.int32)

                        loss = criterion(outputs, labels.to(torch.float32))
                    else:
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        logits, preds = torch.max(outputs, 1)

                        loss = criterion(outputs, labels)

                    # Backward + optimize only if in the training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().tolist())

            if phase == "train":
                scheduler.step()

            # metrics
            epoch_stats = {}
            epoch_stats["loss"] = running_loss / dataset_sizes[phase]
            epoch_stats["accuracy"] = accuracy_score(all_labels, all_preds) * 100
            epoch_stats["F1_Mac"] = (
                f1_score(all_labels, all_preds, average="macro") * 100
            )
            if num_classes == 2:
                all_probs = flatten(all_probs)
                epoch_stats["AUC"] = roc_auc_score(all_labels, all_probs) * 100
            else:
                all_probs = flatten_multi(all_probs)
                epoch_stats["AUC"] = (
                    roc_auc_score(
                        all_labels, all_probs, average="macro", multi_class="ovo"
                    )
                    * 100
                )

            if phase == "train":
                if epoch == 1:
                    train_metrics_df = pd.DataFrame([epoch_stats])
                else:
                    train_metrics_df = pd.concat(
                        [train_metrics_df, pd.DataFrame([epoch_stats])],
                        ignore_index=True,
                    )
                train_metrics_df.to_csv(
                    os.path.join(config["output_folder_path"], f"Train_metrics.csv"),
                    index=False,
                )
            else:
                if epoch == 1:
                    val_metrics_df = pd.DataFrame([epoch_stats])
                else:
                    val_metrics_df = pd.concat(
                        [val_metrics_df, pd.DataFrame([epoch_stats])], ignore_index=True
                    )
                val_metrics_df.to_csv(
                    os.path.join(
                        config["output_folder_path"], f"Validation_metrics.csv"
                    ),
                    index=False,
                )

            # Print
            print(
                "{} Loss: {:.4f} Acc: {:.4f} Macro f1-score: {:.4f} AUC: {:.4f}".format(
                    phase,
                    epoch_stats["loss"],
                    epoch_stats["accuracy"],
                    epoch_stats["F1_Mac"],
                    epoch_stats["AUC"],
                )
            )

            ################################ TOBE FIXED ########################
            if phase == "val" and epoch_stats["F1_Mac"] > best_f1:
                print("New leading Macro f1-score: {}".format(epoch_stats["F1_Mac"]))

                # Save checkpoint
                checkpoint = {
                    "leading_epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                    "leading_val_metrics": epoch_stats,
                    "model": model,
                }
                best_model = model
                best_f1 = epoch_stats["F1_Mac"]
                torch.save(checkpoint, best_model_path_f1)
                print("Checkpoint saved:", best_model_path_f1)
            elif phase == "val" and epoch_stats["loss"] < best_loss:
                print("New leading loss: {}".format(epoch_stats["loss"]))

                # Save checkpoint
                checkpoint = {
                    "leading_epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                    "leading_val_metrics": epoch_stats,
                    "model": model,
                }
                best_loss = epoch_stats["loss"]
                torch.save(checkpoint, best_model_path_loss)
                print("Checkpoint saved:", best_model_path_loss)

        time_elapsed_epoch = time.time() - since_epoch
        print(
            "Epoch {} completed in {:.0f}m {:.0f}s".format(
                epoch, time_elapsed_epoch // 60, time_elapsed_epoch % 60
            )
        )

        plot_metrics_training(
            train_metrics_df, val_metrics_df, ["loss"], "loss", config
        )
        plot_metrics_training(
            train_metrics_df, val_metrics_df, ["F1_Mac"], "F1", config
        )
        plot_metrics_training(
            train_metrics_df, val_metrics_df, ["accuracy"], "accuracy", config
        )
        plot_metrics_training(train_metrics_df, val_metrics_df, ["AUC"], "AUC", config)

    # Time
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    return best_model


def main(config):
    log_file_path = os.path.join(config["output_folder_path"], "output.log")

    sys.stdout = Logger(log_file_path)

    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Configs:")
    pprint(config)
    print()

    shutil.copy(
        args.config,
        os.path.join(config["output_folder_path"], "configs.yml"),
    )

    set_seeds(config["seed"])

    model_name = f"DiT_S_level={config['train']['main_level']}"

    if config["dataset_name"] in ["Fitz17k", "HIBA", "PAD"]:
        dataloaders, dataset_sizes, main_num_classes, SA_num_classes = get_dataloaders(
            root_image_dir=config["root_image_dir"],
            Generated_csv_path=config["Generated_csv_path"],
            sampler_type="WeightedRandom",
            stratify_cols=["low"],
            dataset_name=config["dataset_name"],
            main_level=config["train"]["main_level"],
            SA_level=config["train"]["SA_level"],
            batch_size=config["train"]["batch_size"],
            num_workers=1,
        )
    elif config["dataset_name"] in ["GF3300"]:
        dataloaders, dataset_sizes, main_num_classes, SA_num_classes = get_dataloaders(
            root_image_dir=config["root_image_dir"],
            train_csv_path=config["train_csv_path"],
            val_csv_path=config["val_csv_path"],
            sampler_type="WeightedRandom",
            dataset_name=config["dataset_name"],
            main_level=config["train"]["main_level"],
            SA_level=config["train"]["SA_level"],
            batch_size=config["train"]["batch_size"],
            num_workers=1,
        )
    else:
        raise ValueError("Invalid dataset name")
    num_classes = (
        main_num_classes if config["train"]["branch"] == "main" else SA_num_classes
    )
    model = deit_small_patch16_224(
        pretrained=config["train"]["pretrained"],
        num_classes=num_classes,
        add_hook=True,
        need_ig=False,
    )
    model = model.to(device)
    print(model)

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if config["dataset_name"] in ["Fitz17k", "HIBA", "PAD"]:
        optimizer = optim.Adam(get_params_groups(model), lr=1e-4, weight_decay=1e-5)
        scheduler = LinearWarmup(
            optimizer,
            max_lr=1e-4,
            eta_min=1e-6,
            warmup_epochs=0,
            warmup_iters=1000,
            steps_per_epoch=len(dataloaders["train"]),
        )
    elif config["dataset_name"] in ["GF3300"]:

        lr_warmup_epochs = 10
        lr = 1e-4
        lr_min = 1e-6
        weight_decay = 1e-4
        lr_warmup_decay = 0.01

        optimizer = torch.optim.AdamW(
            get_params_groups(model), lr=lr, weight_decay=weight_decay
        )
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["train"]["n_epochs"] - lr_warmup_epochs,
            eta_min=lr_min,
        )

        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=lr_warmup_decay,
            total_iters=lr_warmup_epochs,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[lr_warmup_epochs],
        )
    else:
        raise ValueError("Invalid dataset name")

    best_model = train_model(
        dataloaders,
        dataset_sizes,
        num_classes,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        model_name,
        config,
    )

    if config["train"]["branch"] == "main":
        val_metrics, _ = eval_model(
            model=best_model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_classes=num_classes,
            device=device,
            level=config["train"]["main_level"],
            model_type=model_name,
            config=config,
            save_preds=True,
        )
    elif config["train"]["branch"] == "SA":
        val_metrics, _ = eval_model_SABranch(
            model=best_model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=device,
            model_type=model_name,
            config=config,
            save_preds=True,
        )
    else:
        raise ValueError("Invalid branch type")

    print("Best validation metrics:")
    pprint(val_metrics)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
