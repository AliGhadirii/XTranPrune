import argparse
import yaml
import time
import os
import sys
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import balanced_accuracy_score

from Datasets.dataloaders import get_dataloaders
from Models.ViT_LRP import deit_small_patch16_224
from Utils.Misc_utils import set_seeds, LinearWarmup, Logger
from Utils.transformers_utils import get_params_groups
from Utils.Metrics import find_threshold
from Evaluation import eval_model_SABranch

import warnings
from sklearn.exceptions import UndefinedMetricWarning


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

    training_results = []
    validation_results = []
    start_epoch = 0
    best_acc = 0

    best_model_path = os.path.join(
        config["output_folder_path"], f"{model_name}_checkpoint.pth"
    )

    if os.path.isfile(best_model_path):
        print("Resuming training from:", best_model_path)
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_loss = checkpoint["best_loss"]
        best_acc = checkpoint["best_acc"]
        best_balanced_acc = checkpoint["best_balanced_acc"]
        leading_epoch = checkpoint["leading_epoch"]
        start_epoch = leading_epoch + 1

    for epoch in range(start_epoch, config["default"]["n_epochs"]):
        print("Epoch {}/{}".format(epoch, config["default"]["n_epochs"] - 1))
        print("-" * 20)
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
            running_corrects = 0
            running_balanced_acc_sum = 0
            cnt = 0

            print(f"Current phase: {phase}")
            # Iterate over data
            for idx, batch in enumerate(dataloaders[phase]):
                # Send inputs and labels to the device
                inputs = batch["image"].to(device)
                labels = batch[config["default"]["level"]]

                labels = torch.from_numpy(np.asarray(labels)).unsqueeze(1).to(device)

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

                    probs = nn.functional.sigmoid(outputs)
                    theshold = find_threshold(
                        probs.cpu().data.numpy(), labels.cpu().data.numpy()
                    )
                    preds = (probs > theshold).to(torch.int32)

                    loss = criterion(outputs, labels.to(torch.float32))

                    # Backward + optimize only if in the training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_balanced_acc_sum += balanced_accuracy_score(
                    labels.data.cpu(), preds.cpu()
                ) * inputs.size(0)

                # Increment
                cnt = cnt + 1

            if phase == "train":
                scheduler.step()

            # metrics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_balanced_acc = running_balanced_acc_sum / dataset_sizes[phase]

            # Print
            print(
                "{} Loss: {:.4f} Acc: {:.4f} Balanced Accuracy: {:.4f} ".format(
                    phase, epoch_loss, epoch_acc, epoch_balanced_acc
                )
            )

            # Save the accuracy and loss
            if phase == "train":
                training_results.append(
                    [phase, epoch, epoch_loss, epoch_acc.item(), epoch_balanced_acc]
                )
            elif phase == "val":
                validation_results.append(
                    [phase, epoch, epoch_loss, epoch_acc.item(), epoch_balanced_acc]
                )

            # Deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                print("New leading accuracy: {}".format(epoch_acc))
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_balanced_acc = epoch_balanced_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # Save checkpoint
                checkpoint = {
                    "leading_epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "best_acc": best_acc,
                    "best_balanced_acc": best_balanced_acc,
                    "config": config,
                }
                torch.save(checkpoint, best_model_path)
                print("Checkpoint saved:", best_model_path)

        time_elapsed_epoch = time.time() - since_epoch
        print(
            "Epoch {} completed in {:.0f}m {:.0f}s".format(
                epoch, time_elapsed_epoch // 60, time_elapsed_epoch % 60
            )
        )

    # Time
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Loss: {:4f}".format(best_loss))
    print("Best val Acc: {:4f}".format(best_acc))
    print("Best val Balanced Acc: {:4f}".format(best_balanced_acc))

    # Load the best model weights
    model.load_state_dict(best_model_wts)

    training_results = pd.DataFrame(training_results)
    training_results.columns = [
        "phase",
        "epoch",
        "loss",
        "accuracy",
        "balanced_accuracy",
    ]

    validation_results = pd.DataFrame(validation_results)
    validation_results.columns = [
        "phase",
        "epoch",
        "loss",
        "accuracy",
        "balanced_accuracy",
    ]

    return model, training_results, validation_results


def main(config):
    log_file_path = os.path.join(config["output_folder_path"], "output.log")

    sys.stdout = Logger(log_file_path)

    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seeds(config["seed"])

    model_name = f"DiT_S_LRP_level={config['default']['level']}"

    dataloaders, dataset_sizes, num_classes, _ = get_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        dataset_name=config["dataset_name"],
        main_level=config["default"]["level"],
        batch_size=config["default"]["batch_size"],
        num_workers=1,
    )

    model = deit_small_patch16_224(
        pretrained=config["default"]["pretrained"],
        num_classes=num_classes,
        add_hook=False,
    )
    model = model.to(device)
    print(model)

    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(get_params_groups(model), lr=1e-4, weight_decay=1e-5)
    scheduler = LinearWarmup(
        optimizer,
        max_lr=1e-4,
        eta_min=1e-6,
        warmup_epochs=0,
        warmup_iters=1000,
        steps_per_epoch=len(dataloaders["train"]),
    )

    if config["default"]["mode"] == "eval":
        best_model_path = os.path.join(
            config["output_folder_path"], f"{model_name}_checkpoint_BASE.pth"
        )
        print("EVAL MODE: Loading the weights from ", best_model_path)

        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_loss = checkpoint["best_loss"]
        best_acc = checkpoint["best_acc"]
        best_balanced_acc = checkpoint["best_balanced_acc"]
        leading_epoch = checkpoint["leading_epoch"]

        print(
            "Best epoch {}: Loss: {:.4f} Acc: {:.4f} Balanced Accuracy: {:.4f} ".format(
                leading_epoch, best_loss, best_acc, best_balanced_acc
            )
        )
    else:
        shutil.copy(
            "Configs/configs_server.yml",
            os.path.join(config["output_folder_path"], "configs.yml"),
        )

        model, training_results, validation_results = train_model(
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

        training_results.to_csv(
            os.path.join(
                config["output_folder_path"],
                f"Training_log_{model_name}_random_holdout.csv",
            ),
            index=False,
        )
        validation_results.to_csv(
            os.path.join(
                config["output_folder_path"],
                f"Validation_log_{model_name}_random_holdout.csv",
            ),
            index=False,
        )

    val_metrics, _ = eval_model_SABranch(
        model,
        dataloaders,
        dataset_sizes,
        device,
        model_name,
        config,
        save_preds=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)
    main(config)
