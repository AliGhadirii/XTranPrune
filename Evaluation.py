import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from pprint import pprint
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    classification_report,
)

from Utils.Metrics import cal_metrics_skin, cal_metrics_eye, find_threshold
from Utils.Misc_utils import set_seeds, Logger, get_stat, get_mask_idx
from Datasets.dataloaders import get_dataloaders
from Models.ViT_LRP import deit_small_patch16_224


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def flatten_all_prob(list_of_lists):
    flattened_list = []
    for sublist in list_of_lists:
        for item in sublist:
            flattened_list.append(item)
    return flattened_list


def eval_model(
    model,
    dataloaders,
    dataset_sizes,
    num_classes,
    device,
    level,
    model_type,
    config,
    save_preds=False,
):
    model = model.eval()
    prediction_list = []
    fitzpatrick_list = []
    fitzpatrick_binary_list = []
    fitzpatrick_scale_list = []
    hasher_list = []
    labels_list = []
    p_list = []
    all_p_list = []
    SA_list = []

    with torch.no_grad():
        total = 0

        for batch in dataloaders["val"]:
            inputs = batch["image"].to(device)
            classes = batch[level]

            if num_classes == 2:
                classes = torch.from_numpy(np.asarray(classes)).unsqueeze(1).to(device)
            else:
                classes = torch.from_numpy(np.asarray(classes)).to(device)

            if config["dataset_name"] in ["Fitz17k", "HIBA", "PAD"]:
                fitzpatrick = batch["fitzpatrick"]
                fitzpatrick_binary = batch["fitzpatrick_binary"]
                fitzpatrick_scale = batch["fitzpatrick_scale"]
                fitzpatrick = torch.from_numpy(np.asarray(fitzpatrick))
                fitzpatrick_binary = torch.from_numpy(np.asarray(fitzpatrick_binary))
                fitzpatrick_scale = torch.from_numpy(np.asarray(fitzpatrick_scale))
                hasher = batch["hasher"]
            elif config["dataset_name"] in ["GF3300"]:
                filename = batch["filename"]
                SA = batch["{}".format(config["train"]["SA_level"])]
                SA = torch.from_numpy(np.asarray(SA))
            else:
                raise ValueError("Invalid dataset name.")

            outputs = model(inputs.float())  # (batchsize, classes num)

            if num_classes == 2:
                probs = torch.nn.functional.sigmoid(outputs)
                theshold = find_threshold(
                    probs.cpu().data.numpy(), classes.cpu().data.numpy()
                )
                preds = (probs > theshold).to(torch.int32)

                # Filling the all probs  with dummy values
                all_probs = torch.zeros_like(probs)
            else:
                all_probs = torch.nn.functional.softmax(outputs, dim=1)
                probs, preds = torch.max(all_probs, 1)

            p_list.append(probs.cpu().tolist())
            prediction_list.append(preds.cpu().tolist())
            all_p_list.append(all_probs.cpu().tolist())
            labels_list.append(classes.tolist())
            total += inputs.shape[0]

            if config["dataset_name"] in ["Fitz17k", "HIBA", "PAD"]:
                fitzpatrick_list.append(fitzpatrick.tolist())
                fitzpatrick_binary_list.append(fitzpatrick_binary.tolist())
                fitzpatrick_scale_list.append(fitzpatrick_scale.tolist())
                hasher_list.append(hasher)
            elif config["dataset_name"] in ["GF3300"]:
                hasher_list.append(filename)
                SA_list.append(SA.tolist())

    if config["dataset_name"] in ["Fitz17k", "HIBA", "PAD"]:
        df_preds = pd.DataFrame(
            {
                "hasher": flatten(hasher_list),
                "label": flatten(labels_list),
                "fitzpatrick": flatten(fitzpatrick_list),
                "fitzpatrick_binary": flatten(fitzpatrick_binary_list),
                "fitzpatrick_scale": flatten(fitzpatrick_scale_list),
                "prediction_probability": flatten(p_list),
                "all_probability": flatten_all_prob(all_p_list),
                "prediction": flatten(prediction_list),
            }
        )
        metrics = cal_metrics_skin(df_preds)
    elif config["dataset_name"] in ["GF3300"]:
        df_preds = pd.DataFrame(
            {
                "filename": flatten(hasher_list),
                "label": flatten(labels_list),
                "{}".format(config["train"]["SA_level"]): flatten(SA_list),
                "prediction_probability": flatten(p_list),
                "all_probability": flatten_all_prob(all_p_list),
                "prediction": flatten(prediction_list),
            }
        )
        metrics = cal_metrics_eye(
            df_preds, SA_level="{}".format(config["train"]["SA_level"])
        )

    if save_preds:

        df_preds.to_csv(
            os.path.join(
                config["output_folder_path"],
                f"validation_results_{model_type}.csv",
            ),
            index=False,
        )

    return metrics, df_preds


def eval_model_SABranch(
    model,
    dataloaders,
    dataset_sizes,
    device,
    model_type,
    config,
    save_preds=False,
):
    model = model.eval().to(device)

    hasher_list = []
    prediction_list = []
    fitzpatrick_list = []
    fitzpatrick_binary_list = []
    gender_list = []
    p_list = []
    SA_list = []

    with torch.no_grad():
        total = 0

        for batch in dataloaders["val"]:
            inputs = batch["image"].to(device)
            if config["dataset_name"] in ["Fitz17k", "HIBA", "PAD"]:
                hasher = batch["hasher"]
                fitzpatrick = batch["fitzpatrick"]
                fitzpatrick_binary = batch["fitzpatrick_binary"]
                fitzpatrick = torch.from_numpy(np.asarray(fitzpatrick))
                fitzpatrick_binary = (
                    torch.from_numpy(np.asarray(fitzpatrick_binary))
                    .unsqueeze(1)
                    .to(device)
                )
            elif config["dataset_name"] in ["GF3300"]:
                filename = batch["filename"]
                SA = batch["{}".format(config["train"]["SA_level"])]
                SA = torch.from_numpy(np.asarray(SA))
            else:
                raise ValueError("Invalid dataset name.")

            outputs = model(inputs.float())  # (batchsize, classes num)

            probs = torch.nn.functional.sigmoid(outputs)

            if config["dataset_name"] in ["Fitz17k", "HIBA", "PAD"]:
                theshold = find_threshold(
                    probs.cpu().data.numpy(), fitzpatrick_binary.cpu().data.numpy()
                )
                preds = (probs > theshold).to(torch.int32)

                hasher_list.append(hasher)
                fitzpatrick_list.append(fitzpatrick.tolist())
                fitzpatrick_binary_list.append(fitzpatrick_binary.cpu().tolist())

            elif config["dataset_name"] in ["GF3300"]:
                assert (
                    "binary" in config["train"]["SA_level"]
                ), "Sopporting binary SA only"

                theshold = find_threshold(
                    probs.cpu().data.numpy(), SA.cpu().data.numpy()
                )
                preds = (probs > theshold).to(torch.int32)
                hasher_list.append(filename)
                SA_list.append(SA.tolist())

            else:
                raise ValueError("Invalid dataset name.")

            p_list.append(probs.cpu().tolist())
            prediction_list.append(preds.cpu().tolist())

            total += inputs.shape[0]

    if config["dataset_name"] in ["Fitz17k", "HIBA", "PAD"]:
        df_preds = pd.DataFrame(
            {
                "hasher": flatten(hasher_list),
                "fitzpatrick_binary": flatten(fitzpatrick_binary_list),
                "prediction": flatten(prediction_list),
                "fitzpatrick": flatten(fitzpatrick_list),
                "prediction_probability": flatten(p_list),
            }
        )
    elif config["dataset_name"] in ["GF3300"]:
        df_preds = pd.DataFrame(
            {
                "filename": flatten(hasher_list),
                f"{config['train']['SA_level']}": flatten(SA_list),
                "prediction": flatten(prediction_list),
                "prediction_probability": flatten(p_list),
            }
        )

    if save_preds:

        df_preds.to_csv(
            os.path.join(
                config["output_folder_path"],
                f"validation_results_{model_type}.csv",
            ),
            index=False,
        )

    y_true = df_preds[config["train"]["SA_level"]].values
    y_pred = df_preds["prediction"].values

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, flatten(p_list))
    f1 = f1_score(y_true, y_pred, average="macro")

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc,
        "F1-score": f1,
    }

    print(classification_report(y_true, y_pred, digits=4))

    return metrics, df_preds


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to configuration yaml file.")
    args = parser.parse_args()
    with open(args.config, "r") as fh:
        config = yaml.load(fh, Loader=yaml.FullLoader)

    print("CUDA is available: {} \n".format(torch.cuda.is_available()))
    print("Starting... \n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seeds(config["seed"])

    if config["dataset_name"] in ["Fitz17k", "HIBA", "PAD"]:
        dataloaders, dataset_sizes, main_num_classes, SA_num_classes = get_dataloaders(
            root_image_dir=config["root_image_dir"],
            Generated_csv_path=config["Generated_csv_path"],
            sampler_type="WeightedRandom",
            dataset_name=config["dataset_name"],
            stratify_cols=["low"],
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
        raise ValueError("Invalid dataset name.")

    # load the model
    checkpoint = torch.load(config["eval"]["weight_path"])
    model = checkpoint["model"]
    print(f"valmetrics read from checkpoint: {checkpoint['leading_val_metrics']}")
    print()
    model = model.eval().to(device)

    val_metrics, _ = eval_model(
        model,
        dataloaders,
        dataset_sizes,
        main_num_classes,
        device,
        config["train"]["main_level"],
        "EvalScript",
        config,
        save_preds=True,
    )

    print("validation metrics:")
    pprint(val_metrics)
    print()
