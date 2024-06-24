import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    classification_report,
)

from Utils.Metrics import cal_metrics, find_threshold
from Utils.Misc_utils import set_seeds, Logger, get_stat, get_mask_idx
from Datasets.dataloaders import get_dataloaders
from Models.ViT_LRP import deit_small_patch16_224


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
    topk_p = []
    topk_n = []
    d1 = []
    d2 = []
    d3 = []
    p1 = []
    p2 = []
    p3 = []
    with torch.no_grad():
        running_corrects = 0
        running_balanced_acc_sum = 0
        total = 0

        for batch in dataloaders["val"]:
            inputs = batch["image"].to(device)
            classes = batch[level]
            fitzpatrick = batch["fitzpatrick"]
            fitzpatrick_binary = batch["fitzpatrick_binary"]
            fitzpatrick_scale = batch["fitzpatrick_scale"]
            fitzpatrick = torch.from_numpy(np.asarray(fitzpatrick))
            fitzpatrick_binary = torch.from_numpy(np.asarray(fitzpatrick_binary))
            fitzpatrick_scale = torch.from_numpy(np.asarray(fitzpatrick_scale))
            hasher = batch["hasher"]

            if num_classes == 2:
                classes = torch.from_numpy(np.asarray(classes)).unsqueeze(1).to(device)
            else:
                classes = torch.from_numpy(np.asarray(classes)).to(device)

            outputs = model(inputs.float())  # (batchsize, classes num)

            if num_classes == 2:
                probs = torch.nn.functional.sigmoid(outputs)
                theshold = find_threshold(
                    probs.cpu().data.numpy(), classes.cpu().data.numpy()
                )
                preds = (probs > theshold).to(torch.int32)
            else:
                all_probs = torch.nn.functional.softmax(outputs, dim=1)
                probs, preds = torch.max(all_probs, 1)

            if level == "low":
                _, preds5 = torch.topk(all_probs, 3)  # topk values, topk indices
                # topk_p.append(np.exp(_.cpu()).tolist())
                topk_p.append((_.cpu()).tolist())
                topk_n.append(preds5.cpu().tolist())

            running_corrects += torch.sum(preds == classes.data)
            running_balanced_acc_sum += (
                balanced_accuracy_score(classes.data.cpu(), preds.cpu())
                * inputs.shape[0]
            )
            p_list.append(probs.cpu().tolist())
            all_p_list.append(all_probs.cpu().tolist())
            prediction_list.append(preds.cpu().tolist())
            labels_list.append(classes.tolist())
            fitzpatrick_list.append(fitzpatrick.tolist())
            fitzpatrick_binary_list.append(fitzpatrick_binary.tolist())
            fitzpatrick_scale_list.append(fitzpatrick_scale.tolist())
            hasher_list.append(hasher)
            total += inputs.shape[0]

        acc = float(running_corrects) / float(dataset_sizes["val"])
        balanced_acc = float(running_balanced_acc_sum) / float(dataset_sizes["val"])

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

    if level == "low":
        for j in topk_n:  # each sample
            for i in j:  # in k
                d1.append(i[0])
                d2.append(i[1])
                d3.append(i[2])
        for j in topk_p:
            for i in j:
                # print(i)
                p1.append(i[0])
                p2.append(i[1])
                p3.append(i[2])

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
                "d1": d1,
                "d2": d2,
                "d3": d3,
                "p1": p1,
                "p2": p2,
                "p3": p3,
            }
        )
    else:
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
    if save_preds:
        num_epoch = config["default"]["n_epochs"]

        df_preds.to_csv(
            os.path.join(
                config["output_folder_path"],
                f"validation_results_{model_type}_epoch={num_epoch}_random_holdout.csv",
            ),
            index=False,
        )

        print(
            f"\nFinal Validation results for {model_type}: Accuracy: {acc}  Balanced Accuracy: {balanced_acc} \n"
        )

    metrics = cal_metrics(df_preds)

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
    model = model.eval()

    hasher_list = []
    prediction_list = []
    fitzpatrick_list = []
    fitzpatrick_binary_list = []
    gender_list = []
    p_list = []

    with torch.no_grad():
        running_corrects = 0
        running_balanced_acc_sum = 0
        total = 0

        for batch in dataloaders["val"]:
            inputs = batch["image"].to(device)
            hasher = batch["hasher"]
            fitzpatrick = batch["fitzpatrick"]
            fitzpatrick_binary = batch["fitzpatrick_binary"]
            fitzpatrick = torch.from_numpy(np.asarray(fitzpatrick))
            fitzpatrick_binary = (
                torch.from_numpy(np.asarray(fitzpatrick_binary)).unsqueeze(1).to(device)
            )
            fitzpatrick_list.append(fitzpatrick.tolist())
            fitzpatrick_binary_list.append(fitzpatrick_binary.cpu().tolist())
            hasher_list.append(hasher)

            if config["default"]["level"] == "gender":
                gender = batch["gender"]
                gender = torch.from_numpy(np.asarray(gender)).unsqueeze(1).to(device)
                gender_list.append(gender.cpu().tolist())

            outputs = model(inputs.float())  # (batchsize, classes num)

            probs = torch.nn.functional.sigmoid(outputs)

            if config["default"]["level"] == "gender":
                theshold = find_threshold(
                    probs.cpu().data.numpy(), gender.cpu().data.numpy()
                )
                preds = (probs > theshold).to(torch.int32)
                running_corrects += torch.sum(preds == gender.data)
                running_balanced_acc_sum += (
                    balanced_accuracy_score(gender.data.cpu(), preds.cpu())
                    * inputs.shape[0]
                )
            else:
                theshold = find_threshold(
                    probs.cpu().data.numpy(), fitzpatrick_binary.cpu().data.numpy()
                )
                preds = (probs > theshold).to(torch.int32)
                running_corrects += torch.sum(preds == fitzpatrick_binary.data)
                running_balanced_acc_sum += (
                    balanced_accuracy_score(fitzpatrick_binary.data.cpu(), preds.cpu())
                    * inputs.shape[0]
                )
            p_list.append(probs.cpu().tolist())
            prediction_list.append(preds.cpu().tolist())

            total += inputs.shape[0]

        acc = float(running_corrects) / float(dataset_sizes["val"])
        balanced_acc = float(running_balanced_acc_sum) / float(dataset_sizes["val"])

    def flatten(list_of_lists):
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], list):
            return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
        return list_of_lists[:1] + flatten(list_of_lists[1:])

    if config["default"]["level"] == "gender":
        df_preds = pd.DataFrame(
            {
                "hasher": flatten(hasher_list),
                "fitzpatrick_binary": flatten(fitzpatrick_binary_list),
                "prediction": flatten(prediction_list),
                "fitzpatrick": flatten(fitzpatrick_list),
                "prediction_probability": flatten(p_list),
                "gender": flatten(gender_list),
            }
        )
    else:
        df_preds = pd.DataFrame(
            {
                "hasher": flatten(hasher_list),
                "fitzpatrick_binary": flatten(fitzpatrick_binary_list),
                "prediction": flatten(prediction_list),
                "fitzpatrick": flatten(fitzpatrick_list),
                "prediction_probability": flatten(p_list),
            }
        )

    if save_preds:
        num_epoch = config["default"]["n_epochs"]
        df_preds.to_csv(
            os.path.join(
                config["output_folder_path"],
                f"validation_results_{model_type}_epoch={num_epoch}_random_holdout.csv",
            ),
            index=False,
        )
        print(
            f"\nFinal Validation results for {model_type}: Accuracy: {acc}  Balanced Accuracy: {balanced_acc} \n"
        )

    y_true = df_preds[config["default"]["level"]].values
    y_pred = df_preds["prediction"].values

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc,
        "F1-score": f1,
    }

    print("validation metrics:")
    print(metrics)

    print(classification_report(y_true, y_pred))

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

    dataloaders, dataset_sizes, main_num_classes, SA_num_classes = get_dataloaders(
        root_image_dir=config["root_image_dir"],
        Generated_csv_path=config["Generated_csv_path"],
        sampler_type="WeightedRandom",
        dataset_name=config["dataset_name"],
        stratify_cols=["low"],
        main_level=config["prune"]["main_level"],
        SA_level=config["prune"]["SA_level"],
        batch_size=config["default"]["batch_size"],
        num_workers=1,
    )

    # load both models
    model = deit_small_patch16_224(
        num_classes=main_num_classes,
        weight_path=config["eval_path"],
    )
    model = model.eval().to(device)

    val_metrics, _ = eval_model(
        model,
        dataloaders,
        dataset_sizes,
        main_num_classes,
        device,
        config["prune"]["main_level"],
        "main model",
        config,
        save_preds=True,
    )

    print(val_metrics)
