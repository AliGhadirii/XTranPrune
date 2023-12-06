import os

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
                probs, preds = torch.max(outputs, 1)

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
    p_list = []

    with torch.no_grad():
        running_corrects = 0
        running_balanced_acc_sum = 0
        total = 0

        for batch in dataloaders["val"]:
            inputs = batch["image"].to(device)
            fitzpatrick = batch["fitzpatrick"]
            fitzpatrick_binary = batch["fitzpatrick_binary"]
            fitzpatrick = torch.from_numpy(np.asarray(fitzpatrick))
            fitzpatrick_binary = (
                torch.from_numpy(np.asarray(fitzpatrick_binary)).unsqueeze(1).to(device)
            )
            hasher = batch["hasher"]

            outputs = model(inputs.float())  # (batchsize, classes num)

            probs = torch.nn.functional.sigmoid(outputs)
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
            fitzpatrick_list.append(fitzpatrick.tolist())
            fitzpatrick_binary_list.append(fitzpatrick_binary.tolist())
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

    y_true = df_preds["fitzpatrick_binary"].values
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
