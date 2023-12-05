import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

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
            classes = torch.from_numpy(np.asarray(classes)).to(device)
            fitzpatrick = torch.from_numpy(np.asarray(fitzpatrick))
            fitzpatrick_binary = torch.from_numpy(np.asarray(fitzpatrick_binary))
            fitzpatrick_scale = torch.from_numpy(np.asarray(fitzpatrick_scale))
            hasher = batch["hasher"]

            outputs = model(inputs.float())  # (batchsize, classes num)

            if num_classes == 2:
                probs = torch.nn.functional.sigmoid(outputs)
                theshold = find_threshold(
                    probs.cpu().data.numpy(), classes.cpu().data.numpy()
                )
                preds = probs > theshold
            else:
                all_probs = torch.nn.functional.softmax(outputs, dim=1)
                probs, preds = torch.topk(all_probs, 1)  # topk values, topk indices

            if level == "low":
                _, preds5 = torch.topk(all_probs, 3)  # topk values, topk indices
                # topk_p.append(np.exp(_.cpu()).tolist())
                topk_p.append((_.cpu()).tolist())
                topk_n.append(preds5.cpu().tolist())

            running_corrects += torch.sum(preds.reshape(-1) == classes.data)
            running_balanced_acc_sum += (
                balanced_accuracy_score(classes.data.cpu(), preds.reshape(-1).cpu())
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
