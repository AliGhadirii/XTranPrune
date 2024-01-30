import os
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    accuracy_score,
)

import matplotlib.pyplot as plt


def cal_metrics(df):
    """
    calculate average accuracy, accuracy per skin type, PQD, DPM, EOM, EOpp0, EOpp1, EOdd, and NAR.
    Skin type in the input df should be in the range of [0,5].
    input val results csv path, type_indices: a list
    output a dic, 'acc_avg': value, 'acc_per_type': array[x,x,x], 'PQD', 'DPM', 'EOM'
    """
    is_binaryCLF = len(df["label"].unique()) == 2

    type_indices = sorted(list(df["fitzpatrick"].unique()))
    type_indices_binary = sorted(list(df["fitzpatrick_binary"].unique()))

    labels_array = np.zeros((len(type_indices), len(df["label"].unique())))
    correct_array = np.zeros((len(type_indices), len(df["label"].unique())))
    predictions_array = np.zeros((len(type_indices), len(df["label"].unique())))
    prob_array = [[] for i in range(len(type_indices))]
    label_array_per_fitz = [[] for i in range(len(type_indices))]

    labels_array_binary = np.zeros((2, len(df["label"].unique())))
    correct_array_binary = np.zeros((2, len(df["label"].unique())))
    predictions_array_binary = np.zeros((2, len(df["label"].unique())))

    positive_list = []  # get positive probability for binary classification
    labels_ft0 = []
    labels_ft1 = []
    predictions_ft0 = []
    predictions_ft1 = []

    for i in range(df.shape[0]):
        prediction = df.iloc[i]["prediction"]
        label = df.iloc[i]["label"]
        type = df.iloc[i]["fitzpatrick"]
        type_binary = df.iloc[i]["fitzpatrick_binary"]

        labels_array[int(type), int(label)] += 1
        predictions_array[int(type), int(prediction)] += 1
        if prediction == label:
            correct_array[int(type), int(label)] += 1

        labels_array_binary[int(type_binary), int(label)] += 1
        predictions_array_binary[int(type_binary), int(prediction)] += 1
        if prediction == label:
            correct_array_binary[int(type_binary), int(label)] += 1

        if is_binaryCLF:
            prob_array[int(type)].append(df.iloc[i]["prediction_probability"])
            label_array_per_fitz[int(type)].append(label)
            if prediction == 0:
                positive_list.append(1.0 - df.iloc[i]["prediction_probability"])
            else:
                positive_list.append(df.iloc[i]["prediction_probability"])

        if type_binary == 0:
            labels_ft0.append(label)
            predictions_ft0.append(prediction)
        else:
            labels_ft1.append(label)
            predictions_ft1.append(prediction)

    correct_array = correct_array[type_indices]
    labels_array = labels_array[type_indices]
    predictions_array = predictions_array[type_indices]

    # Accuracy, accuracy per type
    Accuracy = accuracy_score(df["label"], df["prediction"]) * 100

    acc_array = []
    for i in range(len(type_indices)):
        acc_array.append(
            accuracy_score(
                df[df["fitzpatrick"] == i]["label"],
                df[df["fitzpatrick"] == i]["prediction"],
            )
            * 100
        )
    acc_array = np.array(acc_array)

    # f1_score, f1-score per type (Weighted average)
    F1_W = f1_score(df["label"], df["prediction"], average="weighted") * 100

    F1_W_array = []
    for i in range(len(type_indices)):
        F1_W_array.append(
            f1_score(
                df[df["fitzpatrick"] == i]["label"],
                df[df["fitzpatrick"] == i]["prediction"],
                average="weighted",
            )
            * 100
        )
    F1_W_array = np.array(F1_W_array)

    # f1_score, f1-score per type (Macro average)
    F1_Mac = f1_score(df["label"], df["prediction"], average="macro") * 100

    F1_Mac_array = []
    for i in range(len(type_indices)):
        F1_Mac_array.append(
            f1_score(
                df[df["fitzpatrick"] == i]["label"],
                df[df["fitzpatrick"] == i]["prediction"],
                average="macro",
            )
            * 100
        )
    F1_Mac_array = np.array(F1_Mac_array)

    # PQD
    PQD = acc_array.min() / acc_array.max()

    # DPM
    demo_array = predictions_array / np.sum(predictions_array, axis=1, keepdims=True)
    DPM = np.mean(demo_array.min(axis=0) / demo_array.max(axis=0))

    # EOM
    eo_array = correct_array / labels_array
    EOM = np.mean(np.nanmin(eo_array, axis=0) / np.nanmax(eo_array, axis=0))

    # NAR
    NAR = (acc_array.max() - acc_array.min()) / acc_array.mean()

    # NFR (Weighted)
    NFR_W = (F1_W_array.max() - F1_W_array.min()) / F1_W_array.mean()

    # NAR (Macro)
    NFR_Mac = (F1_Mac_array.max() - F1_Mac_array.min()) / F1_Mac_array.mean()

    # AUC
    if is_binaryCLF:
        # AUC per skin type
        AUC = roc_auc_score(df["label"], df["prediction_probability"]) * 100
        AUC_per_type = []
        for i in range(len(label_array_per_fitz)):
            try:
                AUC_per_type.append(
                    roc_auc_score(label_array_per_fitz[i], prob_array[i]) * 100
                )
            except:
                AUC_per_type.append(np.nan)
        AUC_Gap = max(AUC_per_type) - min(AUC_per_type)
    else:
        AUC = -1
        AUC_per_type = [-1] * len(type_indices)
        AUC_Gap = -1

    ##############################          Metrics with binary Sensative attribute         ##############################

    correct_array_binary = correct_array_binary[type_indices_binary]
    labels_array_binary = labels_array_binary[type_indices_binary]
    predictions_array_binary = predictions_array_binary[type_indices_binary]

    # avg acc, acc per type
    correct_array_sumc_binary, labels_array_sumc_binary = np.sum(
        correct_array_binary, axis=1
    ), np.sum(
        labels_array_binary, axis=1
    )  # sum skin conditions
    acc_array_binary = correct_array_sumc_binary / labels_array_sumc_binary
    avg_acc_binary = (np.sum(correct_array_binary) / np.sum(labels_array_binary)) * 100

    # PQD
    PQD_binary = acc_array_binary.min() / acc_array_binary.max()

    # DPM
    demo_array_binary = predictions_array_binary / np.sum(
        predictions_array_binary, axis=1, keepdims=True
    )
    DPM_binary = np.mean(demo_array_binary.min(axis=0) / demo_array_binary.max(axis=0))

    # EOM
    eo_array_binary = correct_array_binary / labels_array_binary
    EOM_binary = np.mean(
        np.nanmin(eo_array_binary, axis=0) / np.nanmax(eo_array_binary, axis=0)
    )

    # getting class-wise TPR, FPR, TNR for fitzpatrick 0
    conf_matrix_fitz0 = confusion_matrix(labels_ft0, predictions_ft0)

    # Initialize lists to store TPR, TNR, FPR for each class
    class_tpr_fitz0 = []
    class_tnr_fitz0 = []
    class_fpr_fitz0 = []

    for i in range(len(conf_matrix_fitz0)):
        # Calculate TPR for class i
        tpr = conf_matrix_fitz0[i, i] / sum(conf_matrix_fitz0[i, :])
        class_tpr_fitz0.append(tpr)

        # Calculate TNR for class i
        tn = (
            sum(sum(conf_matrix_fitz0))
            - sum(conf_matrix_fitz0[i, :])
            - sum(conf_matrix_fitz0[:, i])
            + conf_matrix_fitz0[i, i]
        )
        fp = sum(conf_matrix_fitz0[:, i]) - conf_matrix_fitz0[i, i]
        fn = sum(conf_matrix_fitz0[i, :]) - conf_matrix_fitz0[i, i]
        tnr = tn / (tn + fp)
        class_tnr_fitz0.append(tnr)

        # Calculate FPR for class i
        fpr = 1 - tnr
        class_fpr_fitz0.append(fpr)

    # getting class-wise TPR, FPR, TNR for fitzpatrick 1

    conf_matrix_fitz1 = confusion_matrix(labels_ft1, predictions_ft1)

    # Check if there is any class that is not in both subgroups to handle it
    try:
        class_idx = (
            set(df[df["fitzpatrick_binary"] == 0]["label"].unique())
            - set(df[df["fitzpatrick_binary"] == 1]["label"].unique())
        ).pop()
        conf_matrix_fitz1 = np.insert(conf_matrix_fitz1, class_idx, 0, axis=1)
        conf_matrix_fitz1 = np.insert(conf_matrix_fitz1, class_idx, 0, axis=0)
        print(f"INFO: class {class_idx} is not in both binary subgroups")
    except:
        class_idx = None

    # Initialize lists to store TPR, TNR, FPR for each class
    class_tpr_fitz1 = []
    class_tnr_fitz1 = []
    class_fpr_fitz1 = []

    for i in range(len(conf_matrix_fitz1)):
        # Calculate TPR for class i
        tpr = conf_matrix_fitz1[i, i] / sum(conf_matrix_fitz1[i, :])
        class_tpr_fitz1.append(tpr)

        # Calculate TNR for class i
        tn = (
            sum(sum(conf_matrix_fitz1))
            - sum(conf_matrix_fitz1[i, :])
            - sum(conf_matrix_fitz1[:, i])
            + conf_matrix_fitz1[i, i]
        )
        fp = sum(conf_matrix_fitz1[:, i]) - conf_matrix_fitz1[i, i]
        fn = sum(conf_matrix_fitz1[i, :]) - conf_matrix_fitz1[i, i]
        tnr = tn / (tn + fp)
        class_tnr_fitz1.append(tnr)

        # Calculate FPR for class i
        fpr = 1 - tnr
        class_fpr_fitz1.append(fpr)

    if class_idx is not None:
        class_tpr_fitz1[class_idx] = np.nan
        class_tnr_fitz1[class_idx] = np.nan
        class_fpr_fitz1[class_idx] = np.nan

    # EOpp0
    EOpp0 = 0
    for c in range(len(class_tnr_fitz0)):
        val = abs(class_tnr_fitz1[c] - class_tnr_fitz0[c])
        if not np.isnan(val):
            EOpp0 += val

    # EOpp1
    EOpp1 = 0
    for c in range(len(class_tpr_fitz0)):
        val = abs(class_tpr_fitz1[c] - class_tpr_fitz0[c])
        if not np.isnan(val):
            EOpp1 += val

    # EOdd
    EOdd = 0
    for c in range(len(class_tpr_fitz0)):
        val = abs(
            class_tpr_fitz1[c]
            - class_tpr_fitz0[c]
            + class_fpr_fitz1[c]
            - class_fpr_fitz0[c]
        )
        if not np.isnan(val):
            EOdd += val

    # NAR
    NAR_binary = (
        acc_array_binary.max() - acc_array_binary.min()
    ) / acc_array_binary.mean()

    return {
        "accuracy": Accuracy,
        "acc_per_type": acc_array,
        "acc_gap": acc_array.max() - acc_array.min(),
        "F1_W": F1_W,
        "F1_per_type_W": F1_W_array,
        "F1_W_gap": max(F1_W_array) - min(F1_W_array),
        "F1_Mac": F1_Mac,
        "F1_per_type_Mac": F1_Mac_array,
        "F1_Mac_gap": max(F1_Mac_array) - min(F1_Mac_array),
        "Worst_F1_Mac": min(F1_Mac_array),
        "PQD": PQD,
        "DPM": DPM,
        "EOM": EOM,
        "EOpp0": EOpp0,
        "EOpp1": EOpp1,
        "EOdd": EOdd,
        "NAR": NAR,
        "NFR_W": NFR_W,
        "NFR_Mac": NFR_Mac,
        "AUC": AUC,
        "AUC_per_type": AUC_per_type,
        "AUC_Gap": AUC_Gap,
        "AUC_min": min(AUC_per_type),
        "acc_avg_binary": avg_acc_binary,
        "acc_per_type_binary": acc_array_binary,
        "PQD_binary": PQD_binary,
        "DPM_binary": DPM_binary,
        "EOM_binary": EOM_binary,
        "NAR_binary": NAR_binary,
    }


def find_threshold(outputs, labels):
    # Calculate precision and recall values for different thresholds
    precision, recall, thresholds = precision_recall_curve(labels, outputs)

    # Calculate F1-score for different thresholds, handling division by zero
    non_zero_denominator_mask = (precision + recall) != 0
    f1_scores = np.zeros_like(precision)
    f1_scores[non_zero_denominator_mask] = (
        2
        * (precision[non_zero_denominator_mask] * recall[non_zero_denominator_mask])
        / (precision[non_zero_denominator_mask] + recall[non_zero_denominator_mask])
    )

    # Find the index of the threshold with the highest F1-score
    best_threshold_index = np.argmax(f1_scores)

    # Get the best threshold
    best_threshold = thresholds[best_threshold_index]
    return best_threshold


def plot_metrics(df, selected_metrics, postfix, config):
    """
    Plot selected metrics over iterations with annotations for each point.

    Args:
    - df (pd.DataFrame): Dataframe containing metrics for each iteration.
    - selected_metrics (list of str): List of metric names to include in the plot.
    """
    iterations = list(range(1, len(df) + 1))
    plt.figure(figsize=(len(df), len(df) * 0.6))

    for metric in selected_metrics:
        plt.plot(iterations, df[metric], label=metric)
        for i, txt in enumerate(df[metric]):
            plt.annotate(
                f"{txt:.3f}",
                (iterations[i], df[metric][i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=12,
            )

    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Metric Values", fontsize=14)
    plt.title("Metrics Over Iterations", fontsize=16)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
    plt.xticks(iterations, fontsize=12)  # Set discrete values on the x-axis
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.savefig(
        os.path.join(
            config["output_folder_path"], f"DeiT_S_LRP_pruning_metrics_{postfix}.png"
        )
    )
    # plt.close()
