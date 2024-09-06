import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    accuracy_score,
)

import matplotlib.pyplot as plt


def cal_metrics_skin(df):

    is_binaryCLF = len(df["label"].unique()) == 2
    num_classes = len(df["label"].unique())

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
        all_probs = np.array(df["all_probability"].to_list())
        AUC = (
            roc_auc_score(df["label"], all_probs, average="macro", multi_class="ovo")
            * 100
        )

        AUC_per_type = []
        for i in range(len(label_array_per_fitz)):
            df_filtered = df[df["fitzpatrick"] == i]
            all_probs = np.array(df_filtered["all_probability"].to_list())
            try:
                AUC_per_type.append(
                    roc_auc_score(
                        df_filtered["label"],
                        all_probs,
                        average="macro",
                        multi_class="ovr",
                    )
                    * 100
                )
            except:
                AUC_per_type.append(np.nan)
        AUC_Gap = max(AUC_per_type) - min(AUC_per_type)

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

    EOpp0_new = np.abs((np.array(class_tnr_fitz1) - np.array(class_tnr_fitz0))).mean()

    # EOpp1
    EOpp1 = 0
    for c in range(len(class_tpr_fitz0)):
        val = abs(class_tpr_fitz1[c] - class_tpr_fitz0[c])
        if not np.isnan(val):
            EOpp1 += val
    EOpp1_new = np.abs((np.array(class_tpr_fitz1) - np.array(class_tpr_fitz0))).mean()

    # EOdd
    EOdd_new = (
        np.abs(
            (np.array(class_tpr_fitz1) - np.array(class_tpr_fitz0))
            + (np.array(class_fpr_fitz1) - np.array(class_fpr_fitz0))
        ).mean()
        / 2
    )
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
        "AUC": AUC,
        "AUC_per_type": AUC_per_type,
        "AUC_Gap": AUC_Gap,
        "AUC_min": min(AUC_per_type),
        "PQD": PQD,
        "DPM": DPM,
        "EOM": EOM,
        "EOpp0": EOpp0,
        "EOpp1": EOpp1,
        "EOdd": EOdd,
        "EOdd_new": EOdd_new,
        "EOpp0_new": EOpp0_new,
        "EOpp1_new": EOpp1_new,
        "NAR": NAR,
        "NFR_W": NFR_W,
        "NFR_Mac": NFR_Mac,
        "acc_avg_binary": avg_acc_binary,
        "acc_per_type_binary": acc_array_binary,
        "PQD_binary": PQD_binary,
        "DPM_binary": DPM_binary,
        "EOM_binary": EOM_binary,
        "NAR_binary": NAR_binary,
    }


def cal_metrics_eye(df, SA_level):

    num_classes = len(df["label"].unique())
    is_binaryCLF = len(df["label"].unique()) == 2

    type_indices = sorted(list(df[SA_level].unique()))
    is_binarySA = len(type_indices) == 2
    if not is_binarySA:
        SA_level_binary = SA_level.replace("binary", "multi")

    labels_array = np.zeros((len(type_indices), len(df["label"].unique())))
    correct_array = np.zeros((len(type_indices), len(df["label"].unique())))
    predictions_array = np.zeros((len(type_indices), len(df["label"].unique())))
    prob_array = [[] for i in range(len(type_indices))]
    label_array_per_SA = [[] for i in range(len(type_indices))]

    positive_list = []
    labels_SA0 = []
    labels_SA1 = []
    predictions_SA0 = []
    predictions_SA1 = []

    for i in range(df.shape[0]):
        prediction = df.iloc[i]["prediction"]
        label = df.iloc[i]["label"]
        type = df.iloc[i][SA_level]
        if not is_binarySA:
            type_binary = df.iloc[i][SA_level_binary]

        labels_array[int(type), int(label)] += 1
        predictions_array[int(type), int(prediction)] += 1
        if prediction == label:
            correct_array[int(type), int(label)] += 1

        if is_binaryCLF:
            prob_array[int(type)].append(df.iloc[i]["prediction_probability"])
            label_array_per_SA[int(type)].append(label)
            if prediction == 0:
                positive_list.append(1.0 - df.iloc[i]["prediction_probability"])
            else:
                positive_list.append(df.iloc[i]["prediction_probability"])

        if is_binarySA:
            if type == 0:
                labels_SA0.append(label)
                predictions_SA0.append(prediction)
            else:
                labels_SA1.append(label)
                predictions_SA1.append(prediction)
        else:
            if type_binary == 0:
                labels_SA0.append(label)
                predictions_SA0.append(prediction)
            else:
                labels_SA1.append(label)
                predictions_SA1.append(prediction)

    correct_array = correct_array[type_indices]
    labels_array = labels_array[type_indices]
    predictions_array = predictions_array[type_indices]

    # Accuracy, accuracy per type
    Accuracy = accuracy_score(df["label"], df["prediction"]) * 100

    acc_array = []
    for i in range(len(type_indices)):
        acc_array.append(
            accuracy_score(
                df[df[SA_level] == i]["label"],
                df[df[SA_level] == i]["prediction"],
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
                df[df[SA_level] == i]["label"],
                df[df[SA_level] == i]["prediction"],
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
                df[df[SA_level] == i]["label"],
                df[df[SA_level] == i]["prediction"],
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
        for i in range(len(label_array_per_SA)):
            try:
                AUC_per_type.append(
                    roc_auc_score(label_array_per_SA[i], prob_array[i]) * 100
                )
            except:
                AUC_per_type.append(np.nan)
        AUC_Gap = max(AUC_per_type) - min(AUC_per_type)
    else:
        all_probs = np.array(df["all_probability"].to_list())
        AUC = (
            roc_auc_score(df["label"], all_probs, average="macro", multi_class="ovo")
            * 100
        )

        AUC_per_type = []
        for i in range(len(label_array_per_SA)):
            df_filtered = df[df[SA_level] == i]
            all_probs = np.array(df_filtered["all_probability"].to_list())
            try:
                AUC_per_type.append(
                    roc_auc_score(
                        df_filtered["label"],
                        all_probs,
                        average="macro",
                        multi_class="ovr",
                    )
                    * 100
                )
            except:
                AUC_per_type.append(np.nan)
        AUC_Gap = max(AUC_per_type) - min(AUC_per_type)

    ##############################          Metrics with binary Sensative attribute         ##############################

    # getting class-wise TPR, FPR, TNR for SA 0
    conf_matrix_SA0 = confusion_matrix(labels_SA0, predictions_SA0)

    # Initialize lists to store TPR, TNR, FPR for each class
    class_tpr_SA0 = []
    class_tnr_SA0 = []
    class_fpr_SA0 = []

    for i in range(len(conf_matrix_SA0)):
        # Calculate TPR for class i
        tpr = conf_matrix_SA0[i, i] / sum(conf_matrix_SA0[i, :])
        class_tpr_SA0.append(tpr)

        # Calculate TNR for class i
        tn = (
            sum(sum(conf_matrix_SA0))
            - sum(conf_matrix_SA0[i, :])
            - sum(conf_matrix_SA0[:, i])
            + conf_matrix_SA0[i, i]
        )
        fp = sum(conf_matrix_SA0[:, i]) - conf_matrix_SA0[i, i]
        fn = sum(conf_matrix_SA0[i, :]) - conf_matrix_SA0[i, i]
        tnr = tn / (tn + fp)
        class_tnr_SA0.append(tnr)

        # Calculate FPR for class i
        fpr = 1 - tnr
        class_fpr_SA0.append(fpr)

    # getting class-wise TPR, FPR, TNR for SA 1

    conf_matrix_SA1 = confusion_matrix(labels_SA1, predictions_SA1)

    # Check if there is any class that is not in both subgroups to handle it
    try:
        if is_binarySA:
            class_idx = (
                set(df[df[SA_level] == 0]["label"].unique())
                - set(df[df[SA_level] == 1]["label"].unique())
            ).pop()
        else:
            class_idx = (
                set(df[df[SA_level_binary] == 0]["label"].unique())
                - set(df[df[SA_level_binary] == 1]["label"].unique())
            ).pop()
        conf_matrix_SA1 = np.insert(conf_matrix_SA1, class_idx, 0, axis=1)
        conf_matrix_SA1 = np.insert(conf_matrix_SA1, class_idx, 0, axis=0)
        print(f"INFO: class {class_idx} is not in both binary subgroups")
    except:
        class_idx = None

    # Initialize lists to store TPR, TNR, FPR for each class
    class_tpr_SA1 = []
    class_tnr_SA1 = []
    class_fpr_SA1 = []

    for i in range(len(conf_matrix_SA1)):
        # Calculate TPR for class i
        tpr = conf_matrix_SA1[i, i] / sum(conf_matrix_SA1[i, :])
        class_tpr_SA1.append(tpr)

        # Calculate TNR for class i
        tn = (
            sum(sum(conf_matrix_SA1))
            - sum(conf_matrix_SA1[i, :])
            - sum(conf_matrix_SA1[:, i])
            + conf_matrix_SA1[i, i]
        )
        fp = sum(conf_matrix_SA1[:, i]) - conf_matrix_SA1[i, i]
        fn = sum(conf_matrix_SA1[i, :]) - conf_matrix_SA1[i, i]
        tnr = tn / (tn + fp)
        class_tnr_SA1.append(tnr)

        # Calculate FPR for class i
        fpr = 1 - tnr
        class_fpr_SA1.append(fpr)

    if class_idx is not None:
        class_tpr_SA1[class_idx] = np.nan
        class_tnr_SA1[class_idx] = np.nan
        class_fpr_SA1[class_idx] = np.nan

    # EOpp0
    EOpp0 = 0
    for c in range(len(class_tnr_SA0)):
        val = abs(class_tnr_SA1[c] - class_tnr_SA0[c])
        if not np.isnan(val):
            EOpp0 += val

    EOpp0_new = np.abs((np.array(class_tnr_SA1) - np.array(class_tnr_SA0))).mean()

    # EOpp1
    EOpp1 = 0
    for c in range(len(class_tpr_SA0)):
        val = abs(class_tpr_SA1[c] - class_tpr_SA0[c])
        if not np.isnan(val):
            EOpp1 += val
    EOpp1_new = np.abs((np.array(class_tpr_SA1) - np.array(class_tpr_SA0))).mean()

    # EOdd
    EOdd_new = (
        np.abs(
            (np.array(class_tpr_SA1) - np.array(class_tpr_SA0))
            + (np.array(class_fpr_SA1) - np.array(class_fpr_SA0))
        ).mean()
        / 2
    )
    EOdd = 0
    for c in range(len(class_tpr_SA0)):
        val = abs(
            class_tpr_SA1[c] - class_tpr_SA0[c] + class_fpr_SA1[c] - class_fpr_SA0[c]
        )
        if not np.isnan(val):
            EOdd += val

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
        "AUC": AUC,
        "AUC_per_type": AUC_per_type,
        "AUC_Gap": AUC_Gap,
        "AUC_min": min(AUC_per_type),
        "PQD": PQD,
        "DPM": DPM,
        "EOM": EOM,
        "EOpp0": EOpp0,
        "EOpp1": EOpp1,
        "EOdd": EOdd,
        "EOdd_new": EOdd_new,
        "EOpp0_new": EOpp0_new,
        "EOpp1_new": EOpp1_new,
        "NAR": NAR,
        "NFR_W": NFR_W,
        "NFR_Mac": NFR_Mac,
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
    if postfix == "PS":
        iterations = list(range(1, len(df) + 1))
    else:
        iterations = list(range(len(df)))
    plt.figure(figsize=(len(df), len(df) * 0.6))

    for metric in selected_metrics:
        iteration_points = []
        metric_points = []

        # Gather only non-None points
        for i in range(len(df)):
            if pd.notna(df[metric][i]):
                iteration_points.append(iterations[i])
                metric_points.append(df[metric][i])

        plt.plot(iteration_points, metric_points, marker="o", label=metric)

        # Add annotation for each valid point
        for i in range(len(iteration_points)):
            plt.annotate(
                (
                    f"{metric_points[i]}"
                    if type(metric_points[i]) == int
                    else f"{metric_points[i]:.3f}"
                ),
                (iteration_points[i], metric_points[i]),
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

    os.makedirs(config["output_folder_path"], exist_ok=True)
    plt.savefig(
        os.path.join(
            config["output_folder_path"], f"DeiT_S_LRP_pruning_metrics_{postfix}.png"
        )
    )
    # plt.close()


def plot_metrics_training(train_df, val_df, selected_metrics, postfix, config):
    """
    Plot selected metrics over iterations comparing training and validation results with annotations for each point.

    Args:
    - train_df (pd.DataFrame): DataFrame containing training metrics for each iteration.
    - val_df (pd.DataFrame): DataFrame containing validation metrics for each iteration.
    - selected_metrics (list of str): List of metric names to include in the plot.
    - postfix (str): Postfix for the output file name.
    - config (dict): Configuration dictionary with output folder path.
    """
    iterations = list(range(1, len(train_df) + 1))
    plt.figure(figsize=(len(train_df), len(train_df) * 0.6))

    for metric in selected_metrics:
        plt.plot(iterations, train_df[metric], label=f"Train {metric}")
        plt.plot(iterations, val_df[metric], label=f"Val {metric}", linestyle="--")

        # Annotate training metrics
        for i, txt in enumerate(train_df[metric]):
            plt.annotate(
                f"{txt:.3f}",
                (iterations[i], train_df[metric][i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=12,
                color="blue",  # Adjust color to match the line color
            )

        # Annotate validation metrics
        for i, txt in enumerate(val_df[metric]):
            plt.annotate(
                f"{txt:.3f}",
                (iterations[i], val_df[metric][i]),
                textcoords="offset points",
                xytext=(0, -10),
                ha="center",
                fontsize=12,
                color="orange",  # Adjust color to match the line color
            )

    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Metric Values", fontsize=14)
    plt.title("Metrics Over Iterations", fontsize=16)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=12)
    plt.xticks(iterations, fontsize=12)  # Set discrete values on the x-axis
    plt.yticks(fontsize=12)
    plt.grid(True)

    plt.tight_layout()  # Adjust layout to prevent clipping of legend
    plt.savefig(
        os.path.join(config["output_folder_path"], f"DeiT-S_metrics_{postfix}.png")
    )
