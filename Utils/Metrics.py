import numpy as np
from sklearn.metrics import roc_curve, auc


def cal_metrics(df, type_indices, is_binary=False):
    """
    calculate average accuracy, accuracy per skin type, PQD, DPM, EOM.
    All known skin types
    input val results csv path, type_indices: a list
    output a dic, 'acc_avg': value, 'acc_per_type': array[x,x,x], 'PQD', 'DPM', 'EOM'
    """
    labels_array = np.zeros((6, len(df["label"].unique())))
    correct_array = np.zeros((6, len(df["label"].unique())))
    predictions_array = np.zeros((6, len(df["label"].unique())))
    positive_list = []  # get positive probability for binary classification
    for i in range(df.shape[0]):
        prediction = df.iloc[i]["prediction"]
        label = df.iloc[i]["label"]
        type = df.iloc[i]["fitzpatrick"] - 1
        labels_array[int(type), int(label)] += 1
        predictions_array[int(type), int(prediction)] += 1
        if prediction == label:
            correct_array[int(type), int(label)] += 1

        if is_binary:
            if prediction == 0:
                positive_list.append(1.0 - df.iloc[i]["prediction_probability"])
            else:
                positive_list.append(df.iloc[i]["prediction_probability"])

    correct_array = correct_array[type_indices]
    labels_array = labels_array[type_indices]
    predictions_array = predictions_array[type_indices]

    # avg acc, acc per type
    correct_array_sumc, labels_array_sumc = np.sum(correct_array, axis=1), np.sum(
        labels_array, axis=1
    )  # sum skin conditions
    acc_array = correct_array_sumc / labels_array_sumc
    avg_acc = np.sum(correct_array) / np.sum(labels_array)

    # PQD
    PQD = acc_array.min() / acc_array.max()

    # DPM
    demo_array = predictions_array / np.sum(predictions_array, axis=1, keepdims=True)
    DPM = np.mean(demo_array.min(axis=0) / demo_array.max(axis=0))

    # EOM
    eo_array = correct_array / labels_array
    EOM = np.mean(np.min(eo_array, axis=0) / np.max(eo_array, axis=0))

    # if is binary classification, output AUC
    if is_binary:
        fpr, tpr, threshold = roc_curve(
            df["label"], positive_list, drop_intermediate=True
        )
        AUC = auc(fpr, tpr)
    else:
        AUC = -1

    return {
        "acc_avg": avg_acc,
        "acc_per_type": acc_array,
        "PQD": PQD,
        "DPM": DPM,
        "EOM": EOM,
        "AUC": AUC,
    }
