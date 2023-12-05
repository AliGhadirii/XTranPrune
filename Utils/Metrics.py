import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve


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

    labels_array = np.zeros((6, len(df["label"].unique())))
    correct_array = np.zeros((6, len(df["label"].unique())))
    predictions_array = np.zeros((6, len(df["label"].unique())))

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

    # NAR
    NAR = (acc_array.max() - acc_array.min()) / acc_array.mean()

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
    avg_acc_binary = np.sum(correct_array_binary) / np.sum(labels_array_binary)

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
        np.min(eo_array_binary, axis=0) / np.max(eo_array_binary, axis=0)
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

    # EOpp0
    EOpp0 = 0
    for c in range(len(class_tnr_fitz0)):
        EOpp0 += abs(class_tnr_fitz1[c] - class_tnr_fitz0[c])

    # EOpp1
    EOpp1 = 0
    for c in range(len(class_tpr_fitz0)):
        EOpp1 += abs(class_tpr_fitz1[c] - class_tpr_fitz0[c])

    # EOdd
    EOdd = 0
    for c in range(len(class_tpr_fitz0)):
        EOdd += abs(
            class_tpr_fitz1[c]
            - class_tpr_fitz0[c]
            + class_fpr_fitz1[c]
            - class_fpr_fitz0[c]
        )

    # NAR
    NAR_binary = (
        acc_array_binary.max() - acc_array_binary.min()
    ) / acc_array_binary.mean()

    # if is binary classification, output AUC
    if is_binaryCLF:
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
        "EOpp0": EOpp0,
        "EOpp1": EOpp1,
        "EOdd": EOdd,
        "NAR": NAR,
        "AUC": AUC,
        "acc_avg_binary": avg_acc_binary,
        "acc_per_type_binary": acc_array_binary,
        "PQD_binary": PQD_binary,
        "DPM_binary": DPM_binary,
        "EOM_binary": EOM_binary,
        "NAR_binary": NAR_binary,
    }


# adapted from https://github.com/LalehSeyyed/Underdiagnosis_NatMed/blob/main/CXP/classification/predictions.py
# and https://github.com/MLforHealth/CXR_Fairness/blob/master/cxr_fairness/metrics.py
def find_threshold(outputs, labels):
    # to find this thresold, first we get the precision and recall without this, from there we calculate f1 score,
    # using f1score, we found this theresold which has best precsision and recall.  Then this threshold activation
    # are used to calculate our binary output.

    PRED_LABEL = ["disease"]

    # create empty dfs
    thrs = []

    for j in range(0, len(outputs)):
        thisrow = {}
        truerow = {}

        # iterate over each entry in prediction vector; each corresponds to
        # individual label
        for k in range(len(PRED_LABEL)):
            thisrow["prob_" + PRED_LABEL[k]] = outputs[j]
            truerow[PRED_LABEL[k]] = labels[j]

    for column in PRED_LABEL:
        thisrow = {}
        thisrow["label"] = column
        thisrow["bestthr"] = np.nan

        p, r, t = precision_recall_curve(labels, outputs)
        # Choose the best threshold based on the highest F1 measure
        f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
        bestthr = t[np.where(f1 == max(f1))]
        thrs.append(bestthr)

        thisrow["bestthr"] = bestthr[0]

    return bestthr[0]
