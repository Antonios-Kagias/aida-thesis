##### Functions to calculate metrics precision, recall, accuracy and F1 score for multi-label classification

import statistics
import itertools
from sklearn.metrics import confusion_matrix

def rank_labels(labels, instance_labels, placeholder = '-'):
    # Create an array with placeholders
    ranks = [placeholder] * len(labels)

    # Dictionary to hold the rank positions of instance labels
    label_positions = {label: pos+1 for pos, label in enumerate(instance_labels)}

    # Fill the ranks array with the correct positions
    for i in range(len(labels)):
        label = labels[i]
        if label in label_positions:
            ranks[i] = label_positions[label]

    return ranks



def compare_ranks(ranks, placeholder = '-'):
    P = [placeholder] * len(list(itertools.combinations(ranks, 2)))

    i = 0
    for a, b in itertools.combinations(ranks, 2):
        # print(a, b)
        if (a == placeholder and b == placeholder):
            P[i] = placeholder
        elif (a == placeholder and b != placeholder):
            P[i] = 0
        elif (a != placeholder and b == placeholder):
            P[i] = 1
        elif (a > b):
            P[i] = 0
        elif (a < b):
            P[i] = 1
        i += 1

    return(P)


def get_tp_fp_tn_fn_ones(confusion_matrix):
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[1][0] + confusion_matrix[2][0]
    tn = confusion_matrix[1][1] + confusion_matrix[1][2] + confusion_matrix[2][1] + confusion_matrix[2][2]
    fn = confusion_matrix[0][1] + confusion_matrix[0][2]

    return tp, fp, tn, fn

def get_tp_fp_tn_fn_zeros(confusion_matrix):
    tp = confusion_matrix[1][1]
    fp = confusion_matrix[0][1] + confusion_matrix[2][1]
    tn = confusion_matrix[0][0] + confusion_matrix[0][2] + confusion_matrix[2][0] + confusion_matrix[2][2]
    fn = confusion_matrix[1][0] + confusion_matrix[1][2]

    return tp, fp, tn, fn

def get_tp_fp_tn_fn_placeholder(confusion_matrix):
    tp = confusion_matrix[2][2]
    fp = confusion_matrix[0][2] + confusion_matrix[1][2]
    tn = confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1]
    fn = confusion_matrix[2][0] + confusion_matrix[2][1]

    return tp, fp, tn, fn


def pre_rec_acc_f1(tp, fp, tn, fn):

    precision = (tp / (tp + fp)) if (tp + fp) != 0 else 0
    recall = (tp / (tp + fn)) if (tp + fn) != 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) != 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, accuracy, f1_score




def calculate_metrics(true_labels, pred_labels, labels):
    
    # store metrics scores to calculate average scores
    pre_list = []
    rec_list = []
    acc_list = []
    f1_list = []

    for i in range(len(true_labels)):
        # calculate label ranks and P array for the ground truth labels
        ranks_true = rank_labels(labels, true_labels[i])
        P_true = compare_ranks(ranks_true)

        # calculate label ranks and P array for the predicted labels
        ranks_pred = rank_labels(labels, pred_labels[i])
        P_pred = compare_ranks(ranks_pred)

        # get the confusion matrix
        cm = confusion_matrix(P_true, P_pred, labels=[1, 0, '-'])
        # print(cm)

        # get the TP, FP, TN, FN values for each label seperately (1, 0 and -)
        tp1, fp1, tn1, fn1 = get_tp_fp_tn_fn_ones(cm)
        # print(tp1, fp1, tn1, fn1)
        tp0, fp0, tn0, fn0 = get_tp_fp_tn_fn_zeros(cm)
        # print(tp0, fp0, tn0, fn0)
        tp_, fp_, tn_, fn_ = get_tp_fp_tn_fn_placeholder(cm)
        # print(tp_, fp_, tn_, fn_)

        # calculate precision, recall, accuracy and f1_score for each label seperately (1, 0 and -)
        pre_1, rec_1, acc_1, f1_1 = pre_rec_acc_f1(tp1, fp1, tn1, fn1)
        pre_0, rec_0, acc_0, f1_0 = pre_rec_acc_f1(tp0, fp0, tn0, fn0)
        pre_, rec_, acc_, f1_ = pre_rec_acc_f1(tp_, fp_, tn_, fn_)

        # add the metrics scores to the appropriate array
        pre_list.extend((pre_1, pre_0, pre_))
        rec_list.extend((rec_1, rec_0, rec_))
        acc_list.extend((acc_1, acc_0, acc_))
        f1_list.extend((f1_1, f1_0, f1_))



    return "{:.2f}".format(statistics.fmean(pre_list)), \
    "{:.2f}".format(statistics.fmean(rec_list)), \
    "{:.2f}".format(statistics.fmean(acc_list)), \
    "{:.2f}".format(statistics.fmean(f1_list))



