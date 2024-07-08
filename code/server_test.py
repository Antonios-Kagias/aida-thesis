import jaro
import string

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.dataset import load_dataset
from skmultilearn.dataset import load_from_arff
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import random
import arff
import statistics
import itertools
from sklearn.metrics import confusion_matrix

# Load a sample dataset
dataset = 'emotions'            # 6 labels
# dataset = 'scene'               # 6 labels
# dataset = 'yeast'               # 14 labels
# dataset = 'birds'               # 19 labels
# dataset = 'Image.arff'          # 5 labels
# dataset = 'Water-quality.arff'  # 14 labels
# dataset = 'CHD_49.arff'          # 6 labels
# dataset = 'mediamill'           # 101 labels
# dataset = 'CAL500.arff'          # 174 labels


# X_train, y_train, feature_names, label_names = load_dataset(dataset, 'train')
# X_test, y_test, _, _ = load_dataset(dataset, 'test')

X, y, feature_names, label_names = load_dataset(dataset, 'undivided')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# X, y = load_from_arff(
#    dataset,
#     label_count=174,
#     label_location='end',
#     load_sparse=False
# )
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)




def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):

    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []

    for i in range(len(y_true)):

        set_true = set( np.where(y_true[i])[0] )
        # set_true = set( np.atleast_1d(y_true).nonzero()[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        # set_pred = set( np.atleast_1d(y_pred).nonzero()[0] )

        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))

        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) /   float(len(set_true.union(set_pred)))

        #print('tmp_a: {0}'.format(tmp_a))

        acc_list.append(tmp_a)

    return np.mean(acc_list)





def strings_to_numpy_array(strings, dataset):

    # Define the characters we're interested in
    if (dataset == 'emotions' or dataset == 'scene' or dataset == 'CHD_49.arff'):
        labels = ['A', 'B', 'C', 'D', 'E', 'F']
    elif (dataset == 'yeast' or dataset == 'Water-quality.arff'):
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
    elif (dataset == 'birds'):
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
    elif (dataset == 'Image.arff'):
        labels = ['A', 'B', 'C', 'D', 'E']
    elif (dataset == 'mediamill'):
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
                  'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                  'y', 'z', '~', '!', '@', '#', '$', '%', '^', '&',
                  '*', '(', ')', '-', '=', '_', '+', '/', '|', '[',
                  ']', '{', '}', ';', ':', ',', '.', '<', '>', '?',
                  '\u1E00', '\u1E01', '\u1E02', '\u1E03', '\u1E04', '\u1E05', '\u1E06', '\u1E07', '\u1E08', '\u1E09',
                  '\u1E0A', '\u1E0B', '\u1E0C', '\u1E0D', '\u1E0E', '\u1E0F', '\u1E10', '\u1E11', '\u1E12', '\u1E13',
                  '\u1E14']
    elif (dataset == 'CAL500.arff'):
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
                  'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                  'y', 'z', '~', '!', '@', '#', '$', '%', '^', '&',
                  '*', '(', ')', '-', '=', '_', '+', '/', '|', '[',
                  ']', '{', '}', ';', ':', ',', '.', '<', '>', '?',
                  '\u1E00', '\u1E01', '\u1E02', '\u1E03', '\u1E04', '\u1E05', '\u1E06', '\u1E07', '\u1E08', '\u1E09',
                  '\u1E0A', '\u1E0B', '\u1E0C', '\u1E0D', '\u1E0E', '\u1E0F', '\u1E10', '\u1E11', '\u1E12', '\u1E13',
                  '\u1E14', '\u1E15', '\u1E16', '\u1E17', '\u1E18', '\u1E19', '\u1E1A', '\u1E1B', '\u1E1C', '\u1E1D',
                  '\u1E1E', '\u1E1F', '\u1E20', '\u1E21', '\u1E22', '\u1E23', '\u1E24', '\u1E25', '\u1E26', '\u1E27',
                  '\u1E28', '\u1E29', '\u1E2A', '\u1E2B', '\u1E2C', '\u1E2D', '\u1E2E', '\u1E2F', '\u1E30', '\u1E31',
                  '\u1E32', '\u1E33', '\u1E34', '\u1E35', '\u1E36', '\u1E37', '\u1E38', '\u1E39', '\u1E3A', '\u1E3B',
                  '\u1E3C', '\u1E3D', '\u1E3E', '\u1E3F', '\u1E40', '\u1E41', '\u1E42', '\u1E43', '\u1E44', '\u1E45',
                  '\u1E46', '\u1E47', '\u1E48', '\u1E49', '\u1E4A', '\u1E4B', '\u1E4C', '\u1E4D', '\u1E4E', '\u1E4F',
                  '\u1E50', '\u1E51', '\u1E52', '\u1E53', '\u1E54', '\u1E55', '\u1E56', '\u1E57', '\u1E58', '\u1E59',
                  '\u1E5A', '\u1E5B', '\u1E5C', '\u1E5D']

    # Initialize the result array with zeros
    result = np.zeros((len(strings), len(labels)), dtype=int)

    # Iterate through each string in the input list
    for i, string in enumerate(strings):
        # Iterate through each character in the string
        for char in string:
            if char in labels:
                # Find the index of the character in the characters list
                index = labels.index(char)
                # Mark the presence of the character
                result[i][index] = 1

    return result




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



    return statistics.fmean(pre_list), statistics.fmean(rec_list),\
    statistics.fmean(acc_list), statistics.fmean(f1_list)








print("Dataset information")
print(f"Name: {dataset}")
print(f"X_train size: {X_train.shape[0]} ({np.round((X_train.shape[0] / (X_train.shape[0] + X_test.shape[0])) * 100, 2)}%)")
print(f"y_train size: {y_train.shape[0]}")
print(f"X_test size: {X_test.shape[0]} ({np.round((X_test.shape[0] / (X_train.shape[0] + X_test.shape[0])) * 100, 2)}%)")
print(f"y_test size: {y_test.shape[0]}")

ytest_array = y_test.toarray()
y_train_array = y_train.toarray()

if (dataset == 'emotions' or dataset == 'scene' or dataset == 'CHD_49.arff'):
    labels = ['A', 'B', 'C', 'D', 'E', 'F']
elif (dataset == 'yeast' or dataset == 'Water-quality.arff'):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
elif (dataset == 'birds'):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
elif (dataset == 'Image.arff'):
	labels = ['A', 'B', 'C', 'D', 'E']
elif (dataset == 'mediamill'):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
              'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
              'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
              'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
              'y', 'z', '~', '!', '@', '#', '$', '%', '^', '&',
              '*', '(', ')', '-', '=', '_', '+', '/', '|', '[',
              ']', '{', '}', ';', ':', ',', '.', '<', '>', '?',
              '\u1E00', '\u1E01', '\u1E02', '\u1E03', '\u1E04', '\u1E05', '\u1E06', '\u1E07', '\u1E08', '\u1E09',
              '\u1E0A', '\u1E0B', '\u1E0C', '\u1E0D', '\u1E0E', '\u1E0F', '\u1E10', '\u1E11', '\u1E12', '\u1E13',
              '\u1E14']
elif (dataset == 'CAL500.arff'):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
              'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
              'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
              'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
              'y', 'z', '~', '!', '@', '#', '$', '%', '^', '&',
              '*', '(', ')', '-', '=', '_', '+', '/', '|', '[',
              ']', '{', '}', ';', ':', ',', '.', '<', '>', '?',
              '\u1E00', '\u1E01', '\u1E02', '\u1E03', '\u1E04', '\u1E05', '\u1E06', '\u1E07', '\u1E08', '\u1E09',
              '\u1E0A', '\u1E0B', '\u1E0C', '\u1E0D', '\u1E0E', '\u1E0F', '\u1E10', '\u1E11', '\u1E12', '\u1E13',
              '\u1E14', '\u1E15', '\u1E16', '\u1E17', '\u1E18', '\u1E19', '\u1E1A', '\u1E1B', '\u1E1C', '\u1E1D',
              '\u1E1E', '\u1E1F', '\u1E20', '\u1E21', '\u1E22', '\u1E23', '\u1E24', '\u1E25', '\u1E26', '\u1E27',
              '\u1E28', '\u1E29', '\u1E2A', '\u1E2B', '\u1E2C', '\u1E2D', '\u1E2E', '\u1E2F', '\u1E30', '\u1E31',
              '\u1E32', '\u1E33', '\u1E34', '\u1E35', '\u1E36', '\u1E37', '\u1E38', '\u1E39', '\u1E3A', '\u1E3B',
              '\u1E3C', '\u1E3D', '\u1E3E', '\u1E3F', '\u1E40', '\u1E41', '\u1E42', '\u1E43', '\u1E44', '\u1E45',
              '\u1E46', '\u1E47', '\u1E48', '\u1E49', '\u1E4A', '\u1E4B', '\u1E4C', '\u1E4D', '\u1E4E', '\u1E4F',
              '\u1E50', '\u1E51', '\u1E52', '\u1E53', '\u1E54', '\u1E55', '\u1E56', '\u1E57', '\u1E58', '\u1E59',
              '\u1E5A', '\u1E5B', '\u1E5C', '\u1E5D']


# Function to replace positions with letters
def replace_with_letters(row):
    return [labels[i] for i, val in enumerate(row) if val == 1]

# Apply the function to each row of each array
ytest_to_letters = [replace_with_letters(row) for row in ytest_array]

# Randomize label order
ytest_final = []

# Make sure that converting to letters always has the same result
random.seed(42)

for j in range(len(ytest_to_letters)):
    ytest_final.append(''.join(random.sample(ytest_to_letters[j], len(ytest_to_letters[j]))))

print(ytest_final)
print(y_train_array)


###################################################### Classifier 1 ######################################################


from collections import defaultdict

def letter_counts_by_position(strings):
    if not strings:
        return {}

    # Determine the maximum length of the strings
    max_length = max(len(string) for string in strings)

    # Initialize a dictionary to store the counts of each letter at each position
    letter_counts = defaultdict(lambda: [0] * max_length)

    # Initialize a dictionary to store the total lengths of strings each letter appears in
    letter_lengths = defaultdict(int)

    # Iterate through each string
    for string in strings:
        string_length = len(string)
        for i, char in enumerate(string):
            letter_counts[char][i] += 1
            letter_lengths[char] += string_length

    return dict(letter_counts), dict(letter_lengths)

def create_unique_majority_string(strings):
    letter_counts, letter_lengths = letter_counts_by_position(strings)

    # Determine the length of the string created
    len_temp = []
    for x in range(len(strings)):
        len_temp.append(len(strings[x]))
    length = round(statistics.mean(len_temp))
    # max_length = max(len(string) for string in strings)

    used_letters = set()
    result = []

    for i in range(length):
        # Collect the count of each letter at position i
        position_counts = {char: counts[i] for char, counts in letter_counts.items()}
        # Sort letters by their counts in descending order
        sorted_letters = sorted(position_counts.items(), key=lambda item: (item[1], -letter_lengths[item[0]]), reverse=True)

        # Find the first letter that hasn't been used yet
        for char, count in sorted_letters:
            if char not in used_letters:
                result.append(char)
                used_letters.add(char)
                break

    return ''.join(result)






from sklearn.neighbors import KNeighborsClassifier

# average jaro-winkler scores
mean_jw_1 = []
# average hamming loss scores
mean_hl_1 = []
# average precision scores
mean_pre_1 = []
# average recall scores
mean_rec_1 = []
# average accuracy scores
mean_acc_1 = []
# average f1 scores
mean_f1_1 = []

def run_clf1():
    for chosen_k in range(1, 31):
        clf1 = KNeighborsClassifier(n_neighbors=chosen_k)
        # square euclidean!!!!!
        clf1.fit(X_train, y_train)
        # y_train_array = y_train.toarray()
        predictions = clf1.predict(X_test)

        # Get neighbors of X_test instances
        neighbors_temp = clf1.kneighbors(X_test, return_distance=False)

        # Convert neighbors' labels to letters
        if (dataset == 'emotions' or dataset == 'scene' or dataset == 'CHD_49.arff'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F']
        elif (dataset == 'yeast' or dataset == 'Water-quality.arff'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        elif (dataset == 'birds'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
        elif (dataset == 'Image.arff'):
            labels = ['A', 'B', 'C', 'D', 'E']
        elif (dataset == 'mediamill'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
					'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
					'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
					'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
					'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
					'y', 'z', '~', '!', '@', '#', '$', '%', '^', '&',
					'*', '(', ')', '-', '=', '_', '+', '/', '|', '[',
					']', '{', '}', ';', ':', ',', '.', '<', '>', '?',
					'\u1E00', '\u1E01', '\u1E02', '\u1E03', '\u1E04', '\u1E05', '\u1E06', '\u1E07', '\u1E08', '\u1E09',
					'\u1E0A', '\u1E0B', '\u1E0C', '\u1E0D', '\u1E0E', '\u1E0F', '\u1E10', '\u1E11', '\u1E12', '\u1E13',
					'\u1E14']
        elif (dataset == 'CAL500.arff'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
					'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
					'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
					'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
					'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
					'y', 'z', '~', '!', '@', '#', '$', '%', '^', '&',
					'*', '(', ')', '-', '=', '_', '+', '/', '|', '[',
					']', '{', '}', ';', ':', ',', '.', '<', '>', '?',
					'\u1E00', '\u1E01', '\u1E02', '\u1E03', '\u1E04', '\u1E05', '\u1E06', '\u1E07', '\u1E08', '\u1E09',
					'\u1E0A', '\u1E0B', '\u1E0C', '\u1E0D', '\u1E0E', '\u1E0F', '\u1E10', '\u1E11', '\u1E12', '\u1E13',
					'\u1E14', '\u1E15', '\u1E16', '\u1E17', '\u1E18', '\u1E19', '\u1E1A', '\u1E1B', '\u1E1C', '\u1E1D',
					'\u1E1E', '\u1E1F', '\u1E20', '\u1E21', '\u1E22', '\u1E23', '\u1E24', '\u1E25', '\u1E26', '\u1E27',
					'\u1E28', '\u1E29', '\u1E2A', '\u1E2B', '\u1E2C', '\u1E2D', '\u1E2E', '\u1E2F', '\u1E30', '\u1E31',
					'\u1E32', '\u1E33', '\u1E34', '\u1E35', '\u1E36', '\u1E37', '\u1E38', '\u1E39', '\u1E3A', '\u1E3B',
					'\u1E3C', '\u1E3D', '\u1E3E', '\u1E3F', '\u1E40', '\u1E41', '\u1E42', '\u1E43', '\u1E44', '\u1E45',
					'\u1E46', '\u1E47', '\u1E48', '\u1E49', '\u1E4A', '\u1E4B', '\u1E4C', '\u1E4D', '\u1E4E', '\u1E4F',
					'\u1E50', '\u1E51', '\u1E52', '\u1E53', '\u1E54', '\u1E55', '\u1E56', '\u1E57', '\u1E58', '\u1E59',
					'\u1E5A', '\u1E5B', '\u1E5C', '\u1E5D']
        neighbors_temp_to_letters = []
        for i in range(len(neighbors_temp)):
            # print(y_train_array[neighbors_temp[0][i]])
            temp_replacement = [replace_with_letters(row) for row in y_train_array[neighbors_temp[i]]]
            neighbors_temp_to_letters.append(temp_replacement)
        # print(neighbors_temp_to_letters)

        # Add all neighbors' labels to the same list (neighbors_final)
        # Randomize label order
        neighbors_final = []
        for j in range(len(neighbors_temp_to_letters)):
            for k in range(chosen_k):
                neighbors_final.append(''.join(random.sample(neighbors_temp_to_letters[j][k], len(neighbors_temp_to_letters[j][k]))))
        # print(neighbors_final)
        # test that number of neighbors is correct: (chosen_k) * (X_test size)
        # print(len(neighbors_final))

        # Run classifier 1 on the dataset
        clf1_predictions = []
        for i in range(0, len(neighbors_final), chosen_k):
            temp_list = []
            for k in range(chosen_k):
                temp_list.append(neighbors_final[i+k])
            # print(temp_list)
            counts, lengths = letter_counts_by_position(temp_list)
            result = create_unique_majority_string(temp_list)
            clf1_predictions.append(result)
        # print(f"Dataset:\t{ytest_final}")
        # print(f"Predictions:\t{clf1_predictions}")

        jw_scores_1 = []
        for i in range(len(ytest_final)):
            # calculate_Jaro_Winkler_2(ytest_final[i], pred_final[i])
            jw_scores_1.append(jaro.jaro_winkler_metric(ytest_final[i], clf1_predictions[i]))
        
        
        clf1_predictions_np = strings_to_numpy_array(clf1_predictions, dataset)


        # print(f"Average Jaro-Winkler score with k = {chosen_k}: {statistics.fmean(jw_scores_1)}")
        # print(f"Hamming loss with k = {chosen_k}: {metrics.hamming_loss(ytest_final, clf1_predictions)}")
        
        # print(f"Average Jaro-Winkler distance with k = {chosen_k}: {1-statistics.fmean(jw_scores_1)}")
        # print(f"Hamming score distance with k = {chosen_k}: {1-hamming_score(ytest_array, clf1_predictions_np)}")
        
        print(f"Computing Jaro-Winkler distance and Hamming score distance with k = {chosen_k}")

        # mean_jw_1.append(statistics.fmean(jw_scores_1))
        # mean_hl_1.append(metrics.hamming_loss(ytest_final, clf1_predictions))
        
        mean_jw_1.append(1-statistics.fmean(jw_scores_1))
        # mean_hl_1.append(1-hamming_score(ytest_array, clf1_predictions_np))
        mean_hl_1.append(metrics.hamming_loss(ytest_array, clf1_predictions_np))
        
        # get precision, recall, accuracy, f1 score
        avg_pre, avg_rec, avg_acc, avg_f1 = calculate_metrics(ytest_final, clf1_predictions, labels)
        mean_pre_1.append(avg_pre)
        mean_rec_1.append(avg_rec)
        mean_acc_1.append(avg_acc)
        mean_f1_1.append(avg_f1)

    print(f"Average precision:\t{statistics.fmean(mean_pre_1):.2f}")
    print(f"Average recall:\t\t{statistics.fmean(mean_rec_1):.2f}")
    print(f"Average accuracy:\t{statistics.fmean(mean_acc_1):.2f}")
    print(f"Average F1 score:\t{statistics.fmean(mean_f1_1):.2f}")
    
    print(f"Average Jaro-Winkler distance:\t{statistics.fmean(mean_jw_1):.2f}")
    # print(f"Average Hamming score distance:\t{statistics.fmean(mean_hl_1):.2f}")
    print(f"Average Hamming loss:\t{statistics.fmean(mean_hl_1):.2f}")





run_clf1()






# importing the required libraries
import matplotlib.pyplot as plt
import numpy as np

# define data values
x = np.arange(1,31)
y1 = mean_jw_1
y2 = mean_hl_1


#plt.plot(x, y1, ls='-', c='orangered', label='clf1 jaro-winkler')
#plt.plot(x, y2, ls='-', c='navy', label='clf1 hamming loss')
#plt.xlabel("Selected k")
#plt.ylabel("Scores")
#plt.title(f"Metrics' scores with clf1 in dataset '{dataset}'")
#plt.ylim(0, 1)
#plt.legend(loc='lower left', ncols=3, fontsize='x-small')
#plt.show()

# linestyles
# 'solid'     '-'
# 'dotted'    ':'
# 'dashed'    '--'
# 'dashdot'   '-.'



###################################################### Classifier 2 ######################################################

from collections import defaultdict

def letter_counts_by_position(strings):
    if not strings:
        return {}

    # Determine the maximum length of the strings
    max_length = max(len(string) for string in strings)

    # Initialize a dictionary to store the counts of each letter at each position
    letter_counts = defaultdict(lambda: [0] * max_length)

    # Initialize a dictionary to store the total lengths of strings each letter appears in
    letter_lengths = defaultdict(int)

    # Iterate through each string
    for string in strings:
        string_length = len(string)
        for i, char in enumerate(string):
            letter_counts[char][i] += 1
            letter_lengths[char] += string_length

    return dict(letter_counts), dict(letter_lengths)

def create_influenced_string(strings):
    if not strings:
        return ''

    letter_counts, letter_lengths = letter_counts_by_position(strings)

    # max_length = max(len(string) for string in strings)
    # Determine the length of the string created
    len_temp = []
    for x in range(len(strings)):
        len_temp.append(len(strings[x]))
    length = round(statistics.mean(len_temp))

    used_letters = set()
    result = []

    for i in range(length):
        # Filter strings based on the current result
        filtered_strings = [s for s in strings if len(s) > i and s[i] not in used_letters]

        letter_counts, letter_lengths = letter_counts_by_position(strings)

        # Recalculate counts and lengths for the current position
        current_letter_counts = defaultdict(int)
        current_letter_lengths = defaultdict(int)

        for string in filtered_strings:
            if len(string) > i:
                char = string[i]
                current_letter_counts[char] += 1
                current_letter_lengths[char] += len(string)

        if not current_letter_counts:
            break

        # Collect the count of each letter at position i
        position_counts = {char: counts[i] for char, counts in letter_counts.items()}
        # Sort letters by their counts in descending order
        sorted_letters = sorted(position_counts.items(), key=lambda item: (item[1], -letter_lengths[item[0]]), reverse=True)


        # Sort by occurrence count (descending) and then by total length (ascending)
        # sorted_letters = sorted(current_letter_counts.keys(), key=lambda c: (current_letter_counts[c], -current_letter_lengths[c]))

        # Choose the letter that satisfies the conditions
        # for char in sorted_letters:
        #     if char not in used_letters:
        #         result.append(char)
        #         used_letters.add(char)
        #         # Update the list of strings to consider for the next position
        #         strings = [s for s in strings if len(s) > i and s[i] == char]
        #         break

        for char, count in sorted_letters:
            if char not in used_letters:
                result.append(char)
                used_letters.add(char)
                strings = [s for s in strings if len(s) > i and s[i] == char]
                # print(strings)
                break

    return ''.join(result)


from sklearn.neighbors import KNeighborsClassifier

# average jaro-winkler scores
mean_jw_2 = []
# average hamming loss scores
mean_hl_2 = []
# average precision scores
mean_pre_2 = []
# average recall scores
mean_rec_2 = []
# average accuracy scores
mean_acc_2 = []
# average f1 scores
mean_f1_2 = []

def run_clf2():
    for chosen_k in range(1, 31):
        clf2 = KNeighborsClassifier(n_neighbors=chosen_k)
        clf2.fit(X_train, y_train)
        # y_train_array = y_train.toarray()
        predictions = clf2.predict(X_test)

        # Get neighbors of X_test instances
        neighbors_temp = clf2.kneighbors(X_test, return_distance=False)

        # Convert neighbors' labels to letters
        if (dataset == 'emotions' or dataset == 'scene' or dataset == 'CHD_49.arff'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F']
        elif (dataset == 'yeast' or dataset == 'Water-quality.arff'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        elif (dataset == 'birds'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
        elif (dataset == 'Image.arff'):
            labels = ['A', 'B', 'C', 'D', 'E']
        elif (dataset == 'mediamill'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
                      'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z', '~', '!', '@', '#', '$', '%', '^', '&',
                      '*', '(', ')', '-', '=', '_', '+', '/', '|', '[',
                      ']', '{', '}', ';', ':', ',', '.', '<', '>', '?',
                      '\u1E00', '\u1E01', '\u1E02', '\u1E03', '\u1E04', '\u1E05', '\u1E06', '\u1E07', '\u1E08', '\u1E09',
                      '\u1E0A', '\u1E0B', '\u1E0C', '\u1E0D', '\u1E0E', '\u1E0F', '\u1E10', '\u1E11', '\u1E12', '\u1E13',
                      '\u1E14']
        elif (dataset == 'CAL500.arff'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
					'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
					'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
					'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
					'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
					'y', 'z', '~', '!', '@', '#', '$', '%', '^', '&',
					'*', '(', ')', '-', '=', '_', '+', '/', '|', '[',
					']', '{', '}', ';', ':', ',', '.', '<', '>', '?',
					'\u1E00', '\u1E01', '\u1E02', '\u1E03', '\u1E04', '\u1E05', '\u1E06', '\u1E07', '\u1E08', '\u1E09',
					'\u1E0A', '\u1E0B', '\u1E0C', '\u1E0D', '\u1E0E', '\u1E0F', '\u1E10', '\u1E11', '\u1E12', '\u1E13',
					'\u1E14', '\u1E15', '\u1E16', '\u1E17', '\u1E18', '\u1E19', '\u1E1A', '\u1E1B', '\u1E1C', '\u1E1D',
					'\u1E1E', '\u1E1F', '\u1E20', '\u1E21', '\u1E22', '\u1E23', '\u1E24', '\u1E25', '\u1E26', '\u1E27',
					'\u1E28', '\u1E29', '\u1E2A', '\u1E2B', '\u1E2C', '\u1E2D', '\u1E2E', '\u1E2F', '\u1E30', '\u1E31',
					'\u1E32', '\u1E33', '\u1E34', '\u1E35', '\u1E36', '\u1E37', '\u1E38', '\u1E39', '\u1E3A', '\u1E3B',
					'\u1E3C', '\u1E3D', '\u1E3E', '\u1E3F', '\u1E40', '\u1E41', '\u1E42', '\u1E43', '\u1E44', '\u1E45',
					'\u1E46', '\u1E47', '\u1E48', '\u1E49', '\u1E4A', '\u1E4B', '\u1E4C', '\u1E4D', '\u1E4E', '\u1E4F',
					'\u1E50', '\u1E51', '\u1E52', '\u1E53', '\u1E54', '\u1E55', '\u1E56', '\u1E57', '\u1E58', '\u1E59',
					'\u1E5A', '\u1E5B', '\u1E5C', '\u1E5D']
        neighbors_temp_to_letters = []
        for i in range(len(neighbors_temp)):
            # print(y_train_array[neighbors_temp[0][i]])
            temp_replacement = [replace_with_letters(row) for row in y_train_array[neighbors_temp[i]]]
            neighbors_temp_to_letters.append(temp_replacement)
        # print(neighbors_temp_to_letters)

        # Add all neighbors' labels to the same list (neighbors_final)
        # Randomize label order
        neighbors_final = []
        for j in range(len(neighbors_temp_to_letters)):
            for k in range(chosen_k):
                neighbors_final.append(''.join(random.sample(neighbors_temp_to_letters[j][k], len(neighbors_temp_to_letters[j][k]))))
        # print(neighbors_final)
        # test that number of neighbors is correct: (chosen_k) * (X_test size)
        # print(len(neighbors_final))

        # Run classifier 1 on the dataset
        clf2_predictions = []
        for i in range(0, len(neighbors_final), chosen_k):
            temp_list = []
            for k in range(chosen_k):
                temp_list.append(neighbors_final[i+k])
            # print(temp_list)
            # counts, lengths = letter_counts_by_position(temp_list)
            result = create_influenced_string(temp_list)
            clf2_predictions.append(result)
        # print(f"Dataset:\t{ytest_final}")
        # print(f"Predictions:\t{clf2_predictions}")

        jw_scores_2 = []
        for i in range(len(ytest_final)):
            # calculate_Jaro_Winkler_2(ytest_final[i], pred_final[i])
            jw_scores_2.append(jaro.jaro_winkler_metric(ytest_final[i], clf2_predictions[i]))
        
        clf2_predictions_np = strings_to_numpy_array(clf2_predictions, dataset)


        # print(f"Average Jaro-Winkler score with k = {chosen_k}: {statistics.fmean(jw_scores_2)}")
        # print(f"Hamming loss with k = {chosen_k}: {metrics.hamming_loss(ytest_final, clf2_predictions)}")
        
        # print(f"Average Jaro-Winkler distance with k = {chosen_k}: {1-statistics.fmean(jw_scores_2)}")
        # print(f"Hamming score distance with k = {chosen_k}: {1-hamming_score(ytest_array, clf2_predictions_np)}")
        
        print(f"Computing Jaro-Winkler distance and Hamming score distance with k = {chosen_k}")

        # mean_jw_2.append(statistics.fmean(jw_scores_2))
        # mean_hl_2.append(metrics.hamming_loss(ytest_final, clf2_predictions))
        
        mean_jw_2.append(1-statistics.fmean(jw_scores_2))
        mean_hl_2.append(metrics.hamming_loss(ytest_array, clf2_predictions_np))
        
        # get precision, recall, accuracy, f1 score
        avg_pre, avg_rec, avg_acc, avg_f1 = calculate_metrics(ytest_final, clf2_predictions, labels)
        mean_pre_2.append(avg_pre)
        mean_rec_2.append(avg_rec)
        mean_acc_2.append(avg_acc)
        mean_f1_2.append(avg_f1)

    print(f"Average precision:\t{statistics.fmean(mean_pre_2):.2f}")
    print(f"Average recall:\t\t{statistics.fmean(mean_rec_2):.2f}")
    print(f"Average accuracy:\t{statistics.fmean(mean_acc_2):.2f}")
    print(f"Average F1 score:\t{statistics.fmean(mean_f1_2):.2f}")
    
    print(f"Average Jaro-Winkler distance:\t{statistics.fmean(mean_jw_2):.2f}")
    # print(f"Average Hamming score distance:\t{statistics.fmean(mean_hl_2):.2f}")
    print(f"Average Hamming loss:\t{statistics.fmean(mean_hl_2):.2f}")


run_clf2()


# importing the required libraries
import matplotlib.pyplot as plt
import numpy as np

# define data values
x = np.arange(1,31)
y11 = mean_jw_1
y12 = mean_hl_1
y21 = mean_jw_2
y22 = mean_hl_2


#plt.plot(x, y11, ls='-', c='orangered', label='clf1 jaro-winkler')
#plt.plot(x, y12, ls='-', c='navy', label='clf1 hamming loss')
#plt.plot(x, y21, ls='--', c='orangered', label='clf2 jaro-winkler')
#plt.plot(x, y22, ls='--', c='navy', label='clf2 hamming loss')
#plt.xlabel("Selected k")
#plt.ylabel("Scores")
#plt.title(f"Metrics' scores with clf1, clf2 in dataset '{dataset}'")
#plt.ylim(0, 1)
#plt.legend(loc='lower left', ncols=3, fontsize='x-small')
#plt.show()

# linestyles
# 'solid'     '-'
# 'dotted'    ':'
# 'dashed'    '--'
# 'dashdot'   '-.'



###################################################### Classifier 3 ######################################################

from collections import defaultdict
import random, statistics

def letter_counts_by_position(strings):
    if not strings:
        return {}

    # Determine the maximum length of the strings
    max_length = max(len(string) for string in strings)

    # Initialize a dictionary to store the counts of each letter at each position
    letter_counts = defaultdict(lambda: [0] * max_length)

    # Initialize a dictionary to store the total lengths of strings each letter appears in
    letter_lengths = defaultdict(int)

    # Iterate through each string
    for string in strings:
        string_length = len(string)
        for i, char in enumerate(string):
            letter_counts[char][i] += 1
            letter_lengths[char] += string_length

    return dict(letter_counts), dict(letter_lengths)

def create_right_shifted_string(strings):
    letter_counts, letter_lengths = letter_counts_by_position(strings)

    # Determine the length of the string created
    len_temp = []
    for x in range(len(strings)):
        len_temp.append(len(strings[x]))
    length = round(statistics.mean(len_temp))
    # max_length = max(len(string) for string in strings)

    used_letters = set()
    result = []

    for i in range(length):
        letter_counts, letter_lengths = letter_counts_by_position(strings)
        # Collect the count of each letter at position i
        position_counts = {char: counts[i] for char, counts in letter_counts.items()}
        # Sort letters by their counts in descending order
        sorted_letters = sorted(position_counts.items(), key=lambda item: (item[1], -letter_lengths[item[0]]), reverse=True)

        # Find the first letter that hasn't been used yet
        for char, count in sorted_letters:
            if char not in used_letters:
                result.append(char)
                used_letters.add(char)
                break

        for i in range(len(strings)):
            if(not strings[i].strip().startswith(char)):
                # print(strings[i])
                # print('true')
                new = strings[i].rjust(len(strings[i])+1, ' ')
                # print(new)
                strings[i] = new


    return ''.join(result).strip()


from sklearn.neighbors import KNeighborsClassifier

# average jaro-winkler scores
mean_jw_3 = []
# average hamming loss scores
mean_hl_3 = []
# average precision scores
mean_pre_3 = []
# average recall scores
mean_rec_3 = []
# average accuracy scores
mean_acc_3 = []
# average f1 scores
mean_f1_3 = []

def run_clf3():
    for chosen_k in range(1, 31):
        clf3 = KNeighborsClassifier(n_neighbors=chosen_k)
        clf3.fit(X_train, y_train)
        # y_train_array = y_train.toarray()
        predictions = clf3.predict(X_test)

        # Get neighbors of X_test instances
        neighbors_temp = clf3.kneighbors(X_test, return_distance=False)

        # Convert neighbors' labels to letters
        if (dataset == 'emotions' or dataset == 'scene' or dataset == 'CHD_49.arff'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F']
        elif (dataset == 'yeast' or dataset == 'Water-quality.arff'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        elif (dataset == 'birds'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
        elif (dataset == 'Image.arff'):
            labels = ['A', 'B', 'C', 'D', 'E']
        elif (dataset == 'mediamill'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
                      'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z', '~', '!', '@', '#', '$', '%', '^', '&',
                      '*', '(', ')', '-', '=', '_', '+', '/', '|', '[',
                      ']', '{', '}', ';', ':', ',', '.', '<', '>', '?',
                      '\u1E00', '\u1E01', '\u1E02', '\u1E03', '\u1E04', '\u1E05', '\u1E06', '\u1E07', '\u1E08', '\u1E09',
                      '\u1E0A', '\u1E0B', '\u1E0C', '\u1E0D', '\u1E0E', '\u1E0F', '\u1E10', '\u1E11', '\u1E12', '\u1E13',
                      '\u1E14']
        elif (dataset == 'CAL500.arff'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
					'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
					'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
					'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
					'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
					'y', 'z', '~', '!', '@', '#', '$', '%', '^', '&',
					'*', '(', ')', '-', '=', '_', '+', '/', '|', '[',
					']', '{', '}', ';', ':', ',', '.', '<', '>', '?',
					'\u1E00', '\u1E01', '\u1E02', '\u1E03', '\u1E04', '\u1E05', '\u1E06', '\u1E07', '\u1E08', '\u1E09',
					'\u1E0A', '\u1E0B', '\u1E0C', '\u1E0D', '\u1E0E', '\u1E0F', '\u1E10', '\u1E11', '\u1E12', '\u1E13',
					'\u1E14', '\u1E15', '\u1E16', '\u1E17', '\u1E18', '\u1E19', '\u1E1A', '\u1E1B', '\u1E1C', '\u1E1D',
					'\u1E1E', '\u1E1F', '\u1E20', '\u1E21', '\u1E22', '\u1E23', '\u1E24', '\u1E25', '\u1E26', '\u1E27',
					'\u1E28', '\u1E29', '\u1E2A', '\u1E2B', '\u1E2C', '\u1E2D', '\u1E2E', '\u1E2F', '\u1E30', '\u1E31',
					'\u1E32', '\u1E33', '\u1E34', '\u1E35', '\u1E36', '\u1E37', '\u1E38', '\u1E39', '\u1E3A', '\u1E3B',
					'\u1E3C', '\u1E3D', '\u1E3E', '\u1E3F', '\u1E40', '\u1E41', '\u1E42', '\u1E43', '\u1E44', '\u1E45',
					'\u1E46', '\u1E47', '\u1E48', '\u1E49', '\u1E4A', '\u1E4B', '\u1E4C', '\u1E4D', '\u1E4E', '\u1E4F',
					'\u1E50', '\u1E51', '\u1E52', '\u1E53', '\u1E54', '\u1E55', '\u1E56', '\u1E57', '\u1E58', '\u1E59',
					'\u1E5A', '\u1E5B', '\u1E5C', '\u1E5D']
        neighbors_temp_to_letters = []
        for i in range(len(neighbors_temp)):
            # print(y_train_array[neighbors_temp[0][i]])
            temp_replacement = [replace_with_letters(row) for row in y_train_array[neighbors_temp[i]]]
            neighbors_temp_to_letters.append(temp_replacement)
        # print(neighbors_temp_to_letters)

        # Add all neighbors' labels to the same list (neighbors_final)
        # Randomize label order
        neighbors_final = []
        for j in range(len(neighbors_temp_to_letters)):
            for k in range(chosen_k):
                neighbors_final.append(''.join(random.sample(neighbors_temp_to_letters[j][k], len(neighbors_temp_to_letters[j][k]))))
        # print(neighbors_final)
        # test that number of neighbors is correct: (chosen_k) * (X_test size)
        # print(len(neighbors_final))

        # Run classifier 1 on the dataset
        clf3_predictions = []
        for i in range(0, len(neighbors_final), chosen_k):
            temp_list = []
            for k in range(chosen_k):
                temp_list.append(neighbors_final[i+k])
            # print(temp_list)
            # counts, lengths = letter_counts_by_position(temp_list)
            result = create_right_shifted_string(temp_list)
            clf3_predictions.append(result)
        # print(f"Dataset:\t{ytest_final}")
        # print(f"Predictions:\t{clf3_predictions}")

        jw_scores_3 = []
        for i in range(len(ytest_final)):
            # calculate_Jaro_Winkler_2(ytest_final[i], pred_final[i])
            jw_scores_3.append(jaro.jaro_winkler_metric(ytest_final[i], clf3_predictions[i]))
        
        clf3_predictions_np = strings_to_numpy_array(clf3_predictions, dataset)


        # print(f"Average Jaro-Winkler score with k = {chosen_k}: {statistics.fmean(jw_scores_3)}")
        # print(f"Hamming loss with k = {chosen_k}: {metrics.hamming_loss(ytest_final, clf3_predictions)}")
        
        # print(f"Average Jaro-Winkler distance with k = {chosen_k}: {1-statistics.fmean(jw_scores_3)}")
        # print(f"Hamming score distance with k = {chosen_k}: {1-hamming_score(ytest_array, clf3_predictions_np)}")
        
        print(f"Computing Jaro-Winkler distance and Hamming score distance with k = {chosen_k}")

        # mean_jw_3.append(statistics.fmean(jw_scores_3))
        # mean_hl_3.append(metrics.hamming_loss(ytest_final, clf3_predictions))
        
        mean_jw_3.append(1-statistics.fmean(jw_scores_3))
        # mean_hl_3.append(1-hamming_score(ytest_array, clf3_predictions_np))
        mean_hl_3.append(metrics.hamming_loss(ytest_array, clf3_predictions_np))
        
        # get precision, recall, accuracy, f1 score
        avg_pre, avg_rec, avg_acc, avg_f1 = calculate_metrics(ytest_final, clf3_predictions, labels)
        mean_pre_3.append(avg_pre)
        mean_rec_3.append(avg_rec)
        mean_acc_3.append(avg_acc)
        mean_f1_3.append(avg_f1)

    print(f"Average precision:\t{statistics.fmean(mean_pre_3):.2f}")
    print(f"Average recall:\t\t{statistics.fmean(mean_rec_3):.2f}")
    print(f"Average accuracy:\t{statistics.fmean(mean_acc_3):.2f}")
    print(f"Average F1 score:\t{statistics.fmean(mean_f1_3):.2f}")
    
    print(f"Average Jaro-Winkler distance:\t{statistics.fmean(mean_jw_3):.2f}")
    # print(f"Average Hamming score distance:\t{statistics.fmean(mean_hl_3):.2f}")
    print(f"Average Hamming loss:\t{statistics.fmean(mean_hl_3):.2f}")


run_clf3()


# importing the required libraries
import matplotlib.pyplot as plt
import numpy as np

# define data values
x = np.arange(1,31)
y11 = mean_jw_1
y12 = mean_hl_1
y21 = mean_jw_2
y22 = mean_hl_2
y31 = mean_jw_3
y32 = mean_hl_3


#plt.plot(x, y11, ls='-', c='orangered', label='clf1 jaro-winkler')
#plt.plot(x, y12, ls='-', c='navy', label='clf1 hamming loss')
#plt.plot(x, y21, ls='--', c='orangered', label='clf2 jaro-winkler')
#plt.plot(x, y22, ls='--', c='navy', label='clf2 hamming loss')
#plt.plot(x, y31, ls='-.', c='orangered', label='clf3 jaro-winkler')
#plt.plot(x, y32, ls='-.', c='navy', label='clf3 hamming loss')
#plt.xlabel("Selected k")
#plt.ylabel("Scores")
#plt.title(f"Metrics' scores with clf1, clf2, clf3 in dataset '{dataset}'")
#plt.ylim(0, 1)
#plt.legend(loc='lower left', ncols=3, fontsize='x-small')
#plt.show()

# linestyles
# 'solid'     '-'
# 'dotted'    ':'
# 'dashed'    '--'
# 'dashdot'   '-.'



###################################################### Classifier 4 ######################################################

def compare_medal_positions(a, b):
    # Compare each position (index) in descending order of importance
    for i in range(len(a[1])):
        if a[1][i] != b[1][i]:
            return b[1][i] - a[1][i]  # Higher count at a position wins
    # If counts are the same, use total length as the tiebreaker
    return a[0][1] - b[0][1]


def create_medal_string(strings):
    letter_counts, letter_lengths = letter_counts_by_position(strings)

    # Determine the length of the string created
    len_temp = []
    for x in range(len(strings)):
        len_temp.append(len(strings[x]))
    length = round(statistics.mean(len_temp))
    # max_length = max(len(string) for string in strings)

    # Create a list of (label, counts) tuples
    # counts_list = [(char, letter_counts) for char, letter_counts in letter_counts.items()]

    counts_list = [(char, counts, letter_lengths[char]) for char, counts in letter_counts.items()]

    # Sort the list based on the counts using the custom comparison function
    # sorted_counts = sorted(counts_list, key=lambda x: x[1], reverse=True)

    # Sort the list based on the counts and the total lengths using the custom comparison function
    sorted_counts = sorted(counts_list, key=lambda x: (x[1], -x[2]), reverse=True)
    # sorted_counts.sort(key=lambda x: (x[1], x[2]), cmp=lambda a, b: compare_medal_positions(a, b))

    # Create the result string based on the sorted order of labels
    result = [char for char, _, _ in sorted_counts]



    # used_letters = set()
    # result = []

    # for i in range(length):
    #     # Collect the count of each letter at position i
    #     position_counts = {char: counts[i] for char, counts in letter_counts.items()}
    #     # Sort letters by their counts in descending order
    #     sorted_letters = sorted(position_counts.items(), key=lambda item: (item[1], -letter_lengths[item[0]]), reverse=True)

    #     # Find the first letter that hasn't been used yet
    #     for char, count in sorted_letters:
    #         if char not in used_letters:
    #             result.append(char)
    #             used_letters.add(char)
    #             break


    if len(result) > length:
        # result = result[:length]
        return ''.join(result[:length]).strip()
    else:
        result = result + [''] * (length - len(result))



    return ''.join(result)


from sklearn.neighbors import KNeighborsClassifier

# average jaro-winkler scores
mean_jw_4 = []
# average hamming loss scores
mean_hl_4 = []
# average precision scores
mean_pre_4 = []
# average recall scores
mean_rec_4 = []
# average accuracy scores
mean_acc_4 = []
# average f1 scores
mean_f1_4 = []

def run_clf4():
    for chosen_k in range(1, 31):
        clf4 = KNeighborsClassifier(n_neighbors=chosen_k)
        clf4.fit(X_train, y_train)
        # y_train_array = y_train.toarray()
        predictions = clf4.predict(X_test)

        # Get neighbors of X_test instances
        neighbors_temp = clf4.kneighbors(X_test, return_distance=False)

        # Convert neighbors' labels to letters
        if (dataset == 'emotions' or dataset == 'scene' or dataset == 'CHD_49.arff'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F']
        elif (dataset == 'yeast' or dataset == 'Water-quality.arff'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        elif (dataset == 'birds'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
        elif (dataset == 'Image.arff'):
            labels = ['A', 'B', 'C', 'D', 'E']
        elif (dataset == 'mediamill'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                      'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
                      'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z', '~', '!', '@', '#', '$', '%', '^', '&',
                      '*', '(', ')', '-', '=', '_', '+', '/', '|', '[',
                      ']', '{', '}', ';', ':', ',', '.', '<', '>', '?',
                      '\u1E00', '\u1E01', '\u1E02', '\u1E03', '\u1E04', '\u1E05', '\u1E06', '\u1E07', '\u1E08', '\u1E09',
                      '\u1E0A', '\u1E0B', '\u1E0C', '\u1E0D', '\u1E0E', '\u1E0F', '\u1E10', '\u1E11', '\u1E12', '\u1E13',
                      '\u1E14']
        elif (dataset == 'CAL500.arff'):
            labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
					'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
					'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
					'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
					'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
					'y', 'z', '~', '!', '@', '#', '$', '%', '^', '&',
					'*', '(', ')', '-', '=', '_', '+', '/', '|', '[',
					']', '{', '}', ';', ':', ',', '.', '<', '>', '?',
					'\u1E00', '\u1E01', '\u1E02', '\u1E03', '\u1E04', '\u1E05', '\u1E06', '\u1E07', '\u1E08', '\u1E09',
					'\u1E0A', '\u1E0B', '\u1E0C', '\u1E0D', '\u1E0E', '\u1E0F', '\u1E10', '\u1E11', '\u1E12', '\u1E13',
					'\u1E14', '\u1E15', '\u1E16', '\u1E17', '\u1E18', '\u1E19', '\u1E1A', '\u1E1B', '\u1E1C', '\u1E1D',
					'\u1E1E', '\u1E1F', '\u1E20', '\u1E21', '\u1E22', '\u1E23', '\u1E24', '\u1E25', '\u1E26', '\u1E27',
					'\u1E28', '\u1E29', '\u1E2A', '\u1E2B', '\u1E2C', '\u1E2D', '\u1E2E', '\u1E2F', '\u1E30', '\u1E31',
					'\u1E32', '\u1E33', '\u1E34', '\u1E35', '\u1E36', '\u1E37', '\u1E38', '\u1E39', '\u1E3A', '\u1E3B',
					'\u1E3C', '\u1E3D', '\u1E3E', '\u1E3F', '\u1E40', '\u1E41', '\u1E42', '\u1E43', '\u1E44', '\u1E45',
					'\u1E46', '\u1E47', '\u1E48', '\u1E49', '\u1E4A', '\u1E4B', '\u1E4C', '\u1E4D', '\u1E4E', '\u1E4F',
					'\u1E50', '\u1E51', '\u1E52', '\u1E53', '\u1E54', '\u1E55', '\u1E56', '\u1E57', '\u1E58', '\u1E59',
					'\u1E5A', '\u1E5B', '\u1E5C', '\u1E5D']
        neighbors_temp_to_letters = []
        for i in range(len(neighbors_temp)):
            # print(y_train_array[neighbors_temp[0][i]])
            temp_replacement = [replace_with_letters(row) for row in y_train_array[neighbors_temp[i]]]
            neighbors_temp_to_letters.append(temp_replacement)
        # print(neighbors_temp_to_letters)

        # Add all neighbors' labels to the same list (neighbors_final)
        # Randomize label order
        neighbors_final = []
        for j in range(len(neighbors_temp_to_letters)):
            for k in range(chosen_k):
                neighbors_final.append(''.join(random.sample(neighbors_temp_to_letters[j][k], len(neighbors_temp_to_letters[j][k]))))
        # print(neighbors_final)
        # test that number of neighbors is correct: (chosen_k) * (X_test size)
        # print(len(neighbors_final))

        # Run classifier 4 on the dataset
        clf4_predictions = []
        for i in range(0, len(neighbors_final), chosen_k):
            temp_list = []
            for k in range(chosen_k):
                temp_list.append(neighbors_final[i+k])
            # print(temp_list)
            counts, lengths = letter_counts_by_position(temp_list)
            result = create_medal_string(temp_list)
            clf4_predictions.append(result)
        # print(f"Dataset:\t{ytest_final}")
        # print(f"Predictions:\t{clf4_predictions}")

        jw_scores_4 = []
        for i in range(len(ytest_final)):
            # calculate_Jaro_Winkler_2(ytest_final[i], pred_final[i])
            jw_scores_4.append(jaro.jaro_winkler_metric(ytest_final[i], clf4_predictions[i]))
        
        clf4_predictions_np = strings_to_numpy_array(clf4_predictions, dataset)


        # print(f"Average Jaro-Winkler score with k = {chosen_k}: {statistics.fmean(jw_scores_4)}")
        # print(f"Hamming loss with k = {chosen_k}: {metrics.hamming_loss(ytest_final, clf4_predictions)}")
        
        # print(f"Average Jaro-Winkler distance with k = {chosen_k}: {1-statistics.fmean(jw_scores_4)}")
        # print(f"Hamming score distance with k = {chosen_k}: {1-hamming_score(ytest_array, clf4_predictions_np)}")
        
        print(f"Computing Jaro-Winkler distance and Hamming score distance with k = {chosen_k}")

        # mean_jw_4.append(statistics.fmean(jw_scores_4))
        # mean_hl_4.append(metrics.hamming_loss(ytest_final, clf4_predictions))
        
        mean_jw_4.append(1-statistics.fmean(jw_scores_4))
        # mean_hl_4.append(1-hamming_score(ytest_array, clf4_predictions_np))
        mean_hl_4.append(metrics.hamming_loss(ytest_array, clf4_predictions_np))
        
        # get precision, recall, accuracy, f1 score
        avg_pre, avg_rec, avg_acc, avg_f1 = calculate_metrics(ytest_final, clf4_predictions, labels)
        mean_pre_4.append(avg_pre)
        mean_rec_4.append(avg_rec)
        mean_acc_4.append(avg_acc)
        mean_f1_4.append(avg_f1)

    print(f"Average precision:\t{statistics.fmean(mean_pre_4):.2f}")
    print(f"Average recall:\t\t{statistics.fmean(mean_rec_4):.2f}")
    print(f"Average accuracy:\t{statistics.fmean(mean_acc_4):.2f}")
    print(f"Average F1 score:\t{statistics.fmean(mean_f1_4):.2f}")
    
    print(f"Average Jaro-Winkler distance:\t{statistics.fmean(mean_jw_4):.2f}")
    # print(f"Average Hamming score distance:\t{statistics.fmean(mean_hl_4):.2f}")
    print(f"Average Hamming loss:\t{statistics.fmean(mean_hl_4):.2f}")


run_clf4()


# importing the required libraries
import matplotlib.pyplot as plt
import numpy as np

# define data values
x = np.arange(1,31)
y11 = mean_jw_1
y12 = mean_hl_1
y21 = mean_jw_2
y22 = mean_hl_2
y31 = mean_jw_3
y32 = mean_hl_3
y41 = mean_jw_4
y42 = mean_hl_4


plt.plot(x, y11, ls='-', c='orangered', label='clf1 jaro-winkler distance')
plt.plot(x, y12, ls='-', c='navy', label='clf1 hamming loss')
plt.plot(x, y21, ls='--', c='orangered', label='clf2 jaro-winkler distance')
plt.plot(x, y22, ls='--', c='navy', label='clf2 hamming loss')
plt.plot(x, y31, ls='-.', c='orangered', label='clf3 jaro-winkler distance')
plt.plot(x, y32, ls='-.', c='navy', label='clf3 hamming loss')
plt.plot(x, y41, ls=':', c='orangered', label='clf4 jaro-winkler distance')
plt.plot(x, y42, ls=':', c='navy', label='clf4 hamming loss')


plt.xlabel("Selected k")
plt.ylabel("Scores")
plt.title(f"Metrics' scores with clf1, clf2, clf3, clf4 in dataset '{dataset}'")
plt.ylim(0, 1)
plt.legend(loc='lower left', ncols=2, fontsize='xx-small')
# plt.show()
plt.savefig(f'{dataset}_1_2_3_4.png')

# linestyles
# 'solid'     '-'
# 'dotted'    ':'
# 'dashed'    '--'
# 'dashdot'   '-.'
# {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}




# print(f"Test: {ytest_final}")
# print(f"Pred: {clf1_pred}")






