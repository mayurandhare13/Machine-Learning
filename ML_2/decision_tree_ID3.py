import numpy as np
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


header = ["age", "workclass", "education", "marital_status", "occupation", "race", "sex", "hours", "country", "income"]

train_file = open('income.train.txt')
dev_file = open('income.dev.txt')
test_file = open('income.test.txt')

train_data = pd.read_csv(train_file, delimiter=',', header=None).values
dev_data = pd.read_csv(dev_file, delimiter=',', header=None).values
test_data = pd.read_csv(test_file, delimiter=',', header=None).values


train_file.close()
dev_file.close()
test_file.close()


def information_gain(left, right, current_uncertainty):
    p = len(left) / (len(left)+len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)

def entropy(rows):
    ent = 0.0
    counts = class_counts(rows)
    print(counts)
    for keys in counts:
        p = (counts[keys] / len(rows))
        ent = -p*math.log(p, 2) 
    print(ent)
    return ent

def unique_vals(rows, col):
    return set([row[col] for row in rows])


def class_counts(rows):
    counts = {} #dictionary of labels -> count
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    print(counts)
    return counts


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:
    
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "is %s %s %s?" % (header[self.column], condition, str(self.value))


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def find_best_split(rows):
    
    best_gain = 0
    best_question = None # keep train of the feature / value that produced it
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])

        for val in values:
            question = Question(col, val)

            true_rows, false_rows = partition(rows, question)

            # skip the split if it doesn't divide the dataset
            if(len(true_rows)==0 or len(false_rows)==0):
                continue
            
            # calc info gain from split
            gain = information_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


def classify(row, node):

    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

def read_data():
    train_file = open('income.train.txt')
    dev_file = open('income.dev.txt')
    test_file = open('income.test.txt')

    train_data = pd.read_csv(train_file, delimiter=',', header=None)
    dev_data = pd.read_csv(dev_file, delimiter=',', header=None)
    test_data = pd.read_csv(test_file, delimiter=',', header=None)

    cs = pd.concat([train_data, dev_data, test_data], keys=[0, 1, 2])

    encoded_data = pd.get_dummies(cs, columns=[1, 2, 3, 4, 5, 6, 8])


    train_file.close()
    dev_file.close()
    test_file.close()

    return encoded_data

def separate_data(encoded_data):

    label = encoded_data[[9]]

    encoded_data = encoded_data.drop([9], axis=1)

    train_data = encoded_data.xs(0)
    dev_data = encoded_data.xs(1)
    test_data = encoded_data.xs(2)

    train_label = label.xs(0)
    dev_label = label.xs(1)
    test_label = label.xs(2)

    return train_data, dev_data, test_data, train_label, dev_label, test_label

def encode_label_features(*ylabels):  # process data into numeric format
    enc = LabelEncoder()
    (y_train, y_dev, y_test) = (ylabels)

    y_train = enc.fit_transform(y_train)
    y_dev = enc.fit_transform(y_dev)
    y_test = enc.fit_transform(y_test)

    # y_train[y_train == 0] = -1
    # y_dev[y_dev == 0] = -1
    # y_test[y_test == 0] = -1

    return y_train, y_dev, y_test

def sklearn_tree():

    dataset = pd.read_csv('income.train.txt', delimiter=',', header=None)
    dataset_test = pd.read_csv('income.test.txt', delimiter=',', header=None)
    X = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 8]]
    y = dataset.iloc[:, 9].values
    y_tes = dataset_test.iloc[:, 9].values

    enc1 = LabelEncoder()
    y_train = enc1.fit_transform(y)

    enc2 = LabelEncoder()
    y_test_2 = enc2.fit_transform(y_tes)

    cs = pd.concat([dataset, dataset_test], keys=[0, 1])
    encoded_data = pd.get_dummies(cs, columns=[1, 2, 3, 4, 5, 6, 8])
    encoded_data = encoded_data.drop([9], axis=1)
    train_data = encoded_data.xs(0)
    test_data = encoded_data.xs(1)


    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(train_data, y_train)
    y_pred = classifier.predict(test_data)
    print(y_pred)
    mistakes = 0
    for i in range(len(y_pred)):
        if y_pred[i] != y_test_2[i]:
            mistakes +=1
    print(mistakes)
    print(len(test_data))
    print("SKlearn Decision Test Score:- ", classifier.score(test_data, y_test_2) * 100, file=f)


if __name__ == '__main__':

    f = open("output_nb.txt", "a+")
    print("-"*75, file = f)
    my_tree = build_tree(train_data)

    for row in test_data[:20]:
        print ("Actual: %s. Predicted: %s" %
               (row[-1], print_leaf(classify(row, my_tree))), file=f)
    sklearn_tree()
    f.close()