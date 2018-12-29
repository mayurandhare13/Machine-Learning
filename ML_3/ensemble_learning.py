import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier


def plotGraph(x_axis, y_axis, title, x_lab, y_lab, filename):
    plt.plot(x_axis, y_axis)
    plt.title(title, loc = 'center')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.savefig(filename)
    plt.close()


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

    y_train = enc.fit_transform(y_train.values.ravel())
    y_dev = enc.fit_transform(y_dev.values.ravel())
    y_test = enc.fit_transform(y_test.values.ravel())

    return y_train, y_dev, y_test

def sklearn_tree():

    encoded_data = read_data()
    train_data, dev_data, test_data, train_label, dev_label, test_label = separate_data(encoded_data)
    y_train, y_dev, y_test = encode_label_features(train_label, dev_label, test_label)


    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(train_data, y_train)
    y_pred = classifier.predict(test_data)
    
    print("="*50)
    print("SKlearn Decision Train Score:- ", classifier.score(train_data, y_train) * 100)   
    print("SKlearn Decision Test Score:- ", classifier.score(test_data, y_test) * 100)
    print("="*50)


    depth = [1, 2, 3, 5, 10]
    bag_size = [10, 20, 40, 60, 80, 100]

    for d in depth:

        bagging_score_train = []
        bagging_score_dev = []
        bagging_score_test = []

        for b_size in bag_size:
            bg = BaggingClassifier(DecisionTreeClassifier(max_depth=d), max_samples= 1.0, max_features = 1.0, n_estimators = b_size)
            bg.fit(train_data, y_train)

            score_train = bg.score(train_data, y_train) * 100
            bagging_score_train.append(score_train)

            score_dev = bg.score(dev_data, y_dev) * 100
            bagging_score_dev.append(score_dev)
    
            score_test = bg.score(test_data, y_test) * 100
            bagging_score_test.append(score_test)

            print("Bagging Decision Train Score, depth {0}, bag size {1}:- {2:2.3f}%".format(d, b_size, score_train))
            print("Bagging Decision Dev Score, depth {0}, bag size {1}:- {2:2.3f}%".format(d, b_size, score_dev))
            print("Bagging Decision Test Score, depth {0}, bag size {1}:- {2:2.3f}%".format(d, b_size, score_test))
            print("-"*60)

        print("="*70)

        plt.plot(bag_size, bagging_score_train, color = 'blue', label = "train score")
        plt.plot(bag_size, bagging_score_dev, color = "red", label = "dev score")
        plt.plot(bag_size, bagging_score_test, color = "green", label = "test score")
        plt.title("bagging depth {}".format(d), loc = 'center')
        plt.xlabel("bag size")
        plt.ylabel("score")
        plt.legend(loc='upper left')
        plt.savefig("bagging_depth_{}.png".format(d))
        plt.close()


    depth = [1, 2, 3]
    for d in depth:

        boosting_score_train = []
        boosting_score_dev = []
        boosting_score_test = []

        for b_size in bag_size:
            adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=d), n_estimators = b_size, learning_rate = 1)
            adb.fit(train_data, y_train)

            score_train = adb.score(train_data, y_train) * 100
            boosting_score_train.append(score_train)

            score_dev = adb.score(dev_data, y_dev) * 100
            boosting_score_dev.append(score_dev)
    
            score_test = adb.score(test_data, y_test) * 100
            boosting_score_test.append(score_test)

            print("Boosting Decision Train Score, depth {0}, bag size {1}:- {2:2.3f}%".format(d, b_size, score_train))
            print("Boosting Decision Dev Score, depth {0}, bag size {1}:- {2:2.3f}%".format(d, b_size, score_dev))
            print("Boosting Decision Test Score, depth {0}, bag size {1}:- {2:2.3f}%".format(d, b_size, score_test))
            print("-"*60)

        print("="*70)

        plt.plot(bag_size, boosting_score_train, color = 'blue', label = "train score")
        plt.plot(bag_size, boosting_score_dev, color = "red", label = "dev score")
        plt.plot(bag_size, boosting_score_test, color = "green", label = "test score")
        plt.title("boosting depth {}".format(d), loc = 'center')
        plt.xlabel("bag size")
        plt.ylabel("score")
        plt.legend(loc='upper left')
        plt.savefig("boosting_depth_{}.png".format(d))
        plt.close()

if __name__ == '__main__':

    sklearn_tree()