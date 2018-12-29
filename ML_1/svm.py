import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import confusion_matrix

f = open("output.txt", "a+")

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

    return (encoded_data, train_data, dev_data)


def encode_label_features(*ylabels):  # process data into numeric format
    enc = LabelEncoder()
    (y_train, y_dev, y_test) = (ylabels)

    y_train = enc.fit_transform(y_train)
    y_dev = enc.fit_transform(y_dev)
    y_test = enc.fit_transform(y_test)

    y_train[y_train == 0] = -1
    y_dev[y_dev == 0] = -1
    y_test[y_test == 0] = -1

    return y_train, y_dev, y_test


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


def train_svm(X, Y, c):
    train_accs = []
    dev_accs = []
    test_accs = []

    accuracy = {}

    for i in c:
        start = time.process_time()

        linear_kernel = SVC(C=i, kernel='linear')
        linear_kernel.fit(X[0], Y[0])
        score_train = linear_kernel.score(X[0], Y[0])
        score_dev = linear_kernel.score(X[1], Y[1])
        score_test = linear_kernel.score(X[2], Y[2])

        accuracy[i] = score_dev
        train_accs.append(score_train)
        dev_accs.append(score_dev)
        test_accs.append(score_test)
        end = time.process_time()
        print("i = {}, time takes = {} min".format(i, (end - start) / 60), file=f)

    best_acc = [key for (key, value) in accuracy.items() if value == max(accuracy.values())][0]

    return best_acc, train_accs, dev_accs, test_accs

def train_svm_cm(train, dev, test_x, test_y, c_best):
    combine = pd.concat([train, dev], keys = [0, 1])
    combine_y = combine[[9]]
    combine_x = combine.drop([9], axis = 1)
    combine_x = pd.get_dummies(combine_x, columns = [1,2,3,4,5,6,8])

    enc = LabelEncoder()
    combine_y = enc.fit_transform(combine_y)
    combine_y[combine_y == 0] = -1

    combine_x = combine_x.iloc[:].values

    classifier = SVC(C=c_best, kernel='linear')
    train_c_svm = classifier.fit(combine_x, combine_y).score(combine_x, combine_y)
    test_c_svm = classifier.score(test_x, test_y)
    y_predict = classifier.predict(test_x)
    print("SVM C classifier ", train_c_svm, test_c_svm, file = f)
    cnf_matrix = confusion_matrix(test_y, y_predict)
    print("confusion matrix", cnf_matrix, file = f)

    return cnf_matrix


def train_svm_poly(X, Y, c):
    num_SV = []
    train_score = []
    test_score = []
    dev_score = []

    for deg in [2, 3, 4]:
        start = time.process_time()

        poly_kernel = SVC(C = c, kernel='poly', degree=deg)
        poly_kernel.fit(X[0], Y[0])
        num_SV.append(sum(poly_kernel.n_support_))
        train_score.append(poly_kernel.score(X[0], Y[0]))
        dev_score.append(poly_kernel.score(X[1], Y[1]))
        test_score.append(poly_kernel.score(X[2], Y[2]))

        end = time.process_time()
        print("deg = {}, time takes = {} min".format(deg, (end - start) / 60), file=f)
    return num_SV, train_score, dev_score, test_score


if __name__ == "__main__":
    (encoded_data,train,dev) = read_data()
    train_x, dev_x, test_x, train_y, dev_y, test_y = separate_data(encoded_data)
    train_y, dev_y, test_y = encode_label_features(train_y, dev_y, test_y)

    x_features = (train_x, dev_x, test_x)
    y_labels = (train_y, dev_y, test_y)

    c = [10**-4, 10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3, 10**4]

    print("\nLinear SVM\n", file=f)
    best_acc, train_accs, dev_accs, test_accs = train_svm(x_features, y_labels, c)
    print("Linear SVM train accuracy: ", train_accs, file=f)
    print("Linear SVM dev accuracy: ", dev_accs, file=f)
    print("Linear SVM test accuracy: ", test_accs, file=f)
    plotGraph(list(c), train_accs, "Train Linear SVM", "Hyper-Parameter", "train accuracy", "SVM Train.png")
    plotGraph(list(c), dev_accs, "Dev Linear SVM", "Hyper-Parameter","dev accuracy", "SVM Dev.png")
    plotGraph(list(c), test_accs, "Dev Linear SVM", "Hyper-Parameter","test accuracy", "SVM Test.png")

    cnf_matrix = train_svm_cm(train, dev, test_x, test_y, c)

    print("\nSVM Polynomial\n", file=f)
    num_SV, train_score, dev_score, test_score = train_svm_poly(x_features, y_labels, best_acc)
    print("\nPolynomial SVM train accuracy: ", train_score, file=f)
    print("\nPolynomial SVM dev accuracy: ", dev_score, file=f)
    print("\nPolynomial SVM test accuracy: ", test_score, file=f)
    plotGraph(train_score, num_SV, "Train Poly SVM", "Poly train accuracy", "Number of Support Vectors", "Poly SVM Train.png")
    plotGraph(dev_score, num_SV, "Dev Poly SVM", "Poly dev accuracy", "Number of Support Vectors", "Poly SVM Dev.png")
    plotGraph(test_score, num_SV, "Test Poly SVM", "Poly test accuracy", "Number of Support Vectors", "Poly SVM Test.png")

    f.close()