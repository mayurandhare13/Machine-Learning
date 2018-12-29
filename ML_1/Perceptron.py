import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

f = open("output.txt", 'w+')

def plotGraph(x_axis, y_axis, title, x_lab, y_lab, filename):
    plt.plot(x_axis, y_axis)
    plt.title(title, loc = 'center')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.savefig(filename)
    plt.close()


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


def cal_tao(x_t, y_t, wx):
    x_sq = np.dot(x_t, x_t)
    tow = (1 - (y_t * wx) ) / x_sq
    return tow


def passive_aggression(X, y, x_dev, y_dev, x_test, y_test):
    train_accs = []
    dev_accs = []
    test_accs = []
    w = np.zeros(X.shape[1])
    mistakes = []

    for i in range(5):
        count = 0
        for t in range(X.shape[0]):
            wx = np.dot(w, X[t])
            y_predict = np.sign(wx)
            if y_predict == 0.00:
                y_predict = -1

            if y_predict != y[t]:
                count = count + 1
                tao = cal_tao(X[t], y[t], wx)
                w = w + tao * (y[t] * X[t])
        mistakes.append(count)

        train_acc = test_perceptron(X, y, w)
        train_accs.append(train_acc)

        dev_acc = test_perceptron(x_dev, y_dev, w)
        dev_accs.append(dev_acc)

        test_acc = test_perceptron(x_test, y_test, w)
        test_accs.append(test_acc)

    return train_accs, dev_accs, test_accs, mistakes


def perceptron(X, y, x_dev, y_dev, x_test, y_test):
    train_accs = []
    dev_accs = []
    test_accs = []
    w = np.zeros(X.shape[1])
    mistakes = []

    for i in range(5):
        count = 0
        for t in range(X.shape[0]):
            wx = np.dot(w, X[t])
            y_predict = np.sign(wx)
            if y_predict == 0.00:
                y_predict = -1

            if y_predict != y[t]:
                count = count + 1
                w = w + 1 * (y[t] * X[t])
        mistakes.append(count)
    
        train_acc = test_perceptron(X, y, w)
        train_accs.append(train_acc)

        dev_acc = test_perceptron(x_dev, y_dev, w)
        dev_accs.append(dev_acc)

        test_acc = test_perceptron(x_test, y_test, w)
        test_accs.append(test_acc)

    return train_accs, dev_accs, test_accs, mistakes


def average_perceptron(X, y, x_dev, y_dev, x_test, y_test):
    train_accs = []
    dev_accs = []
    test_accs = []
    w = np.zeros(X.shape[1])
    u = np.zeros(X.shape[1])
    b = 0
    beta = 0
    c = 1

    start = time.process_time()
    for i in range(5):
        for t in range(X.shape[0]):
            wx = np.dot(w, X[t]) + b
            if wx <= 0:
                w = w + (y[t] * X[t])
                b = b + y[t]
                u = u + c*y[t]*X[t]
                beta = beta + c*y[y]
            c = c + 1
    end = time.process_time()

    train_acc = test_perceptron(X, y, w)
    train_accs.append(train_acc)

    dev_acc = test_perceptron(x_dev, y_dev, w)
    dev_accs.append(dev_acc)

    test_acc = test_perceptron(x_test, y_test, w)
    test_accs.append(test_acc)
    print("time for avg perceptron: {}".format(end - start), file=f)

    w_final = w - (u/c)
    bias = b - (beta/c)
    return w_final, bias, train_accs, dev_accs, test_accs


def perceptron_naive(X, y, x_dev, y_dev, x_test, y_test):
    train_accs = []
    dev_accs = []
    test_accs = []
    count = 0
    w = np.zeros(X.shape[1])
    wsum = 0
    mistakes = []
    ETA = 1

    start = time.process_time()

    for i in range(5):
        mistake = 0
        for t in range(X.shape[0]):
            wx = np.dot(w, X[t])
            y_predict = np.sign(wx)
            if y_predict == 0.00:
                y_predict = -1

            if y_predict != y[t]:
                wsum = wsum + w
                count = count + 1
                mistake = mistake + 1
                w = w + ETA * (y[t] * X[t])

        mistakes.append(mistake)

        train_acc = test_perceptron(X, y, w)
        train_accs.append(train_acc)

        dev_acc = test_perceptron(x_dev, y_dev, w)
        dev_accs.append(dev_acc)

        test_acc = test_perceptron(x_test, y_test, w)
        test_accs.append(test_acc)
   
    end = time.process_time()
    print("time for naive avg perceptron: {} sec".format(end - start), file=f)

    wavg = wsum / X.shape[0]

    return train_accs, dev_accs, test_accs


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


def test_perceptron(X, y, w):
    mistake = 0

    for t in range(X.shape[0]):
        wx = np.dot(w, X[t])
        y_predict = np.sign(wx)
        if y_predict == 0.00:
            y_predict = -1

        if y_predict != y[t]:
            mistake = mistake + 1

    return (1 - (mistake / X.shape[0]))*100


def gen_learning_curve(train_x,test_x,dev_x,train_y,test_y,dev_y, function_name):
    count=5000
    dev_gen_acu=[]
    test_gen_acu=[]
    train_gen_acu=[]
    nexamples=[]

    while count<=25000:
        temp_train_x=train_x[:count,:]
        temp_train_y=train_y[:count]
        if(function_name == "passive_aggression"):
            (dev_acu, test_acu, train_acu, mistake) = passive_aggression(temp_train_x, temp_train_y, dev_x, dev_y, test_x, test_y)
        elif(function_name == "perceptron"):
            (dev_acu, test_acu, train_acu, mistake) = perceptron(temp_train_x, temp_train_y, dev_x, dev_y, test_x, test_y)
        dev_gen_acu.append(dev_acu[-1])
        test_gen_acu.append(test_acu[-1])
        train_gen_acu.append(train_acu[-1])
        nexamples.append(count)
        count=count+5000
    print(dev_gen_acu)
    return (dev_gen_acu, test_gen_acu, train_gen_acu, nexamples)


if __name__ == "__main__":
    encoded_data = read_data()
    train_x, dev_x, test_x, train_y, dev_y, test_y = separate_data(encoded_data)

    x_train = train_x.iloc[:, :].values
    x_test = test_x.iloc[:, :].values
    x_dev = dev_x.iloc[:, :].values

    print("We have {} number of features in training data".format(x_train.shape[1]))

    y_train, y_dev, y_test = encode_label_features(train_y, dev_y, test_y)

    dev_gen_acu, test_gen_acu, train_gen_acu, nexamples = gen_learning_curve(x_train, x_test, x_dev, y_train, y_test, y_dev, "passive_aggression")
    plotGraph(list(nexamples), list(train_gen_acu), "General PA Train", "Iterations", "accuracy", "General PA Train.png")
    plotGraph(list(nexamples), list(dev_gen_acu), "General PA Dev", "Iterations", "accuracy", "General PA Dev.png")
    plotGraph(list(nexamples), list(test_gen_acu), "General PA Test", "Iterations", "accuracy", "General PA Test.png")
    
    dev_gen_acu, test_gen_acu, train_gen_acu, nexamples = gen_learning_curve(x_train, x_test, x_dev, y_train, y_test, y_dev, "perceptron")
    plotGraph(list(nexamples), list(train_gen_acu), "General Perceptron Train", "Iterations", "accuracy", "General Perceptron Train.png")
    plotGraph(list(nexamples), list(dev_gen_acu), "General Perceptron Dev", "Iterations", "accuracy", "General Perceptron Dev.png")
    plotGraph(list(nexamples), list(test_gen_acu), "General Perceptron Test", "Iterations", "accuracy", "General Perceptron Test.png")

    train_acc, dev_acc, test_acc, mistakes = perceptron(x_train, y_train, x_dev, y_dev, x_test, y_test)
    print("\nTraining Accuracy Perceptron: ",train_acc, file = f)
    print("\nDev Accuracy Perceptron: ",dev_acc, file = f)
    print("\nTest Accuracy Perceptron: ",test_acc, file = f)
    plotGraph(range(5), train_acc, "Perceptron Train", "Iterations", "accuracy", "Train Perceptron.png")
    plotGraph(range(5), test_acc, "Perceptron Test", "Iterations", "accuracy", "Test Perceptron.png")
    plotGraph(range(5), dev_acc, "Perceptron Dev", "Iterations", "accuracy", "Dev Perceptron.png")
    plotGraph(range(5), mistakes, "Perceptron Learning Curve", "Iterations", "mistakes", "Perceptron Learning Curve.png")

    avg_wt, avg_bias, av_train_score, av_dev_score, av_test_score = average_perceptron(x_train, y_train, x_dev, y_dev, x_test, y_test)
    print("\nAverage Perceptron Training Score: ",av_train_score, file = f)
    print("\nAverage Perceptron Dev Score: ", av_dev_score, file = f)
    print("\nAverage Perceptron Test Score: ", av_test_score, file = f)

    n_train_acc, n_dev_acc, n_test_acc = perceptron_naive(x_train, y_train, x_dev, y_dev, x_test, y_test)
    print("\nAverage Perceptron Naive Training Score: ",n_train_acc, file = f)
    print("\nAverage Perceptron Naive Dev Score: ", n_dev_acc, file = f)
    print("\nAverage Perceptron Naive Test Score: ", n_test_acc, file = f)

    # plotGraph(range(1, 6), train_acc, "Perceptron Naive Train", "Iterations", "accuracy", "Train Perceptron Naive.png")
    # plotGraph(range(1, 6), test_acc, "Perceptron Naive Test", "Iterations", "accuracy", "Test Perceptron Naive.png")
    # plotGraph(range(1, 6), dev_acc, "Perceptron Naive Dev", "Iterations", "accuracy", "Dev Perceptron Naive.png")

    train_acc, dev_acc, test_acc, mistakes = passive_aggression(x_train, y_train, x_dev, y_dev, x_test, y_test)
    print("\nPassive Aggression Training Score: ",train_acc, file = f)
    print("\nPassive Aggression Dev Score: ", dev_acc, file = f)
    print("\nPassive Aggression Test Score: ", test_acc, file = f)

    plotGraph(range(1, 6), train_acc, "PA Train Accuracy Curve", "Iterations", "accuracy", "Train PA.png")
    plotGraph(range(1, 6), test_acc, "PA Test Accuracy Curve", "Iterations", "accuracy", "Test PA.png")
    plotGraph(range(1, 6), dev_acc, "PA Dev  Accuracy Curve", "Iterations", "accuracy", "Dev PA.png")
    plotGraph(range(5), mistakes, "PA Learning Curve", "Iterations", "mistakes", "PA Learning Curve.png")

    ans = '''
    As we can see from the learning graph, Perceptron reduced mistakes per iteration more quickly than Passive Aggression.
    And Perceptron converges quickly.
    '''
    print(ans, file = f)
    f.close()