import math
import operator
import time
import pandas as pd
from functools import reduce
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
f = open("output.txt", "w+")

class Naive_bayes:

    def __init__(self, vocab):
        self.vocab = vocab
        self.probability = {}

    def predict_dict(self, features, labels):
        label_0, label_1 = self.class_prob(labels)
        self.probability = self.feature_prob(features, labels, label_0, label_1)
        return self.probability, label_0/len(features), label_1/len(features)

    def class_prob(self, labels):
        label_0 = 0
        label_1 = 0
        for i, val in enumerate(labels):
            if val == 0:
                label_0 += 1
        
        label_1 = len(labels) - label_0
        return label_0, label_1


    def feature_prob(self, messages, labels, l_0_prob, l_1_prob):
        total_obs = len(messages)
        f = len(messages[0])
        feature_count_present = [0] * f
        feature_count_absent = [0] * f

        for feature in messages:
            for i, val in enumerate(feature):
                if val == 1:
                    feature_count_present[i] += 1
                else:
                    feature_count_absent[i] += 1

        f_count_label_0_present = [0] * f
        f_count_label_0_absent = [0] * f
        f_count_label_1_present = [0] * f
        f_count_label_1_absent = [0] * f

        for i, label in enumerate(labels):
            if label == 0:
                for i, val in enumerate(messages[i]):
                    if val == 1:
                        f_count_label_0_present[i] += 1
                    else:
                        f_count_label_0_absent[i] += 1

            elif label == 1:
                for i, val in enumerate(messages[i]):
                    if val == 1:
                        f_count_label_1_present[i] += 1
                    else:
                        f_count_label_1_absent[i] += 1

        result_l_0_present = []
        result_l_0_absent = []
        result_l_1_present = []
        result_l_1_absent = []
        for i in range(f):
            result_l_0_present.append((f_count_label_0_present[i]+1) / (feature_count_present[i]+2))
            result_l_0_absent.append((f_count_label_0_absent[i]+1) / (feature_count_absent[i]+2))
            result_l_1_present.append((f_count_label_1_present[i]+1) / (feature_count_present[i]+2))
            result_l_1_absent.append((f_count_label_1_absent[i]+1) / (feature_count_absent[i]+2))

        probabilities = {}

        for i in range(len(self.vocab)):
            word = {}
            word_present = {}
            word_absent = {}
            word_absent[0] = result_l_0_absent[i]
            word_absent[1] = result_l_1_absent[i]
            word_present[0] = result_l_0_present[i]
            word_present[1] = result_l_1_present[i]
            word[0] = word_absent
            word[1] = word_present
            probabilities[vocab[i]] = word

        return probabilities


    def predict_label(self, messages, dictionary: dict, class_1_prob: float, class_0_prob: float) -> int:
        future_prob = 1.0
        wise_prob = 1.0
        for word in messages.split():
            if word in dictionary:
                future_prob *= dictionary[word][1][1]
                wise_prob *= dictionary[word][1][0]

        future_prob *= class_1_prob
        wise_prob *= class_0_prob

        print("Future Prob. --> {}".format(future_prob), file=f)
        print("Wise Saying Prob. --> {}".format(wise_prob), file=f)

        if future_prob > wise_prob:
            return 1
        else:
            return 0


def sklearn_predict(X_train, Y_train, X_test, Y_test):
    classifier = MultinomialNB()
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)

    print("Confusion Matrix SKLEARN NAIVE BAYES", file=f)
    print(cm, file=f)

    print("SKLEARN NB Train Score:- ", classifier.score(X_train, Y_train) * 100, file=f)
    print("SKLEARN NB Test Score:- ", classifier.score(X_test, Y_test) * 100, file=f)
    print('-'*50, file=f)



def logistics(X_train, Y_train, X_test, Y_test):
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    
    cm = confusion_matrix(Y_test, Y_pred)

    print("Confusion Matrix LOGISTICS", file=f)
    print(cm, file=f)
    print("LOGISTIC Train Score:- ", classifier.score(X_train, Y_train) * 100, file=f)
    print("LOGISTIC Test Score:- ", classifier.score(X_test, Y_test) * 100, file=f)
    print('-'*50, file=f)


def read_files():
    stop_list = []
    with open("stoplist.txt", "r") as stop_file:
        stop_list = stop_file.read().split('\n')

    train_list = []
    with open("traindata.txt", "r") as train_file:
        train_list = train_file.read().split('\n')

    train_label_list = []
    with open("trainlabels.txt", "r") as train_label_file:
        train_label_list = train_label_file.read().split('\n')

    test_label_list = []
    with open("testlabels.txt", "r") as test_label_file:
        test_label_list = test_label_file.read().split('\n')

    test_list = []
    with open("testdata.txt", "r") as test_file:
        test_list = test_file.read().split('\n')

    return stop_list, train_list, train_label_list, test_list, test_label_list


def preprocess_test(data_list, label_list, vocab):
    data_set = []

    labels_set = list(map(int, label_list))

    for line in data_list:
        new_line = []
        for word in line.split():
            if word not in stop_list:
                new_line.append(''.join(word))
        data_set.append(" ".join(new_line))

    feature_sets = []
    for msg in data_set:
        msg_dict = dict.fromkeys(vocab, 0)
        for word in msg.split():
            if(word in msg_dict):
                msg_dict[word] = 1
        feature_sets.append(msg_dict)
    
    features = []
    for feature in feature_sets:
        sorted_set = sorted(feature.items(), key=operator.itemgetter(0))
        sorted_set_occ = [x[1] for x in sorted_set]
        features.append(sorted_set_occ)
    
    return features, labels_set, data_set


def preprocess(data_list, label_list):
    vocab = set()
    data_set = []

    for line in data_list:
        new_line = []
        for word in line.split():
            if word not in stop_list:
                vocab.add(word)
                new_line.append(''.join(word))

        data_set.append(" ".join(new_line))

    sorted_vocab = sorted(vocab)
    labels_set = list(map(int, label_list))

    feature_sets = []
    for msg in data_set:
        msg_dict = dict.fromkeys(vocab, 0)
        for word in msg.split():
            msg_dict[word] = 1
        feature_sets.append(msg_dict)
    
    features = []
    for feature in feature_sets:
        sorted_set = sorted(feature.items(), key=operator.itemgetter(0))
        sorted_set_occ = [x[1] for x in sorted_set]
        features.append(sorted_set_occ)
    
    return sorted_vocab, features, labels_set, data_set


def score(model: Naive_bayes, vocab: list, messages: list, labels: list, future_prob: float, wise_prob: float) -> float:

    num_mistakes = 0
    total_messages = len(messages)
    for i in range(len(messages)):
        message = messages[i]
        label = labels[i]
        print("\n", message, file=f)
        predict_label = model.predict_label(message, model.probability, future_prob, wise_prob)
        if predict_label != label:
            num_mistakes += 1

    score = (1 - (num_mistakes / total_messages)) * 100
    return score


if __name__ == "__main__":

    stop_list, train_list, train_label_list, test_list, test_label_list = read_files()
    vocab, train_features, train_labels, train_data = preprocess(train_list, train_label_list)

    test_features, test_label, test_data = preprocess_test(test_list, test_label_list, vocab)
    
    nb_model = Naive_bayes(vocab)
    probability, wise_prob, future_prob = nb_model.predict_dict(train_features, train_labels)
    

    train_accuracy = score(nb_model, vocab, train_data, train_labels, future_prob, wise_prob)
    test_accuracy = score(nb_model, vocab, test_data, test_label, future_prob, wise_prob)

    print('-'*50, file=f)
    print("\nTRAINING accuracy: {0:2.3f}%".format(train_accuracy), file=f)
    print("TESTING accuracy: {0:2.3f}%\n".format(test_accuracy), file=f)
    print('-'*50, file=f)

    logistics(train_features, train_labels, test_features, test_label)
    sklearn_predict(train_features, train_labels, test_features, test_label)
    f.close()