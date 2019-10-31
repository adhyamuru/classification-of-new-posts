import sys
from sklearn import datasets, preprocessing
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier


# For plotting Learning curve for Unigram Representation
def plot_learning_curve(estimatorList, data, target, test, ylim=None, xlim=None):

    tvalue = twenty_train.target[0]
    for i in range(len(twenty_train.target)):
        if(tvalue != twenty_train.target[i]):
            split = i+1
            break

    colorList = ['blue', 'red', 'black', 'green']
    labelList = ['Naive Bayes', 'Logistic Regression', 'SVM', 'Random Forest']
    plt.figure()
    plt.title("Learning Curve for NB, LR, SVM, RF")
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel("Training data")
    plt.ylabel("F-1 Score")
    plt.grid()
    for color, clf, label in zip(colorList, estimatorList, labelList):
        f1values = []
        training_data_set = []
        for i in range(split, len(data), 200):
            training_data_set.append(i)
            # print(data[:i])
            clf.fit(data[:i], target[:i])
            predicted = clf.predict(test.data)
            f1values.append(metrics.f1_score(test.target, predicted, average='macro',
                                             labels=np.unique(predicted)))

        plt.plot(training_data_set, f1values, 'o-', color=color, label=label)
    plt.legend(loc="best")
    return plt

#Displaying the performance metrics - precison, recall and f1 score
def performanceMetrics(name, ngram, predicted, twenty_test):
    if (ngram == (1, 1)):
        string = name + ",UB,"
    else:
        string = name + ",BB,"
    string += str(round(metrics.precision_score(twenty_test.target, predicted, average='macro'), 3)) + "," \
              + str(round(metrics.recall_score(twenty_test.target, predicted, average='macro'), 3)) + "," \
              + str(round(metrics.f1_score(twenty_test.target, predicted, average='macro'), 3))

    return string


def naiveBayes(ngram, twenty_train, twenty_test):
    # Pipeline for Tokenization and applying Multinomial Naive Bayes
    clf = Pipeline([('vect', CountVectorizer(ngram_range=ngram, binary=True, token_pattern="[\w]+")),
                    ('clf', MultinomialNB()), ])
    clf.fit(twenty_train.data, twenty_train.target)

    # Testing the trained classifier
    predicted = clf.predict(twenty_test.data)

    # Metrics
    string = performanceMetrics("NB", ngram, predicted, twenty_test)

    if (ngram == (1, 1)):
        return string, clf

    return string


def logisticRegression(ngram, twenty_train, twenty_test):
    # Pipeline for Tokenization and applying Logistic Regression
    clf = Pipeline([('vect', CountVectorizer(ngram_range=ngram, binary=True, token_pattern="[\w]+")),
                    ('clf', LogisticRegression()), ])
    clf.fit(twenty_train.data, twenty_train.target)

    # Testing the trained classifier
    predicted = clf.predict(twenty_test.data)

    # Metrics
    string = performanceMetrics("LR", ngram, predicted, twenty_test)

    if (ngram == (1, 1)):
        return string, clf

    return string


def supportVectorMachine(ngram, twenty_train, twenty_test):
    # Pipeline for Tokenization and applying SVM
    clf = Pipeline([('vect', CountVectorizer(ngram_range=ngram, binary=True)),
                    ('clf', LinearSVC()), ])

    clf.fit(twenty_train.data, twenty_train.target)

    # Testing the trained classifier
    predicted = clf.predict(twenty_test.data)

    # Metrics
    string = performanceMetrics("SVM", ngram, predicted, twenty_test)

    if (ngram == (1, 1)):
        return string, clf

    return string


def randomForest(ngram, twenty_train, twenty_test):
    clf = Pipeline([('vect', CountVectorizer(ngram_range=ngram, binary=True)),
                    ('clf', RandomForestClassifier(n_estimators=10, max_depth=None,
                                        min_samples_split=2, random_state=0)), ])

    clf.fit(twenty_train.data, twenty_train.target)

    # Testing the trained classifier
    predicted = clf.predict(twenty_test.data)

    # Metrics
    string = performanceMetrics("RF", ngram, predicted, twenty_test)

    if (ngram == (1, 1)):
        return string, clf

    return string

# Main function - Parsing the command line arguments
if __name__ == '__main__':
    if (len(sys.argv) == 5):
        trainSet = sys.argv[1]
        testSet = sys.argv[2]
        outputFile = sys.argv[3]
        displayLc = int(sys.argv[4])
    else:
        print("Invalid arguments\n")
        exit()

    # Loading Training data
    # twenty_train is the data bunch which is constructed as a dictionary like object
    twenty_train = sklearn.datasets.load_files(trainSet, description=None, categories=None, load_content=True,
                                               shuffle=False, encoding='utf-8', decode_error='ignore', random_state=0)

    # Loading Test data
    # twenty_test is the data for testing
    twenty_test = sklearn.datasets.load_files(testSet, description=None, categories=None, load_content=True,
                                              shuffle=False, encoding='utf-8', decode_error='ignore', random_state=0)
    # target_names contains the list of category names

    # The code below removes the Headers in training and test data
    for index, item in enumerate(twenty_train.data):
        val = item.find("\n\n")
        if (val > 0):
            twenty_train.data[index] = item[val:]

    for index, item in enumerate(twenty_test.data):
        val = item.find("\n\n")
        if (val > 0):
            twenty_test.data[index] = item[val:]

    list = []

    string, clf1 = naiveBayes((1, 1), twenty_train, twenty_test)
    list.append(string)
    string = naiveBayes((2, 2), twenty_train, twenty_test)
    list.append(string)

    string, clf2 = logisticRegression((1, 1), twenty_train, twenty_test)
    list.append(string)
    string = logisticRegression((2, 2), twenty_train, twenty_test)
    list.append(string)

    string, clf3 = supportVectorMachine((1, 1), twenty_train, twenty_test)
    list.append(string)
    string = supportVectorMachine((2, 2), twenty_train, twenty_test)
    list.append(string)

    string, clf4 = randomForest((1, 1), twenty_train, twenty_test)
    list.append(string)
    string = randomForest((2, 2), twenty_train, twenty_test)
    list.append(string)

    clfList = []

    fhandle = open(outputFile, "w+")
    for string in list:
        fhandle.write(string + "\n")
    fhandle.close()

    if (displayLc == 1):
        clfList = [clf1, clf2, clf3,clf4]
        plot_learning_curve(clfList, twenty_train.data, twenty_train.target,
                            twenty_test, ylim=(0.2, 1.01))
        plt.show()
