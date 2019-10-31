
'''
Logistic Regression with Stop words Removal and converting to lower case.
L2 Regularization with C parameter
Output : F1 score
'''
import sys
from sklearn import datasets
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def performanceMetrics(predicted, twenty_test):
    string = str(round(metrics.f1_score(twenty_test.target, predicted, average ='weighted'), 3))
    return string


def finalMethod(twenty_train, twenty_test):
    clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1),
                                             stop_words='english', token_pattern="[\w]+")),
                    ('classification', LogisticRegression(penalty='l2', dual=False, C=100))])

    twenty_train.data = [item.lower() for item in twenty_train.data]
    twenty_test.data = [item.lower() for item in twenty_test.data]
    clf.fit(twenty_train.data, twenty_train.target)

    predicted = clf.predict(twenty_test.data)

    # Metrics
    string = performanceMetrics(predicted, twenty_test)
    return string

if __name__ == '__main__':
    if (len(sys.argv) == 4):
        trainSet = sys.argv[1]
        testSet = sys.argv[2]
        outputFile = sys.argv[3]
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

    # To remove the Headers in training and test data
    for index, item in enumerate(twenty_train.data):
        val = item.find("\n\n")
        if(val > 0):
            twenty_train.data[index] = item[val:]

    for index, item in enumerate(twenty_test.data):
        val = item.find("\n\n")
        if (val > 0):
            twenty_test.data[index] = item[val:]

    string = finalMethod(twenty_train, twenty_test)

    #Writing to output file
    fhandle = open(outputFile, "w+")
    fhandle.write(string)
    fhandle.close()