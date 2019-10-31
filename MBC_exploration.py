import sys
from sklearn import datasets
import numpy as np
import sklearn
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def performanceMetrics(name, predicted, twenty_test):
    string = name+ ","
    string += str(round(metrics.precision_score(twenty_test.target, predicted, average='macro'), 3)) + "," \
              + str(round(metrics.recall_score(twenty_test.target, predicted, average='macro'), 3)) + "," \
              + str(round(metrics.f1_score(twenty_test.target, predicted, average ='macro'), 3))

    return string

def lrCountVectorizer(name, twenty_train, twenty_test):
    # Pipeline for Tokenization and applying SVM
    clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1), token_pattern="[\w]+",)),
                     ('clf', LogisticRegression()),])

    clf.fit(twenty_train.data, twenty_train.target)

    # Testing the trained classifier
    predicted = clf.predict(twenty_test.data)

    # Metrics
    string = performanceMetrics(name, predicted, twenty_test)
    return string


def lrTfidfVectorizer(name, twenty_train, twenty_test):
    clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,1), token_pattern="[\w]+")),
                    ('clf', LogisticRegression()), ])

    clf.fit(twenty_train.data, twenty_train.target)

    predicted = clf.predict(twenty_test.data)

    # Metrics
    string = performanceMetrics(name, predicted, twenty_test)
    return string


def lrStopWords(name, twenty_train, twenty_test):
    clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,1), token_pattern="[\w]+",
                    stop_words = 'english')), ('clf', LogisticRegression(max_iter=1500)), ])

    twenty_train.data = [item.lower() for item in twenty_train.data]
    twenty_test.data = [item.lower() for item in twenty_test.data]
    clf.fit(twenty_train.data, twenty_train.target)

    predicted = clf.predict(twenty_test.data)

    # Metrics
    string = performanceMetrics(name, predicted, twenty_test)
    return string


def lrStemming(twenty_train, twenty_test):
    #print(twenty_train.data)
    dataList=[]
    porter_stemmer = PorterStemmer()
    dataList.append(twenty_train.data)
    dataList.append(twenty_test.data)
    for data in dataList:
        for index, item in enumerate(data):
            stemmed = ""
            #print("Item",index, item)
            list = item.split()
            for word in list:
                #print("Word is:",word)
                if(word.startswith('>')):
                    word = word[1:]
                stemmed += porter_stemmer.stem(word) + " "
            data[index] = stemmed
    #print(twenty_train.data)
    return twenty_train, twenty_test


    #svmTfidfVectorizer(ngram, twenty_train, twenty_test)


def featureSelection1(name, twenty_train, twenty_test, c=None):
    if(c == None):
        clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1), token_pattern="[\w]+",
                    stop_words = 'english')),
                    ('feature_selection', LogisticRegression(penalty="l1", dual=False)),
                    ])
    else:
        clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), token_pattern="[\w]+",
                                                 stop_words='english')),
                        ('classification', LogisticRegression(penalty="l1", dual=False, C=c)),
                        ])

    twenty_train.data = [item.lower() for item in twenty_train.data]
    twenty_test.data = [item.lower() for item in twenty_test.data]
    clf.fit(twenty_train.data, twenty_train.target)
    predicted = clf.predict(twenty_test.data)

    # Metrics
    string = performanceMetrics(name, predicted, twenty_test)
    return string


def featureSelection2(name, twenty_train, twenty_test,c=None):
    if(c == None):
        clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1), stop_words='english',token_pattern="[\w]+")),
                    ('classification', LogisticRegression(penalty='l2', dual=False))])
    else:
        clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1),
                                                 stop_words='english',token_pattern="[\w]+")),
                        ('classification', LogisticRegression(penalty='l2', dual=False, C=c))])

    #lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(twenty_train.data, twenty_train.target)
    twenty_train.data = [item.lower() for item in twenty_train.data]
    twenty_test.data = [item.lower() for item in twenty_test.data]
    clf.fit(twenty_train.data, twenty_train.target)

    predicted = clf.predict(twenty_test.data)

    # Metrics
    string = performanceMetrics(name, predicted, twenty_test)
    return string

# Main function - Parsing the command line arguments
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

    # Design choices with Count Vectorizer
    list = []
    stemmed_twenty_test = twenty_test
    stemmed_twenty_train = twenty_train
    string = lrCountVectorizer("LR, CountVectorizer", twenty_train, twenty_test)
    list.append(string)
    string = featureSelection1("LR, CountVectorizer + Lowercase & StopWords+ L1", twenty_train, twenty_test)
    list.append(string)
    string = featureSelection2("LR, Count Vectorizer + Lowercase & Stopwords + L2", twenty_train, twenty_test)
    list.append(string)

    stemmed_twenty_train, stemmed_twenty_test = lrStemming(stemmed_twenty_train, stemmed_twenty_test)
    string = featureSelection2("LR, Count Vectorizer + Lowercase & Stopwords + Stemming + L2",
                        stemmed_twenty_train, stemmed_twenty_test)
    list.append(string)

    #Various design choices with TFIDF Vectorizer

    string = lrTfidfVectorizer("LR, TfidfVectorizer + Stemming", stemmed_twenty_train, stemmed_twenty_test)
    list.append(string)
    string = lrStopWords("LR, TfidfVectorizer + Lowercase & Stopwords + Max No. of Iterations", twenty_train, twenty_test)
    list.append(string)
    string = featureSelection2("LR, TfidfVectorizer + Lowercase & Stopwords + L2+ C(regularization Constant)",
                               twenty_train, twenty_test, 100)
    list.append(string)

    string = featureSelection1("LR, TfidfVectorizer + Stemming + L1 + C(Regularization Constant)",
                          stemmed_twenty_train, stemmed_twenty_test, 100)
    list.append(string)

    #Writing to output file
    fhandle = open(outputFile, "w+")
    for string in list:
        fhandle.write(string + "\n")
    fhandle.close()