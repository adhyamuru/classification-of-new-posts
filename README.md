# Classification-Of-News-Posts

<b> 1.	Unigram Baseline (UB) & Bigram Baseline (BB) Representation. </b> <br>
The results are as follows: <br>

|Classifier	           |Unigram/ Bigram	|Precision	|Recall 	|F1-Score|
|----------------------|----------------|-----------|---------|--------|
|Naïve Bayes	         |              UB|	     0.895|	   0.836|   0.842|
|Naïve Bayes	         |              BB|	     0.868|	   0.792|	  0.783|
|Logistic Regression	 |              UB|	     0.877|    0.879| 	0.877|
|Logistic Regression	 |              BB|	     0.811|	   0.805|	  0.799|
|SVM	                 |              UB|	     0.849|	   0.857|	  0.858|
|SVM	                 |              BB|	     0.771|	   0.750|	  0.747|
|Random Forest	       |              UB|	     0.730|	   0.732|	  0.715|
|Random Forest	       |              BB|	     0.641|	   0.641|	  0.626|

 
I implemented both Unigram and Bigram Representation for the 4 classifiers – Naïve Bayes, Logistic Regression, SVM and Random Forests. The Unigram and Bigram representation was obtained using CountVectorizer with binary=True parameter. <br>

The performance of Logistic Regression was the best. Followed by Naïve Bayes and SVM which differed minorly. In the final place, was Random Forest. <br>

Naïve Bayes is faster to build but assumes that features are conditionally independent, though that might not be true in all real-world cases. Compared to Naïve Bayes, Logistic Regression model is a bit slower to build but performs better when data sizes are huge. <br>

Initially, Naïve Bayes performs better. But as the training data size increases, the curve depicting F1 score of Logistic Regression becomes higher and gradually at the end, we can see that Logistic Regression performs better. <br>

A point to note is that Logistic Regression gives the probabilities scores of the observation and not classes as the output. It is robust to mild multi-collinearity and in severe cases of multi-collinearity, we can use it with L2 Regularization. <br>

SVM makes the classes linearly separable by adding an extra dimension by projecting the feature space into kernel space. It can handle a huge feature space, but when the number of observations become too large, they are not so efficient and selecting the right kernel is sometimes a challenge. With non-linear kernels, it is too costly to train SVMs on data. <br>

Random Forest does not overfit data. They are faster to build. But they are black boxes which are hard to interpret. They take up lot of memory as well and are slow to evaluate. <br> 

<b> 2.	My Best Configuration </b> <br>
I have chosen Logistic Regression due to its high F1-Score and the above explained reasons. I explored the various Feature Representations, Feature Selection, Hyperparameters for it in MBC_exploration.py
The design choices that I chose are: <br>
1.	CountVectorizer <br>
2.	TFIDFVectorizer <br>
3.	Lower case and filtering stop words <br>
4.	Stemming using Porter Stemmer <br>
5.	L1 Regularizer <br>
6.	L2 Regularizer <br>
7.	Regularization Constant <br>
8.	Max number of iterations <br>

For the feature extractors, I have chosen Count Vectorizer and TFIDFVectorizer. Count Vectorizer uses the count of the words observed. Since longer documents have higher average count values, it would be beneficial to include the term frequencies normalized by the inverse of the number of documents in which the word is observed. This is done by TFIDFVectorizer. <br>
In terms of preprocessing, I have removed Stop words and performed stemming. Stop words like ‘a’, ‘is’, ‘the’, etc. do not give any additional information about the data. Hence, I have removed stop words. Stemming reduces the inflected words to their base forms. It groups the frequencies of the various inflected forms of a word to one form which is its base term. <br> 
For feature selection, I have used L1 and L2 Regularizers which prevent the over-fitting of data. Regularization improves the performance on new unseen data. If data is severely multicollinear, then using with L2 regularization make the model perform better. <br>
For hyper parameters, I have used max_iter and Regularization Constant C. <br>

After trying several possible combinations of feature extractors, regularizers, parameters, the highest F1 score was obtained when combining TFIDF Vectorizer with L2 Regularizer with regularization constant C and using lower case and filtering stop words. 
I have exported the same configuration for my MBC_final.py. <br>

REFERENCES
1.	http://scikit-learn.org/stable/tutorial/basic/tutorial.html 
2.	https://medium.com/@sangha_deb/naive-bayes-vs-logistic-regression-a319b07a5d4c
3.	https://www.edvancer.in/logistic-regression-vs-decision-trees-vs-svm-part1/
4.	https://www.edvancer.in/logistic-regression-vs-decision-trees-vs-svm-part2/
5.	https://medium.com/rants-on-machine-learning/the-unreasonable-effectiveness-of-random-forests-f33c3ce28883
6.	https://pythonprogramming.net/stemming-nltk-tutorial/
7.	http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
8.	http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
9.	https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
10.	https://www.codeschool.com/blog/2016/03/25/machine-learning-working-with-stop-words-stemming-and-spam/

