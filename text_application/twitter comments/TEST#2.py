from __future__ import division
from __future__ import print_function
import numpy as np

from pyod.models.copod import COPOD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

#提取dataset中的评论并且
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
train = pd.read_csv('./train_2.csv',encoding='unicode_escape')
train.columns = ['A','B']
test = pd.read_csv('./test_2.csv',encoding='unicode_escape')
test.columns = ['A','B']


train_model = TfidfVectorizer().fit(train['A'])
train_tfidf = train_model.transform(train['A'])     # 得到tf-idf矩阵，稀疏矩阵表示法
train_tfidf = train_tfidf.toarray()
train_label = train['B']
train_label = train_label == 1
print(train_label)
test_model = TfidfVectorizer().fit(test['A'])
test_tfidf = train_model.transform(test['A'])     # 得到tf-idf矩阵，稀疏矩阵表示法
test_tfidf = test_tfidf.toarray()
test_label = test['B']
test_label = test_label == 1
print(test_label)
print(train_tfidf.shape, test_tfidf.shape)


if __name__ == "__main__":
    # train COPOD detector
    clf_name = 'COPOD'
    clf = COPOD() 

    print("before fit")
    clf.fit(train_tfidf)
    print("after fit")

    np.savetxt("out.csv", np.c_[train_label, clf.labels_, clf.decision_scores_], delimiter=",", fmt='%1.3f')


    # get the prediction labels and outlier scores of the training data
    train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
    train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
    test_pred = clf.predict(test_tfidf)  # outlier labels (0 or 1)
    test_scores = clf.decision_function(test_tfidf)  # outlier scores

    # evaluate and print the results
    print("\nOn Training Data:")
    evaluate_print(clf_name, train_label, train_scores)
    print(train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, test_label, test_scores)
    print(test_scores)

  