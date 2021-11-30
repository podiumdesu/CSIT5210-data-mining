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
train = pd.read_csv('./train_1.csv',encoding='unicode_escape')
train.columns = ['A','B','C','D','E','F']
test = pd.read_csv('./test_1.csv',encoding='unicode_escape')
test.columns = ['A','B','C','D','E','F']
# print(datas['F'][1])

train_model = TfidfVectorizer().fit(train['F'])
train_tfidf = train_model.transform(train['F'])     # 得到tf-idf矩阵，稀疏矩阵表示法
train_tfidf = train_tfidf.toarray()
train_label = train['A']
train_label = train_label != 0

print(train_label)
input()

test_model = TfidfVectorizer().fit(test['F'])
test_tfidf = train_model.transform(test['F'])     # 得到tf-idf矩阵，稀疏矩阵表示法
test_tfidf = test_tfidf.toarray()
test_label = test['A']
test_label = test_label != 0

print("training set dimensions: {}".format(train_tfidf.shape))
print("testing set dimensions: {}".format(test_tfidf.shape))



if __name__ == "__main__":
    # contamination = 0.1  # percentage of outliers
    # n_train = 5000  # number of training points
    # n_test = 100  # number of testing points

    # # Generate sample data
    # X_train, y_train, X_test, y_test = \
    #     generate_data(n_train=n_train,
    #                   n_test=n_test,
    #                   n_features=2,
    #                   contamination=contamination,
    #                   random_state=42)

    # train COPOD detector
    clf_name = 'COPOD'
    
    clf = COPOD()

    print("before fit")

    # you could try parallel version as well.
    # clf = COPOD(n_jobs=2)
    clf.fit(train_tfidf)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test)

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
    # print(train_scores)
    print("\nOn Test Data:")
    evaluate_print(clf_name, test_label, test_scores)
    # print(test_scores)

    # visualize the results
    # visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred,
    #           y_test_pred, show_figure=True, save_figure=False)