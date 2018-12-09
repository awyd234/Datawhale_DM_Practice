# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, auc, roc_curve
from sklearn import tree


def cal_auc(y_test, predictions):
    false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:, 1])
    roc_auc = auc(false_positive_rate, recall)
    return roc_auc


def main():
    data_all = pd.read_csv('data_all.csv', encoding='gbk')
    features = [x for x in data_all.columns if x not in ['status']]
    x = data_all[features]
    y = data_all['status']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2018)

    lr = LogisticRegression(random_state=2018)
    lr.fit(x_train, y_train)
    print('LR fit finished, score: {}'.format(lr.score(x_test, y_test)))
    predictions = lr.predict_proba(x_test)
    print('LR fit finished, auc: {}'.format(cal_auc(y_test, predictions)))

    # clf = SVC(kernel='linear', C=0.4)
    # clf = SVC(kernel='linear')
    clf = SVC(random_state=2018, probability=True)
    clf.fit(x_train, y_train)
    print('SVM fit finished, score: {}'.format(clf.score(x_test, y_test)))
    predictions = clf.predict_proba(x_test)
    print('SVM fit finished, auc: {}'.format(cal_auc(y_test, predictions)))

    clf = tree.DecisionTreeClassifier(random_state=2018)
    clf.fit(x_train, y_train)
    print('DecisionTree fit finished, score: {}'.format(clf.score(x_test, y_test)))
    predictions = clf.predict_proba(x_test)
    print('DecisionTree fit finished, auc: {}'.format(cal_auc(y_test, predictions)))


if __name__ == '__main__':
    main()
