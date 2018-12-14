# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt


def fit_and_evaluate_model(model, x_train_standard, y_train, x_test_standard, y_test, model_name):
    '''
    train model and evaluate it
    :param model: 
    :param x_train_standard: 
    :param y_train: 
    :param x_test_standard: 
    :param y_test: 
    :param model_name: 
    :return: 
    '''
    clf = model.fit(x_train_standard, y_train)

    clf.fit(x_train_standard, y_train)

    # 模型对训练集的预测值
    y_train_prediction_value = clf.predict(x_train_standard)

    # 模型对测试集的预测值
    y_test_prediction_value = clf.predict(x_test_standard)

    # 模型对训练集的各类预测概率
    y_train_prediction_prob = clf.predict_proba(x_train_standard)
    # 模型对测试集的各类预测概率
    y_test_prediction_prob = clf.predict_proba(x_test_standard)

    train_accuracy_score = metrics.accuracy_score(y_train, y_train_prediction_value)
    test_accuracy_score = metrics.accuracy_score(y_test, y_test_prediction_value)

    train_precision_score = metrics.precision_score(y_train, y_train_prediction_value)
    test_precision_score = metrics.precision_score(y_test, y_test_prediction_value)

    train_recall = metrics.recall_score(y_train, y_train_prediction_value)
    test_recall = metrics.recall_score(y_test, y_test_prediction_value)

    train_f1_score = metrics.f1_score(y_train, y_train_prediction_value)
    test_f1_score = metrics.f1_score(y_test, y_test_prediction_value)

    train_auc_score = metrics.roc_auc_score(y_train, y_train_prediction_prob[:, 1])
    test_auc_score = metrics.roc_auc_score(y_test, y_test_prediction_prob[:, 1])

    print("------------------------{model_name}------------------------".format(**locals()))
    print("Train Set:")
    print("Accuracy: {train_accuracy_score}".format(**locals()))
    print("Precision: {train_precision_score}".format(**locals()))
    print("F1_score: {train_f1_score}".format(**locals()))
    print("Recall: {train_recall}".format(**locals()))
    print("Auc: {train_auc_score}".format(**locals()))

    print("Test Set:")
    print("Accuracy: {test_accuracy_score}".format(**locals()))
    print("Precision: {test_precision_score}".format(**locals()))
    print("F1_score: {test_f1_score}".format(**locals()))
    print("Recall: {test_recall}".format(**locals()))
    print("Auc: {test_auc_score}".format(**locals()))

    # ROC曲线绘制
    train_fprs, train_tprs, train_thresholds = metrics.roc_curve(y_train, y_train_prediction_prob[:, 1])
    test_fprs, test_tprs, test_thresholds = metrics.roc_curve(y_test, y_test_prediction_prob[:, 1])
    plt.plot(train_fprs, train_tprs)
    plt.plot(test_fprs, test_tprs)
    plt.plot([0, 1], [0, 1], "--")
    plt.title("{model_name} ROC curve".format(**locals()))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(
        labels=["Train Set AUC:" + str(round(train_accuracy_score, 5)), "Test Set AUC:" + str(round(test_accuracy_score, 5))],
        loc="lower right")
    plt.show()

    print("-----------------------------------------------------------")


def main():
    data_all = pd.read_csv('data_all.csv', encoding='gbk')
    features = [x for x in data_all.columns if x not in ['status']]
    x = data_all[features]
    y = data_all['status']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2018)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_standard = scaler.transform(x_train)
    x_test_standard = scaler.transform(x_test)
    # x_train_standard = x_train
    # x_test_standard = x_test

    lr = LogisticRegression(random_state=2018)
    fit_and_evaluate_model(lr, x_train_standard, y_train, x_test_standard, y_test, 'LogisticRegression')

    # clf = SVC(kernel='linear', C=0.4)
    # clf = SVC(kernel='linear')
    svm_clf = SVC(random_state=2018, probability=True)
    fit_and_evaluate_model(svm_clf, x_train_standard, y_train, x_test_standard, y_test, 'SVM')

    dt_clf = tree.DecisionTreeClassifier(random_state=2018)
    fit_and_evaluate_model(dt_clf, x_train_standard, y_train, x_test_standard, y_test, 'DecisionTree')

    random_forest_clf = RandomForestClassifier(random_state=2018)
    fit_and_evaluate_model(random_forest_clf, x_train_standard, y_train, x_test_standard, y_test, 'RandomForest')

    gbdt_clf = GradientBoostingClassifier(random_state=2018)
    fit_and_evaluate_model(gbdt_clf, x_train_standard, y_train, x_test_standard, y_test, 'GBDT')

    xgbt_clf = XGBClassifier(random_state=2018)
    fit_and_evaluate_model(xgbt_clf, x_train_standard, y_train, x_test_standard, y_test, 'XGBoost')

    light_gbm_clf = LGBMClassifier(random_state=2018)
    fit_and_evaluate_model(light_gbm_clf, x_train_standard, y_train, x_test_standard, y_test, 'LightGBM')


if __name__ == '__main__':
    main()
