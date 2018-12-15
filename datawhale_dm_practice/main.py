# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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
    clf = model.fit(x_train_standard, y_train.ravel())
    best_params = clf.best_params_
    best_score = clf.best_score_

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
    print('''
        best params: {best_params},
        best_score: {best_score},
    '''.format(**locals()))
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
    x = data_all.drop(columns=['status']).values
    y = data_all[['status']].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2018)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_standard = scaler.transform(x_train)
    x_test_standard = scaler.transform(x_test)
    # x_train_standard = x_train
    # x_test_standard = x_test

    lr = LogisticRegression(random_state=2018)
    lr_grid_search_param = {
        'C': [0.5 * _ for _ in range(1, 10, 1)]
    }
    grid_search = GridSearchCV(lr, param_grid=lr_grid_search_param, cv=5, n_jobs=1, scoring='roc_auc')
    fit_and_evaluate_model(grid_search, x_train_standard, y_train, x_test_standard, y_test, 'LogisticRegression')

    # clf = SVC(kernel='linear', C=0.4)
    # clf = SVC(kernel='linear')
    svm_clf = SVC(random_state=2018, probability=True)
    svm_grid_search_param = {
        'C': [0.5 * _ for _ in range(1, 10, 1)]
    }

    grid_search = GridSearchCV(svm_clf, param_grid=svm_grid_search_param, cv=5, n_jobs=1, scoring='roc_auc')
    fit_and_evaluate_model(grid_search, x_train_standard, y_train, x_test_standard, y_test, 'SVM')
    dt_clf = tree.DecisionTreeClassifier(random_state=2018)
    dt_grid_search_param = [
        {
            'max_depth': range(1, 20, 1),
            'max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }
    ]
    grid_search = GridSearchCV(dt_clf, param_grid=dt_grid_search_param, cv=5, n_jobs=1, scoring='roc_auc')
    fit_and_evaluate_model(grid_search, x_train_standard, y_train, x_test_standard, y_test, 'DecisionTree')

    random_forest_clf = RandomForestClassifier(random_state=2018)
    random_forest_grid_search_param = {
        'n_estimators': range(250, 300, 5)
    }
    grid_search = GridSearchCV(random_forest_clf, param_grid=random_forest_grid_search_param, cv=5, n_jobs=1, scoring='roc_auc')
    fit_and_evaluate_model(grid_search, x_train_standard, y_train, x_test_standard, y_test, 'RandomForest')

    gbdt_clf = GradientBoostingClassifier(random_state=2018)
    gbdt_grid_search_param = {
        'n_estimators': range(250, 300, 5),
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    }
    grid_search = GridSearchCV(gbdt_clf, param_grid=gbdt_grid_search_param, cv=5, n_jobs=1, scoring='roc_auc')
    fit_and_evaluate_model(grid_search, x_train_standard, y_train, x_test_standard, y_test, 'GBDT')

    xgbt_clf = XGBClassifier(random_state=2018)
    xgbt_grid_search_param = {
        'max_depth': range(1, 5, 1),
        'min_child_weight': range(1, 6, 1)
    }
    grid_search = GridSearchCV(xgbt_clf, param_grid=xgbt_grid_search_param, cv=5, n_jobs=1, scoring='roc_auc')
    fit_and_evaluate_model(grid_search, x_train_standard, y_train, x_test_standard, y_test, 'XGBoost')

    light_gbm_clf = LGBMClassifier(random_state=2018)
    light_gbm_grid_search_param = {
        'max_depth': range(3, 8, 1),
        'num_leaves': range(20, 200, 5)
    }
    grid_search = GridSearchCV(light_gbm_clf, param_grid=light_gbm_grid_search_param, cv=5, n_jobs=1, scoring='roc_auc')
    fit_and_evaluate_model(grid_search, x_train_standard, y_train, x_test_standard, y_test, 'LightGBM')


if __name__ == '__main__':
    main()
