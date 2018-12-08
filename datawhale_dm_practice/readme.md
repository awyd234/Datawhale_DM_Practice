背景介绍

由Datawhale组织的「一周算法实践」活动，通过短期实践一个比较完整的数据挖掘项目，迅速了解相关实际过程



任务描述

【任务1.1 - 模型构建】给定数据集，数据三七分，随机种子2018。（在任务1中什么都不用考虑，即不需数据处理和模型调参）调用sklearn的包，简单构建逻辑回归、SVM和决策树3个模型，评分方式任意（e.g. 准确度和auc值）。



1. 环境搭建

1.1 Virtualenv安装Python3.6虚拟环境

    virtualenv --python=python3.6 --prompt='(datawhale_dm_practice)' .env

1.2 安装sklearn

    pip install -i install -i https://pypi.doubanio.com/simple/ sklearn

由于python官方源访问不太稳定，此处选用豆瓣Python源



2. 数据划分

根据要求，三七分数据，7成数据作为训练集，三成数据作为测试集，并选取随机种子2018

    from sklearn.model_selection import train_test_split
    
    data_all = pd.read_csv('data_all.csv', encoding='gbk')
    features = [x for x in data_all.columns if x not in ['status']]
    x = data_all[features]
    y = data_all['status']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2018)

2.1 X和y值划分

此题将status列作为y值，其它列作为X值

2.2 train_test_split函数

2.2.1 源码函数说明

    Split arrays or matrices into random train and test subsets
    
    Quick utility that wraps input validation and
    ``next(ShuffleSplit().split(X, y))`` and application to input data
    into a single call for splitting (and optionally subsampling) data in a
    oneliner.
    
    Read more in the :ref:`User Guide <cross_validation>`.
    
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    
    test_size : float, int or None, optional (default=0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default, the value is set to 0.25.
        The default will change in version 0.21. It will remain 0.25 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.
    
    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    
    shuffle : boolean, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    
    stratify : array-like or None (default=None)
        If not None, data is split in a stratified fashion, using this as
        the class labels.
    
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    
        .. versionadded:: 0.16
            If the input is sparse, the output will be a
            ``scipy.sparse.csr_matrix``. Else, output type is the same as the
            input type.

2.2.2 参数说明

- test_size，测试集大小，可以int，float或者None，此处指定为0.3
  - 值为int时，取样本数目
  - 值为float时，取样本占比
  - 如果没有指定，或者说为None，如果没有指定train_size参数，则取默认值0.25，否则取除了train_size余下的数据
- train_size，训练集大小，同样可以int，float或者None，默认值None，与test_size区别处，如果train_size不设置，则直接取除了test_size余下的数据
- random_state，随机数种子，也就是该组随机数的编号
  - 随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则
    - 种子不同，产生不同的随机数，如果设置为None，每次都会产生不一样的随机数
    - 种子相同，即使实例不同也产生相同的随机数，对于重复数据很有帮助
    - 具体原理暂时不具体研究



3. 逻辑回归LogisticRegression

    from sklearn.linear_model import LogisticRegression
    
    lr = LogisticRegression(random_state=2018)
    lr.fit(x_train, y_train)
    print('LR fit finished, score: {}'.format(lr.score(x_test, y_test)))

输出

    LR fit finished, score: 0.7484232655921513

3.1 初始化随机树

此处仍然选择随机种子2018，其它参数暂不详述

3.2 训练模型

LogisticRegression.fit() 函数源码说明如下

    Fit the model according to the given training data.
    
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
    	Training vector, where n_samples is the number of samples and n_features is the number of features.
    
    y : array-like, shape (n_samples,)
    	Target vector relative to X.
    
    sample_weight : array-like, shape (n_samples,) optional
    	Array of weights that are assigned to individual samples. If not provided, then each sample is given unit weight.
    
    .. versionadded:: 0.17
    *sample_weight* support to LogisticRegression.
    
    Returns
    -------
    self : object

- X, y以矩阵形式传入
- sample_weight是每条测试数据的权重，同样以array形式传入，作为可选参数，如果不指定，则默认权重相同，都为1个单元

3.3 模型评价

3.3.1 准确率计算

此处使用ClassifierMixin.score()函数，源码中说明如下

    Returns the mean accuracy on the given test data and labels.
    
    In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.
            
    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features)
    	Test samples.
    
    y : array-like, shape = (n_samples) or (n_samples, n_outputs)
        True labels for X.
    
    sample_weight : array-like, shape = [n_samples], optional
    	Sample weights.
    
    Returns
    -------
    score : float
    	Mean accuracy of self.predict(X) wrt. y.

给定测试数据以及标签，计算平均准确率

3.3.2 AUC计算

    from sklearn.metrics import auc, roc_curve
    from sklearn import tree
    
    def cal_auc(y_test, predictions):
        false_positive_rate, recall, thresholds = roc_curve(y_test, predictions[:, 1])
        roc_auc = auc(false_positive_rate, recall)
        return roc_auc
    
    predictions = lr.predict_proba(x_test)
    print('LR fit finished, auc: {}'.format(cal_auc(y_test, predictions)))

此模型输出结果

    LR fit finished, auc: 0.5674574609036754

predict_proba()函数根据模型生成各个测试数据在各class的概率，源码中说明如下

    Compute probabilities of possible outcomes for samples in X.
    
    The model need to have probability information computed at training
    time: fit with attribute `probability` set to True.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    	For kernel="precomputed", the expected shape of X is [n_samples_test, n_samples_train]
    
    Returns
    -------
    T : array-like, shape (n_samples, n_classes)
    	Returns the probability of the sample for each class in the model. The columns correspond to the classes in sorted order, as they appear in the attribute `classes_`.
    
    Notes
    -----
    The probability model is created using cross validation, so the results can be slightly different than those obtained by predict. Also, it will produce meaningless results on very small datasets.





4. 支持向量机SVM

    from sklearn.svm import SVC
    
    clf = SVC(random_state=2018, probability=True)
    clf.fit(x_train, y_train)
    print('SVM fit finished, score: {}'.format(lr.score(x_test, y_test)))
    predictions = clf.predict_proba(x_test)
    print('SVM fit finished, auc: {}'.format(cal_auc(y_test, predictions)))

输出

    SVM fit finished, score: 0.7484232655921513
    SVM fit finished, auc: 0.5



5. 决策树DecisionTreeClassifier

    from sklearn import tree
    
    clf = tree.DecisionTreeClassifier(random_state=2018)
    clf.fit(x_train, y_train)
    print('DecisionTree fit finished, score: {}'.format(lr.score(x_test, y_test)))
    predictions = clf.predict_proba(x_test)
    print('DecisionTree fit finished, auc: {}'.format(cal_auc(y_test, predictions)))

输出

    DecisionTree fit finished, score: 0.7484232655921513
    DecisionTree fit finished, auc: 0.5942367479369453


