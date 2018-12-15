# datawhale_dm_practice

## 背景介绍

由Datawhale组织的「一周算法实践」活动，通过短期实践一个比较完整的数据挖掘项目，迅速了解相关实际过程


## 任务描述

### Task 1.1 模型构建
#### 任务描述
给定数据集，数据三七分，随机种子2018。（在任务1中什么都不用考虑，即不需数据处理和模型调参）调用sklearn的包，简单构建逻辑回归、SVM和决策树3个模型，评分方式任意（e.g. 准确度和auc值）。
#### 说明链接
[「一周算法实践」Task1.1 模型构建 - WindTrack](http://windtrack.xyz/2018/12/09/%E3%80%8C%E4%B8%80%E5%91%A8%E7%AE%97%E6%B3%95%E5%AE%9E%E8%B7%B5%E3%80%8DTask1-1-%E6%A8%A1%E5%9E%8B%E6%9E%84%E5%BB%BA)

### Task 1.2 模型构建之集成模型
#### 任务描述
构建随机森林、GBDT、XGBoost和LightGBM这4个模型，评分方式任意。
#### 说明链接
[「一周算法实践」Task1.2 模型构建之集成模型 - WindTrack](http://windtrack.xyz/2018/12/10/%E3%80%8C%E4%B8%80%E5%91%A8%E7%AE%97%E6%B3%95%E5%AE%9E%E8%B7%B5%E3%80%8DTask1-2-%E6%A8%A1%E5%9E%8B%E6%9E%84%E5%BB%BA%E4%B9%8B%E9%9B%86%E6%88%90%E6%A8%A1%E5%9E%8B)

### Task 2 模型评估

#### 任务描述

记录7个模型（在Task1的基础上）关于accuracy、precision，recall和F1-score、auc值的评分表格，并画出Roc曲线。

#### 说明链接

[「一周算法实践」Task2 模型评估 - WindTrack](http://windtrack.xyz/2018/12/13/%E3%80%8C%E4%B8%80%E5%91%A8%E7%AE%97%E6%B3%95%E5%AE%9E%E8%B7%B5%E3%80%8DTask2-%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0/)

### Task 3 模型调优

#### 任务描述

使用网格搜索法对7个模型进行调优（调参时采用五折交叉验证的方式），并进行模型评估，记得展示代码的运行结果

#### 说明链接

[「一周算法实践」Task3 模型调优 - WindTrack](http://windtrack.xyz/2018/12/15/%E3%80%8C%E4%B8%80%E5%91%A8%E7%AE%97%E6%B3%95%E5%AE%9E%E8%B7%B5%E3%80%8DTask3-%E6%A8%A1%E5%9E%8B%E8%B0%83%E4%BC%98/)