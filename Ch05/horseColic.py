#!/usr/bin/env python
# encoding=utf-8

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from logRegres import stocGradientAscent, sigmoid


""" 利用logistic回归来进行分类 -- 从疝气病症状预测病马的死亡率 """


def predict(inX, weights):
    probability = sigmoid(sum(np.array(inX)*weights))
    if probability > 0.5:
        return 1
    else:
        return 0


def loadDatasetFromFile(filename):
    dataset = []
    labels = []
    with open(filename) as infile:
        for line in infile:
            datas = line.strip().split('\t')
            row = list(map(lambda x: float(x), datas[:21]))
            dataset.append(row)
            labels.append(float(datas[21]))
    return np.array(dataset), np.array(labels)


def testColicPredict(num_iter=1000):
    """在马的疝气病数据上训练 logistic 回归模型

    Parameters
    ----------
    num_iter

    Returns
    -------

    """
    # 训练模型
    train = {}
    train['dataset'], train['labels'] = loadDatasetFromFile(
        'horseColicTraining.txt'
    )
    train['weights'] = stocGradientAscent(
        train['dataset'],
        train['labels'],
        epoches=num_iter
    )
    # 测试
    errorCount = 0
    test = {}
    test['dataset'], test['labels'] = loadDatasetFromFile(
        'horseColicTest.txt'
    )
    m, _n = test['dataset'].shape
    for rowno, row in enumerate(test['dataset']):
        if predict(row, train['weights']) != test['labels'][rowno]:
            errorCount += 1
    errorRate = 1.0*errorCount / m
    print("Error rate: {:.4f}".format(errorRate))
    return errorRate


def multiTestColicPredict(numTests=10):
    errorSum = 0.0
    # 多次运行结果可能不同, 因为使用随机选取的向量来更新回归系数
    for k in range(numTests):
        errorSum += testColicPredict()
    print('after %d iterations the average error rate is: %f'
          % (numTests, errorSum/float(numTests))
    )

# dataset, labels = getDataset()
# weights = {
#     0: getGradientAsecent(dataset, labels),
#     1: getStochasticGradientAsecent_0(dataset, labels),
#     2: stocGradientAscent(dataset, labels),
# }
# plotBestFit(dataset, labels, weights[0])
# plotBestFit(dataset, labels, weights[1])
# plotBestFit(dataset, labels, weights[2])

# multiTestColicPredict(10)

# import horseColic
# horseColic.testColicPredict()
