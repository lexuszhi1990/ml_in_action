#!/usr/bin/env python
# encoding=utf-8

"""
Logistic回归:
根据现有数据对分类边界线建立回归公式, 以此进行分类.
"回归"来源于最佳拟合, 训练分类器即使用最优化算法寻找最佳拟合参数.
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

# http://blog.csdn.net/whai362/article/details/51860379

def loadDataset(filename='testSet.txt'):
    dataset = []
    labels = []
    with open(filename) as infile:
        for line in infile:
            datas = line.strip().split()
            dataset.append([
                1.0,
                float(datas[0]),
                float(datas[1]),
            ])
            labels.append(int(datas[-1]))
        return numpy.array(dataset), labels

def sigmoid(inX):
    """ 海维赛德阶跃函数 """
    return 1.0 / (1 + numpy.exp(-inX))

def plotBestFit(dataset, labels, weights):
    """绘制数据分界线

    Parameters
    ----------
    weights : list of floats
        系数

    """
    m, _n = dataset.shape
    # 收集绘制的数据
    cord = {
        '1': {
            'x': [],
            'y': [],
        },
        '2': {
            'x': [],
            'y': [],
        },
    }
    for i in range(m):
        if labels[i] == 1:
            cord['1']['x'].append(dataset[i, 1])
            cord['1']['y'].append(dataset[i, 2])
        else:
            cord['2']['x'].append(dataset[i, 1])
            cord['2']['y'].append(dataset[i, 2])
    # 绘制图形
    figure = plt.figure()
    subplot = figure.add_subplot(111)
    # 绘制散点
    subplot.scatter(
        cord['1']['x'], cord['1']['y'],
        s=30, c='red', marker='s'
    )
    subplot.scatter(
        cord['2']['x'], cord['2']['y'],
        s=30, c='green'
    )
    # 绘制直线
    x = numpy.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x)/ weights[2]
    subplot.plot(x, y)
    # 标签
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def gradientAscent(dataset, labels, epoches=1000, alpha=0.01):
    xMat = np.mat(dataset); yMat = np.mat(labels).T
    m, n = np.shape(xMat)
    weights = np.ones((n, 1))
    for i in xrange(epoches):
        h = sigmoid(xMat*weights)
        weights += alpha * xMat.T * (yMat - h)
    return weights

def stocGradientAscent(dataset, labels, epoches=1000, lr=0.01):
    xArr = np.array(dataset); yArr = np.array(labels)
    m, n = np.shape(xArr)
    # weights = np.ones((1, n))
    weights = np.ones(n)
    for j in xrange(epoches):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4.0/(i+j+1) + lr
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(np.sum(xArr[randIndex] * weights))
            weights += alpha * (yArr[randIndex] - h) * xArr[randIndex]
            del(dataIndex[randIndex])

    return weights

# import logRegres
# reload(logRegres)
# dataset, labels = logRegres.loadDataset()
# weights = logRegres.gradientAscent(dataset, labels)
# logRegres.plotBestFit(dataset, labels, weights)
# weights = logRegres.stocGradientAscent(dataset, labels)
# logRegres.plotBestFit(dataset, labels, weights)
