#!/usr/bin/env python
# encoding=utf-8

"""
SVM - 支持向量机
===
介绍的是SVM的其中一种实现 -- 序列最小化(SMO, Sequential Minimal Optimization)算法
`分隔超平面` -- 将数据集分隔开来的超平面, 也就是分类的决策边界.
`间隔` -- 找到离分隔超平面最近的点, 确保他们离分隔面的距离尽可能远, 这其中点到分隔面的距离就是间隔.
    我们希望间隔尽可能地大, 以保证分类器尽可能健壮
`支持向量` -- 离分隔超平面最近的那些点
"""

from __future__ import print_function

import logging

import numpy as np

logging.basicConfig(
    # level=logging.DEBUG,
    level=logging.INFO,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)


def load_dataset(filename='./testSet.txt'):
    dataset = []
    labels = []
    with open(filename) as infile:
        for line in infile:
            datas = line.strip().split('\t')
            dataset.append([float(datas[0]), float(datas[1])])
            # dataset.append(np.array(datas[0:-1], dtype=np.float))
            labels.append(float(datas[-1]))
    return dataset, labels


def selectJrand(i,m):
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def random_select_j(i, m):
    """ 返回任一 [0, m) 之间且不等于 i 的数 """
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def adjust_alpha(aj, upper_bound, lower_bound):
    if aj > upper_bound:
        aj = upper_bound
    if lower_bound > aj:
        aj = lower_bound
    return aj

def estimate(alphas, labels, dataset, index, b):
    fx = float(
        np.multiply(alphas, labels).T
        * (dataset*dataset[index, :].T)
    ) + b
    e = fx - float(labels[index])
    return e

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """
    Platt的SMO算法简化版.
    = = = =
    每次循环中选择两个alpha进行优化处理.一旦找到一堆合适的alpha,
    那么就增大其中一个同时减少另外一个.
    * 两个alpha必须在间隔边界之外
    * 两个alpha还没有进行过区间化处理或者不在边界上

    Parameters
    ----------
    dataset
        数据集
    labels
        类型标签
    constant
        常数, 用于控制"最大化间隔"和"保证大部分点的函数间隔小于1.0"
    toler
        容错率
    max_iter
        最大循环次数

    Returns
    -------
    """
    dataMatrix = np.mat(dataMatIn); labelMat = np.mat(classLabels).transpose()
    b = 0; m,n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta>=0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                #update i by the same amount as j
                #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print("iteration number: %d" % iter)
    return b,alphas

def smo_simple(dataset, labels, constant, toler, max_iter):
    """
    Platt的SMO算法简化版.
    = = = =
    每次循环中选择两个alpha进行优化处理.一旦找到一堆合适的alpha,
    那么就增大其中一个同时减少另外一个.
    * 两个alpha必须在间隔边界之外
    * 两个alpha还没有进行过区间化处理或者不在边界上

    Parameters
    ----------
    dataset
        数据集
    labels
        类型标签
    constant
        常数, 用于控制"最大化间隔"和"保证大部分点的函数间隔小于1.0"
    toler
        容错率
    max_iter
        最大循环次数

    Returns
    -------

    """
    dataset = np.mat(dataset)
    labels = np.mat(labels).T
    b = 0
    m, n = dataset.shape
    # 初始化alpha向量
    alphas = np.mat(np.zeros((m, 1)))
    num_iter = 0
    while num_iter < max_iter:
        # 对数据集中每个数据向量
        num_alpha_pairs_changed = False  # alpha 是否已经优化
        for i in range(m):
            # 计算 alpha[i] 的预测值, 估算其是否可以被优化
            Ei = estimate(alphas, labels, dataset, i, b)
            # 测试正/负间隔距离, alpha值, 是否满足KKT条件
            if not ((labels[i] * Ei < -toler and alphas[i] < constant)
                    or (labels[i] * Ei > toler and alphas[i] > 0)):
                logging.debug('alpha[{0}]不需要调整.'.format(i))
                continue

            # 选择第二个 alpha[j]
            j = random_select_j(i, m)
            # alpha[j] 的预测值
            Ej = estimate(alphas, labels, dataset, j, b)

            # 保存旧值以便与调整后比较
            alphaI_old = alphas[i].copy()
            alphaJ_old = alphas[j].copy()

            # 计算 lower_bound/upper_bound, 调整 alpha[j] 至 (0, C) 之间
            if labels[i] != labels[j]:
                lower_bound = max(0, alphas[j] - alphas[i])
                upper_bound = min(constant, constant + alphas[j] - alphas[i])
            else:
                lower_bound = max(0, alphas[j] + alphas[i] - constant)
                upper_bound = min(constant, alphas[j] + alphas[i])
            if lower_bound == upper_bound:
                logging.debug('lower_bound == upper_bound == {0}'.format(lower_bound))
                continue

            # 计算 alpha[j] 的最优修改量
            delta = (
                2.0 * dataset[i, :] * dataset[j, :].T
                - dataset[i, :] * dataset[i, :].T
                - dataset[j, :] * dataset[j, :].T
            )
            # 如果 delta==0, 则需要退出for循环的当前迭代过程.
            # 简化版中不处理这种少量出现的特殊情况
            if delta >= 0:
                logging.warning('{0}(delta) >= 0'.format(delta))
                continue

            # 计算新的 alpha[j]
            alphas[j] -= labels[j] * (Ei - Ej) / delta
            alphas[j] = adjust_alpha(alphas[j], upper_bound, lower_bound)
            # 若 alpha[j] 的改变量太少, 不采用
            delta_j = abs(alphas[j] - alphaJ_old)
            if delta_j < 0.00001:
                logging.debug('j 变化量太少, 不采用. ({0})'.format(delta_j))
                continue

            # 对 alpha[i] 做 alpha[j] 同样大小, 方向相反的改变
            alphas[i] += labels[j] * labels[i] * (alphaJ_old - alphas[j])

            # 给两个 alpha 值设置常量 b
            b1 = (
                b - Ei
                - labels[i] * (alphas[i] - alphaI_old) * dataset[i, :] * dataset[i, :].T
                - labels[j] * (alphas[j] - alphaJ_old) * dataset[i, :] * dataset[j, :].T
            )
            b2 = (
                b - Ej
                - labels[i] * (alphas[i] - alphaI_old) * dataset[i, :] * dataset[j, :].T
                - labels[j] * (alphas[j] - alphaJ_old) * dataset[j, :] * dataset[j, :].T
            )
            if 0 < alphas[i] < constant:
                b = b1
            elif 0 < alphas[j] < constant:
                b = b2
            else:
                b = (b1 + b2) / 2.0

            num_alpha_pairs_changed = True
            logging.debug('numIter: {:d} i:{:d}, pairs changed {}'.format(
                num_iter, i, num_alpha_pairs_changed
            ))
        if num_alpha_pairs_changed == 0:
            num_iter += 1
        else:
            num_iter = 0
        logging.debug('iteration number: {0}'.format(num_iter))
    return b, alphas

def aaaa(s):
    print(s)

# import svm
# reload(svm)
# dataArr, labelArr = svm.load_dataset()
# b, alphas = svm.smoSimple(labelArr, dataArr, 0.6, 0.001, 40)
# b, alphas = svm.smo_simple(dataArr, labelArr, 0.6, 0.001, 40)

# Ei = svm.estimate(alphas, labelArr, dataArr, i, b)
