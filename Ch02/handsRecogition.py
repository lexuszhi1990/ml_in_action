#!/usr/bin/env python
# encoding=utf-8

"""
k-近邻算法
===
存在训练样本集, 且样本集中每个数据都存在标签, 即已知样本集中每一组数据与所属分类的对应关系.

当输入没有标签的新数据后, 将新数据的每个特征与样本集中数据对应的特征进行比较,
算法提取样本集中特征最相似的k组数据(最近邻)的分类标签, (一般k<20),
取k个最相似的数据中出现次数最多的分类, 作为新数据的分类.
"""

from __future__ import print_function

import os
import operator

import numpy
import matplotlib
import matplotlib.pyplot as plt

""" 手写识别系统 """


def VectorDebugPrint(vector):
    for i in range(32):
        print(''.join(
            list(map(
                lambda x: str(int(x)),
                vector[i*32:(i+1)*32]
            ))
        ))

def normalize(dataset):
  """ 对dataset进行归一化处理, 使得输入的特征权重一致 """
  minVals = dataset.min(0)  # 获取每一列的最小值
  maxVals = dataset.max(0)  # 获取每一列的最大值
  ranges = maxVals - minVals  # 每一列的范围
  m, n = dataset.shape
  # 归一化 (Xi - Xmin) / (Xmax - Xmin)
  normDataset = (dataset - numpy.tile(minVals, (m, 1))) / numpy.tile(ranges, (m, 1))
  return normDataset, ranges, minVals

def TranslateImg2Vector(filename):
    """ 把'图像文件'转换为1024维的向量 """
    vector = numpy.zeros((1, 1024))
    with open(filename, 'r') as infile:
        for lineno, line in enumerate(infile):
            for rowno in range(32):
                vector[0, 32*lineno+rowno] = int(line[rowno])
        return vector


def GetDigitsDatasetFromDir(dataset_dir='./digits/trainingDigits'):
    """从文件夹中获取数据集, labels

    Parameters
    ----------
    dataset_dir 文件夹名称

    Returns
    -------
    numpy.array, labels : 数据集, 数据集元素对应的标签
    """
    filenames = os.listdir(dataset_dir)

    labels = [None] * len(filenames)
    dataset = numpy.zeros((len(filenames), 1024))

    for i, filename in enumerate(filenames):
        fileclass = filename.split('.')[0].split('_')[0]
        filepath = os.path.join(dataset_dir, filename)
        dataset[i, :], labels[i] = TranslateImg2Vector(filepath), fileclass
    return dataset, labels

def predict(testVec, train_dataset, train_labels, k):
    diffMat = numpy.tile(testVec, (train_dataset.shape[0], 1)) - train_dataset
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedIndices = distances.argsort()
    classCount={}
    for x in range(k):
        voteLabel = train_labels[sortedIndices[x]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def TestHandwritingNumber(trainDir='./digits/trainingDigits', testDir='digits/testDigits', k=3):
    train_dataset, train_labels = GetDigitsDatasetFromDir(trainDir)
    # normDataset, ranges, minVals = normalize(train_dataset)

    test_dataset, test_labels = GetDigitsDatasetFromDir(testDir)
    numError = 0
    numTestVectors = len(test_labels)
    for testVec, testLabel in zip(test_dataset, test_labels):
        result = predict(testVec, train_dataset, train_labels, k)
        if result != testLabel:
            numError += 1
            print('× Predict/Real {0}/{1}'.format(result, testLabel))
        else:
            print('√ Predict/Real {0}/{1}'.format(result, testLabel))
    print('Total error rate: {0:.1%}'.format(1.0*numError / numTestVectors))

# import handsRecogition
# reload(handsRecogition)
# train_dataset, train_labels = handsRecogition.GetDigitsDatasetFromDir()
# test_dataset, test_labels = handsRecogition.GetDigitsDatasetFromDir('digits/testDigits')
# result = handsRecogition.predict(test_dataset[0], train_dataset, train_labels, 3)

if __name__ == '__main__':
  pass
