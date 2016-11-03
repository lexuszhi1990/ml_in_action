#!/usr/bin/env python
# encoding=utf-8


from __future__ import print_function

import os
import operator

import numpy
import matplotlib
import matplotlib.pyplot as plt

# 带归一化的kNN分类器

def getFileDataset(filename='datingTestSet.txt'):
  """把文本中的数据转换为数据集, labels返回

  Parameters
  ----------
  filename : string
      文本名

  Returns
  -------
  numpy.array, list : 文本中的数据矩阵, 数据对应的标签列表
  """
  with open(filename) as infile:
      lines = infile.readlines()
      numberOflines = len(lines)
      dataset = numpy.zeros((numberOflines, 3))
      dataLabels = []
      for index, line in enumerate(lines):
          listFromLine = line.strip().split()
          dataset[index,:] = listFromLine[0:3]
          dataLabels.append(int(listFromLine[-1]))
      return dataset, dataLabels

def drawPlot(dataset, labels, v, h):
  """绘制散点图

  Parameters
  ----------
  dataset : numpy.array
      数据集
  labels : list of int
      标签值
  """
  fig = plt.figure()
  ax = fig.add_subplot(111)
  _ = ax.scatter(
      dataset[:, v], dataset[:, h],
      s=15.0*numpy.array(labels),   # 大小
      c=15.0*numpy.array(labels)    # 颜色
  )
  plt.show()

def normalize( dataset):
  """ 对dataset进行归一化处理, 使得输入的特征权重一致 """
  minVals = dataset.min(0)  # 获取每一列的最小值
  maxVals = dataset.max(0)  # 获取每一列的最大值
  ranges = maxVals - minVals  # 每一列的范围
  m, n = dataset.shape
  # 归一化 (Xi - Xmin) / (Xmax - Xmin)
  normDataset = (dataset - numpy.tile(minVals, (m, 1))) / numpy.tile(ranges, (m, 1))
  return normDataset, ranges, minVals

def predict( inX, normDataset, labels, k):
  if k <= 0:
      raise ValueError('K > 0')

  # 利用矩阵运算, 每个 dataset 的分量都减去inX
  diffMat = numpy.tile(inX, (normDataset.shape[0], 1)) - normDataset
  # 计算欧式距离 sqrt(sum())
  distances = ((diffMat**2).sum(axis=1))**0.5
  # 对数据从小到大次序排列，确定前k个距离最小元素所在的主要分类
  sortedDistInd = distances.argsort()
  classCount={}
  for i in range(k):
      voteIlabel = labels[sortedDistInd[i]]
      classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
  # 返回最相近的类
  sortedClassCount = sorted(
      classCount.items(), key=operator.itemgetter(1), reverse=True
  )
  return sortedClassCount[0][0]


def test(cls, testfile='datingTestSet.txt', k=3, ratio=0.10):
    dataset, labels = getFileDataset(testfile)
    m, n = dataset.shape
    numTestVectors = int(m * ratio)
    numError = 0

    model = cls(dataset[numTestVectors:m, :], labels[numTestVectors:m])
    for i in range(numTestVectors):
        result = model.predict(dataset[i, :], k)
        if result != labels[i]:
            numError += 1
            print('× Predict/Real {0}/{1}'.format(result, labels[i]))
        else:
            print('√ Predict/Real {0}/{1}'.format(result, labels[i]))
    print('Total error rate: {0:.1%}'.format(1.0*numError / numTestVectors))

def TestClassifyPerson(dataset_filename='datingTestSet.txt'):
  result2str = {
      1: '完全不感兴趣',
      2: '可能喜欢',
      3: '很有可能喜欢',
  }
  print('请输入该人的相关信息:')
  percentageTimeOfPlayGames = float(
      input('消耗在玩游戏上的时间百分比?\n： ')
  )
  flyMiles = float(
      input('每年搭乘飞机的飞行里程数?\n： ')
  )
  iceCream = float(
      input('每周消费的冰淇淋公升数?\n： ')
  )

  dataset, labels = getFileDataset(dataset_filename)
  # drawPlot(dataset, labels)
  normDataset, ranges, minVals = normalize(dataset)
  # drawPlot(normDataset, labels, 0, 1)
  # flyMiles, percentageTimeOfPlayGames, iceCream = 3000, 0.3, 1
  inVector = numpy.array([flyMiles, percentageTimeOfPlayGames, iceCream]) # 先对输入特征进行归一化处理
  inVector = (inVector - minVals)/ranges
  classifierResult = predict(inVector, normDataset, labels, 4)

  print(
      '预测你对这个人:', result2str[classifierResult]
  )

# import kNNWithNormalize
# reload(kNNWithNormalize)

# dataset, labels = kNNWithNormalize.getFileDataset()
# kNNWithNormalize.drawPlot(dataset, labels)
# normDataset, ranges, minVals = kNNWithNormalize.normalize(dataset)
# kNNWithNormalize.drawPlot(normDataset, labels, 0, 1)
# flyMiles, percentageTimeOfPlayGames, iceCream = 3000, 0.3, 1
# inVector = numpy.array([flyMiles, percentageTimeOfPlayGames, iceCream]) # 先对输入特征进行归一化处理
# inVector = (inVector - minVals)/ranges
# kNNWithNormalize.predict(inVector, normDataset, labels, 4)

