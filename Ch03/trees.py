#!/usr/bin/env python
# encoding=utf-8

"""
决策树
"""
from __future__ import print_function

import math
import operator
import pickle
from collections import defaultdict
import numpy as np

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [1, 0, 'other'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calculateShannonEntropy(dataSet):
  entropy = 0
  # dataSet, labels = createDataSet()
  totalNum = len(dataSet)
  labelList = defaultdict(int)
  for data in dataSet:
    labelList[data[-1]] += 1
  for label in labelList:
    probability = 1.0 * labelList[label] / totalNum
    entropy -= probability * math.log(probability, 2)

  return entropy

def splitDateSet(dataSet, axis, value):
  resultDateSet = []
  for featureVector in dataSet:
    if value == featureVector[axis]:
      resultDateSet.append(featureVector[:axis] + featureVector[axis+1:])

  return resultDateSet

def split(dataSet, axis):
  subDatasets = defaultdict(list)
  for featureVector in dataSet:
    value = featureVector[axis]
    subDatasets[value].append(featureVector[:axis] + featureVector[axis+1:])

  return subDatasets


def searchBestFeatureToSplit(dataSet):
  totalFeatureNum = len(dataSet[0]) - 1
  baseEntropy = calculateShannonEntropy(dataSet)
  bestFeature = -1
  bestInfoGain = 0.0

  for i in range(totalFeatureNum):
    featureList = [data[i] for data in dataSet]
    uniqueValues = set(featureList)
    newEntropy = 0.0
    for value in uniqueValues:
      subDatasets = splitDateSet(dataSet, i, value)
      subEntropy = calculateShannonEntropy(subDatasets)
      probability = len(subDatasets) * 1.0 / len(dataSet)
      newEntropy += probability * subEntropy

    newInfoGain = baseEntropy - newEntropy
    if newInfoGain > bestInfoGain:
      bestInfoGain = newInfoGain
      bestFeature = i

  return bestFeature

def majorityCnt(classList):
  # classCount = {}
  # for vote in classList:
  #   if vote not in classCount.keys():
  #     classCount[vote] = 0
  #   classCount[vote] += 1
  classCount_dict = defaultdict(int)
  for data in classList:
    classCount_dict[data] += 1
  result = sorted(classCount_dict.iteritems(), key=operator.itemgetter(1), reverse=True)

  return result[0][0]


def createTree(dataSet, labels):
  classList = [example[-1] for example in dataSet]
  if classList.count(classList[0]) == len(classList):
    return classList[0]
  if len(dataSet[0]) == 1:
    return majorityCnt(classList)

  bestFeature = searchBestFeatureToSplit(dataSet)
  bestLabel = labels[bestFeature]
  myTree = {bestLabel: {}}
  del(labels[bestFeature])
  featureValues = [example[bestFeature] for example in dataSet]
  uniqueValues = set(featureValues)
  for value in uniqueValues:
    subLabel = labels[:]
    myTree[bestLabel][value] = createTree(splitDateSet(dataSet, bestFeature, value), subLabel)

  return myTree

def classify(decisionTree, labels, testVector):
  rootLabel = decisionTree.keys()[0]
  secondDict = decisionTree[rootLabel]
  rootLabelIndex = labels.index(rootLabel)

  for key in secondDict.keys():
    if testVector[rootLabelIndex] == key:
      if type(secondDict[key]).__name__ == 'dict':
        classLabel = classify(secondDict[key], labels, testVector)
      else:
        classLabel = secondDict[key]

  return classLabel


def createLensesTree(lenses_file='./lenses.txt'):
  fr = open(lenses_file)
  lenses = [inst.strip().split() for inst in fr.readlines()]
  lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
  lenseTree =  createTree(lenses, lensesLabels)
  print(lenseTree)

  return lenseTree, ['age', 'prescript', 'astigmatic', 'tearRate']

# import trees
# reload(trees)
# dataSet, labels = trees.createDataSet()
# trees.calculateShannonEntropy(dataSet)
# trees.splitDateSet(dataSet, 0, 0)
# trees.split(dataSet, 0)
# trees.searchBestFeatureToSplit(dataSet)
# myTree = trees.createTree(dataSet, labels)
# trees.classify(myTree, labels, [1,1])

# lenseTree, lensesLabels = trees.createLensesTree()
# trees.classify(lenseTree, lensesLabels, ['pre', 'hyper', 'no', 'reduced'])
