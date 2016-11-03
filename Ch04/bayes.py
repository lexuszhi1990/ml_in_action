#!/usr/bin/env python
# encoding=utf-8

"""
朴素贝叶斯
===
朴素贝叶斯是贝叶斯决策理论的一部分.
## 贝叶斯决策
* 核心思想 -> 选择具有最高概率的决策

## 朴素贝叶斯分类器
朴素贝叶斯分类器是用于文档分类的常用算法
* 把每个次的出现或者不出现作为一个特征
* 假设特征之间相互独立, 即一个单词出现的可能性和其他相邻单词没有关系
* 每个特征同等重要

"""

from __future__ import print_function

import numpy as np
from numpy import random
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
  vocabSet = set([])
  for data in dataSet:
    vocabSet |= set(data)

  return list(vocabSet)

def setOfWordsVector(vocabSet, words):
  vector = [0]*len(vocabSet)
  for word in words:
    vector[vocabSet.index(word)] = 1

  return vector

def bagOfWords2VecMN(vocabSet, words):
  vector = [0]*len(vocabSet)
  for word in words:
    if word in vocabSet:
      vector[vocabSet.index(word)] += 1

  return vector

def setOfTrainMatix(vocabSet, postingList):
  trainMatrix = []
  for post in postingList:
    trainMatrix.append(setOfWordsVector(vocabSet, post))

  return trainMatrix

def trainNaiveBayes(trainMatrix, trainCategories):
  totalTrainDocs = len(trainMatrix)
  pAbusive = np.sum(trainCategories) / float(totalTrainDocs)
  wordsNumber = len(trainMatrix[0])
  pClass1 = np.ones(wordsNumber); pClass0 = np.ones(wordsNumber)
  p0Denom = 2.0; p1Denom = 2.0

  for i in xrange(totalTrainDocs):
    if trainCategories[i] == 1:
      pClass1 += trainMatrix[i]
      p1Denom += np.sum(trainMatrix[i])
    else:
      pClass0 += trainMatrix[i]
      p0Denom += np.sum(trainMatrix[i])

  p1Vector = pClass1/p1Denom
  p0Vector = pClass0/p1Denom

  return p0Vector, p1Vector, pAbusive

def testNaiveBayes(testWords, vocabSet, p0Vector, p1Vector, pAbusive):
  testVector = np.array(setOfWordsVector(vocabSet, testWords))
  p1 = np.sum(testVector * p1Vector) + np.log(pAbusive)
  p0 = np.sum(testVector * p0Vector) + np.log(pAbusive)
  if p1 > p0:
    return 1
  else:
    return 0

def classifyNB(testVector, p0Vector, p1Vector, pAbusive):
  p1 = np.sum(testVector * p1Vector) + np.log(pAbusive)
  p0 = np.sum(testVector * p0Vector) + np.log(pAbusive)
  if p1 > p0:
    return 1
  else:
    return 0

def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = range(50); testSet=[]           #create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNaiveBayes(np.array(trainMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText


# import bayes
# reload(bayes)
# postingList, classVec = bayes.loadDataSet()
# vocabSet = bayes.createVocabList(postingList)
# bayes.setOfWordsVector(vocabSet, postingList[0])
# trainMatrix = bayes.setOfTrainMatix(vocabSet, postingList)
# pAbusive, p1Vector, p0Vector = bayes.trainNaiveBayes(trainMatrix, classVec)
# bayes.testNaiveBayes(['buying', 'worthless', 'dog'], vocabSet, p0Vector, p1Vector, pAbusive)
# bayes.spamTest()



# testVector = bayes.setOfWordsVector(vocabSet, ['buying', 'worthless', 'dog'])
