#!/usr/bin/env python
# encoding=utf-8

from __future__ import print_function

import copy
import logging

import numpy as np

TRACE = logging.DEBUG - 1
logging.basicConfig(
    level=logging.DEBUG,
    # level=TRACE,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split()
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def linearSolve(dataSet):   #helper function used in two places
    m,n = np.shape(dataSet)
    X = np.mat(np.ones((m,n))); Y = np.mat(np.ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse, try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet): #create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    """

    Parameters
    ----------
    leafType : int
        TYPE_VALUE 普通回归树
        TYPE_MODEL 模型回归树
    errType : float
        分裂叶节点时, 数据集方差和下降值最小值
    ops : (int, int)
        容许的误差下降值，叶节点中最少包含的样本数

    Returns
    -------
    (int, float) : 对数据集划分的最好特征的index, 划分值
    """
    # 如果所有值都相等, 生成一个叶节点
    tolS = ops[0]; tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf; bestValue = 0; bestFeatureIndex = 0
    for featureIndex in range(n-1):
        for splitValue in set(dataSet[:, featureIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featureIndex, splitValue)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestFeatureIndex = featureIndex
                bestS = newS
                bestValue = splitValue

    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestFeatureIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)

    return bestFeatureIndex, bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
    if np.shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'],2)) +\
            np.sum(np.power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = np.sum(np.power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else: return tree
    else: return tree

def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = np.mat(np.zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat

# import regressionTrees
# reload(regressionTrees)
# regressionTrees.loadDataSet()
# testMat = np.mat(np.eye(4))
# testMat[np.nonzero(testMat[:,3] > 0.5)[0], :]
# >>>  matrix([[ 0.,  0.,  0.,  1.]])
# mat0, mat1 = regressionTrees.binSplitDataSet(testMat, 1, 0.5)

# myMat = np.mat(regressionTrees.loadDataSet('ex00.txt'))
# regressionTrees.chooseBestSplit(myMat)
# regressionTrees.createTree(myMat)

# myMat1 = np.mat(regressionTrees.loadDataSet('ex0.txt'))
# regressionTrees.chooseBestSplit(myMat1)
# regressionTrees.createTree(myMat1)

# myMat2 = np.mat(regressionTrees.loadDataSet('ex2.txt'))
# myTree = regressionTrees.createTree(myMat2)
# myMatTest = np.mat(regressionTrees.loadDataSet('ex2test.txt'))
# regressionTrees.prune(myTree, myMatTest)

# 9.4
# myExpMat2 = np.mat(regressionTrees.loadDataSet('exp2.txt'))
# regressionTrees.createTree(myExpMat2, regressionTrees.modelLeaf, regressionTrees.modelErr, (1,10))

# trainMat = np.mat(regressionTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
# testMat = np.mat(regressionTrees.loadDataSet('bikeSpeedVsIq_test.txt'))
# bikeSpeedVsIqTree = regressionTrees.createTree(trainMat, ops=(1,10))
# yHat = regressionTrees.createForeCast(bikeSpeedVsIqTree, testMat[:, 0])
# np.corrcoef(yHat, testMat[:, 1], rowvar=0)

# bikeSpeedVsIqModelTree = regressionTrees.createTree(trainMat, regressionTrees.modelLeaf, regressionTrees.modelErr, (1,10))
# yHat = regressionTrees.createForeCast(bikeSpeedVsIqModelTree, testMat[:, 0], regressionTrees.modelTreeEval)
# np.corrcoef(yHat, testMat[:, 1], rowvar=0)
