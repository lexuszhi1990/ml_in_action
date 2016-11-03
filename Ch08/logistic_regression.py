#!/usr/bin/env python
# encoding=utf-8

import logging

# http://stackoverflow.com/questions/18199853/error-could-not-create-library-python-2-7-site-packages-xlrd-permission-den
# pip install --upgrade matplotlib --trusted-host pypi.douban.com --user
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.DEBUG,
    # level=logging.INFO,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)
TRACE = logging.DEBUG - 1

def loadDatasetFromFile(filename='./ex0.txt'):
    dataset = []
    labels = []
    num_features = None
    with open(filename) as infile:
        for line in infile:
            line = line.strip().split('\t')
            if num_features is None:
                num_features = len(line)
            dataset.append(list(map(float, line[:-1])))
            labels.append(float(line[-1]))
        return dataset, labels

def standRegres(xArr,yArr):
    """使用普通最小二乘法求回归系数"""
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        return
    ws = xTx.I * (xMat.T*yMat)
    # w = np.linalg.solve(xTx, xMatrix.T * yMatrix)
    return ws

def lwlr(testPoint, xArr, yArr, k=0.5):
  xMat = np.mat(xArr); yMat = np.mat(yArr).T
  m, _n = np.shape(xMat)
  weights = np.mat(np.eye(m))
  for x in xrange(m):
    diffMat = testPoint - xMat[x, :]
    weights[x, x] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
  xTx = xMat.T * (weights * xMat)
  if np.linalg.det(xTx) == 0.0:
    print "This matrix is singular, cannot do inverse"
    return

  ws = xTx.I * (xMat.T * (weights * yMat))
  return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
  m, _n = np.shape(xArr)
  yHat = np.zeros(m)
  for i in xrange(m):
    yHat[i] = lwlr(testArr[i], xArr, yArr, k)

  return yHat

def ridgeRegression(xArr, yArr, lam=0.2):
  xMat, yMat = np.mat(xArr), np.mat(yArr)
  denom = xMat.T * xMat + np.eye(xMat.shape[1]) * lam
  if np.linalg.det(denom) == 0.0:
    print "This matrix is singular, cannot do inverse"
    return

  ws = denom.I * (xMat.T * yMat)
  return ws

def ridgeTest(filename='./abalone.txt'):
  xArr, yArr = loadDatasetFromFile(filename)
  xMat = np.mat(xArr)
  yMat = np.mat(yArr).T
  # 标准化Y
  yMean = np.mean(yMat, 0)
  yMat = yMat - yMean     # to eliminate X0 take np.mean off of Y
  # 标准化X的每一维
  xMeans = np.mean(xMat, 0)   # calc np.mean then subtract it off
  xVar = np.var(xMat, 0)      # calc variance of Xi then divide by it
  xMat = (xMat - xMeans) / xVar

  numTestPts = 30
  _m, n = xMat.shape
  wMatrix = np.zeros((numTestPts, n))
  for i in range(numTestPts):
      ws = ridgeRegression(xMat, yMat, np.exp(i - 10))
      wMatrix[i, :] = ws.T
  return wMatrix

def drawRidgeRegress(wMatrix):
  figure = plt.figure()
  ax = figure.add_subplot(111)
  ax.plot(wMatrix)
  plt.show()

def drawRegressionLine(xArr, yArr, ws):
  figure = plt.figure()
  ax = figure.add_subplot(111)
  xMat = np.mat(xArr); yMat = np.mat(yArr)
  ax.scatter(
      # xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0]
      xMat[:, 1].A1, yMat.T.A1,
      s=2, c='red'
  )  # 原始数据

  # sorted_index = xMat[:, 1].argsort(axis=0)
  # xSort = xMat[sorted_index][:, 0, :]
  # yHat = xMat * ws
  # ax.plot(xSort[:, 1], yHat[sorted_index])  # 拟合曲线
  xCopy = xMat.copy()
  xCopy.sort(0)
  yHat = xCopy * ws
  ax.plot(xCopy[:,1], yHat)

  plt.show()

def drawLwRegressionLine(xArr, yArr, yHat):
  figure = plt.figure()
  ax = figure.add_subplot(111)
  xMat = np.mat(xArr); yMat = np.mat(yArr)

  sortedIndex = xMat[:, 1].argsort(axis=0)
  xSort = xMat[sortedIndex][:, 0, :]
  ax.plot(xSort[:, 1], yHat[sortedIndex])  # 拟合曲线

  ax.scatter(
      # xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0]
      xMat[:, 1].A1, yMat.T.A1,
      s=2, c='red'
  )  # 原始数据

  plt.show()

def rssError(yArray, yHatArr):
    """计算预测误差"""
    yArray = np.array(yArray)
    yHatArr = np.array(yHatArr)
    return ((yArray - yHatArr)**2).sum()

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat,0)   #calc mean then subtract it off
    inVar = np.var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

def stageWise(xArr, yArr, lr, iterations):
  xMatrix = np.mat(xArr); yMatrix=np.mat(yArr).T
  xMatrix = regularize(xMatrix)
  yIn = np.mean(yArr, axis=0)
  yMatrix = yMatrix - yIn
  m, n = np.shape(xMatrix)
  ws = np.zeros((n, 1)); wsTest = ws.copy(); wsMax = ws.copy()
  returnMat = np.zeros((iterations,n))
  for i in range(iterations):
    print ws.T
    lowestError = np.inf
    for j in range(n):
      for sign in [-1, 1]:
        wsTest = ws.copy()
        wsTest[j] += lr*sign
        yTest = xMatrix * wsTest
        rssE = rssError(yMatrix.A, yTest.A)
        if rssE < lowestError:
          lowestError = rssE
          wsMax = wsTest

    ws = wsMax.copy()
    returnMat[i,:]=ws.T

  return returnMat

# import logistic_regression
# reload(logistic_regression)

# xArr, yArr = logistic_regression.loadDatasetFromFile()
# ws = logistic_regression.standRegres(xArr, yArr)
# logistic_regression.drawRegressionLine(xArr, yArr, ws)
# k=0.5
# lwlrWs = logistic_regression.lwlr(xArr[0], xArr, yArr, k)
# yHat = logistic_regression.lwlrTest(xArr, xArr, yArr, k)
# logistic_regression.drawLwRegressionLine(xArr, yArr, yHat)
# wMatrix = logistic_regression.ridgeTest()
# logistic_regression.drawRidgeRegress(wMatrix)

# stageWise for abalone
# xArr, yArr = logistic_regression.loadDatasetFromFile('./abalone.txt')
# returnMat = logistic_regression.stageWise(xArr, yArr, 0.05, 2000)
# logistic_regression.drawRidgeRegress(returnMat)
#
