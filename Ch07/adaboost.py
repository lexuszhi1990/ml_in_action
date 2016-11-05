#!/usr/bin/env python
# encoding=utf-8

"""
AdaBoost -- Adaptive boosting
===
通过串行训练多个分类器, 每一个分类器根据已训练出来的分类器的性能来进行训练,
每个新的分类器集中关注被已有分类器错分的那些数据来获得新的分类器.
最终把所有分类器的结果加权求和.
"""
import logging

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.DEBUG,
    # level=logging.INFO,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)
TRACE = logging.DEBUG - 1

def loadDataSet(filename):
    dataset = []
    labels = []
    with open(filename) as infile:
        for line in infile:
            datas = line.strip().split('\t')
            dataset.append(np.array(datas[0:-1], dtype=np.float))
            labels.append(float(datas[-1]))
    return dataset, labels

def loadSimpData():
    dataArr = np.matrix([[ 1. ,  2.1],
                        [ 2. ,  1.1],
                        [ 1.3,  1. ],
                        [ 1. ,  1. ],
                        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataArr,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    #just classify the data
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0 # 在特征的可能值上通过递增步长遍历的次数
    bestStump = {} # 记录对于给定权重向量D, 最佳的单层决策树
    bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf #init error sum, to +infinity
    for i in range(n): #loop over all feature dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr, classLabels, epoches=10):
    weekClassArr = []
    m, n = np.shape(dataArr)
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in xrange(epoches):
        bestStump,error,classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5*np.log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weekClassArr.append(bestStump)
        print "classEst: ",classEst.T
        print "current classEst: ", ((np.mat(classLabels).T == classEst).T)

        # 如果正确分类，D = D*np.exp(-alpha)/Sum(D)
        # 如果错误分类，D = D*np.exp(alpha)/Sum(D)
        # np.multiply( -1 * np.mat(classLabels).T, classEst).T 得到分类的正确与错误，即 matrix([[ 1., -1., -1., -1., -1.]])
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha * classEst
        print "aggClassEst: ", aggClassEst.T
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m,1)))
        print "agg est:", (np.sign(aggClassEst) == np.mat(classLabels).T).T
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate
        if errorRate == 0.0:
            print(weekClassArr)
            break

    return weekClassArr, aggClassEst

def addClassify(dotToClassify, clasiifierArr):
    dataMatrix = np.mat(dotToClassify)
    m, n = np.shape(dataMatrix)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in xrange(len(clasiifierArr)):
        classEst = stumpClassify(dataMatrix, clasiifierArr[i]['dim'], clasiifierArr[i]['thresh'], clasiifierArr[i]['ineq'])
        aggClassEst += clasiifierArr[i]['alpha'] * classEst
        print aggClassEst

    return np.sign(aggClassEst)

def plotROCCurve(predStrengths, labels):
    """
    ROC曲线(Receiver Operating Characteristic curve)
    ROC曲线给出当阈值变化时假阳率和真阳率的变化情况
    """
    cursor = (1.0, 1.0)  # 绘制光标的位置
    ySum = 0.0  # variable to calculate AUC
    numPositiveClass = sum(np.array(labels) == 1.0)  # 正例的数目
    step = {
        'x': 1.0 / numPositiveClass,
        'y': 1.0 / (len(labels) - numPositiveClass),
    }
    sortedIndicies = predStrengths.A1.argsort()  # get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies:
        if labels[index] == 1.0:
            deltaX = 0
            deltaY = step['x']
        else:
            deltaX = step['y']
            deltaY = 0
            ySum += cursor[1]
        # draw line from cursor to (cursor[0]-deltaX, cursor[1]-deltaY)
        logging.debug('Drawing line from {} -> {}'.format(
            cursor, (cursor[0]-deltaX, cursor[1]-deltaY)
        ))
        ax.plot(
            [cursor[0], cursor[0]-deltaX],
            [cursor[1], cursor[1]-deltaY],
            c='b'
        )
        cursor = (cursor[0] - deltaX, cursor[1] - deltaY)
    ax.plot([0, 1], [0, 1], 'b--')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    logging.info('曲线下面积AUC(Area Under the Curve): {}'.format(ySum * step['y']))

# import adaboost
# reload(adaboost)
# dataArr, classLabels = adaboost.loadSimpData()
# D=np.mat(np.ones((5,1))/5)
# bestStump,minError,bestClasEst = adaboost.buildStump(dataArr,classLabels, D)
# weekClassArr, aggClassEst = adaboost.adaBoostTrainDS(dataArr,classLabels)
# adaboost.addClassify([0,0], weekClassArr)
# adaboost.plotROCCurve(aggClassEst.T, classLabels)
