#!/usr/bin/env python
# encoding=utf-8

'''
PCA 主成分分析
# TODO: Factor Analysis / Independent Component Analysis
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging

TRACE = logging.DEBUG - 1
logging.basicConfig(
    level=logging.DEBUG,
    # level=TRACE,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)

def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(np.float,line) for line in stringArr]
    return np.mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = np.cov(meanRemoved, rowvar=0)
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))
    eigValInd = np.argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i].A))[0],i]) #values that are not NaN (a number)
        datMat[np.nonzero(np.isnan(datMat[:,i].A))[0],i] = meanVal  #set NaN values to mean
    return datMat

# import pca
# reload(pca)
# dataMat = pca.loadDataSet('./testSet.txt')
# lowDDataMat, reconMat = pca.pca(dataMat, 1)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90)
# ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=90, c='red')
# plt.show()

# secomDataMat = pca.replaceNanWithMean()
# secomDataMean = np.mean(secomDataMat, axis=0)
# secomDataMeanRemoved = secomDataMat - secomDataMean
# secomDataCov = np.cov(secomDataMeanRemoved, rowvar=0)
# eigVals, tigVectors = np.linalg.eig(np.mat(secomDataCov))
# eigValInd = np.argsort(eigVals)
# eigValInd = eigValInd[:-(8+1):-1]
# eigVals[eigValInd]
# redEigVects = tigVectors[:,eigValInd]

# references:
# http://blog.csdn.net/zhongkelee/article/details/44064401
