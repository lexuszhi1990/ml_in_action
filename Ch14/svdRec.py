#!/usr/bin/env python
# encoding=utf-8

'''
svd 简化数据
'''

import numpy as np
from numpy import linalg as la

def loadExData():
    return np.mat([[0, 0, 0, 2, 2],
                   [0, 0, 0, 3, 3],
                   [0, 0, 0, 1, 1],
                   [1, 1, 1, 0, 0],
                   [2, 2, 2, 0, 0],
                   [5, 5, 5, 0, 0],
                   [1, 1, 1, 0, 0]])

def loadExData2():
    return np.mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
                   [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
                   [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
                   [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
                   [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
                   [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
                   [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
                   [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
                   [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
                   [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
                   [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])

def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5+0.5*np.corrcoef(inA, inB, rowvar = 0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0: continue
        overLap = np.nonzero(np.logical_and(dataMat[:,item].A>0, \
                                      dataMat[:,j].A>0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def svdEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    Sig4 = np.mat(np.eye(4)*Sigma[:4]) #arrange Sig4 into a diagonal matrix
    xformedItems = dataMat.T * U[:,:4] * Sig4.I  #create transformed items
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print 'the %d and %d similarity is: %f' % (item, j, similarity)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user,:].A==0)[1]#find unrated items
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print 1,
            else: print 0,
        print ''

def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print "****original matrix******"
    printMat(myMat, thresh)
    U,Sigma,VT = la.svd(myMat)
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV):#construct diagonal matrix from vector
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print "****reconstructed matrix using %d singular values******" % numSV
    printMat(reconMat, thresh)

# import svdRec
# reload(svdRec)
# dataMat = svdRec.loadExData()
# svdRec.ecludSim(dataMat[:, 1], dataMat[:, 2])
# svdRec.pearsSim(dataMat[:, 1], dataMat[:, 2])
# svdRec.cosSim(dataMat[:, 1], dataMat[:, 2])

# myMat = svdRec.loadExData()
# myMat[0,1] = myMat[0,0] = myMat[1,0] = myMat[2,0] = 4
# myMat[3,3] = 2
# svdRec.recommend(myMat, 2)
# svdRec.recommend(myMat, 2, simMeas=svdRec.ecludSim)
# svdRec.recommend(myMat, 2, simMeas=svdRec.pearsSim)

# svdRec.recommend(myMat, 2, simMeas=svdRec.pearsSim, estMethod=svdRec.svdEst)
# svdRec.recommend(myMat, 2, simMeas=svdRec.ecludSim, estMethod=svdRec.svdEst)


# svdRec.imgCompress(2)

# reference:
# http://www.cnblogs.com/peizhe123/p/5113357.html
