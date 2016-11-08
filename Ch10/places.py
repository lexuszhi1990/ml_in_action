#!/usr/bin/env python
# encoding=utf-8

from numpy import *
import matplotlib
import matplotlib.pyplot as plt

import kMeans

def clusterClubs(kMeansFunc=kMeans.biKmeans, distMeas=kMeans.distSLC, numClust=5):
    datList = []
    for line in open('./places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = kMeansFunc(datMat, numClust, distMeas)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

# import places
# reload(places)
# places.clusterClubs(kMeansFunc=kMeans.kMeans, numClust=5)
# places.clusterClubs(kMeansFunc=kMeans.biKmeans, numClust=5)
