#!/usr/bin/env python
# encoding=utf-8

import logging
import numpy as np

TRACE = logging.DEBUG - 1
logging.basicConfig(
    level=logging.DEBUG,
    # level=TRACE,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])

    C1.sort()
    return map(frozenset, C1)#use frozen set so we
                            #can use it as a key in a dict

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can): ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k): #creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            # 如果前k-2项相同时，合并两项
            # list([1,3])[:1] #=>[1]
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: #if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j]) #set union
    return retList

def apriori(dataSet,minSupport):
    L = []
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L.append(L1)
    k = 2
    while (len(L[k-2]) > 0):
        ck = aprioriGen(L[k-2], k)
        lk, supData = scanD(D, ck, minSupport)
        L.append(lk)
        supportData.update(supData)
        k += 1

    return L, supportData

def calculateConfidience(frequentSet, H, supportData, brl, minConf=0.7):
    pruneH = []
    for conseq in H:
        # P --> H : support(P | H]) / support(P)
        conf = supportData[frequentSet] / supportData[frequentSet - conseq]
        if conf >= minConf:
            print(frequentSet-conseq, '--->', conseq, conf)
            brl.append((frequentSet-conseq, conseq, conf))
            pruneH.append(conseq)

    return pruneH

def rulesFromConsq(frequentSet, H, supportData, blr, minConf=0.7):
    m = len(H[0])
    if len(frequentSet) > (m+1):
        hmp1 = aprioriGen(H, m+1)
        hmp1 = calculateConfidience(frequentSet, hmp1, supportData, blr, minConf)
        if len(hmp1) > 1:
            rulesFromConsq(frequentSet, hmp1, supportData, blr, minConf)

def generateRules(L, supportData, minConf):
    bigRuleList = []
    for i in xrange(1,len(L)):
        for frequentSet in L[i]:
            H1 = [frozenset([item]) for item in frequentSet]
            if i > 1:
                rulesFromConsq(frequentSet, H1, supportData, bigRuleList, minConf)
            else:
                calculateConfidience(frequentSet, H1, supportData, bigRuleList, minConf)

def testWithMushroom(filename='./mushroom.dat'):
    mushroomDateSet = [line.split() for line in open(filename).readlines()]
    ML, MSupoortData = apriori.apriori(mushroomDateSet, minSupport = 0.5)
    apriori.generateRules(ML, MSupoortData, 0.5)


# import apriori
# reload(apriori)
# dataSet = apriori.loadDataSet()
# C1 = apriori.createC1(dataSet)
# L1,supportData = apriori.scanD(dataSet, C1, 0.5)
# L1,supportData = apriori.aprioriGen(L1, 2)
# L, supportData = apriori.apriori(dataSet, 0.5)

# frequentSet = L[1][0]
# H1 = [frozenset([item]) for item in frequentSet]
# supportData[frequentSet]/ supportData[frequentSet - H1[0]]
# apriori.calculateConfidience(frequentSet, H1, supportData, bigRuleList, minConf)

# frequentSet = L[2][1]
# H1 = [frozenset([item]) for item in frequentSet]
# apriori.rulesFromConsq(frequentSet, H1, supportData, bigRuleList, minConf)

# apriori.generateRules(L, supportData, 0.5)

# mushroom.dat

