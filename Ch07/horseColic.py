#!/usr/bin/env python
# encoding=utf-8

import logging
import numpy as np
import matplotlib.pyplot as plt

import adaboost

logging.basicConfig(
    level=logging.DEBUG,
    # level=logging.INFO,
    format='[%(levelname)s %(module)s line:%(lineno)d] %(message)s',
)
TRACE = logging.DEBUG - 1

datArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
weekClassArr, aggClassEst = adaboost.adaBoostTrainDS(datArr, labelArr, 20)

testDatArr, testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
prediction = adaboost.addClassify(testDatArr, weekClassArr)

errArr = np.mat(np.ones((67,1)))
errArr[prediction != np.mat(testLabelArr).T]
errArr[prediction != np.mat(testLabelArr).T].sum()/67
