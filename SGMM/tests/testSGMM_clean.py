#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:10:04 2019

@author: george
"""

import sys
sys.path.append("/home/george/github/code/loaders")
sys.path.append("..")
sys.path.append("../../metrics")


import numpy as np
import pandas as pd



#def warn(*args, **kwargs):
#    pass
#import warnings
#warnings.warn = warn
#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
from supervisedGmm import SupervisedGMM

from metricsFunctions import sgmmResults
#from superGmmMother import superGmmMother
from loaders2 import loader
#from mlModels import logisticRegressionCv2, neural_nets, randomforests,\
#kmeansLogRegr

np.random.seed( seed = 0)
###############################################################################

#READING DATA SETTING COLUMNS NAMES FOR METRICS
dat = '/home/george/github/data/'
file1 = dat+'sparcs00.h5'
file2 = dat+'sparcs01.h5'

data, dataS, idx = loader(10000, 300, file1, file2)


cols = data.columns
#drop drgs and length of stay
colA = cols[761:1100]
colB = cols[0]
data = data.drop(colA, axis = 1)
data = data.drop(colB, axis = 1)
colss = data.columns.tolist()

columns = ['cluster', 'size', 'high_cost%','low_cost%', 
                       'TP', 'TN', 'FP', 'FN', 
                       'FPR', 'specificity', 'sensitivity', 'precision',
                       'accuracy', 'balanced accuracy', 'f1', 'auc']
###############################################################################

##Fitting SGMM
#alpha = [1]
n_clusters = 10
cv = 2
scoring = 'neg_log_loss'
mcov = 'diag'
mx_it2 = 10
mx_it = 1000
warm = 0
vrb = 0
adaR = 0
km = 1
parallel = 0
parallel2 = 0
solver = 'liblinear'
model = SupervisedGMM(  n_clusters = n_clusters, EM_iter = mx_it2, tol_EM = 10**(-5),
                         lg_iter = mx_it, mcov = mcov, adaR = adaR,
                         transduction = 0, verbose = vrb, scoring = scoring,
                         cv = cv, warm = warm, tol_lg = 10**(-2), solver = solver,
                         parallel = parallel, parallel2 = parallel2)



#SPLIT THE DATA
Xtrain, Xtest, ytrain, ytest = model.split( data = data.values)

#FIT THE MODEL 
start = time.time()
model = model.fit( Xtrain = Xtrain, Xtest = [], ytrain = ytrain,
                  mod = 1, kmeans = km, simple = 0, hard_cluster = 0)
end = time.time() - start
print( " Algorith run in {}s".format( end ))

probTest = model.predict_proba( Xtest )
probTrain = model.predict_proba( Xtrain )



res2 = sgmmResults(model, probTest, probTrain, ytest, ytrain)
