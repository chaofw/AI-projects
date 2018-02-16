#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 17:01:23 2018

@author: Chaofeng Wang
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import Series, DataFrame


##############################################
##pre-processing using pandas



train = pd.read_csv("train.csv")
## train.info()

## fillna to fill missing data

## delete non-sense features
train = train.drop(['Id'], axis = 1)

##analysis
#sns.factorplot('Col_3', data = train, kind = 'count', order = ['a','b','c','d','e','f']) ##to see count

ColType = train.dtypes

## to see relation of categorial variables and score
#for ii in range(1,33):
#    if ColType[ii] == object:
#        sns.factorplot(ColType.index[ii],'Score', data = train) ##to see relation between features and score

##one-hot mapping
train_temp = train.copy()

for ii in range(33):
    if ColType[ii] == 'object':
        col_str = ColType.index[ii]
        Col_dummies = pd.get_dummies(train[col_str])
        col_temp = []
        for jj in range(Col_dummies.columns.size):
            col_temp.append(col_str + '_' + Col_dummies.columns[jj])
        Col_dummies.columns = col_temp
        train_temp.drop([ColType.index[ii]], axis = 1, inplace = True)
        train_temp = train_temp.join(Col_dummies)
        
X_train = train_temp.drop('Score',axis = 1)
Y_train = train_temp['Score']


##############################################################
###for test data
test = pd.read_csv('test.csv')

test = test.drop(['Id'], axis = 1)


test_temp = test.copy()

for ii in range(33):
    if ColType[ii] == 'object':
        col_str = ColType.index[ii]
        Col_dummies = pd.get_dummies(test[col_str])
        col_temp = []
        for jj in range(Col_dummies.columns.size):
            col_temp.append(col_str + '_' + Col_dummies.columns[jj])
        Col_dummies.columns = col_temp
        test_temp.drop([ColType.index[ii]], axis = 1, inplace = True)
        test_temp = test_temp.join(Col_dummies)
        
X_test = test_temp
    

