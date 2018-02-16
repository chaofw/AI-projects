#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 16:29:32 2018

@author: Chaofeng Wang
"""

##
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np


##############################################################
DecisionTree = RandomForestRegressor()

DecisionTree.fit(X_train, Y_train)

Y_test = DecisionTree.predict(X_test)


accuracy = DecisionTree.score(X_train, Y_train)

Id = pd.DataFrame({'Id': range(8000)})

Y_test = pd.DataFrame(Y_test)

Y_test.insert(0, 'Id', Id)

Y_test.to_csv('submission.csv', index = False)
###############################################################
