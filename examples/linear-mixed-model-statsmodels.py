# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 14:11:49 2018

@author: yiyuezhuo
"""

import statsmodels.api as sm
import statsmodels.formula.api as smf

import numpy as np
import pandas as pd

rail = pd.read_csv('data/Rail.csv',index_col=0)

fit = smf.mixedlm("travel ~ 1", rail, groups=rail["Rail"]).fit()
print(fit.summary())