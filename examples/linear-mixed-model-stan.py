# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 13:41:56 2018

@author: yiyuezhuo
"""

import pystan
import pandas as pd

alphabet_to_int = {a:i+1 for i,a in enumerate('ABCDEF')}
rail = pd.read_csv('data/Rail.csv',index_col=0)
N = len(rail)

standard_mu = 0.0
in_rail = [alphabet_to_int[t] for t in rail['Rail'].tolist()]
y = rail['travel'].as_matrix()

data = dict(N=N,
            Nb=6,
            rail=in_rail,
            y=y)

sm = pystan.StanModel(file = 'linear-mixed-model.stan')

fit = sm.vb(data)
trace = sm.sampling(data)