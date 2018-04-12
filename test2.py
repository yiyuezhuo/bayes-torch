# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:47:11 2018

@author: yiyuezhuo
"""

from core import Parameter,Data,optimizing,vb,sampling
from distributions import norm_log_prob

import numpy as np

_X = np.random.random(10)
print(_X.mean(),_X.std())

# torch-bayes model

mu = Parameter(0.0)
sigma = Data(1.0)
X = Data(_X)

def target():
    target = norm_log_prob(X,mu,sigma).sum()
    return target

optimizing(target)
print('optimizing: mu={}'.format(mu.data.numpy()))
res = vb(target)
print('optimizing: mu={} omega={} sigma={}'.format(res[0],res[1],np.exp(res[1])))

# stan model

import pystan
stan_code = '''
data {
    int<lower=1> N;
    real y[N];
}
parameters {
    real mu;
}
model {
    y ~ normal(mu, 1);
}
'''
sm = pystan.StanModel(model_code = stan_code)
res2 = sm.optimizing(data = dict(N = len(_X), y = _X))
print('optimizing(stan): mu={}'.format(res2['mu']))
res3 = sm.vb(data = dict(N = len(_X), y = _X))
res3a=np.array(res3['sampler_params'])
print('vb(stan): mu={} sigma={}'.format(res3a[:,0].mean(),res3a[:,0].std()))

