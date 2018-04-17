# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 08:24:28 2018

@author: yiyuezhuo
"""

from bayestorch.core import Parameter,Data,optimizing,vb,sampling
from bayestorch.distributions import norm_log_prob

import numpy as np

_X = np.random.random(10)
print(_X.mean())
#0.6720278728059971

# torch-bayes model

mu = Parameter(0.0)
sigma = Data(1.0)
X = Data(_X)

def target():
    target = norm_log_prob(X,mu,sigma).sum()
    return target

optimizing(target)
print('optimizing: mu={}'.format(mu.data.numpy()))
# optimizing: mu=[0.68792978] omega=[-1.09238617] sigma=[0.33541518]
res = vb(target)
print('vb: mu={} omega={} sigma={}'.format(res[0],res[1],np.exp(res[1])))
#vb: mu=[0.66530417] omega=[-1.05874966] sigma=[0.34688927]
trace = sampling(target)
print('sampling: mu={} sigma={}'.format(np.mean(trace), np.std(trace)))
