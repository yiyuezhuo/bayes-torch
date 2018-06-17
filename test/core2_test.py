# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 21:37:17 2018

@author: yiyuezhuo
"""

from bayestorch.core2 import Parameter,Data,optimizing,vb,sampling,reset
import torch
from torch.distributions import Normal

_X = torch.rand(10)
print(_X.mean())
#0.6720278728059971

# torch-bayes model

mu = Parameter(0.0)
sigma = Data(1.0)
X = Data(_X)

def target():
    target = Normal(mu,sigma).log_prob(X).sum(-1)
    return target

optimizing(target)
print(f'optimizing: mu={mu.data}')

res = vb(target)
q_mu = res.params['mu']
print(f'vb mu={q_mu["loc"]} omega={q_mu["omega"]} sigma={torch.exp(q_mu["omega"])}')

reset()

mu = Parameter(0.0)

res = vb(target)
q_mu = res.params['mu']
print(f'vb mu={q_mu["loc"]} omega={q_mu["omega"]} sigma={torch.exp(q_mu["omega"])}')

