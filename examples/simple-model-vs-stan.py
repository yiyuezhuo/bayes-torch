# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 21:37:17 2018

@author: yiyuezhuo
"""

from bayestorch import Parameter,Data,optimizing,vb,sampling,reset
import torch
from torch.distributions import Normal

_X = torch.arange(10)
print(_X.mean())

# torch-bayes model

mu = Parameter(0.0)
sigma = Data(1.0)
X = Data(_X)

def target():
    target = Normal(mu,sigma).log_prob(X).sum(0)
    return target

optimizing(target)
print(f'optimizing: mu={mu.data}')

res = vb(target)
q_mu = res.params['mu']
print(f'vb mu={q_mu["loc"]} omega={q_mu["omega"]} sigma={torch.exp(q_mu["omega"])}')

reset()

mu = Parameter(0.0)

res = vb(target, q_size = 10, n_epoch=200)
q_mu = res.params['mu']
print(f'vb mu={q_mu["loc"]} omega={q_mu["omega"]} sigma={torch.exp(q_mu["omega"])}')

trace = sampling(target,trace_length=300)
mu_trace = torch.tensor([t['mu'].item() for t in trace])
print('sampling: mu={} sigma={}'.format(torch.mean(mu_trace), torch.std(mu_trace)))


# stan model


import numpy as np
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

_X = _X.numpy()
res2 = sm.optimizing(data = dict(N = len(_X), y = _X))
print(f'optimizing(stan): mu={res2["mu"]}')
res3 = sm.vb(data = dict(N = len(_X), y = _X))
res3a=np.array(res3['sampler_params'])
print(f'vb(stan): mu={res3a[0,:].mean()} sigma={res3a[0,:].std()}')
