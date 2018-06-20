# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 21:37:17 2018

@author: yiyuezhuo
"""

from bayestorch.core3 import Parameter,Data,optimizing,vb,sampling,reset
import torch
from torch.distributions import Normal

#_X = torch.rand(10)
_X = torch.arange(10)
print(_X.mean())
#0.6720278728059971

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

trace = sampling(target)
mu_trace = torch.tensor([t['mu'].item() for t in trace])
print('sampling: mu={} sigma={}'.format(torch.mean(mu_trace), torch.std(mu_trace)))


reset()

mu = Parameter(0.0)

res = vb(target, q_size = 10)
q_mu = res.params['mu']
print(f'vb mu={q_mu["loc"]} omega={q_mu["omega"]} sigma={torch.exp(q_mu["omega"])}')

reset()

mu = Parameter(0.0)

res = vb(target, n_epoch = 2000)
q_mu = res.params['mu']
print(f'vb mu={q_mu["loc"]} omega={q_mu["omega"]} sigma={torch.exp(q_mu["omega"])}')

reset()

mu = Parameter(0.0)


trace = sampling(target)
mu_trace = torch.tensor([t['mu'].item() for t in trace])
print('sampling: mu={} sigma={}'.format(torch.mean(mu_trace), torch.std(mu_trace)))

mu = Parameter(4.5)

trace = sampling(target,trace_length=300)
mu_trace = torch.tensor([t['mu'].item() for t in trace])
print('sampling: mu={} sigma={}'.format(torch.mean(mu_trace), torch.std(mu_trace)))
