# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 21:37:17 2018

@author: yiyuezhuo
"""

from bayestorch import Parameter,Data,optimizing,vb,sampling,reset
import torch
from torch.distributions import Normal
torch.manual_seed(42)

_X = torch.arange(10)
print(_X.mean())
# 4.5000

'''
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

# torch-bayes model

mu = Parameter(0.0) # optimizing/vb/sampling init value
sigma = Data(1.0)
X = Data(_X)

def target():
    target = Normal(mu,sigma).log_prob(X).sum(0)
    return target

optimizing(target)
print(f'optimizing: mu={mu.data}')
# 4.499998092651367

res = vb(target)
q_mu = res.params['mu']
print(f'vb mu={q_mu["loc"]} omega={q_mu["omega"]} sigma={torch.exp(q_mu["omega"])}')
# mu=4.400146484375 sd=0.3164467215538025


trace = sampling(target)
mu_trace = torch.tensor([t['mu'].item() for t in trace])
print('sampling: mu={} sigma={}'.format(torch.mean(mu_trace), torch.std(mu_trace)))
# mu=4.517702102661133 sd=0.33891919255256653

reset()

mu = Parameter(0.0)

res = vb(target, q_size = 10)
q_mu = res.params['mu']
print(f'vb mu={q_mu["loc"]} omega={q_mu["omega"]} sigma={torch.exp(q_mu["omega"])}')
# mu=4.484315395355225 sd=0.33686521649360657

reset()

mu = Parameter(0.0)

res = vb(target, n_epoch = 2000)
q_mu = res.params['mu']
print(f'vb mu={q_mu["loc"]} omega={q_mu["omega"]} sigma={torch.exp(q_mu["omega"])}')
# mu=4.518213272094727 sd=0.3279591500759125

reset()

mu = Parameter(0.0)


trace = sampling(target)
mu_trace = torch.tensor([t['mu'].item() for t in trace])
print('sampling: mu={} sigma={}'.format(torch.mean(mu_trace), torch.std(mu_trace)))
# mu=4.258555889129639 sd=0.6691126227378845
# That mu is less than true value and sd is greater than true is due to the 
# 'burn-in' trace were not droped. 


mu = Parameter(4.5)

trace = sampling(target,trace_length=300)
mu_trace = torch.tensor([t['mu'].item() for t in trace])
print('sampling: mu={} sigma={}'.format(torch.mean(mu_trace), torch.std(mu_trace)))
# mu=4.466207027435303 sd=0.28977543115615845