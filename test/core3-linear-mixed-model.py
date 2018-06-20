# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 08:28:58 2018

@author: yiyuezhuo
"""

import bayestorch as bt
import pandas as pd
import torch
from torch import tensor

from bayestorch.core3 import Parameter,Data,optimizing,vb,sampling,reset
from torch.distributions import Normal

alphabet_to_int = {a:i for i,a in enumerate('ABCDEF')}


rail = pd.read_csv('data/Rail.csv',index_col=0)

reset()

standard_mu = Data(0.)
in_rail = Data([alphabet_to_int[t] for t in rail['Rail'].tolist()])
y = Data(tensor(rail['travel'].as_matrix()).float())

b = Parameter([-9.3730793 , -26.1230793 ,  13.6269207 ,  22.12691879, -12.3730793 ,  12.1269207])
mu = Parameter(66.49797821)


omega_b = Parameter(0.)
omega_epsilon = Parameter(0.)
# set they as parameter will raise numerical problem
#omega_b = Data(0) 
#omega_epsilon = Data(0)


def target():
    # transformed parameters
    sigma_b = torch.exp(omega_b)
    sigma_epsilon = torch.exp(omega_epsilon)
    #in_rail: n_sample * (q_size)
    yhat = mu + b.gather(0,in_rail)
    
    # model
    
    # b: dim_b * (q_size)
    log_prob_b = Normal(standard_mu, sigma_b).log_prob(b).sum(0)
    log_prob_y = Normal(yhat, sigma_epsilon).log_prob(y).sum(0)
    return log_prob_b + log_prob_y




optimizing(target)

fit=vb(target,q_size=3,lr=1e-5)
fit2=vb(target,n_epoch=1000)
#fit3=vb(target,n_epoch=10000)


print('vb:')
print(fit)


def explain_fit(fit):
    #fit_mu,fit_omega = fit
    
    def print_key(key):
        print(f'fit {key} mu')
        print(fit.params[key]['loc'])
        print(f'fit {key} sigma')
        print(torch.exp(fit.params[key]['omega']))
    
    print_key('b')
    print_key('mu')
    print_key('omega_b')
    print_key('omega_epsilon')
    
    

print('--------advi(epoch=100)----------')
explain_fit(fit) # The sigma seems lack of converge
print('--------advi(epoch=1000)----------')
explain_fit(fit2) 
# The result very close to statsmodels and lme4(Well the both of two are not close anyway)
'''
print('--------advi(epoch=10000)----------')
explain_fit(fit3)
print('---------hmc(n_sample=100)-------')
explain_fit((trace.mean(axis=0),trace.std(axis=0)))

print('---------hmc(n_sample=1000)-------')
trace = np.array(sampling(target,n_epoch=1000))
explain_fit((trace.mean(axis=0),trace.std(axis=0)))
'''