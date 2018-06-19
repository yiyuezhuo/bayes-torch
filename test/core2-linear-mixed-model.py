# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 08:28:58 2018

@author: yiyuezhuo
"""

import bayestorch as bt
import pandas as pd
#import numpy as np
import torch
from torch import tensor

from bayestorch.core2 import Parameter,Data,optimizing,vb,sampling,reset
#from bayestorch.distributions import norm_log_prob
from torch.distributions import Normal

alphabet_to_int = {a:i for i,a in enumerate('ABCDEF')}


rail = pd.read_csv('data/Rail.csv',index_col=0)

reset()

standard_mu = Data(0.)
in_rail = Data([alphabet_to_int[t] for t in rail['Rail'].tolist()])
y = Data(tensor(rail['travel'].as_matrix()).float())

#b = Parameter(torch.zeros(6))
#mu = Parameter(0.0)
b = Parameter([-9.3730793 , -26.1230793 ,  13.6269207 ,  22.12691879, -12.3730793 ,  12.1269207])
mu = Parameter(66.49797821)


omega_b = Parameter(0.)
omega_epsilon = Parameter(0.)
#omega_b = Data(0) # set it as parameter will raise numerical problem
#omega_epsilon = Data(0)


def target():
    # transformed parameters
    sigma_b = torch.exp(omega_b)
    sigma_epsilon = torch.exp(omega_epsilon)
    #yhat = mu + b[in_rail]
    yhat = mu.unsqueeze(-1).expand_as(in_rail) + b.gather(-1,in_rail)
    
    # model
    #log_prob_b = norm_log_prob(b, standard_mu, sigma_b).sum()
    #log_prob_y = norm_log_prob(y, yhat, sigma_epsilon).sum()
    
    _standard_mu = standard_mu.unsqueeze(-1).expand_as(b)
    _sigma_b = sigma_b.unsqueeze(-1).expand_as(b)
    _sigma_epsilon = sigma_epsilon.unsqueeze(-1).expand_as(yhat)
    
    log_prob_b = Normal(_standard_mu,_sigma_b).log_prob(b).sum(-1)
    log_prob_y = Normal(yhat,_sigma_epsilon).log_prob(y).sum(-1)
    return log_prob_b + log_prob_y



#res=vb(target)
#target().backward()
#diagnosis(target, n_epoch=5)

optimizing(target)

fit=vb(target,q_size=3,lr=1e-5)
fit2=vb(target,n_epoch=1000)
fit3=vb(target,n_epoch=2000)


print('vb:')
print(fit)

'''
print('sampling:')
trace = np.array(sampling(target))
print(trace.mean(axis=0),trace.std(axis=0))
'''

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
    
    
    '''
    print('fit sigma_b mu')
    print(np.exp(fit_mu[index[omega_b]]))
    print('fit sigma_b sigma')
    print('???')
    
    
    print('fit omega_epsilon mu')
    print(fit_mu[index[omega_epsilon]])
    print('fit omega_epsilon sigma')
    print(np.exp(fit_omega[index[omega_epsilon]]))
    
    print('fit sigma_epsilon mu')
    print(np.exp(fit_mu[index[omega_epsilon]]))
    print('fit sigma_epsilon sigma')
    print('???')
    '''

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