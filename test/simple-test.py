# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 20:48:21 2018

@author: yiyuezhuo
"""

import unittest

from bayestorch import Parameter,Data,optimizing,vb,sampling,reset
import torch
from torch.distributions import Normal

_X = torch.arange(10)
mean_X = _X.mean().item()

mu = Parameter(0.0)
sigma = Data(1.0)
X = Data(_X)

def target():
    target = Normal(mu,sigma).log_prob(X).sum(0)
    return target

def reset_mu(_mu):
    global mu
    reset() # delete mu from __globals__
    mu = Parameter(_mu)

class TestSimpleModel(unittest.TestCase):
    
    def test_optimizng(self):
        reset_mu(0.0)
        
        optimizing(target)
        self.assertLess( abs(mu.item() - mean_X), 1.0)
    
    def test_vb(self):
        reset_mu(0.0)
        
        res = vb(target)
        q_mu = res.params['mu']
        self.assertLess( abs(q_mu['loc'] - mean_X), 1.0)
        self.assertLess( q_mu['omega'], 1.0)

    def test_sampling(self):
        reset_mu(mean_X)
        
        trace = sampling(target)
        mu_trace = torch.tensor([t['mu'].item() for t in trace])
        self.assertLess( abs(mu_trace.mean() - mean_X), 1.0)
        
    def test_vb_qsize(self):
        reset_mu(0.0)
        
        res = vb(target, q_size = 7)
        q_mu = res.params['mu']
        self.assertLess( abs(q_mu['loc'] - mean_X), 1.0)
        self.assertLess( q_mu['omega'], 1.0)
        
    def test_vb_n_epoch(self):
        reset_mu(0.0)
        
        res = vb(target, n_epoch = 777)
        q_mu = res.params['mu']
        self.assertLess( abs(q_mu['loc'] - mean_X), 1.0)
        self.assertLess( q_mu['omega'], 1.0)

if __name__ == '__main__':
    unittest.main()
