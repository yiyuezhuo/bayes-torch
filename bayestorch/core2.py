# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 20:58:07 2018

@author: yiyuezhuo
"""

'''
A relative gerneralized framework will be employed, such as apply common 
distributions rather than meanfield or full-rank multivariate normal.

Pytorch 0.4 new features are required.
'''

import torch
import contextlib


# The two helper classes will be used to annotate what is parameters should 
# be optimized
class Parameter(torch.Tensor):
    def __new__(cls,*args,**kwargs):
        kwargs['requires_grad'] = True
        tensor = torch.tensor(*args,**kwargs)
        tensor.__class__ = cls
        return tensor

class Data(torch.Tensor):
    '''
    This class may be unnecessary.
    '''
    def __new__(cls,*args,**kwargs):
        kwargs['requires_grad'] = False
        tensor = torch.tensor(*args,**kwargs)
        tensor.__class__ = cls
        return tensor
    
def collect_parameters(target_f):
    parameters_dict = {}
    for key,value in target_f.__globals__.items():
        if isinstance(value, Parameter):
            parameters_dict[key] = value
    return parameters_dict

class NormalReparametrization:
    def __init__(self,loc,omega):
        self.loc = loc
        self.omega = omega
        self.dis = torch.distributions.Normal(loc.size(),omega.size())
    def sample(self):
        return self.dis.sample() * torch.exp(self.omega) + self.loc
    
class VariationalMeanFieldDistribution:
    def __init__(self,target_f, q_size=1):
        self.target_f = target_f
        self.q_size = 1
        
        self.params = {}
        for name,variable in collect_parameters(target_f).items():
            param = dict(loc = variable,
                         omega = torch.zeros(variable.shape, requires_grad=True),
                         size = torch.Size([q_size]) + variable.shape)
            self.params[name] = param
    def sample(self):
        for name,param in self.params.items():
            loc,scale,size = param['loc'],torch.exp(param['omega']),param['size']
            noise = torch.distributions.Normal(0,1).sample(size)
            self.target_f.__globals__[name] = noise * scale + loc
    def parameters(self):
        rl = []
        for name,param in self.params.items():
            rl.extend([param['loc'], param['omega']])
        return rl
    def log_prob(self):
        logq = 0.0
        for name,param in self.params.items():
            q_dis = torch.distributions.Normal(param['loc'],torch.exp(param['omega']))
            logq += q_dis.log_prob(self.target_f.__globals__[name])
        return logq
            


@contextlib.contextmanager
def transform_meanfield(target_f):
    '''
    Providing a function that replace all variables labeled Parameter 
    in target_f.__global__ with sample of variational distribution.
    
    When exit, the state will be reset and variational parameter 
    loc value will not be rewrite to original value. The variational parameter,
    should be collected in block.
    
    '''
    cache = collect_parameters(target_f)
    
    q_dis = VariationalMeanFieldDistribution(target_f)
    
    yield q_dis
    
    target_f.__globals__.update(cache)
    
def vb_meanfield(target_f, n_epoch = 100, lr=0.01, q_size = 10):
    
    with transform_meanfield(target_f) as q_dis: 
    
        optimizer = torch.optim.SGD(q_dis.parameters(), lr=lr)
        for i in range(n_epoch):
            q_dis.sample()
            logp = target_f()
            logq = q_dis.log_prob()
            target = logp.sum(-1) + logq.sum(-1)
            loss = -target
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    return q_dis



current_target_f = None

from functools import wraps

def cache_target_f(f):
    @wraps(f)
    def _f(target_f,*args,**kwargs):
        global current_target_f
        current_target_f = target_f
        return f(target_f,*args,**kwargs)
    return _f

@cache_target_f
def vb(target_f,method='meanfield', *args, **kwargs):
    return globals()[f'vb_{method}'](target_f, *args, **kwargs)

@cache_target_f
def sampling(target_f,method='hmc',*args,**kwargs):
    raise NotImplemented
    
@cache_target_f
def optimizing(target_f, lr=0.01, n_epoch = 1000):
    parameters_dict = collect_parameters(target_f)
    
    optimizer = torch.optim.SGD(parameters_dict.values(), lr=lr)
    
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        target = target_f()
        loss = -target
        loss.backward()
        optimizer.step()

def reset(target_f = None):
    if target_f is None:
        if current_target_f is None:
            return
        target_f = current_target_f
    for name,variable in collect_parameters(target_f).items():
        del target_f.__globals__[name]
    
    