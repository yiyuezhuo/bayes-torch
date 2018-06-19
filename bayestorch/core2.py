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
    
def collect_variable_labeled_class(target_f,classes):
    parameters_dict = {}
    for key,value in target_f.__globals__.items():
        if isinstance(value, classes):
            parameters_dict[key] = value
    return parameters_dict

    
def collect_parameters(target_f):
    return collect_variable_labeled_class(target_f, Parameter)

def collect_parameter_datas(target_f):
    return collect_variable_labeled_class(target_f, (Parameter,Data))


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
            q_log_prob = q_dis.log_prob(self.target_f.__globals__[name])
            for i in range(len(self.params[name]['size'])-1):
                q_log_prob = q_log_prob.sum(-1)
            logq += q_log_prob
            #s_dis = torch.distributions.Normal(0.,1.)
            #z = (self.target_f.__globals__[name] - param['loc'])/torch.exp(param['omega'])
            #logq += s_dis.log_prob(z)
        return logq
            


@contextlib.contextmanager
def transform_meanfield(target_f, q_size = 1):
    '''
    Add q size dimention to all variables labeled Parameter and Data
    in target_f.__globals__ with sample of variational distribution.
    
    When exit, the state will be reset and variational parameter 
    loc value rewrite to original value. The other variational parameters,
    can be collected in block.
    
    '''
    cache = collect_parameter_datas(target_f)
    
    for name,variable in cache.items():
        if isinstance(variable,Data):
            extended = variable.unsqueeze(0).repeat(q_size,*(1,)*len(variable.shape))
            target_f.__globals__[name] = extended
    
    q_dis = VariationalMeanFieldDistribution(target_f, q_size = q_size)
    
    yield q_dis
    
    target_f.__globals__.update(cache)
    
def vb_meanfield(target_f, n_epoch = 100, lr=0.01, q_size = 1):
    
    with transform_meanfield(target_f, q_size = q_size) as q_dis: 
    
        optimizer = torch.optim.SGD(q_dis.parameters(), lr=lr)
        for i in range(n_epoch):
            q_dis.sample()
            logp = target_f()
            logq = q_dis.log_prob()
            target = logp.mean(0) - logq.mean(0) # reduce q_size dimention
            loss = -target
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    return q_dis # Though calling sample or other method may cause misleading.
                 # Maybe returning only trimed params is better choice?
                 
def vb_fullrank(target_f, n_epoch = 100, lr=0.01, q_size = 1):
    raise NotImplementedError
    
def sampling_hmc(target_f, leap_frog_step = 0.01, leap_frog_length = 20, 
                 trace_length = 100):
    pd = collect_parameters(target_f)
    trace = []
    potential_trace = []
    
    def grad():
        # zero_grad manually
        for name in pd:
            target_f.__globals__[name].grad = None
        target = target_f()
        target.backward()
        return target
    
    def get_potential(target, r):
        #target = target_f()
        potential = target.detach()
        for name in pd:
            #potential -= r[name] @ r[name]
            potential -= 0.5 * torch.sum(r[name] * r[name]) 
            # It's interesting to see torch.dot or @ does't support size 0 tensor
        return potential
    

    
    for trace_i in range(trace_length):
        r = {}
        for name in pd:
            r[name]  = torch.randn_like(target_f.__globals__[name])
            
        trace.append({name: target_f.__globals__[name].detach() for name in pd})
        potential_trace.append(get_potential(target_f(), r))

        # leap-frog procedure
        for frog_i in range(leap_frog_length):
            grad()
            for name in pd:
                r[name] += (leap_frog_step/2) * target_f.__globals__[name].grad
                theta = target_f.__globals__[name].detach() + leap_frog_step * r[name]
                #print(r[name],theta)
                target_f.__globals__[name] = Parameter(theta)
            target = grad()
            for name in pd:
                r[name] += (leap_frog_step/2) * target_f.__globals__[name].grad
        
        potential = get_potential(target, r)
        log_accept = potential - potential_trace[-1]
        #print(log_accept,potential,potential_trace[-1])
        #print(log_accept)
        if log_accept > 0 or (torch.rand(1) < torch.exp(log_accept)).item():
            #print('accept',)
            trace[-1] = {name: target_f.__globals__[name].detach() for name in pd}
            potential_trace[-1] = potential
        
    return trace
                
                



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
    return globals()[f'sampling_{method}'](target_f,*args,**kwargs)
    
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
    
    