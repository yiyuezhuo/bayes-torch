# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 08:26:47 2018

@author: yiyuezhuo
"""

import torch
from torch.autograd import Variable
import numpy as np
import scipy

'''
core主要封装参数定义之类的东西，把它们记录到当前model里。不然动态图定义一时爽，
更新起来火葬场。

程序主体应该是这样，我们在顶层用helper function Paramter声明一些参数，
这些参数同时被记录到匿名的module.current_model的参数列表里。
然后这些参数的计算过程写到一个target函数里，这个函数应该返回一个target Variable
这个表示计算概率函数和生成动态图的过程，
最后我们调用module.api去隐式调用model的方法进行相关操作。一般蕴含着不断重建
动态图的过程。
'''



class Model:
    def __init__(self):
        self.parameters = []
        self.n_parameters = 0
        self.size_parameters = []
    def add_parameter(self, variable):
        self.parameters.append(variable)
        self.n_parameters += np.prod(variable.size())
        self.size_parameters.append(variable.size())
    def set_parameter_meanfield(self, mu, omega):
        n_parameters = self.n_parameters
        param_samples_eta = np.random.normal(size=n_parameters)
        param_samples = param_samples_eta*np.exp(omega) + mu
        self.set_parameter(param_samples)
        return param_samples_eta
    def set_parameter(self, values, is_float = True):
        # values is a flatten array
        start = 0
        for param in self.parameters:
            param_size = np.prod(param.size())
            section = values[start:start+param_size].reshape(param.size())
            tensor = torch.from_numpy(section)
            if is_float:
                tensor = tensor.float()
            param.data = tensor
            start += param_size
    def collect_parameter_grad(self):
        grad = np.empty(self.n_parameters)
        start = 0
        for param in self.parameters:
            param_size = np.prod(param.size())
            grad[start:start+param_size] = param.grad.data.numpy()
            start += param_size
        return grad

    def grad_q_meanfield(self, mu, omega, target_f, q_size=10, lr = 0.01):
        n_parameters = self.n_parameters
        
        mu_grad = np.zeros(n_parameters)
        omega_grad = np.zeros(n_parameters)
        
        
        optimizer = torch.optim.SGD(self.parameters, lr=lr)
        # optimizer只用于清空梯度，不用它来更新
        
        for i in range(q_size):
            
            param_samples_eta = self.set_parameter_meanfield(mu, omega)
            
            optimizer.zero_grad()
            target = target_f()
            target.backward()
            
            param_grad = self.collect_parameter_grad()
            mu_grad += param_grad
            omega_grad += param_grad * param_samples_eta
        
        mu_grad /= q_size
        omega_grad /= q_size
        omega_grad *= np.exp(omega)
        omega_grad += 1.0
        
        return mu_grad,omega_grad
            
    def vb_meanfield(self, target_f, n_epoch = 100, lr=0.01, q_size = 10):
        n_parameters = self.n_parameters
        mu = np.zeros(n_parameters)
        omega = np.zeros(n_parameters)
        
        #optimizer = torch.optim.SGD(self.parameters, lr=lr)
        
        for i in range(n_epoch):
            mu_grad,omega_grad = self.grad_q_meanfield(mu, omega, target_f, q_size = q_size)
            mu += lr * mu_grad
            omega += lr * omega_grad
        
        return mu,omega
        
        
    def vb_fullrank(self, target_f):
        raise NotImplementedError
    def vb(self, target_f, method = 'meanfield', **kwargs):
        if method == 'meanfield':
            return self.vb_meanfield(target_f, **kwargs)
        if method == 'fullrank':
            return self.vb_fullrank(target_f, **kwargs)
        raise NotImplementedError
    def sampling(self):
        trace = []
        raise NotImplementedError
    def optimizing(self, target_f, lr=0.01, n_epoch = 1000):
        '''
        optimizing的结果作为状态仍然存在原动态图参数里\
        target_f当然不应该是带变量的，因为这个类就是为了消除这些操作的
        '''
        optimizer = torch.optim.SGD(self.parameters, lr=lr)
        
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            target = target_f()
            loss = -target
            loss.backward()
            optimizer.step()
        
            
def easy_variable(data, is_float=True, requires_grad=False):
    if not hasattr(data, '__len__'):# is scaler without
        data = [data]
    if not isinstance(data, torch.Tensor):
        array = torch.from_numpy(np.array(data))
        if is_float:
            array = array.float()
    return Variable(array, requires_grad=requires_grad)
    
def Parameter(data, is_float=True, model = None):
    variable = easy_variable(data, 
                             is_float = is_float,
                             requires_grad = True)
    if model is None:
        model = current_model
    model.add_parameter(variable)
    return variable

def Data(data, is_float=True, model = None):
    variable = easy_variable(data, 
                             is_float = is_float,
                             requires_grad = False)
    return variable

'''
def optimizing(*args, **kwargs):
    return current_model.optimizing(*args, **kwargs)

def vb(*args, **kwargs):
    return current_model.vb(*args, **kwargs)

def sampling(*args, **kwargs):
    return current_model.vb(*args, **kwargs)
'''
current_model = Model()
optimizing = current_model.optimizing
vb = current_model.vb
sampling = current_model.sampling