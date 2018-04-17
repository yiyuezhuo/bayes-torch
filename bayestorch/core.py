# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 08:26:47 2018

@author: yiyuezhuo
"""

import torch
from torch.autograd import Variable
import numpy as np
#import scipy

'''


Core mainly wrap the parameter defining process, those parameter will
be recorded into "current" model automatically. Otherwise, you can
define a dynamic as intuititive way and feels awkward when you try to manage
your parameter spreading everywhere and write a update function.

Our program should be consist of
1. Define some parameter using helper function defined in this module, then
   those paramters will be appended into a list of anonymous model object.
2. The computation process relate those parameter should be written into a so
   called "target" function(stan style without fake sampling statement).
   The function may return a target torch Variable.
3. Finally, we call module.optimzing/vb/sampiling to run our inference.
   Every calling will build a new dynamic graph.
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
        # values is a flatten numpy array
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
            grad[start:start+param_size] = param.grad.data.numpy().copy().ravel()
            start += param_size
        return grad
    def collect_parameter(self):
        res = np.empty(self.n_parameters)
        start = 0
        for param in self.parameters:
            param_size = np.prod(param.size())
            res[start:start+param_size] = param.data.numpy().copy().ravel()
            start += param_size
        return res
    def grad_q_meanfield(self, target_f, mu, omega, q_size=10, lr = 0.01):
        n_parameters = self.n_parameters
        
        mu_grad = np.zeros(n_parameters)
        omega_grad = np.zeros(n_parameters)
        
        optimizer = torch.optim.SGD(self.parameters, lr=lr)
        # optimizer only serve to zero_grad.
        
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
            
    def vb_meanfield(self, target_f, mu=None,omega=None, zero_init=False, n_epoch = 100, lr=0.01, q_size = 10):
        n_parameters = self.n_parameters
        
        if mu is None:
            if zero_init:
                mu = np.zeros(n_parameters)
            else:
                mu = np.array(self.collect_parameter())
        if omega is None:
            omega = np.zeros(n_parameters) # sigma=exp(omega) = 1
        
        #optimizer = torch.optim.SGD(self.parameters, lr=lr)
        
        for i in range(n_epoch):
            mu_grad,omega_grad = self.grad_q_meanfield(target_f, mu, omega, q_size = q_size)
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
    def sampling_hmc(self, target_f, epsilon=0.01, L=20, M=100):
        
        def grad(theta):
            self.set_parameter(theta)
            return self.grad(target_f)
        def likelihood(theta):
            self.set_parameter(theta)
            return target_f().data.numpy()
        
        sample =[]
        sample.append(self.collect_parameter())
        accept_list = []
        
        for m in range(M):
            r0 = np.random.normal(size=self.n_parameters)
            r = r0
            theta0 = sample[-1]
            theta = theta0.copy()
            
            for i in range(L):
                r = r + 0.5 * epsilon * grad(theta)
                theta = theta + epsilon * r
                self.set_parameter(theta)
                r = r + 0.5 * epsilon * grad(theta)
            odd_up = np.exp(likelihood(theta)-0.5*np.dot(r,r))
            odd_bottom = np.exp(likelihood(theta0)-0.5*np.dot(r0,r0))
            alpha = min(1,odd_up/odd_bottom)
            accept_list.append(alpha)
            
            if np.random.random() < alpha:
                sample.append(theta)
            else:
                sample.append(theta0)
        return sample
        

    def sampling(self, target_f, method= 'hmc', **kwargs):
        #trace = []
        #raise NotImplementedError
        if method == 'hmc':
            return self.sampling_hmc(target_f, **kwargs)
        if method == 'nuts':
            return self.sampling_nuts(target_f, **kwargs)
        raise NotImplementedError
    def optimizing(self, target_f, lr=0.01, n_epoch = 1000):
        '''
        After optimizing run, the result is stored in original torch variable.
        target_f don't have any parameter, since the class serve to delete the trouble
        '''
        optimizer = torch.optim.SGD(self.parameters, lr=lr)
        
        for epoch in range(n_epoch):
            optimizer.zero_grad()
            target = target_f()
            loss = -target
            loss.backward()
            optimizer.step()
    def grad(self, target_f):
        fake_lr = 1.0
        optimizer = torch.optim.SGD(self.parameters, lr=fake_lr)
        optimizer.zero_grad()
        target = target_f()
        target.backward()
        return self.collect_parameter_grad()
            
def easy_variable(data, is_float=True, requires_grad=False):
    if not hasattr(data, '__len__'):# is scaler
        data = [data]
    if not isinstance(data, torch.Tensor):
        array = torch.from_numpy(np.array(data)) # np.array copyed data
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