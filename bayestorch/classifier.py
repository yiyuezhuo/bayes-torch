# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 08:44:50 2018

@author: yiyuezhuo
"""

import numpy as np
import torch
#from .distributions import norm_log_prob,torch_norm_log_prob
from torch.distributions import Normal
from scipy.special import logsumexp
import scipy.stats as stats
from .utils import torch_logsumexp,torch_transpose,torch_tile


def numpy_norm_naive_bayes_predict(X,mu,sd,logPC):
    # X: sample_size * features
    # mu: class_size * features
    # sd: class_size * featrues
    # log_PC class_size
    n_class = logPC.shape[0]
    n_feature = X.shape[1]
    n_sample = X.shape[0]
    _X = np.tile(X,[1,n_class]).reshape([n_sample,n_class,n_feature])
    #cp = norm_log_prob(_X,mu,sd).sum(axis=2) + logPC  
    cp = stats.norm(mu,sd).logpdf(_X).sum(axis=-1) + logPC
    log_predict_prob = (cp.T - logsumexp(cp,axis=1)).T
    return log_predict_prob

def torch_norm_naive_bayes_predict(X,mu,sd,logPC):
    # X: (q_size) * sample_size * features
    # mu: (q_size) * class_size * features or q_size * class_size * features
    # sd: (q_size) * class_size * featrues or q_size * class_size * features
    # log_PC (q_size) * class_size
    # So called `q_size` is number of particles for some approxminate expectation. 
    n_class = logPC.shape[-1]
    #n_feature = X.shape[-1]
    #n_sample = X.shape[-2]
    # q_size = X.shape[-3]
    
    X_expand = [-1] * (len(X.shape) + 1)
    X_expand[-2] = n_class
    _X = X.unsqueeze(-2).expand(X_expand)
    #_X = torch_tile(X,[1,n_class]).reshape(n_sample,n_class,n_feature)
    #cp = torch_norm_log_prob(_X, mu, sd).sum(dim=2) + logPC
    
    # X: (q_size) * sample_size * features
    # _X: (q_size) * sample_size * n_class * features
    # mu/sd: (q_size) * n_class * features
    # _mu/_sd (q_size) * sample_size * n_class * feature
    _mu = mu.unsqueeze(-3).expand_as(_X)
    _sd = sd.unsqueeze(-3).expand_as(_X)
    cp = Normal(_mu, _sd).log_prob(_X).sum(-1) + logPC
    # cp: (q_size) * sample_size * n_class
    log_predict_prob = (cp - torch_logsumexp(cp,dim=-1).unsqueeze(-1).expand_as(cp))
    return log_predict_prob

norm_naive_bayes_predict = torch_norm_naive_bayes_predict