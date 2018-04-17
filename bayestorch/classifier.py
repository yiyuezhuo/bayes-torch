# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 08:44:50 2018

@author: yiyuezhuo
"""

import numpy as np
import torch
from .distributions import norm_log_prob,torch_norm_log_prob
from scipy.special import logsumexp
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
    cp = norm_log_prob(_X,mu,sd).sum(axis=2) + logPC  
    log_predict_prob = (cp.T - logsumexp(cp,axis=1)).T
    return log_predict_prob

def torch_norm_naive_bayes_predict(X,mu,sd,logPC):
    # X: sample_size * features
    # mu: class_size * features
    # sd: class_size * featrues
    # log_PC class_size
    n_class = logPC.size()[0]
    n_feature = X.size()[1]
    n_sample = X.size()[0]
    _X = torch_tile(X,[1,n_class]).resize(n_sample,n_class,n_feature)
    cp = torch_norm_log_prob(_X, mu, sd).sum(dim=2) + logPC  
    log_predict_prob = torch_transpose((torch_transpose(cp) - torch_logsumexp(cp,dim=1)))
    return log_predict_prob

norm_naive_bayes_predict = torch_norm_naive_bayes_predict