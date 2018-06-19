# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 08:28:07 2018

@author: yiyuezho
"""

import numpy as np
from scipy import stats
import torch


class GridSampler2d:
    def __init__(self,prob,x,y):
        # prob: n_sample * 2 -> n_sample
        self.prob = prob
        self.x,self.y = x,y
        self.x_step = x[1] - x[0]
        self.y_step = y[1] - y[0]
        xx, yy = np.meshgrid(x,y)
        self.xx,self.yy = xx,yy
        xy = np.c_[xx.ravel(),yy.ravel()]
        self.xy = xy
        _prob = prob(xy)
        self.grid_prob = _prob / _prob.sum()
    def sample(self, size):
        base = self.xy[np.random.choice(len(self.xy), size, p=self.grid_prob)]
        x_noise = stats.uniform(-self.x_step/2, self.x_step/2).rvs(size)
        y_noise = stats.uniform(-self.y_step/2, self.x_step/2).rvs(size)
        base[:,0] += x_noise
        base[:,1] += y_noise
        return base
    
def sigmoid(x):
    # torch have this function in Variable
    return 1/(1+np.exp(-x))

def numpy_soft_cut_ge(x,threshold,tense=1.0):
    return sigmoid(tense*(x-threshold))

def numpy_soft_cut_le(x,threshold,tense=1.0):
    return sigmoid(tense*(-x+threshold))

def torch_logsumexp(X,dim=0):
    # numpy have this function
    return torch.log(torch.exp(X).sum(dim=dim))

def torch_tile(X,repeats):
    # torch.repeat ~ numpy.tile
    return X.repeat(*repeats)


def torch_soft_cut_ge(x,threshold,tense=1.0):
    return torch.sigmoid(tense*(x-threshold))

def torch_soft_cut_le(x,threshold,tense=1.0):
    return torch.sigmoid(tense*(-x+threshold))

soft_cut_le = torch_soft_cut_le
soft_cut_ge = torch_soft_cut_ge
logsumexp = torch_logsumexp

def torch_transpose(x):
    # torch don't have "default"(2d matrix) transpose
    return x.transpose(0,1)


def torch_cdist(A,B):
    '''
    A: m * n
    B: t * n
    ->
    m * t
    
    or 
    
    A: q * m * n
    B: q * t * n
    ->
    q * m * t
    '''
    # scipy.spatial.distance.cdist(A, B).min(axis=1) # numpy version
    '''
    if len(A.shape) == len(B.shape):
    #d = A.repeat(1,B.size()[0]).resize(A.size()[0]*B.size()[0],2).resize(A.size()[0],B.size()[0],2) - B
        d = A.repeat(1,B.shape[0]).reshape(A.shape[0]*B.shape[0],2).reshape(A.shape[0],B.shape[0],2) - B
    else:
        _B = B.unsqueeze(2).repeat(1,1,A.shape[0],1)
        d = A - _B
    '''
    length = len(A.shape) + 1
    A_expand = [-1] * length 
    A_expand[-2] = B.shape[-2]
    B_expand = [-1] * length
    B_expand[-3] = A.shape[-2]
    d = A.unsqueeze(-2).expand(*A_expand) - B.unsqueeze(-3).expand(*B_expand)
    return torch.sqrt((d**2).sum(dim=-1)) 


cdist = torch_cdist

def cat(seq,*args,dim=0,**kwargs):
    '''
    broadcast version of torch.cat
    '''
    min_len = max_len = len(seq[0].shape)
    max_idx = 0
    for i,t in enumerate(seq):
        if len(t.shape) > max_len:
            max_idx = i
            max_len = len(t.shape)
        elif len(t.shape) < min_len:
            min_len = len(t.shape)
    
    if min_len == max_len:
        return torch.cat(seq,*args,dim=dim,**kwargs)
    
    shape = seq[max_idx].shape
    for i in range(len(seq)):
        res = len(shape) - len(t.shape)
        for j in range(res):
            seq[i] = seq[i].unsqueeze(0)
        seq[i] = seq[i].repeat(shape[:res] + (1,)*len(seq[i].shape))
    
    return torch.cat(seq,*args,dim=dim,**kwargs)