# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 08:34:54 2018

@author: yiyuezhuo
"""

import numpy as np
import torch
from torch.autograd import Variable

def numpy_norm_log_prob(X,mu,sd,drop_constant = False):
    
    temp = -(X - mu)**2/(2*sd) 
    if not drop_constant:
        temp = temp - 0.5 * np.log(2.0 * np.pi * sd)
    return temp

def torch_norm_log_prob(X,mu,sd):    
    temp = -(X - mu)**2/(2*sd) 
    temp = temp - 0.5 * torch.log(2.0 * np.pi * sd)
    return temp

norm_log_prob = torch_norm_log_prob