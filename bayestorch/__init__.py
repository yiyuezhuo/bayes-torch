# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 08:04:11 2018

@author: yiyuezhuo
"""

'''
Greeting our great elder toad!
         /          \   /          \
        /            \ /            \
        |        /\  | |   /\       |
       /\            / \            /\    
      /  \ _________/   \__________/  \         
     /                                 \
    (    O                         O    )
     \    \_                     _/    /
       \_   ---------------------   _/
         ----___________________---- __-->
        /   / =  =  |\ /| =  =  =\       >      
      /    /  =  =  |/ \| =  =  = \ __-- >
    /     /=  =  =  =  =  =  =  =  \            

Source: http://chris.com/ascii/index.php?art=animals/frogs
'''

from .core import Parameter,Data,Variable
from .core import optimizing,vb,sampling
from .core import reset,index
import bayestorch.core as core
import bayestorch.distributions as distributions
import bayestorch.classifier as classifier
import bayestorch.utils as utils

