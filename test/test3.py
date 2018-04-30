# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 19:56:38 2018

@author: yiyuezhuo
"""

from bayestorch.core import Parameter,Data,Variable,optimizing,vb,sampling
from bayestorch.distributions import norm_log_prob
from bayestorch.classifier import norm_naive_bayes_predict
from bayestorch.utils import GridSampler2d,cdist,soft_cut_ge,soft_cut_le

import numpy as np
import matplotlib.pyplot as plt
import torch

import bayestorch.core as core
model = core.current_model

# data
friend_point = np.array([[0.0,5],[0.9,3.5],[1.4,3.0],[1.8,2.5],[2.1,1.1],[2.5,0.9],[3.2,0.5]])
enemy_point = np.array([[1.0,5.3],[1.3,4.4],[2.0,3.3],[3.0,2.5],[4.0,2.0]])

battle_point = np.array([
       [ 1.19218743,  4.71590436],
       [ 1.38600962,  5.5512839 ],
       [ 3.05023886,  2.91432179],
       [ 1.07274493,  4.92151802],
       [ 2.35362575,  3.8626928 ],
       [ 1.23971936,  4.04078796],
       [ 1.78757744,  2.81426836],
       [ 1.48919788,  3.05258488],
       [ 4.56533233,  0.55658402],
       [ 4.24201312,  3.5746528 ]])

X = np.vstack([friend_point, enemy_point])
Y = np.ones(X.shape[0],dtype=int)
Y[:friend_point.shape[0]] = 0


_logPC = np.log(np.array([np.sum(Y==0)/len(Y),np.sum(Y==1)/len(Y)]))


# display data
def display_data():
    plt.scatter(friend_point[:,0],friend_point[:,1],color='blue',label='friend')
    plt.scatter(enemy_point[:,0],enemy_point[:,1],color='red',label='enemy')
    plt.scatter(battle_point[:,0],battle_point[:,1],color='green',label='battle')
    plt.legend()
display_data()
plt.show()

# model
friend = Data(friend_point)
battle = Data(battle_point)
enemy = Parameter(enemy_point) # set real value as init value, though maybe a randomed init is more proper

conflict_threshold = 0.2
distance_threshold = 1.0
tense = 10.0
alpha = 5.0
prior_threshold = 5.0
prior_tense = 5.0

def target():
    friend_enemy = torch.cat((friend, enemy),0)
    distance = cdist(battle, friend_enemy).min(dim=1)[0]
    logPC = Data(_logPC)

    mu = Variable(torch.zeros(2,2)) #目前外层还有个同名的numpy.array mu,sd变量不要搞混了
    sd = Variable(torch.zeros(2,2))
    
    mu[0,:] = friend.mean(dim=0)
    mu[1,:] = enemy.mean(dim=0)
    sd[0,:] = friend.std(dim=0)
    sd[1,:] = enemy.std(dim=0)
    
    conflict = torch.exp(norm_naive_bayes_predict(battle, mu, sd, logPC)).prod(dim=1)
    p = soft_cut_ge(conflict,conflict_threshold, tense = tense) * soft_cut_le(distance, distance_threshold, tense = tense)
    
    target= torch.sum(torch.log(p))
    return target

def target2():
    target1 = target()
    # location prior
    target2 = target1 + torch.sum(enemy.sum(dim=1))
    return target2


# display the optimizing result helper function

def show_change(show=True):
    display_data()
    for i in range(enemy_point.shape[0]):
        #s = 0.1
        plt.arrow(enemy_point[i][0], enemy_point[i][1], enemy.data[i][0] - enemy_point[i][0], enemy.data[i][1] - enemy_point[i][1],head_width=0.1)
    plt.legend()
    if show:
        plt.show()
    
def reset():
    enemy.data = torch.from_numpy(enemy_point).float()
    
def show_ellipse(mu,sd):
    from matplotlib.patches import Ellipse
    
    ax = plt.subplot(111)
    show_change(show=False)
    #res_reshaped = [r.reshape(enemy_point.shape) for r in res]
    for i in range(enemy_point.shape[0]):
        mu_x,mu_y = mu[i]
        sd_x,sd_y = sd[i]
        e=Ellipse((mu_x,mu_y), sd_x, sd_y, 0)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.1)
        ax.add_artist(e)
        
    plt.show()
    
def show_vb(vb_res):
    res = vb_res
    model.set_parameter(res[0])
    res_reshaped = [r.reshape(enemy_point.shape) for r in res]
    mu = res_reshaped[0]
    sd = np.exp(res_reshaped[1])
    show_ellipse(mu,sd)

    
def _show_vb(vb_res):
    from matplotlib.patches import Ellipse
    
    #res = vb(target,**kwargs)
    res = vb_res
    model.set_parameter(res[0])
    
    ax = plt.subplot(111)
    show_change(show=False)
    res_reshaped = [r.reshape(enemy_point.shape) for r in res]
    for i in range(enemy_point.shape[0]):
        mu_x,mu_y = res_reshaped[0][i]
        omega_x,omega_y = res_reshaped[1][i]
        e=Ellipse((mu_x,mu_y), np.exp(omega_x), np.exp(omega_y), 0)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.1)
        ax.add_artist(e)
    
    plt.show()
    
def show_sampling(trace):
    trace = np.array(trace).reshape((len(trace),)+enemy_point.shape)
    mu = trace.mean(axis=0)
    sd = trace.std(axis=0)
    model.set_parameter(mu.ravel())
    show_ellipse(mu,sd)

# experiment
reset()
optimizing(target)
show_change(target)

reset()
optimizing(target2)
show_change(target)
    
reset()
res = vb(target)
show_vb(res)

reset()
res = vb(target2)
show_vb(res)


reset()
trace = sampling(target)
show_sampling(trace)

reset()
trace = sampling(target2)
show_sampling(trace)