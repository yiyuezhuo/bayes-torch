# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 19:56:38 2018

@author: yiyuezhuo
"""

from bayestorch import Parameter,Data,optimizing,vb,sampling
from bayestorch.utils import torch_logsumexp,soft_cut_ge,soft_cut_le
from torch.distributions import Normal

import matplotlib.pyplot as plt
import torch
from torch import tensor


# data
friend_point = tensor([[0.0,5],[0.9,3.5],[1.4,3.0],[1.8,2.5],[2.1,1.1],[2.5,0.9],[3.2,0.5]])
enemy_point = tensor([[1.0,5.3],[1.3,4.4],[2.0,3.3],[3.0,2.5],[4.0,2.0]])

battle_point = tensor([
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

X = torch.cat([friend_point, enemy_point])
Y = torch.ones(X.shape[0], dtype=torch.int32)
Y[:friend_point.shape[0]] = 0


_logPC = torch.log(tensor([
        torch.sum(Y==0).float()/len(Y),
        torch.sum(Y==1).float()/len(Y)]))


# display data
def display_data():
    plt.scatter(friend_point[:,0],friend_point[:,1],color='blue',label='friend')
    plt.scatter(enemy_point[:,0],enemy_point[:,1],color='red',label='enemy')
    plt.scatter(battle_point[:,0],battle_point[:,1],color='green',label='battle')
    plt.legend()
display_data()
plt.show()

# helper function

def cdist(A,B):
    '''
    A: m * n
    B: t * n
    ->
    m * t
    
    or 
    
    A: m * n * q
    B: t * n * q
    ->
    m * t * q
    '''
    _A = A.unsqueeze(1).expand((A.shape[0],) + B.shape)
    d = _A - B
    return torch.sqrt(d**2).sum(dim=2)
    
def norm_naive_bayes_predict(X,mu,sd,logPC):
    # X: n_sample * feature * (q_size)
    # mu/sd: n_class * feature * (q_size)
    # logPC : n_class * (q_size)
    
    # _X: n_sample * n_class * feature * (q_size)
    _X = X.unsqueeze(1).expand((X.shape[0],) + mu.shape)
    # cp: n_sample * n_class * (q_size)
    cp = Normal(mu,sd).log_prob(_X).sum(2) + logPC
    log_predict_prob = cp - torch_logsumexp(cp,dim=1).unsqueeze(1).expand_as(cp)
    return log_predict_prob
    
    


# model
friend = Data(friend_point)
battle = Data(battle_point)
enemy = Parameter(enemy_point) # set real value as init value, though maybe a randomed init is more proper

logPC = Data(_logPC)

conflict_threshold = 0.2
distance_threshold = 1.0
tense = 10.0
alpha = 5.0
prior_threshold = 5.0
prior_tense = 5.0

def target():
    friend_enemy = torch.cat((friend, enemy),0)
    
    distance = cdist(battle, friend_enemy).min(dim=1)[0]
    
    
    mu = torch.stack([friend.mean(0),enemy.mean(0)],0)
    sd = torch.stack([friend.std(0),enemy.std(0)],0)
    
    conflict = torch.exp(norm_naive_bayes_predict(battle, mu, sd, logPC)).prod(1)
    p = soft_cut_ge(conflict,conflict_threshold, tense = tense) *  \
        soft_cut_le(distance, distance_threshold, tense = tense)
    
    target= torch.log(p).sum(0)
    return target

def target2():
    target1 = target()
    # location prior
    target2 = target1 + enemy.sum(0).sum(0)
    return target2


# display the optimizing result helper function

def show_change(show=True):
    display_data()
    for i in range(enemy_point.shape[0]):
        #s = 0.1
        plt.arrow(enemy_point[i][0], enemy_point[i][1], 
                  enemy.data[i][0] - enemy_point[i][0], 
                  enemy.data[i][1] - enemy_point[i][1],head_width=0.1)
    plt.legend()
    if show:
        plt.show()

def reset():
    global enemy
    enemy = Parameter(enemy_point)
    
def show_ellipse(mu,sd):
    from matplotlib.patches import Ellipse
    
    ax = plt.subplot(111)
    show_change(show=False)
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
    loc = res.params['enemy']['loc']
    omega = res.params['enemy']['omega']

    mu = loc
    sd = torch.exp(omega)
    show_ellipse(mu,sd)

    
def _show_vb(vb_res):
    from matplotlib.patches import Ellipse
    
    res = vb_res
    
    ax = plt.subplot(111)
    show_change(show=False)
    for i in range(enemy_point.shape[0]):
        loc = res.params['enemy']['loc']
        omega = res.params['enemy']['omega']
        mu_x,mu_y = loc[:,0], loc[:,1]
        omega_x,omega_y = omega[:,0],omega[:,1]
        e=Ellipse((mu_x,mu_y), torch.exp(omega_x), torch.exp(omega_y), 0)
        e.set_clip_box(ax.bbox)
        e.set_alpha(0.1)
        ax.add_artist(e)
    
    plt.show()
    
def show_sampling(trace):
    _trace = torch.stack([t['enemy'] for t in trace])
    mu = _trace.mean(0)
    sd = _trace.std(0)
    global enemy
    enemy = Parameter(mu)
    show_ellipse(mu,sd)

# experiment
reset()
optimizing(target)
show_change(target)


reset()
optimizing(target2)
show_change(target)


reset()
res = vb(target,q_size=3)
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
