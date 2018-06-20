
# Bayes-torch: A light weight bayes inference framework

Though there're a lot of bayes inference modeling lib/language
such as [stan](https://github.com/stan-dev/stan), 
[edward](https://github.com/blei-lab/edward) (tensorflow) 
[pymc](https://github.com/pymc-devs/pymc3) (theano), 
[pyro](https://github.com/uber/pyro) (pytorch) the relation between
their computation ground and absract high API is awkward.

So the project is found to implment stan-like API on the flexible
autograd library, pytorch. It's a light-weight framework, you will
directly write joint likelihood function to run inference instead of
fake sampling statment in stan, pymc or ugly style in Edward, 
weired namebinding in pyro.

## Example

We can implement following stan model as such:

```
data {
    int<lower=1> N;
    real y[N];
}
parameters {
    real mu;
}
model {
    y ~ normal(mu, 1);
}
```

torch-bayes model code:

```python
mu = Parameter(0.0) # optimizing/vb/sampling init value
sigma = Data(1.0)
X = Data(_X)

def target():
    target = Normal(mu,sigma).log_prob(X).sum(0)
    return target
```

Full code comparing two framework:

```python
from bayestorch import Parameter,Data,optimizing,vb,sampling,reset
import torch
from torch.distributions import Normal

_X = torch.arange(10)
print(_X.mean())

# torch-bayes model

mu = Parameter(0.0)
sigma = Data(1.0)
X = Data(_X)

def target():
    target = Normal(mu,sigma).log_prob(X).sum(0)
    return target

optimizing(target)
print(f'optimizing: mu={mu.data}')

res = vb(target)
q_mu = res.params['mu']
print(f'vb mu={q_mu["loc"]} omega={q_mu["omega"]} sigma={torch.exp(q_mu["omega"])}')

reset()

mu = Parameter(0.0)

res = vb(target, q_size = 10, n_epoch=200)
q_mu = res.params['mu']
print(f'vb mu={q_mu["loc"]} omega={q_mu["omega"]} sigma={torch.exp(q_mu["omega"])}')

trace = sampling(target,trace_length=300)
mu_trace = torch.tensor([t['mu'].item() for t in trace])
print('sampling: mu={} sigma={}'.format(torch.mean(mu_trace), torch.std(mu_trace)))


# stan model


import numpy as np
import pystan
stan_code = '''
data {
    int<lower=1> N;
    real y[N];
}
parameters {
    real mu;
}
model {
    y ~ normal(mu, 1);
}
'''
sm = pystan.StanModel(model_code = stan_code)

_X = _X.numpy()
res2 = sm.optimizing(data = dict(N = len(_X), y = _X))
print(f'optimizing(stan): mu={res2["mu"]}')
res3 = sm.vb(data = dict(N = len(_X), y = _X))
res3a=np.array(res3['sampler_params'])
print(f'vb(stan): mu={res3a[0,:].mean()} sigma={res3a[0,:].std()}')

```

Enemy location detecting example:

```python
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
```

<img src="images/example.png">

## Basic principle

bayes-torch(BT) use target_f.__globals__ to access and change variables labeled `Parameter` or `Data`. 
It implies some limit to way to introduce parameters, for example you can't define a list of `Parameter`
and expect BT can find it.

In `optimizing`, BT run standard SGD algorithm. 
In `sampling`, BT will frequently replace variable with another(same shape) using HMC. 
In `vb`, BT will map a variable to a normal variational distribution object, that contain variational
parameters mu and omega (omega = log(sigma)). The last dimention will be used by `vb` innerly.
So code such as `sum(-1)` or `sum()`(reduce to scaler) will raise error.

Anyway, the thin implementation(`core.py`) only consists of 300- lines, 
you can always check origin code to figure out what happen anyway.

## Reference

Automatic Differentiation Variational Inference

Kucukelbir, Alp and Tran, Dustin and Ranganath, Rajesh and Gelman, Andrew and Blei, David M

https://arxiv.org/abs/1603.00788

Fully automatic variational inference of differentiable probability models

Kucukelbir, Alp and Ranganath, Rajesh and Gelman, Andrew and Blei, David

http://andrewgelman.com/wp-content/uploads/2014/12/pp_workshop_nips2014.pdf

The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo.

Hoffman, Matthew D and Gelman, Andrew

https://arxiv.org/abs/1111.4246