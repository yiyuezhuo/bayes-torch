# A bayes inference library based on pytorch

Though there're a lot of bayes inference modeling lib/language
such as stan,edward(tensorflow) and pymc(theano), the relation between
their computation ground and absract high API is awkward.

So the project is found to implment stan-like API on the flexible
autograd library, pytorch. Tt's a light-weight framework, you will
directly write joint likelihood function to run inference instead of
fake sampling statment in stan, pymc or ugly style in Edward.

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

```
mu = Parameter(0.0) # optimizing/vb/sampling init value
sigma = Data(1.0)
X = Data(_X)

def target():
    target = norm_log_prob(X,mu,sigma).sum()
    return target
```

Full script:

```
from core import Parameter,Data,optimizing,vb,sampling
from distributions import norm_log_prob

import numpy as np

_X = np.random.random(10)
print(_X.mean(),_X.std())

# torch-bayes model

mu = Parameter(0.0)
sigma = Data(1.0)
X = Data(_X)

def target():
    target = norm_log_prob(X,mu,sigma).sum()
    return target

optimizing(target)
print('optimizing: mu={}'.format(mu.data.numpy()))
res = vb(target)
print('vb: mu={} omega={} sigma={}'.format(res[0],res[1],np.exp(res[1])))

# stan model

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
res2 = sm.optimizing(data = dict(N = len(_X), y = _X))
print('optimizing(stan): mu={}'.format(res2['mu']))
res3 = sm.vb(data = dict(N = len(_X), y = _X))
res3a=np.array(res3['sampler_params'])
print('vb(stan): mu={} sigma={}'.format(res3a[:,0].mean(),res3a[:,0].std()))

```