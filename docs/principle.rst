.. principle 

Principle
===========

The goal of bayes-torch is to write latent-variable model as Stan-like style
in pytorch. I personally can't bear the slow compiling time, inflexible in 
writing and clumsy support for ADVI (though it firstly present it) of Stan.
In other hand, 
pymc, edward, pyro don't interest me for some reason, so I write the framework
for illustration and light model writing. There're a lot of VAE example in 
those so called probability programming "language". But in my opnion, the
"serious" model such as VAE to be written in original framework such as pytorch
may be more better in a "high-level" wrapper that let you can't find the 
reality feeling behind those abstraction.

bayes-torch is easy to understand, you are required to formulate the joint 
probability function by a directive computing process consisting of 
some function or callable object. As you pass the function to algorithmes 
provied by bayes-torch, bayes-torch may run optimizing in-place(optimzing,
MAP), replace original variable with new one due to HMC random dynamic(samping),
or "transform" original parameters with some "variational distribution"
including new variational parameters (vb). 