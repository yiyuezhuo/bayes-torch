data {
  int<lower=0> N;
  int<lower=0> Nb;
  int<lower=1,upper=Nb> rail[N];
  vector[N] y;
}
parameters {
  vector[Nb] b;
  real mu;
  real<lower=0> sigmarail;
  real<lower=0> sigmaepsilon;
}
transformed parameters {
  vector[N] yhat;

  for (i in 1:N)
    yhat[i] = mu + b[rail[i]];

}
model {
  b ~ normal(0, sigmarail);

  y ~ normal(yhat, sigmaepsilon);
}