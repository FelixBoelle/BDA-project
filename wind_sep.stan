//separate model for GPD
functions {
  real gpareto_lpdf(vector y, real ymin, real k, real sigma) {
    // generalised Pareto log pdf 
    int N = rows(y);
    real inv_k = inv(k);
    if (k<0 && max(y-ymin)/sigma > -inv_k)
      reject("k<0 and max(y-ymin)/sigma > -1/k; found k, sigma =", k, sigma)
    if (sigma<=0)
      reject("sigma<=0; found sigma =", sigma)
    if (fabs(k) > 1e-15)
      return -(1+inv_k)*sum(log1p((y-ymin) * (k/sigma))) -N*log(sigma);
    else
      return -sum(y-ymin)/sigma -N*log(sigma); // limit k->0
  }
  real gpareto_rng(real ymin, real k, real sigma) {
    // generalised Pareto rng
    if (sigma<=0)
      reject("sigma<=0; found sigma =", sigma)
    if (fabs(k) > 1e-15)
      return ymin + (uniform_rng(0,1)^-k -1) * sigma / k;
    else
      return ymin - sigma*log(uniform_rng(0,1)); // limit k->0
   }
  real gpareto_lccdf(vector y, real ymin, real k, real sigma) {
    // generalised Pareto log ccdf
    real inv_k = inv(k);
    if (k<0 && max(y-ymin)/sigma > -inv_k)
      reject("k<0 and max(y-ymin)/sigma > -1/k; found k, sigma =", k, sigma)
    if (sigma<=0)
      reject("sigma<=0; found sigma =", sigma)
    if (fabs(k) > 1e-15)
      return (-inv_k)*sum(log1p((y-ymin) * (k/sigma)));
    else
      return -sum(y-ymin)/sigma; // limit k->0
  }
}

data {
  int<lower=0> N; //Total number of data points
  vector[N] y; //The combined data points
  int<lower=0> K; // Number of cities
  vector[K] ymin;
  vector[K] ymax;
  int dstart[K]; 
  int dend[K];
}
transformed data {
  real mindif = min(ymax-ymin);
}
parameters {
  vector[K] sigma;
  vector<lower=-max(sigma)/mindif>[K] k;
}
model {
  for (i in 1:K){
    y[dstart[i]:dend[i]] ~ gpareto(ymin[i], k[i], sigma[i]);
  }
}
generated quantities {
  vector[N] log_lik;
  vector[N] yrep;
  vector[N] yrep_fut;

  for (i in 1:K) {
    for (n in dstart[i]:dend[i]) {
      log_lik[n] = gpareto_lpdf(rep_vector(y[n],1) | ymin[i], k[i], sigma[i]);
      yrep[n] = gpareto_rng(ymin[i], k[i], sigma[i]);
      yrep_fut[n] = gpareto_rng(ymin[i], 0.1, sigma[i]);
    }
  }
}