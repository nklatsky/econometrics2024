// From Stan website: https://mc-stan.org/docs/2_21/stan-users-guide/stochastic-volatility-models.html

// Note: substitute more imformative priors where appropriate


data {
  int<lower=0> T;   // # time points (equally spaced)
  vector[T] y;      // mean corrected return at time t
}

parameters {
  real mu;                     // mean log volatility
  real<lower=-1, upper=1> phi; // persistence of volatility (auto-regression coefficient)
  real<lower=0.01> sigma;         // white noise shock scale (set lower bound slightly above zero to avoid computational issues)
  vector[T] h_std;             // std log volatility time t (for speed optimization)
}


transformed parameters {
  vector[T] h = h_std * sigma;  // now h ~ normal(0, sigma)
  h[1] /= sqrt(1 - phi * phi);  // rescale h[1]
  h += mu;
  for (t in 2:T)
    h[t] += phi * (h[t-1] - mu);

  vector[T] scale = exp(h / 2) + 1e-10 ;  // Compute scale parameters for the normal distribution (in transformed parameters block to enable inspection) ((also add 1e-10 to avoid errors))
}


model {

  phi ~ normal(0.90, 0.1) T[-1, 1];  // Truncated Normal Prior (formerly: phi ~ uniform(-1, 1);)

  sigma ~ inv_gamma(3, 2);   // Formerly cauchy(0, 5); 

  mu ~  std_normal();                // Formerly cauchy(0, 10);

  h_std ~ std_normal();              // (for speed optimization)
  
  y ~ normal(0, scale);         // Vectorized

}