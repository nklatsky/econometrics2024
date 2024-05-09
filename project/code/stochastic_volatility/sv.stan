// From Stan website: https://mc-stan.org/docs/2_21/stan-users-guide/stochastic-volatility-models.html

// Note: substitute more imformative priors where appropriate


data {
  int<lower=0> T;   // # time points (equally spaced)
  vector[T] y;      // mean corrected return at time t
}

parameters {
  real mu;                     // mean log volatility
  real<lower=-1, upper=1> phi; // persistence of volatility (auto-regression coefficient)
  real<lower=0.01, upper=3> sigma;         // white noise shock scale (set lower bound slightly above zero to avoid computational issues, restrict upper bound to avoid extreme values ())
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

  phi ~ normal(0.75, 0.1) T[-1, 1];  // Truncated Normal Prior (formerly: phi ~ uniform(-1, 1);)

  sigma ~ inv_gamma(2, 0.5);   // Non-negative dsitribution, centered near zero

  mu ~  normal(-0.5,1);                // Formerly cauchy(0, 10);

  h_std ~ std_normal();              // (for speed optimization)
  
  y ~ normal(0, scale);         // Vectorized

}


generated quantities {

  real mu_sim = normal_rng(-0.5, 1);                     // Sample mu from its prior
  real phi_sim;                                       // Declare phi_sim
  real sigma_sim;                          // Declare sigma_sim
  vector[T] h_std_sim;                                // Declare the vector for standard normal variates
  vector[T] h_sim;                                    // Simulated log volatility
  vector[T] scale_sim;                                // Simulated scale for y_sim
  vector[T] y_sim;                                    // Simulated data based on the prior
  
  // Fill h_std_sim with standard normal random variates
  for (i in 1:T) {
    h_std_sim[i] = std_normal_rng();
  }


  // Sample sigma using rejection sampling to ensure it stays within bounds (0,5)
  while (1) {
    sigma_sim = inv_gamma_rng(2, 0.5);
    if (sigma_sim <= 3) break;
  }


  // Sample phi using rejection sampling to ensure it falls within (-1, 1)
  while (1) {
    phi_sim = normal_rng(0.75, 0.1);
    if (phi_sim > -1 && phi_sim < 1)
      break;
  }

  // Initialize h_sim and scale_sim
  h_sim[1] = h_std_sim[1] * sigma_sim / sqrt(1 - phi_sim * phi_sim);
  h_sim[1] += mu_sim;
  scale_sim[1] = exp(h_sim[1] / 2) + 1e-10;

  for (t in 2:T) {
    h_sim[t] = h_std_sim[t] * sigma_sim;
    h_sim[t] += phi_sim * (h_sim[t-1] - mu_sim) + mu_sim;
    scale_sim[t] = exp(h_sim[t] / 2) + 1e-10;
  }

  // Simulate y_sim from the generated scale parameters
  for (t in 1:T) {
    y_sim[t] = normal_rng(0, scale_sim[t]);
  }
}