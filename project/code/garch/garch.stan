

data {
  int<lower=0> T;
  vector[T] y;      // mean corrected return at time t
  real<lower=0> sigma1;
}

parameters {
  real<lower=0.01> mu;
  real<lower=0.01,upper=0.99> alpha1; // slightly less than 1 to avoid errors
  real<lower=0.01,upper=(1-alpha1)> beta1; // for stationarity
}

transformed parameters {
  vector<lower=0>[T] sigma;
  sigma[1] = sigma1;
  for (t in 2:T)
    sigma[t] = sqrt(mu + alpha1 * pow(y[t-1], 2) + beta1 * pow(sigma[t-1], 2)) + 1e-10; // plus small constant to prevent underflow
}

model {

// Priors
  mu ~ inv_gamma(3, 2);          // Prior for the baseline variance component
  alpha1 ~ beta(2, 5);         // Prior for the coefficient of past squared residuals
  beta1 ~ beta(5, 2);          // Prior for the coefficient of past variances

// Likelihood
  y ~ normal(0, sigma);
}


generated quantities {

// Prior Predictive Check

  real<lower=0> mu_sim = inv_gamma_rng(3, 2);
  real<lower=0, upper=0.99> alpha1_sim = beta_rng(2, 5);
  real<lower=0, upper=1 - alpha1_sim> beta1_sim;

  vector<lower=0>[T] sigma_sim;
  vector[T] y_sim;

  sigma_sim[1] = sigma1;
  y_sim[1] = normal_rng(0, sigma_sim[1]);
  
  for (t in 2:T) {

    // Dynamically adjust upper bound for beta1_sim based on current value of alpha1_sim
    beta1_sim = beta_rng(5, 2) * (1 - alpha1_sim);

    sigma_sim[t] = sqrt(mu_sim + alpha1_sim * pow(y_sim[t-1], 2) + beta1_sim * pow(sigma_sim[t-1], 2)) + 1e-10;
    y_sim[t] = normal_rng(0, sigma_sim[t]);
  }



// Posterior Predictive Check


}

