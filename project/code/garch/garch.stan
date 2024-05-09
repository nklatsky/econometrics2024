

data {
  int<lower=0> T_train; // len of training set
  int<lower=0> T_test; // len of test set
  vector[T_train] y_train;      // (training data) mean corrected return at time t
  vector[T_test] y_test; // (test data)
  real<lower=0> sigma_init; // initial condition for volatility
}

parameters {
  real<lower=0.01> mu;
  real<lower=0.01,upper=0.99> alpha1; // slightly less than 1 to avoid errors
  real<lower=0.01,upper=(1-alpha1)> beta1; // for stationarity
}

transformed parameters {
  vector<lower=0>[T_train] sigma_train;
  sigma_train[1] = sigma_init;
  for (t in 2:T_train)
    sigma_train[t] = sqrt(mu + alpha1 * pow(y_train[t-1], 2) + beta1 * pow(sigma_train[t-1], 2)) + 1e-10; // plus small constant to prevent underflow
}

model {

// Priors
  mu ~ inv_gamma(1.5, 0.25);          // Prior for the baseline variance component
  alpha1 ~ beta(2, 5);           // Prior for the coefficient of past squared residuals
  beta1 ~ beta(5, 2);            // Prior for the coefficient of past variances

// Likelihood for training data
  y_train ~ normal(0, sigma_train); // 

}


generated quantities {

// Prior Predictive Check (against training data)

  real<lower=0> mu_sim = inv_gamma_rng(1.5, 0.25);
  real<lower=0, upper=0.99> alpha1_sim = beta_rng(2, 5);
  real<lower=0, upper=1 - alpha1_sim> beta1_sim;

  vector<lower=0>[T_train] sigma_sim;
  vector[T_train] y_sim;

  sigma_sim[1] = sigma_init;
  y_sim[1] = normal_rng(0, sigma_sim[1]);
  
  for (t in 2:T_train) {

    // Dynamically adjust upper bound for beta1_sim based on current value of alpha1_sim
    beta1_sim = beta_rng(5, 2) * (1 - alpha1_sim);

    sigma_sim[t] = sqrt(mu_sim + alpha1_sim * pow(y_sim[t-1], 2) + beta1_sim * pow(sigma_sim[t-1], 2)) + 1e-10;
    y_sim[t] = normal_rng(0, sigma_sim[t]);
  }



// Posterior Predictive Check (against training data)

  vector[T_train] y_post;  // This will hold the posterior predictive simulated data

  // Using the posterior samples to simulate new data
  y_post[1] = normal_rng(0, sigma_train[1]);  // Initial value based on fitted sigma
  
  for (t in 2:T_train) {
    y_post[t] = normal_rng(0, sqrt(mu + alpha1 * pow(y_post[t-1], 2) + beta1 * pow(sigma_train[t-1], 2)) + 1e-10);
  }

}

