// From Stan website: https://mc-stan.org/docs/2_21/stan-users-guide/stochastic-volatility-models.html

// Note: substitute more imformative priors where appropriate


data {
  int<lower=0> T_train;   // len of training data
  vector[T_train] y_train;      // training data

  int<lower=0> T_test;  // len of test data
  vector[T_test] y_test; // len of test data
}

parameters {
  real mu;                     // mean log volatility
  real<lower=-1, upper=1> phi; // persistence of volatility (auto-regression coefficient)
  real<lower=0.01, upper=3> sigma;         // white noise shock scale (set lower bound slightly above zero to avoid computational issues, restrict upper bound to avoid extreme values ())
  vector[T_train] h_std_train;             // std log volatility time t (for speed optimization)
}


transformed parameters {
  vector[T_train] h_train = h_std_train * sigma;  // now h ~ normal(0, sigma)
  h_train[1] /= sqrt(1 - phi * phi);  // rescale h_train[1]
  h_train += mu;
  for (t in 2:T_train)
    h_train[t] += phi * (h_train[t-1] - mu);

  vector[T_train] scale = exp(h_train / 2) + 1e-10 ;  // Compute scale parameters for the normal distribution (in transformed parameters block to enable inspection) ((also add 1e-10 to avoid errors))
}


model {

  phi ~ normal(0.75, 0.1) T[-1, 1];  // Truncated Normal Prior (formerly: phi ~ uniform(-1, 1);)

  sigma ~ inv_gamma(2, 0.5);   // Non-negative dsitribution, centered near zero

  mu ~  normal(-0.5,1);                // Formerly cauchy(0, 10);

  h_std_train ~ std_normal();              // (for speed optimization)
  
  y_train ~ normal(0, scale);         // Vectorized [on training set]

}


generated quantities {

  // Prior Predictive Check

  real mu_sim = normal_rng(-0.5, 1);                     // Sample mu from its prior
  real phi_sim;                                       // Declare phi_sim
  real sigma_sim;                          // Declare sigma_sim
  vector[T_train] h_std_sim;                                // Declare the vector for standard normal variates
  vector[T_train] h_sim;                                    // Simulated log volatility
  vector[T_train] scale_sim;                                // Simulated scale for y_sim
  vector[T_train] y_sim;                                    // Simulated data based on the prior
  
  // Fill h_std_sim with standard normal random variates
  for (i in 1:T_train) {
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

  for (t in 2:T_train) {
    h_sim[t] = h_std_sim[t] * sigma_sim;
    h_sim[t] += phi_sim * (h_sim[t-1] - mu_sim) + mu_sim;
    scale_sim[t] = exp(h_sim[t] / 2) + 1e-10;
  }

  // Simulate y_sim from the generated scale parameters
  for (t in 1:T_train) {
    y_sim[t] = normal_rng(0, scale_sim[t]);
  }


  // Posterior Predictive Check
  vector[T_train] y_post;  // To hold posterior predictive simulated data
  vector[T_train] h_post;  // To hold posterior predictive log volatilities
  vector[T_train] scale_post;  // Scale parameters for the normal distribution based on the posterior
  
  h_post[1] = h_train[1];  // Starting from the first fitted volatility in training set
  scale_post[1] = exp(h_post[1] / 2) + 1e-10;  // Computing the scale
  
  // Simulate y_post[1] from the normal distribution with mean 0 and scale_post[1]
  y_post[1] = normal_rng(0, scale_post[1]);
  
  for (t in 2:T_train) {
    // Update h_post based on posterior sampled parameters
    h_post[t] = phi * (h_post[t-1] - mu) + mu + sigma * std_normal_rng();
    scale_post[t] = exp(h_post[t] / 2) + 1e-10;  // Compute the scale parameter

    // Simulate y_post[t] from the normal distribution with mean 0 and the updated scale_post[t]
    y_post[t] = normal_rng(0, scale_post[t]);
  }
     
// Out of Sample Forecasting
  vector[T_test] y_pred;   // Forecasts for the test data
  vector[T_test] h_pred;   // Log volatilities for the forecasts
  vector[T_test] scale_pred; // Scale parameters for the forecasts

  h_pred[1] = mu + (phi * (h_train[T_train] - mu)) + sigma * std_normal_rng();  // Init first forecast using last h_train
  scale_pred[1] = exp(h_pred[1] / 2) + 1e-10;  // Compute scale for the first forecast
  y_pred[1] = normal_rng(0, scale_pred[1]);  // Forecast the first test data point

  for (t in 2:T_test) {
    h_pred[t] = mu + (phi * (h_pred[t-1] - mu)) + sigma * std_normal_rng();  // Sequential forecast
    scale_pred[t] = exp(h_pred[t] / 2) + 1e-10;  // Compute scale for each forecast
    y_pred[t] = normal_rng(0, scale_pred[t]);  // Forecast subsequent test data points
  }

// Log-Likelihood of Observed Test Set Data Under Predictive Posterior distribution
  vector[T_test] log_likelihood; // For each forecast period

  for (t in 1:T_test) {

    log_likelihood[t] = normal_lpdf(y_test[t] | 0, scale_pred[t]); // log-likelihood of the t_th observation of y_test given the posterior predictive density defined by scale_pred[t]
  
  }

}