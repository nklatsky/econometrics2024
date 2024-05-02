/* 

Construct a model in Stan such that we model the return on the vector of currencies y as a linear model with beta'x as the mean, 
where x is the avg return on the USD in that period and beta is the vector of coefficients.

The error term will follow a multivariate normal distribution with a full covariance matrix

*/

data {
  int<lower=0> N; // number of observations
  int<lower=0> K; // number of predictors
  matrix[N,K] X; // predictor matrix
  matrix[N,K] y; // response matrix
}

