#!/usr/bin/env python3

import cmdstanpy
# cmdstanpy.install_cmdstan()
from cmdstanpy import CmdStanModel

import numpy as np
import pandas as pd


data_path = "data/kaggle_fx_data_log_diff.csv"
data = pd.read_csv(data_path)
jpy_returns = data["JPY"].values

# Define dependent variable
y = jpy_returns

# Load Stan model
model = CmdStanModel(stan_file='stochastic_volatility.stan')

# Prepare data for Stan model
data = {
    'T': len(y),
    'y': y
}

# Fit the model
fit = model.sample(data=data)

# Print summary of the model fit
print(fit.summary())
# Print diagnosis
print(fit.diagnose())

