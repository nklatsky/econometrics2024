#!/usr/bin/env python3

import cmdstanpy
# cmdstanpy.install_cmdstan()
from cmdstanpy import CmdStanModel

import numpy as np
import pandas as pd

import os
print("Current working directory:", os.getcwd())

data_path = "project/code/fx_data_approx_percent_returns.csv"
data = pd.read_csv(data_path)

# Define dependent variable
y = data["DXY"].values


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

