#!/usr/bin/env python3
import os
print("current working directory:", os.getcwd())

import cmdstanpy
# cmdstanpy.install_cmdstan()
from cmdstanpy import CmdStanModel

import numpy as np
import pandas as pd


data_path = "econometrics/econometrics2024/project/code/fx_data_approx_percent_returns.csv"
data = pd.read_csv(data_path)

# Define dependent variable
y = data["DXY"].values

stan_filepath = "econometrics/econometrics2024/project/code/stochastic_volatility/stochastic_volatility.stan"

# Load Stan model
model = CmdStanModel(stan_file=stan_filepath)

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

