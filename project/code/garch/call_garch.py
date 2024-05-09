#!/usr/bin/env python3

import os
print("current working directory:", os.getcwd())

import cmdstanpy
# cmdstanpy.install_cmdstan()
from cmdstanpy import CmdStanModel

import numpy as np
import pandas as pd


data_path = "../../data/DXY_approx_percent_returns.csv"
data = pd.read_csv(data_path)

# Define dependent variable
y = data["DXY"].values
y = y - np.mean(y) # De-mean returns

stan_filepath = "garch.stan"

# Load Stan model
model = CmdStanModel(stan_file=stan_filepath)

# Prepare data for Stan model
data = {
    'T': len(y),
    'y': y,
    'sigma1': 1
}

# Fit the model
fit = model.sample(data=data)

# Print summary of the model fit
print(fit.summary())
# Print diagnosis
print(fit.diagnose())

## Diagnostics
def write_samples_to_csv(fit, variables, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for variable in variables:
        samples = fit.stan_variable(variable)
        samples_df = pd.DataFrame(samples)
        samples_df.to_csv(os.path.join(output_dir, f"{variable}_samples.csv"), index=False)

# variables_to_write = ["mu", "alpha1", "beta1", "mu_sim", "alpha1_sim", "beta1_sim", "sigma_sim", "y_sim", "y_post"]
variables_to_write = ["y_sim", "y_post"]
write_samples_to_csv(fit, variables_to_write, "sampler_outputs")

