#!/usr/bin/env python3

import os
print("current working directory:", os.getcwd())

import cmdstanpy
# cmdstanpy.install_cmdstan()
from cmdstanpy import CmdStanModel

import numpy as np
import pandas as pd

# Read train_test.csv
train_test_split = pd.read_csv("../../train_test.csv")
train_size = train_test_split['train'][0]
test_size = train_test_split['test'][0]

data_path = "../../data/DXY_approx_percent_returns.csv"
data = pd.read_csv(data_path)

# Define dependent variable
y = data["DXY"].values
y = y - np.mean(y) # De-mean returns

# calculate the splitting point as the train_size * len of data, rounded
split_point = int(train_size * len(y))

# Load Stan model
stan_filepath = "sv.stan"
model = CmdStanModel(stan_file=stan_filepath)

# Train-test split
y_train = y[:split_point]
y_test = y[split_point:]


# Prepare data for Stan model
data = {
    'T_train': len(y_train),
    'T_test': len(y_test),
    'y_train': y_train,
    'y_test': y_test
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

# variables_to_write = ["phi", "sigma", "mu", "h_std", "mu_sim", "phi_sim", "sigma_sim", "h_std_sim", "h_sim", "scale_sim", "y_sim", "y_post"]
variables_to_write = ["y_sim", "y_post"]
write_samples_to_csv(fit, variables_to_write, "sampler_outputs")