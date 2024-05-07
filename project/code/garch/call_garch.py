#!/usr/bin/env python3

# Note: Filepaths are relative to the ECONOMETRICS2024 directory

import os
print("current working directory:", os.getcwd())

import cmdstanpy
# cmdstanpy.install_cmdstan()
from cmdstanpy import CmdStanModel

import numpy as np
import pandas as pd


data_path = "../data/fx_data_approx_percent_returns.csv"
data = pd.read_csv(data_path)

# Define dependent variable
y = data["DXY"].values

stan_filepath = "garch.stan"

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

## Error checking
## Access the samples for the parameters of interest
# h_samples = fit.stan_variable("h")
# sigma_samples = fit.stan_variable("sigma")
# scale_samples = fit.stan_variable("scale")
# #

## Save samples to CSV 
## Create folder to store sampler outputs (if it doesn't exist)
# os.makedirs("outputs", exist_ok=True)

## convert to DataFrame
# h_samples = pd.DataFrame(h_samples)
# sigma_samples = pd.DataFrame(sigma_samples)
# scale_samples = pd.DataFrame(scale_samples)

## Write to CSV
# h_samples.to_csv("outputs/h_samples.csv", index=False)
# sigma_samples.to_csv("outputs/sigma_samples.csv", index=False)
# scale_samples.to_csv("outputs/scale_samples.csv", index=False)



