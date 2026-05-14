"""Round-trip estimator for the BLP synthetic data.

Loads the CSVs written by simulate.py, builds a pyblp.Problem with the same
formulations and demographics, and calls problem.solve() starting from the
true parameter values. Prints a side-by-side comparison of truth vs estimate.

A clean round-trip (estimates near truth, objective near 0) is the
definitive sanity check that the synthetic data is BLP-shaped.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pyblp

# Loosen pyblp's collinearity/inversion thresholds so the 2SLS weighting
# matrix doesn't fall back to a pseudo-inverse and trip the inverse check.
pyblp.options.collinear_atol = pyblp.options.collinear_rtol = 0.0
pyblp.options.singular_tol = 1e-14
pyblp.options.pseudo_inverses = True

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")

product_data = pd.read_csv(os.path.join(OUTPUT_DIR, "product_data.csv"))
agent_data = pd.read_csv(os.path.join(OUTPUT_DIR, "agent_data.csv"))
with open(os.path.join(OUTPUT_DIR, "truth.pkl"), "rb") as fh:
    truth = pickle.load(fh)

print(f"loaded {len(product_data)} product-market rows, {len(agent_data)} agent rows")

# Same three product formulations and same agent formulation used in simulate.py.
product_formulations = (
    pyblp.Formulation("1 + prices + x1 + x2 + x3 + x4 + x5"),
    pyblp.Formulation("1 + prices + x1 + x2 + x3"),
    pyblp.Formulation("1 + x1 + x2 + w1 + w2"),
)
agent_formulation = pyblp.Formulation("0 + income + age + hh_size + education")

problem = pyblp.Problem(
    product_formulations=product_formulations,
    product_data=product_data,
    agent_formulation=agent_formulation,
    agent_data=agent_data,
    costs_type="linear",
)
print(problem)

# Start from the true Sigma, Pi, and price coefficient. With a supply side,
# pyblp does not concentrate alpha out of the GMM objective, so we must
# supply an initial value for the price coefficient (NaN for the other
# beta entries means "concentrate this one out").
beta_init = np.full(truth["beta"].shape, np.nan)
beta_init[1] = truth["beta"][1]   # price coefficient

results = problem.solve(
    sigma=truth["sigma"],
    pi=truth["pi"],
    beta=beta_init,
    optimization=pyblp.Optimization("bfgs", {"gtol": 1e-5}),
    method="1s",
)
print(results)

print("\n=== Truth vs estimate ===")
print("\nSigma (lower triangular):")
print("  truth diag    :", np.diag(truth["sigma"]))
print("  estimate diag :", np.diag(results.sigma))

print("\nPi:")
print("  truth    :\n", truth["pi"])
print("  estimate :\n", results.pi)

print("\nBeta (concentrated out):")
print("  truth    :", truth["beta"])
print("  estimate :", results.beta.flatten())

print("\nGamma (concentrated out):")
print("  truth    :", truth["gamma"])
print("  estimate :", results.gamma.flatten())

print(f"\nGMM objective at estimate : {float(results.objective):.6e}")
