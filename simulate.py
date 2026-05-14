"""Synthetic data generator for a BLP random coefficients logit model.

Builds 50 markets x 10 products with 5 firms, 5 observed demand
characteristics, 2 cost shifters, and 4 demographics. Prices are the
Bertrand-Nash equilibrium implied by the true parameters, solved by
pyblp.Simulation.replace_endogenous().

Outputs (under ./output):
  - product_data.csv   product-market rows with shares, prices, instruments
  - agent_data.csv     integration nodes, weights, and demographics
  - truth.pkl          dict of true parameters (beta, sigma, pi, gamma, xi, omega)
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import pyblp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
T = 50                                           # markets
J = 10                                           # products per market
FIRM_PATTERN = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # length J
SEED = 0

# Demand-side linear: 1 + prices + x1..x5  (K1 = 7)
# Constant set negative so the outside good keeps non-trivial share at
# typical prices; mean price coefficient strongly negative so alpha_i
# stays negative for every consumer (required for Bertrand equilibrium).
BETA = np.array([-1.0, -1.5, 1.0, 0.8, 0.6, -0.4, 0.3])

# Demand-side nonlinear: 1 + prices + x1 + x2 + x3  (K2 = 5)
# Modest heterogeneity on price so alpha_i = alpha + sigma*nu + pi*d stays
# negative with high probability over the support of (nu, d).
SIGMA = np.diag([0.5, 0.3, 0.4, 0.3, 0.2])

# Demographics (D = 4): income, age, hh_size, education
# Pi has K2 rows and D cols. Sparse: each demographic interacts with one
# nonlinear characteristic. Price-on-income kept modest for stability.
PI = np.array([
    # income  age   hh_size  educ
    [ 0.00,  0.20,  0.00,  0.00],   # constant
    [-0.20,  0.00,  0.00,  0.00],   # prices  (richer => less price-sensitive)
    [ 0.00,  0.00,  0.30,  0.00],   # x1
    [ 0.00,  0.00,  0.00,  0.20],   # x2
    [ 0.20,  0.00,  0.00,  0.00],   # x3
])

# Supply-side (linear marginal cost): 1 + x1 + x2 + w1 + w2  (K3 = 5)
# Constant + positive slopes on bounded characteristics keep marginal cost
# safely positive across all products (otherwise the Bertrand FOC iteration
# can blow up when c < 0 implies negative equilibrium prices).
GAMMA = np.array([1.0, 0.3, 0.2, 0.4, 0.3])

XI_SD = 0.2
OMEGA_SD = 0.1

I_PER_MARKET = 200        # Monte Carlo nodes per market for share integration

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: exogenous product skeleton
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)

N = T * J
market_ids = np.repeat(np.arange(T), J)
firm_ids = np.tile(FIRM_PATTERN, T)

X = rng.uniform(0.0, 1.0, size=(N, 5))   # x1..x5
W = rng.uniform(0.0, 1.0, size=(N, 2))   # w1, w2 (bounded so marginal cost stays positive)

product_data = pd.DataFrame({
    "market_ids": market_ids,
    "firm_ids": firm_ids,
    "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2], "x4": X[:, 3], "x5": X[:, 4],
    "w1": W[:, 0], "w2": W[:, 1],
})

xi = rng.normal(0.0, XI_SD, size=N)
omega = rng.normal(0.0, OMEGA_SD, size=N)


# ---------------------------------------------------------------------------
# Step 2: agent data (Monte Carlo integration nodes + demographics)
# ---------------------------------------------------------------------------
# Draw nodes ~ N(0, I_K2) directly. Uniform weights = 1 / I_PER_MARKET so the
# weighted sum approximates expectation over the standard normal heterogeneity.
K2 = SIGMA.shape[0]
n_agents = T * I_PER_MARKET

agent_market_ids = np.repeat(np.arange(T), I_PER_MARKET)
nodes = rng.standard_normal(size=(n_agents, K2))
weights = np.full(n_agents, 1.0 / I_PER_MARKET)

# Standardize raw-scale demographics so that |Pi * d| is on the same
# order as |Sigma * nu|. Tail agents on age (SD=5) or hh_size (Poisson, range 0-8)
# otherwise pick up huge random-coefficient contributions, saturate inside-good
# shares to 1, and make the demand Jacobian near-singular.
raw_poisson = np.clip(rng.poisson(lam=2.5, size=n_agents), 1, None).astype(float)
demographics = pd.DataFrame({
    "income": rng.lognormal(mean=0.0, sigma=0.5, size=n_agents),
    "age": rng.standard_normal(size=n_agents),                           # already ~N(0,1)
    "hh_size": (raw_poisson - raw_poisson.mean()) / raw_poisson.std(),   # standardized
    "education": rng.standard_normal(size=n_agents),
})

agent_data = pd.DataFrame({"market_ids": agent_market_ids, "weights": weights})
for k in range(K2):
    agent_data[f"nodes{k}"] = nodes[:, k]
agent_data = pd.concat([agent_data.reset_index(drop=True), demographics.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# Step 3: build pyblp.Simulation and solve equilibrium
# ---------------------------------------------------------------------------
product_formulations = (
    pyblp.Formulation("1 + prices + x1 + x2 + x3 + x4 + x5"),
    pyblp.Formulation("1 + prices + x1 + x2 + x3"),
    pyblp.Formulation("1 + x1 + x2 + w1 + w2"),
)
agent_formulation = pyblp.Formulation("0 + income + age + hh_size + education")

simulation = pyblp.Simulation(
    product_formulations=product_formulations,
    product_data=product_data,
    beta=BETA,
    sigma=SIGMA,
    pi=PI,
    gamma=GAMMA,
    agent_formulation=agent_formulation,
    agent_data=agent_data,
    xi=xi,
    omega=omega,
    costs_type="linear",
    seed=SEED,
)
print(simulation)

sim_results = simulation.replace_endogenous(
    iteration=pyblp.Iteration("simple", {"atol": 1e-12, "max_evaluations": 5000}),
    error_behavior="warn",
)
print("\n--- SimulationResults ---")
print(sim_results)

# pyblp raises on the *first* transient numpy overflow inside the ζ-contraction
# even when the fixed point recovers. Gate on the real convergence diagnostic.
assert int(sim_results.fp_converged.sum()) == T, (
    f"only {int(sim_results.fp_converged.sum())} / {T} markets converged"
)


# ---------------------------------------------------------------------------
# Step 4: assemble final DataFrames with instruments
# ---------------------------------------------------------------------------
# Convert results back to a DataFrame.
out_product = pd.DataFrame(pyblp.data_to_dict(sim_results.product_data))

# Build BLP-style instruments from the *exogenous* part of X1 and X3.
# Drop the constant from the IV formulations: build_blp_instruments returns
# sums of rivals' / other-firm characteristics, which, taken together with
# a constant, are perfectly collinear with the constant.
demand_iv = pyblp.build_blp_instruments(
    pyblp.Formulation("0 + x1 + x2 + x3 + x4 + x5"),
    out_product,
)
supply_iv = pyblp.build_blp_instruments(
    pyblp.Formulation("0 + x1 + x2 + w1 + w2"),
    out_product,
)
for k in range(demand_iv.shape[1]):
    out_product[f"demand_instruments{k}"] = demand_iv[:, k]
for k in range(supply_iv.shape[1]):
    out_product[f"supply_instruments{k}"] = supply_iv[:, k]


# ---------------------------------------------------------------------------
# Step 5: persist outputs and print a sanity summary
# ---------------------------------------------------------------------------
prod_path = os.path.join(OUTPUT_DIR, "product_data.csv")
agent_path = os.path.join(OUTPUT_DIR, "agent_data.csv")
truth_path = os.path.join(OUTPUT_DIR, "truth.pkl")

out_product.to_csv(prod_path, index=False)
agent_data.to_csv(agent_path, index=False)
with open(truth_path, "wb") as fh:
    pickle.dump(
        {
            "beta": BETA, "sigma": SIGMA, "pi": PI, "gamma": GAMMA,
            "xi": xi, "omega": omega,
            "T": T, "J": J,
        },
        fh,
    )

# Sanity numbers.
shares_by_market = out_product.groupby("market_ids")["shares"].sum()
costs = sim_results.compute_costs()
markups = out_product["prices"].to_numpy() - costs.flatten()
print("\n--- sanity ---")
print(f"inside-good share, mean across markets : {shares_by_market.mean():.4f}")
print(f"inside-good share, max  across markets : {shares_by_market.max():.4f}")
print(f"mean markup (p - c)                    : {markups.mean():.4f}")
print(f"share of products with p > c           : {(markups > 0).mean():.4f}")
print(f"wrote {prod_path}  ({out_product.shape[0]} rows, {out_product.shape[1]} cols)")
print(f"wrote {agent_path} ({agent_data.shape[0]} rows, {agent_data.shape[1]} cols)")
print(f"wrote {truth_path}")
