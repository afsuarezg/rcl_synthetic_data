"""Synthetic data generator for a BLP random coefficients logit model.

Builds T markets x J products with 5 firms, 5 observed demand
characteristics, 2 cost shifters, and 4 demographics. Equilibrium prices
are the Bertrand-Nash solution implied by the true parameters, computed
by pyblp.Simulation.replace_endogenous().

Outputs (under --output-dir, default output/seed_{seed}/):
  - product_data.csv   product-market rows with shares, prices, instruments
  - agent_data.csv     integration nodes, weights, demographics
  - truth.pkl          dict of true parameters (beta, sigma, pi, gamma, xi, omega)

Example:
  python simulate.py                       # defaults: T=200, J=10, seed=0
  python simulate.py --seed 1
  python simulate.py --T 20 --J 5 --seed 42
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import pyblp

# ---------------------------------------------------------------------------
# Fixed model parameters (the data-generating "truth")
# ---------------------------------------------------------------------------
FIRM_PATTERN = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])  # length 10 (per market when J=10)

# Demand-side linear: 1 + prices + x1..x5  (K1 = 7)
# Constant negative so outside good keeps non-trivial share; mean price
# coefficient strongly negative so alpha_i stays negative everywhere.
BETA = np.array([-1.0, -1.5, 1.0, 0.8, 0.6, -0.4, 0.3])

# Demand-side nonlinear: 1 + prices + x1 + x2 + x3  (K2 = 5)
SIGMA = np.diag([0.5, 0.3, 0.4, 0.3, 0.2])

# Demographics (D = 4): income, age, hh_size, education. Pi is 5x4.
PI = np.array([
    # income  age   hh_size  educ
    [ 0.00,  0.20,  0.00,  0.00],   # constant
    [-0.20,  0.00,  0.00,  0.00],   # prices  (price-on-income)
    [ 0.00,  0.00,  0.30,  0.00],   # x1
    [ 0.00,  0.00,  0.00,  0.20],   # x2
    [ 0.20,  0.00,  0.00,  0.00],   # x3
])

# Supply-side (linear marginal cost): 1 + x1 + x2 + w1 + w2  (K3 = 5)
GAMMA = np.array([1.0, 0.3, 0.2, 0.4, 0.3])

XI_SD = 0.2
OMEGA_SD = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=200, help="number of markets")
    p.add_argument("--J", type=int, default=10, help="products per market")
    p.add_argument("--I-per-market", type=int, default=200,
                   help="integration nodes per market (Halton)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="default: output/seed_{seed}/")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(here, "output", f"seed_{args.seed}")
    os.makedirs(output_dir, exist_ok=True)

    T, J = args.T, args.J
    rng = np.random.default_rng(args.seed)

    # ---------------- Step 1: exogenous product skeleton ----------------
    N = T * J
    market_ids = np.repeat(np.arange(T), J)
    if J == FIRM_PATTERN.size:
        firm_ids = np.tile(FIRM_PATTERN, T)
    else:
        # Cycle through firms 1..5 when J != 10.
        firm_ids = np.tile(np.arange(1, 6).repeat(int(np.ceil(J / 5)))[:J], T)

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

    # ---------------- Step 2: agent data ----------------
    # Halton draws via pyblp.Integration give O(1/I^2) variance vs O(1/I) for
    # Monte Carlo, and pyblp builds the nodes/weights itself when we pass
    # integration=. We supply demographics ourselves (one row per node).
    K2 = SIGMA.shape[0]
    n_agents = T * args.I_per_market
    agent_market_ids = np.repeat(np.arange(T), args.I_per_market)

    raw_poisson = np.clip(rng.poisson(lam=2.5, size=n_agents), 1, None).astype(float)
    agent_data = pd.DataFrame({
        "market_ids": agent_market_ids,
        # Standardized so |Pi * d| is comparable to |Sigma * nu|.
        "income": rng.lognormal(mean=0.0, sigma=0.5, size=n_agents),
        "age": rng.standard_normal(size=n_agents),
        "hh_size": (raw_poisson - raw_poisson.mean()) / raw_poisson.std(),
        "education": rng.standard_normal(size=n_agents),
    })

    integration = pyblp.Integration("halton", args.I_per_market,
                                    specification_options={"seed": args.seed})

    # ---------------- Step 3: Simulation + equilibrium ----------------
    product_formulations = (
        pyblp.Formulation("1 + prices + x1 + x2 + x3 + x4 + x5"),
        pyblp.Formulation("1 + prices + x1 + x2 + x3"),
        pyblp.Formulation("1 + x1 + x2 + w1 + w2"),
    )
    agent_formulation = pyblp.Formulation("0 + income + age + hh_size + education")

    simulation = pyblp.Simulation(
        product_formulations=product_formulations,
        product_data=product_data,
        beta=BETA, sigma=SIGMA, pi=PI, gamma=GAMMA,
        agent_formulation=agent_formulation,
        agent_data=agent_data,
        integration=integration,
        xi=xi, omega=omega,
        costs_type="linear",
        seed=args.seed,
    )
    print(simulation)

    sim_results = simulation.replace_endogenous(
        iteration=pyblp.Iteration("simple", {"atol": 1e-12, "max_evaluations": 5000}),
        error_behavior="warn",
    )
    print("\n--- SimulationResults ---")
    print(sim_results)

    # pyblp raises on transient numpy overflow inside the contraction even when
    # the fixed point recovers. Gate on the real convergence diagnostic.
    converged = int(sim_results.fp_converged.sum())
    assert converged == T, f"only {converged} / {T} markets converged"

    # ---------------- Step 4: assemble outputs with instruments ----------------
    out_product = pd.DataFrame(pyblp.data_to_dict(sim_results.product_data))

    # BLP rivals-sum instruments (drop constant; otherwise collinear with it).
    demand_iv_blp = pyblp.build_blp_instruments(
        pyblp.Formulation("0 + x1 + x2 + x3 + x4 + x5"), out_product,
    )
    supply_iv_blp = pyblp.build_blp_instruments(
        pyblp.Formulation("0 + x1 + x2 + w1 + w2"), out_product,
    )
    # Gandhi-Houde differentiation instruments — much stronger for Sigma/Pi.
    demand_iv_diff = pyblp.build_differentiation_instruments(
        pyblp.Formulation("0 + x1 + x2 + x3 + x4 + x5"), out_product,
    )
    supply_iv_diff = pyblp.build_differentiation_instruments(
        pyblp.Formulation("0 + x1 + x2 + w1 + w2"), out_product,
    )
    demand_iv = np.column_stack([demand_iv_blp, demand_iv_diff])
    supply_iv = np.column_stack([supply_iv_blp, supply_iv_diff])
    for k in range(demand_iv.shape[1]):
        out_product[f"demand_instruments{k}"] = demand_iv[:, k]
    for k in range(supply_iv.shape[1]):
        out_product[f"supply_instruments{k}"] = supply_iv[:, k]

    # pyblp built nodes/weights and merged with our demographics on the
    # Simulation object (not on the results — agent data is exogenous).
    out_agent = pd.DataFrame(pyblp.data_to_dict(simulation.agent_data))

    # ---------------- Step 5: persist ----------------
    prod_path = os.path.join(output_dir, "product_data.csv")
    agent_path = os.path.join(output_dir, "agent_data.csv")
    truth_path = os.path.join(output_dir, "truth.pkl")

    out_product.to_csv(prod_path, index=False)
    out_agent.to_csv(agent_path, index=False)
    with open(truth_path, "wb") as fh:
        pickle.dump(
            {"beta": BETA, "sigma": SIGMA, "pi": PI, "gamma": GAMMA,
             "xi": xi, "omega": omega, "T": T, "J": J, "seed": args.seed},
            fh,
        )

    # Sanity numbers.
    shares_by_market = out_product.groupby("market_ids")["shares"].sum()
    costs = sim_results.compute_costs()
    markups = out_product["prices"].to_numpy() - costs.flatten()
    print("\n--- sanity ---")
    print(f"markets converged                      : {converged} / {T}")
    print(f"inside-good share, mean across markets : {shares_by_market.mean():.4f}")
    print(f"inside-good share, max  across markets : {shares_by_market.max():.4f}")
    print(f"mean markup (p - c)                    : {markups.mean():.4f}")
    print(f"share of products with p > c           : {(markups > 0).mean():.4f}")
    print(f"demand IVs                             : {demand_iv.shape[1]} "
          f"({demand_iv_blp.shape[1]} BLP + {demand_iv_diff.shape[1]} diff)")
    print(f"supply IVs                             : {supply_iv.shape[1]} "
          f"({supply_iv_blp.shape[1]} BLP + {supply_iv_diff.shape[1]} diff)")
    print(f"wrote {prod_path}  ({out_product.shape[0]} rows, {out_product.shape[1]} cols)")
    print(f"wrote {agent_path} ({out_agent.shape[0]} rows, {out_agent.shape[1]} cols)")
    print(f"wrote {truth_path}")


if __name__ == "__main__":
    main()
