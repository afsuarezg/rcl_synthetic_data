"""Counterfactual post-merger Bertrand-Nash equilibrium.

Loads a pre-merger synthetic draw (product_data.csv, agent_data.csv,
truth.pkl) produced by simulate.py, relabels firm_ids per --merge, and
re-solves equilibrium prices holding the demand/cost primitives, xi, omega,
and agent draws fixed. Writes product_data_postmerger_{tag}.csv alongside
the inputs.

Example:
  uv run python merger.py --seed 0 --merge 1+2
  uv run python merger.py --seed 0 --merge 1+2,3+4 --tag big_merger
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import pyblp


def parse_merge(spec: str) -> dict[int, int]:
    """Parse "1+2,3+4" into {2: 1, 4: 3}: acquired firm -> acquirer."""
    mapping: dict[int, int] = {}
    for group in spec.split(","):
        parts = [int(p) for p in group.split("+") if p.strip()]
        if len(parts) < 2:
            raise ValueError(f"merge group {group!r} needs at least two firms")
        acquirer, *acquired = parts
        for f in acquired:
            if f in mapping or f == acquirer:
                raise ValueError(f"firm {f} appears twice in merge spec")
            mapping[f] = acquirer
    return mapping


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--merge", type=str, required=True,
                   help='comma-separated merge groups, e.g. "1+2" or "1+2,3+4"')
    p.add_argument("--input-dir", type=str, default=None,
                   help="default: output/seed_{seed}/")
    p.add_argument("--tag", type=str, default=None,
                   help="suffix for output filename; default derived from --merge")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.input_dir or os.path.join(here, "output", f"seed_{args.seed}")

    with open(os.path.join(input_dir, "truth.pkl"), "rb") as fh:
        truth = pickle.load(fh)
    pre_product = pd.read_csv(os.path.join(input_dir, "product_data.csv"))
    agent_data = pd.read_csv(os.path.join(input_dir, "agent_data.csv"))

    merge_map = parse_merge(args.merge)
    pre_firms = pre_product["firm_ids"].to_numpy()
    post_firms = np.array([merge_map.get(int(f), int(f)) for f in pre_firms])
    if np.array_equal(pre_firms, post_firms):
        raise ValueError(f"merge spec {args.merge!r} did not change any firm_ids "
                         f"(present firms: {sorted(set(int(f) for f in pre_firms))})")

    # Counterfactual product data: keep exogenous covariates and unobservables
    # fixed; only ownership changes. Drop equilibrium outputs and ownership-
    # dependent instruments so pyblp rebuilds them at the new ownership.
    post_product = pre_product.copy()
    post_product["firm_ids"] = post_firms
    drop = [c for c in post_product.columns
            if c in ("prices", "shares")
            or c.startswith("demand_instruments")
            or c.startswith("supply_instruments")]
    post_product = post_product.drop(columns=drop)

    product_formulations = (
        pyblp.Formulation("1 + prices + x1 + x2 + x3 + x4 + x5"),
        pyblp.Formulation("1 + prices + x1 + x2 + x3"),
        pyblp.Formulation("1 + x1 + x2 + w1 + w2"),
    )
    agent_formulation = pyblp.Formulation("0 + income + age + hh_size + education")

    simulation = pyblp.Simulation(
        product_formulations=product_formulations,
        product_data=post_product,
        beta=truth["beta"], sigma=truth["sigma"], pi=truth["pi"], gamma=truth["gamma"],
        agent_formulation=agent_formulation,
        agent_data=agent_data,
        xi=truth["xi"], omega=truth["omega"],
        costs_type="linear",
        seed=truth.get("seed", args.seed),
    )
    print(simulation)

    sim_results = simulation.replace_endogenous(
        iteration=pyblp.Iteration("simple", {"atol": 1e-12, "max_evaluations": 5000}),
        error_behavior="warn",
    )
    print("\n--- post-merger SimulationResults ---")
    print(sim_results)

    T = int(truth["T"])
    converged = int(sim_results.fp_converged.sum())
    assert converged == T, f"only {converged} / {T} markets converged"

    out_product = pd.DataFrame(pyblp.data_to_dict(sim_results.product_data))

    # Rebuild ownership-dependent instruments at the new firm_ids.
    demand_iv_blp = pyblp.build_blp_instruments(
        pyblp.Formulation("0 + x1 + x2 + x3 + x4 + x5"), out_product,
    )
    supply_iv_blp = pyblp.build_blp_instruments(
        pyblp.Formulation("0 + x1 + x2 + w1 + w2"), out_product,
    )
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

    tag = args.tag or "_".join(f"{a}{b}" for b, a in sorted(merge_map.items()))
    prod_path = os.path.join(input_dir, f"product_data_postmerger_{tag}.csv")
    out_product.to_csv(prod_path, index=False)

    meta_path = os.path.join(input_dir, f"merger_{tag}.pkl")
    with open(meta_path, "wb") as fh:
        pickle.dump({"merge_map": merge_map, "seed": args.seed, "spec": args.merge}, fh)

    # ---------------- Sanity / merger summary ----------------
    pre_prices = pre_product["prices"].to_numpy()
    post_prices = out_product["prices"].to_numpy()
    pre_shares = pre_product["shares"].to_numpy()
    post_shares = out_product["shares"].to_numpy()
    pre_inside_share = pre_product.groupby("market_ids")["shares"].sum().mean()
    post_inside_share = out_product.groupby("market_ids")["shares"].sum().mean()

    # A product is "merging" if its pre-merger firm appears anywhere in the merge map.
    touched = set(merge_map) | set(merge_map.values())
    is_merging = np.array([int(f) in touched for f in pre_firms])

    def pct(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (b / a - 1.0) * 100.0

    print("\n--- merger summary ---")
    print(f"merge map (acquired -> acquirer)       : {merge_map}")
    print(f"markets converged                      : {converged} / {T}")
    print(f"inside-good share, pre  / post         : {pre_inside_share:.4f} / {post_inside_share:.4f}")
    print(f"mean price change, all products        : {(post_prices - pre_prices).mean():+.4f}  "
          f"({pct(pre_prices, post_prices).mean():+.2f}%)")
    if is_merging.any():
        print(f"mean price change, merging firms       : {(post_prices - pre_prices)[is_merging].mean():+.4f}  "
              f"({pct(pre_prices[is_merging], post_prices[is_merging]).mean():+.2f}%)")
    if (~is_merging).any():
        print(f"mean price change, non-merging firms   : {(post_prices - pre_prices)[~is_merging].mean():+.4f}  "
              f"({pct(pre_prices[~is_merging], post_prices[~is_merging]).mean():+.2f}%)")
    print(f"mean share change, merging firms       : "
          f"{(post_shares - pre_shares)[is_merging].mean():+.4f}" if is_merging.any() else "")
    print(f"wrote {prod_path}  ({out_product.shape[0]} rows, {out_product.shape[1]} cols)")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
