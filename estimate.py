"""Round-trip estimator for the BLP synthetic data.

Loads CSVs written by simulate.py, builds a pyblp.Problem with matching
formulations and demographics, and calls problem.solve() from one or more
starting points. A clean round-trip (estimates near truth, GMM objective
near zero) is the sanity check that the synthetic data is BLP-shaped.

Example:
  python estimate.py                         # default output/seed_0
  python estimate.py --output-dir output/seed_1
  python estimate.py --method 2s --n-starts 10
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
import pyblp

# Loosen pyblp's inversion thresholds so the 2SLS weighting matrix falls
# back to a pseudo-inverse rather than aborting when MD+MS is large.
pyblp.options.collinear_atol = pyblp.options.collinear_rtol = 0.0
pyblp.options.singular_tol = 1e-14
pyblp.options.pseudo_inverses = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-dir", type=str, default=None,
                   help="default: output/seed_0/")
    p.add_argument("--method", type=str, default="2s", choices=["1s", "2s"],
                   help="GMM method (1-step or 2-step)")
    p.add_argument("--n-starts", type=int, default=5,
                   help="number of optimizer starts (start 0 = truth)")
    p.add_argument("--start-seed", type=int, default=12345,
                   help="seed for sampling perturbations of the starts")
    p.add_argument("--gtol", type=float, default=1e-5)
    p.add_argument("--iv-mode", choices=["both", "diff_only"], default="both",
                   help="which demand/supply instrument blocks to include in GMM. "
                        "'both' = BLP rivals-sum + Gandhi-Houde differentiation; "
                        "'diff_only' = differentiation only.")
    return p.parse_args()


def apply_iv_mode(product_data: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Filter and renumber instrument columns in product_data per `mode`.

    simulate.py writes BLP rivals-sum instruments first (demand 0..9, supply 0..7)
    followed by Gandhi-Houde differentiation instruments (demand 10..19, supply
    8..15). pyblp requires demand_instruments / supply_instruments to be
    contiguously numbered from 0 (utilities/basics.py: extract_matrix), so when
    we drop the BLP block we must also renumber the surviving diff block.
    """
    if mode == "both":
        return product_data

    if mode == "diff_only":
        out = product_data.copy()
        out = out.drop(columns=[f"demand_instruments{k}" for k in range(10)])
        out = out.drop(columns=[f"supply_instruments{k}" for k in range(8)])
        out = out.rename(columns={f"demand_instruments{10 + k}": f"demand_instruments{k}"
                                  for k in range(10)})
        out = out.rename(columns={f"supply_instruments{8 + k}": f"supply_instruments{k}"
                                  for k in range(8)})
        return out

    raise ValueError(f"unknown iv-mode {mode!r}")


def perturb(rng: np.random.Generator, x: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Multiplicative-magnitude perturbation, leaving zeros zero."""
    mask = x != 0
    out = x.copy()
    out[mask] = x[mask] + rng.normal(0.0, scale * np.abs(x[mask]))
    return out


def flatten_params(sigma: np.ndarray, pi: np.ndarray,
                   beta: np.ndarray, gamma: np.ndarray) -> dict[str, float]:
    """Map every scalar parameter to a stable label, used for both truth and estimates."""
    out: dict[str, float] = {}
    for k in range(sigma.shape[0]):
        out[f"sigma_{k}_{k}"] = float(sigma[k, k])
    for k in range(pi.shape[0]):
        for d in range(pi.shape[1]):
            out[f"pi_{k}_{d}"] = float(pi[k, d])
    for k, v in enumerate(np.asarray(beta).flatten()):
        out[f"beta_{k}"] = float(v)
    for k, v in enumerate(np.asarray(gamma).flatten()):
        out[f"gamma_{k}"] = float(v)
    return out


def main() -> None:
    args = parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(here, "output", "seed_0")

    product_data = pd.read_csv(os.path.join(output_dir, "product_data.csv"))
    agent_data = pd.read_csv(os.path.join(output_dir, "agent_data.csv"))
    with open(os.path.join(output_dir, "truth.pkl"), "rb") as fh:
        truth = pickle.load(fh)

    product_data = apply_iv_mode(product_data, args.iv_mode)
    n_demand_iv = sum(c.startswith("demand_instruments") for c in product_data.columns)
    n_supply_iv = sum(c.startswith("supply_instruments") for c in product_data.columns)
    print(f"loaded {len(product_data)} product-market rows, "
          f"{len(agent_data)} agent rows from {output_dir}")
    print(f"iv-mode={args.iv_mode}: {n_demand_iv} demand IVs, {n_supply_iv} supply IVs")

    variant_dir = os.path.join(output_dir, f"iv_{args.iv_mode}")
    os.makedirs(variant_dir, exist_ok=True)

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

    # Build initial beta vector: NaN entries get concentrated out, alpha
    # (price coefficient) must be optimized explicitly when supply is present.
    def beta_init(beta_truth: np.ndarray, alpha_start: float) -> np.ndarray:
        b = np.full(beta_truth.shape, np.nan)
        b[1] = alpha_start
        return b

    optimization = pyblp.Optimization("bfgs", {"gtol": args.gtol})
    rng = np.random.default_rng(args.start_seed)

    estimates_dir = os.path.join(variant_dir, "estimates")
    os.makedirs(estimates_dir, exist_ok=True)
    truth_params = flatten_params(truth["sigma"], truth["pi"],
                                  truth["beta"], truth["gamma"])

    records: list[dict] = []
    best_results = None
    best_obj = np.inf
    for i in range(args.n_starts):
        if i == 0:
            sigma0, pi0, alpha0 = truth["sigma"], truth["pi"], truth["beta"][1]
            tag = "truth"
        else:
            sigma0 = perturb(rng, truth["sigma"])
            pi0 = perturb(rng, truth["pi"])
            alpha0 = float(perturb(rng, np.array([truth["beta"][1]]))[0])
            tag = f"perturbed#{i}"

        t0 = time.perf_counter()
        try:
            res = problem.solve(
                sigma=sigma0, pi=pi0,
                beta=beta_init(truth["beta"], alpha0),
                optimization=optimization,
                method=args.method,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  start {i} ({tag}): solve failed — {exc.__class__.__name__}: {exc}")
            records.append({
                "start_id": i, "tag": tag, "objective": np.nan,
                "converged": False, "elapsed_sec": elapsed,
                "error_class": exc.__class__.__name__, "estimates": None,
            })
            continue
        elapsed = time.perf_counter() - t0

        pkl_path = os.path.join(estimates_dir, f"start_{i:02d}.pkl")
        with open(pkl_path, "wb") as fh:
            pickle.dump(res, fh)

        obj = float(res.objective)
        # pyblp ProblemResults exposes optimizer success differently across
        # versions; fall back to "solve returned" = converged for older builds.
        converged = bool(getattr(res, "converged", True))
        print(f"  start {i:>2} ({tag:>13s}): objective = {obj:.6e}  "
              f"converged={converged}  ({elapsed:.1f}s)")
        records.append({
            "start_id": i, "tag": tag, "objective": obj,
            "converged": converged, "elapsed_sec": elapsed,
            "error_class": "",
            "estimates": flatten_params(res.sigma, res.pi, res.beta, res.gamma),
        })
        if obj < best_obj:
            best_obj = obj
            best_results = res

    rows: list[dict] = []
    for rec in records:
        est = rec["estimates"]
        for pname, tval in truth_params.items():
            evalue = est[pname] if est is not None else np.nan
            rows.append({
                "start_id": rec["start_id"],
                "tag": rec["tag"],
                "param_name": pname,
                "truth": tval,
                "estimate": evalue,
                "abs_error": abs(evalue - tval) if est is not None else np.nan,
                "objective": rec["objective"],
                "converged": rec["converged"],
                "elapsed_sec": rec["elapsed_sec"],
                "error_class": rec["error_class"],
            })
    summary_path = os.path.join(variant_dir, "estimates_summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    n_pkl = sum(1 for r in records if r["estimates"] is not None)
    print(f"\nwrote {summary_path} ({len(rows)} rows)")
    print(f"wrote {n_pkl} pickle(s) to {estimates_dir}/")

    if best_results is None:
        raise RuntimeError("every optimizer start failed")

    print(f"\nbest of {args.n_starts} starts: objective = {best_obj:.6e}")
    print(best_results)

    print("\n=== Truth vs estimate ===")
    print("\nSigma (diagonal):")
    print("  truth    :", np.diag(truth["sigma"]))
    print("  estimate :", np.diag(best_results.sigma))

    print("\nPi:")
    print("  truth    :\n", truth["pi"])
    print("  estimate :\n", best_results.pi)

    print("\nBeta:")
    print("  truth    :", truth["beta"])
    print("  estimate :", best_results.beta.flatten())

    print("\nGamma:")
    print("  truth    :", truth["gamma"])
    print("  estimate :", best_results.gamma.flatten())


if __name__ == "__main__":
    main()
