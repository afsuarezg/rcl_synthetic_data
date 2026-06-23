"""Validate estimated-RCL merger price predictions against the true counterfactual.

For unique_spec/seed_0 we have BOTH:
  - the true post-merger equilibrium (product_data_postmerger_12.csv), produced by
    merger.py via Simulation.replace_endogenous at the DGP truth, new ownership 2->1.
  - full estimation runs (iv_both / iv_diff_only, 20 starts each), each start_XX.pkl
    a complete pyblp.ProblemResults.

Prediction recipe (the realistic merger-analyst path):
  costs   = res.compute_costs()                      # invert DEMAND FOC at observed p_pre
  p_post  = res.compute_prices(firm_ids=merge_ids, costs=costs)   # re-solve post-merger FOC
This isolates how DEMAND-estimation error propagates to merger price effects; the
estimated gamma / cost shifters never enter (compute_costs backs costs out of demand).
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp

warnings.simplefilter("ignore")
pyblp.options.verbose = False

SEED_DIR = Path("output/unique_spec/seed_0")
TRUE_POST = SEED_DIR / "product_data_postmerger_12.csv"


def pct(pre: np.ndarray, post: np.ndarray) -> float:
    return float((post / pre - 1.0).mean() * 100.0)


def load_true_post() -> pd.DataFrame:
    return pd.read_csv(TRUE_POST)


def build_truth_simulation() -> pyblp.SimulationResults:
    with (SEED_DIR / "truth.pkl").open("rb") as fh:
        truth = pickle.load(fh)
    product = pd.read_csv(SEED_DIR / "product_data.csv")
    agents = pd.read_csv(SEED_DIR / "agent_data.csv")
    drop = [c for c in product.columns
            if c in ("prices", "shares")
            or c.startswith("demand_instruments")
            or c.startswith("supply_instruments")]
    sim = pyblp.Simulation(
        product_formulations=(
            pyblp.Formulation("1 + prices + x1 + x2 + x3 + x4 + x5"),
            pyblp.Formulation("1 + prices + x1 + x2 + x3"),
            pyblp.Formulation("1 + x1 + x2 + w1 + w2"),
        ),
        product_data=product.drop(columns=drop),
        beta=truth["beta"], sigma=truth["sigma"], pi=truth["pi"], gamma=truth["gamma"],
        agent_formulation=pyblp.Formulation("0 + income + age + hh_size + education"),
        agent_data=agents, xi=truth["xi"], omega=truth["omega"],
        costs_type="linear", seed=truth.get("seed"),
    )
    return sim.replace_endogenous(
        iteration=pyblp.Iteration("simple", {"atol": 1e-12, "max_evaluations": 5000}),
        error_behavior="warn",
    )


def predict_from(res, p_true_post: np.ndarray, p_pre: np.ndarray,
                 merge_ids: np.ndarray, orig_ids: np.ndarray, is_merging: np.ndarray):
    """Returns dict of accuracy metrics for one results object."""
    costs = res.compute_costs()
    # Round-trip: re-solving at ORIGINAL ownership must reproduce observed p_pre.
    p_pre_rt = res.compute_prices(firm_ids=orig_ids, costs=costs).flatten()
    rt_err = float(np.abs(p_pre_rt - p_pre).max())

    p_pred = res.compute_prices(firm_ids=merge_ids, costs=costs).flatten()
    err = p_pred - p_true_post                       # = Δp_pred - Δp_true (same baseline)
    return {
        "roundtrip_max_abs_err": rt_err,
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "max_abs_err": float(np.abs(err).max()),
        "pred_dp_merging_pct": pct(p_pre[is_merging], p_pred[is_merging]),
        "true_dp_merging_pct": pct(p_pre[is_merging], p_true_post[is_merging]),
        "pred_dp_nonmerging_pct": pct(p_pre[~is_merging], p_pred[~is_merging]),
        "true_dp_nonmerging_pct": pct(p_pre[~is_merging], p_true_post[~is_merging]),
        "corr_dp": float(np.corrcoef(p_pred - p_pre, p_true_post - p_pre)[0, 1]),
    }


def main() -> None:
    pre = pd.read_csv(SEED_DIR / "product_data.csv")
    post = load_true_post()
    assert np.allclose(pre["x1"].to_numpy(), post["x1"].to_numpy()), "row order mismatch"
    assert np.array_equal(pre["market_ids"], post["market_ids"]), "market mismatch"

    p_pre = pre["prices"].to_numpy()
    p_true_post = post["prices"].to_numpy()
    orig_ids = pre["firm_ids"].to_numpy()
    merge_ids = pd.Series(orig_ids).replace(2, 1).to_numpy()   # matches merger_12.pkl {2:1}
    is_merging = np.isin(orig_ids, [1, 2])

    print("=" * 78)
    print("  Merger-prediction validation  —  unique_spec/seed_0,  merge 1+2 (2->1)")
    print("=" * 78)
    print(f"  products: {len(pre)}   merging-firm products: {is_merging.sum()}")
    print(f"  TRUE counterfactual (merger.py / replace_endogenous at truth):")
    print(f"     Δprice merging firms : {pct(p_pre[is_merging], p_true_post[is_merging]):+.2f}%")
    print(f"     Δprice non-merging   : {pct(p_pre[~is_merging], p_true_post[~is_merging]):+.2f}%")

    # ---- Anchor: compute_prices at TRUTH must reproduce the merger.py CSV ----
    print("\n--- ANCHOR: compute_prices at truth vs merger.py replace_endogenous ---")
    truth_res = build_truth_simulation()
    tcosts = truth_res.compute_costs()
    p_truth_pred = truth_res.compute_prices(firm_ids=merge_ids, costs=tcosts).flatten()
    anchor_err = float(np.abs(p_truth_pred - p_true_post).max())
    print(f"  max|compute_prices(truth) - product_data_postmerger_12.csv| = {anchor_err:.2e}")
    print(f"  (validates the prediction machinery; small => any gap below is estimation)")

    # ---- Estimated runs ----
    for mode in ("iv_both", "iv_diff_only"):
        est_dir = SEED_DIR / mode / "estimates"
        summ = pd.read_csv(SEED_DIR / mode / "estimates_summary.csv")
        obj = (summ.groupby(["start_id", "tag"], as_index=False)["objective"]
               .first().sort_values("objective"))
        rows = []
        for _, r in obj.iterrows():
            sid = int(r["start_id"])
            pkl = est_dir / f"start_{sid:02d}.pkl"
            if not pkl.exists():
                continue
            with pkl.open("rb") as fh:
                res = pickle.load(fh)
            m = predict_from(res, p_true_post, p_pre, merge_ids, orig_ids, is_merging)
            m.update(start_id=sid, tag=r["tag"], objective=float(r["objective"]))
            rows.append(m)
        df = pd.DataFrame(rows)

        best = df.iloc[0]          # lowest objective (already sorted)
        truth_start = df[df["tag"] == "truth"].iloc[0]
        print(f"\n================  {mode}  ================")
        print(f"  best-objective start  (#{int(best.start_id)}, obj={best.objective:.3f}) "
              f"— what a real analyst would pick:")
        print(f"     Δprice merging : pred {best.pred_dp_merging_pct:+.2f}%  vs true "
              f"{best.true_dp_merging_pct:+.2f}%")
        print(f"     Δprice non-mrg : pred {best.pred_dp_nonmerging_pct:+.2f}%  vs true "
              f"{best.true_dp_nonmerging_pct:+.2f}%")
        print(f"     price RMSE {best.rmse:.4f}  MAE {best.mae:.4f}  "
              f"max|err| {best.max_abs_err:.4f}  corr(Δp) {best.corr_dp:.3f}")
        print(f"     pre-merger round-trip max abs err = {best.roundtrip_max_abs_err:.2e}")
        print(f"  truth-start (#{int(truth_start.start_id)}): "
              f"Δmerging pred {truth_start.pred_dp_merging_pct:+.2f}%  RMSE {truth_start.rmse:.4f}")
        print(f"  spread across {len(df)} starts: "
              f"Δmerging pred [{df.pred_dp_merging_pct.min():+.2f}%, "
              f"{df.pred_dp_merging_pct.max():+.2f}%]   "
              f"RMSE [{df.rmse.min():.4f}, {df.rmse.max():.4f}]")
        out = SEED_DIR / mode / "merger_prediction_validation.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
