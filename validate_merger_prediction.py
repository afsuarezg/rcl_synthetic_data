"""Validate estimated-RCL merger predictions against the true counterfactual.

Two modes (see --sweep):

  default (single seed, full IV runs) — unique_spec/seed_0, which has BOTH:
    - the true post-merger equilibrium (product_data_postmerger_12.csv), from
      merger.py via Simulation.replace_endogenous at truth, ownership 2->1, and
    - full estimation runs (iv_both / iv_diff_only, 20 starts each).

  --sweep (spec comparison) — multiple_specs/seed_0, scoring the merger
    prediction for all 60 demand specifications x their starts. No post-merger
    CSV exists there, so the true counterfactual is recomputed once from the
    DGP truth (build_truth_simulation), which the single-mode anchor shows
    reproduces merger.py to ~1e-12.

Prediction recipe (the realistic merger-analyst path):
  costs   = res.compute_costs()                      # invert DEMAND FOC at observed p_pre
  p_post  = res.compute_prices(firm_ids=merge_ids, costs=costs)   # re-solve post-merger FOC
This isolates how DEMAND-estimation error propagates to merger effects; the
estimated gamma / cost shifters never enter (compute_costs backs costs out of demand).
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp

warnings.simplefilter("ignore")
pyblp.options.verbose = False

MERGE_FROM, MERGE_TO = 2, 1          # merger_12.pkl mapping: {2: 1}
MERGING_FIRMS = (MERGE_TO, MERGE_FROM)


def pct(pre: np.ndarray, post: np.ndarray) -> float:
    return float((post / pre - 1.0).mean() * 100.0)


def build_truth_simulation(seed_dir: Path) -> pyblp.SimulationResults:
    """Reconstruct the DGP Simulation at truth (pre-merger ownership) and solve."""
    with (seed_dir / "truth.pkl").open("rb") as fh:
        truth = pickle.load(fh)
    product = pd.read_csv(seed_dir / "product_data.csv")
    agents = pd.read_csv(seed_dir / "agent_data.csv")
    drop = [c for c in product.columns
            if c in ("prices", "shares", "product_slot")
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


def mean_delta_hhi(res, merge_ids: np.ndarray, new_shares: np.ndarray) -> float:
    """Mean over markets of (post-merger HHI - pre-merger HHI), per compute_elasticities."""
    pre_hhi = np.asarray(res.compute_hhi()).mean()
    post_hhi = np.asarray(res.compute_hhi(firm_ids=merge_ids, shares=new_shares)).mean()
    return float(post_hhi - pre_hhi)


def predict_from(res, p_true_post: np.ndarray, p_pre: np.ndarray, merge_ids: np.ndarray,
                 orig_ids: np.ndarray, is_merging: np.ndarray) -> dict:
    """Accuracy metrics (prices + Δ-HHI) for one ProblemResults/SimulationResults."""
    costs = res.compute_costs()
    # Round-trip: re-solving at ORIGINAL ownership must reproduce observed p_pre.
    p_pre_rt = res.compute_prices(firm_ids=orig_ids, costs=costs).flatten()
    rt_err = float(np.abs(p_pre_rt - p_pre).max())

    pp = res.compute_prices(firm_ids=merge_ids, costs=costs)      # (N, 1)
    p_pred = pp.flatten()
    new_shares = res.compute_shares(pp)
    err = p_pred - p_true_post                       # = Δp_pred - Δp_true (same baseline)
    return {
        "roundtrip_max_abs_err": rt_err,
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "max_abs_err": float(np.abs(err).max()),
        "pred_dp_merging_pct": pct(p_pre[is_merging], p_pred[is_merging]),
        "pred_dp_nonmerging_pct": pct(p_pre[~is_merging], p_pred[~is_merging]),
        "corr_dp": float(np.corrcoef(p_pred - p_pre, p_true_post - p_pre)[0, 1]),
        "pred_delta_hhi": mean_delta_hhi(res, merge_ids, new_shares),
    }


def setup(seed_dir: Path):
    """Common pre-merger arrays + merge mapping for a seed dir."""
    pre = pd.read_csv(seed_dir / "product_data.csv")
    p_pre = pre["prices"].to_numpy()
    orig_ids = pre["firm_ids"].to_numpy()
    merge_ids = pd.Series(orig_ids).replace(MERGE_FROM, MERGE_TO).to_numpy()
    is_merging = np.isin(orig_ids, MERGING_FIRMS)
    return pre, p_pre, orig_ids, merge_ids, is_merging


def truth_benchmark(seed_dir: Path, p_pre, merge_ids, orig_ids, is_merging):
    """True post-merger prices + Δ-HHI from the DGP, plus the machinery anchor err."""
    truth_res = build_truth_simulation(seed_dir)
    tcosts = truth_res.compute_costs()
    pp = truth_res.compute_prices(firm_ids=merge_ids, costs=tcosts)
    p_true_post = pp.flatten()
    true_shares = truth_res.compute_shares(pp)
    bench = {
        "true_dp_merging_pct": pct(p_pre[is_merging], p_true_post[is_merging]),
        "true_dp_nonmerging_pct": pct(p_pre[~is_merging], p_true_post[~is_merging]),
        "true_delta_hhi": mean_delta_hhi(truth_res, merge_ids, true_shares),
    }
    return p_true_post, bench


# ---------------------------------------------------------------------------
# Mode 1: single seed, full IV runs (unique_spec/seed_0)
# ---------------------------------------------------------------------------

def run_single(seed_dir: Path) -> None:
    pre, p_pre, orig_ids, merge_ids, is_merging = setup(seed_dir)
    post = pd.read_csv(seed_dir / "product_data_postmerger_12.csv")
    assert np.allclose(pre["x1"], post["x1"]), "row order mismatch"
    assert np.array_equal(pre["market_ids"], post["market_ids"]), "market mismatch"
    p_true_post = post["prices"].to_numpy()

    print("=" * 78)
    print(f"  Merger-prediction validation  —  {seed_dir},  merge 1+2 (2->1)")
    print("=" * 78)
    print(f"  products: {len(pre)}   merging-firm products: {is_merging.sum()}")
    print(f"  TRUE counterfactual (merger.py / replace_endogenous at truth):")
    print(f"     Δprice merging firms : {pct(p_pre[is_merging], p_true_post[is_merging]):+.2f}%")
    print(f"     Δprice non-merging   : {pct(p_pre[~is_merging], p_true_post[~is_merging]):+.2f}%")

    print("\n--- ANCHOR: compute_prices at truth vs merger.py replace_endogenous ---")
    p_truth_pred, _ = truth_benchmark(seed_dir, p_pre, merge_ids, orig_ids, is_merging)
    print(f"  max|compute_prices(truth) - product_data_postmerger_12.csv| = "
          f"{np.abs(p_truth_pred - p_true_post).max():.2e}")

    for mode in ("iv_both", "iv_diff_only"):
        est_dir = seed_dir / mode / "estimates"
        summ = pd.read_csv(seed_dir / mode / "estimates_summary.csv")
        obj = (summ.groupby(["start_id", "tag"], as_index=False)["objective"]
               .first().sort_values("objective"))
        rows = []
        for _, r in obj.iterrows():
            pkl = est_dir / f"start_{int(r['start_id']):02d}.pkl"
            if not pkl.exists():
                continue
            with pkl.open("rb") as fh:
                res = pickle.load(fh)
            m = predict_from(res, p_true_post, p_pre, merge_ids, orig_ids, is_merging)
            m.update(start_id=int(r["start_id"]), tag=r["tag"], objective=float(r["objective"]))
            rows.append(m)
        df = pd.DataFrame(rows)
        best, truth_start = df.iloc[0], df[df["tag"] == "truth"].iloc[0]
        print(f"\n================  {mode}  ================")
        print(f"  best-objective start (#{int(best.start_id)}, obj={best.objective:.3f}):")
        print(f"     Δprice merging : pred {best.pred_dp_merging_pct:+.2f}%  (true "
              f"{pct(p_pre[is_merging], p_true_post[is_merging]):+.2f}%)   "
              f"RMSE {best.rmse:.4f}  corr {best.corr_dp:.3f}")
        print(f"  truth-start (#{int(truth_start.start_id)}): "
              f"Δmerging pred {truth_start.pred_dp_merging_pct:+.2f}%  RMSE {truth_start.rmse:.4f}")
        print(f"  spread across {len(df)} starts: Δmerging "
              f"[{df.pred_dp_merging_pct.min():+.2f}%, {df.pred_dp_merging_pct.max():+.2f}%]  "
              f"RMSE [{df.rmse.min():.4f}, {df.rmse.max():.4f}]")
        out = seed_dir / mode / "merger_prediction_validation.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Mode 2: sweep all 60 demand specifications (multiple_specs/seed_0)
# ---------------------------------------------------------------------------

def starts_by_objective(summ: pd.DataFrame) -> list[int]:
    """Start ids ordered by ascending objective, dropping errored (NaN) starts.

    The first element is the best start (mirrors run_specs best_per_spec); the
    rest are the next-best fallbacks used when the best pickle fails to load.
    """
    per = summ.groupby("start_id", as_index=False)["objective"].first()
    per = per[per["objective"].notna()]
    return [int(s) for s in per.sort_values("objective")["start_id"].tolist()]


def _predict_pickle(pkl: Path, label: str, sid: int, p_true_post, p_pre,
                    merge_ids, orig_ids, is_merging) -> dict | None:
    """Load one start pickle and run predict_from; warn & return None on failure."""
    try:
        with pkl.open("rb") as fh:
            res = pickle.load(fh)
        return predict_from(res, p_true_post, p_pre, merge_ids, orig_ids, is_merging)
    except Exception as exc:                            # noqa: BLE001 - log & skip
        warnings.warn(f"{label} start {sid}: {exc.__class__.__name__}: {exc}")
        return None


def run_sweep(seed_dir: Path, mode: str, all_starts: bool = False) -> None:
    pre, p_pre, orig_ids, merge_ids, is_merging = setup(seed_dir)
    p_true_post, bench = truth_benchmark(seed_dir, p_pre, merge_ids, orig_ids, is_merging)

    spec_root = seed_dir / mode / "specs"
    spec_dirs = sorted(d for d in spec_root.glob("spec_*") if d.is_dir())
    print("=" * 90)
    print(f"  Merger-prediction sweep — {seed_dir} / {mode},  {len(spec_dirs)} specs,  merge 1+2")
    print("=" * 90)
    print(f"  TRUE: Δprice merging {bench['true_dp_merging_pct']:+.2f}%   "
          f"Δprice non-merging {bench['true_dp_nonmerging_pct']:+.2f}%   "
          f"Δ-HHI {bench['true_delta_hhi']:+.1f}")

    long_rows, spec_rows = [], []
    for sd in spec_dirs:
        label = sd.name.replace("spec_", "")
        summ_path = sd / "estimates_summary.csv"
        if not summ_path.exists():
            continue
        summ = pd.read_csv(summ_path)
        ordered = starts_by_objective(summ)
        bid = ordered[0] if ordered else None
        est_dir = sd / "estimates"
        per_start = []
        if all_starts:
            # Original behavior: predict_from on every start (full per-start spread).
            for pkl in sorted(est_dir.glob("start_*.pkl")):
                sid = int(pkl.stem.split("_")[1])
                m = _predict_pickle(pkl, label, sid, p_true_post, p_pre,
                                    merge_ids, orig_ids, is_merging)
                if m is None:
                    continue
                m.update(spec_label=label, start_id=sid, is_best=(sid == bid))
                per_start.append(m)
                long_rows.append(m)
        else:
            # Default: every downstream report (37, 38-40, 41) uses only the best
            # start per spec, so simulate just that one -- ~60 predict_from calls
            # instead of 60*N. Walk starts by ascending objective and stop at the
            # first that loads, falling back to the next-best if the best pickle is
            # missing/corrupt so the spec never silently drops out of the reports.
            for sid in ordered:
                pkl = est_dir / f"start_{sid:02d}.pkl"
                m = _predict_pickle(pkl, label, sid, p_true_post, p_pre,
                                    merge_ids, orig_ids, is_merging)
                if m is None:
                    continue
                m.update(spec_label=label, start_id=sid, is_best=(sid == bid))
                per_start.append(m)
                long_rows.append(m)
                break
        if not per_start:
            continue
        pdf = pd.DataFrame(per_start)
        b = pdf[pdf["is_best"]].iloc[0] if pdf["is_best"].any() else pdf.iloc[0]
        spec_rows.append({
            "spec_label": label,
            "n_starts": len(pdf),
            "best_start_id": int(b["start_id"]),
            "pred_dp_merging_pct": float(b["pred_dp_merging_pct"]),
            "dp_merging_err": float(b["pred_dp_merging_pct"] - bench["true_dp_merging_pct"]),
            "pred_delta_hhi": float(b["pred_delta_hhi"]),
            "delta_hhi_err": float(b["pred_delta_hhi"] - bench["true_delta_hhi"]),
            "price_rmse": float(b["rmse"]),
            "corr_dp": float(b["corr_dp"]),
            "roundtrip_max_abs_err": float(b["roundtrip_max_abs_err"]),
            "dp_merging_min": float(pdf["pred_dp_merging_pct"].min()),
            "dp_merging_max": float(pdf["pred_dp_merging_pct"].max()),
            "rmse_min": float(pdf["rmse"].min()),
            "rmse_max": float(pdf["rmse"].max()),
        })

    spec_df = pd.DataFrame(spec_rows).sort_values("price_rmse").reset_index(drop=True)
    for k, v in bench.items():
        spec_df[f"bench_{k}"] = v

    out_by_spec = seed_dir / mode / "merger_prediction_by_spec.csv"
    out_long = seed_dir / mode / "merger_prediction_by_spec_long.csv"
    spec_df.to_csv(out_by_spec, index=False)
    pd.DataFrame(long_rows).to_csv(out_long, index=False)

    def show(rows: pd.DataFrame, title: str) -> None:
        print(f"\n  {title}")
        print(f"  {'spec':<48}{'Δmerge%':>9}{'Δ-HHI':>9}{'ΔHHIerr':>9}"
              f"{'RMSE':>8}{'corr':>7}")
        for _, r in rows.iterrows():
            print(f"  {r.spec_label:<48}{r.pred_dp_merging_pct:>+9.2f}{r.pred_delta_hhi:>+9.1f}"
                  f"{r.delta_hhi_err:>+9.1f}{r.price_rmse:>8.4f}{r.corr_dp:>7.3f}")

    show(spec_df.head(8), "MOST accurate specs (lowest price RMSE):")
    show(spec_df.tail(8).iloc[::-1], "LEAST accurate specs (highest price RMSE):")
    n_wrong = int((spec_df["corr_dp"] < 0).sum())
    print(f"\n  specs with wrong-sign Δp (corr<0): {n_wrong} / {len(spec_df)}")
    print(f"  Δ-HHI error range: [{spec_df.delta_hhi_err.min():+.1f}, "
          f"{spec_df.delta_hhi_err.max():+.1f}]  (true Δ-HHI = {bench['true_delta_hhi']:+.1f})")
    scope = "all starts" if all_starts else "best start per spec"
    print(f"\n  wrote {out_by_spec}  ({len(spec_df)} specs)")
    print(f"  wrote {out_long}  ({len(long_rows)} spec-start rows, {scope})")


# ---------------------------------------------------------------------------
# Mode 3: per-product (per-observation) prices for the best spec
# ---------------------------------------------------------------------------

def _slot_labels(pre: pd.DataFrame, is_merging: np.ndarray) -> pd.Series:
    """One label per merging obs: firm<id><a|b>. The DGP redraws products every
    market (no persistent product identity), so the 4 merging "products" are 4
    ownership slots -- firm 1's two products and firm 2's two -- of exchangeable
    draws. a/b is the within-(market,firm) order; it carries no meaning beyond
    splitting each firm's pair."""
    ab = pre.groupby(["market_ids", "firm_ids"]).cumcount().map({0: "a", 1: "b"})
    labels = "firm" + pre["firm_ids"].astype(int).astype(str) + ab
    return labels[is_merging].reset_index(drop=True)


def run_per_product(seed_dir: Path, mode: str) -> None:
    """Per-merging-observation truth vs. best-spec predicted price increase.

    Reads the best spec (rank-1 by price RMSE) from merger_prediction_by_spec.csv,
    re-solves the post-merger FOC for that one estimate, and writes the predicted
    and true %Δprice for every merging product-market observation.
    """
    by_spec_csv = seed_dir / mode / "merger_prediction_by_spec.csv"
    if not by_spec_csv.exists():
        raise SystemExit(f"{by_spec_csv} not found -- run --sweep first.")
    by_spec = (pd.read_csv(by_spec_csv)
               .sort_values("price_rmse").reset_index(drop=True))
    best = by_spec.iloc[0]
    label, bid = str(best["spec_label"]), int(best["best_start_id"])
    pkl = seed_dir / mode / "specs" / f"spec_{label}" / "estimates" / f"start_{bid:02d}.pkl"
    if not pkl.exists():
        raise SystemExit(f"best-spec pickle not found: {pkl}")

    pre, p_pre, orig_ids, merge_ids, is_merging = setup(seed_dir)
    p_true_post, bench = truth_benchmark(seed_dir, p_pre, merge_ids, orig_ids, is_merging)
    with pkl.open("rb") as fh:
        res = pickle.load(fh)
    costs = res.compute_costs()
    p_pred = res.compute_prices(firm_ids=merge_ids, costs=costs).flatten()

    obs = pre.loc[is_merging, ["market_ids", "firm_ids"]].reset_index(drop=True)
    obs["slot"] = _slot_labels(pre, is_merging)
    obs["p_pre"] = p_pre[is_merging]
    obs["p_true_post"] = p_true_post[is_merging]
    obs["p_pred"] = p_pred[is_merging]
    obs["true_dp_pct"] = (obs["p_true_post"] / obs["p_pre"] - 1.0) * 100.0
    obs["pred_dp_pct"] = (obs["p_pred"] / obs["p_pre"] - 1.0) * 100.0
    obs["spec_label"] = label
    obs["start_id"] = bid

    out = seed_dir / mode / "merger_prediction_by_product.csv"
    obs.to_csv(out, index=False)

    print("=" * 90)
    print(f"  Per-product merger prediction — {seed_dir} / {mode}")
    print("=" * 90)
    print(f"  best spec (rank 1 by price RMSE): {label}  (start {bid})")
    print(f"  merging obs: {len(obs)}   true Δp merging {bench['true_dp_merging_pct']:+.2f}%")
    corr = float(np.corrcoef(obs["pred_dp_pct"], obs["true_dp_pct"])[0, 1])
    rmse = float(np.sqrt(((obs["pred_dp_pct"] - obs["true_dp_pct"]) ** 2).mean()))
    print(f"  overall %Δp  corr {corr:.4f}   RMSE {rmse:.4f}")
    print(f"  {'slot':<8}{'n':>5}{'true_mean':>11}{'pred_mean':>11}")
    for slot, g in obs.groupby("slot"):
        print(f"  {slot:<8}{len(g):>5}{g['true_dp_pct'].mean():>11.3f}"
              f"{g['pred_dp_pct'].mean():>11.3f}")
    print(f"  wrote {out}  ({len(obs)} obs)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sweep", action="store_true",
                   help="run the 60-spec sweep instead of the single-seed validation")
    p.add_argument("--per-product", action="store_true",
                   help="write per-merging-observation truth vs best-spec predicted "
                        "%%Δprice (needs the --sweep CSV); default seed output/multiple_specs/seed_0")
    p.add_argument("--seed-dir", type=Path, default=None,
                   help="seed dir (default: output/unique_spec/seed_0 single, "
                        "output/multiple_specs/seed_0 sweep)")
    p.add_argument("--mode", choices=["iv_both", "iv_diff_only"], default="iv_both",
                   help="instrument mode for --sweep (default iv_both)")
    p.add_argument("--all-starts", action="store_true",
                   help="run predict_from on every estimation start (original "
                        "behavior, slow: 60*N_starts merger solves). Default: best "
                        "start per spec only (~60 solves), which is all every "
                        "downstream report uses")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_product:
        seed_dir = args.seed_dir or Path("output/multiple_specs/seed_0")
        run_per_product(seed_dir, args.mode)
    elif args.sweep:
        seed_dir = args.seed_dir or Path("output/multiple_specs/seed_0")
        run_sweep(seed_dir, args.mode, all_starts=args.all_starts)
    else:
        seed_dir = args.seed_dir or Path("output/unique_spec/seed_0")
        run_single(seed_dir)


if __name__ == "__main__":
    main()
