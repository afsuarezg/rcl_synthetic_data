"""Driver for the BLP specification sweep on seed_0 (iv_both).

Enumerates a grid of (X2 chars × demographics) specifications, dispatches each
to estimate.py via subprocess, and aggregates per-spec outputs into shared
CSVs under output/seed_N/iv_<mode>/specs/.

Designed for both a SLURM job array on Sherlock and local sanity tests:

  # list the 60 specs (one per line, with their index)
  uv run python run_specs.py --list

  # run one spec by its grid index (this is what each SLURM array task calls)
  uv run python run_specs.py --seed 0 --spec-index 0  --n-starts 5

  # run a slice locally (e.g. for a small sanity check)
  uv run python run_specs.py --seed 0 --indices 0,5  --n-starts 2

  # aggregate finished spec dirs into specs_summary_{long,best}.csv
  uv run python run_specs.py --seed 0 --aggregate-only

Resume is count-aware: a spec is skipped only when it already has at least
--n-starts solved starts (start_NN.pkl on disk) AND its estimates_summary.csv
exists. Otherwise it is handed to estimate.py, which resumes the existing
pickles and runs only the missing starts, then rewrites the summary. So
re-running with a higher --n-starts just tops each spec up — no need to delete
summaries first.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import subprocess
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Grid definition — must match the plan (4 X2 × 15 demos = 60 specs).
# ---------------------------------------------------------------------------
X2_SUBSETS: list[list[str]] = [
    ["x1", "x2", "x3"],   # truth
    ["x2", "x3"],         # drop x1
    ["x1", "x3"],         # drop x2
    ["x1", "x2"],         # drop x3
]

_DEMOS = ["income", "age", "hh_size", "education"]
DEMO_SUBSETS: list[list[str]] = [
    list(s) for r in range(1, len(_DEMOS) + 1) for s in itertools.combinations(_DEMOS, r)
]
# Total = 4 * 15 = 60 (the historical --grid full).

# All/all-minus-one subsets for the --grid cube experiment: the full set plus
# every leave-one-out. Keeps the experiment balanced across the three axes
# instead of exploding the demographic dimension.
DEMO_SUBSETS_AMO: list[list[str]] = (
    [list(_DEMOS)] + [[d for d in _DEMOS if d != drop] for drop in _DEMOS]
)  # 1 + 4 = 5

_COSTS = ["w1", "w2"]
COST_SUBSETS_AMO: list[list[str]] = (
    [list(_COSTS)] + [[c for c in _COSTS if c != drop] for drop in _COSTS]
)  # 1 + 2 = 3:  {w1,w2}, {w2}, {w1}


def build_specs(grid: str) -> list:
    """Enumerate the spec grid as (x2_vars, demo_vars, cost_vars) triples.

    full -- 4 X2 x 15 demos = 60 demand-only specs (cost_vars=None: supply formula
            stays at truth, no --cost-vars passed). Identical to the historical
            grid, so SLURM array indices are preserved.
    cube -- 4 X2 x 5 demos x 3 cost = 60 specs, all/all-minus-one on every axis.
    """
    if grid == "full":
        return [(x2, demos, None) for x2 in X2_SUBSETS for demos in DEMO_SUBSETS]
    if grid == "cube":
        return [(x2, demos, cost)
                for x2 in X2_SUBSETS
                for demos in DEMO_SUBSETS_AMO
                for cost in COST_SUBSETS_AMO]
    raise ValueError(f"unknown grid {grid!r}")


def spec_label(x2: list[str], demos: list[str], cost: list[str] | None = None) -> str:
    label = f"x2-{'_'.join(x2)}__demos-{'_'.join(demos)}"
    if cost is not None:
        label += f"__cost-{'_'.join(cost)}"
    return label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iv-mode", choices=["both", "diff_only"], default="both")
    p.add_argument("--grid", choices=["full", "cube"], default="full",
                   help="full = 4 X2 x 15 demos (historical, demand-only); "
                        "cube = 4 X2 x 5 demos x 3 cost-shifter subsets "
                        "(all/all-minus-one on every axis = 60 specs).")
    p.add_argument("--n-starts", type=int, default=5,
                   help="multistarts per spec (passed through to estimate.py)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="default: output/seed_{seed}/")

    # Dispatch modes (exactly one of these or default-run-all).
    p.add_argument("--list", action="store_true",
                   help="print the index→spec table and exit")
    p.add_argument("--spec-index", type=int, default=None,
                   help="run only the spec at this 0-based grid index")
    p.add_argument("--indices", type=str, default=None,
                   help="comma-separated grid indices to run (e.g. '0,5,12')")
    p.add_argument("--aggregate-only", action="store_true",
                   help="skip estimation; just build specs_summary_*.csv from "
                        "existing spec subdirs")

    p.add_argument("--python", type=str, default=None,
                   help="python interpreter for the estimate.py subprocess "
                        "(default: same as the one running this script)")
    p.add_argument("--no-uv", action="store_true",
                   help="invoke estimate.py via plain python (default uses "
                        "`uv run python` so deps resolve)")
    return p.parse_args()


def seed_dir(args: argparse.Namespace) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return args.output_dir or os.path.join(here, "output", f"seed_{args.seed}")


def spec_dir(args: argparse.Namespace, label: str) -> str:
    return os.path.join(seed_dir(args), f"iv_{args.iv_mode}", "specs",
                        f"spec_{label}")


def cmd_list(args: argparse.Namespace) -> int:
    specs = build_specs(args.grid)
    print(f"{'idx':>3}  {'X2':<14}  {'demos':<34}  cost")
    print(f"{'-' * 3:>3}  {'-' * 14:<14}  {'-' * 34:<34}  {'-' * 8}")
    for idx, (x2, demos, cost) in enumerate(specs):
        cost_s = ','.join(cost) if cost is not None else '(truth)'
        print(f"{idx:>3}  {','.join(x2):<14}  {','.join(demos):<34}  {cost_s}")
    print(f"\nTotal: {len(specs)} specs  (grid={args.grid})")
    return 0


def dispatch_one(args: argparse.Namespace, idx: int) -> int:
    specs = build_specs(args.grid)
    if idx < 0 or idx >= len(specs):
        print(f"spec-index {idx} out of range [0, {len(specs)})", file=sys.stderr)
        return 2
    x2, demos, cost = specs[idx]
    label = spec_label(x2, demos, cost)
    sdir = spec_dir(args, label)
    summary_path = os.path.join(sdir, "estimates_summary.csv")
    # Count-aware resume: per-start pickles are estimate.py's resume source (a
    # failed start writes none), so this is exactly the set it will skip-resume.
    # Fast-skip only when the spec already has >= n_starts solved AND its summary
    # is on disk; otherwise hand off to estimate.py, which resumes existing
    # pickles and runs only the missing indices, then (re)writes the summary.
    n_done = len(glob.glob(os.path.join(sdir, "estimates", "start_*.pkl")))
    if n_done >= args.n_starts and os.path.exists(summary_path):
        print(f"[skip] spec {idx} ({label}): {n_done} starts solved "
              f">= n_starts={args.n_starts}, summary present")
        return 0
    if n_done >= args.n_starts:
        # Enough starts solved but the summary is missing: estimate.py reloads
        # the pickles (no re-solve) and rewrites estimates_summary.csv.
        print(f"[resume] spec {idx} ({label}): {n_done} starts on disk, "
              f"rebuilding missing summary")
    elif n_done:
        print(f"[resume] spec {idx} ({label}): {n_done} starts on disk, "
              f"topping up to {args.n_starts}")

    here = os.path.dirname(os.path.abspath(__file__))
    cmd: list[str]
    if args.no_uv:
        py = args.python or sys.executable
        cmd = [py, os.path.join(here, "estimate.py")]
    else:
        cmd = ["uv", "run", "python", os.path.join(here, "estimate.py")]
    cmd += [
        "--output-dir", seed_dir(args),
        "--iv-mode", args.iv_mode,
        "--n-starts", str(args.n_starts),
        "--x2-vars", ",".join(x2),
        "--demos-vars", ",".join(demos),
        "--spec-label", label,
    ]
    if cost is not None:
        cmd += ["--cost-vars", ",".join(cost)]
    print(f"[run ] spec {idx}/{len(specs) - 1}  {label}")
    print("       $ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=here)
    return res.returncode


def aggregate(args: argparse.Namespace) -> int:
    """Concatenate every spec's estimates_summary.csv into specs_summary_long.csv,
    then derive specs_summary_best.csv (best converged start per spec)."""
    iv_dir = os.path.join(seed_dir(args), f"iv_{args.iv_mode}")
    specs_root = os.path.join(iv_dir, "specs")
    if not os.path.isdir(specs_root):
        print(f"no specs dir at {specs_root}", file=sys.stderr)
        return 1

    rows: list[pd.DataFrame] = []
    n_found = 0
    n_missing = 0
    for idx, (x2, demos, cost) in enumerate(build_specs(args.grid)):
        label = spec_label(x2, demos, cost)
        sdir = os.path.join(specs_root, f"spec_{label}")
        csv = os.path.join(sdir, "estimates_summary.csv")
        if not os.path.exists(csv):
            n_missing += 1
            continue
        df = pd.read_csv(csv)
        df.insert(0, "spec_idx", idx)
        df.insert(1, "spec_label", label)
        df.insert(2, "x2_vars", ",".join(x2))
        df.insert(3, "demo_vars", ",".join(demos))
        # cost_vars: 'w1,w2' (truth) for the demand-only grid where cost is None.
        df.insert(4, "cost_vars", ",".join(cost) if cost is not None else "w1,w2")
        rows.append(df)
        n_found += 1
    if not rows:
        print(f"no spec results found under {specs_root}", file=sys.stderr)
        return 1

    long_df = pd.concat(rows, ignore_index=True)
    long_path = os.path.join(specs_root, "specs_summary_long.csv")
    long_df.to_csv(long_path, index=False)
    print(f"wrote {long_path}  ({len(long_df)} rows; "
          f"{n_found} specs, {n_missing} missing)")

    # specs_summary_best.csv: pick lowest-objective converged start per spec,
    # then keep one row per (spec, param_name).
    conv = long_df[long_df["converged"] & long_df["objective"].notna()].copy()
    if conv.empty:
        print("no converged starts to summarize")
        return 0
    # For each spec, the best start_id is argmin objective. Use a join.
    best_starts = (conv.groupby("spec_label")["objective"]
                       .idxmin()
                       .map(lambda i: conv.loc[i, "start_id"])
                       .rename("best_start_id"))
    best_df = (conv.merge(best_starts, left_on="spec_label", right_index=True)
                   .query("start_id == best_start_id")
                   .drop(columns=["best_start_id"]))
    best_path = os.path.join(specs_root, "specs_summary_best.csv")
    best_df.to_csv(best_path, index=False)
    print(f"wrote {best_path}  ({len(best_df)} rows)")

    # Quick ranking.
    rank = (best_df.groupby("spec_label")["abs_error"]
                   .agg(["mean", "max", "count"])
                   .sort_values("mean"))
    print("\nTop 5 specs by mean |est − truth|:")
    print(rank.head(5).to_string())
    print("\nBottom 5 specs by mean |est − truth|:")
    print(rank.tail(5).to_string())
    return 0


def main() -> int:
    args = parse_args()
    if args.list:
        return cmd_list(args)
    if args.aggregate_only:
        return aggregate(args)
    if args.spec_index is not None:
        return dispatch_one(args, args.spec_index)
    if args.indices is not None:
        indices = [int(s) for s in args.indices.split(",") if s.strip()]
        rc = 0
        for idx in indices:
            r = dispatch_one(args, idx)
            if r != 0 and rc == 0:
                rc = r
        # Aggregate at the end so the user gets the rollup even from a slice.
        aggregate(args)
        return rc

    # Default: run every spec sequentially, then aggregate.
    rc = 0
    for idx in range(len(build_specs(args.grid))):
        r = dispatch_one(args, idx)
        if r != 0 and rc == 0:
            rc = r
    aggregate(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
