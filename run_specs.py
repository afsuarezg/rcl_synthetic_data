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

Resume is automatic: any spec that already has estimates_summary.csv on disk
is skipped (matching estimate.py's per-start resume on the inside).
"""
from __future__ import annotations

import argparse
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
# Total = 4 * 15 = 60.
SPECS: list[tuple[list[str], list[str]]] = [
    (x2, demos) for x2 in X2_SUBSETS for demos in DEMO_SUBSETS
]


def spec_label(x2: list[str], demos: list[str]) -> str:
    return f"x2-{'_'.join(x2)}__demos-{'_'.join(demos)}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iv-mode", choices=["both", "diff_only"], default="both")
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


def cmd_list() -> int:
    print(f"{'idx':>3}  {'X2':<14}  demos")
    print(f"{'-' * 3:>3}  {'-' * 14:<14}  {'-' * 40}")
    for idx, (x2, demos) in enumerate(SPECS):
        print(f"{idx:>3}  {','.join(x2):<14}  {','.join(demos)}")
    print(f"\nTotal: {len(SPECS)} specs")
    return 0


def dispatch_one(args: argparse.Namespace, idx: int) -> int:
    if idx < 0 or idx >= len(SPECS):
        print(f"spec-index {idx} out of range [0, {len(SPECS)})", file=sys.stderr)
        return 2
    x2, demos = SPECS[idx]
    label = spec_label(x2, demos)
    sdir = spec_dir(args, label)
    summary_path = os.path.join(sdir, "estimates_summary.csv")
    if os.path.exists(summary_path):
        print(f"[skip] spec {idx} ({label}): summary exists at {summary_path}")
        return 0

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
    print(f"[run ] spec {idx}/{len(SPECS) - 1}  {label}")
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
    for idx, (x2, demos) in enumerate(SPECS):
        label = spec_label(x2, demos)
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
        return cmd_list()
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
    for idx in range(len(SPECS)):
        r = dispatch_one(args, idx)
        if r != 0 and rc == 0:
            rc = r
    aggregate(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
