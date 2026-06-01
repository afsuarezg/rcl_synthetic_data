#!/bin/bash
#
# =============================================================================
# Post-processing backfill: build the analysis/ and graphs/ folders on Sherlock.
# =============================================================================
#
# What this does, in one paragraph:
#   The sweep job (slurm/main_job.sh) leaves behind the per-spec estimates and
#   the rollup CSVs under
#       /oak/stanford/groups/polinsky/rcl_synthetic_data/output/seed_0/iv_both/specs/
#   but it does NOT run the three post-estimation modules. This script runs
#   them, in dependency order, against that existing OAK tree:
#       0) git pull --ff-only       -> sync code first (skip with GIT_PULL=0)
#       1) compute_elasticities.py  -> elasticities_detail.csv,
#                                       post_estimation_summary.csv (per iv) and
#                                       truth_elasticities.csv /
#                                       truth_post_estimation.csv (per seed)
#       2) analyze_specs.py         -> output/seed_0/iv_both/analysis/NN_*.txt
#       3) plot_specs.py            -> output/seed_0/iv_both/graphs/NN_*.png
#   All outputs land in place under the same OAK directory.
#
# THE ONE GOTCHA — the --root path:
#   These three scripts default --root to output/multiple_specs, but the sweep
#   on Sherlock writes to output/seed_0/... directly (no multiple_specs level).
#   discover() globs `<root>/seed_*`, so --root must point at the parent of the
#   seed_* dirs, i.e. ./output. Passing the wrong root makes the scripts find
#   nothing and exit 0 silently — see the precondition check below.
#
# How to submit (from anywhere on Sherlock):
#       sbatch slurm/postprocess.sbatch
#
#   This operates IN PLACE on the existing OAK working copy. It runs
#   `git pull --ff-only` first to sync the committed code (skippable with
#   GIT_PULL=0), but unlike main_job.sh it never clones or `git reset --hard`,
#   and the output/ tree is gitignored so the pull never touches generated data.
#   Note: a subsequent `sbatch slurm/main_job.sh` does `git reset --hard`, which
#   would discard this file if it isn't committed to the repo. Commit + push it
#   if you want it to survive the next sweep submission.
# =============================================================================

#SBATCH --job-name=blp-postproc            # name shown in `squeue`
#SBATCH --output=blp_postproc.%j.out       # stdout file (%j = SLURM job id)
#SBATCH --error=blp_postproc.%j.err        # stderr file
#SBATCH --time=04:00:00                    # max wallclock
#SBATCH -p normal                          # partition / queue
#SBATCH -c 4                               # CPU cores (single node)
#SBATCH --mem=16GB                         # total RAM

set -euo pipefail

# Where the working copy + data live on Sherlock. Same default as main_job.sh.
# Override on the command line if you put the repo elsewhere:
#     sbatch --export=PROJECT_DIR=$HOME/rcl_synthetic_data slurm/postprocess.sbatch
PROJECT_DIR="${PROJECT_DIR:-/oak/stanford/groups/polinsky/rcl_synthetic_data}"

# The seed_* parent. The scripts' default (output/multiple_specs) is WRONG for
# this layout; ./output is correct because the sweep writes output/seed_0/...
ROOT="${ROOT:-output}"

cd "$PROJECT_DIR"

# Sync the committed code so post-processing runs against the current
# plot_specs.py / analyze_specs.py, not whatever the OAK clone last had.
# --ff-only is safe: it never makes a merge commit and aborts cleanly if the
# working copy has diverged (uncommitted edits to tracked files will block it).
# output/ is gitignored, so the pull never touches generated data. Skip with
# GIT_PULL=0, e.g. to run a hand-edited local change:
#     sbatch --export=GIT_PULL=0 slurm/postprocess.sbatch
if [ "${GIT_PULL:-1}" != "0" ]; then
    echo "[$(date -Iseconds)] git pull --ff-only in $PROJECT_DIR"
    git pull --ff-only
fi

# Make the repo's flat modules (specs_io.py, ...) importable. pyproject.toml
# declares a [project], so uv treats the repo as a package and does NOT put the
# repo root on sys.path; on Sherlock `python -c "import specs_io"` and the three
# scripts below would then fail with ModuleNotFoundError (cwd isn't auto-added
# to sys.path there, unlike a default local shell). Setting PYTHONPATH fixes the
# import for the precondition check AND all three scripts.
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# uv on PATH (sbatch runs a non-login shell, so .bashrc may not have sourced it).
if ! command -v uv >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
fi

# Modern GCC so any source-build fallbacks (matplotlib's contourpy) can compile.
ml gcc/12.4.0 2>/dev/null || ml gcc/10.1.0 2>/dev/null || ml gcc

uv sync --quiet      # refresh .venv/ to match uv.lock; no-op if up to date

# -----------------------------------------------------------------------------
# PRECONDITION — prove discover() actually sees the sweep before the expensive
# elasticity recompute. An empty list means specs_summary_long.csv is missing
# (the sweep's aggregation step never ran), and all three scripts would no-op.
# Fix by re-running the sweep / aggregation first:
#     uv run python run_specs.py --seed 0 --iv-mode both --n-starts <N>
#   or  sbatch slurm/aggregate_specs.sbatch
# -----------------------------------------------------------------------------
echo "[$(date -Iseconds)] checking discover('$ROOT') ..."
FOUND=$(uv run python -c "import specs_io as s, pathlib as p; print(len(list(s.discover(p.Path('$ROOT')))))")
echo "[$(date -Iseconds)] discover() found $FOUND populated (seed,iv) sweep(s)"
if [ "$FOUND" -eq 0 ]; then
    echo "ERROR: no sweep with specs_summary_long.csv under '$ROOT'." >&2
    echo "       Run run_specs.py (or aggregate_specs.sbatch) first." >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# STEP 1 / 3 — Elasticities + merger post-estimation (prerequisite for the
# elasticity analyses/plots, 20-35). Skips work whose CSVs already exist;
# pass --force to recompute.
# -----------------------------------------------------------------------------
echo "[$(date -Iseconds)] STEP 1/3 compute_elasticities.py"
uv run python compute_elasticities.py --root "$ROOT"

# -----------------------------------------------------------------------------
# STEP 2 / 3 — Text analysis reports -> <seed>/iv_*/analysis/NN_*.txt
# -----------------------------------------------------------------------------
echo "[$(date -Iseconds)] STEP 2/3 analyze_specs.py"
uv run python analyze_specs.py --root "$ROOT"

# -----------------------------------------------------------------------------
# STEP 3 / 3 — Figures -> <seed>/iv_*/graphs/NN_*.png
# -----------------------------------------------------------------------------
echo "[$(date -Iseconds)] STEP 3/3 plot_specs.py"
uv run python plot_specs.py --root "$ROOT"

echo "[$(date -Iseconds)] done."
echo "analysis: $PROJECT_DIR/$ROOT/seed_0/iv_both/analysis/"
echo "graphs:   $PROJECT_DIR/$ROOT/seed_0/iv_both/graphs/"
echo "pull both to your laptop with:"
echo "  rsync -avz asuarezg@login.sherlock.stanford.edu:'$PROJECT_DIR/$ROOT/seed_0/iv_both/' \\"
echo "             './output/seed_0/iv_both/' --include='analysis/***' --include='graphs/***' --exclude='*'"
