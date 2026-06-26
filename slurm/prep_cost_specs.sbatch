#!/bin/bash
#
# Data prep for the cost-shifter (--grid cube) BLP sweep on Sherlock.
#
# Simulates the persistent-product DGP into output/multiple_specs_cost/seed_0/
# (the SAME DGP as the local validation tree) so the array job has its inputs.
# Idempotent: skips simulation if truth.pkl already exists.
#
# It also refreshes the OAK clone to the latest committed code first (clone-or-
# update, pinned to the add-solve-diagnostics branch), so it works turnkey from
# the OnDemand UI with no manual `git pull`. Because this job runs first, the
# array + aggregate jobs that follow read the freshly-updated code.
#
# Submit this FIRST, then chain the array + aggregate on it (afterok):
#   PREP=$(sbatch --parsable slurm/prep_cost_specs.sbatch)
#   ARR=$(sbatch  --parsable --dependency=afterok:$PREP \
#                 --export=ALL,N_STARTS=50 slurm/run_cost_specs.sbatch)
#   sbatch --dependency=afterok:$ARR slurm/aggregate_cost_specs.sbatch

#SBATCH --job-name=cost-specs-prep
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/logs/cost_prep_%j.out
#SBATCH --error=slurm/logs/cost_prep_%j.err

set -euo pipefail
mkdir -p slurm/logs

# -----------------------------------------------------------------------------
# Code refresh (clone-or-update), so submitting through the OnDemand UI always
# runs the latest committed code — no manual `git pull` on the login node. This
# is the single place the shared OAK clone is updated: it runs FIRST in the
# chain, before the 60-task array, so the array tasks read fresh code without
# each having to `git reset` (which would race on the shared clone).
#
# Mirrors main_job.sh's clone-or-update, but PINS the branch carrying the
# --grid cube / persistent-product code (main does NOT have it yet). First run:
# clone that branch. Later runs: fetch + `reset --hard` to its tip. `reset
# --hard` only touches tracked files, so the untracked output/ tree (and its
# resume state) is preserved; it also discards any uncommitted edits made on the
# Sherlock copy — intentional, this is a disposable working copy. Override on
# the sbatch line, e.g.  --export=ALL,BRANCH=main  or  ...,PROJECT_DIR=/path.
# -----------------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/afsuarezg/rcl_synthetic_data.git}"
BRANCH="${BRANCH:-add-solve-diagnostics}"
PROJECT_DIR="${PROJECT_DIR:-/oak/stanford/groups/polinsky/rcl_synthetic_data}"

if [ ! -d "$PROJECT_DIR/.git" ]; then
    echo "[$(date -Iseconds)] cloning $REPO_URL ($BRANCH) into $PROJECT_DIR"
    git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
else
    cd "$PROJECT_DIR"
    echo "[$(date -Iseconds)] updating $PROJECT_DIR to origin/$BRANCH"
    git fetch --quiet origin
    git reset --hard "origin/$BRANCH"
fi
git log --oneline -1

if [ ! -f pyproject.toml ]; then
    echo "ERROR: no pyproject.toml in $PWD — set PROJECT_DIR to the repo root." >&2
    exit 1
fi

# Sherlock's system /usr/bin/gcc (4.8.5) ships a libstdc++ that lacks
# GLIBCXX_3.4.20, which pandas' compiled extensions require at import time.
# Load a modern GCC via Lmod so its newer libstdc++ is on the library path
# (same line as main_job.sh / postprocess.sbatch).
ml gcc/12.4.0 2>/dev/null || ml gcc/10.1.0 2>/dev/null || ml gcc

# `uv sync` materialises .venv/ from pyproject.toml + uv.lock. Idempotent.
uv sync --quiet

OUT="output/multiple_specs_cost/seed_0"
if [ -f "$OUT/truth.pkl" ]; then
    echo "[$(date -Iseconds)] data already present at $OUT — skipping simulate"
else
    echo "[$(date -Iseconds)] simulating persistent-product DGP into $OUT"
    uv run python simulate.py \
        --seed 0 \
        --output-dir "$OUT" \
        --shared-characteristics \
        --shared-costs \
        --market-demographics \
        --product-availability
fi

echo "[$(date -Iseconds)] prep done"
