#!/bin/bash
#
# =============================================================================
# Single-job submission for the 60-spec BLP sweep on Sherlock.
# =============================================================================
#
# What this script does, in one paragraph:
#   It clones the rcl_synthetic_data repo into the polinsky group's OAK space
#   (/oak/stanford/groups/polinsky/rcl_synthetic_data), sets up a Python
#   environment with uv, simulates a synthetic BLP dataset (seed 0), runs
#   60 model specifications (a grid of X2 variables × demographic controls)
#   against that dataset using both instrument sets and 5 optimizer restarts
#   per spec, then renders a recovery-error heatmap comparing each spec's
#   estimated parameters to the known ground truth. The whole pipeline
#   completes in ~5 hours of wallclock when run serially in one job.
#
# When to use this vs. the array form:
#   This file runs all 60 specs back-to-back in a single SLURM job — simpler
#   to reason about and easier to debug. For a much faster parallel run that
#   submits 60 independent tasks (each spec on its own node, ~30 min wall):
#
#       sbatch slurm/run_specs.sbatch
#
#   Use the array form once you've validated the pipeline works end-to-end.
#
# How to submit:
#   1) Edit REPO_URL below if you've forked or moved the repo.
#   2) From any directory on Sherlock (or via OnDemand's Job Composer):
#
#         sbatch slurm/main_job.sh
#
# =============================================================================

# -----------------------------------------------------------------------------
# SLURM directives. Each `#SBATCH` line is parsed by SLURM at submission time
# (NOT executed as a bash comment). They tell the scheduler what resources
# the job needs and where to write its output.
# -----------------------------------------------------------------------------
#SBATCH --job-name=blp-specs-main          # name shown in `squeue`
#SBATCH --output=blp_specs.%j.out          # stdout file (%j = SLURM job id)
#SBATCH --error=blp_specs.%j.err           # stderr file
#SBATCH --time=46:00:00                    # max wallclock; SLURM kills the job past this
#SBATCH -p normal                          # partition / queue
#SBATCH -c 4                               # CPU cores allocated (single node)
#SBATCH --mem=16GB                         # total RAM the job may use

# Bash safety flags:
#   -e            abort the script the moment any command returns non-zero
#   -u            treat references to undefined variables as errors
#   -o pipefail   a failure anywhere in a pipe propagates to the pipeline's exit code
# Without these, a failed `simulate.py` could silently let the rest of the
# pipeline run on stale or missing data.
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration.
#
# REPO_URL    — where to git-clone the project from. Edit this once if you fork.
# PROJECT_DIR — where on Sherlock the working copy lives. We use the polinsky
#               group's OAK space: it's persistent (not purged like $SCRATCH),
#               large, and shared across the group, so results survive between
#               runs and don't need to be rsync'd off immediately. OAK has
#               lower throughput than $SCRATCH, but this pipeline is
#               compute-bound rather than I/O-bound, so that's fine.
#
# Both variables use the `${VAR:-default}` syntax, which lets you override
# them on the sbatch command line without editing this file:
#
#     sbatch --export=REPO_URL=git@github.com:other/rcl_synthetic_data.git slurm/main_job.sh
#     sbatch --export=PROJECT_DIR=$SCRATCH/rcl_synthetic_data slurm/main_job.sh
# -----------------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/afsuarezg/rcl_synthetic_data.git}"
PROJECT_DIR="${PROJECT_DIR:-/oak/stanford/groups/polinsky/rcl_synthetic_data}"

# ADD_STARTS — how many *new* optimizer starts to add per spec on this
# submission. The first run with no existing starts produces ADD_STARTS starts
# per spec (default 5, matching the historical behavior). Every subsequent
# submission counts the existing `start_*.pkl` files and adds ADD_STARTS more
# on top, so resubmitting the job accumulates robustness against local minima
# rather than redoing work. Override on the sbatch command line:
#
#     sbatch --export=ADD_STARTS=10 slurm/main_job.sh
#
# Wallclock cost scales roughly linearly: ADD_STARTS=5 is ~5 h serial, the
# same as a from-scratch 5-start run. The 46 h walltime below has headroom
# for ~9 increments before you need to bump --time.
ADD_STARTS="${ADD_STARTS:-5}"

# -----------------------------------------------------------------------------
# Code refresh (clone-or-update).
#
# On the first run $PROJECT_DIR doesn't exist yet, so we `git clone`. On every
# subsequent run we `fetch` + `reset --hard origin/HEAD` to bring the tracked
# files exactly in line with the remote tip. Untracked files — notably the
# `output/` tree from previous runs — are left untouched, which is what gives
# us resume behavior: `run_specs.py` skips any spec whose
# `estimates_summary.csv` already exists.
#
# `reset --hard` will discard any uncommitted edits made directly on the
# Sherlock copy. That's intentional: this directory is a disposable working
# copy, not a place to develop.
#
# If you ever want a truly clean slate (e.g. the previous run's outputs are
# from an incompatible code version), `rm -rf "$PROJECT_DIR"` once from the
# Sherlock shell and resubmit.
# -----------------------------------------------------------------------------
if [ ! -d "$PROJECT_DIR/.git" ]; then
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
else
    cd "$PROJECT_DIR"
    git fetch --quiet origin
    git reset --hard origin/HEAD
fi

# -----------------------------------------------------------------------------
# Python environment, via uv (https://astral.sh/uv).
#
# uv is a fast Python package manager. It reads `pyproject.toml` + `uv.lock`
# from the repo and materializes a `.venv/` containing the exact dependency
# versions committed there — no `module load python/...`, no `pip install`,
# no requirements drift between machines.
#
# If uv isn't already on PATH (first run on a fresh Sherlock account), its
# installer drops a binary into ~/.local/bin and we source the env-setup
# script so this shell session can find it.
# -----------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[$(date -Iseconds)] installing uv into \$HOME/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
fi

# Sherlock's system /usr/bin/gcc is 4.8.5, which predates C++17. Load a
# modern GCC via Lmod so that any pip/uv source-build fallbacks (notably
# matplotlib's contourpy) can compile. If none of these versions exist on
# the host, run `ml avail gcc` to find a current one and edit this line.
ml gcc/12.4.0 2>/dev/null || ml gcc/10.1.0 2>/dev/null || ml gcc

uv sync --quiet      # create/refresh .venv/ to match uv.lock; no-op if up to date

# -----------------------------------------------------------------------------
# STEP 1 / 3 — Simulate the synthetic dataset.
#
# `simulate.py --seed 0` generates the ground-truth product / market /
# consumer data and the corresponding observed shares and prices,
# deterministically from the given seed. The `truth.pkl` file it writes
# holds the parameters used to generate the data; step 3 later compares
# estimated parameters back against this file to compute recovery error.
#
# This step is idempotent: if `output/seed_0/truth.pkl` already exists
# (e.g., from a prior run where you commented out the `rm -rf` above),
# we skip regeneration to save time.
# -----------------------------------------------------------------------------
if [ ! -f output/seed_0/truth.pkl ]; then
    echo "[$(date -Iseconds)] simulating seed_0"
    uv run python simulate.py --seed 0
else
    echo "[$(date -Iseconds)] seed_0 dataset already present, skipping simulate.py"
fi

# -----------------------------------------------------------------------------
# STEP 2 / 3 — The 60-spec BLP estimation sweep.
#
# `run_specs.py` enumerates the cross-product of:
#     - X2 specifications (which non-price product characteristics enter
#       the model as random coefficients), and
#     - demographic-interaction sets
# yielding 60 BLP model variants. For each variant it estimates the model
# from multiple random starting values (keeping the best fit) using both
# instrument sets simultaneously (`--iv-mode both`, i.e., differentiation
# IVs *and* BLP-style IVs), and writes per-spec results to
#     output/seed_0/iv_both/specs/spec_<label>/estimates_summary.csv
#
# Accumulating starts across re-submissions:
#   Each submission adds ADD_STARTS more optimizer starts to every spec.
#   We count the existing `start_*.pkl` files (taking the max across specs
#   so all reach the same target) and ask run_specs.py for
#   `existing + ADD_STARTS` total starts. Then we delete the per-spec
#   `estimates_summary.csv` files: that's the *only* thing standing between
#   us and the per-start resume inside estimate.py, which loads any
#   `start_<NN>.pkl` already on disk and only runs the new ones. The
#   summary CSVs are regenerated from the (old + new) pickles at the end
#   of each spec, so deleting them costs nothing.
#
#   First-time runs: EXISTING=0, so N_STARTS=ADD_STARTS — identical to the
#   historical 5-start behavior when ADD_STARTS is left at its default.
#
# After all specs finish, run_specs.py aggregates them into rollup CSVs
# next to the per-spec subdirectories.
# -----------------------------------------------------------------------------
SPECS_ROOT="output/seed_0/iv_both/specs"

if [ -d "$SPECS_ROOT" ]; then
    # Group pkls by their containing spec_<label>/ dir (everything before
    # /estimates/), count per group, take the max so every spec is brought
    # up to the same N_STARTS this run.
    EXISTING=$(find "$SPECS_ROOT" -path '*/estimates/start_*.pkl' \
               | awk -F'/estimates/' '{print $1}' | sort | uniq -c \
               | awk '{print $1}' | sort -n | tail -1)
    EXISTING="${EXISTING:-0}"
else
    EXISTING=0
fi
N_STARTS=$(( EXISTING + ADD_STARTS ))

# Clear per-spec summaries so run_specs.py's per-spec skip doesn't fire.
# Also clear the rollups so they rebuild cleanly from the new summaries.
find "$SPECS_ROOT" -name estimates_summary.csv -delete 2>/dev/null || true
rm -f "$SPECS_ROOT/specs_summary_long.csv" "$SPECS_ROOT/specs_summary_best.csv"

echo "[$(date -Iseconds)] sweep: existing=$EXISTING add=$ADD_STARTS -> n_starts=$N_STARTS"
uv run python run_specs.py --seed 0 --iv-mode both --n-starts "$N_STARTS"

# -----------------------------------------------------------------------------
# STEP 3 / 3 — Visualize recovery error.
#
# `viz_specs.py` reads the rollup from step 2, compares each spec's
# estimated parameters against the ground truth in `truth.pkl`, and renders
# a heatmap showing which (X2 × demographics) combinations recover the true
# parameters most accurately. The output PNG lives under
#     output/seed_0/iv_both/specs/
# -----------------------------------------------------------------------------
echo "[$(date -Iseconds)] rendering heatmap"
uv run python viz_specs.py --seed 0 --iv-mode both

# -----------------------------------------------------------------------------
# Done. Print where outputs live and a copy-paste-ready rsync command for
# pulling a local copy back to your laptop. Outputs persist on OAK (it's not
# purged like $SCRATCH), so they're safe to leave there between runs.
# -----------------------------------------------------------------------------
echo "[$(date -Iseconds)] done."
echo "outputs:  $PROJECT_DIR/output/seed_0/iv_both/specs/"
echo "rsync from your laptop with:"
echo "  rsync -avz asuarezg@login.sherlock.stanford.edu:'$PROJECT_DIR/output/seed_0/iv_both/specs/' \\"
echo "             ./output/seed_0/iv_both/specs/"
