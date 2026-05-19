#!/bin/bash
#
# =============================================================================
# Single-job submission for the 60-spec BLP sweep on Sherlock.
# =============================================================================
#
# What this script does, in one paragraph:
#   It clones the rcl_synthetic_data repo into $SCRATCH, sets up a Python
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
#SBATCH --time=24:00:00                    # max wallclock; SLURM kills the job past this
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
# PROJECT_DIR — where on Sherlock the working copy lives. $SCRATCH is the right
#               place for compute-heavy jobs: it's high-throughput, large, and
#               purged on a rolling schedule. Don't leave anything precious
#               here long-term — rsync results back to your laptop after each run.
#
# Both variables use the `${VAR:-default}` syntax, which lets you override
# them on the sbatch command line without editing this file:
#
#     sbatch --export=REPO_URL=git@github.com:other/rcl_synthetic_data.git slurm/main_job.sh
# -----------------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/afsuarezg/rcl_synthetic_data.git}"
PROJECT_DIR="${PROJECT_DIR:-$SCRATCH/rcl_synthetic_data}"

# -----------------------------------------------------------------------------
# Fresh checkout. Every run wipes $PROJECT_DIR and re-clones from REPO_URL,
# guaranteeing the code we execute matches the current tip of the remote
# branch. Outputs from previous runs that lived under $PROJECT_DIR/output/
# are destroyed in the process.
#
# To preserve outputs across runs (e.g., to resume an interrupted sweep),
# comment out the `rm -rf` line. `run_specs.py` is itself resume-friendly:
# it skips any spec that already has an `estimates_summary.csv` on disk.
# -----------------------------------------------------------------------------
# rm -rf "$PROJECT_DIR"
git clone "$REPO_URL" "$PROJECT_DIR"
cd "$PROJECT_DIR"

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
# yielding 60 BLP model variants. For each variant it:
#     - estimates the model 5 times from different random starting values
#       (`--n-starts 5`), keeping the best fit,
#     - uses both instrument sets simultaneously (`--iv-mode both`,
#       i.e., differentiation IVs *and* BLP-style IVs), and
#     - writes per-spec results to
#         output/seed_0/iv_both/specs/spec_<label>/estimates_summary.csv
#
# The script is resume-friendly: any spec whose `estimates_summary.csv` is
# already on disk is skipped, so a re-submission picks up where a previous
# run left off (provided $PROJECT_DIR survived — see the note about `rm -rf`).
#
# After all specs finish, run_specs.py aggregates them into rollup CSVs
# next to the per-spec subdirectories.
# -----------------------------------------------------------------------------
echo "[$(date -Iseconds)] running 60-spec sweep (iv_both, 5 starts each)"
uv run python run_specs.py --seed 0 --iv-mode both --n-starts 5

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
# pulling them back to your laptop. Remember: $SCRATCH is purged on a
# rolling schedule, so don't leave results sitting there indefinitely.
# -----------------------------------------------------------------------------
echo "[$(date -Iseconds)] done."
echo "outputs:  $PROJECT_DIR/output/seed_0/iv_both/specs/"
echo "rsync from your laptop with:"
echo "  rsync -avz asuarezg@login.sherlock.stanford.edu:'$PROJECT_DIR/output/seed_0/iv_both/specs/' \\"
echo "             ./output/seed_0/iv_both/specs/"
