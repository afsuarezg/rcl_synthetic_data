#!/bin/bash
#
# SLURM array script for the cost-shifter (--grid cube) BLP sweep on Sherlock.
#
# 60 array tasks, one per cube spec (4 X2 x 5 demos x 3 cost-shifter subsets).
# Each task runs N_STARTS optimizer starts for its spec and writes to a disjoint
# per-spec subdir under output/multiple_specs_cost/seed_0/iv_both/specs/.
#
# N_STARTS is an env override (default 5). Full submission chain:
#
#   PREP=$(sbatch --parsable slurm/prep_cost_specs.sbatch)
#   ARR=$(sbatch  --parsable --dependency=afterok:$PREP \
#                 --export=ALL,N_STARTS=50 slurm/run_cost_specs.sbatch)
#   sbatch --dependency=afterok:$ARR slurm/aggregate_cost_specs.sbatch
#
# (Drop the --export to use the default 5-start validation run.)
#
# Preconditions:
#   - PROJECT_DIR (default: the OAK clone) must be a checkout containing the
#     --grid cube code (commit 7904265 on add-solve-diagnostics). These scripts
#     run whatever code is at PROJECT_DIR — they do NOT force a git checkout.
#   - The DGP inputs must already exist (run prep_cost_specs.sbatch first).
#
# To RAISE N_STARTS on a later submission, just resubmit with the higher value —
# run_specs is count-aware: it skips a spec only when it already has >= N_STARTS
# solved start_*.pkl, otherwise it hands the spec to estimate.py, which resumes
# the existing pickles and runs only the missing starts. No need to delete the
# per-spec estimates_summary.csv first.

#SBATCH --job-name=cost-specs
#SBATCH --array=0-59
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/logs/cost_spec_%A_%a.out
#SBATCH --error=slurm/logs/cost_spec_%A_%a.err

set -euo pipefail
mkdir -p slurm/logs

# Move to the repo root. Defaults to the OAK clone (as in main_job.sh /
# postprocess.sbatch); override with --export=ALL,PROJECT_DIR=/path/to/repo.
PROJECT_DIR="${PROJECT_DIR:-/oak/stanford/groups/polinsky/rcl_synthetic_data}"
cd "$PROJECT_DIR" || { echo "ERROR: PROJECT_DIR=$PROJECT_DIR not found" >&2; exit 1; }
if [ ! -f pyproject.toml ]; then
    echo "ERROR: no pyproject.toml in $PWD — set PROJECT_DIR to the repo root." >&2
    exit 1
fi

# Sherlock's system /usr/bin/gcc (4.8.5) ships a libstdc++ that lacks
# GLIBCXX_3.4.20, which pandas' compiled extensions require at import time.
# Load a modern GCC via Lmod so its newer libstdc++ is on the library path
# (same line as main_job.sh / postprocess.sbatch).
ml gcc/12.4.0 2>/dev/null || ml gcc/10.1.0 2>/dev/null || ml gcc

# `uv sync` materialises .venv/ from pyproject.toml + uv.lock. Idempotent;
# safe to run inside every array task (no-ops after the first one).
uv sync --quiet

N_STARTS="${N_STARTS:-5}"

echo "[$(date -Iseconds)] cube spec ${SLURM_ARRAY_TASK_ID}/59 starting on $(hostname) (n_starts=$N_STARTS)"
echo "node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  mem: ${SLURM_MEM_PER_NODE}M"

uv run python run_specs.py \
    --grid cube \
    --output-dir output/multiple_specs_cost/seed_0 \
    --iv-mode both \
    --n-starts "$N_STARTS" \
    --spec-index "${SLURM_ARRAY_TASK_ID}"

echo "[$(date -Iseconds)] cube spec ${SLURM_ARRAY_TASK_ID} done"
