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
#   - The checkout you submit from must contain the --grid cube code
#     (commit 7904265 on add-solve-diagnostics): these scripts run whatever is
#     at $SLURM_SUBMIT_DIR — they do NOT force a git checkout.
#   - The DGP inputs must already exist (run prep_cost_specs.sbatch first).
#
# To RAISE N_STARTS on a later submission, first clear the per-spec summaries so
# run_specs re-dispatches (estimate.py then resumes existing start_*.pkl and runs
# only the new ones — same mechanism as main_job.sh):
#   find output/multiple_specs_cost/seed_0/iv_both/specs -name estimates_summary.csv -delete

#SBATCH --job-name=cost-specs
#SBATCH --array=0-59
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/logs/cost_spec_%A_%a.out
#SBATCH --error=slurm/logs/cost_spec_%A_%a.err

set -euo pipefail
mkdir -p slurm/logs

# Move to the repo root (the dir that contains run_specs.py and pyproject.toml).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

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
