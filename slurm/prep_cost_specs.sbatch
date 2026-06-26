#!/bin/bash
#
# Data prep for the cost-shifter (--grid cube) BLP sweep on Sherlock.
#
# Simulates the persistent-product DGP into output/multiple_specs_cost/seed_0/
# (the SAME DGP as the local validation tree) so the array job has its inputs.
# Idempotent: skips simulation if truth.pkl already exists.
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

# Move to the repo root. Defaults to the OAK clone (as in main_job.sh /
# postprocess.sbatch); override with --export=ALL,PROJECT_DIR=/path/to/repo.
PROJECT_DIR="${PROJECT_DIR:-/oak/stanford/groups/polinsky/rcl_synthetic_data}"
cd "$PROJECT_DIR" || { echo "ERROR: PROJECT_DIR=$PROJECT_DIR not found" >&2; exit 1; }
if [ ! -f pyproject.toml ]; then
    echo "ERROR: no pyproject.toml in $PWD — set PROJECT_DIR to the repo root." >&2
    exit 1
fi

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
