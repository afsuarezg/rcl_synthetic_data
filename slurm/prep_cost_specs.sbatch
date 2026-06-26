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

# Move to the repo root (the dir that contains simulate.py and pyproject.toml).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

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
