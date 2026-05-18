#!/bin/bash
#
# Aggregation + heatmap step. Run AFTER slurm/run_specs.sbatch completes.
#
# Usage:
#   # one-shot, depend on a finished array:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm/aggregate_specs.sbatch
#
#   # or, if you submit the array and aggregate in sequence:
#   ARRAY_ID=$(sbatch --parsable slurm/run_specs.sbatch)
#   sbatch --dependency=afterok:${ARRAY_ID} slurm/aggregate_specs.sbatch

#SBATCH --job-name=blp-specs-agg
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm/logs/aggregate_%j.out
#SBATCH --error=slurm/logs/aggregate_%j.err

set -euo pipefail
mkdir -p slurm/logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Same env setup as the array script.
uv sync --quiet

echo "[$(date -Iseconds)] aggregating spec results"
uv run python run_specs.py --seed 0 --iv-mode both --aggregate-only

echo "[$(date -Iseconds)] rendering heatmap"
uv run python viz_specs.py --seed 0 --iv-mode both

echo "[$(date -Iseconds)] done"
