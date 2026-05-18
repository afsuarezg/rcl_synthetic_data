#!/bin/bash
#
# SLURM array script for the 60-spec BLP sweep on Sherlock.
#
# Submit with:
#   sbatch slurm/run_specs.sbatch
#
# Each array task estimates ONE specification (by its index in run_specs.py's
# SPECS list, 0..59). Tasks are independent — they write to disjoint per-spec
# subdirs under output/seed_0/iv_both/specs/.
#
# After the array finishes successfully, submit slurm/aggregate_specs.sbatch
# with a job-dependency on this array's JobID to build the rollup CSVs.

#SBATCH --job-name=blp-specs
#SBATCH --array=0-59
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/logs/spec_%A_%a.out
#SBATCH --error=slurm/logs/spec_%A_%a.err

set -euo pipefail
mkdir -p slurm/logs

# Move to the repo root (the dir that contains run_specs.py and pyproject.toml).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Sherlock module setup. Tune to your environment — comment out anything that
# is already on $PATH for your account.
#
# Example (typical Sherlock Python user):
#   module load python/3.12
#
# uv is the preferred runner; if you don't have it on $PATH, install once with:
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#
# `uv sync` materialises .venv/ from pyproject.toml + uv.lock. Idempotent;
# safe to run inside every array task (it just no-ops after the first one).
uv sync --quiet

echo "[$(date -Iseconds)] spec ${SLURM_ARRAY_TASK_ID}/59 starting on $(hostname)"
echo "node: $(hostname)  cpus: ${SLURM_CPUS_PER_TASK}  mem: ${SLURM_MEM_PER_NODE}M"

uv run python run_specs.py \
    --seed 0 \
    --iv-mode both \
    --n-starts 5 \
    --spec-index "${SLURM_ARRAY_TASK_ID}"

echo "[$(date -Iseconds)] spec ${SLURM_ARRAY_TASK_ID} done"
