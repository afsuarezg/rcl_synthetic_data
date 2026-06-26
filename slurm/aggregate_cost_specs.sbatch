#!/bin/bash
#
# Aggregation + analysis for the cost-shifter (--grid cube) sweep.
# Run AFTER slurm/run_cost_specs.sbatch completes (afterok dependency):
#
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm/aggregate_cost_specs.sbatch
#
# Builds the rollup CSVs (specs_summary_long.csv carries a cost_vars column) and
# renders the cross-spec text reports + figures under output/multiple_specs_cost/.

#SBATCH --job-name=cost-specs-agg
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm/logs/cost_aggregate_%j.out
#SBATCH --error=slurm/logs/cost_aggregate_%j.err

set -euo pipefail
mkdir -p slurm/logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

uv sync --quiet

# --grid cube is required so aggregate() enumerates the cube spec labels (with
# their __cost-... suffixes); the default full grid would find none of them.
echo "[$(date -Iseconds)] aggregating cube spec results"
uv run python run_specs.py \
    --grid cube \
    --output-dir output/multiple_specs_cost/seed_0 \
    --iv-mode both \
    --aggregate-only

echo "[$(date -Iseconds)] analysis + plots"
uv run python analyze_specs.py --root output/multiple_specs_cost --basin-threshold 2.0
uv run python plot_specs.py    --root output/multiple_specs_cost

# Optional (slow — reconstructs per-spec DGP simulations for elasticities/merger):
# uv run python compute_elasticities.py --root output/multiple_specs_cost

echo "[$(date -Iseconds)] done"
