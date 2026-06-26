#!/bin/bash
#
# Aggregation + full post-estimation analysis for the cost-shifter (--grid cube)
# sweep. Run AFTER slurm/run_cost_specs.sbatch completes (afterok dependency):
#
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm/aggregate_cost_specs.sbatch
#
# Pipeline (each step's inputs come from earlier steps):
#   1. run_specs --aggregate-only  -> specs_summary_{long,best}.csv (with cost_vars)
#   2. validate_merger_prediction  -> merger_prediction_by_spec / by_product CSVs
#   3. analyze_specs / plot_specs   -> text reports + figures (incl. merger #37/#38,
#                                      which need step 2's CSVs)
#   4. compute_elasticities         -> per-spec elasticity/markup/HHI .txt + .png
#
# Steps 2 and 4 read the per-spec start_*.pkl ProblemResults directly and are the
# expensive ones (step 4 touches EVERY pickle = 60 specs x N starts). They are
# guarded so a failure/timeout there does not block the essential recovery
# outputs (steps 1+3); compute_elasticities skips finished specs on a re-run
# (no --force), so re-submitting this job resumes it.

#SBATCH --job-name=cost-specs-agg
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/logs/cost_aggregate_%j.out
#SBATCH --error=slurm/logs/cost_aggregate_%j.err

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

uv sync --quiet

SEED_DIR="output/multiple_specs_cost/seed_0"
ROOT="output/multiple_specs_cost"

# 1. Rollups. --grid cube is required so aggregate() enumerates the cube spec
#    labels (with their __cost-... suffixes); the default full grid finds none.
echo "[$(date -Iseconds)] (1/4) aggregating cube spec results"
uv run python run_specs.py \
    --grid cube \
    --output-dir "$SEED_DIR" \
    --iv-mode both \
    --aggregate-only

# 2. Merger counterfactual (reads best-spec pickles; needs the rollup from step 1).
#    Guarded: a failure here must not block the recovery analysis in step 3.
echo "[$(date -Iseconds)] (2/4) merger validation (sweep + per-product)"
uv run python validate_merger_prediction.py --sweep       --seed-dir "$SEED_DIR" --mode iv_both \
    || echo "WARNING: validate_merger_prediction --sweep failed (rc=$?); continuing"
uv run python validate_merger_prediction.py --per-product --seed-dir "$SEED_DIR" --mode iv_both \
    || echo "WARNING: validate_merger_prediction --per-product failed (rc=$?); continuing"

# 3. Essential recovery analysis (CSV-only; reads the rollup + any merger CSVs).
echo "[$(date -Iseconds)] (3/4) text reports + figures"
uv run python analyze_specs.py --root "$ROOT" --basin-threshold 2.0
uv run python plot_specs.py    --root "$ROOT"

# 4. Elasticities + per-spec merger sim (reads EVERY pickle; slowest). Guarded and
#    resumable: re-run this job to continue (it skips specs whose outputs exist).
echo "[$(date -Iseconds)] (4/4) elasticities + merger sim (slow)"
uv run python compute_elasticities.py --root "$ROOT" \
    || echo "WARNING: compute_elasticities failed/timed out (rc=$?); re-run this job to resume"

echo "[$(date -Iseconds)] done"
