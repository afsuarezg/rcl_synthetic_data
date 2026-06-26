#!/bin/bash
#
# Aggregation + full post-estimation analysis for the cost-shifter (--grid cube)
# sweep. Run AFTER slurm/run_cost_specs.sbatch completes (afterok dependency):
#
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> slurm/aggregate_cost_specs.sbatch
#
# Pipeline (each step's inputs come from earlier steps):
#   1. run_specs --aggregate-only  -> specs_summary_{long,best}.csv (with cost_vars)
#   2. compute_elasticities         -> elasticities_detail.csv + post_estimation_summary.csv
#   3. validate_merger_prediction  -> merger_prediction_by_spec / by_product CSVs
#   4. analyze_specs / plot_specs   -> ALL NN_*.txt + NN_*.png, including the
#                                      elasticity reports 20-35 (read step 2's CSVs)
#                                      and the merger reports #37/#38 (read step 3's)
#
# ORDER MATTERS. analyze_specs/plot_specs (step 4) write reports 20-35 from the
# elasticity CSVs and 37/38 from the merger CSVs, so both producers (steps 2 and 3)
# must run BEFORE step 4 -- otherwise those reports are silently skipped on the
# first pass (they only appear on a re-submission, once the CSVs exist).
#
# Steps 2 and 3 read the per-spec start_*.pkl ProblemResults directly and are the
# expensive ones (step 2 touches EVERY pickle = 60 specs x N starts). Both are
# guarded so a failure/timeout there does NOT block step 4: analyze_specs/plot_specs
# still run and emit every report whose inputs exist -- an elasticity timeout costs
# only reports 20-35, not the recovery reports 01-15 or merger 37/38.
# compute_elasticities skips finished specs on a re-run (no --force), so
# re-submitting this job resumes it and fills in any reports missed on a timeout.

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

# Sherlock's system /usr/bin/gcc (4.8.5) ships a libstdc++ that lacks
# GLIBCXX_3.4.20, which pandas' compiled extensions require at import time.
# Load a modern GCC via Lmod so its newer libstdc++ is on the library path
# (same line as main_job.sh / postprocess.sbatch).
ml gcc/12.4.0 2>/dev/null || ml gcc/10.1.0 2>/dev/null || ml gcc

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

# 2. Elasticities + per-spec post-estimation (reads EVERY pickle; slowest step).
#    Must run BEFORE analyze_specs/plot_specs, which read these CSVs for reports
#    20-35. Guarded + resumable: a timeout doesn't block step 4 (it just loses
#    20-35); re-submitting this job resumes it (skips specs whose outputs exist).
echo "[$(date -Iseconds)] (2/4) elasticities + post-estimation (slow)"
uv run python compute_elasticities.py --root "$ROOT" \
    || echo "WARNING: compute_elasticities failed/timed out (rc=$?); re-run this job to resume"

# 3. Merger counterfactual (reads best-spec pickles; needs the rollup from step 1).
#    Produces the CSVs that analyze_specs/plot_specs turn into reports 37/38.
#    Guarded: a failure here must not block the recovery analysis in step 4.
echo "[$(date -Iseconds)] (3/4) merger validation (sweep + per-product)"
uv run python validate_merger_prediction.py --sweep       --seed-dir "$SEED_DIR" --mode iv_both \
    || echo "WARNING: validate_merger_prediction --sweep failed (rc=$?); continuing"
uv run python validate_merger_prediction.py --per-product --seed-dir "$SEED_DIR" --mode iv_both \
    || echo "WARNING: validate_merger_prediction --per-product failed (rc=$?); continuing"

# 4. Text reports + figures. Reads the rollup (step 1), the elasticity CSVs
#    (step 2) and the merger CSVs (step 3); emits every NN_*.txt + NN_*.png whose
#    inputs exist. Reports 20-35 appear only if step 2 succeeded, 37/38 only if
#    step 3 did -- the rest (01-15) always.
echo "[$(date -Iseconds)] (4/4) text reports + figures"
uv run python analyze_specs.py --root "$ROOT" --basin-threshold 2.0
uv run python plot_specs.py    --root "$ROOT"

echo "[$(date -Iseconds)] done"
