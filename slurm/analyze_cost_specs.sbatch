#!/bin/bash
#
# Reports-and-figures-only re-run for the cost-shifter (--grid cube) sweep.
# A trimmed slurm/aggregate_cost_specs.sbatch that runs ONLY step 4
# (analyze_specs + plot_specs) and skips the expensive producers (aggregation,
# elasticities, merger validation).
#
# Use this to regenerate the NN_*.txt reports and NN_*.png figures after editing
# analyze_specs.py / plot_specs.py, WITHOUT recomputing the upstream CSVs:
#
#   sbatch slurm/analyze_cost_specs.sbatch
#
# PREREQUISITE: a prior slurm/aggregate_cost_specs.sbatch run must have already
# produced the inputs this step reads (all under output/multiple_specs_cost/):
#   - specs_summary_{long,best}.csv          (step 1, run_specs --aggregate-only)
#   - elasticities_detail.csv / post_estimation_summary.csv  (step 2)  -> reports 20-35
#   - merger_prediction_by_spec / by_product CSVs            (step 3)  -> reports 37-41
# Reports whose inputs are missing are silently skipped (01-15 always emit). If a
# CSV is stale/absent, re-run the full slurm/aggregate_cost_specs.sbatch instead.
#
# This is read-heavy but light on CPU/mem (no pickle scans) -> short time + low mem.

#SBATCH --job-name=cost-specs-analyze
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/logs/cost_analyze_%j.out
#SBATCH --error=slurm/logs/cost_analyze_%j.err

set -euo pipefail
mkdir -p slurm/logs

# Move to the repo root. Defaults to the OAK clone (as in main_job.sh /
# postprocess.sbatch); override with --export=ALL,PROJECT_DIR=/path/to/repo.
PROJECT_DIR="${PROJECT_DIR:-/oak/stanford/groups/polinsky/rcl_synthetic_data}"
cd "$PROJECT_DIR" || { echo "ERROR: PROJECT_DIR=$PROJECT_DIR not found" >&2; exit 1; }

# Sync code so each submission runs the intended branch, not "whatever the OAK
# clone was left on" -- no manual git pull after pushing local changes. checkout
# -f -B (not `git pull`) makes the result independent of the previously checked-
# out branch (pull only fast-forwards the CURRENT branch; if the clone sits on
# another branch the working tree never moves -> silent stale run). Destructive
# to *tracked* files only; output/ is gitignored, so generated data + resume
# state are untouched. sbatch runs a submit-time snapshot, so overwriting this
# script mid-run is safe. Override the ref with --export=ALL,GIT_REF=main; skip
# syncing (run hand-edited on-disk code) with --export=ALL,GIT_PULL=0.
GIT_REF="${GIT_REF:-add-solve-diagnostics}"
if [ "${GIT_PULL:-1}" != "0" ]; then
    echo "[$(date -Iseconds)] syncing $PROJECT_DIR -> origin/$GIT_REF"
    git fetch origin --quiet --prune
    if ! git rev-parse --verify --quiet "origin/$GIT_REF" >/dev/null; then
        echo "ERROR: origin/$GIT_REF not found after fetch." >&2
        echo "       Set GIT_REF to a branch that has been pushed to origin." >&2
        exit 1
    fi
    git checkout -f -B "$GIT_REF" "origin/$GIT_REF"
    echo "[$(date -Iseconds)] now at $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"
fi

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

ROOT="output/multiple_specs_cost"

# Text reports + figures only. Reads the rollup, elasticity, and merger CSVs
# produced by a prior aggregate run; emits every NN_*.txt + NN_*.png whose inputs
# exist. Reports 20-35 require step 2's CSVs, 37-41 require step 3's; the rest
# (01-15) always emit.
echo "[$(date -Iseconds)] (1/1) text reports + figures (analysis-only re-run)"
uv run python analyze_specs.py --root "$ROOT" --basin-threshold 2.0
uv run python plot_specs.py    --root "$ROOT"

echo "[$(date -Iseconds)] done"
