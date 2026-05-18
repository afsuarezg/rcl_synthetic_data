#!/bin/bash
#
# Single-job submission for the 60-spec BLP sweep on Sherlock.
#
# Modeled on the rcl-market-assessment slurm/example_main_job.sh template.
# Runs all 60 specs serially in one SLURM job (≈ 5 h wallclock). For a much
# faster parallel run, use the array form instead:
#
#     sbatch slurm/run_specs.sbatch     # 60-task array, ≈ 30 min wall
#
# Submit:
#
#     # 1) edit REPO_URL below
#     # 2) cd into a directory where slurm/logs/ can be created (or run from $HOME)
#     sbatch slurm/main_job.sh

#SBATCH --job-name=blp-specs-main
#SBATCH --output=slurm/logs/blp_specs.%j.out
#SBATCH --error=slurm/logs/blp_specs.%j.err
#SBATCH --time=24:00:00
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=16GB

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration. Edit REPO_URL before first use.
# PROJECT_DIR follows the example_main_job.sh convention: live in $SCRATCH.
# Override either on the sbatch command line, e.g.:
#     sbatch --export=REPO_URL=git@github.com:user/rcl_synthetic_data.git slurm/main_job.sh
# ---------------------------------------------------------------------------
REPO_URL="${REPO_URL:-<set-this-to-your-git-remote-url>}"
PROJECT_DIR="${PROJECT_DIR:-$SCRATCH/rcl_synthetic_data}"

# ---------------------------------------------------------------------------
# Fresh checkout. Comment the `rm -rf` line if you want to keep prior outputs
# under $PROJECT_DIR/output/ and just refresh the code in place.
# ---------------------------------------------------------------------------
rm -rf "$PROJECT_DIR"
git clone "$REPO_URL" "$PROJECT_DIR"
cd "$PROJECT_DIR"
mkdir -p slurm/logs

# ---------------------------------------------------------------------------
# uv: self-bootstrap if missing, then materialise .venv/ from uv.lock.
# Avoids needing `ml python/...` on Sherlock.
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[$(date -Iseconds)] installing uv into \$HOME/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    source "$HOME/.local/bin/env"
fi
uv sync --quiet

# ---------------------------------------------------------------------------
# 1) Generate the synthetic dataset if not already on disk. Deterministic
#    given --seed, so this is idempotent / skip-on-resume.
# ---------------------------------------------------------------------------
if [ ! -f output/seed_0/truth.pkl ]; then
    echo "[$(date -Iseconds)] simulating seed_0"
    uv run python simulate.py --seed 0
else
    echo "[$(date -Iseconds)] seed_0 dataset already present, skipping simulate.py"
fi

# ---------------------------------------------------------------------------
# 2) Run the 60-spec sweep + aggregate the rollup CSVs in one call.
#    Each spec writes its own subdir under
#    output/seed_0/iv_both/specs/spec_<label>/ and is resume-friendly
#    (any spec with an existing estimates_summary.csv is skipped).
# ---------------------------------------------------------------------------
echo "[$(date -Iseconds)] running 60-spec sweep (iv_both, 5 starts each)"
uv run python run_specs.py --seed 0 --iv-mode both --n-starts 5

# ---------------------------------------------------------------------------
# 3) Render the recovery-error heatmap.
# ---------------------------------------------------------------------------
echo "[$(date -Iseconds)] rendering heatmap"
uv run python viz_specs.py --seed 0 --iv-mode both

echo "[$(date -Iseconds)] done."
echo "outputs:  $PROJECT_DIR/output/seed_0/iv_both/specs/"
echo "rsync from your laptop with:"
echo "  rsync -avz <SUNETID>@login.sherlock.stanford.edu:'$PROJECT_DIR/output/seed_0/iv_both/specs/' \\"
echo "             ./output/seed_0/iv_both/specs/"
