# Running the BLP specification sweep on Sherlock

End-to-end recipe for running `run_specs.py` (60-spec sweep) as a SLURM
array job on Stanford's Sherlock cluster. Replace `YOURSUNETID` with your
own Stanford ID throughout.

## 1. Put the repo on Sherlock

SSH in:

```bash
ssh YOURSUNETID@login.sherlock.stanford.edu
```

Either clone (preferred):

```bash
cd $HOME
git clone <your-git-remote-url> rcl_synthetic_data
cd rcl_synthetic_data
```

…or `scp` from your laptop:

```bash
# run on your laptop, not on Sherlock
scp -r "/Users/andres/Documents/Mergers RCL/rcl_synthetic_data" \
       YOURSUNETID@login.sherlock.stanford.edu:~/
```

`$HOME` (≈ 15 GB quota) is fine. If you expect to keep many seeds, use
`$OAK/<group>/users/YOURSUNETID/` instead.

## 2. Install `uv` and materialise the venv

`uv` is not pre-installed but installs to `$HOME/.local/bin` without admin:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env                       # adds uv to $PATH this session
echo 'source $HOME/.local/bin/env' >> ~/.bashrc   # persist for future logins
```

Materialise the venv from `pyproject.toml` + `uv.lock`:

```bash
cd $HOME/rcl_synthetic_data
uv sync
```

First run ≈ 2 min (downloads pyblp 1.2.0, numpy<2, scipy<1.14, matplotlib,
…). Subsequent runs are instant.

## 3. Get the simulated dataset onto Sherlock

The sweep reads `output/seed_0/{product_data.csv, agent_data.csv, truth.pkl}`,
which are `.gitignore`'d. Two options:

**(a) Re-simulate on Sherlock** — deterministic, byte-identical:

```bash
uv run python simulate.py --seed 0
```

Takes ~30 s.

**(b) Copy from your laptop:**

```bash
# run on your laptop
scp "/Users/andres/Documents/Mergers RCL/rcl_synthetic_data/output/seed_0/product_data.csv" \
    "/Users/andres/Documents/Mergers RCL/rcl_synthetic_data/output/seed_0/agent_data.csv" \
    "/Users/andres/Documents/Mergers RCL/rcl_synthetic_data/output/seed_0/truth.pkl" \
    YOURSUNETID@login.sherlock.stanford.edu:~/rcl_synthetic_data/output/seed_0/
```

Either works. (a) is simpler.

## 4. Sanity-check on the login node

Run **one spec at one start** (~90 s) before queueing the array, to confirm
the environment is sane:

```bash
uv run python run_specs.py --seed 0 --iv-mode both --n-starts 1 --spec-index 0
```

If it prints `best of 1 starts: objective = …` and writes
`output/seed_0/iv_both/specs/spec_x2-x1_x2_x3__demos-income/`, you're good.

**Do not** run anything heavier on the login node — Sherlock kills processes
that exceed login-node limits (~15 min wallclock, 4 GB RAM, low CPU). For
anything bigger, use `sbatch` or `srun --pty bash`.

## 5. (Optional) Tune the sbatch scripts for your account

Open `slurm/run_specs.sbatch` and add a `#SBATCH --partition=…` line for
your group's queue. List your accessible partitions with:

```bash
sh_part
```

Typical choices:

| partition | notes |
|---|---|
| `normal` | Stanford-wide pool, fair-share scheduling. Default if you do nothing. |
| `<owners>` | Your PI's `owners` partition (if any) — short queue but preemptable. |

If `uv` isn't on `$PATH` in non-login shells, also uncomment the
`source $HOME/.local/bin/env` line near the top of the sbatch script (the
template already documents this).

## 6. Submit the array job

Dispatches 60 independent jobs (one per spec), each requesting 4 CPUs, 8 GB
RAM, 2 h wallclock. Captures the job id in a shell variable so we can chain:

```bash
ARRAY_JOB_ID=$(sbatch --parsable slurm/run_specs.sbatch)
echo "array job: $ARRAY_JOB_ID"
```

Queue the aggregation step with `--dependency=afterok` so it only runs
when the array completes successfully:

```bash
sbatch --dependency=afterok:${ARRAY_JOB_ID} slurm/aggregate_specs.sbatch
```

The whole array typically finishes within the wall time of the slowest single
spec (≈ 25–30 min on average Sherlock hardware) once the scheduler has
allocated all tasks.

## 7. Monitor

```bash
squeue --me                                           # everything pending / running
squeue --me -t RUNNING                                # only running
sacct -j $ARRAY_JOB_ID                                # status per array task
sacct -j $ARRAY_JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Watch a single task's stdout in real time:

```bash
tail -f slurm/logs/spec_${ARRAY_JOB_ID}_0.out         # array task index 0
```

Kill the whole array if something looks wrong:

```bash
scancel $ARRAY_JOB_ID
```

## 8. After it finishes — outputs

Aggregation writes:

```
output/seed_0/iv_both/specs/
  spec_<label>/                          # 60 of these
    spec.json
    estimates/start_00.pkl … start_04.pkl
    estimates_summary.csv
  specs_summary_long.csv                 # 60 specs × 5 starts × variable params
  specs_summary_best.csv                 # one row per (spec, param) using best start
  specs_heatmap.png / .svg               # recovery-error heatmap
```

Pull everything back to your laptop:

```bash
# run on your laptop
rsync -avz YOURSUNETID@login.sherlock.stanford.edu:~/rcl_synthetic_data/output/seed_0/iv_both/specs/ \
           "/Users/andres/Documents/Mergers RCL/rcl_synthetic_data/output/seed_0/iv_both/specs/"
```

Or just the small rollup files (no pickles):

```bash
rsync -avz \
  YOURSUNETID@login.sherlock.stanford.edu:'~/rcl_synthetic_data/output/seed_0/iv_both/specs/specs_*' \
  "/Users/andres/Documents/Mergers RCL/rcl_synthetic_data/output/seed_0/iv_both/specs/"
```

## Common gotchas

- **Job pending forever.** `sacct -j $ARRAY_JOB_ID -X --format=JobID,State,Reason`
  shows why. `Reason=ReqNodeNotAvail` on an `owners` partition usually means
  the partition is full — fall back to `normal`.
- **`uv: command not found` inside the job.** `sbatch` runs a non-login
  shell, so `.bashrc` may not source `uv`'s env. Uncomment
  `source $HOME/.local/bin/env` near the top of the sbatch script.
- **Out-of-memory.** Bump `#SBATCH --mem=8G` to `16G`. pyblp peaks at
  ~3 GB for this problem size, so 8 GB has headroom; bump only if you see
  `OOMKilled`.
- **One task fails, others succeed.** Find it with
  `sacct -j $ARRAY_JOB_ID --format=JobID,State,ExitCode | grep -v COMPLETED`,
  then resubmit only the failed indices:
  ```bash
  sbatch --array=14,37 slurm/run_specs.sbatch
  ```
  The resume logic in both `run_specs.py` (skips specs whose summary CSV
  exists) and `estimate.py` (skips per-start pickles that exist) means
  re-running is safe — successful work is reused, only the missing pieces
  are recomputed.
- **Want to add another seed.** Simulate it first
  (`uv run python simulate.py --seed N`), then edit `--seed 0` → `--seed N`
  in both sbatch scripts and re-submit. Outputs land under
  `output/seed_N/iv_both/specs/`.
