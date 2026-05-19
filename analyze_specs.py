"""analyze_specs.py -- post-hoc analysis of synthetic-RCL multistart sweeps.

Walks `output/multiple_specs/seed_*/iv_*/specs/`, reads each
`specs_summary_long.csv`, and writes one text report per analysis to:

    output/multiple_specs/seed_X/iv_Y/analysis/NN_<name>.txt

When >=2 seeds are present for the same iv_mode, additional cross-seed
analyses are written to:

    output/multiple_specs/analysis/NN_<name>.txt

The synthetic-data context differs from the BLP/Nevo sweeps in two ways
that the analyses here lean on:

  - **Truth is known**, so analyses 08-15 compare estimate -> truth (RMSE,
    bias, false-positive pi). These have no analog in the BLP scripts.
  - **One start per spec is the truth-warm start** (initialized at the DGP
    parameter vector). It dominates by RMSE trivially, so by default it is
    excluded from multistart-discovery analyses. It is reported separately
    as a reference (the GMM objective evaluated at truth).

Usage:
    uv run python analyze_specs.py
    uv run python analyze_specs.py --root output/multiple_specs --basin-threshold 2.0
"""
from __future__ import annotations

import argparse
import contextlib
import io
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import specs_io as sio


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _hdr(title: str) -> None:
    print('=' * 80)
    print(f'  {title}')
    print('=' * 80)


def _sep(width: int = 80, char: str = '-') -> None:
    print(char * width)


def _run_and_save(out_dir: Path, filename: str, func, *args, **kwargs) -> None:
    """Call func(*args, **kwargs), tee output to stdout, and save to out_dir/filename."""
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        func(*args, **kwargs)
    text = buf.getvalue()
    print(text, end='')
    (out_dir / filename).write_text(text, encoding='utf-8')


def _safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Return (Pearson, Spearman) between two equally-sized Series. NaN on degenerate input."""
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float('nan'), float('nan')
    return float(x.corr(y)), float(x.rank().corr(y.rank()))


def _rmse(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float('nan')


# ---------------------------------------------------------------------------
# Per-(seed, iv) analyses
# ---------------------------------------------------------------------------

def objective_ranking(df: pd.DataFrame) -> None:
    """01. Specs ranked by best perturbed-start objective; truth-warm value as reference."""
    _hdr('01. Objective ranking across specifications (truth-warm = reference)')
    best = sio.best_per_spec(df)
    starts = sio.starts_table(df)

    truth_obj = dict(zip(starts[starts.is_truth_start].spec_label,
                         starts[starts.is_truth_start].objective))

    rows = []
    for spec, sid in best.items():
        sub = starts[(starts.spec_label == spec) & (starts.start_id == sid)].iloc[0]
        rows.append((spec, sid, sub.objective, bool(sub.converged), truth_obj.get(spec)))
    rows.sort(key=lambda r: r[2])

    print(f'  {"Rank":>4}  {"GMM obj":>10}  {"truth_ref":>10}  {"Start":>5}  {"Conv":>5}  Specification')
    _sep()
    for i, (spec, sid, obj, conv, tref) in enumerate(rows, 1):
        tref_s = f'{tref:>10.4f}' if tref is not None else f'{"NA":>10}'
        print(f'  {i:>4}  {obj:>10.4f}  {tref_s}  {sid:>5}  {str(conv):>5}  {spec}')
    print()
    print(f'  N specs: {len(rows)}.  "truth_ref" = GMM objective at truth-warm start.')
    print()


def multistart_stability(df: pd.DataFrame) -> None:
    """02. Spread of objective across perturbed starts per spec."""
    _hdr('02. Multistart stability -- perturbed-start objective spread per spec')
    starts = sio.starts_table(df)
    pert = starts[~starts.is_truth_start]
    g = pert.groupby('spec_label')['objective']
    summary = g.agg(['count', 'min', 'max', 'mean', 'std']).rename(columns={'count': 'n'})
    summary['spread'] = summary['max'] - summary['min']
    summary = summary.sort_values('spread', ascending=False)

    truth_obj = dict(zip(starts[starts.is_truth_start].spec_label,
                         starts[starts.is_truth_start].objective))

    print(f'  Spread = max(obj) - min(obj) across perturbed starts only.')
    print(f'  {"N":>3}  {"min":>10}  {"max":>10}  {"spread":>10}  {"truth_ref":>10}  Specification')
    _sep()
    for spec, r in summary.iterrows():
        tref = truth_obj.get(spec, float('nan'))
        print(f'  {int(r.n):>3}  {r["min"]:>10.4f}  {r["max"]:>10.4f}  '
              f'{r["spread"]:>10.4f}  {tref:>10.4f}  {spec}')
    print()


def convergence_audit(df: pd.DataFrame) -> None:
    """03. Per-start convergence flags + listing of any failures."""
    _hdr('03. Convergence audit')
    starts = sio.starts_table(df)
    n_total = len(starts)
    n_conv = int(starts.converged.sum())
    n_fail = int((~starts.converged).sum())

    print(f'  Total starts:    {n_total}')
    print(f'  Converged:       {n_conv}')
    print(f'  Non-converged:   {n_fail}')
    print()
    failing = starts[~starts.converged].sort_values(['spec_label', 'start_id'])
    if failing.empty:
        print('  No non-converged starts.')
    else:
        print('  Non-converged starts:')
        print(f'  {"Spec":<55}  {"Start":>5}  {"Tag":>12}  {"GMM obj":>10}')
        _sep()
        for _, r in failing.iterrows():
            print(f'  {r.spec_label:<55}  {int(r.start_id):>5}  '
                  f'{str(r.tag):>12}  {r.objective:>10.4f}')
    print()


def global_minimum(df: pd.DataFrame) -> None:
    """04. Global best objective across (spec, perturbed start); per-spec distance to it."""
    _hdr('04. Global minimum across all specs (perturbed starts, converged only)')
    starts = sio.starts_table(df)
    pool = starts[(~starts.is_truth_start) & starts.converged]
    if pool.empty:
        print('  No converged perturbed starts.')
        return

    best = pool.loc[pool.objective.idxmin()]
    print(f'  Global best:  spec={best.spec_label}  start={int(best.start_id)}  '
          f'obj={best.objective:.4f}')
    print()

    by_spec = pool.groupby('spec_label').objective.min().sort_values()
    print(f'  {"Rank":>4}  {"GMM obj":>10}  {"d_from_best":>12}  Specification')
    _sep()
    for i, (spec, obj) in enumerate(by_spec.items(), 1):
        print(f'  {i:>4}  {obj:>10.4f}  {obj - best.objective:>12.4f}  {spec}')
    print()


def two_basin_analysis(df: pd.DataFrame, threshold: float) -> None:
    """05. Classify perturbed starts into 'near-global' vs 'far' basins."""
    _hdr(f'05. Two-basin analysis -- threshold = {threshold} (perturbed starts only)')
    starts = sio.starts_table(df)
    pool = starts[(~starts.is_truth_start) & starts.converged]
    if pool.empty:
        print('  No converged perturbed starts.')
        return
    gmin = pool.objective.min()
    a = pool[pool.objective - gmin <= threshold]
    b = pool[pool.objective - gmin > threshold]
    for label, basin in [('A (near global)', a), ('B (far)', b)]:
        if basin.empty:
            print(f'  Basin {label}: (none)')
            continue
        spread = basin.objective.max() - basin.objective.min()
        print(f'  Basin {label}: n={len(basin)}  '
              f'obj=[{basin.objective.min():.4f}, {basin.objective.max():.4f}]  '
              f'spread={spread:.4f}  n_specs={basin.spec_label.nunique()}')
    print()


def runtime(df: pd.DataFrame) -> None:
    """06. Elapsed wall-clock seconds per (spec, start)."""
    _hdr('06. Runtime per specification (seconds)')
    starts = sio.starts_table(df)
    g = starts.groupby('spec_label')['elapsed_sec']
    summary = g.agg(['mean', 'median', 'min', 'max']).sort_values('mean', ascending=False)
    print(f'  Total wall time: {starts.elapsed_sec.sum():.1f} s')
    print()
    print(f'  {"mean":>10}  {"median":>10}  {"min":>10}  {"max":>10}  Specification')
    _sep()
    for spec, r in summary.iterrows():
        print(f'  {r["mean"]:>10.2f}  {r["median"]:>10.2f}  {r["min"]:>10.2f}  '
              f'{r["max"]:>10.2f}  {spec}')
    print()


def price_coef(df: pd.DataFrame) -> None:
    """07. Price coefficient (beta_1) across perturbed starts per spec."""
    _hdr('07. Price coefficient (beta_1) across starts')
    sub = df[df.param_name == 'beta_1']
    if sub.empty:
        print('  No beta_1 rows.')
        return
    truth = sub.truth.iloc[0]
    pert = sub[~sub.is_truth_start]
    g = pert.groupby('spec_label')['estimate']
    summary = g.agg(['mean', 'std', 'min', 'max']).sort_values('mean')
    print(f'  Truth: beta_1 = {truth:.4f}')
    print()
    print(f'  {"mean":>10}  {"std":>10}  {"min":>10}  {"max":>10}  '
          f'{"|mean-truth|":>12}  Specification')
    _sep()
    for spec, r in summary.iterrows():
        print(f'  {r["mean"]:>10.4f}  {r["std"]:>10.4f}  {r["min"]:>10.4f}  '
              f'{r["max"]:>10.4f}  {abs(r["mean"] - truth):>12.4f}  {spec}')
    print()


def recovery_rmse_by_group(df: pd.DataFrame) -> None:
    """08. RMSE/MAE by parameter group (beta/sigma/pi/gamma) for best perturbed start."""
    _hdr('08. Recovery RMSE/MAE by parameter group (best perturbed start)')
    best = sio.best_per_spec(df)
    rows = []
    for spec, sid in best.items():
        sub = df[(df.spec_label == spec) & (df.start_id == sid)]
        rec = {'spec': spec, 'total_RMSE': _rmse(sub.abs_error.values),
               'total_MAE': float(sub.abs_error.mean())}
        for grp in ('beta', 'sigma', 'pi', 'gamma'):
            g_err = sub.loc[sub.param_group == grp, 'abs_error'].values
            rec[f'{grp}_RMSE'] = _rmse(g_err)
            rec[f'{grp}_MAE'] = float(np.mean(g_err)) if g_err.size else float('nan')
        rows.append(rec)
    rows.sort(key=lambda r: r['total_RMSE'])

    cols = ['total_RMSE', 'total_MAE',
            'beta_RMSE', 'sigma_RMSE', 'pi_RMSE', 'gamma_RMSE']
    header = '  ' + '  '.join(f'{c:>10}' for c in cols) + '  Specification'
    print(header)
    _sep(len(header))
    for r in rows:
        vals = '  '.join(f'{r.get(c, float("nan")):>10.4f}' for c in cols)
        print(f'  {vals}  {r["spec"]}')
    print()


def recovery_vs_objective(df: pd.DataFrame) -> None:
    """09. Does lower GMM objective => better recovery? Correlate across specs."""
    _hdr('09. Recovery error vs. GMM objective (best perturbed start)')
    best = sio.best_per_spec(df)
    pairs = []
    for spec, sid in best.items():
        sub = df[(df.spec_label == spec) & (df.start_id == sid)]
        pairs.append({
            'spec': spec,
            'obj': float(sub.objective.iloc[0]),
            'rmse': _rmse(sub.abs_error.values),
        })
    d = pd.DataFrame(pairs).sort_values('obj')
    pearson, spearman = _safe_corr(d.obj, d.rmse)
    print(f'  N specs:               {len(d)}')
    print(f'  Pearson(obj, RMSE)  =  {pearson:.4f}')
    print(f'  Spearman(obj, RMSE) =  {spearman:.4f}')
    print()
    print(f'  {"obj":>10}  {"total_RMSE":>10}  Specification')
    _sep()
    for _, r in d.iterrows():
        print(f'  {r.obj:>10.4f}  {r.rmse:>10.4f}  {r.spec}')
    print()


def param_level_error(df: pd.DataFrame) -> None:
    """10. Per-physical-parameter |error| distribution across specs that share it."""
    _hdr('10. Per-parameter error distribution (best perturbed start, by physical name)')
    best = sio.best_per_spec(df)
    rec_rows = [df[(df.spec_label == s) & (df.start_id == sid)] for s, sid in best.items()]
    rec = pd.concat(rec_rows)
    g = rec.groupby('param_physical')['abs_error']
    summary = g.agg(['count', 'mean', 'median', 'std', 'min', 'max']).sort_values(
        'mean', ascending=False)
    print(f'  {"n":>4}  {"mean":>10}  {"median":>10}  {"std":>10}  '
          f'{"min":>10}  {"max":>10}  Parameter')
    _sep()
    for pname, r in summary.iterrows():
        std = 0.0 if pd.isna(r["std"]) else r["std"]
        print(f'  {int(r["count"]):>4}  {r["mean"]:>10.4f}  {r["median"]:>10.4f}  '
              f'{std:>10.4f}  {r["min"]:>10.4f}  {r["max"]:>10.4f}  {pname}')
    print()


def pi_zero_false_positive(df: pd.DataFrame) -> None:
    """11. For pi entries with truth=0, distribution of |estimate| -- phantom interactions."""
    _hdr('11. Pi false-positives -- |estimate| where truth = 0 (best perturbed start)')
    best = sio.best_per_spec(df)
    rows = []
    all_abs = []
    for spec, sid in best.items():
        sub = df[(df.spec_label == spec)
                 & (df.start_id == sid)
                 & (df.param_group == 'pi')
                 & (df.truth == 0.0)]
        if not sub.empty:
            absest = sub.estimate.abs()
            rows.append((spec, len(sub), float(absest.mean()), float(absest.max())))
            all_abs.extend(absest.tolist())
    rows.sort(key=lambda r: r[2], reverse=True)

    if all_abs:
        arr = np.asarray(all_abs)
        print(f'  Pooled over all specs: n={len(arr)}  '
              f'mean|est|={arr.mean():.4f}  median|est|={np.median(arr):.4f}  '
              f'p95|est|={np.percentile(arr, 95):.4f}  max|est|={arr.max():.4f}')
        print()

    print(f'  {"n_zero":>6}  {"mean|est|":>10}  {"max|est|":>10}  Specification')
    _sep()
    for spec, n, ma, mx in rows:
        print(f'  {n:>6}  {ma:>10.4f}  {mx:>10.4f}  {spec}')
    print()


def omitted_x2_bias(df: pd.DataFrame) -> None:
    """12. Recovery MAE grouped by which {x1,x2,x3} are in X2 (omitted-variable bias)."""
    _hdr('12. Recovery MAE by X2 inclusion pattern (best perturbed start)')
    best = sio.best_per_spec(df)
    rec = pd.concat([df[(df.spec_label == s) & (df.start_id == sid)]
                     for s, sid in best.items()])
    print(f'  {"X2 set":<12}  {"n_specs":>7}  {"beta_MAE":>10}  '
          f'{"sigma_MAE":>10}  {"pi_MAE":>10}  {"gamma_MAE":>10}')
    _sep()
    for x2_set, sub in rec.groupby('x2_vars'):
        ns = sub.spec_label.nunique()
        bM = sub.loc[sub.param_group == 'beta', 'abs_error'].mean()
        sM = sub.loc[sub.param_group == 'sigma', 'abs_error'].mean()
        pM = sub.loc[sub.param_group == 'pi', 'abs_error'].mean()
        gM = sub.loc[sub.param_group == 'gamma', 'abs_error'].mean()
        print(f'  {x2_set:<12}  {ns:>7}  {bM:>10.4f}  {sM:>10.4f}  '
              f'{pM:>10.4f}  {gM:>10.4f}')
    print()


def demo_overfit(df: pd.DataFrame) -> None:
    """13. Demographic cardinality vs. recovery (and phantom-pi rate)."""
    _hdr('13. Demographic cardinality vs. recovery (best perturbed start)')
    best = sio.best_per_spec(df)
    rec = pd.concat([df[(df.spec_label == s) & (df.start_id == sid)]
                     for s, sid in best.items()])
    print(f'  {"n_demos":>8}  {"n_specs":>8}  {"beta_MAE":>10}  '
          f'{"sigma_MAE":>10}  {"pi_MAE":>10}  {"pi_zero_MAE":>12}')
    _sep()
    for n, sub in rec.groupby('n_demos'):
        ns = sub.spec_label.nunique()
        bM = sub.loc[sub.param_group == 'beta', 'abs_error'].mean()
        sM = sub.loc[sub.param_group == 'sigma', 'abs_error'].mean()
        pM = sub.loc[sub.param_group == 'pi', 'abs_error'].mean()
        pZ = sub.loc[(sub.param_group == 'pi') & (sub.truth == 0),
                     'estimate'].abs().mean()
        print(f'  {int(n):>8}  {ns:>8}  {bM:>10.4f}  {sM:>10.4f}  '
              f'{pM:>10.4f}  {pZ:>12.4f}')
    print()


def param_stability_within_spec(df: pd.DataFrame) -> None:
    """14. Within-spec std of estimate across perturbed starts (per param)."""
    _hdr('14. Within-spec parameter stability across perturbed starts (std of estimate)')
    pert = df[~df.is_truth_start]
    g = pert.groupby(['spec_label', 'param_name'])['estimate']
    std = g.std().rename('std').reset_index()
    spec_mean = (std.groupby('spec_label')['std'].mean()
                 .sort_values(ascending=False))
    print(f'  Per-spec average across-start std (mean over params):')
    print(f'  {"mean_std":>10}  Specification')
    _sep()
    for spec, s in spec_mean.items():
        print(f'  {s:>10.4f}  {spec}')
    print()


def best_vs_truth_start(df: pd.DataFrame) -> None:
    """15. Best-by-objective vs. best-by-RMSE among perturbed starts (per spec)."""
    _hdr('15. Best-by-objective vs. best-by-RMSE (perturbed starts only)')
    rows = []
    for spec, sub in df[~df.is_truth_start].groupby('spec_label'):
        per_start = []
        for sid, ssub in sub.groupby('start_id'):
            obj = float(ssub.objective.iloc[0])
            rmse = _rmse(ssub.abs_error.values)
            per_start.append((int(sid), obj, rmse))
        by_obj = sorted(per_start, key=lambda r: r[1])
        by_rmse = sorted(per_start, key=lambda r: r[2])
        rows.append({
            'spec': spec,
            'obj_start': by_obj[0][0], 'obj_min': by_obj[0][1],
            'rmse_start': by_rmse[0][0], 'rmse_min': by_rmse[0][2],
            'match': by_obj[0][0] == by_rmse[0][0],
        })
    n_match = sum(r['match'] for r in rows)
    print(f'  Best-by-obj == best-by-RMSE in {n_match} / {len(rows)} specs '
          f'({100 * n_match / len(rows):.1f}%)')
    print()
    print(f'  {"obj_start":>9}  {"rmse_start":>10}  {"match":>5}  '
          f'{"obj_min":>10}  {"rmse_min":>10}  Specification')
    _sep()
    for r in sorted(rows, key=lambda x: (not x['match'], x['spec'])):
        print(f'  {r["obj_start"]:>9}  {r["rmse_start"]:>10}  '
              f'{str(r["match"]):>5}  {r["obj_min"]:>10.4f}  '
              f'{r["rmse_min"]:>10.4f}  {r["spec"]}')
    print()


# ---------------------------------------------------------------------------
# Cross-seed analyses (run when >=2 seeds for a given iv_mode)
# ---------------------------------------------------------------------------

def _stack_best(seed_dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Stack the best-start row per spec across seeds; adds a `seed` column."""
    parts = []
    for seed, df in seed_dfs.items():
        best = sio.best_per_spec(df)
        for spec, sid in best.items():
            sub = df[(df.spec_label == spec) & (df.start_id == sid)].copy()
            sub['seed'] = seed
            parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def estimate_across_seeds(seed_dfs: dict[int, pd.DataFrame]) -> None:
    """16. Mean / std of best-start estimate per (spec, param) across seeds."""
    _hdr('16. Estimate mean +/- std across seeds (best perturbed start per spec)')
    combined = _stack_best(seed_dfs)
    g = combined.groupby(['spec_label', 'param_physical', 'truth'], dropna=False)['estimate']
    summary = g.agg(['mean', 'std', 'count']).reset_index()
    summary['bias'] = summary['mean'] - summary['truth']
    big = summary[summary.bias.abs() > 0.1].sort_values('bias',
                                                       key=lambda s: s.abs(),
                                                       ascending=False)
    print(f'  N seeds: {len(seed_dfs)}.  Showing |bias| > 0.1 (top 50):')
    print(f'  {"truth":>8}  {"mean":>8}  {"bias":>8}  {"std":>8}  {"n":>3}  '
          f'{"Param":<20}  Spec')
    _sep()
    for _, r in big.head(50).iterrows():
        std = 0.0 if pd.isna(r["std"]) else r["std"]
        print(f'  {r.truth:>8.4f}  {r["mean"]:>8.4f}  {r["bias"]:>8.4f}  '
              f'{std:>8.4f}  {int(r["count"]):>3}  '
              f'{str(r.param_physical):<20}  {r.spec_label}')
    print()


def recovery_across_seeds(seed_dfs: dict[int, pd.DataFrame]) -> None:
    """17. Per-spec total RMSE: mean and std across seeds."""
    _hdr('17. Recovery RMSE per spec, across seeds')
    rows = []
    for seed, df in seed_dfs.items():
        best = sio.best_per_spec(df)
        for spec, sid in best.items():
            sub = df[(df.spec_label == spec) & (df.start_id == sid)]
            rows.append({'seed': seed, 'spec': spec,
                         'rmse': _rmse(sub.abs_error.values)})
    d = pd.DataFrame(rows)
    g = d.groupby('spec')['rmse']
    summary = g.agg(['mean', 'std', 'min', 'max', 'count']).sort_values('mean')
    print(f'  {"mean":>10}  {"std":>10}  {"min":>10}  {"max":>10}  '
          f'{"n_seeds":>7}  Specification')
    _sep()
    for spec, r in summary.iterrows():
        std = 0.0 if pd.isna(r["std"]) else r["std"]
        print(f'  {r["mean"]:>10.4f}  {std:>10.4f}  {r["min"]:>10.4f}  '
              f'{r["max"]:>10.4f}  {int(r["count"]):>7}  {spec}')
    print()


def best_spec_consistency(seed_dfs: dict[int, pd.DataFrame]) -> None:
    """18. Which spec is "best" (lowest objective) most often across seeds?"""
    _hdr('18. Best-spec consistency across seeds')
    counts: Counter[str] = Counter()
    for seed, df in seed_dfs.items():
        best = sio.best_per_spec(df)
        starts = sio.starts_table(df)
        pool = starts[(~starts.is_truth_start) & starts.converged]
        if pool.empty:
            continue
        per_spec_min = pool.groupby('spec_label').objective.min()
        winner = per_spec_min.idxmin()
        counts[winner] += 1
    print(f'  Across {len(seed_dfs)} seeds:')
    for spec, n in counts.most_common():
        print(f'    {n:>3} times: {spec}')
    print()


def price_coef_across_seeds(seed_dfs: dict[int, pd.DataFrame]) -> None:
    """19. Price coefficient (beta_1) mean and dispersion across seeds, per spec."""
    _hdr('19. Price coefficient (beta_1) across seeds')
    rows = []
    truth = None
    for seed, df in seed_dfs.items():
        sub = df[(df.param_name == 'beta_1') & (~df.is_truth_start)]
        if sub.empty:
            continue
        if truth is None:
            truth = float(sub.truth.iloc[0])
        for spec, ssub in sub.groupby('spec_label'):
            rows.append({'seed': seed, 'spec': spec,
                         'mean_within': ssub.estimate.mean(),
                         'std_within': ssub.estimate.std()})
    if not rows:
        print('  No beta_1 rows.')
        return
    d = pd.DataFrame(rows)
    agg = d.groupby('spec').agg(
        mean_over_seeds=('mean_within', 'mean'),
        sd_over_seeds=('mean_within', 'std'),
        avg_within_seed_std=('std_within', 'mean'),
        n_seeds=('mean_within', 'count'),
    ).sort_values('mean_over_seeds')
    print(f'  Truth: beta_1 = {truth}')
    print(f'  {"mean":>10}  {"sd_seeds":>10}  {"avg_within":>10}  '
          f'{"n_seeds":>7}  Specification')
    _sep()
    for spec, r in agg.iterrows():
        sd = 0.0 if pd.isna(r["sd_over_seeds"]) else r["sd_over_seeds"]
        print(f'  {r["mean_over_seeds"]:>10.4f}  {sd:>10.4f}  '
              f'{r["avg_within_seed_std"]:>10.4f}  {int(r["n_seeds"]):>7}  {spec}')
    print()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root', type=Path, default=None,
                   help='multiple_specs root (default: ./output/multiple_specs)')
    p.add_argument('--basin-threshold', type=float, default=2.0,
                   help='Two-basin classifier distance from global best (default: 2.0)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root or Path(__file__).parent / 'output' / 'multiple_specs'
    if not root.exists():
        raise SystemExit(f'Root not found: {root}')

    per_iv: dict[str, dict[int, pd.DataFrame]] = {}

    for seed, iv_mode, specs_dir in sio.discover(root):
        df = sio.load_long(specs_dir)
        out_dir = specs_dir.parent / 'analysis'
        print(f'\n### per-(seed,iv)  seed={seed}  iv={iv_mode}  '
              f'rows={len(df)}  specs={df.spec_label.nunique()}  '
              f'-> {out_dir.relative_to(root.parent)}')

        _run_and_save(out_dir, '01_objective_ranking.txt', objective_ranking, df)
        _run_and_save(out_dir, '02_multistart_stability.txt', multistart_stability, df)
        _run_and_save(out_dir, '03_convergence_audit.txt', convergence_audit, df)
        _run_and_save(out_dir, '04_global_minimum.txt', global_minimum, df)
        _run_and_save(out_dir, '05_two_basin_analysis.txt',
                      two_basin_analysis, df, args.basin_threshold)
        _run_and_save(out_dir, '06_runtime.txt', runtime, df)
        _run_and_save(out_dir, '07_price_coef.txt', price_coef, df)
        _run_and_save(out_dir, '08_recovery_rmse_by_group.txt',
                      recovery_rmse_by_group, df)
        _run_and_save(out_dir, '09_recovery_vs_objective.txt',
                      recovery_vs_objective, df)
        _run_and_save(out_dir, '10_param_level_error.txt', param_level_error, df)
        _run_and_save(out_dir, '11_pi_zero_false_positive.txt',
                      pi_zero_false_positive, df)
        _run_and_save(out_dir, '12_omitted_x2_bias.txt', omitted_x2_bias, df)
        _run_and_save(out_dir, '13_demo_overfit.txt', demo_overfit, df)
        _run_and_save(out_dir, '14_param_stability_within_spec.txt',
                      param_stability_within_spec, df)
        _run_and_save(out_dir, '15_best_vs_truth_start.txt',
                      best_vs_truth_start, df)

        per_iv.setdefault(iv_mode, {})[seed] = df

    for iv_mode, seed_dfs in per_iv.items():
        if len(seed_dfs) < 2:
            print(f'\n[skip cross-seed for iv_{iv_mode}: only {len(seed_dfs)} seed]')
            continue
        # Multiple iv modes coexist? namespace under iv_<mode>/; else flat.
        flat = len(per_iv) == 1
        cross_dir = root / 'analysis' if flat else root / 'analysis' / f'iv_{iv_mode}'
        print(f'\n### cross-seed  iv={iv_mode}  seeds={sorted(seed_dfs)}  '
              f'-> {cross_dir.relative_to(root.parent)}')
        _run_and_save(cross_dir, '16_estimate_across_seeds.txt',
                      estimate_across_seeds, seed_dfs)
        _run_and_save(cross_dir, '17_recovery_across_seeds.txt',
                      recovery_across_seeds, seed_dfs)
        _run_and_save(cross_dir, '18_best_spec_consistency.txt',
                      best_spec_consistency, seed_dfs)
        _run_and_save(cross_dir, '19_price_coef_across_seeds.txt',
                      price_coef_across_seeds, seed_dfs)


if __name__ == '__main__':
    main()
