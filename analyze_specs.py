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
import traceback
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
    try:
        with contextlib.redirect_stdout(buf):
            func(*args, **kwargs)
    except Exception as exc:
        # Don't let one analysis kill the whole report sweep (esp. on a long
        # SLURM run). Record the traceback in the report and on stdout, continue.
        buf.write(f'\n[ERROR in {filename}: {exc.__class__.__name__}: {exc}]\n')
        buf.write(traceback.format_exc())
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
# Elasticity analyses (analyses 20-32, mirroring BLP analyses 10-22)
# These run only when compute_elasticities.py has produced the CSVs.
# ---------------------------------------------------------------------------

def _truth_own_elas_mean(truth_elas: pd.DataFrame) -> float:
    """Mean over j of truth own-price elasticity (eps_jj)."""
    return float(truth_elas.loc[truth_elas.own_price, 'elasticity'].mean())


def elasticity_own_summary(elas: pd.DataFrame, df_long: pd.DataFrame,
                           truth_elas: pd.DataFrame | None) -> None:
    """20. Own-price elasticity summary by spec, ranked by GMM objective."""
    _hdr('20. Own-price elasticity summary by spec (best perturbed, ranked by GMM obj)')
    best = sio.best_per_spec(df_long)
    starts = sio.starts_table(df_long)
    obj_map = {r.spec_label: r.objective
               for _, r in starts[~starts.is_truth_start & starts.converged].iterrows()
               if r.start_id == best.get(r.spec_label)}

    rows = []
    for spec, sid in best.items():
        own = elas[(elas.spec_label == spec) & (elas.start_id == sid)
                   & elas.own_price]
        if own.empty:
            continue
        vals = own.elasticity.values
        rows.append({
            'spec': spec, 'obj': obj_map.get(spec, float('nan')),
            'mean': float(np.mean(vals)), 'median': float(np.median(vals)),
            'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'min': float(np.min(vals)), 'max': float(np.max(vals)),
        })
    rows.sort(key=lambda r: r['obj'])

    if truth_elas is not None:
        print(f'  Truth mean own-price elasticity = {_truth_own_elas_mean(truth_elas):.4f}')
        print()

    print(f'  {"Rank":>4}  {"GMM obj":>10}  {"mean":>9}  {"median":>9}  '
          f'{"std":>9}  {"min":>9}  {"max":>9}  Specification')
    _sep()
    for i, r in enumerate(rows, 1):
        print(f'  {i:>4}  {r["obj"]:>10.4f}  {r["mean"]:>9.4f}  '
              f'{r["median"]:>9.4f}  {r["std"]:>9.4f}  {r["min"]:>9.4f}  '
              f'{r["max"]:>9.4f}  {r["spec"]}')
    print()


def elasticity_multistart_stability(elas: pd.DataFrame,
                                    df_long: pd.DataFrame) -> None:
    """21. Per-product own-price elasticity spread across perturbed starts."""
    _hdr('21. Own-price elasticity stability across perturbed starts (per spec, per product)')
    rows = []
    for spec, sub in elas[~elas.is_truth_start & elas.own_price].groupby('spec_label'):
        g = sub.groupby('product_j')['elasticity']
        agg = g.agg(['mean', 'min', 'max', 'std', 'count']).rename(columns={'count': 'n'})
        agg['spread'] = agg['max'] - agg['min']
        for prod, r in agg.iterrows():
            rows.append({
                'spec': spec, 'product': int(prod),
                'n': int(r['n']), 'mean': r['mean'], 'spread': r['spread'],
                'std': (0.0 if pd.isna(r['std']) else r['std']),
            })
    if not rows:
        print('  No multi-start elasticity data.')
        return
    # rank specs by max spread
    by_spec_max = {}
    for r in rows:
        by_spec_max[r['spec']] = max(by_spec_max.get(r['spec'], 0), r['spread'])
    rows.sort(key=lambda r: (-by_spec_max[r['spec']], r['spec'], r['product']))

    print(f'  {"prod":>4}  {"n":>3}  {"mean":>9}  {"spread":>9}  '
          f'{"std":>9}  Specification')
    _sep()
    last_spec = None
    for r in rows[:200]:  # cap
        sep = '' if r['spec'] == last_spec else ''
        if r['spec'] != last_spec:
            last_spec = r['spec']
        print(f'  {r["product"]:>4}  {r["n"]:>3}  {r["mean"]:>9.4f}  '
              f'{r["spread"]:>9.4f}  {r["std"]:>9.4f}  {r["spec"]}')
    if len(rows) > 200:
        print(f'  ... ({len(rows) - 200} more rows omitted)')
    print()


def elasticity_top_substitutes(elas: pd.DataFrame, df_long: pd.DataFrame,
                               k: int = 3) -> None:
    """22. Top-k substitutes for each product in the rank-1 spec."""
    _hdr(f'22. Top-{k} substitutes per product (rank-1 spec, cross-price elasticities)')
    starts = sio.starts_table(df_long)
    best = sio.best_per_spec(df_long)
    # rank-1 spec = lowest best-start objective
    obj_pairs = [(spec, float(starts[(starts.spec_label == spec)
                  & (starts.start_id == sid)].objective.iloc[0]))
                 for spec, sid in best.items()]
    obj_pairs.sort(key=lambda x: x[1])
    rank1_spec, rank1_obj = obj_pairs[0]
    rank1_sid = best[rank1_spec]

    sub = elas[(elas.spec_label == rank1_spec) & (elas.start_id == rank1_sid)
               & (~elas.own_price)]
    print(f'  Spec (rank 1):  {rank1_spec}  (GMM obj = {rank1_obj:.4f}, start {rank1_sid})')
    print()
    for j in sorted(sub.product_j.unique()):
        s = sub[sub.product_j == j].sort_values('elasticity', ascending=False).head(k)
        firms = ', '.join(
            f'k={int(r.product_k)} (firm {int(r.firm_k)}, e={r.elasticity:+.4f}'
            f'{", same firm" if r.same_firm else ""})'
            for _, r in s.iterrows()
        )
        own = elas[(elas.spec_label == rank1_spec) & (elas.start_id == rank1_sid)
                   & (elas.product_j == j) & (elas.product_k == j)].elasticity.iloc[0]
        print(f'  j={j} (firm {sub[sub.product_j==j].firm_j.iloc[0]}, '
              f'own-elas={own:.4f}):  {firms}')
    print()


def elasticity_asymmetry(elas: pd.DataFrame, df_long: pd.DataFrame) -> None:
    """23. Cross-price elasticity asymmetry |e_jk - e_kj| (rank-1 spec)."""
    _hdr('23. Cross-price elasticity asymmetry (rank-1 spec)')
    starts = sio.starts_table(df_long)
    best = sio.best_per_spec(df_long)
    obj_pairs = [(spec, float(starts[(starts.spec_label == spec)
                  & (starts.start_id == sid)].objective.iloc[0]))
                 for spec, sid in best.items()]
    obj_pairs.sort(key=lambda x: x[1])
    rank1_spec = obj_pairs[0][0]
    rank1_sid = best[rank1_spec]
    sub = elas[(elas.spec_label == rank1_spec) & (elas.start_id == rank1_sid)
               & ~elas.own_price].copy()
    # build (j,k) -> e_jk lookup
    lookup = {(int(r.product_j), int(r.product_k)): float(r.elasticity)
              for _, r in sub.iterrows()}
    diffs = []
    for (j, k), v in lookup.items():
        if j < k and (k, j) in lookup:
            diffs.append({'j': j, 'k': k, 'e_jk': v, 'e_kj': lookup[(k, j)],
                          'abs_diff': abs(v - lookup[(k, j)])})
    diffs.sort(key=lambda r: r['abs_diff'], reverse=True)

    print(f'  Spec: {rank1_spec}')
    print(f'  N pairs:  {len(diffs)}')
    if not diffs:
        print('  No symmetric (j,k)/(k,j) elasticity pairs available.')
        print()
        return
    vals = np.array([r['abs_diff'] for r in diffs])
    print(f'  |e_jk - e_kj|  mean={vals.mean():.4f}  median={np.median(vals):.4f}  '
          f'max={vals.max():.4f}')
    print()
    print(f'  Top 20 most asymmetric pairs:')
    print(f'  {"j":>3}  {"k":>3}  {"e_jk":>9}  {"e_kj":>9}  {"|diff|":>9}')
    _sep(50)
    for r in diffs[:20]:
        print(f'  {r["j"]:>3}  {r["k"]:>3}  {r["e_jk"]:>9.4f}  '
              f'{r["e_kj"]:>9.4f}  {r["abs_diff"]:>9.4f}')
    print()


def elasticity_spec_spearman(elas: pd.DataFrame, df_long: pd.DataFrame) -> None:
    """24. Spearman rank correlation of own-price elasticities across spec pairs."""
    _hdr('24. Cross-spec Spearman rank correlation of own-price elasticities')
    best = sio.best_per_spec(df_long)
    # build (spec, product_j) -> own_elas
    rows = []
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)
                   & elas.own_price]
        for _, r in sub.iterrows():
            rows.append((spec, int(r.product_j), float(r.elasticity)))
    d = pd.DataFrame(rows, columns=['spec', 'product', 'elas'])
    pivot = d.pivot(index='product', columns='spec', values='elas')
    rho = pivot.rank().corr(method='pearson')  # rank-rank = spearman
    # report distribution of pairwise correlations
    arr = rho.values
    iu = np.triu_indices_from(arr, k=1)
    offdiag = arr[iu]
    print(f'  N specs:  {rho.shape[0]}.  Off-diagonal correlations:  '
          f'n={len(offdiag)}  mean={offdiag.mean():.4f}  '
          f'median={np.median(offdiag):.4f}  '
          f'min={offdiag.min():.4f}  max={offdiag.max():.4f}')
    print()
    # rank specs by mean correlation with all others
    mean_rho = (rho.sum(axis=1) - 1) / (rho.shape[0] - 1)
    print(f'  Specs ranked by mean Spearman correlation with all other specs:')
    print(f'  {"mean_rho":>10}  Specification')
    _sep()
    for spec, v in mean_rho.sort_values(ascending=False).items():
        print(f'  {v:>10.4f}  {spec}')
    print()


def elasticity_firm_substitution(elas: pd.DataFrame, df_long: pd.DataFrame) -> None:
    """25. Within-firm vs between-firm cross-price elasticities."""
    _hdr('25. Within-firm vs between-firm cross-price elasticities (best perturbed start)')
    best = sio.best_per_spec(df_long)
    rows = []
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)
                   & ~elas.own_price]
        within = sub[sub.same_firm].elasticity
        between = sub[~sub.same_firm].elasticity
        rows.append({
            'spec': spec,
            'within_mean': float(within.mean()),
            'between_mean': float(between.mean()),
            'ratio': float(within.mean() / between.mean())
                if between.mean() != 0 else float('nan'),
            'n_within': int(len(within)),
            'n_between': int(len(between)),
        })
    rows.sort(key=lambda r: r['ratio'], reverse=True)

    print(f'  Ratio = within_mean / between_mean. Larger ratios mean stronger '
          f'within-firm substitution.')
    print(f'  {"within":>9}  {"between":>9}  {"ratio":>7}  '
          f'{"n_w":>4}  {"n_b":>4}  Specification')
    _sep()
    for r in rows:
        print(f'  {r["within_mean"]:>9.4f}  {r["between_mean"]:>9.4f}  '
              f'{r["ratio"]:>7.2f}  {r["n_within"]:>4}  {r["n_between"]:>4}  '
              f'{r["spec"]}')
    print()


def elasticity_own_cross_spec_stability(elas: pd.DataFrame,
                                        df_long: pd.DataFrame) -> None:
    """26. Product-level own-elasticity coefficient of variation across specs."""
    _hdr('26. Per-product own-price elasticity cross-spec variability (CV)')
    best = sio.best_per_spec(df_long)
    rows = []
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)
                   & elas.own_price]
        for _, r in sub.iterrows():
            rows.append((int(r.product_j), int(r.firm_j), float(r.elasticity)))
    d = pd.DataFrame(rows, columns=['product', 'firm', 'elas'])
    g = d.groupby(['product', 'firm'])['elas']
    summary = g.agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    summary['cv'] = (summary['std'] / summary['mean'].abs()).round(4)
    summary = summary.sort_values('cv', ascending=False)
    print(f'  {"prod":>4}  {"firm":>4}  {"n":>3}  {"mean":>9}  {"std":>9}  '
          f'{"min":>9}  {"max":>9}  {"CV":>7}')
    _sep()
    for _, r in summary.iterrows():
        print(f'  {int(r["product"]):>4}  {int(r["firm"]):>4}  '
              f'{int(r["count"]):>3}  {r["mean"]:>9.4f}  '
              f'{(0.0 if pd.isna(r["std"]) else r["std"]):>9.4f}  '
              f'{r["min"]:>9.4f}  {r["max"]:>9.4f}  '
              f'{(0.0 if pd.isna(r["cv"]) else r["cv"]):>7.4f}')
    print()


def elasticity_cross_cross_spec_stability(elas: pd.DataFrame,
                                          df_long: pd.DataFrame,
                                          k: int = 5) -> None:
    """27. Cross-spec CV for the top-k substitute pairs in the rank-1 spec."""
    _hdr(f'27. Cross-price elasticity cross-spec variability for top-{k} substitute pairs')
    starts = sio.starts_table(df_long)
    best = sio.best_per_spec(df_long)
    obj_pairs = [(spec, float(starts[(starts.spec_label == spec)
                  & (starts.start_id == sid)].objective.iloc[0]))
                 for spec, sid in best.items()]
    obj_pairs.sort(key=lambda x: x[1])
    rank1_spec, _ = obj_pairs[0]
    rank1_sid = best[rank1_spec]
    rank1 = elas[(elas.spec_label == rank1_spec) & (elas.start_id == rank1_sid)
                 & ~elas.own_price]

    # Pick top-k by elasticity in rank-1 spec
    top = rank1.sort_values('elasticity', ascending=False).head(k)
    pairs = list(zip(top.product_j.astype(int), top.product_k.astype(int)))
    print(f'  Reference spec: {rank1_spec}')
    print(f'  Tracking {k} top-substitute (j,k) pairs across all specs:')
    print()
    print(f'  {"(j,k)":>7}  {"firms":>7}  {"n":>3}  {"mean":>9}  {"std":>9}  '
          f'{"min":>9}  {"max":>9}  {"CV":>7}')
    _sep()
    for j, k_ in pairs:
        vals = []
        for spec, sid in best.items():
            row = elas[(elas.spec_label == spec) & (elas.start_id == sid)
                       & (elas.product_j == j) & (elas.product_k == k_)]
            if not row.empty:
                vals.append(float(row.elasticity.iloc[0]))
        arr = np.array(vals)
        mean = arr.mean(); std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        firm_j = int(rank1[(rank1.product_j == j) & (rank1.product_k == k_)
                            ].firm_j.iloc[0])
        firm_k = int(rank1[(rank1.product_j == j) & (rank1.product_k == k_)
                            ].firm_k.iloc[0])
        cv = std / abs(mean) if mean != 0 else float('nan')
        print(f'  ({j},{k_:>1})  ({firm_j},{firm_k})  {len(arr):>3}  '
              f'{mean:>9.4f}  {std:>9.4f}  {arr.min():>9.4f}  '
              f'{arr.max():>9.4f}  {cv:>7.4f}')
    print()


def elasticity_pairwise_mad(elas: pd.DataFrame, df_long: pd.DataFrame) -> None:
    """28. Pairwise mean-absolute-deviation between specs (own-price)."""
    _hdr('28. Spec pairwise MAD of own-price elasticities')
    best = sio.best_per_spec(df_long)
    own_by_spec: dict[str, np.ndarray] = {}
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)
                   & elas.own_price].sort_values('product_j')
        own_by_spec[spec] = sub.elasticity.to_numpy()

    specs = list(own_by_spec.keys())
    n = len(specs)
    mat = np.zeros((n, n))
    for i, si in enumerate(specs):
        for j, sj in enumerate(specs):
            mat[i, j] = np.mean(np.abs(own_by_spec[si] - own_by_spec[sj]))

    # mean off-diagonal per spec
    mean_mad = (mat.sum(axis=1) - np.diag(mat)) / (n - 1)
    order = np.argsort(mean_mad)
    print(f'  N specs: {n}.  Off-diagonal mean MAD across all spec pairs.')
    print(f'  Min mean-MAD (most centrally located spec):  '
          f'{specs[order[0]]}  ({mean_mad[order[0]]:.4f})')
    print(f'  Max mean-MAD (most distant from others):    '
          f'{specs[order[-1]]}  ({mean_mad[order[-1]]:.4f})')
    print()
    print(f'  {"mean_MAD":>10}  Specification')
    _sep()
    for i in order:
        print(f'  {mean_mad[i]:>10.4f}  {specs[i]}')
    print()


def elasticity_pair_across_sims(elas: pd.DataFrame, df_long: pd.DataFrame,
                                truth_elas: pd.DataFrame | None) -> None:
    """30. Four elasticities (e_jj, e_kk, e_jk, e_kj) across perturbed starts of best-mean-obj spec."""
    _hdr('30. (e_jj, e_kk, e_jk, e_kj) across perturbed starts (best-mean-obj spec)')
    # best-mean-obj spec: lowest mean objective among perturbed starts
    starts = sio.starts_table(df_long)
    pert = starts[~starts.is_truth_start]
    mean_obj = pert.groupby('spec_label').objective.mean().sort_values()
    spec = mean_obj.index[0]
    # pick (j,k) = (0,1) as a stable default (both products of firm 1)
    j, k = 0, 1
    print(f'  Spec: {spec}  (mean obj across perturbed starts = {mean_obj.iloc[0]:.4f})')
    print(f'  Pair: j={j}  k={k}')
    print()

    sub = elas[(elas.spec_label == spec) & ~elas.is_truth_start]
    rows = []
    for sid, ssub in sub.groupby('start_id'):
        def _e(jj, kk):
            r = ssub[(ssub.product_j == jj) & (ssub.product_k == kk)]
            return float(r.elasticity.iloc[0]) if not r.empty else float('nan')
        rows.append({
            'start_id': int(sid),
            'e_jj': _e(j, j), 'e_kk': _e(k, k),
            'e_jk': _e(j, k), 'e_kj': _e(k, j),
        })
    if truth_elas is not None:
        def _et(jj, kk):
            r = truth_elas[(truth_elas.product_j == jj)
                           & (truth_elas.product_k == kk)]
            return float(r.elasticity.iloc[0]) if not r.empty else float('nan')
        truth_row = {'start_id': 'TRUTH',
                     'e_jj': _et(j, j), 'e_kk': _et(k, k),
                     'e_jk': _et(j, k), 'e_kj': _et(k, j)}
    else:
        truth_row = None

    print(f'  {"start":>5}  {"e_jj":>9}  {"e_kk":>9}  {"e_jk":>9}  {"e_kj":>9}')
    _sep(60)
    for r in rows:
        print(f'  {r["start_id"]:>5}  {r["e_jj"]:>9.4f}  {r["e_kk"]:>9.4f}  '
              f'{r["e_jk"]:>9.4f}  {r["e_kj"]:>9.4f}')
    if truth_row is not None:
        print(f'  {truth_row["start_id"]:>5}  {truth_row["e_jj"]:>9.4f}  '
              f'{truth_row["e_kk"]:>9.4f}  {truth_row["e_jk"]:>9.4f}  '
              f'{truth_row["e_kj"]:>9.4f}')
    print()


def elasticity_pair_best_sim_across_specs(elas: pd.DataFrame,
                                          df_long: pd.DataFrame,
                                          truth_elas: pd.DataFrame | None) -> None:
    """31. Same four elasticities for one (j,k) pair, best start per spec."""
    _hdr('31. (e_jj, e_kk, e_jk, e_kj) at best perturbed start across specs')
    j, k = 0, 1
    best = sio.best_per_spec(df_long)
    starts = sio.starts_table(df_long)
    rows = []
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)]
        def _e(jj, kk):
            r = sub[(sub.product_j == jj) & (sub.product_k == kk)]
            return float(r.elasticity.iloc[0]) if not r.empty else float('nan')
        obj_row = starts[(starts.spec_label == spec) & (starts.start_id == sid)]
        rows.append({
            'spec': spec,
            'obj': float(obj_row.objective.iloc[0]),
            'e_jj': _e(j, j), 'e_kk': _e(k, k),
            'e_jk': _e(j, k), 'e_kj': _e(k, j),
        })
    rows.sort(key=lambda r: r['obj'])

    if truth_elas is not None:
        def _et(jj, kk):
            r = truth_elas[(truth_elas.product_j == jj)
                           & (truth_elas.product_k == kk)]
            return float(r.elasticity.iloc[0]) if not r.empty else float('nan')
        print(f'  Pair (j,k) = ({j},{k})')
        print(f'  Truth:  e_jj={_et(j,j):.4f}  e_kk={_et(k,k):.4f}  '
              f'e_jk={_et(j,k):.4f}  e_kj={_et(k,j):.4f}')
        print()

    print(f'  {"obj":>9}  {"e_jj":>9}  {"e_kk":>9}  {"e_jk":>9}  '
          f'{"e_kj":>9}  Specification')
    _sep()
    for r in rows:
        print(f'  {r["obj"]:>9.4f}  {r["e_jj"]:>9.4f}  {r["e_kk"]:>9.4f}  '
              f'{r["e_jk"]:>9.4f}  {r["e_kj"]:>9.4f}  {r["spec"]}')
    print()


# ---------------------------------------------------------------------------
# Truth-driven analyses (analyses 33-35) -- unique to the synthetic data
# ---------------------------------------------------------------------------

def elasticity_recovery_own(elas: pd.DataFrame, df_long: pd.DataFrame,
                            truth_elas: pd.DataFrame) -> None:
    """33. Per-spec MAE / RMSE / signed bias on own-price elasticities vs truth."""
    _hdr('33. Own-elasticity recovery vs. DGP truth (best perturbed start)')
    best = sio.best_per_spec(df_long)
    truth_own = truth_elas[truth_elas.own_price].set_index('product_j')['elasticity']

    rows = []
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)
                   & elas.own_price].set_index('product_j')['elasticity']
        common = sub.index.intersection(truth_own.index)
        err = (sub.loc[common] - truth_own.loc[common]).to_numpy()
        rows.append({
            'spec': spec,
            'mae': float(np.mean(np.abs(err))),
            'rmse': float(np.sqrt(np.mean(err ** 2))),
            'bias': float(np.mean(err)),
            'min_err': float(err.min()),
            'max_err': float(err.max()),
        })
    rows.sort(key=lambda r: r['rmse'])
    print(f'  Truth own-elas vector: {truth_own.tolist()}')
    print(f'  Errors = estimate - truth')
    print()
    print(f'  {"rmse":>9}  {"mae":>9}  {"bias":>9}  '
          f'{"min_err":>9}  {"max_err":>9}  Specification')
    _sep()
    for r in rows:
        print(f'  {r["rmse"]:>9.4f}  {r["mae"]:>9.4f}  {r["bias"]:>9.4f}  '
              f'{r["min_err"]:>9.4f}  {r["max_err"]:>9.4f}  {r["spec"]}')
    print()


def elasticity_recovery_cross(elas: pd.DataFrame, df_long: pd.DataFrame,
                              truth_elas: pd.DataFrame) -> None:
    """34. Per-spec MAE on full J x J elasticity matrix; split same-firm vs different-firm."""
    _hdr('34. Full elasticity-matrix recovery vs. truth (same-firm vs between-firm)')
    best = sio.best_per_spec(df_long)
    truth_jk = truth_elas.set_index(['product_j', 'product_k'])[
        ['elasticity', 'same_firm']]
    rows = []
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)] \
            .set_index(['product_j', 'product_k'])['elasticity']
        common = sub.index.intersection(truth_jk.index)
        s = sub.loc[common]
        t = truth_jk.loc[common, 'elasticity']
        sf = truth_jk.loc[common, 'same_firm'].astype(bool)
        err = (s - t).to_numpy()
        sf_arr = sf.to_numpy()
        same = err[sf_arr]
        diff = err[~sf_arr]
        rows.append({
            'spec': spec,
            'mae_all': float(np.mean(np.abs(err))),
            'mae_same': float(np.mean(np.abs(same))) if same.size else float('nan'),
            'mae_diff': float(np.mean(np.abs(diff))) if diff.size else float('nan'),
            'mae_own': float(np.mean(np.abs(
                [e for (j, k), e in zip(common, err) if j == k]))),
            'mae_off': float(np.mean(np.abs(
                [e for (j, k), e in zip(common, err) if j != k]))),
        })
    rows.sort(key=lambda r: r['mae_all'])

    print(f'  {"mae_all":>9}  {"mae_own":>9}  {"mae_off":>9}  '
          f'{"mae_same":>9}  {"mae_diff":>9}  Specification')
    _sep()
    for r in rows:
        print(f'  {r["mae_all"]:>9.4f}  {r["mae_own"]:>9.4f}  '
              f'{r["mae_off"]:>9.4f}  {r["mae_same"]:>9.4f}  '
              f'{r["mae_diff"]:>9.4f}  {r["spec"]}')
    print()


def post_estimation_recovery(post: pd.DataFrame,
                             truth_post: pd.DataFrame) -> None:
    """35. Per-spec recovery error on post-estimation quantities."""
    _hdr('35. Post-estimation recovery vs. truth (best perturbed start)')
    cols = ['mean_own_elas', 'mean_outside_div', 'mean_markup', 'mean_hhi',
            'mean_delta_markup', 'mean_delta_hhi', 'mean_delta_cs']
    truth = truth_post.iloc[0]
    print(f'  Truth values:')
    for c in cols:
        if c in truth.index:
            print(f'    {c:>20s} = {truth[c]:.4f}')
    print()

    rows = []
    for _, r in post.iterrows():
        rec = {'spec': r.spec_label}
        for c in cols:
            if c in r.index and c in truth.index and pd.notna(r[c]) and pd.notna(truth[c]):
                rec[f'd_{c}'] = float(r[c] - truth[c])
        rows.append(rec)
    # rank by |d_mean_delta_hhi| if available, else d_mean_own_elas
    rank_col = 'd_mean_delta_hhi' if 'd_mean_delta_hhi' in rows[0] else 'd_mean_own_elas'
    rows.sort(key=lambda r: abs(r.get(rank_col, 0.0)))

    print(f'  Per-spec deltas (estimate - truth), ranked by |{rank_col}|:')
    show_cols = ['d_mean_own_elas', 'd_mean_markup',
                 'd_mean_delta_markup', 'd_mean_delta_hhi', 'd_mean_delta_cs']
    hdr = '  ' + '  '.join(f'{c.replace("d_mean_", "d_"):>10}' for c in show_cols) + '  Specification'
    print(hdr)
    _sep(len(hdr))
    for r in rows:
        vals = '  '.join(f'{r.get(c, float("nan")):>10.4f}' for c in show_cols)
        print(f'  {vals}  {r["spec"]}')
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

        # Elasticity analyses (20-35) -- only when CSVs are present.
        if sio.has_elasticities(specs_dir):
            elas = sio.load_elasticities(specs_dir)
            post = sio.load_post_estimation(specs_dir)
            seed_dir = specs_dir.parent.parent
            truth_elas = (sio.load_truth_elasticities(seed_dir)
                          if sio.has_truth_elasticities(seed_dir) else None)
            truth_post = (sio.load_truth_post_estimation(seed_dir)
                          if (seed_dir / 'truth_post_estimation.csv').exists()
                          else None)

            _run_and_save(out_dir, '20_elasticity_own_summary.txt',
                          elasticity_own_summary, elas, df, truth_elas)
            _run_and_save(out_dir, '21_elasticity_multistart_stability.txt',
                          elasticity_multistart_stability, elas, df)
            _run_and_save(out_dir, '22_elasticity_top_substitutes.txt',
                          elasticity_top_substitutes, elas, df)
            _run_and_save(out_dir, '23_elasticity_asymmetry.txt',
                          elasticity_asymmetry, elas, df)
            _run_and_save(out_dir, '24_elasticity_spec_spearman.txt',
                          elasticity_spec_spearman, elas, df)
            _run_and_save(out_dir, '25_elasticity_firm_substitution.txt',
                          elasticity_firm_substitution, elas, df)
            _run_and_save(out_dir, '26_elasticity_own_cross_spec_stability.txt',
                          elasticity_own_cross_spec_stability, elas, df)
            _run_and_save(out_dir, '27_elasticity_cross_cross_spec_stability.txt',
                          elasticity_cross_cross_spec_stability, elas, df)
            _run_and_save(out_dir, '28_elasticity_pairwise_mad.txt',
                          elasticity_pairwise_mad, elas, df)
            _run_and_save(out_dir, '30_elasticity_pair_across_sims.txt',
                          elasticity_pair_across_sims, elas, df, truth_elas)
            _run_and_save(out_dir, '31_elasticity_pair_best_sim_across_specs.txt',
                          elasticity_pair_best_sim_across_specs,
                          elas, df, truth_elas)

            if truth_elas is not None:
                _run_and_save(out_dir, '33_elasticity_recovery_own.txt',
                              elasticity_recovery_own, elas, df, truth_elas)
                _run_and_save(out_dir, '34_elasticity_recovery_cross.txt',
                              elasticity_recovery_cross, elas, df, truth_elas)
            if truth_post is not None:
                _run_and_save(out_dir, '35_post_estimation_recovery.txt',
                              post_estimation_recovery, post, truth_post)
        else:
            print(f'[skip elasticity analyses for seed={seed} iv={iv_mode}: '
                  f'run compute_elasticities.py first]')

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
