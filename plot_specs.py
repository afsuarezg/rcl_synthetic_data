"""plot_specs.py -- publication-quality figures for analyze_specs.py reports.

Reads the same long-format CSVs as analyze_specs.py and saves one PNG per
analysis to:

    output/multiple_specs/seed_X/iv_Y/graphs/*.png

Cross-seed figures (when >=2 seeds for an iv_mode) go to:

    output/multiple_specs/graphs/*.png

Usage:
    uv run python plot_specs.py
    uv run python plot_specs.py --root output/multiple_specs --basin-threshold 2.0
"""
from __future__ import annotations

import argparse
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import specs_io as sio


sns.set_theme(style='ticks', font_scale=0.9)
PALETTE = sns.color_palette('Set2')
GROUP_COLORS = {
    'beta':  PALETTE[0],
    'sigma': PALETTE[1],
    'pi':    PALETTE[2],
    'gamma': PALETTE[3],
}
COL_NEAR = PALETTE[1]   # basin A: near global
COL_FAR  = PALETTE[2]   # basin B: far
COL_REF  = '#6c6c6c'    # truth reference
DPI = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def _abbrev(spec: str) -> str:
    """Shorten 'x2-x1_x2_x3__demos-income_age_hh_size_education' -> 'x123|inc,age,hh,edu'."""
    try:
        x2, demos = spec.split('__demos-')
        x2 = x2.replace('x2-', '').replace('_', '')  # x1_x2_x3 -> x1x2x3
    except ValueError:
        return spec
    demo_short = {
        'income': 'inc', 'age': 'age',
        'hh_size': 'hh', 'education': 'edu',
    }
    parts = demos.split('_')
    out_demos = []
    i = 0
    while i < len(parts):
        # hh_size is the only multi-token demo
        if i + 1 < len(parts) and parts[i] == 'hh' and parts[i + 1] == 'size':
            out_demos.append(demo_short['hh_size']); i += 2
        else:
            out_demos.append(demo_short.get(parts[i], parts[i])); i += 1
    return f'{x2} | {",".join(out_demos)}'


def _rmse(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float('nan')


def _fig_height(n_rows: int, per_row: float = 0.18, min_h: float = 4.0,
                max_h: float = 22.0) -> float:
    return max(min_h, min(max_h, per_row * n_rows + 1.0))


# ---------------------------------------------------------------------------
# Per-(seed, iv) plots
# ---------------------------------------------------------------------------

def plot_objective_ranking(df: pd.DataFrame, out_dir: Path) -> None:
    starts = sio.starts_table(df)
    best = sio.best_per_spec(df)
    rows = []
    for spec, sid in best.items():
        sub = starts[(starts.spec_label == spec) & (starts.start_id == sid)].iloc[0]
        tref = starts[(starts.spec_label == spec) & (starts.is_truth_start)]
        rows.append((spec, sub.objective,
                     tref.objective.iloc[0] if not tref.empty else np.nan))
    rows.sort(key=lambda r: r[1])
    labels = [_abbrev(s) for s, _, _ in rows]
    best_obj = [b for _, b, _ in rows]
    truth_obj = [t for _, _, t in rows]

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(rows))))
    y = np.arange(len(rows))
    ax.barh(y, best_obj, color=PALETTE[0], alpha=0.9, label='best perturbed')
    ax.scatter(truth_obj, y, marker='|', color=COL_REF, s=80,
               label='truth-warm reference', zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('GMM objective')
    ax.set_title('01. Objective ranking across specifications')
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    _save(fig, out_dir, '01_objective_ranking.png')


def plot_multistart_stability(df: pd.DataFrame, out_dir: Path) -> None:
    starts = sio.starts_table(df)
    pert = starts[~starts.is_truth_start].copy()
    spreads = (pert.groupby('spec_label').objective.agg(lambda s: s.max() - s.min())
               .sort_values(ascending=False))
    order = spreads.index.tolist()
    pert['order'] = pert.spec_label.map({s: i for i, s in enumerate(order)})
    truth_obj = starts[starts.is_truth_start].set_index('spec_label').objective

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(order))))
    sns.stripplot(data=pert, y='order', x='objective', orient='h',
                  jitter=0.18, size=4, color=PALETTE[0], ax=ax)
    # truth-warm reference
    tx = [truth_obj.get(s, np.nan) for s in order]
    ax.scatter(tx, np.arange(len(order)), marker='|', color=COL_REF, s=80,
               label='truth-warm reference', zorder=3)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_abbrev(s) for s in order], fontsize=7)
    ax.set_xlabel('GMM objective (per perturbed start)')
    ax.set_title('02. Multistart objective spread per spec (sorted by spread)')
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    _save(fig, out_dir, '02_multistart_stability.png')


def plot_convergence_audit(df: pd.DataFrame, out_dir: Path) -> None:
    starts = sio.starts_table(df)
    pert = starts[~starts.is_truth_start]
    by_spec = pert.groupby('spec_label').agg(
        n=('converged', 'size'),
        n_conv=('converged', 'sum'),
    )
    by_spec['n_fail'] = by_spec.n - by_spec.n_conv
    by_spec = by_spec.sort_values(['n_fail', 'spec_label'], ascending=[False, True])

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(by_spec))))
    y = np.arange(len(by_spec))
    ax.barh(y, by_spec.n_conv, color=GROUP_COLORS['sigma'], label='converged')
    ax.barh(y, by_spec.n_fail, left=by_spec.n_conv, color=GROUP_COLORS['pi'],
            label='non-converged')
    ax.set_yticks(y)
    ax.set_yticklabels([_abbrev(s) for s in by_spec.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('# perturbed starts')
    ax.set_title(f'03. Convergence audit (failing specs first)  '
                 f'-- {int(by_spec.n_fail.sum())} of {int(by_spec.n.sum())} starts failed')
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    _save(fig, out_dir, '03_convergence_audit.png')


def plot_global_minimum(df: pd.DataFrame, out_dir: Path) -> None:
    starts = sio.starts_table(df)
    pool = starts[(~starts.is_truth_start) & starts.converged]
    if pool.empty:
        return
    by_spec = pool.groupby('spec_label').objective.min().sort_values()
    gmin = by_spec.min()

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(by_spec))))
    y = np.arange(len(by_spec))
    delta = by_spec.values - gmin
    ax.barh(y, delta, color=PALETTE[0])
    ax.set_yticks(y)
    ax.set_yticklabels([_abbrev(s) for s in by_spec.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f'GMM objective - global best (= {gmin:.3f})')
    ax.set_title('04. Distance from global minimum (best perturbed start per spec)')
    sns.despine(ax=ax)
    _save(fig, out_dir, '04_global_minimum.png')


def plot_two_basin(df: pd.DataFrame, out_dir: Path, threshold: float) -> None:
    starts = sio.starts_table(df)
    pool = starts[(~starts.is_truth_start) & starts.converged]
    if pool.empty:
        return
    gmin = pool.objective.min()
    bins = np.linspace(pool.objective.min(), pool.objective.max(), 30)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    a = pool[pool.objective - gmin <= threshold].objective
    b = pool[pool.objective - gmin > threshold].objective
    ax.hist([a, b], bins=bins, stacked=True,
            color=[COL_NEAR, COL_FAR],
            label=[f'Basin A: <= {threshold} from best (n={len(a)})',
                   f'Basin B: > {threshold} from best (n={len(b)})'])
    ax.axvline(gmin, color='black', linestyle='--', linewidth=1,
               label=f'global best = {gmin:.2f}')
    ax.axvline(gmin + threshold, color=COL_FAR, linestyle=':', linewidth=1)
    ax.set_xlabel('GMM objective')
    ax.set_ylabel('# converged perturbed starts')
    ax.set_title('05. Two-basin classification')
    ax.legend()
    sns.despine(ax=ax)
    _save(fig, out_dir, '05_two_basin_analysis.png')


def plot_runtime(df: pd.DataFrame, out_dir: Path) -> None:
    starts = sio.starts_table(df)
    order = (starts.groupby('spec_label').elapsed_sec.median()
             .sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(8, _fig_height(len(order))))
    sns.boxplot(data=starts, y='spec_label', x='elapsed_sec',
                order=order, orient='h', color=PALETTE[0], ax=ax,
                fliersize=2, linewidth=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_abbrev(s) for s in order], fontsize=7)
    ax.set_xlabel('elapsed (sec)')
    ax.set_ylabel('')
    ax.set_title('06. Per-start wall-clock runtime')
    sns.despine(ax=ax)
    _save(fig, out_dir, '06_runtime.png')


def plot_price_coef(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df.param_name == 'beta_1') & (~df.is_truth_start)].copy()
    if sub.empty:
        return
    truth = float(df[df.param_name == 'beta_1'].truth.iloc[0])
    order = (sub.groupby('spec_label').estimate.mean()
             .sort_values().index.tolist())
    sub['order'] = sub.spec_label.map({s: i for i, s in enumerate(order)})

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(order))))
    sns.stripplot(data=sub, y='order', x='estimate', orient='h',
                  jitter=0.18, size=4, color=PALETTE[0], ax=ax)
    ax.axvline(truth, color='black', linestyle='--', linewidth=1,
               label=f'truth = {truth}')
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_abbrev(s) for s in order], fontsize=7)
    ax.set_xlabel('beta_1 estimate (price coefficient)')
    ax.set_title('07. Price coefficient across perturbed starts')
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    _save(fig, out_dir, '07_price_coef.png')


def plot_recovery_rmse(df: pd.DataFrame, out_dir: Path) -> None:
    best = sio.best_per_spec(df)
    rows = []
    for spec, sid in best.items():
        sub = df[(df.spec_label == spec) & (df.start_id == sid)]
        rec = {'spec': spec, 'total': _rmse(sub.abs_error.values)}
        for grp in ('beta', 'sigma', 'pi', 'gamma'):
            rec[grp] = _rmse(sub.loc[sub.param_group == grp, 'abs_error'].values)
        rows.append(rec)
    d = pd.DataFrame(rows).sort_values('total')

    fig, ax = plt.subplots(figsize=(9, _fig_height(len(d))))
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for grp in ('beta', 'sigma', 'pi', 'gamma'):
        vals = d[grp].fillna(0).values
        ax.barh(y, vals, left=left, color=GROUP_COLORS[grp], label=grp)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([_abbrev(s) for s in d.spec], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('summed RMSE across param groups (~= total RMSE * sqrt(...))')
    ax.set_title('08. Recovery RMSE stacked by parameter group (best perturbed start)')
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    _save(fig, out_dir, '08_recovery_rmse_by_group.png')


def plot_recovery_vs_objective(df: pd.DataFrame, out_dir: Path) -> None:
    best = sio.best_per_spec(df)
    rows = []
    for spec, sid in best.items():
        sub = df[(df.spec_label == spec) & (df.start_id == sid)]
        rows.append({
            'spec': spec,
            'obj': float(sub.objective.iloc[0]),
            'rmse': _rmse(sub.abs_error.values),
        })
    d = pd.DataFrame(rows)
    pearson = d.obj.corr(d.rmse)
    spearman = d.obj.rank().corr(d.rmse.rank())

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(d.obj, d.rmse, s=30, color=PALETTE[0], alpha=0.85, edgecolor='white')
    if d.obj.std() > 0:
        b, a = np.polyfit(d.obj, d.rmse, 1)
        xs = np.linspace(d.obj.min(), d.obj.max(), 100)
        ax.plot(xs, a + b * xs, color=COL_REF, linewidth=1, linestyle='--')
    ax.set_xlabel('GMM objective (best perturbed start)')
    ax.set_ylabel('Total RMSE vs. truth')
    ax.set_title(f'09. Recovery vs. objective    '
                 f'Pearson={pearson:.3f}    Spearman={spearman:.3f}')
    sns.despine(ax=ax)
    _save(fig, out_dir, '09_recovery_vs_objective.png')


def plot_param_level_error(df: pd.DataFrame, out_dir: Path) -> None:
    best = sio.best_per_spec(df)
    rec = pd.concat([df[(df.spec_label == s) & (df.start_id == sid)]
                     for s, sid in best.items()])
    summary = (rec.groupby(['param_physical', 'param_group'])['abs_error']
               .agg(['mean', 'std', 'count']).reset_index()
               .sort_values('mean', ascending=False))

    fig, ax = plt.subplots(figsize=(9, _fig_height(len(summary), per_row=0.22)))
    y = np.arange(len(summary))
    colors = [GROUP_COLORS.get(g, PALETTE[4]) for g in summary.param_group]
    ax.barh(y, summary['mean'], xerr=summary['std'].fillna(0),
            color=colors, ecolor=COL_REF, error_kw={'linewidth': 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels(summary.param_physical, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('mean |error| across specs that include this parameter (+/- std)')
    ax.set_title('10. Per-parameter recovery error (sorted by mean)')
    # Group legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in GROUP_COLORS.values()]
    ax.legend(handles, list(GROUP_COLORS), loc='lower right')
    sns.despine(ax=ax)
    _save(fig, out_dir, '10_param_level_error.png')


def plot_pi_zero(df: pd.DataFrame, out_dir: Path) -> None:
    best = sio.best_per_spec(df)
    parts = []
    for spec, sid in best.items():
        sub = df[(df.spec_label == spec) & (df.start_id == sid)
                 & (df.param_group == 'pi') & (df.truth == 0.0)]
        if not sub.empty:
            t = sub[['spec_label']].copy()
            t['abs_est'] = sub.estimate.abs().values
            parts.append(t)
    if not parts:
        return
    pooled = pd.concat(parts)
    order = (pooled.groupby('spec_label').abs_est.mean()
             .sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(order))))
    sns.boxplot(data=pooled, y='spec_label', x='abs_est',
                order=order, orient='h', color=GROUP_COLORS['pi'], ax=ax,
                fliersize=2, linewidth=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_abbrev(s) for s in order], fontsize=7)
    ax.set_xlabel('|estimate| where truth = 0')
    ax.set_ylabel('')
    ax.set_title(f'11. Phantom pi interactions (truth = 0); pooled n = {len(pooled)}')
    sns.despine(ax=ax)
    _save(fig, out_dir, '11_pi_zero_false_positive.png')


def plot_omitted_x2(df: pd.DataFrame, out_dir: Path) -> None:
    best = sio.best_per_spec(df)
    rec = pd.concat([df[(df.spec_label == s) & (df.start_id == sid)]
                     for s, sid in best.items()])
    summary = (rec.groupby(['x2_vars', 'param_group'])['abs_error']
               .mean().reset_index())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=summary, x='x2_vars', y='abs_error', hue='param_group',
                hue_order=['beta', 'sigma', 'pi', 'gamma'],
                palette=GROUP_COLORS, ax=ax)
    ax.set_xlabel('X2 specification (x2_vars)')
    ax.set_ylabel('mean |error| across specs')
    ax.set_title('12. Recovery MAE by X2 inclusion pattern')
    ax.legend(title='param group', loc='upper right')
    sns.despine(ax=ax)
    _save(fig, out_dir, '12_omitted_x2_bias.png')


def plot_demo_overfit(df: pd.DataFrame, out_dir: Path) -> None:
    best = sio.best_per_spec(df)
    rec = pd.concat([df[(df.spec_label == s) & (df.start_id == sid)]
                     for s, sid in best.items()])
    rec_pz = rec[(rec.param_group == 'pi') & (rec.truth == 0)].copy()
    rec_pz['abs_error'] = rec_pz.estimate.abs()

    grp_summary = (rec.groupby(['n_demos', 'param_group'])['abs_error']
                   .mean().reset_index())
    pz_summary = (rec_pz.groupby('n_demos').abs_error.mean()
                  .reset_index().rename(columns={'abs_error': 'mean_abs'}))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for grp in ('beta', 'sigma', 'pi', 'gamma'):
        d = grp_summary[grp_summary.param_group == grp]
        ax.plot(d.n_demos, d.abs_error, marker='o',
                color=GROUP_COLORS[grp], label=grp)
    ax.plot(pz_summary.n_demos, pz_summary.mean_abs, marker='s',
            color=GROUP_COLORS['pi'], linestyle='--', label='pi (truth=0 only)')
    ax.set_xlabel('# demographic vars in spec')
    ax.set_ylabel('mean |error|')
    ax.set_title('13. Recovery error vs. # demographic vars')
    ax.legend(title='group', loc='upper right')
    sns.despine(ax=ax)
    _save(fig, out_dir, '13_demo_overfit.png')


def plot_param_stability(df: pd.DataFrame, out_dir: Path) -> None:
    pert = df[~df.is_truth_start]
    pivot = (pert.groupby(['spec_label', 'param_physical'])['estimate']
             .std().reset_index()
             .pivot(index='spec_label', columns='param_physical', values='estimate'))
    order = pivot.mean(axis=1).sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[order]
    # column order: group then alpha
    def _g(p):
        return {'beta': 0, 'sigma': 1, 'pi': 2, 'gamma': 3}.get(p.split('_')[0], 4)
    cols = sorted(pivot.columns, key=lambda c: (_g(c), c))
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(cols)),
                                    _fig_height(len(pivot), per_row=0.22,
                                                min_h=6, max_h=28)))
    sns.heatmap(pivot, cmap='magma_r', cbar_kws={'label': 'std across perturbed starts'},
                ax=ax, linewidths=0)
    ax.set_yticklabels([_abbrev(s) for s in pivot.index], fontsize=8, rotation=0)
    ax.set_xticklabels(pivot.columns, fontsize=8, rotation=90)
    ax.set_xlabel('parameter')
    ax.set_ylabel('')
    ax.set_title('14. Within-spec estimate std across perturbed starts')
    _save(fig, out_dir, '14_param_stability_within_spec.png')


def plot_best_vs_truth_start(df: pd.DataFrame, out_dir: Path) -> None:
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
            'spec': spec, 'obj_min': by_obj[0][1], 'rmse_min': by_rmse[0][2],
            'rmse_at_best_obj': next(r[2] for r in per_start if r[0] == by_obj[0][0]),
            'match': by_obj[0][0] == by_rmse[0][0],
        })
    d = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = [GROUP_COLORS['sigma'] if m else GROUP_COLORS['pi'] for m in d.match]
    ax.scatter(d.rmse_min, d.rmse_at_best_obj, c=colors, s=35,
               alpha=0.85, edgecolor='white')
    lo = min(d.rmse_min.min(), d.rmse_at_best_obj.min()) * 0.95
    hi = max(d.rmse_min.max(), d.rmse_at_best_obj.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color=COL_REF, linestyle='--', linewidth=1)
    ax.set_xlabel('lowest RMSE among perturbed starts')
    ax.set_ylabel('RMSE at lowest-objective start')
    n_match = int(d.match.sum())
    ax.set_title(f'15. RMSE: best-by-obj vs. best-by-RMSE      '
                 f'match in {n_match}/{len(d)} specs')
    sns.despine(ax=ax)
    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor=GROUP_COLORS['sigma'],
                   markeredgecolor='white', label='best-by-obj == best-by-RMSE'),
        plt.Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor=GROUP_COLORS['pi'],
                   markeredgecolor='white', label='disagree'),
    ]
    ax.legend(handles=handles, loc='upper left')
    _save(fig, out_dir, '15_best_vs_truth_start.png')


# ---------------------------------------------------------------------------
# Cross-seed plots
# ---------------------------------------------------------------------------

def plot_bias_across_seeds(seed_dfs: dict[int, pd.DataFrame], out_dir: Path) -> None:
    parts = []
    for seed, df in seed_dfs.items():
        best = sio.best_per_spec(df)
        for spec, sid in best.items():
            sub = df[(df.spec_label == spec) & (df.start_id == sid)].copy()
            sub['seed'] = seed
            parts.append(sub)
    combined = pd.concat(parts, ignore_index=True)
    mean_est = (combined.groupby(['spec_label', 'param_physical', 'truth'])
                .estimate.mean().reset_index())
    mean_est['bias'] = mean_est.estimate - mean_est.truth
    pivot = mean_est.pivot(index='spec_label', columns='param_physical',
                           values='bias')

    def _g(p):
        return {'beta': 0, 'sigma': 1, 'pi': 2, 'gamma': 3}.get(p.split('_')[0], 4)
    cols = sorted(pivot.columns, key=lambda c: (_g(c), c))
    pivot = pivot[cols]
    order = pivot.abs().mean(axis=1).sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[order]

    vmax = float(np.nanpercentile(np.abs(pivot.values), 95))
    fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(cols)),
                                    _fig_height(len(pivot), per_row=0.22,
                                                min_h=6, max_h=28)))
    sns.heatmap(pivot, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                cbar_kws={'label': 'mean estimate - truth'}, ax=ax)
    ax.set_yticklabels([_abbrev(s) for s in pivot.index], fontsize=8, rotation=0)
    ax.set_xticklabels(pivot.columns, fontsize=8, rotation=90)
    ax.set_xlabel('parameter')
    ax.set_ylabel('')
    ax.set_title('16. Mean bias (estimate - truth) across seeds')
    _save(fig, out_dir, '16_estimate_across_seeds.png')


def plot_recovery_across_seeds(seed_dfs: dict[int, pd.DataFrame], out_dir: Path) -> None:
    rows = []
    for seed, df in seed_dfs.items():
        best = sio.best_per_spec(df)
        for spec, sid in best.items():
            sub = df[(df.spec_label == spec) & (df.start_id == sid)]
            rows.append({'seed': seed, 'spec_label': spec,
                         'rmse': _rmse(sub.abs_error.values)})
    d = pd.DataFrame(rows)
    order = d.groupby('spec_label').rmse.mean().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(order))))
    sns.boxplot(data=d, y='spec_label', x='rmse', order=order, orient='h',
                color=PALETTE[0], ax=ax, fliersize=2, linewidth=0.8)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_abbrev(s) for s in order], fontsize=7)
    ax.set_xlabel('total RMSE')
    ax.set_ylabel('')
    ax.set_title('17. Recovery RMSE per spec, across seeds')
    sns.despine(ax=ax)
    _save(fig, out_dir, '17_recovery_across_seeds.png')


def plot_best_spec_consistency(seed_dfs: dict[int, pd.DataFrame], out_dir: Path) -> None:
    counts: Counter[str] = Counter()
    for seed, df in seed_dfs.items():
        starts = sio.starts_table(df)
        pool = starts[(~starts.is_truth_start) & starts.converged]
        if pool.empty:
            continue
        winner = pool.groupby('spec_label').objective.min().idxmin()
        counts[winner] += 1
    items = counts.most_common()

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(items), per_row=0.25)))
    y = np.arange(len(items))
    ax.barh(y, [n for _, n in items], color=PALETTE[0])
    ax.set_yticks(y)
    ax.set_yticklabels([_abbrev(s) for s, _ in items], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('# seeds where this spec was the global best')
    ax.set_title('18. Best-spec consistency across seeds')
    sns.despine(ax=ax)
    _save(fig, out_dir, '18_best_spec_consistency.png')


def plot_price_coef_across_seeds(seed_dfs: dict[int, pd.DataFrame], out_dir: Path) -> None:
    rows = []
    truth = None
    for seed, df in seed_dfs.items():
        sub = df[(df.param_name == 'beta_1') & (~df.is_truth_start)]
        if sub.empty:
            continue
        if truth is None:
            truth = float(sub.truth.iloc[0])
        agg = sub.groupby('spec_label').estimate.mean().reset_index()
        agg['seed'] = seed
        rows.append(agg)
    if not rows:
        return
    d = pd.concat(rows, ignore_index=True)
    order = d.groupby('spec_label').estimate.mean().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(order))))
    sns.stripplot(data=d, y='spec_label', x='estimate', order=order, orient='h',
                  jitter=0.18, size=5, color=PALETTE[0], ax=ax)
    if truth is not None:
        ax.axvline(truth, color='black', linestyle='--', linewidth=1,
                   label=f'truth = {truth}')
        ax.legend(loc='lower right')
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_abbrev(s) for s in order], fontsize=7)
    ax.set_xlabel('within-seed mean beta_1 (perturbed starts)')
    ax.set_ylabel('')
    ax.set_title('19. Price coefficient across seeds')
    sns.despine(ax=ax)
    _save(fig, out_dir, '19_price_coef_across_seeds.png')


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root', type=Path, default=None,
                   help='multiple_specs root (default: ./output/multiple_specs)')
    p.add_argument('--basin-threshold', type=float, default=2.0,
                   help='Two-basin classifier distance (default: 2.0)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root or Path(__file__).parent / 'output' / 'multiple_specs'
    if not root.exists():
        raise SystemExit(f'Root not found: {root}')

    per_iv: dict[str, dict[int, pd.DataFrame]] = {}

    for seed, iv_mode, specs_dir in sio.discover(root):
        df = sio.load_long(specs_dir)
        out_dir = specs_dir.parent / 'graphs'
        print(f'plot: seed={seed} iv={iv_mode} -> {out_dir.relative_to(root.parent)}')

        plot_objective_ranking(df, out_dir)
        plot_multistart_stability(df, out_dir)
        plot_convergence_audit(df, out_dir)
        plot_global_minimum(df, out_dir)
        plot_two_basin(df, out_dir, args.basin_threshold)
        plot_runtime(df, out_dir)
        plot_price_coef(df, out_dir)
        plot_recovery_rmse(df, out_dir)
        plot_recovery_vs_objective(df, out_dir)
        plot_param_level_error(df, out_dir)
        plot_pi_zero(df, out_dir)
        plot_omitted_x2(df, out_dir)
        plot_demo_overfit(df, out_dir)
        plot_param_stability(df, out_dir)
        plot_best_vs_truth_start(df, out_dir)

        per_iv.setdefault(iv_mode, {})[seed] = df

    for iv_mode, seed_dfs in per_iv.items():
        if len(seed_dfs) < 2:
            print(f'[skip cross-seed plots for iv_{iv_mode}: only {len(seed_dfs)} seed]')
            continue
        flat = len(per_iv) == 1
        cross_dir = root / 'graphs' if flat else root / 'graphs' / f'iv_{iv_mode}'
        print(f'plot: cross-seed iv={iv_mode} -> {cross_dir.relative_to(root.parent)}')
        plot_bias_across_seeds(seed_dfs, cross_dir)
        plot_recovery_across_seeds(seed_dfs, cross_dir)
        plot_best_spec_consistency(seed_dfs, cross_dir)
        plot_price_coef_across_seeds(seed_dfs, cross_dir)


if __name__ == '__main__':
    main()
