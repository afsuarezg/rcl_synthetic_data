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
import re
import textwrap
import traceback
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
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

_TITLE_NUM = re.compile(r'^\s*\d{1,2}\.\s+')


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Strip the leading analysis index ("26. ") from every title; filenames keep
    # the number so on-disk ordering is unaffected.
    for ax in fig.axes:
        t = ax.get_title()
        if t:
            ax.set_title(_TITLE_NUM.sub('', t))
    for txt in fig.texts:  # suptitle lives in fig.texts
        s = txt.get_text()
        if s:
            txt.set_text(_TITLE_NUM.sub('', s))
    fig.savefig(out_dir / name, dpi=DPI, bbox_inches='tight')
    plt.close(fig)


def _safe_plot(fn, *args) -> None:
    """Run one plot; on failure log the traceback and continue rather than kill
    the whole run (esp. on a long SLURM job). Mirrors the try/except that
    analyze_specs._run_and_save wraps each report in. Any half-built figure is
    discarded so it can't leak into a later plot."""
    try:
        fn(*args)
    except Exception as exc:
        print(f'  [ERROR in {fn.__name__}: {exc.__class__.__name__}: {exc}]')
        traceback.print_exc()
        plt.close('all')


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


def _hue_colors(values, hue, palette='tab10'):
    """Per-point colour list aligned with `values`, using the same sorted-unique
    -> palette mapping seaborn applies for categorical hue."""
    cats = sorted(pd.unique(hue))
    pal = sns.color_palette(palette, len(cats))
    cmap = dict(zip(cats, pal))
    return [cmap[h] for h in hue]


def _clip_with_outlier_markers(ax, positions, values, *, orient='v',
                               colors=None, color=COL_REF, marker_size=40):
    """Clip the value-axis to Tukey 3*IQR bounds; render out-of-range points as
    labeled edge triangles instead of letting them stretch the axis.

    orient='v': values on y -> clip ylim, '^'(top)/'v'(bottom) triangles at x.
    orient='h': values on x -> clip xlim, '>'(right)/'<'(left) triangles at y.

    The original seaborn markers beyond the clip are hidden by the new limit
    (clip_on=True); our triangles use clip_on=False to sit at the edge.
    Co-located outliers (same position + side) collapse to one triangle, labeled
    with the value (single) or a count (e.g. '4x'). No-op (axis left to
    autoscale) when <6 points, zero IQR, or no outliers exist.
    """
    # A tight clip otherwise triggers matplotlib's '1e-9' offset stamp. Only a
    # ScalarFormatter supports these toggles; a categorical axis has none.
    val_ax = ax.yaxis if orient == 'v' else ax.xaxis
    fmt = val_ax.get_major_formatter()
    if isinstance(fmt, matplotlib.ticker.ScalarFormatter):
        fmt.set_useOffset(False)
        fmt.set_scientific(False)

    pts = [(p, float(v), (colors[i] if colors is not None else color))
           for i, (p, v) in enumerate(zip(positions, values))
           if not (isinstance(v, float) and np.isnan(v))]
    if len(pts) < 6:
        return
    s = sorted(v for _, v, _ in pts)
    n = len(s)
    q1, q3 = s[n // 4], s[(3 * n) // 4]
    iqr = q3 - q1
    if iqr <= 0:
        return
    lo, hi = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    outliers = [(p, v, c) for p, v, c in pts if v < lo or v > hi]
    if not outliers:
        return

    # Floor a near-degenerate window so we never produce a sliver axis.
    median = s[n // 2]
    min_span = max(0.1 * abs(median), 1e-6)
    if hi - lo < min_span:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
    rng = hi - lo
    pad = 0.05 * rng
    near = hi + pad * 0.5   # just inside the high edge
    far = lo - pad * 0.5    # just inside the low edge
    if orient == 'v':
        ax.set_ylim(lo - pad, hi + pad)
    else:
        ax.set_xlim(lo - pad, hi + pad)

    # Collapse outliers sharing a (position, side) into one triangle.
    groups: dict[tuple, list] = {}
    for p, v, c in outliers:
        groups.setdefault((p, v > hi), []).append((v, c))
    for (p, high), members in groups.items():
        c = members[0][1]
        edge = near if high else far
        label = (f'{members[0][0]:.2f}' if len(members) == 1
                 else f'{len(members)}x')
        bbox = dict(boxstyle='round,pad=0.15', facecolor='white',
                    edgecolor='none', alpha=0.75)
        if orient == 'v':
            marker = '^' if high else 'v'
            ax.scatter([p], [edge], marker=marker, color=c, s=marker_size + 20,
                       edgecolor='black', linewidth=0.8, zorder=5, clip_on=False)
            ax.annotate(label, xy=(p, edge),
                        xytext=(0, -10 if high else 10),
                        textcoords='offset points', ha='center',
                        va='top' if high else 'bottom', fontsize=7, zorder=6,
                        bbox=bbox)
        else:
            marker = '>' if high else '<'
            ax.scatter([edge], [p], marker=marker, color=c, s=marker_size + 20,
                       edgecolor='black', linewidth=0.8, zorder=5, clip_on=False)
            ax.annotate(label, xy=(edge, p),
                        xytext=(-10 if high else 10, 0),
                        textcoords='offset points',
                        ha='right' if high else 'left', va='center',
                        fontsize=7, zorder=6, bbox=bbox)


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
    # Keep one extreme start from stretching the x-axis: clip to the bulk and
    # mark out-of-range objectives at the edge with their value (cf. #20/#27).
    _clip_with_outlier_markers(ax, pert.order.tolist(), pert.objective.tolist(),
                               orient='h', color=PALETTE[0])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_abbrev(s) for s in order], fontsize=7)
    ax.set_xlabel('GMM objective (per perturbed start)')
    ax.set_title('02. Multistart objective spread per spec (sorted by spread)')
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    _save(fig, out_dir, '02_multistart_stability.png')


def plot_objective_spec_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-spec GMM-objective summary across perturbed starts: a mean bar with a
    ±1 std error bar and a dashed min/max range, sorted by mean. Complements 01
    (best only) and 02 (raw per-start points) by aggregating the same objectives
    into mean ± std ± range. The truth-warm start is excluded from the statistics
    (near-zero by construction) but drawn as a per-spec reference tick."""
    starts = sio.starts_table(df)
    pert = starts[~starts.is_truth_start]
    grp = pert.groupby('spec_label').objective
    stats = pd.DataFrame({
        'mean': grp.mean(),
        'std':  grp.std(ddof=0),
        'min':  grp.min(),
        'max':  grp.max(),
        'n':    grp.size(),
    }).sort_values('mean')
    if stats.empty:
        return
    order = stats.index.tolist()
    truth_obj = starts[starts.is_truth_start].set_index('spec_label').objective

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(order))))
    y = np.arange(len(order))
    ax.barh(y, stats['mean'], color=PALETTE[0], alpha=0.9, label='mean')
    ax.errorbar(stats['mean'], y, xerr=stats['std'].fillna(0.0), fmt='none',
                ecolor='#444', elinewidth=1.0, capsize=3, capthick=1.0,
                label='±1 std', zorder=3)
    # dashed min/max range with end-cap ticks
    for yi, lo, hi in zip(y, stats['min'], stats['max']):
        ax.plot([lo, hi], [yi, yi], color='#999', lw=0.7, ls='--', zorder=2)
        ax.plot([lo, hi], [yi, yi], marker='|', color='#666', ms=6,
                ls='none', zorder=2)
    # truth-warm reference tick per spec
    tx = [truth_obj.get(s, np.nan) for s in order]
    ax.scatter(tx, y, marker='|', color=COL_REF, s=80,
               label='truth-warm reference', zorder=4)
    # n= annotation at the right edge of each row
    x_max = float(np.nanmax(stats['max'].to_numpy()))
    for yi, n in zip(y, stats['n']):
        ax.text(x_max * 1.01, yi, f'n={int(n)}', va='center', ha='left',
                fontsize=6, color='#555')
    ax.set_yticks(y)
    ax.set_yticklabels([_abbrev(s) for s in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('GMM objective')
    ax.set_title('36. Objective by specification (mean ± std, range)')
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    _save(fig, out_dir, '36_objective_spec_comparison.png')


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
# Elasticity plots (analyses 20-35)
# ---------------------------------------------------------------------------

def plot_elasticity_own_summary(elas, df_long, truth_elas, out_dir):
    """20. Boxplot of own-price elasticities per spec, sorted by GMM objective."""
    best = sio.best_per_spec(df_long)
    starts = sio.starts_table(df_long)
    rows = []
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)
                   & elas.own_price]
        if sub.empty:
            continue
        obj = float(starts[(starts.spec_label == spec)
                           & (starts.start_id == sid)].objective.iloc[0])
        for _, r in sub.iterrows():
            rows.append({'spec': spec, 'obj': obj,
                         'elasticity': float(r.elasticity)})
    d = pd.DataFrame(rows)
    order = (d[['spec', 'obj']].drop_duplicates()
             .sort_values('obj').spec.tolist())

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(order))))
    sns.boxplot(data=d, y='spec', x='elasticity', order=order, orient='h',
                color=PALETTE[0], ax=ax, fliersize=2, linewidth=0.8)
    spec_pos = {s: i for i, s in enumerate(order)}
    _clip_with_outlier_markers(
        ax, [spec_pos[s] for s in d.spec], d.elasticity.tolist(),
        orient='h', color=PALETTE[0])
    if truth_elas is not None:
        tval = float(truth_elas[truth_elas.own_price].elasticity.mean())
        ax.axvline(tval, color='black', linestyle='--', linewidth=1,
                   label=f'truth mean = {tval:.3f}')
        ax.legend(loc='lower right')
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_abbrev(s) for s in order], fontsize=7)
    ax.set_xlabel('own-price elasticity')
    ax.set_ylabel('')
    ax.set_title('20. Own-price elasticity distribution per spec '
                 '(sorted by GMM objective)')
    sns.despine(ax=ax)
    _save(fig, out_dir, '20_elasticity_own_summary.png')


def plot_elasticity_multistart_stability(elas, df_long, out_dir):
    """21. Dotplot: per-product own-elasticity across perturbed starts, per spec."""
    sub = elas[~elas.is_truth_start & elas.own_price].copy()
    # rank specs by max spread
    spread = (sub.groupby(['spec_label', 'product_j']).elasticity
              .agg(lambda s: s.max() - s.min()))
    by_spec = spread.groupby(level=0).max().sort_values(ascending=False)
    # take top 10 most-unstable specs
    keep = by_spec.head(10).index.tolist()
    sub = sub[sub.spec_label.isin(keep)]

    fig, ax = plt.subplots(figsize=(9, _fig_height(len(keep), per_row=0.5,
                                                   min_h=5, max_h=14)))
    spec_order = keep
    sub['spec_order'] = sub.spec_label.map({s: i for i, s in enumerate(spec_order)})
    sns.stripplot(data=sub, y='spec_order', x='elasticity', hue='product_j',
                  orient='h', native_scale=True,
                  jitter=0.18, dodge=True, palette='tab10', size=4, ax=ax)
    _clip_with_outlier_markers(
        ax, sub.spec_order.tolist(), sub.elasticity.tolist(), orient='h',
        colors=_hue_colors(sub.elasticity, sub.product_j))
    ax.set_yticks(np.arange(len(spec_order)))
    ax.set_yticklabels([_abbrev(s) for s in spec_order], fontsize=8)
    ax.set_xlabel('own-price elasticity')
    ax.set_ylabel('')
    ax.set_title('21. Own-price elasticity stability across perturbed starts '
                 '(top 10 most unstable specs)')
    ax.legend(title='product', loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=7, ncol=2)
    sns.despine(ax=ax)
    _save(fig, out_dir, '21_elasticity_multistart_stability.png')


def plot_elasticity_top_substitutes(elas, df_long, out_dir):
    """22. Cross-elasticity heatmap (10x10) for the rank-1 spec, firm blocks marked."""
    starts = sio.starts_table(df_long)
    best = sio.best_per_spec(df_long)
    obj_pairs = sorted(
        [(spec, float(starts[(starts.spec_label == spec)
                             & (starts.start_id == sid)].objective.iloc[0]))
         for spec, sid in best.items()], key=lambda x: x[1])
    rank1_spec = obj_pairs[0][0]
    rank1_sid = best[rank1_spec]
    sub = elas[(elas.spec_label == rank1_spec) & (elas.start_id == rank1_sid)]
    if sub.empty:
        print(f'  [skip 22: rank-1 spec {rank1_spec!r} start {rank1_sid} '
              f'has no elasticities]')
        return
    pivot = sub.pivot(index='product_j', columns='product_k', values='elasticity')

    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = float(np.abs(pivot.values).max())
    sns.heatmap(pivot, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt='.3f', annot_kws={'size': 7},
                cbar_kws={'label': 'elasticity'}, ax=ax)
    # firm-block boundaries (FIRM_PATTERN = 1,1,2,2,3,3,4,4,5,5 -> lines at 2,4,6,8)
    for k in (2, 4, 6, 8):
        ax.axvline(k, color='black', linewidth=0.8)
        ax.axhline(k, color='black', linewidth=0.8)
    ax.set_xlabel('product k')
    ax.set_ylabel('product j')
    ax.set_title(f'22. Cross-elasticity matrix (rank-1 spec)\n{_abbrev(rank1_spec)}')
    _save(fig, out_dir, '22_elasticity_top_substitutes.png')


def plot_elasticity_asymmetry(elas, df_long, out_dir):
    """23. Asymmetry |e_jk - e_kj| scatter for the rank-1 spec."""
    starts = sio.starts_table(df_long)
    best = sio.best_per_spec(df_long)
    rank1_spec = sorted(
        [(spec, float(starts[(starts.spec_label == spec)
                             & (starts.start_id == sid)].objective.iloc[0]))
         for spec, sid in best.items()], key=lambda x: x[1])[0][0]
    rank1_sid = best[rank1_spec]
    sub = elas[(elas.spec_label == rank1_spec) & (elas.start_id == rank1_sid)
               & ~elas.own_price]
    lookup = {(int(r.product_j), int(r.product_k)): float(r.elasticity)
              for _, r in sub.iterrows()}
    pairs = []
    for (j, k), v in lookup.items():
        if j < k and (k, j) in lookup:
            pairs.append((j, k, v, lookup[(k, j)]))
    if not pairs:
        print(f'  [skip 23: rank-1 spec {rank1_spec!r} start {rank1_sid} '
              f'has no symmetric (j,k)/(k,j) cross-price pairs]')
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    xs = [p[2] for p in pairs]
    ys = [p[3] for p in pairs]
    ax.scatter(xs, ys, s=30, color=PALETTE[0], alpha=0.85, edgecolor='white')
    lo = min(min(xs), min(ys)) * 0.95
    hi = max(max(xs), max(ys)) * 1.05
    ax.plot([lo, hi], [lo, hi], color=COL_REF, linestyle='--', linewidth=1)
    ax.set_xlabel('e_jk')
    ax.set_ylabel('e_kj')
    ax.set_title(f'23. Cross-elasticity asymmetry (rank-1 spec)\n{_abbrev(rank1_spec)}')
    sns.despine(ax=ax)
    _save(fig, out_dir, '23_elasticity_asymmetry.png')


def plot_elasticity_spec_spearman(elas, df_long, out_dir):
    """24. Heatmap of pairwise Spearman correlations of own-elas across specs."""
    best = sio.best_per_spec(df_long)
    rows = []
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)
                   & elas.own_price]
        for _, r in sub.iterrows():
            rows.append((spec, int(r.product_j), float(r.elasticity)))
    d = pd.DataFrame(rows, columns=['spec', 'product', 'elas'])
    pivot = d.pivot(index='product', columns='spec', values='elas')
    rho = pivot.rank().corr()
    order = rho.mean().sort_values(ascending=False).index.tolist()
    rho = rho.loc[order, order]

    fig, ax = plt.subplots(figsize=(max(8, 0.18 * len(order)),
                                    max(7, 0.18 * len(order))))
    sns.heatmap(rho, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Spearman rho'}, ax=ax,
                xticklabels=[_abbrev(s) for s in order],
                yticklabels=[_abbrev(s) for s in order])
    ax.tick_params(axis='x', labelsize=6, rotation=90)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_title('24. Spearman correlation of own-price elasticities across specs')
    _save(fig, out_dir, '24_elasticity_spec_spearman.png')


def plot_elasticity_firm_substitution(elas, df_long, out_dir):
    """25. 5x5 firm-substitution heatmap for the rank-1 spec."""
    starts = sio.starts_table(df_long)
    best = sio.best_per_spec(df_long)
    rank1_spec = sorted(
        [(spec, float(starts[(starts.spec_label == spec)
                             & (starts.start_id == sid)].objective.iloc[0]))
         for spec, sid in best.items()], key=lambda x: x[1])[0][0]
    rank1_sid = best[rank1_spec]
    sub = elas[(elas.spec_label == rank1_spec) & (elas.start_id == rank1_sid)
               & ~elas.own_price]
    if sub.empty:
        print(f'  [skip 25: rank-1 spec {rank1_spec!r} start {rank1_sid} '
              f'has no cross-price elasticities]')
        return
    pivot = sub.groupby(['firm_j', 'firm_k']).elasticity.mean().unstack()

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    vmax = float(np.abs(pivot.values).max())
    sns.heatmap(pivot, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                annot=True, fmt='.3f', ax=ax,
                cbar_kws={'label': 'mean cross-elasticity'})
    ax.set_xlabel('firm k')
    ax.set_ylabel('firm j')
    ax.set_title(f'25. Within/between-firm mean cross-elasticity\n'
                 f'rank-1 spec: {_abbrev(rank1_spec)}')
    _save(fig, out_dir, '25_elasticity_firm_substitution.png')


def plot_elasticity_own_cross_spec_stability(elas, df_long, out_dir):
    """26. Per-product own-elas across specs (stripplot)."""
    best = sio.best_per_spec(df_long)
    rows = []
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)
                   & elas.own_price]
        for _, r in sub.iterrows():
            rows.append({'spec': spec, 'product': int(r.product_j),
                         'firm': int(r.firm_j), 'elas': float(r.elasticity)})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.stripplot(data=d, x='product', y='elas', hue='firm',
                  palette='tab10', jitter=0.2, size=4, ax=ax)
    _clip_with_outlier_markers(
        ax, d['product'].tolist(), d.elas.tolist(), orient='v',
        colors=_hue_colors(d.elas, d.firm))
    ax.set_xlabel('product')
    ax.set_ylabel('own-price elasticity (best perturbed start)')
    ax.set_title('26. Own-price elasticity per product across specs')
    sns.despine(ax=ax)
    _save(fig, out_dir, '26_elasticity_own_cross_spec_stability.png')


def plot_elasticity_cross_cross_spec_stability(elas, df_long, out_dir, k=5):
    """27. Top-k substitute pair elasticities across specs (stripplot)."""
    starts = sio.starts_table(df_long)
    best = sio.best_per_spec(df_long)
    rank1_spec = sorted(
        [(spec, float(starts[(starts.spec_label == spec)
                             & (starts.start_id == sid)].objective.iloc[0]))
         for spec, sid in best.items()], key=lambda x: x[1])[0][0]
    rank1_sid = best[rank1_spec]
    rank1 = elas[(elas.spec_label == rank1_spec) & (elas.start_id == rank1_sid)
                 & ~elas.own_price]
    top = rank1.sort_values('elasticity', ascending=False).head(k)
    pairs = list(zip(top.product_j.astype(int), top.product_k.astype(int)))

    rows = []
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)]
        for (j, k_) in pairs:
            r = sub[(sub.product_j == j) & (sub.product_k == k_)]
            if not r.empty:
                rows.append({'pair': f'({j},{k_})', 'spec': spec,
                             'elas': float(r.elasticity.iloc[0])})
    d = pd.DataFrame(rows)
    if d.empty:
        print(f'  [skip 27: no cross-price elasticities for top-{k} '
              f'substitute pairs of rank-1 spec {rank1_spec!r}]')
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.stripplot(data=d, x='pair', y='elas', color=PALETTE[0],
                  jitter=0.18, size=4, ax=ax)
    pair_pos = {p: i for i, p in enumerate(dict.fromkeys(d.pair))}
    _clip_with_outlier_markers(
        ax, [pair_pos[p] for p in d.pair], d.elas.tolist(), orient='v',
        color=PALETTE[0])
    ax.set_xlabel('(j,k) pair')
    ax.set_ylabel('cross-price elasticity')
    ax.set_title(f'27. Top-{k} substitute pair elasticities across specs')
    sns.despine(ax=ax)
    _save(fig, out_dir, '27_elasticity_cross_cross_spec_stability.png')


def plot_elasticity_pairwise_mad(elas, df_long, out_dir):
    """28. 60x60 MAD heatmap of own-price elasticities between specs."""
    best = sio.best_per_spec(df_long)
    own_by_spec = {}
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
    mean_mad = (mat.sum(axis=1) - np.diag(mat)) / (n - 1)
    order = np.argsort(mean_mad)
    mat_ord = mat[np.ix_(order, order)]
    labels = [_abbrev(specs[i]) for i in order]

    fig, ax = plt.subplots(figsize=(max(8, 0.18 * n), max(7, 0.18 * n)))
    sns.heatmap(mat_ord, cmap='magma_r',
                cbar_kws={'label': 'mean |e_jj - e_jj_other|'},
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.tick_params(axis='x', labelsize=6, rotation=90)
    ax.tick_params(axis='y', labelsize=6)
    ax.set_title('28. Pairwise MAD of own-price elasticities between specs')
    _save(fig, out_dir, '28_elasticity_pairwise_mad.png')


def plot_elasticity_pair_across_sims(elas, df_long, truth_elas, out_dir):
    """30. 4-panel: e_jj, e_kk, e_jk, e_kj across perturbed starts of best-mean-obj spec."""
    starts = sio.starts_table(df_long)
    pert = starts[~starts.is_truth_start]
    spec = pert.groupby('spec_label').objective.mean().sort_values().index[0]
    j, k = 0, 1
    sub = elas[(elas.spec_label == spec) & ~elas.is_truth_start]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    quads = [('e_jj', j, j), ('e_kk', k, k), ('e_jk', j, k), ('e_kj', k, j)]
    for ax, (name, jj, kk) in zip(axes.flat, quads):
        vals = []
        for sid, ssub in sub.groupby('start_id'):
            r = ssub[(ssub.product_j == jj) & (ssub.product_k == kk)]
            if not r.empty:
                vals.append((int(sid), float(r.elasticity.iloc[0])))
        if vals:
            xs, ys = zip(*sorted(vals))
            ax.scatter(xs, ys, s=40, color=PALETTE[0], zorder=3,
                       edgecolor='white')
            _clip_with_outlier_markers(ax, list(xs), list(ys), orient='v',
                                       color=PALETTE[0])
        if truth_elas is not None:
            r = truth_elas[(truth_elas.product_j == jj)
                           & (truth_elas.product_k == kk)]
            if not r.empty:
                ax.axhline(float(r.elasticity.iloc[0]),
                           color='black', linestyle='--', linewidth=1,
                           label=f'truth = {float(r.elasticity.iloc[0]):.3f}')
                ax.legend(loc='lower right', fontsize=7)
        ax.set_title(f'{name}  (j={jj}, k={kk})')
        ax.set_xlabel('start_id')
        ax.set_ylabel('elasticity')
        sns.despine(ax=ax)
    fig.suptitle(f'30. Pair (j={j}, k={k}) across perturbed starts '
                 f'(best-mean-obj spec: {_abbrev(spec)})')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out_dir, '30_elasticity_pair_across_sims.png')


def plot_elasticity_pair_best_sim_across_specs(elas, df_long, truth_elas, out_dir):
    """31. 4-panel: same (j,k) pair across specs, best-start each, ranked by obj."""
    j, k = 0, 1
    best = sio.best_per_spec(df_long)
    starts = sio.starts_table(df_long)
    rows = []
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)]
        obj = float(starts[(starts.spec_label == spec)
                           & (starts.start_id == sid)].objective.iloc[0])
        def _e(jj, kk):
            r = sub[(sub.product_j == jj) & (sub.product_k == kk)]
            return float(r.elasticity.iloc[0]) if not r.empty else float('nan')
        rows.append({'spec': spec, 'obj': obj,
                     'e_jj': _e(j, j), 'e_kk': _e(k, k),
                     'e_jk': _e(j, k), 'e_kj': _e(k, j)})
    d = pd.DataFrame(rows).sort_values('obj').reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, (name, jj, kk) in zip(axes.flat,
                                  [('e_jj', j, j), ('e_kk', k, k),
                                   ('e_jk', j, k), ('e_kj', k, j)]):
        ax.scatter(d.obj, d[name], s=30, color=PALETTE[0],
                   edgecolor='white')
        _clip_with_outlier_markers(ax, d.obj.tolist(), d[name].tolist(),
                                   orient='v', color=PALETTE[0])
        if truth_elas is not None:
            r = truth_elas[(truth_elas.product_j == jj)
                           & (truth_elas.product_k == kk)]
            if not r.empty:
                ax.axhline(float(r.elasticity.iloc[0]),
                           color='black', linestyle='--', linewidth=1,
                           label=f'truth = {float(r.elasticity.iloc[0]):.3f}')
                ax.legend(loc='lower right', fontsize=7)
        ax.set_title(f'{name}  (j={jj}, k={kk})')
        ax.set_xlabel('GMM objective')
        ax.set_ylabel('elasticity')
        sns.despine(ax=ax)
    fig.suptitle(f'31. Pair (j={j}, k={k}) across specs (best perturbed start each)')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out_dir, '31_elasticity_pair_best_sim_across_specs.png')


def plot_elasticity_recovery_own(elas, df_long, truth_elas, out_dir):
    """33. Per-spec own-elasticity RMSE vs truth (horizontal bar)."""
    best = sio.best_per_spec(df_long)
    truth_own = truth_elas[truth_elas.own_price].set_index('product_j')['elasticity']
    rows = []
    for spec, sid in best.items():
        sub = elas[(elas.spec_label == spec) & (elas.start_id == sid)
                   & elas.own_price].set_index('product_j')['elasticity']
        common = sub.index.intersection(truth_own.index)
        err = (sub.loc[common] - truth_own.loc[common]).to_numpy()
        rows.append({'spec': spec,
                     'rmse': float(np.sqrt(np.mean(err ** 2))),
                     'bias': float(np.mean(err))})
    d = pd.DataFrame(rows).sort_values('rmse')

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(d))))
    y = np.arange(len(d))
    colors = [COL_NEAR if b >= 0 else COL_FAR for b in d.bias]
    ax.barh(y, d.rmse, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([_abbrev(s) for s in d.spec], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('RMSE(estimate - truth) on own-price elasticity')
    ax.set_title('33. Own-elasticity recovery vs. truth  '
                 '(orange = negative bias, green = positive)')
    sns.despine(ax=ax)
    _save(fig, out_dir, '33_elasticity_recovery_own.png')


def plot_elasticity_recovery_cross(elas, df_long, truth_elas, out_dir):
    """34. Per-spec MAE: same-firm vs between-firm (grouped bar)."""
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
        sf = truth_jk.loc[common, 'same_firm'].astype(bool).to_numpy()
        err = (s - t).to_numpy()
        rows.append({'spec': spec,
                     'mae_same': float(np.mean(np.abs(err[sf]))),
                     'mae_diff': float(np.mean(np.abs(err[~sf])))})
    d = pd.DataFrame(rows).sort_values('mae_same')

    fig, ax = plt.subplots(figsize=(8, _fig_height(len(d))))
    y = np.arange(len(d))
    h = 0.4
    ax.barh(y - h/2, d.mae_same, height=h, color=GROUP_COLORS['sigma'],
            label='same firm')
    ax.barh(y + h/2, d.mae_diff, height=h, color=GROUP_COLORS['pi'],
            label='different firm')
    ax.set_yticks(y)
    ax.set_yticklabels([_abbrev(s) for s in d.spec], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('MAE on cross-elasticity (estimate - truth)')
    ax.set_title('34. Cross-elasticity recovery: same-firm vs different-firm')
    ax.legend(loc='lower right')
    sns.despine(ax=ax)
    _save(fig, out_dir, '34_elasticity_recovery_cross.png')


def plot_post_estimation_recovery(post, truth_post, df_long, out_dir):
    """35. Scatter of merger-prediction error vs parameter-RMSE per spec."""
    truth = truth_post.iloc[0]
    best = sio.best_per_spec(df_long)
    # parameter RMSE per spec (from analyses #08-#09 logic)
    rec_rmse = {}
    for spec, sid in best.items():
        sub = df_long[(df_long.spec_label == spec) & (df_long.start_id == sid)]
        rec_rmse[spec] = float(np.sqrt(np.mean(sub.abs_error.values ** 2)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    targets = [('mean_own_elas', 'own-elas error'),
               ('mean_markup', 'markup error'),
               ('mean_delta_hhi', 'Δ-HHI (merger) error')]
    for ax, (col, label) in zip(axes, targets):
        xs, ys = [], []
        for _, r in post.iterrows():
            if pd.notna(r[col]) and pd.notna(truth[col]):
                xs.append(rec_rmse.get(r.spec_label, np.nan))
                ys.append(float(r[col]) - float(truth[col]))
        ax.scatter(xs, ys, s=30, color=PALETTE[0], alpha=0.85,
                   edgecolor='white')
        ax.axhline(0, color=COL_REF, linestyle='--', linewidth=1)
        ax.set_xlabel('parameter total RMSE')
        ax.set_ylabel(f'{label}  (estimate - truth)')
        ax.set_title(col)
        sns.despine(ax=ax)
    fig.suptitle('35. Post-estimation recovery vs. parameter recovery')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, out_dir, '35_post_estimation_recovery.png')


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

        _safe_plot(plot_objective_ranking, df, out_dir)
        _safe_plot(plot_multistart_stability, df, out_dir)
        _safe_plot(plot_objective_spec_comparison, df, out_dir)
        _safe_plot(plot_convergence_audit, df, out_dir)
        _safe_plot(plot_global_minimum, df, out_dir)
        _safe_plot(plot_two_basin, df, out_dir, args.basin_threshold)
        _safe_plot(plot_runtime, df, out_dir)
        _safe_plot(plot_price_coef, df, out_dir)
        _safe_plot(plot_recovery_rmse, df, out_dir)
        _safe_plot(plot_recovery_vs_objective, df, out_dir)
        _safe_plot(plot_param_level_error, df, out_dir)
        _safe_plot(plot_pi_zero, df, out_dir)
        _safe_plot(plot_omitted_x2, df, out_dir)
        _safe_plot(plot_demo_overfit, df, out_dir)
        _safe_plot(plot_param_stability, df, out_dir)
        _safe_plot(plot_best_vs_truth_start, df, out_dir)

        # Elasticity plots (20-35) -- only when CSVs are present.
        if sio.has_elasticities(specs_dir):
            elas = sio.load_elasticities(specs_dir)
            post = sio.load_post_estimation(specs_dir)
            seed_dir = specs_dir.parent.parent
            truth_elas = (sio.load_truth_elasticities(seed_dir)
                          if sio.has_truth_elasticities(seed_dir) else None)
            truth_post = (sio.load_truth_post_estimation(seed_dir)
                          if (seed_dir / 'truth_post_estimation.csv').exists()
                          else None)

            _safe_plot(plot_elasticity_own_summary, elas, df, truth_elas, out_dir)
            _safe_plot(plot_elasticity_multistart_stability, elas, df, out_dir)
            _safe_plot(plot_elasticity_top_substitutes, elas, df, out_dir)
            _safe_plot(plot_elasticity_asymmetry, elas, df, out_dir)
            _safe_plot(plot_elasticity_spec_spearman, elas, df, out_dir)
            _safe_plot(plot_elasticity_firm_substitution, elas, df, out_dir)
            _safe_plot(plot_elasticity_own_cross_spec_stability, elas, df, out_dir)
            _safe_plot(plot_elasticity_cross_cross_spec_stability, elas, df, out_dir)
            _safe_plot(plot_elasticity_pairwise_mad, elas, df, out_dir)
            _safe_plot(plot_elasticity_pair_across_sims, elas, df, truth_elas, out_dir)
            _safe_plot(plot_elasticity_pair_best_sim_across_specs, elas, df, truth_elas, out_dir)
            if truth_elas is not None:
                _safe_plot(plot_elasticity_recovery_own, elas, df, truth_elas, out_dir)
                _safe_plot(plot_elasticity_recovery_cross, elas, df, truth_elas, out_dir)
            if truth_post is not None:
                _safe_plot(plot_post_estimation_recovery, post, truth_post, df, out_dir)
        else:
            print(f'[skip elasticity plots for seed={seed} iv={iv_mode}: '
                  f'run compute_elasticities.py first]')

        per_iv.setdefault(iv_mode, {})[seed] = df

    for iv_mode, seed_dfs in per_iv.items():
        if len(seed_dfs) < 2:
            print(f'[skip cross-seed plots for iv_{iv_mode}: only {len(seed_dfs)} seed]')
            continue
        flat = len(per_iv) == 1
        cross_dir = root / 'graphs' if flat else root / 'graphs' / f'iv_{iv_mode}'
        print(f'plot: cross-seed iv={iv_mode} -> {cross_dir.relative_to(root.parent)}')
        _safe_plot(plot_bias_across_seeds, seed_dfs, cross_dir)
        _safe_plot(plot_recovery_across_seeds, seed_dfs, cross_dir)
        _safe_plot(plot_best_spec_consistency, seed_dfs, cross_dir)
        _safe_plot(plot_price_coef_across_seeds, seed_dfs, cross_dir)


if __name__ == '__main__':
    main()
