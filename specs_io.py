"""specs_io.py -- shared loaders/helpers for analyze_specs.py and plot_specs.py.

Discovers the multi-spec output tree under output/multiple_specs/, loads each
per-(seed, iv_mode) long-format results CSV, and enriches it with:

  - `is_truth_start`  True iff `tag == 'truth'` (start initialized at DGP truth,
                      not a perturbation). Treat differently from the random
                      restarts in any multistart-style summary.
  - `param_group`     'beta' | 'sigma' | 'pi' | 'gamma'
  - `param_physical`  Spec-invariant name for cross-spec aggregation, e.g.
                      'sigma_x1', 'pi_prices_income'. For beta/gamma this is
                      identical to `param_name` (indices are spec-invariant).

The X2 / agent formulas needed for the physical mapping are pulled from each
spec's `spec.json`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import pandas as pd


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover(root: Path) -> Iterator[tuple[int, str, Path]]:
    """Yield (seed, iv_mode, specs_dir) for every populated spec sweep under root.

    `root` should be `output/multiple_specs/`. Skips sweeps missing the
    long-format CSV (treated as not-yet-aggregated).
    """
    for seed_dir in sorted(root.glob('seed_*')):
        m = re.match(r'seed_(\d+)$', seed_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        for iv_dir in sorted(seed_dir.glob('iv_*')):
            iv_mode = iv_dir.name[len('iv_'):]
            specs_dir = iv_dir / 'specs'
            if (specs_dir / 'specs_summary_long.csv').exists():
                yield seed, iv_mode, specs_dir


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

_TERM_RE = re.compile(r'\s*\+\s*')


def _parse_formula(formula: str, drop_zero: bool = False) -> list[str]:
    """Split a pyblp-style formula like '1 + prices + x1 + x2' into its terms.

    If `drop_zero` (agent_formula starts with '0 +'), drop the leading '0'.
    """
    terms = [t.strip() for t in _TERM_RE.split(formula) if t.strip()]
    if drop_zero and terms and terms[0] == '0':
        terms = terms[1:]
    return terms


def _term_label(term: str) -> str:
    """Pretty label for a formula term (the constant becomes 'const')."""
    return 'const' if term == '1' else term


def _load_spec_formulas(specs_dir: Path) -> dict[str, dict]:
    """Return {spec_label: {'x2_terms': [...], 'demo_terms': [...]}}."""
    out: dict[str, dict] = {}
    for spec_dir in specs_dir.glob('spec_*'):
        spec_label = spec_dir.name[len('spec_'):]
        with (spec_dir / 'spec.json').open() as f:
            s = json.load(f)
        x2 = [_term_label(t) for t in _parse_formula(s['x2_formula'])]
        demo = [_term_label(t) for t in _parse_formula(s['agent_formula'], drop_zero=True)]
        out[spec_label] = {'x2_terms': x2, 'demo_terms': demo}
    return out


_SIGMA_RE = re.compile(r'^sigma_(\d+)_(\d+)$')
_PI_RE    = re.compile(r'^pi_(\d+)_(\d+)$')


def _physical_name(param_name: str, x2_terms: list[str], demo_terms: list[str]) -> str:
    """Map a positional param_name to a spec-invariant physical name.

    Examples:
      sigma_2_2, x2_terms=['const','prices','x1','x2']  -> 'sigma_x1'
      pi_1_0,   x2_terms=[...,'prices',...], demo=['income']  -> 'pi_prices_income'
      beta_1, gamma_0  -> unchanged
    """
    m = _SIGMA_RE.match(param_name)
    if m:
        i = int(m.group(1))
        if 0 <= i < len(x2_terms):
            return f'sigma_{x2_terms[i]}'
        return param_name
    m = _PI_RE.match(param_name)
    if m:
        i, j = int(m.group(1)), int(m.group(2))
        if 0 <= i < len(x2_terms) and 0 <= j < len(demo_terms):
            return f'pi_{x2_terms[i]}_{demo_terms[j]}'
        return param_name
    return param_name


def load_long(specs_dir: Path) -> pd.DataFrame:
    """Read specs_summary_long.csv and enrich it with derived columns."""
    df = pd.read_csv(specs_dir / 'specs_summary_long.csv')
    for c in ('truth', 'estimate', 'abs_error', 'objective', 'elapsed_sec'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['converged'] = df['converged'].astype(str).str.lower() == 'true'
    df['error_class'] = df['error_class'].fillna('').astype(str)
    df['is_truth_start'] = df['tag'].astype(str) == 'truth'
    df['param_group'] = df['param_name'].str.extract(r'^([a-z]+)', expand=False)

    formulas = _load_spec_formulas(specs_dir)
    def _phys(row):
        f = formulas.get(row['spec_label'])
        if f is None:
            return row['param_name']
        return _physical_name(row['param_name'], f['x2_terms'], f['demo_terms'])
    df['param_physical'] = df.apply(_phys, axis=1)
    # n_x2 / n_demos help group-by analyses
    df['n_x2'] = df['x2_vars'].fillna('').astype(str).apply(
        lambda s: 0 if not s else len([t for t in s.split(',') if t])
    )
    df['n_demos'] = df['demo_vars'].fillna('').astype(str).apply(
        lambda s: 0 if not s else len([t for t in s.split(',') if t])
    )
    return df


# ---------------------------------------------------------------------------
# Best-start selection
# ---------------------------------------------------------------------------

def best_per_spec(df: pd.DataFrame, include_truth_start: bool = False) -> dict[str, int]:
    """Return {spec_label: start_id} for the best start of each spec.

    Best = lowest objective among `converged == True`; falls back to lowest
    objective overall if no start converged. Starts that errored during the
    solve (`error_class` set) are excluded entirely: an errored solve produced
    no usable ProblemResults — it has no pickle, hence no elasticities — so it
    must never be crowned "best" (a spuriously low objective could otherwise win
    the fallback). A spec whose every (non-truth) start errored is omitted from
    the result rather than represented by an unusable start. The `error_class`
    filter is skipped if the column is absent (pre-diagnostics summaries).

    By default the truth-warm start (`is_truth_start=True`, started at the DGP
    parameter) is *excluded* — its presence is informative as a reference but it
    can dominate naive multistart comparisons (it always has near-zero recovery
    error by construction). Pass `include_truth_start=True` to keep it in the pool.
    """
    has_err = 'error_class' in df.columns
    cols = ['start_id', 'objective', 'converged', 'is_truth_start']
    if has_err:
        cols = cols + ['error_class']
    out: dict[str, int] = {}
    for spec, sub in df.groupby('spec_label'):
        starts = sub.drop_duplicates('start_id')[cols]
        if not include_truth_start:
            starts = starts[~starts['is_truth_start']]
        if has_err:
            errored = starts['error_class'].fillna('').astype(str).str.strip() != ''
            starts = starts[~errored]
        if starts.empty:
            continue
        valid = starts[starts['converged']]
        pool = valid if not valid.empty else starts
        out[spec] = int(pool.sort_values('objective').iloc[0]['start_id'])
    return out


def starts_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (spec_label, start_id): objective, converged, runtime, etc."""
    return (
        df.drop_duplicates(['spec_label', 'start_id'])[
            ['spec_label', 'start_id', 'tag', 'is_truth_start',
             'converged', 'objective', 'elapsed_sec', 'n_x2', 'n_demos']
        ]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Elasticity / post-estimation loaders (written by compute_elasticities.py)
# ---------------------------------------------------------------------------

def has_elasticities(specs_dir: Path) -> bool:
    """True iff elasticities_detail.csv and post_estimation_summary.csv exist."""
    return (specs_dir / 'elasticities_detail.csv').exists() \
        and (specs_dir / 'post_estimation_summary.csv').exists()


def load_elasticities(specs_dir: Path) -> pd.DataFrame:
    """Load elasticities_detail.csv with numeric coercion."""
    df = pd.read_csv(specs_dir / 'elasticities_detail.csv')
    df['converged'] = df['converged'].astype(str).str.lower() == 'true'
    df['is_truth_start'] = df['is_truth_start'].astype(str).str.lower() == 'true'
    df['own_price'] = df['own_price'].astype(str).str.lower() == 'true'
    df['same_firm'] = df['same_firm'].astype(int) == 1 \
        if df['same_firm'].dtype == 'O' else df['same_firm'].astype(bool)
    for c in ('elasticity',):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ('product_j', 'product_k', 'firm_j', 'firm_k', 'start_id'):
        df[c] = df[c].astype(int)
    return df


def load_post_estimation(specs_dir: Path) -> pd.DataFrame:
    """Load post_estimation_summary.csv (one row per spec, best perturbed start)."""
    df = pd.read_csv(specs_dir / 'post_estimation_summary.csv')
    for c in ('mean_own_elas', 'mean_outside_div', 'mean_markup', 'mean_hhi',
              'mean_delta_markup', 'mean_delta_hhi', 'mean_delta_cs'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def has_truth_elasticities(seed_dir: Path) -> bool:
    return (seed_dir / 'truth_elasticities.csv').exists()


def load_truth_elasticities(seed_dir: Path) -> pd.DataFrame:
    """Load truth_elasticities.csv for the given seed_dir."""
    df = pd.read_csv(seed_dir / 'truth_elasticities.csv')
    df['own_price'] = df['own_price'].astype(str).str.lower() == 'true'
    df['same_firm'] = df['same_firm'].astype(int) == 1 \
        if df['same_firm'].dtype == 'O' else df['same_firm'].astype(bool)
    df['elasticity'] = pd.to_numeric(df['elasticity'], errors='coerce')
    for c in ('product_j', 'product_k', 'firm_j', 'firm_k'):
        df[c] = df[c].astype(int)
    return df


def load_truth_post_estimation(seed_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(seed_dir / 'truth_post_estimation.csv')
    return df


def elasticities_best_per_spec(elas_df: pd.DataFrame, best_map: dict[str, int]
                               ) -> pd.DataFrame:
    """Filter elas_df to just the best-start row per spec."""
    frames = [elas_df[(elas_df.spec_label == s) & (elas_df.start_id == sid)]
              for s, sid in best_map.items()]
    return pd.concat(frames, ignore_index=True) if frames else elas_df.iloc[:0]
