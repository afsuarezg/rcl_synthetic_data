"""compute_elasticities.py -- post-hoc elasticity & merger-sim export.

Walks output/multiple_specs/seed_*/iv_*/specs/spec_*/estimates/start_NN.pkl,
loads each pyblp.ProblemResults, computes elasticities (averaged across
markets), and writes two CSVs per (seed, iv) sweep:

  output/multiple_specs/seed_X/iv_Y/specs/elasticities_detail.csv
      Columns: spec_label, start_id, tag, is_truth_start, converged,
               product_j, product_k, firm_j, firm_k,
               elasticity, own_price, same_firm
      One row per (spec, start, j, k); 60 specs x 5 starts x 100 (j,k) pairs
      = 30,000 rows for the default sweep.

  output/multiple_specs/seed_X/iv_Y/specs/post_estimation_summary.csv
      Columns: spec_label, start_id,
               mean_own_elas, mean_outside_div,
               mean_markup, mean_hhi,
               mean_delta_markup, mean_delta_hhi, mean_delta_cs
      One row per spec (best perturbed start only). Merger sim is firm 2
      acquires firm 1 (BLP tutorial convention).

And two CSVs per seed (truth depends only on seed, not on iv mode):

  output/multiple_specs/seed_X/truth_elasticities.csv
      Same schema as elasticities_detail.csv (without spec/start columns).
  output/multiple_specs/seed_X/truth_post_estimation.csv
      Same supply/demand/merger schema, evaluated at the DGP truth.

Usage:
  uv run python compute_elasticities.py
  uv run python compute_elasticities.py --root output/multiple_specs
  uv run python compute_elasticities.py --skip-merger        # demand only
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp

import specs_io as sio


# ---------------------------------------------------------------------------
# pyblp options (mirror estimate.py)
# ---------------------------------------------------------------------------
pyblp.options.collinear_atol = pyblp.options.collinear_rtol = 0.0
pyblp.options.singular_tol = 1e-14
pyblp.options.pseudo_inverses = True


# ---------------------------------------------------------------------------
# Synthetic product-id helpers
# ---------------------------------------------------------------------------

def detect_firm_pattern(product_data: pd.DataFrame) -> np.ndarray:
    """Return the firm-ids vector for one market (assumed identical across markets)."""
    first = product_data['market_ids'].iloc[0]
    return product_data.loc[product_data['market_ids'] == first, 'firm_ids'].to_numpy()


# ---------------------------------------------------------------------------
# Elasticity aggregation
# ---------------------------------------------------------------------------

def elasticity_pairs(elasticities_flat, product_data: pd.DataFrame,
                     firm_pattern: np.ndarray) -> pd.DataFrame:
    """Aggregate a pyblp .compute_elasticities() result into one row per (j, k).

    `elasticities_flat` is the (N,) array returned by .compute_elasticities();
    each entry is a 1-D array of length J_t (the j-th row of the J_t x J_t
    matrix for that market). product_data is the data the result was solved on
    (sorted by market_ids).
    """
    product_data = product_data.sort_values('market_ids').reset_index(drop=True)
    markets = np.sort(product_data['market_ids'].unique())
    pair_vals: dict[tuple[int, int], list[float]] = {}
    flat_idx = 0
    for market_id in markets:
        mask = product_data['market_ids'] == market_id
        J_t = int(mask.sum())
        # Stack J_t length-J_t arrays -> J_t x J_t
        E_t = np.stack(list(elasticities_flat[flat_idx:flat_idx + J_t]))
        for j in range(J_t):
            for k in range(J_t):
                pair_vals.setdefault((j, k), []).append(float(E_t[j, k]))
        flat_idx += J_t

    rows = []
    for (j, k), vals in pair_vals.items():
        rows.append({
            'product_j': j,
            'product_k': k,
            'firm_j': int(firm_pattern[j]),
            'firm_k': int(firm_pattern[k]),
            'elasticity': float(np.mean(vals)),
            'own_price': j == k,
            'same_firm': int(firm_pattern[j]) == int(firm_pattern[k]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Post-estimation rows (demand, supply, merger)
# ---------------------------------------------------------------------------

def _diagonal_mean(matrix_per_market) -> float:
    """Mirror of pyblp's extract_diagonal_means(): mean over markets of the
    per-market diagonal entries (i.e., mean of own-price elasticity)."""
    vals: list[float] = []
    flat_idx = 0
    # matrix_per_market is shape (N,) of 1-D arrays length J_t (single row), so
    # for elasticities/diversion we re-stack to access the diagonal. Easier:
    # iterate per market and pull diag.
    return float(np.nan)  # not used -- replaced by structured loop below


def demand_columns(elasticities_flat, diversions_flat,
                   product_data: pd.DataFrame) -> tuple[float, float]:
    """Mean own-price elasticity and mean diversion to outside, averaged over
    markets and own products."""
    product_data = product_data.sort_values('market_ids').reset_index(drop=True)
    markets = np.sort(product_data['market_ids'].unique())
    own_elas, out_div = [], []
    flat_idx = 0
    for market_id in markets:
        mask = product_data['market_ids'] == market_id
        J_t = int(mask.sum())
        E = np.stack(list(elasticities_flat[flat_idx:flat_idx + J_t]))
        D = np.stack(list(diversions_flat[flat_idx:flat_idx + J_t]))
        own_elas.extend(np.diag(E).tolist())
        # Diversion ratio matrix in pyblp: rows = j, cols = k. The "outside" column
        # is the (J+1)-th column iff included; pyblp's compute_diversion_ratios()
        # returns an extra column for the outside good when name='all' (default).
        # Per-row sum of off-diag + outside should be ~1.
        # The outside diversion is the LAST column when D.shape[1] > J_t.
        if D.shape[1] > J_t:
            out_div.extend(D[:, -1].tolist())
        else:
            # Fallback: 1 - sum of inside diversions
            inside = D.sum(axis=1) - np.diag(D)  # off-diagonal inside sum
            out_div.extend((1.0 - inside).tolist())
        flat_idx += J_t
    return float(np.mean(own_elas)), float(np.mean(out_div))


def supply_columns(res, product_data: pd.DataFrame,
                   skip_merger: bool) -> dict[str, float]:
    """Compute markup, HHI, and (optionally) merger Δs. Returns NaN values on
    any pyblp error, mirroring blp_blp.py:485 except-clause."""
    out = {'mean_markup': np.nan, 'mean_hhi': np.nan,
           'mean_delta_markup': np.nan, 'mean_delta_hhi': np.nan,
           'mean_delta_cs': np.nan}
    try:
        costs = res.compute_costs()
        markups = res.compute_markups(costs=costs)
        hhi = res.compute_hhi()
        out['mean_markup'] = float(np.asarray(markups).mean())
        out['mean_hhi'] = float(np.asarray(hhi).mean())

        if skip_merger:
            return out

        # Merger: firm 2 acquires firm 1 (replicates BLP convention).
        merger_ids = product_data['firm_ids'].replace(2, 1).to_numpy()
        cs = res.compute_consumer_surpluses()
        new_prices = res.compute_prices(firm_ids=merger_ids, costs=costs)
        new_shares = res.compute_shares(new_prices)
        new_markups = res.compute_markups(new_prices, costs)
        new_hhi = res.compute_hhi(firm_ids=merger_ids, shares=new_shares)
        new_cs = res.compute_consumer_surpluses(new_prices)

        out['mean_delta_markup'] = float(np.asarray(new_markups - markups).mean())
        out['mean_delta_hhi'] = float(np.asarray(new_hhi - hhi).mean())
        out['mean_delta_cs'] = float(np.asarray(new_cs - cs).mean())
    except (AttributeError, pyblp.exceptions.MultipleErrors, Exception) as exc:
        warnings.warn(f'supply/merger sim failed: {exc.__class__.__name__}: {exc}')
    return out


def post_estimation_row(res, product_data: pd.DataFrame,
                        skip_merger: bool) -> dict[str, float]:
    """One row of post_estimation_summary.csv content for a single result."""
    elas = res.compute_elasticities()
    div = res.compute_diversion_ratios()
    mean_own, mean_out_div = demand_columns(elas, div, product_data)
    row = {'mean_own_elas': mean_own, 'mean_outside_div': mean_out_div}
    row.update(supply_columns(res, product_data, skip_merger))
    return row


# ---------------------------------------------------------------------------
# Truth elasticities (per-seed)
# ---------------------------------------------------------------------------

def build_truth_simulation(seed_dir: Path) -> pyblp.SimulationResults:
    """Reconstruct the DGP Simulation at truth and solve endogenous prices.

    Mirrors merger.py:79-101 but uses the original (pre-merger) firm_ids.
    """
    with (seed_dir / 'truth.pkl').open('rb') as fh:
        truth = pickle.load(fh)
    product = pd.read_csv(seed_dir / 'product_data.csv')
    agents = pd.read_csv(seed_dir / 'agent_data.csv')

    # Drop equilibrium outputs + ownership-dependent instruments so pyblp
    # rebuilds them at truth ownership.
    drop = [c for c in product.columns
            if c in ('prices', 'shares')
            or c.startswith('demand_instruments')
            or c.startswith('supply_instruments')]
    sim_input = product.drop(columns=drop)

    simulation = pyblp.Simulation(
        product_formulations=(
            pyblp.Formulation('1 + prices + x1 + x2 + x3 + x4 + x5'),
            pyblp.Formulation('1 + prices + x1 + x2 + x3'),
            pyblp.Formulation('1 + x1 + x2 + w1 + w2'),
        ),
        product_data=sim_input,
        beta=truth['beta'], sigma=truth['sigma'],
        pi=truth['pi'], gamma=truth['gamma'],
        agent_formulation=pyblp.Formulation('0 + income + age + hh_size + education'),
        agent_data=agents,
        xi=truth['xi'], omega=truth['omega'],
        costs_type='linear',
        seed=truth.get('seed'),
    )
    return simulation.replace_endogenous(
        iteration=pyblp.Iteration('simple', {'atol': 1e-12, 'max_evaluations': 5000}),
        error_behavior='warn',
    )


def truth_post_estimation_row(truth_res, product_data: pd.DataFrame,
                              skip_merger: bool) -> dict[str, float]:
    """Same demand+supply+merger columns evaluated at the DGP truth.

    SimulationResults exposes the same compute_* helpers as ProblemResults.
    """
    return post_estimation_row(truth_res, product_data, skip_merger)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root', type=Path, default=None,
                   help='multiple_specs root (default: ./output/multiple_specs)')
    p.add_argument('--skip-merger', action='store_true',
                   help='Skip merger simulation; emit demand+markup+HHI only.')
    p.add_argument('--force', action='store_true',
                   help='Re-run even if output CSVs already exist.')
    return p.parse_args()


def _load_starts_summary(specs_dir: Path) -> pd.DataFrame:
    """Read the long-format summary just to get per-start metadata."""
    return sio.load_long(specs_dir)


def main() -> None:
    args = parse_args()
    root = args.root or Path(__file__).parent / 'output' / 'multiple_specs'
    if not root.exists():
        raise SystemExit(f'Root not found: {root}')

    for seed_dir in sorted(root.glob('seed_*')):
        if not (seed_dir / 'truth.pkl').exists():
            print(f'skip {seed_dir.name}: no truth.pkl')
            continue
        seed_product = pd.read_csv(seed_dir / 'product_data.csv')
        firm_pattern = detect_firm_pattern(seed_product)
        print(f'\n=== {seed_dir.name}  firm pattern={firm_pattern.tolist()} ===')

        # --- Truth (per-seed) ---
        truth_elas_path = seed_dir / 'truth_elasticities.csv'
        truth_post_path = seed_dir / 'truth_post_estimation.csv'
        if args.force or not truth_elas_path.exists():
            print('  computing truth elasticities ...')
            truth_res = build_truth_simulation(seed_dir)
            t_elas = elasticity_pairs(
                truth_res.compute_elasticities(), seed_product, firm_pattern)
            t_elas.to_csv(truth_elas_path, index=False)
            t_post = pd.DataFrame([truth_post_estimation_row(
                truth_res, seed_product, args.skip_merger)])
            t_post.to_csv(truth_post_path, index=False)
            print(f'  -> {truth_elas_path.name}  ({len(t_elas)} rows)')
            print(f'  -> {truth_post_path.name}')
        else:
            print(f'  truth CSVs exist; skipping (use --force to recompute)')

        # --- Per-(seed, iv) sweep ---
        for iv_dir in sorted(seed_dir.glob('iv_*')):
            specs_dir = iv_dir / 'specs'
            if not (specs_dir / 'specs_summary_long.csv').exists():
                continue
            elas_path = specs_dir / 'elasticities_detail.csv'
            post_path = specs_dir / 'post_estimation_summary.csv'
            if not args.force and elas_path.exists() and post_path.exists():
                print(f'  iv_{iv_dir.name[3:]}: CSVs exist; skipping')
                continue

            df_long = _load_starts_summary(specs_dir)
            # one row per (spec_label, start_id)
            starts_meta = (df_long.drop_duplicates(['spec_label', 'start_id'])
                           [['spec_label', 'start_id', 'tag', 'is_truth_start',
                             'converged']])
            best_map = sio.best_per_spec(df_long)
            n_total = len(starts_meta)
            print(f'  iv_{iv_dir.name[3:]}: {n_total} pickles to process ...')

            detail_frames: list[pd.DataFrame] = []
            post_rows: list[dict] = []
            n_done = 0
            for spec_subdir in sorted(specs_dir.glob('spec_*')):
                spec_label = spec_subdir.name[len('spec_'):]
                spec_meta = starts_meta[starts_meta.spec_label == spec_label]
                best_sid = best_map.get(spec_label)

                for _, m in spec_meta.iterrows():
                    sid = int(m.start_id)
                    pkl = spec_subdir / 'estimates' / f'start_{sid:02d}.pkl'
                    if not pkl.exists():
                        warnings.warn(f'missing pickle: {pkl}')
                        continue
                    try:
                        with pkl.open('rb') as fh:
                            res = pickle.load(fh)
                        elas = res.compute_elasticities()
                    except Exception as exc:
                        warnings.warn(f'{pkl.name} failed: '
                                      f'{exc.__class__.__name__}: {exc}')
                        continue

                    pairs = elasticity_pairs(elas, seed_product, firm_pattern)
                    pairs['spec_label'] = spec_label
                    pairs['start_id'] = sid
                    pairs['tag'] = str(m.tag)
                    pairs['is_truth_start'] = bool(m.is_truth_start)
                    pairs['converged'] = bool(m.converged)
                    detail_frames.append(pairs)

                    if sid == best_sid:
                        row = {'spec_label': spec_label, 'start_id': sid}
                        row.update(post_estimation_row(
                            res, seed_product, args.skip_merger))
                        post_rows.append(row)

                    n_done += 1
                    if n_done % 50 == 0:
                        print(f'    {n_done}/{n_total} ...')

            detail_df = pd.concat(detail_frames, ignore_index=True)
            cols = ['spec_label', 'start_id', 'tag', 'is_truth_start', 'converged',
                    'product_j', 'product_k', 'firm_j', 'firm_k',
                    'elasticity', 'own_price', 'same_firm']
            detail_df = detail_df[cols]
            detail_df.to_csv(elas_path, index=False)
            pd.DataFrame(post_rows).to_csv(post_path, index=False)
            print(f'  -> {elas_path.relative_to(root.parent)}  '
                  f'({len(detail_df)} rows)')
            print(f'  -> {post_path.relative_to(root.parent)}  '
                  f'({len(post_rows)} rows)')


if __name__ == '__main__':
    main()
