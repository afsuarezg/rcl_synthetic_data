"""Heatmap of recovery error across BLP specifications.

Reads output/seed_{seed}/iv_{mode}/specs/specs_summary_best.csv (the rollup
written by run_specs.py --aggregate-only) and renders one figure where rows
are specs and columns are parameters; cell color encodes |estimate − truth|.
Cells where a spec didn't estimate a given parameter are grey (NaN).

Example:
  uv run python viz_specs.py                       # seed 0, iv_both
  uv run python viz_specs.py --seed 0 --iv-mode both
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iv-mode", choices=["both", "diff_only"], default="both")
    p.add_argument("--output-dir", type=str, default=None,
                   help="default: output/seed_{seed}/")
    return p.parse_args()


# Parameter ordering: linear demand → sigma diag → pi cells → gamma.
def param_sort_key(name: str) -> tuple[int, str]:
    if name.startswith("beta_"):
        return (0, name)
    if name.startswith("sigma_"):
        return (1, name)
    if name.startswith("pi_"):
        return (2, name)
    if name.startswith("gamma_"):
        return (3, name)
    return (9, name)


def spec_sort_key(label: str) -> tuple[int, int, str]:
    """Sort specs by (X2 size desc, demos size desc, label) so similar
    specs cluster in the figure."""
    # Label format: x2-x1_x2_x3__demos-income_age
    try:
        x2_part, demo_part = label.split("__")
        x2_vars = x2_part.removeprefix("x2-").split("_") if x2_part != "x2-" else []
        demo_vars = demo_part.removeprefix("demos-").split("_") if demo_part != "demos-" else []
        return (-len(x2_vars), -len(demo_vars), label)
    except Exception:
        return (0, 0, label)


def main() -> None:
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    seed_dir = args.output_dir or os.path.join(here, "output", f"seed_{args.seed}")
    specs_dir = os.path.join(seed_dir, f"iv_{args.iv_mode}", "specs")
    best_csv = os.path.join(specs_dir, "specs_summary_best.csv")
    if not os.path.exists(best_csv):
        raise SystemExit(f"missing {best_csv} — run "
                         f"`uv run python run_specs.py --seed {args.seed} "
                         f"--iv-mode {args.iv_mode} --aggregate-only` first")
    df = pd.read_csv(best_csv)

    # Wide matrix: rows = specs, cols = params, value = abs_error.
    wide = df.pivot_table(index="spec_label", columns="param_name",
                          values="abs_error", aggfunc="first")
    # Sort.
    wide = wide.reindex(sorted(wide.index, key=spec_sort_key))
    wide = wide.reindex(columns=sorted(wide.columns, key=param_sort_key))

    n_specs, n_params = wide.shape
    print(f"heatmap: {n_specs} specs × {n_params} parameters")

    # Figure size scales with grid. Cells ≈ 12px wide, 14px tall.
    fig_w = max(8.0, 0.16 * n_params + 4.0)
    fig_h = max(5.0, 0.20 * n_specs + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h),
                           gridspec_kw={"left": 0.30, "right": 0.94,
                                        "top": 0.94, "bottom": 0.16})

    # log10 of |err| + floor → keep magnitudes comparable across many decades.
    floor = 1e-3
    data = np.log10(wide.to_numpy() + floor)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#dddddd")
    masked = np.ma.masked_invalid(data)

    vmax = float(np.nanmax(masked))
    vmin = float(np.nanmin(masked))
    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
                   interpolation="nearest")

    ax.set_xticks(np.arange(n_params))
    ax.set_xticklabels(wide.columns, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(n_specs))
    ax.set_yticklabels(wide.index, fontsize=7)
    ax.set_xlabel("parameter")
    ax.set_ylabel("specification")
    ax.set_title(
        f"BLP recovery error across specs — seed_{args.seed}  (iv_{args.iv_mode})\n"
        f"cell color = log₁₀(|estimate − truth| + {floor:g});  grey = not estimated",
        fontsize=10, pad=8,
    )

    # Faint dividers between parameter groups.
    group_starts = []
    prev_group = None
    for j, name in enumerate(wide.columns):
        g = param_sort_key(name)[0]
        if g != prev_group and j > 0:
            group_starts.append(j)
        prev_group = g
    for j in group_starts:
        ax.axvline(j - 0.5, color="white", lw=1.2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(f"log₁₀(|est − truth| + {floor:g})", fontsize=8)

    png = os.path.join(specs_dir, "specs_heatmap.png")
    svg = os.path.join(specs_dir, "specs_heatmap.svg")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    print(f"wrote {png}")
    print(f"wrote {svg}")


if __name__ == "__main__":
    main()
