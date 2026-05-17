"""Side-by-side recovery comparison of two instrument-set variants.

Reads output/seed_N/iv_both/ and output/seed_N/iv_diff_only/, picks each
variant's best converged start, pulls 95% CIs from pyblp's standard errors,
and renders one figure with four caterpillar panels (beta, sigma, pi, gamma).
Each parameter row shows:
  * an open diamond at truth,
  * a filled dot + 95% CI horizontal whisker for iv_both (color A),
  * a filled dot + 95% CI horizontal whisker for iv_diff_only (color B),
    vertically offset within the row so the two CIs don't overlap.

Top-of-figure stats summarize objective, mean |error|, and CI coverage per
variant. Output: output/seed_N/iv_compare.png + .svg.

Example:
  uv run python compare_iv_modes.py                  # seed 0
  uv run python compare_iv_modes.py --seed 1
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse labeling & grouping from the single-variant viz script.
from viz_recovery import (
    GROUPS, GROUP_LABEL, PARAM_PRETTY, NL_NAMES, DEMO_NAMES,
    load_recovery_table, param_group,
)

VARIANTS = ("both", "diff_only")
VARIANT_COLOR = {
    "both":      "#1f77b4",  # blue
    "diff_only": "#d62728",  # red
}
VARIANT_LEGEND = {
    "both":      "iv_both (BLP + GH-diff)",
    "diff_only": "iv_diff_only (GH-diff only)",
}
ROW_OFFSET = 0.18  # vertical separation between the two variants within a row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None,
                   help="default: output/seed_{seed}/")
    return p.parse_args()


def pi_pretty(k: int, d: int) -> str:
    return rf"$\pi_{{{NL_NAMES[k]},\,{DEMO_NAMES[d]}}}$"


def _fallback_label(name: str) -> str:
    if name.startswith("pi_"):
        _, k, d = name.split("_")
        return pi_pretty(int(k), int(d))
    return name


def label_for(name: str) -> str:
    return PARAM_PRETTY.get(name) or _fallback_label(name)


def load_both(seed_dir: str) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    rows_by_variant: dict[str, pd.DataFrame] = {}
    meta_by_variant: dict[str, dict] = {}
    for v in VARIANTS:
        variant_dir = os.path.join(seed_dir, f"iv_{v}")
        if not os.path.isdir(variant_dir):
            raise FileNotFoundError(
                f"missing variant directory: {variant_dir}. "
                f"Run estimate.py --iv-mode {v} first."
            )
        rows, meta = load_recovery_table(variant_dir)
        rows_by_variant[v] = rows
        meta_by_variant[v] = meta
    return rows_by_variant, meta_by_variant


def aligned_table(rows_by_variant: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build one long table: rows = parameters that are estimated in BOTH variants.

    Each parameter appears once; columns hold both variants' estimates / SEs /
    truth (truth is invariant across variants by construction).
    """
    a = rows_by_variant["both"][["param_name", "group", "label", "truth",
                                  "estimate", "se"]].rename(
        columns={"estimate": "est_both", "se": "se_both"}
    )
    b = rows_by_variant["diff_only"][["param_name", "estimate", "se"]].rename(
        columns={"estimate": "est_diff", "se": "se_diff"}
    )
    merged = a.merge(b, on="param_name", how="inner")
    return merged


def make_figure(table: pd.DataFrame,
                meta_by_variant: dict[str, dict],
                seed: int) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(
        1, 4, wspace=0.55,
        left=0.06, right=0.985, top=0.74, bottom=0.10,
    )

    for col, grp in enumerate(GROUPS):
        ax = fig.add_subplot(gs[0, col])
        sub = table[table["group"] == grp].copy()
        sub = sub.sort_values("truth", kind="stable", ascending=True).reset_index(drop=True)
        y = np.arange(len(sub))

        for v, sign in [("both", +1), ("diff_only", -1)]:
            color = VARIANT_COLOR[v]
            est = sub[f"est_{ 'both' if v == 'both' else 'diff' }"].to_numpy()
            se  = sub[f"se_{  'both' if v == 'both' else 'diff' }"].to_numpy()
            y_off = y + sign * ROW_OFFSET
            ax.hlines(y_off, est - 1.96 * se, est + 1.96 * se,
                      color=color, lw=2.0, alpha=0.85)
            ax.plot(est, y_off, "o", color=color, ms=5.5, zorder=3,
                    label=VARIANT_LEGEND[v] if col == 0 else None)

        # Truth diamond, centered on the row.
        ax.plot(sub["truth"], y, marker="D", linestyle="", ms=7.5,
                markerfacecolor="white", markeredgecolor="#222222",
                markeredgewidth=1.3, zorder=4,
                label="truth" if col == 0 else None)
        ax.axvline(0, color="#aaaaaa", lw=0.7, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"], fontsize=9)
        ax.set_title(GROUP_LABEL[grp], fontsize=10, pad=6)
        ax.margins(y=0.10)
        if col == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=8,
                      bbox_to_anchor=(0.0, -0.06))

    # Top-of-figure stats annotation.
    def variant_stats(v: str, key_est: str, key_se: str) -> str:
        est = table[key_est].to_numpy()
        se  = table[key_se].to_numpy()
        tr  = table["truth"].to_numpy()
        within = (np.abs(est - tr) <= 1.96 * se).sum()
        return (
            f"{VARIANT_LEGEND[v]}\n"
            f"  GMM objective         : {meta_by_variant[v]['best_obj']:.3g}\n"
            f"  best start            : #{meta_by_variant[v]['best_start']}\n"
            f"  mean |est − truth|    : {np.abs(est - tr).mean():.3f}\n"
            f"  truth inside 95% CI   : {within} / {len(table)}"
        )

    txt_left  = variant_stats("both",      "est_both", "se_both")
    txt_right = variant_stats("diff_only", "est_diff", "se_diff")
    fig.text(0.06, 0.92, txt_left,  ha="left", va="top",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=VARIANT_COLOR["both"], lw=1.0))
    fig.text(0.55, 0.92, txt_right, ha="left", va="top",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=VARIANT_COLOR["diff_only"], lw=1.0))

    fig.suptitle(
        f"BLP / RCL recovery — seed_{seed} — instrument-set comparison",
        fontsize=14, y=0.97, x=0.06, ha="left", weight="bold",
    )
    return fig


def main() -> None:
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    seed_dir = args.output_dir or os.path.join(here, "output", f"seed_{args.seed}")

    rows_by_variant, meta_by_variant = load_both(seed_dir)
    table = aligned_table(rows_by_variant)
    print(f"comparing {len(table)} parameters estimated in both variants")
    for v in VARIANTS:
        m = meta_by_variant[v]
        print(f"  iv_{v:<10s}: best start #{m['best_start']}  objective={m['best_obj']:.4g}")

    fig = make_figure(table, meta_by_variant, seed=args.seed)
    png_path = os.path.join(seed_dir, "iv_compare.png")
    svg_path = os.path.join(seed_dir, "iv_compare.svg")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
