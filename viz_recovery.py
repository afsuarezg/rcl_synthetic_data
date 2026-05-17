"""Visualize parameter recovery for a single estimation run.

Loads estimates_summary.csv and the corresponding pickled ProblemResults
from output/seed_{seed}/, picks the best converged start, pulls 95% CIs
from pyblp's standard errors, and writes recovery.png + recovery.svg:

  Top panel    estimate vs truth scatter (all estimated params), 45-degree
               identity line, vertical 95% CI whiskers, colored by group.
  Bottom panel four-column forest plot (one per parameter block: beta,
               sigma, pi, gamma) with estimate + 95% CI horizontal bar and
               truth marker.

Pi entries pinned to zero in the DGP are excluded — they aren't part of
the recovery test.

Example:
  uv run python viz_recovery.py                  # output/seed_0
  uv run python viz_recovery.py --seed 1
"""
from __future__ import annotations

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUPS = ("beta", "sigma", "pi", "gamma")
GROUP_COLORS = {
    "beta":  "#1f77b4",
    "sigma": "#ff7f0e",
    "pi":    "#2ca02c",
    "gamma": "#d62728",
}
GROUP_LABEL = {
    "beta":  r"$\beta$  (linear demand)",
    "sigma": r"$\Sigma$  (random-coef SDs)",
    "pi":    r"$\Pi$  (demographic interactions)",
    "gamma": r"$\gamma$  (marginal-cost shifters)",
}
PARAM_PRETTY = {
    # beta: 1 + prices + x1..x5
    "beta_0": r"$\beta_{\rm const}$",
    "beta_1": r"$\alpha$ (price)",
    "beta_2": r"$\beta_{x_1}$",
    "beta_3": r"$\beta_{x_2}$",
    "beta_4": r"$\beta_{x_3}$",
    "beta_5": r"$\beta_{x_4}$",
    "beta_6": r"$\beta_{x_5}$",
    # sigma diag: 1 + prices + x1 + x2 + x3
    "sigma_0_0": r"$\sigma_{\rm const}$",
    "sigma_1_1": r"$\sigma_{\rm price}$",
    "sigma_2_2": r"$\sigma_{x_1}$",
    "sigma_3_3": r"$\sigma_{x_2}$",
    "sigma_4_4": r"$\sigma_{x_3}$",
    # gamma: 1 + x1 + x2 + w1 + w2
    "gamma_0": r"$\gamma_{\rm const}$",
    "gamma_1": r"$\gamma_{x_1}$",
    "gamma_2": r"$\gamma_{x_2}$",
    "gamma_3": r"$\gamma_{w_1}$",
    "gamma_4": r"$\gamma_{w_2}$",
}
# pi rows index the nonlinear chars (1, prices, x1, x2, x3); cols index
# demographics (income, age, hh_size, education).
NL_NAMES = ("const", "price", "x_1", "x_2", "x_3")
DEMO_NAMES = ("income", "age", "hh\\_size", "educ")


def pi_pretty(k: int, d: int) -> str:
    return rf"$\pi_{{{NL_NAMES[k]},\,{DEMO_NAMES[d]}}}$"


def param_group(name: str) -> str:
    return name.split("_", 1)[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None,
                   help="default: output/seed_{seed}/")
    return p.parse_args()


def load_recovery_table(output_dir: str) -> tuple[pd.DataFrame, dict]:
    summary = pd.read_csv(os.path.join(output_dir, "estimates_summary.csv"))
    converged = summary[summary["converged"] & summary["objective"].notna()]
    if converged.empty:
        raise RuntimeError(f"no converged starts in {output_dir}")
    best_start = int(converged.loc[converged["objective"].idxmin(), "start_id"])
    best_obj = float(converged["objective"].min())

    pkl_path = os.path.join(output_dir, "estimates", f"start_{best_start:02d}.pkl")
    with open(pkl_path, "rb") as fh:
        res = pickle.load(fh)

    # Build SE lookup keyed by the same scheme as flatten_params() in estimate.py.
    se: dict[str, float] = {}
    sigma_se = np.asarray(res.sigma_se)
    for k in range(sigma_se.shape[0]):
        se[f"sigma_{k}_{k}"] = float(sigma_se[k, k])
    pi_se = np.asarray(res.pi_se)
    for k in range(pi_se.shape[0]):
        for d in range(pi_se.shape[1]):
            se[f"pi_{k}_{d}"] = float(pi_se[k, d])
    for k, v in enumerate(np.asarray(res.beta_se).flatten()):
        se[f"beta_{k}"] = float(v)
    for k, v in enumerate(np.asarray(res.gamma_se).flatten()):
        se[f"gamma_{k}"] = float(v)

    rows = summary[summary["start_id"] == best_start].copy()
    rows["se"] = rows["param_name"].map(se)
    rows["group"] = rows["param_name"].map(param_group)
    # Pretty labels for the forest plot.
    rows["label"] = rows["param_name"].map(
        lambda n: PARAM_PRETTY.get(n) or _fallback_label(n)
    )
    # Drop parameters pinned to zero (no SE; both truth and estimate ~ 0).
    estimated = rows["se"].notna()
    rows = rows.loc[estimated].reset_index(drop=True)
    meta = {"best_start": best_start, "best_obj": best_obj, "results": res}
    return rows, meta


def _fallback_label(name: str) -> str:
    if name.startswith("pi_"):
        _, k, d = name.split("_")
        return pi_pretty(int(k), int(d))
    return name


def make_figure(rows: pd.DataFrame, meta: dict, seed: int) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })
    fig = plt.figure(figsize=(15, 10))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.0],
        hspace=0.32,
        left=0.06, right=0.985, top=0.92, bottom=0.07,
    )
    # Top row: centered square scatter so aspect="equal" doesn't leave huge gaps.
    top = outer[0].subgridspec(1, 3, width_ratios=[0.18, 1.0, 0.18], wspace=0.05)
    bot = outer[1].subgridspec(1, 4, wspace=0.55)

    # ------------------------------ Top: scatter ------------------------------
    ax = fig.add_subplot(top[0, 1])
    lo = min(rows["truth"].min(), rows["estimate"].min()) - 0.25
    hi = max(rows["truth"].max(), rows["estimate"].max()) + 0.25
    ax.plot([lo, hi], [lo, hi], color="#444444", lw=1.0, zorder=1,
            label="estimate = truth")
    for grp in GROUPS:
        sub = rows[rows["group"] == grp]
        if sub.empty:
            continue
        ax.errorbar(
            sub["truth"], sub["estimate"],
            yerr=1.96 * sub["se"],
            fmt="o", ms=6.5, mew=0.0,
            color=GROUP_COLORS[grp], ecolor=GROUP_COLORS[grp],
            elinewidth=1.0, capsize=2.5, alpha=0.92,
            label=GROUP_LABEL[grp], zorder=3,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("true parameter value")
    ax.set_ylabel("estimate  (95% CI)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=1,
              bbox_to_anchor=(0.995, 0.025))

    # Summary stats annotation.
    within = (np.abs(rows["estimate"] - rows["truth"]) <= 1.96 * rows["se"]).sum()
    txt = (
        f"GMM objective: {meta['best_obj']:.3g}\n"
        f"best start: #{meta['best_start']}\n"
        f"params: {len(rows)} estimated\n"
        f"truth inside 95% CI: {within}/{len(rows)}\n"
        f"mean |est − truth|: {rows['abs_error'].mean():.3f}"
    )
    ax.text(
        0.025, 0.975, txt, transform=ax.transAxes, ha="left", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#bbbbbb", lw=0.8),
    )

    # --------------------------- Bottom: forest plots -------------------------
    for col, grp in enumerate(GROUPS):
        ax = fig.add_subplot(bot[0, col])
        sub = rows[rows["group"] == grp].copy()
        # Largest truth at top, smallest at bottom — sort descending and let
        # the default y-axis (low at bottom) do the rest.
        sub = sub.sort_values("truth", kind="stable", ascending=True)
        sub = sub.reset_index(drop=True)
        y = np.arange(len(sub))
        color = GROUP_COLORS[grp]
        ax.hlines(y, sub["estimate"] - 1.96 * sub["se"],
                  sub["estimate"] + 1.96 * sub["se"],
                  color=color, lw=2.0, alpha=0.85)
        ax.plot(sub["estimate"], y, "o", color=color, ms=6.5, zorder=3,
                label="estimate")
        ax.plot(sub["truth"], y, marker="D", linestyle="", ms=7.5,
                markerfacecolor="white", markeredgecolor="#222222",
                markeredgewidth=1.3, zorder=4, label="truth")
        ax.axvline(0, color="#aaaaaa", lw=0.7, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["label"], fontsize=9)
        ax.set_title(GROUP_LABEL[grp], fontsize=10, pad=6)
        ax.margins(y=0.08)
        if col == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=8,
                      bbox_to_anchor=(0.0, -0.05))

    fig.suptitle(
        f"BLP / RCL parameter recovery — seed_{seed}",
        fontsize=14, y=0.985, x=0.07, ha="left", weight="bold",
    )
    return fig


def main() -> None:
    args = parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(here, "output", f"seed_{args.seed}")

    rows, meta = load_recovery_table(output_dir)
    print(f"plotting {len(rows)} estimated parameters from "
          f"start #{meta['best_start']} (objective={meta['best_obj']:.4g})")
    fig = make_figure(rows, meta, seed=args.seed)

    png_path = os.path.join(output_dir, "recovery.png")
    svg_path = os.path.join(output_dir, "recovery.svg")
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"wrote {png_path}")
    print(f"wrote {svg_path}")


if __name__ == "__main__":
    main()
