"""Diagnostics for pyblp solves that fail or do not converge.

`extract_solve_diagnostics()` turns either a returned `pyblp.ProblemResults` or
a raised exception into a flat, CSV-friendly dict, so a start that errors or
stalls is recorded as a row instead of being silently dropped. Filter the
resulting table to ``outcome != "converged"`` to isolate exactly the simulations
that did not converge or raised.

The fields answer three questions:
  * What kind of failure?       -> stage, outcome, error_classes, error_message
  * Close or hopeless?          -> objective(_finite), converged,
                                   projected_gradient_norm, optimization_iterations,
                                   objective_evaluations, nonfinite_theta
  * Data/spec or unlucky start? -> fp_nonconverged_count, fp_iterations_max,
                                   contraction_evaluations, clipped_shares,
                                   clipped_costs, hessian_min_eig

This module is intentionally dependency-light and import-safe: every extraction
is guarded so collecting diagnostics can never itself raise.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:                                            # pyblp is always present in this repo
    import pyblp
    _PYBLP_VERSION = getattr(pyblp, "__version__", "")
except Exception:                               # noqa: BLE001 - never fail on import
    pyblp = None                                # type: ignore[assignment]
    _PYBLP_VERSION = ""


# Stable column order so every driver writes the same schema.
DIAGNOSTIC_FIELDS: tuple[str, ...] = (
    "stage",                    # pipeline step: stage1_solve, stage2_prefilter, stage2_opt_iv, solve, ...
    "seed",
    "outcome",                  # "error" | "no_converge" | "converged"
    "error_classes",            # "|"-joined specific pyblp exception subclasses (MultipleErrors unwrapped)
    "error_message",
    "note",                     # free-form context, e.g. a pre-filter rejection reason
    "objective",
    "objective_finite",
    "converged",
    "projected_gradient_norm",  # first-order optimality at termination
    "optimization_iterations",
    "objective_evaluations",
    "fp_nonconverged_count",    # market x objective-evaluation delta-contraction failures (fp_converged is 2D)
    "fp_iterations_max",
    "contraction_evaluations",
    "clipped_shares",           # count of clipped share values (numerical-stress flag)
    "clipped_costs",
    "nonfinite_theta",          # non-finite entries in theta at termination
    "hessian_min_eig",          # min eigenvalue of the reduced Hessian (saddle / non-PD check)
    "elapsed_sec",
    "pyblp_version",
)


# ── small, never-raising coercions ───────────────────────────────────────────

def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(x: Any) -> Any:
    if x is None:
        return ""
    try:
        return int(x)
    except (TypeError, ValueError):
        return ""


def _count_true(arr: Any) -> Any:
    """Count truthy entries in a boolean/array clip flag; '' if unavailable."""
    if arr is None:
        return ""
    try:
        return int(np.asarray(arr).sum())
    except Exception:                           # noqa: BLE001
        return ""


def _count_nonfinite(arr: Any) -> Any:
    if arr is None:
        return ""
    try:
        a = np.asarray(arr, dtype=float)
        return int((~np.isfinite(a)).sum())
    except Exception:                           # noqa: BLE001
        return ""


def _min_eig(mat: Any) -> Any:
    """Smallest eigenvalue of a symmetric matrix; '' if unavailable, nan if non-finite."""
    if mat is None:
        return ""
    try:
        m = np.asarray(mat, dtype=float)
        if m.ndim != 2 or m.size == 0 or m.shape[0] != m.shape[1]:
            return ""
        if not np.all(np.isfinite(m)):
            return float("nan")
        return float(np.linalg.eigvalsh(m).min())
    except Exception:                           # noqa: BLE001
        return ""


def _unwrap_error_classes(exc: BaseException) -> str:
    """Return '|'-joined specific error subclass names.

    pyblp.exceptions.MultipleErrors stores its constituents in `_errors`; a
    singular error is returned as itself, so the fallback is the class name.
    """
    inner = getattr(exc, "_errors", None)
    if inner:
        try:
            return "|".join(dict.fromkeys(type(e).__name__ for e in inner))
        except Exception:                       # noqa: BLE001
            pass
    return type(exc).__name__


def _result_diagnostics(res: Any) -> dict:
    """Pull numerical-health fields off a (possibly partially-formed) result."""
    fp_conv = getattr(res, "fp_converged", None)
    fp_iter = getattr(res, "fp_iterations", None)
    contr = getattr(res, "contraction_evaluations", None)
    objective = _safe_float(getattr(res, "objective", np.nan))
    return {
        "objective": objective,
        "objective_finite": bool(np.isfinite(objective)),
        "converged": bool(getattr(res, "converged", False)),
        "projected_gradient_norm": _safe_float(getattr(res, "projected_gradient_norm", np.nan)),
        "optimization_iterations": _safe_int(getattr(res, "optimization_iterations", None)),
        "objective_evaluations": _safe_int(getattr(res, "objective_evaluations", None)),
        "fp_nonconverged_count": (
            int((~np.asarray(fp_conv).astype(bool)).sum()) if fp_conv is not None else ""
        ),
        "fp_iterations_max": (
            int(np.max(fp_iter)) if fp_iter is not None and np.size(fp_iter) else ""
        ),
        "contraction_evaluations": (
            int(np.sum(contr)) if contr is not None and np.size(contr) else ""
        ),
        "clipped_shares": _count_true(getattr(res, "clipped_shares", None)),
        "clipped_costs": _count_true(getattr(res, "clipped_costs", None)),
        "nonfinite_theta": _count_nonfinite(getattr(res, "theta", None)),
        "hessian_min_eig": _min_eig(getattr(res, "reduced_hessian", None)),
    }


def extract_solve_diagnostics(
    res_or_exc: Any,
    *,
    stage: str,
    seed: Any = None,
    elapsed_sec: Optional[float] = None,
    note: str = "",
) -> dict:
    """Flatten a returned ProblemResults or a raised exception into a diagnostics row.

    Parameters
    ----------
    res_or_exc:
        A ``pyblp.ProblemResults`` for a solve that returned (whether or not it
        converged), or a ``BaseException`` for a solve that raised.
    stage:
        Where in the pipeline this came from (e.g. "stage1_solve",
        "stage2_prefilter", "stage2_opt_iv", "solve").
    seed:
        The start's seed / id, for reproduction and failure-clustering analysis.
    elapsed_sec:
        Wall-clock seconds for the attempt, if measured (instant failure vs.
        timeout are different problems).
    note:
        Free-form context, e.g. a pre-filter rejection reason.

    Returns a dict containing every key in ``DIAGNOSTIC_FIELDS``.
    """
    row: dict[str, Any] = {f: "" for f in DIAGNOSTIC_FIELDS}
    row.update({
        "stage": stage,
        "seed": "" if seed is None else seed,
        "note": note,
        "elapsed_sec": float("nan") if elapsed_sec is None else float(elapsed_sec),
        "pyblp_version": _PYBLP_VERSION,
    })

    if isinstance(res_or_exc, BaseException):
        row["outcome"] = "error"
        row["error_classes"] = _unwrap_error_classes(res_or_exc)
        row["error_message"] = str(res_or_exc)[:500]
        return row

    res = res_or_exc
    try:
        row.update(_result_diagnostics(res))
    except Exception as exc:                     # noqa: BLE001 - diagnostics must not raise
        row["outcome"] = "error"
        row["error_classes"] = type(exc).__name__
        row["error_message"] = f"diagnostics extraction failed: {exc}"[:500]
        return row

    row["outcome"] = "converged" if row.get("converged") else "no_converge"
    return row
