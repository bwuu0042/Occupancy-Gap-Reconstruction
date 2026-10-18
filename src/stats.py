"""Day-level (cluster) bootstrap confidence intervals for gap-error comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_boot_ci(values, groups, n_boot: int = 2000, seed: int = 1):
    """95% CI of a mean, resampling whole groups (days) with replacement.

    Gaps on the same day overlap in time, so their errors are correlated.
    Resampling days instead of single gaps respects that.
    """
    g = pd.DataFrame({"v": values, "g": groups}).groupby("g")["v"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    return np.percentile(s[idx].sum(axis=1) / c[idx].sum(axis=1), [2.5, 97.5])


def cluster_boot_rmse_diff(mse_a, mse_b, groups, n_boot: int = 2000, seed: int = 1):
    """95% CI of RMSE(a) - RMSE(b), resampling whole days.

    Inputs are per-gap mean squared errors. RMSE is the square root of the
    mean of those, so the bootstrap works on per-day sums.
    """
    g = (pd.DataFrame({"a": mse_a, "b": mse_b, "g": groups})
         .groupby("g").agg(a=("a", "sum"), b=("b", "sum"), c=("a", "count")))
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n_boot, len(g)))
    cnt = g["c"].to_numpy()[idx].sum(axis=1)
    ra = np.sqrt(g["a"].to_numpy()[idx].sum(axis=1) / cnt)
    rb = np.sqrt(g["b"].to_numpy()[idx].sum(axis=1) / cnt)
    return np.percentile(ra - rb, [2.5, 97.5])


def compare_to_reference(res: pd.DataFrame, day_of: np.ndarray, reference: str,
                         order: list, metric: str = "mae") -> pd.DataFrame:
    """Compare each method with ``reference`` on busy gaps, per gap length.

    Args:
        res: Output of ``score_gaps``.
        day_of: Array giving the calendar day of every bin index.
        reference: Method name to compare against.
        order: Gap-length labels, in display order.
        metric: ``"mae"`` or ``"rmse"``.

    Returns:
        One row per (gap, method) with the difference from the reference
        (positive = worse), its 95% interval, and whether it excludes zero.
    """
    busy = res[res["busy"]]
    col = "mae" if metric == "mae" else "mse"
    rows = []
    for label in order:
        w = busy[busy["gap"] == label].pivot_table(index="start", columns="method", values=col)
        groups = day_of[w.index.to_numpy()]
        for m in w.columns:
            if m == reference:
                continue
            if metric == "mae":
                d = (w[m] - w[reference]).to_numpy()
                lo, hi = cluster_boot_ci(d, groups)
                diff = d.mean()
            else:
                lo, hi = cluster_boot_rmse_diff(w[m].to_numpy(), w[reference].to_numpy(), groups)
                diff = np.sqrt(w[m].mean()) - np.sqrt(w[reference].mean())
            rows.append({"gap": label, "method": m, "n": len(w), "diff": diff,
                         "ci_low": lo, "ci_high": hi, "real": (lo > 0) or (hi < 0)})
    return pd.DataFrame(rows)
