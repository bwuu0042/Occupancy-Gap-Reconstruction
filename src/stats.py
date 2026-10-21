"""Day-level (cluster) bootstrap confidence intervals for gap-error comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_boot_ci(values, groups, n_boot: int = 2000, seed: int = 1):
    """95% confidence interval of a mean, resampling whole groups (days).

    Gaps on the same day overlap in time, so their errors are correlated.
    Resampling days instead of single gaps respects that and avoids
    intervals that are too narrow.

    Args:
        values: One number per gap (for example the MAE difference between
            two methods).
        groups: Group label per gap (for example its calendar day).
        n_boot: Number of bootstrap resamples.
        seed: Seed of the random generator, for reproducibility.

    Returns:
        np.ndarray: ``[lower, upper]`` 2.5th and 97.5th percentiles of the
        resampled mean.
    """
    # Per-day sums and counts let each resample be computed with array maths.
    g = pd.DataFrame({"v": values, "g": groups}).groupby("g")["v"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    return np.percentile(s[idx].sum(axis=1) / c[idx].sum(axis=1), [2.5, 97.5])


def cluster_boot_rmse_diff(mse_a, mse_b, groups, n_boot: int = 2000, seed: int = 1):
    """95% confidence interval of RMSE(a) - RMSE(b), resampling whole days.

    RMSE is the square root of the mean squared error, so the bootstrap works
    on per-day sums of the per-gap mean squared errors and takes the square
    root after averaging.

    Args:
        mse_a: Per-gap mean squared error of method a (people squared).
        mse_b: Per-gap mean squared error of method b, same gaps and order.
        groups: Group label per gap (for example its calendar day).
        n_boot: Number of bootstrap resamples.
        seed: Seed of the random generator, for reproducibility.

    Returns:
        np.ndarray: ``[lower, upper]`` percentiles of the RMSE difference
        (positive means method a is worse).
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
    """Compare each method with a reference method on busy gaps.

    Args:
        res: Output of ``experiment.score_gaps``.
        day_of: Array giving the calendar day of every bin index.
        reference: Name of the method to compare against.
        order: Gap-length labels in the order to report.
        metric: ``"mae"`` or ``"rmse"``.

    Returns:
        pd.DataFrame: One row per (gap length, method) with ``n`` (number of
        busy gaps), ``diff`` (method minus reference; positive = worse),
        ``ci_low`` and ``ci_high`` (95% interval) and ``real`` (True if the
        interval excludes zero).
    """
    busy = res[res["busy"]]                       # diagnostic subset (uses the hidden truth)
    col = "mae" if metric == "mae" else "mse"
    rows = []
    for label in order:
        # One row per gap, one column per method, so methods are paired by gap.
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
