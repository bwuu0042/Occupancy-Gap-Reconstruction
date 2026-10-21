"""Features and simulated-gap training sets for learned gap reconstruction."""

from __future__ import annotations

import numpy as np

# Column order of the matrix returned by gap_features. experiment.py relies on
# column 0 (profile), 5 (distance to nearest edge) and 7 (interpolated anomaly).
FEATURE_NAMES = ["profile", "anom_before", "anom_after", "dist_before",
                 "dist_after", "dist_nearest", "gap_len", "lin_interp_anom"]


def gap_features(y: np.ndarray, i: int, gap: int, slots: np.ndarray,
                 profile: np.ndarray) -> np.ndarray:
    """Build the feature matrix, one row per hidden bin, for one gap.

    Only the bins at ``i - 1`` and ``i + gap`` (the known edges), the weekly
    profile and the gap geometry are used. The hidden bins ``y[i:i + gap]``
    are never read, so the features cannot leak the answer (a test checks
    this).

    Args:
        y: Headcount array (people).
        i: Index of the first hidden bin (must be >= 1).
        gap: Number of hidden bins; ``y[i + gap]`` must exist.
        slots: Weekly slot number of every bin.
        profile: Weekly profile from ``weekly_profile``.

    Returns:
        np.ndarray: Array of shape ``(gap, 8)`` with columns listed in
        ``FEATURE_NAMES``: profile value, anomaly at each edge, distance to
        each edge, distance to the nearest edge, gap length, and the linearly
        interpolated anomaly.
    """
    t = np.arange(i, i + gap)
    prof_t = profile[slots[t]]
    before = y[i - 1] - profile[slots[i - 1]]
    after = y[i + gap] - profile[slots[i + gap]]
    d_before, d_after = t - (i - 1), (i + gap) - t
    lin_resid = np.linspace(before, after, gap + 2)[1:-1]
    # before/after are single numbers, so repeat them down the gap to give
    # every column the same length.
    return np.column_stack([prof_t, np.full(gap, before), np.full(gap, after),
                            d_before, d_after, np.minimum(d_before, d_after),
                            np.full(gap, gap), lin_resid])


def make_training_set(y: np.ndarray, slots: np.ndarray, profile: np.ndarray,
                      lo: int, hi: int, n_gaps: int, seed: int,
                      min_g: int = 3, max_g: int = 288):
    """Simulate gaps inside ``y[lo:hi]`` and return features and targets.

    Gap lengths are drawn log-uniformly between ``min_g`` and ``max_g`` bins
    so short and long gaps are both represented. Each gap and both edges must
    be known. The target is the true headcount minus the weekly profile (the
    "anomaly"), which is easier to learn than the raw headcount.

    Args:
        y: Headcount array (people) with NaN for unknown bins.
        slots: Weekly slot number of every bin.
        profile: Weekly profile (built from training data only).
        lo: First index that may be used (exclusive of the edge bin).
        hi: Index after the last usable bin; keep within the training period.
        n_gaps: Number of gaps to simulate.
        seed: Seed of the random generator, for reproducibility.
        min_g: Shortest gap in bins.
        max_g: Longest gap in bins.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(X, Y)`` where ``X`` has shape
        ``(n_rows, 8)`` and ``Y`` the anomaly per row. A gap is skipped if no
        valid position is found in 50 tries, so ``n_rows`` can be slightly
        smaller than expected.
    """
    rng = np.random.default_rng(seed)
    lengths = np.exp(rng.uniform(np.log(min_g), np.log(max_g), n_gaps)).astype(int)
    X, Y = [], []
    for g in lengths:
        for _ in range(50):                       # retry until the window is fully known
            i = int(rng.integers(lo + 1, hi - g - 1))
            if not np.isnan(y[i - 1:i + g + 1]).any():
                X.append(gap_features(y, i, g, slots, profile))
                Y.append(y[i:i + g] - profile[slots[i:i + g]])
                break
    return np.vstack(X), np.concatenate(Y)
