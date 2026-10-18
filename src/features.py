"""Features and simulated-gap training sets for learned gap reconstruction."""

from __future__ import annotations

import numpy as np


def gap_features(y: np.ndarray, i: int, gap: int, slots: np.ndarray,
                 profile: np.ndarray) -> np.ndarray:
    """Feature matrix, one row per hidden bin, for a gap starting at ``i``.

    Only the bins at ``i - 1`` and ``i + gap`` (the known edges), the weekly
    profile and the gap geometry are used. The hidden bins ``y[i:i + gap]``
    are never read.

    Columns: profile, anomaly before, anomaly after, distance to the last
    known bin, distance to the next known bin, distance to the nearest edge,
    gap length, linearly interpolated anomaly.
    """
    t = np.arange(i, i + gap)
    prof_t = profile[slots[t]]
    before = y[i - 1] - profile[slots[i - 1]]
    after = y[i + gap] - profile[slots[i + gap]]
    d_before, d_after = t - (i - 1), (i + gap) - t
    lin_resid = np.linspace(before, after, gap + 2)[1:-1]
    return np.column_stack([prof_t, np.full(gap, before), np.full(gap, after),
                            d_before, d_after, np.minimum(d_before, d_after),
                            np.full(gap, gap), lin_resid])


def make_training_set(y: np.ndarray, slots: np.ndarray, profile: np.ndarray,
                      lo: int, hi: int, n_gaps: int, seed: int,
                      min_g: int = 3, max_g: int = 288):
    """Simulate gaps inside ``y[lo:hi]`` and return features and anomaly targets.

    Gap lengths are log-uniform between ``min_g`` and ``max_g`` bins. Each gap
    and both of its edges must be known. The target is the true headcount
    minus the weekly profile.
    """
    rng = np.random.default_rng(seed)
    lengths = np.exp(rng.uniform(np.log(min_g), np.log(max_g), n_gaps)).astype(int)
    X, Y = [], []
    for g in lengths:
        for _ in range(50):
            i = int(rng.integers(lo + 1, hi - g - 1))
            if not np.isnan(y[i - 1:i + g + 1]).any():
                X.append(gap_features(y, i, g, slots, profile))
                Y.append(y[i:i + g] - profile[slots[i:i + g]])
                break
    return np.vstack(X), np.concatenate(Y)
