"""Gap simulation and rule-based reconstruction methods for headcount series.

All reconstruction functions share one convention: ``y`` is the full series,
``i`` is the index of the first hidden bin and ``gap`` is the number of hidden
bins. Only the known edge bins ``y[i - 1]`` and ``y[i + gap]`` may be read; the
hidden bins ``y[i:i + gap]`` are never used (a test checks this).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def eligible_starts(y: np.ndarray, gap: int, context: int = 1) -> np.ndarray:
    """Find positions where a gap can be hidden with a clean ground truth.

    A position is eligible only if every hidden bin and the ``context`` bins
    on each side are known (not NaN), so the true values and both edges exist.

    Args:
        y: Headcount array (people) with NaN for unknown bins.
        gap: Number of hidden bins (bins, not minutes).
        context: Known bins required on each side of the gap.

    Returns:
        np.ndarray: Integer indices of the first hidden bin for every valid
        placement. Empty if no placement is valid.
    """
    known = (~np.isnan(y)).astype(int)
    window = gap + 2 * context
    # A sliding sum equals the window length only when every bin is known.
    full = np.convolve(known, np.ones(window, dtype=int), mode="valid") == window
    return np.flatnonzero(full) + context


def time_slots(index: pd.DatetimeIndex, slot_minutes: int = 30) -> np.ndarray:
    """Map local timestamps to weekly slots (weekday x time of day).

    Local clock time is used so that "9 am Monday" stays in the same slot
    across daylight-saving changes.

    Args:
        index: Timezone-aware (local time) timestamps.
        slot_minutes: Slot width in minutes; should divide 60 or 1440 evenly.

    Returns:
        np.ndarray: Integer slot number per timestamp, from 0 (Monday 00:00)
        to ``7 * 24 * 60 / slot_minutes - 1``.
    """
    per_day = 24 * 60 // slot_minutes
    return (index.dayofweek * per_day
            + (index.hour * 60 + index.minute) // slot_minutes).to_numpy()


def weekly_profile(y: np.ndarray, slots: np.ndarray, train_end: int,
                   slot_minutes: int = 30) -> np.ndarray:
    """Compute the mean headcount of each weekly slot from training data only.

    Only bins before ``train_end`` are used, so the validation and test
    periods never influence the profile (a test checks this).

    Args:
        y: Headcount array (people) with NaN for unknown bins.
        slots: Weekly slot number of every bin (see :func:`time_slots`).
        train_end: Index of the first bin that must not be used.
        slot_minutes: Slot width in minutes; sets the number of slots.

    Returns:
        np.ndarray: Mean headcount per slot, length ``7 * 24 * 60 /
        slot_minutes``. A slot with no known training bin is NaN (and numpy
        emits a "Mean of empty slice" warning).
    """
    n_slots = 7 * 24 * 60 // slot_minutes
    return np.array([np.nanmean(y[:train_end][slots[:train_end] == s])
                     for s in range(n_slots)])


def carry_forward(y: np.ndarray, i: int, gap: int) -> np.ndarray:
    """Baseline: repeat the last known value across the gap.

    Args:
        y: Headcount array (people).
        i: Index of the first hidden bin (must be >= 1).
        gap: Number of hidden bins.

    Returns:
        np.ndarray: ``gap`` copies of ``y[i - 1]``.
    """
    return np.full(gap, y[i - 1])


def linear_interp(y: np.ndarray, i: int, gap: int) -> np.ndarray:
    """Straight line from the last known bin to the first known bin after.

    Args:
        y: Headcount array (people).
        i: Index of the first hidden bin (must be >= 1).
        gap: Number of hidden bins; ``y[i + gap]`` must exist.

    Returns:
        np.ndarray: ``gap`` interpolated values strictly between the edges.
    """
    # linspace includes both edge values, so drop the first and last point.
    return np.linspace(y[i - 1], y[i + gap], gap + 2)[1:-1]


def decay_profile(y: np.ndarray, i: int, gap: int, slots: np.ndarray,
                  profile: np.ndarray, tau_bins: float = 24) -> np.ndarray:
    """Weekly profile plus an edge anomaly that fades with distance.

    The anomaly (observed minus profile) at each gap edge is interpolated
    across the gap and multiplied by ``exp(-d / tau_bins)``, where ``d`` is
    the distance in bins to the nearest known value. Near an edge the
    reconstruction follows the observed level; far from both edges it falls
    back to the typical weekly pattern. A very small ``tau_bins`` gives the
    plain weekly profile and a very large one gives profile-corrected linear
    interpolation. Output is clipped at zero because headcount cannot be
    negative.

    Args:
        y: Headcount array (people).
        i: Index of the first hidden bin (must be >= 1).
        gap: Number of hidden bins; ``y[i + gap]`` must exist.
        slots: Weekly slot number of every bin (see :func:`time_slots`).
        profile: Weekly profile from :func:`weekly_profile`.
        tau_bins: Fade time constant in bins (24 bins = 2 h at 5 min).

    Returns:
        np.ndarray: ``gap`` reconstructed headcounts (people), all >= 0.
    """
    t = np.arange(i, i + gap)
    before = y[i - 1] - profile[slots[i - 1]]    # anomaly at the left edge
    after = y[i + gap] - profile[slots[i + gap]]  # anomaly at the right edge
    resid = np.linspace(before, after, gap + 2)[1:-1]
    # Distance (in bins) from each hidden bin to the nearer known edge.
    nearest = np.minimum(t - (i - 1), (i + gap) - t)
    return np.maximum(profile[slots[t]] + resid * np.exp(-nearest / tau_bins), 0)
