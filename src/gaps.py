"""Gap simulation and reconstruction methods for headcount series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def eligible_starts(y: np.ndarray, gap: int, context: int = 1) -> np.ndarray:
    """Return start indices where a hidden gap and its edges are all known.

    Args:
        y: Headcount array with NaN for unknown bins.
        gap: Number of hidden bins.
        context: Known bins required on each side of the gap.

    Returns:
        Indices of the first hidden bin for every valid placement.
    """
    known = (~np.isnan(y)).astype(int)
    window = gap + 2 * context
    full = np.convolve(known, np.ones(window, dtype=int), mode="valid") == window
    return np.flatnonzero(full) + context


def time_slots(index: pd.DatetimeIndex, slot_minutes: int = 30) -> np.ndarray:
    """Map local timestamps to weekly slots (weekday x time of day)."""
    per_day = 24 * 60 // slot_minutes
    return (index.dayofweek * per_day
            + (index.hour * 60 + index.minute) // slot_minutes).to_numpy()


def weekly_profile(y: np.ndarray, slots: np.ndarray, train_end: int,
                   slot_minutes: int = 30) -> np.ndarray:
    """Mean headcount per weekly slot, using only bins before ``train_end``."""
    n_slots = 7 * 24 * 60 // slot_minutes
    return np.array([np.nanmean(y[:train_end][slots[:train_end] == s])
                     for s in range(n_slots)])


def carry_forward(y: np.ndarray, i: int, gap: int) -> np.ndarray:
    """Repeat the last known value across the gap."""
    return np.full(gap, y[i - 1])


def linear_interp(y: np.ndarray, i: int, gap: int) -> np.ndarray:
    """Straight line from the last known bin to the first known bin after."""
    return np.linspace(y[i - 1], y[i + gap], gap + 2)[1:-1]


def decay_profile(y: np.ndarray, i: int, gap: int, slots: np.ndarray,
                  profile: np.ndarray, tau_bins: float = 24) -> np.ndarray:
    """Weekly profile plus an edge anomaly that fades with distance.

    The anomaly (observed minus profile) at each gap edge is interpolated
    across the gap and multiplied by ``exp(-d / tau_bins)``, where ``d`` is
    the distance in bins to the nearest known value. Output is clipped at zero.
    """
    t = np.arange(i, i + gap)
    before = y[i - 1] - profile[slots[i - 1]]
    after = y[i + gap] - profile[slots[i + gap]]
    resid = np.linspace(before, after, gap + 2)[1:-1]
    nearest = np.minimum(t - (i - 1), (i + gap) - t)
    return np.maximum(profile[slots[t]] + resid * np.exp(-nearest / tau_bins), 0)
