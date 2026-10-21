import warnings

import numpy as np
import pandas as pd
import pytest

from src.gaps import (carry_forward, decay_profile, eligible_starts,
                      linear_interp, weekly_profile)
from src.series import build_headcount_series
from src.stats import cluster_boot_ci, cluster_boot_rmse_diff


# ---------- input handling ----------

def test_gap_must_be_positive_and_inside_series():
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError):
        linear_interp(y, 3, 0)          # empty gap
    with pytest.raises(ValueError):
        linear_interp(y, 0, 3)          # touches the left edge
    with pytest.raises(ValueError):
        carry_forward(y, 8, 2)          # touches the right edge


def test_nan_edge_is_rejected():
    y = np.arange(10, dtype=float)
    y[2] = np.nan                        # left edge of the gap i=3
    with pytest.raises(ValueError):
        linear_interp(y, 3, 3)
    y = np.arange(10, dtype=float)
    y[6] = np.nan                        # right edge of the gap i=3, gap=3
    with pytest.raises(ValueError):
        decay_profile(y, 3, 3, np.zeros(10, int), np.zeros(1))


def test_decay_rejects_bad_tau():
    with pytest.raises(ValueError):
        decay_profile(np.ones(100), 10, 5, np.zeros(100, int), np.ones(1), tau_bins=0)


def test_eligible_starts_rejects_bad_gap_and_handles_short_series():
    with pytest.raises(ValueError):
        eligible_starts(np.ones(10), gap=0)
    assert len(eligible_starts(np.ones(3), gap=5)) == 0


def test_weekly_profile_validates_arguments():
    y, s = np.ones(400), np.arange(400) % 336
    with pytest.raises(ValueError):
        weekly_profile(y, s, train_end=0)
    with pytest.raises(ValueError):
        weekly_profile(y, s, train_end=500)
    with pytest.raises(ValueError):
        weekly_profile(y, s, train_end=300, slot_minutes=7)
    with pytest.raises(ValueError):
        weekly_profile(y, s[:-1], train_end=300)
    with pytest.raises(ValueError):
        weekly_profile(y, s, train_end=300, on_empty="ignore")


def test_weekly_profile_empty_slot_warns_or_raises():
    y, s = np.ones(300), np.arange(300) % 336      # slots 300-335 never seen
    with pytest.warns(UserWarning, match="no known training bin"):
        p = weekly_profile(y, s, train_end=300)
    assert np.isnan(p[310]) and p[0] == 1
    with pytest.raises(ValueError):
        weekly_profile(y, s, train_end=300, on_empty="raise")


def test_weekly_profile_ignores_nan_bins():
    s = np.arange(672) % 336
    y = np.concatenate([np.full(336, 2.0), np.full(336, 4.0)])
    y[336 + 5] = np.nan                              # slot 5 known only in week 1
    p = weekly_profile(y, s, train_end=672)
    assert p[0] == 3.0 and p[5] == 2.0


def test_series_rejects_empty_events():
    empty = pd.DataFrame({"time": pd.to_datetime([]).tz_localize("UTC"),
                          "occupancystatus": [], "headcount": []})
    with pytest.raises(ValueError):
        build_headcount_series(empty)


def test_bootstrap_rejects_mismatched_input():
    with pytest.raises(ValueError):
        cluster_boot_ci(np.ones(5), np.arange(4))
    with pytest.raises(ValueError):
        cluster_boot_rmse_diff(np.ones(5), np.ones(4), np.arange(5))


# ---------- verification against known answers ----------

def _flat_setup(level=2.0, n=100):
    return np.full(336, level), np.arange(n) % 336


def test_decay_matches_analytic_answer():
    """Constant profile P and equal edge anomalies a: output = P + a*exp(-d/tau)."""
    P, a, tau, i, gap = 2.0, 3.0, 3.0, 10, 6
    profile, slots = _flat_setup(P)
    y = np.full(100, P)
    y[i - 1] = y[i + gap] = P + a
    y[i:i + gap] = 99                                # hidden truth must be ignored
    d = np.array([1, 2, 3, 3, 2, 1])                 # distance to nearest edge
    expected = P + a * np.exp(-d / tau)
    assert np.allclose(decay_profile(y, i, gap, slots, profile, tau_bins=tau), expected)


def test_decay_limits_and_clipping():
    P, a, i, gap = 2.0, 3.0, 10, 6
    profile, slots = _flat_setup(P)
    y = np.full(100, P)
    y[i - 1] = y[i + gap] = P + a
    # tiny tau -> plain weekly profile; huge tau -> profile + full anomaly
    assert np.allclose(decay_profile(y, i, gap, slots, profile, tau_bins=1e-6), P)
    assert np.allclose(decay_profile(y, i, gap, slots, profile, tau_bins=1e9), P + a)
    # a strongly negative anomaly must be clipped at zero, never negative
    y[i - 1] = y[i + gap] = P - 5
    assert (decay_profile(y, i, gap, slots, profile, tau_bins=1e9) == 0).all()


# ---------- reproducibility after refactoring ----------

def test_weekly_profile_matches_original_loop():
    """The bincount version must equal the original per-slot nanmean loop."""
    rng = np.random.default_rng(7)
    y = rng.integers(0, 6, 3000).astype(float)
    y[rng.random(3000) < 0.2] = np.nan
    s = np.arange(3000) % 336
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # old loop warns on empty slots
        old = np.array([np.nanmean(y[:2500][s[:2500] == k]) for k in range(336)])
        new = weekly_profile(y, s, 2500)
    assert np.allclose(new, old, equal_nan=True)
