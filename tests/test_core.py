import numpy as np
import pandas as pd

from src.gaps import (carry_forward, decay_profile, eligible_starts,
                      linear_interp, weekly_profile)
from src.series import build_headcount_series


def _events(rows):
    return pd.DataFrame({
        "time": pd.to_datetime([r[0] for r in rows]).tz_localize("Australia/Melbourne"),
        "occupancystatus": [r[1] for r in rows],
        "headcount": [r[2] for r in rows],
    })


def test_eligible_starts_skips_nan_windows():
    y = np.array([1, 2, np.nan, 4, 5, 6, 7], dtype=float)
    assert list(eligible_starts(y, gap=2)) == [4]


def test_linear_interp_known_values():
    y = np.array([0, 9, 9, 9, 4], dtype=float)
    assert np.allclose(linear_interp(y, 1, 3), [1, 2, 3])


def test_carry_forward_repeats_last_value():
    y = np.array([3, 9, 9, 0], dtype=float)
    assert np.allclose(carry_forward(y, 1, 2), [3, 3])


def test_methods_ignore_hidden_bins():
    """No leakage: changing the hidden truth must not change any prediction."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 800).astype(float)
    slots = np.arange(800) % 336
    profile = weekly_profile(y, slots, train_end=700)
    i, gap = 750, 20
    y2 = y.copy()
    y2[i:i + gap] = 99
    for fn in (carry_forward, linear_interp):
        assert np.allclose(fn(y, i, gap), fn(y2, i, gap))
    assert np.allclose(decay_profile(y, i, gap, slots, profile),
                       decay_profile(y2, i, gap, slots, profile))


def test_profile_uses_training_period_only():
    y = np.ones(672)
    slots = np.arange(672) % 336
    p1 = weekly_profile(y, slots, train_end=336)
    y[336:] = 50
    assert np.allclose(p1, weekly_profile(y, slots, train_end=336))


def test_series_holds_empty_room():
    ev = _events([("2025-03-03 00:00", "CurrentlyOccupied", 3),
                  ("2025-03-03 00:10", "NotOccupied", 0),
                  ("2025-03-03 02:00", "NotOccupied", 0)])
    s = build_headcount_series(ev)
    assert s.loc["2025-03-03 01:00"] == 0


def test_series_expires_occupied_state_after_hold():
    ev = _events([("2025-03-03 00:00", "CurrentlyOccupied", 3),
                  ("2025-03-03 02:00", "NotOccupied", 0)])
    s = build_headcount_series(ev, max_hold="30min")
    assert s.loc["2025-03-03 00:25"] == 3
    assert np.isnan(s.loc["2025-03-03 00:30"])
