"""Score every reconstruction method on the same simulated gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import gap_features
from .models import TAU_BINS

# Gap length label -> number of 5-minute bins.
GAP_LENGTHS = {"15 min": 3, "30 min": 6, "1 h": 12, "2 h": 24,
               "4 h": 48, "8 h": 96, "12 h": 144, "24 h": 288}


def predict_gaps(y: np.ndarray, starts: np.ndarray, gap: int, slots: np.ndarray,
                 profile: np.ndarray, models: dict) -> dict:
    """Predict every gap in ``starts`` with every method.

    Features are built once and reused by the decay rule and the learned
    models, so all methods see exactly the same information.

    Args:
        y: Headcount array (people) with NaN for unknown bins.
        starts: Indices of the first hidden bin of each gap.
        gap: Gap length in bins (same for all gaps).
        slots: Weekly slot number of every bin.
        profile: Weekly profile from the training period.
        models: Fitted learned models from ``models.fit_frozen_models``.

    Returns:
        dict: Method name -> array of shape ``(len(starts), gap)`` with the
        predicted headcount (people). Keys: ``carry_forward``,
        ``linear_interp``, ``weekly_profile``, ``decay_profile`` and one per
        learned model.
    """
    X = np.vstack([gap_features(y, i, gap, slots, profile) for i in starts])
    prof = X[:, 0].reshape(len(starts), gap)               # column 0: profile value
    out = {
        "carry_forward": np.repeat(y[starts - 1][:, None], gap, axis=1),
        "linear_interp": np.array([np.linspace(y[i - 1], y[i + gap], gap + 2)[1:-1]
                                   for i in starts]),
        "weekly_profile": prof,
        # Column 7 is the interpolated edge anomaly, column 5 the distance to
        # the nearest edge; this reproduces ``gaps.decay_profile``.
        "decay_profile": np.maximum(
            prof + (X[:, 7] * np.exp(-X[:, 5] / TAU_BINS)).reshape(prof.shape), 0),
    }
    for name, m in models.items():
        # Learned models predict the anomaly; add the profile back and clip at 0.
        out[name] = np.maximum(prof + m.predict(X).reshape(prof.shape), 0)
    return out


def score_gaps(y: np.ndarray, starts: np.ndarray, slots: np.ndarray,
               profile: np.ndarray, models: dict,
               gap_lengths: dict = GAP_LENGTHS) -> pd.DataFrame:
    """Score all methods on the same hidden gaps, for every gap length.

    Args:
        y: Headcount array (people) with NaN for unknown bins.
        starts: Gap start indices; all must be eligible for the longest gap.
        slots: Weekly slot number of every bin.
        profile: Weekly profile from the training period.
        models: Fitted learned models.
        gap_lengths: Mapping from label to gap length in bins.

    Returns:
        pd.DataFrame: One row per (gap length, start, method) with columns
        ``gap``, ``start``, ``method``, ``mae`` and ``mse`` (people and people
        squared), ``bias`` (prediction minus truth; negative = under-predicts)
        and ``busy`` (True if the true gap contains at least one person).
        ``busy`` uses the hidden truth, so it is a diagnostic and could not be
        known in a real outage.
    """
    frames = []
    for label, g in gap_lengths.items():
        truth = np.array([y[i:i + g] for i in starts])
        busy = truth.max(axis=1) >= 1
        for name, h in predict_gaps(y, starts, g, slots, profile, models).items():
            e = h - truth                                      # signed error per bin
            frames.append(pd.DataFrame({"gap": label, "start": starts, "method": name,
                                        "mae": np.abs(e).mean(1), "mse": (e ** 2).mean(1),
                                        "bias": e.mean(1), "busy": busy}))
    return pd.concat(frames, ignore_index=True)
