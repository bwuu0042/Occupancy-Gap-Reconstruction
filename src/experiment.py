"""Score every reconstruction method on the same simulated gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import gap_features
from .models import TAU_BINS

GAP_LENGTHS = {"15 min": 3, "30 min": 6, "1 h": 12, "2 h": 24,
               "4 h": 48, "8 h": 96, "12 h": 144, "24 h": 288}


def predict_gaps(y, starts, gap, slots, profile, models) -> dict:
    """Predict every gap in ``starts`` with every method.

    Returns a dict of method name -> array of shape (n_gaps, gap).
    """
    X = np.vstack([gap_features(y, i, gap, slots, profile) for i in starts])
    prof = X[:, 0].reshape(len(starts), gap)
    out = {
        "carry_forward": np.repeat(y[starts - 1][:, None], gap, axis=1),
        "linear_interp": np.array([np.linspace(y[i - 1], y[i + gap], gap + 2)[1:-1]
                                   for i in starts]),
        "weekly_profile": prof,
        "decay_profile": np.maximum(
            prof + (X[:, 7] * np.exp(-X[:, 5] / TAU_BINS)).reshape(prof.shape), 0),
    }
    for name, m in models.items():
        out[name] = np.maximum(prof + m.predict(X).reshape(prof.shape), 0)
    return out


def score_gaps(y, starts, slots, profile, models, gap_lengths=GAP_LENGTHS) -> pd.DataFrame:
    """One row per (gap length, start, method) with MAE, MSE, bias and busy flag."""
    frames = []
    for label, g in gap_lengths.items():
        truth = np.array([y[i:i + g] for i in starts])
        busy = truth.max(axis=1) >= 1
        for name, h in predict_gaps(y, starts, g, slots, profile, models).items():
            e = h - truth
            frames.append(pd.DataFrame({"gap": label, "start": starts, "method": name,
                                       "mae": np.abs(e).mean(1), "mse": (e ** 2).mean(1),
                                       "bias": e.mean(1), "busy": busy}))
    return pd.concat(frames, ignore_index=True)
