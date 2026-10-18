"""Frozen model settings chosen on the validation period."""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

TAU_BINS = 24          # decay time constant (2 h of 5-minute bins)
TREE_DEPTH = 5
SVR_C = 0.03


def fit_frozen_models(X: np.ndarray, Y: np.ndarray) -> dict:
    """Fit the three learned models with the settings frozen before testing.

    The SVR is trained on 8,000 rows and the absolute-error tree on 60,000
    (both random subsamples) because of their training cost.
    """
    svr_idx = np.random.default_rng(5).choice(len(X), min(8000, len(X)), replace=False)
    abs_idx = np.random.default_rng(3).choice(len(X), min(60000, len(X)), replace=False)
    return {
        "tree_sq_d5": DecisionTreeRegressor(max_depth=TREE_DEPTH, min_samples_leaf=50,
                                            random_state=0).fit(X, Y),
        "tree_abs_d5": DecisionTreeRegressor(criterion="absolute_error",
                                             max_depth=TREE_DEPTH, min_samples_leaf=50,
                                             random_state=0).fit(X[abs_idx], Y[abs_idx]),
        "svr_c003": make_pipeline(StandardScaler(),
                                  SVR(kernel="rbf", C=SVR_C, epsilon=0.1, gamma="scale")
                                  ).fit(X[svr_idx], Y[svr_idx]),
    }
