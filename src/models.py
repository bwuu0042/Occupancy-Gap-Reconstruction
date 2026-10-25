"""Learned-model settings and decay time constant selected on validation data.

The learned-model settings and decay time constant were selected using G.20
validation gaps and applied unchanged to G.25. Rule-based test results were
inspected during development of the decay profile, so the test period was not
an untouched holdout.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

TAU_BINS = 24          # decay time constant: 24 bins = 2 h of 5-minute bins
TREE_DEPTH = 5         # shallowest depth before validation error rises (overfitting)
SVR_C = 0.03           # SVR regularisation; 0.01-0.1 gave near-identical results


def fit_frozen_models(X: np.ndarray, Y: np.ndarray) -> dict:
    """Fit the three learned models using the validation-selected settings.

    The SVR is trained on 8,000 rows and the absolute-error tree on 60,000
    (random subsamples with fixed seeds) because their training cost grows
    quickly with the number of rows. The squared-error tree uses all rows.

    Args:
        X: Feature matrix from ``features.make_training_set``, shape
            ``(n_rows, 8)``.
        Y: Anomaly target (people), length ``n_rows``.

    Returns:
        dict: Fitted models keyed ``"tree_sq_d5"``, ``"tree_abs_d5"`` and
        ``"svr_c003"``. Each has a ``predict`` method returning the anomaly.
    """
    # Fixed seeds so the subsamples (and therefore the results) are repeatable.
    svr_idx = np.random.default_rng(5).choice(len(X), min(8000, len(X)), replace=False)
    abs_idx = np.random.default_rng(3).choice(len(X), min(60000, len(X)), replace=False)
    return {
        # Squared-error tree: its leaves predict the mean anomaly.
        "tree_sq_d5": DecisionTreeRegressor(max_depth=TREE_DEPTH, min_samples_leaf=50,
                                            random_state=0).fit(X, Y),
        # Absolute-error tree: leaves predict the median, which suits data that
        # is mostly zero with rare large values.
        "tree_abs_d5": DecisionTreeRegressor(criterion="absolute_error",
                                             max_depth=TREE_DEPTH, min_samples_leaf=50,
                                             random_state=0).fit(X[abs_idx], Y[abs_idx]),
        # SVR is sensitive to feature scale, so standardise inside a pipeline.
        "svr_c003": make_pipeline(StandardScaler(),
                                  SVR(kernel="rbf", C=SVR_C, epsilon=0.1, gamma="scale")
                                  ).fit(X[svr_idx], Y[svr_idx]),
    }
