import numpy as np

from src.features import gap_features, make_training_set
from src.gaps import weekly_profile


def _setup(n=1200):
    rng = np.random.default_rng(1)
    y = rng.integers(0, 6, n).astype(float)
    slots = np.arange(n) % 336
    profile = weekly_profile(y, slots, train_end=700)
    return y, slots, profile


def test_features_never_read_hidden_bins():
    y, slots, profile = _setup()
    i, gap = 900, 30
    y2 = y.copy()
    y2[i:i + gap] = 999
    assert np.allclose(gap_features(y, i, gap, slots, profile),
                       gap_features(y2, i, gap, slots, profile))


def test_feature_shape():
    y, slots, profile = _setup()
    assert gap_features(y, 900, 30, slots, profile).shape == (30, 8)


def test_training_gaps_stay_inside_training_period():
    y, slots, profile = _setup()
    y_marked = y.copy()
    y_marked[800:] = np.nan            # nothing after 800 may be used
    X, Y = make_training_set(y_marked, slots, profile, 1, 700, 50, seed=0)
    assert not np.isnan(X).any() and not np.isnan(Y).any()
    assert len(X) == len(Y)


def test_training_set_is_reproducible():
    y, slots, profile = _setup()
    a = make_training_set(y, slots, profile, 1, 700, 30, seed=3)
    b = make_training_set(y, slots, profile, 1, 700, 30, seed=3)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
