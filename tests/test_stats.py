import numpy as np

from src.stats import cluster_boot_ci, cluster_boot_rmse_diff


def test_ci_of_constant_difference_is_that_constant():
    lo, hi = cluster_boot_ci(np.full(40, 0.5), np.repeat(np.arange(10), 4))
    assert np.isclose(lo, 0.5) and np.isclose(hi, 0.5)


def test_rmse_diff_of_identical_methods_is_zero():
    mse = np.random.default_rng(0).random(40)
    lo, hi = cluster_boot_rmse_diff(mse, mse, np.repeat(np.arange(10), 4))
    assert np.isclose(lo, 0) and np.isclose(hi, 0)


def test_ci_contains_sample_mean_and_has_sensible_width():
    rng = np.random.default_rng(1)
    v = rng.normal(2.0, 1.0, 400)
    lo, hi = cluster_boot_ci(v, np.repeat(np.arange(100), 4))
    assert lo < v.mean() < hi
    # standard error is about 1/sqrt(400) = 0.05, so a 95% CI is about 0.2 wide
    assert 0.1 < hi - lo < 0.4
