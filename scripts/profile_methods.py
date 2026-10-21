"""Performance study: timing, fit times, profile, memory and FLOPs.

Usage:
    python -m scripts.profile_methods \
        --series /path/headcount_5min_G20_G25.parquet \
        --events /path/occ_G20_G25_events.parquet \
        --summary results/final_test_summary_busy_gaps.csv \
        --out docs/performance

``--events`` is optional; without it the series-build memory is skipped.
Tables are written as CSV files and the cProfile output as ``cprofile.txt``.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import resource
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.experiment import GAP_LENGTHS, score_gaps
from src.features import FEATURE_NAMES, gap_features, make_training_set
from src.gaps import (carry_forward, decay_profile, eligible_starts,
                      linear_interp, time_slots, weekly_profile)
from src.models import SVR_C, TAU_BINS, TREE_DEPTH
from src.series import build_headcount_series

INTERPRETABILITY = {
    "carry_forward": "very high",
    "linear_interp": "very high",
    "weekly_profile": "high",
    "decay_profile": "high (one parameter, tau)",
    "tree_sq_d5": "medium (readable splits)",
    "tree_abs_d5": "medium (readable splits)",
    "svr_c003": "low (kernel model)",
}


def timed(fn):
    """Run ``fn()`` and return ``(result, seconds)``."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def peak_mb(fn):
    """Run ``fn()`` under tracemalloc and return ``(result, peak_MB)``.

    tracemalloc sees Python and numpy allocations but not memory that C
    libraries (libsvm, for example) allocate themselves.
    """
    tracemalloc.start()
    out = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, peak / 1e6


def fit_models(Xt, Yt):
    """Fit the three frozen models one by one and time each fit.

    Mirrors ``src.models.fit_frozen_models`` (same settings and seeds) so each
    fit can be timed separately.

    Returns:
        tuple[dict, pd.DataFrame]: Fitted models and a table of fit times.
    """
    sv = np.random.default_rng(5).choice(len(Xt), min(8000, len(Xt)), replace=False)
    sb = np.random.default_rng(3).choice(len(Xt), min(60000, len(Xt)), replace=False)
    specs = {
        "tree_sq_d5": (DecisionTreeRegressor(max_depth=TREE_DEPTH, min_samples_leaf=50,
                                             random_state=0), np.arange(len(Xt))),
        "tree_abs_d5": (DecisionTreeRegressor(criterion="absolute_error", max_depth=TREE_DEPTH,
                                              min_samples_leaf=50, random_state=0), sb),
        "svr_c003": (make_pipeline(StandardScaler(),
                                   SVR(kernel="rbf", C=SVR_C, epsilon=0.1, gamma="scale")), sv),
    }
    models, rows = {}, []
    for name, (m, idx) in specs.items():
        _, secs = timed(lambda: m.fit(Xt[idx], Yt[idx]))
        models[name] = m
        rows.append({"model": name, "train_rows": len(idx), "fit_seconds": secs})
    return models, pd.DataFrame(rows)


def per_gap_timing(y, pick, slots, prof, models):
    """Time every method per gap, for every gap length.

    Learned models are charged for building their features as well as for
    ``predict``, since the features are needed to use them.

    Returns:
        pd.DataFrame: ``gap``, ``method``, ``ms_per_gap``.
    """
    n = len(pick)

    def per(fn):
        t0 = time.perf_counter()
        for i in pick:
            fn(i)
        return 1000 * (time.perf_counter() - t0) / n

    rows = []
    for label, g in GAP_LENGTHS.items():
        X, t_feat = timed(lambda: np.vstack([gap_features(y, i, g, slots, prof) for i in pick]))
        ms = {
            "carry_forward": per(lambda i: carry_forward(y, i, g)),
            "linear_interp": per(lambda i: linear_interp(y, i, g)),
            "weekly_profile": per(lambda i: prof[slots[i:i + g]]),
            "decay_profile": per(lambda i: decay_profile(y, i, g, slots, prof, TAU_BINS)),
        }
        for name, m in models.items():
            _, dt = timed(lambda: m.predict(X))
            ms[name] = 1000 * (t_feat + dt) / n
        rows += [{"gap": label, "method": k, "ms_per_gap": v} for k, v in ms.items()]
    return pd.DataFrame(rows)


def flops_table(models) -> pd.DataFrame:
    """Rough arithmetic operations per hidden bin and per 24 h gap.

    Counts one multiply, add, compare or exponential as one operation. These
    are order-of-magnitude estimates, not measurements.
    """
    n_feat = len(FEATURE_NAMES)
    feat = 12                                           # gap_features work per bin (approx.)
    n_sv = models["svr_c003"].named_steps["svr"].support_vectors_.shape[0]
    per_sv = 3 * n_feat + 2                             # RBF: diff, square, add, scale, exp, weight
    per_bin = {
        "carry_forward": 0,
        "linear_interp": 2,
        "weekly_profile": 0,
        "decay_profile": 6,                             # lerp, exp, multiply, add, clip
        "tree_sq_d5": feat + TREE_DEPTH,
        "tree_abs_d5": feat + TREE_DEPTH,
        "svr_c003": feat + 2 * n_feat + n_sv * per_sv,
    }
    return pd.DataFrame([{"method": k, "ops_per_bin": v, "ops_per_24h_gap": v * 288,
                          "note": f"{n_sv} support vectors" if k == "svr_c003" else ""}
                         for k, v in per_bin.items()])


def main() -> None:
    """Run the performance study and write all tables to ``--out``.

    Raises:
        KeyError: If the series file has no ``60_G20`` column.
        ValueError: If no eligible 24 h test gap exists.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--events", type=Path, default=None)
    p.add_argument("--summary", type=Path, default=Path("results/final_test_summary_busy_gaps.csv"))
    p.add_argument("--out", type=Path, default=Path("docs/performance"))
    p.add_argument("--n-gaps", type=int, default=200, help="test gaps per length for timing")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    s = pd.read_parquet(args.series)["60_G20"]
    y = s.to_numpy()
    n = len(y)
    tr, te = int(0.6 * n), int(0.8 * n)
    slots = time_slots(s.index)
    prof = weekly_profile(y, slots, tr)

    st = eligible_starts(y, 288)
    st = st[(st >= te + 1) & (st + 289 <= n)]
    if len(st) == 0:
        raise ValueError("no eligible 24 h test gap")
    pick = np.random.default_rng(9001).choice(st, size=min(args.n_gaps, len(st)), replace=False)

    # ---- stage times and fit times (no tracing, so timings are not inflated)
    (Xt, Yt), t_train = timed(lambda: make_training_set(y, slots, prof, 1, tr, 4000, seed=11))
    models, fits = fit_models(Xt, Yt)
    print(f"training set: {Xt.shape[0]} rows in {t_train:.2f} s")
    print(fits.round(2).to_string(index=False))
    fits.to_csv(args.out / "fit_times.csv", index=False)

    timing = per_gap_timing(y, pick, slots, prof, models)
    timing.to_csv(args.out / "timing_per_gap.csv", index=False)
    print("\nms per gap:")
    print(timing.pivot_table(index="gap", columns="method", values="ms_per_gap")
          .loc[list(GAP_LENGTHS)].round(3).to_string())

    # ---- cProfile of the full scoring step
    pr = cProfile.Profile()
    pr.enable()
    _, t_score = timed(lambda: score_gaps(y, pick, slots, prof, models))
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(15)
    (args.out / "cprofile.txt").write_text(
        f"score_gaps on {len(pick)} gaps x {len(GAP_LENGTHS)} lengths: {t_score:.2f} s\n\n"
        + buf.getvalue())
    print(f"\nscore_gaps: {t_score:.2f} s; cProfile saved to cprofile.txt")

    # ---- peak memory (Python/numpy allocations only)
    mem = []
    if args.events is not None:
        ev = pd.read_parquet(args.events)
        sid = ev["floorspaceid"].value_counts().idxmax()    # busiest space; cost is similar
        one = ev[ev["floorspaceid"] == sid]
        _, mb = peak_mb(lambda: build_headcount_series(one))
        mem.append({"stage": "series build (one room)", "peak_MB": mb})
    _, mb = peak_mb(lambda: make_training_set(y, slots, prof, 1, tr, 4000, seed=11))
    mem.append({"stage": "training set (4000 gaps)", "peak_MB": mb})
    _, mb = peak_mb(lambda: score_gaps(y, pick[:50], slots, prof, models))
    mem.append({"stage": "scoring (50 gaps x 8 lengths)", "peak_MB": mb})
    mem = pd.DataFrame(mem)
    mem.to_csv(args.out / "memory.csv", index=False)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print("\n", mem.round(1).to_string(index=False))
    print(f"whole-process peak RSS (includes C libraries): {rss:.0f} MB")

    # ---- FLOPs
    fl = flops_table(models)
    fl.to_csv(args.out / "flops.csv", index=False)
    print("\n", fl.to_string(index=False))

    # ---- baseline vs selected comparison (G.20 accuracy from the saved results)
    summ = pd.read_csv(args.summary)
    acc = (summ[(summ["room"] == "60_G20") & summ["gap"].isin(["1 h", "24 h"])]
           .pivot_table(index="method", columns="gap", values="MAE")
           .rename(columns={"1 h": "MAE_1h", "24 h": "MAE_24h"}))
    ms = timing.groupby("method")["ms_per_gap"].mean().rename("mean_ms_per_gap")
    comp = acc.join(ms).join(fl.set_index("method")["ops_per_24h_gap"])
    comp["interpretability"] = pd.Series(INTERPRETABILITY)
    comp.round(3).to_csv(args.out / "comparison.csv")
    print("\n", comp.round(3).to_string())


if __name__ == "__main__":
    main()
