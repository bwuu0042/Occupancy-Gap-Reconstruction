"""Sensitivity of gap reconstruction to bin size and weekly-slot width.

Usage:
    python -m scripts.sensitivity --series /path/headcount_5min_G20_G25.parquet --out docs/sensitivity
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.gaps import (carry_forward, decay_profile, eligible_starts,
                      linear_interp, time_slots, weekly_profile)

# Gap lengths are fixed in minutes so results stay comparable across bin sizes.
GAP_MINUTES = {"1 h": 60, "4 h": 240, "12 h": 720}


def run(s: pd.Series, bin_min: int, slot_min: int, tau_hours: float = 2.0,
        n_gaps: int = 1000, seed: int = 9001) -> pd.DataFrame:
    """Measure busy-gap MAE and run time for one (bin size, slot width) setting.

    Only the rule-based methods are used, so no model is trained and nothing
    is re-tuned. The series is coarsened by taking the first value in each
    bin.

    Args:
        s: Headcount series (people) at 5-minute resolution.
        bin_min: Bin width in minutes (5, 15, 30 or 60).
        slot_min: Weekly-profile slot width in minutes.
        tau_hours: Decay time constant in hours.
        n_gaps: Maximum number of test gaps per gap length.
        seed: Seed of the random generator, for reproducibility.

    Returns:
        pd.DataFrame: One row per (gap length, method) with ``bin_min``,
        ``slot_min``, ``gap``, ``method``, ``n_busy`` (gaps containing
        occupancy), ``mae_busy`` (people) and ``ms_per_gap`` (milliseconds).
    """
    sr = s if bin_min == 5 else s.resample(f"{bin_min}min").first()
    y = sr.to_numpy()
    n = len(y)
    train_hi, test_lo = int(0.6 * n), int(0.8 * n)   # same 60/20/20 split as the main experiment
    slots = time_slots(sr.index, slot_min)
    prof = weekly_profile(y, slots, train_hi, slot_min)
    tau_bins = tau_hours * 60 / bin_min              # keep tau fixed in hours, not bins
    methods = {
        "carry_forward": carry_forward,
        "linear_interp": linear_interp,
        "weekly_profile": lambda y_, i, g: prof[slots[i:i + g]],
        "decay_profile": lambda y_, i, g: decay_profile(y_, i, g, slots, prof, tau_bins),
    }
    rows = []
    for label, minutes in GAP_MINUTES.items():
        g = minutes // bin_min                       # gap length in bins at this resolution
        st = eligible_starts(y, g)
        st = st[(st >= test_lo + 1) & (st + g + 1 <= n)]   # test period only
        pick = np.random.default_rng(seed).choice(st, size=min(n_gaps, len(st)), replace=False)
        for name, fn in methods.items():
            t0 = time.perf_counter()
            errs, busy = [], []
            for i in pick:
                truth = y[i:i + g]
                errs.append(np.abs(fn(y, i, g) - truth).mean())
                busy.append(truth.max() >= 1)
            secs = time.perf_counter() - t0
            errs, busy = np.array(errs), np.array(busy)
            rows.append({"bin_min": bin_min, "slot_min": slot_min, "gap": label,
                         "method": name, "n_busy": int(busy.sum()),
                         "mae_busy": errs[busy].mean(), "ms_per_gap": 1000 * secs / len(pick)})
    return pd.DataFrame(rows)


def main() -> None:
    """Run the bin-size and slot-width sensitivity study from the command line.

    Writes ``bin_size.csv`` and ``slot_width.csv`` to ``--out`` and prints
    the busy-gap MAE tables and the per-gap run time.

    Raises:
        KeyError: If the series file has no ``60_G20`` column.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--series", type=Path, required=True, help="headcount parquet file")
    p.add_argument("--out", type=Path, required=True, help="folder for the CSV tables")
    args = p.parse_args()
    s = pd.read_parquet(args.series)["60_G20"]

    bins = pd.concat([run(s, b, 30) for b in (5, 15, 30, 60)], ignore_index=True)
    slots = pd.concat([run(s, 5, w) for w in (15, 30, 60)], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    bins.to_csv(args.out / "bin_size.csv", index=False)
    slots.to_csv(args.out / "slot_width.csv", index=False)
    print("bin size, busy MAE:")
    print(bins.pivot_table(index=["gap", "method"], columns="bin_min", values="mae_busy").round(3))
    print("\nslot width, busy MAE:")
    print(slots.pivot_table(index=["gap", "method"], columns="slot_min", values="mae_busy").round(3))
    print("\nms per gap (5-min, 30-min slots):")
    print(bins[bins["bin_min"] == 5].groupby("method")["ms_per_gap"].mean().round(3))


if __name__ == "__main__":
    main()
