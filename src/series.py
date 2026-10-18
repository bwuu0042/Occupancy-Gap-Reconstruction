"""Convert irregular occupancy event logs into regular headcount series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_headcount_series(events: pd.DataFrame, freq: str = "5min",
                           max_hold: str = "30min") -> pd.Series:
    """Resample one space's event log onto a regular time grid.

    The headcount in each bin is the last reported value. While the room is
    ``NotOccupied`` the value is held until the next event. While the room is
    occupied it is held for at most ``max_hold``; after that the bin is NaN
    (state unknown), so real outages never become ground truth.

    Args:
        events: Data frame with tz-aware ``time``, ``occupancystatus`` and
            ``headcount`` columns, for a single space.
        freq: Bin width, a pandas offset string.
        max_hold: Longest time an occupied state is held without a new event.

    Returns:
        Series of headcounts indexed by Australia/Melbourne time.
    """
    e = events.sort_values("time").copy()
    e["t"] = e["time"].dt.tz_convert("UTC")
    nxt = e["t"].shift(-1)
    cap = e["t"] + pd.Timedelta(max_hold)
    quiet_ok = e["occupancystatus"].astype(str).eq("NotOccupied")
    e["valid_until"] = nxt.where(quiet_ok, cap.where(cap < nxt, nxt))

    grid = pd.DataFrame({"t": pd.date_range(e["t"].min().ceil(freq),
                                            e["t"].max().floor(freq), freq=freq)})
    grid["t"] = grid["t"].astype(e["t"].dtype)
    merged = pd.merge_asof(grid, e[["t", "headcount", "valid_until"]],
                           on="t", direction="backward")
    merged.loc[merged["t"] >= merged["valid_until"], "headcount"] = np.nan

    out = merged.set_index("t")["headcount"]
    out.index = out.index.tz_convert("Australia/Melbourne")
    return out
