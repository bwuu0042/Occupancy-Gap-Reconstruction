"""Convert irregular occupancy event logs into regular headcount series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_headcount_series(events: pd.DataFrame, freq: str = "5min",
                           max_hold: str = "30min") -> pd.Series:
    """Resample one space's event log onto a regular time grid.

    The sensor only writes a row when the state changes, so silence is
    ambiguous: it can mean "nothing changed" or "the feed stopped". The rule
    used here resolves this conservatively. While the room is ``NotOccupied``
    the last value is held until the next event (an empty room stays empty).
    While the room is occupied the value is held for at most ``max_hold``;
    after that the bin is NaN (state unknown), so a real outage never becomes
    ground truth for the gap experiments.

    Args:
        events: Events of a single space with tz-aware ``time``,
            ``occupancystatus`` ("NotOccupied" or an occupied label) and
            ``headcount`` (people, >= 0) columns.
        freq: Bin width as a pandas offset string, for example ``"5min"``.
        max_hold: Longest time an occupied state is held without a new event.

    Returns:
        pd.Series: Headcount per bin (people), NaN where the state is unknown,
        indexed by Australia/Melbourne time. The first and last bins are
        rounded inwards to whole ``freq`` steps.

    Raises:
        KeyError: If a required column is missing from ``events``.
    """
    e = events.sort_values("time").copy()
    # Work in UTC so daylight-saving changes cannot create duplicate or
    # missing bins; convert back to local time at the end.
    e["t"] = e["time"].dt.tz_convert("UTC")
    nxt = e["t"].shift(-1)                       # time of the following event
    cap = e["t"] + pd.Timedelta(max_hold)        # latest an occupied state may be held
    quiet_ok = e["occupancystatus"].astype(str).eq("NotOccupied")
    # Empty room: valid until the next event. Occupied: valid until the next
    # event or the hold limit, whichever comes first. The final event has no
    # successor, so its validity is left open (NaT).
    e["valid_until"] = nxt.where(quiet_ok, cap.where(cap < nxt, nxt))

    grid = pd.DataFrame({"t": pd.date_range(e["t"].min().ceil(freq),
                                            e["t"].max().floor(freq), freq=freq)})
    grid["t"] = grid["t"].astype(e["t"].dtype)
    # For each grid time take the most recent event at or before it.
    merged = pd.merge_asof(grid, e[["t", "headcount", "valid_until"]],
                           on="t", direction="backward")
    # Bins past the validity limit become unknown instead of carrying forward.
    merged.loc[merged["t"] >= merged["valid_until"], "headcount"] = np.nan

    out = merged.set_index("t")["headcount"]
    out.index = out.index.tz_convert("Australia/Melbourne")
    return out
