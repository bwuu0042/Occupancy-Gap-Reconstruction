"""Load the sensor metadata and extract occupancy events for chosen spaces."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

OCC_COLUMNS = ["id", "deviceid", "floorspaceid", "occupancystatus",
               "headcount", "collecteddate"]


def make_room_key(df: pd.DataFrame) -> pd.Series:
    """Build a building_room key that ignores punctuation and case.

    Labels such as ``G.57`` and ``G57`` give the same key, so the two
    metadata sheets can be matched.

    Args:
        df: Metadata table with a ``Building`` column (for example 60.0 or
            "60") and a ``Floor/Room`` column (for example "G.57").

    Returns:
        pd.Series: One text key per row, such as ``60_G57``, with the same
        index as ``df``.

    Raises:
        KeyError: If the ``Building`` or ``Floor/Room`` column is missing.
    """
    building = df["Building"].astype(str).str.replace(r"\.0$", "", regex=True)
    room = (df["Floor/Room"].astype(str).str.upper()
            .str.replace(r"[^A-Z0-9]", "", regex=True))
    return building + "_" + room


def load_metadata(workbook) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the ENV (sheet 0) and OCC (sheet 1) mapping sheets as text.

    Reading as text keeps the 13-digit sensor IDs intact. Both frames get a
    ``room_key`` column, and the ENV frame gets a clean ``sensor_id``.

    Args:
        workbook: Path to ``Sensor ID and Locations.xlsx``. The ENV sheet must
            be the first sheet and the OCC sheet the second.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: ``(env, occ)``. All original
        columns are text. ``env`` has the added columns ``sensor_id`` and
        ``room_key``; ``occ`` has the added column ``room_key``.

    Raises:
        FileNotFoundError: If ``workbook`` does not exist.
        ValueError: If the workbook has fewer than two sheets.
        KeyError: If an expected column (``ID``, ``Building`` or
            ``Floor/Room``) is missing.
    """
    env = pd.read_excel(workbook, sheet_name=0, dtype=str)
    occ = pd.read_excel(workbook, sheet_name=1, dtype=str)
    env["sensor_id"] = env["ID"].str.replace(r"\.0$", "", regex=True)
    env["room_key"] = make_room_key(env)
    occ["room_key"] = make_room_key(occ)
    return env, occ


def scan_spaces(occ_zip, chunksize: int = 500_000) -> Counter:
    """Count events per ``floorspaceid`` by reading the big file in chunks.

    Args:
        occ_zip: Path to the zipped occupancy CSV.
        chunksize: Rows read per chunk. Larger chunks are faster but use more
            memory.

    Returns:
        Counter: Mapping from each ``floorspaceid`` to its number of events.

    Raises:
        FileNotFoundError: If ``occ_zip`` does not exist.
        ValueError: If the file has no ``floorspaceid`` column.
    """
    counts: Counter = Counter()
    for chunk in pd.read_csv(occ_zip, usecols=["floorspaceid"], dtype=str,
                             chunksize=chunksize):
        counts.update(chunk["floorspaceid"].value_counts().to_dict())
    return counts


def extract_events(occ_zip, space_ids, chunksize: int = 500_000) -> pd.DataFrame:
    """Keep only the events of ``space_ids`` from the large occupancy file.

    Times are converted to Australia/Melbourne local time, headcount and id
    to numbers, and text columns to categories.

    Args:
        occ_zip: Path to the zipped occupancy CSV (about 9 million rows).
        space_ids: Iterable of ``floorspaceid`` strings to keep, for example
            the IDs of G.20 and G.25.
        chunksize: Rows read per chunk. The file is streamed so it never has
            to fit in memory whole; larger chunks are faster but use more
            memory.

    Returns:
        pd.DataFrame: One row per event of the requested spaces, with columns
        ``id``, ``deviceid``, ``floorspaceid``, ``occupancystatus``,
        ``headcount`` and a timezone-aware ``time`` (Australia/Melbourne).
        Values that cannot be parsed become NaN or NaT instead of raising an
        error.

    Raises:
        FileNotFoundError: If ``occ_zip`` does not exist.
        ValueError: If no event matches ``space_ids``, or the file lacks the
            expected columns.
    """
    space_ids = set(space_ids)
    pieces = []
    for chunk in pd.read_csv(occ_zip, dtype=str, usecols=OCC_COLUMNS,
                             chunksize=chunksize):
        keep = chunk[chunk["floorspaceid"].isin(space_ids)]
        if keep.empty:
            continue
        pieces.append(keep.assign(
            time=pd.to_datetime(keep["collecteddate"], utc=True, errors="coerce")
                   .dt.tz_convert("Australia/Melbourne"),
            headcount=pd.to_numeric(keep["headcount"], errors="coerce"),
            id=pd.to_numeric(keep["id"], errors="coerce"),
        ).drop(columns="collecteddate"))
    if not pieces:
        raise ValueError("No events found for the requested space IDs; "
                         "check the IDs and the archive.")
    events = pd.concat(pieces, ignore_index=True)
    for col in ["deviceid", "floorspaceid", "occupancystatus"]:
        events[col] = events[col].astype("category")
    return events


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate (space, timestamp) rows, keeping the highest ``id``.

    Args:
        events: Event table from :func:`extract_events`, with the columns
            ``floorspaceid``, ``time`` and ``id``.

    Returns:
        pd.DataFrame: Events sorted by space, time and id, with one row per
        space and timestamp and a fresh index.

    Raises:
        KeyError: If a required column is missing.
    """
    return (events.sort_values(["floorspaceid", "time", "id"])
                  .drop_duplicates(["floorspaceid", "time"], keep="last")
                  .reset_index(drop=True))
