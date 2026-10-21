"""Rebuild the processed data files from the raw downloads.

Usage:
    python -m scripts.build_data --raw /path/to/raw --out /path/to/processed

Expects in --raw:
    Sensor ID and Locations.xlsx
    the occupancy zip (name contains 'occupancySensor')
Writes to --out:
    occ_G20_G25_events.parquet
    headcount_5min_G20_G25.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import clean_events, extract_events, load_metadata, scan_spaces
from src.series import build_headcount_series

# Rooms studied; only these two appear in both the occupancy data and the
# metadata sheet.
TARGET_ROOMS = ["60_G20", "60_G25"]


def main() -> None:
    """Run the data build from the command line.

    Reads the workbook and the occupancy zip from ``--raw``, extracts and
    cleans the events of the target rooms, builds the 5-minute headcount
    series, and writes both to ``--out`` as Parquet files.

    Raises:
        StopIteration: If the workbook or occupancy zip is not found in
            ``--raw``.
        SystemExit: If a target room has no events in the data.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True, help="folder with the raw files")
    p.add_argument("--out", type=Path, required=True, help="folder for processed files")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    workbook = next(args.raw.glob("Sensor ID*.xlsx"))
    occ_zip = next(args.raw.glob("*occupancySensor*.zip"))

    # Join the metadata (room names) to the space IDs that actually occur in
    # the occupancy data, keeping only the target rooms.
    _, occ_meta = load_metadata(workbook)
    meta = occ_meta.dropna(subset=["SpaceId"]).drop_duplicates("SpaceId")
    counts = scan_spaces(occ_zip)
    in_data = meta[meta["SpaceId"].isin(counts)]
    mapping = {r.room_key: r.SpaceId for r in in_data.itertuples()
               if r.room_key in TARGET_ROOMS}
    print("spaces in data:", len(counts), "| mapped target rooms:", mapping)
    missing = set(TARGET_ROOMS) - set(mapping)
    if missing:
        raise SystemExit(f"No data for {missing}; check the metadata mapping.")

    events = clean_events(extract_events(occ_zip, mapping.values()))
    events.to_parquet(args.out / "occ_G20_G25_events.parquet")

    # One regular 5-minute series per room.
    series = {room: build_headcount_series(events[events["floorspaceid"] == sid])
              for room, sid in mapping.items()}
    pd.DataFrame(series).to_parquet(args.out / "headcount_5min_G20_G25.parquet")
    print("wrote", len(events), "events and", len(series), "series")


if __name__ == "__main__":
    main()
