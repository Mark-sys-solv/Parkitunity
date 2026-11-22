"""Download the historical occupancy of all Zurich garages from ParkenDD.

Usage:
    python -m src.data.download_history                          # last 12 months
    python -m src.data.download_history --start 2025-01-01 --end 2025-06-30
    python -m src.data.download_history --garage zuerichparkhausurania   # one garage (quick test)
    python -m src.data.download_history --no-db                  # only write the CSV files

Output:
    data/raw/history_<garage_id>.csv   one file per finished garage (raw, UTC)
    data/processed/history.csv         all garages: timestamp, garage_id, garage_name, free, total
    data/garages.csv                   id, name, lat, lon, total
    + import into the SQLite database (config.DB_PATH) used by training and the app

The API needs ~10 s per request, so 12 months for all garages take several
hours. That is why every garage is saved on its own as soon as it is finished.
If the download stops, simply start it again: garages that already have a
file are skipped. At the end all files are merged into history.csv.
If the API does not answer for several weeks in a row (server or internet
down), the download stops by itself – start it again later.

The files do not remember which period they cover. To download another period
(or one garage again), delete the matching data/raw/history_*.csv files first.

How the API works (https://opendatazurich.github.io/parkendd-api, tested Nov 2026):
- List of garages:  https://api.parkendd.de/Zuerich
- History of one garage:
  https://api.parkendd.de/Zuerich/<garage_id>/timespan?version=1.1&from=2024-09-23T00:00:00&to=2024-09-29T23:59:59
- At most 7 days per request (longer ranges give HTTP 400), and at most about
  2016 rows per answer – if a week has more measurements, we ask again from
  the last timestamp we got.
- One measurement roughly every 5 minutes. Timestamps are in UTC.
- Data is available from March 2021 on, with a gap from 18 July to 18 October 2026.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, GARAGES_CSV_PATH, HISTORY_CSV_PATH, PARKENDD_URL, RAW_DATA_DIR, ZURICH_TZ
from src.data import database
from src.data.http_utils import get_json

logger = logging.getLogger(__name__)

DEFAULT_DAYS = 365  # default period: the last 12 months
CHUNK_DAYS = 7  # the API allows at most 7 days per request
ROW_LIMIT = 2000  # answers with this many rows may be cut off -> ask for the rest
PAUSE_S = 0.5  # be polite: wait between requests
TIMEOUT_S = 60  # history requests can take a while
GAP_HOURS = 6  # a pause between two measurements longer than this counts as a gap
MAX_FAILURES_IN_A_ROW = 3  # this many failed weeks in a row -> the API is down, stop the download
API_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"  # what the API expects and returns (UTC, no zone suffix)
PARTS_DIR = RAW_DATA_DIR  # one file per finished garage: data/raw/history_<garage_id>.csv
RAW_COLUMNS = ["timestamp", "garage_id", "free"]  # raw measurements, timestamp in UTC as the API sends it


# --- Talking to the API -----------------------------------------------------------


def fetch_garage_list() -> pd.DataFrame:
    """All Zurich garages with metadata: id, name, lat, lon, total.

    Unlike ``parking_api.fetch_occupancy`` this keeps garages that currently
    send no data – they may still have a history.
    """
    data = get_json(PARKENDD_URL, timeout=TIMEOUT_S, retries=2)
    if not isinstance(data, dict) or "lots" not in data:
        raise SystemExit(f"Could not load the list of garages from {PARKENDD_URL} – try again later.")
    rows = []
    for lot in data["lots"]:
        coords = lot.get("coords") or {}
        rows.append(
            {
                "id": lot["id"],
                "name": lot.get("name") or lot["id"],
                "lat": coords.get("lat"),
                "lon": coords.get("lng"),
                "total": lot.get("total") or None,  # 0 means "unknown"
            }
        )
    garages = pd.DataFrame(rows, columns=["id", "name", "lat", "lon", "total"])
    garages["total"] = garages["total"].astype("Int64")  # whole numbers, empty if unknown
    return garages


def week_chunks(start: datetime, end: datetime, days: int = CHUNK_DAYS) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into pieces of at most ``days`` days."""
    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=days), end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return chunks


def fetch_timespan(garage_id: str, start: datetime, end: datetime, pause_s: float = PAUSE_S) -> list[dict] | None:
    """All measurements of one garage between start and end (UTC, at most 7 days).

    Returns:
        List of ``{"timestamp": "...", "free": int}`` (may be empty), or
        ``None`` if the API failed (the error is logged by ``get_json``).
    """
    url = f"{PARKENDD_URL}/{garage_id}/timespan"
    rows: list[dict] = []
    cursor = start
    while cursor < end:
        params = {"version": "1.1", "from": cursor.strftime(API_TIME_FORMAT), "to": end.strftime(API_TIME_FORMAT)}
        data = get_json(url, params=params, timeout=TIMEOUT_S, retries=2)
        if not isinstance(data, dict):
            return rows or None  # keep what we already have
        batch = data.get("data") or []
        rows.extend(batch)
        if len(batch) < ROW_LIMIT:
            break  # we got everything
        # The answer may have been cut off: continue right after the last measurement.
        last = datetime.strptime(batch[-1]["timestamp"], API_TIME_FORMAT)
        if last <= cursor:
            break  # safety: never loop forever
        cursor = last + timedelta(seconds=1)
        time.sleep(pause_s)
    return rows


# --- Downloading everything ------------------------------------------------------------


class ApiUnavailable(Exception):
    """ParkenDD did not answer several times in a row – no point in going on."""


def part_file(garage_id: str, folder: Path) -> Path:
    """File with the finished download of one garage, e.g. data/raw/history_zuerichparkhausaccu.csv."""
    return folder / f"history_{garage_id}.csv"


def download_garage(garage_id: str, start: datetime, end: datetime, pause_s: float = PAUSE_S) -> tuple[pd.DataFrame, list[str]]:
    """Download the history of one garage week by week.

    Returns:
        (measurements with columns timestamp [UTC string], garage_id, free,
        first day of every week that failed, e.g. ["2026-03-02"])

    Raises:
        ApiUnavailable: if MAX_FAILURES_IN_A_ROW weeks in a row failed.
    """
    records: list[dict] = []
    failed_weeks: list[str] = []
    failures_in_a_row = 0
    for chunk_start, chunk_end in week_chunks(start, end):
        rows = fetch_timespan(garage_id, chunk_start, chunk_end, pause_s)
        if rows is None:
            failed_weeks.append(f"{chunk_start:%Y-%m-%d}")
            failures_in_a_row += 1
            if failures_in_a_row >= MAX_FAILURES_IN_A_ROW:
                raise ApiUnavailable(f"{failures_in_a_row} weeks in a row failed for {garage_id}")
            rows = []
        else:
            failures_in_a_row = 0
        records.extend({"timestamp": r["timestamp"], "garage_id": garage_id, "free": r["free"]} for r in rows)
        time.sleep(pause_s)
    return pd.DataFrame(records, columns=RAW_COLUMNS), failed_weeks


def save_part(part: pd.DataFrame, path: Path) -> None:
    """Save the measurements of one garage.

    We first write a temporary file and rename it at the end. If the program
    is stopped while writing, there is no half-written history_*.csv that
    would wrongly be skipped next time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    part.to_csv(temporary, index=False)
    temporary.replace(path)


def download(garages: pd.DataFrame, start: datetime, end: datetime, folder: Path, pause_s: float = PAUSE_S) -> list[str]:
    """Download every garage that has no file in ``folder`` yet.

    - After each garage, its measurements are saved as history_<garage_id>.csv.
    - Garages that already have such a file are skipped (resume after a stop).
    - Garages where a week failed are not saved, so the next run tries them again.
    - Ctrl+C, or an API that stops answering, ends the download early;
      finished garages stay saved.

    Returns:
        Failed requests as text, e.g. ["zuerichparkhausaccu 2026-03-02"].
    """
    failed: list[str] = []
    try:
        for number, garage in enumerate(garages.itertuples(), start=1):
            progress = f"[{number}/{len(garages)}] {garage.name}"
            path = part_file(garage.id, folder)
            if path.exists():
                logger.info("%s: already downloaded – skipped (%s)", progress, path.name)
                continue
            part, failed_weeks = download_garage(garage.id, start, end, pause_s)
            if failed_weeks:
                failed.extend(f"{garage.id} {week}" for week in failed_weeks)
                logger.warning("%s: %d week(s) failed – not saved, run again to retry", progress, len(failed_weeks))
                continue
            save_part(part, path)
            logger.info("%s: %d measurements saved to %s", progress, len(part), path.name)
    except ApiUnavailable as error:
        logger.error("ParkenDD does not answer (%s) – stopped. Finished garages are saved, run again later.", error)
    except KeyboardInterrupt:
        logger.warning("Stopped by user – finished garages are saved, run again to continue.")
    return failed


def merge_parts(garage_ids: list[str], start: datetime, end: datetime, folder: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read the files of these garages and put them into one table.

    Only measurements between start and end are kept, in case a file comes
    from an earlier run with a longer period.

    Returns:
        (raw measurements with columns timestamp [UTC string], garage_id, free,
        ids of garages that have no file yet)
    """
    parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for garage_id in garage_ids:
        path = part_file(garage_id, folder)
        if not path.exists():
            missing.append(garage_id)
            continue
        part = pd.read_csv(path)
        if not part.empty:  # a garage without measurements has a file with only the header
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=RAW_COLUMNS), missing
    raw = pd.concat(parts, ignore_index=True)
    times = pd.to_datetime(raw["timestamp"])
    raw = raw[(times >= start) & (times < end)]
    return raw.reset_index(drop=True), missing


def clean_history(raw: pd.DataFrame, garages: pd.DataFrame) -> pd.DataFrame:
    """Final table: Zurich time, garage name and capacity added, no duplicates, sorted."""
    history = raw.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True).dt.tz_convert(ZURICH_TZ)
    names = garages.set_index("id")
    history["garage_name"] = history["garage_id"].map(names["name"])
    history["total"] = history["garage_id"].map(names["total"]).astype("Int64")
    history["free"] = history["free"].astype("Int64")
    history = history.drop_duplicates(subset=["garage_id", "timestamp"])
    history = history.sort_values(["timestamp", "garage_id"]).reset_index(drop=True)
    return history[["timestamp", "garage_id", "garage_name", "free", "total"]]


def save_csv(history: pd.DataFrame, garages: pd.DataFrame, history_path: Path, garages_path: Path) -> None:
    """Write both CSV files. Timestamps keep their UTC offset, e.g. 2026-11-22T13:10:00+01:00."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    garages_path.parent.mkdir(parents=True, exist_ok=True)
    out = history.copy()
    out["timestamp"] = out["timestamp"].map(lambda t: t.isoformat())
    out.to_csv(history_path, index=False)
    garages.to_csv(garages_path, index=False)


def import_into_db(history: pd.DataFrame, db_path: Path, batch_size: int = 100_000) -> int:
    """Copy the history into the SQLite database used by training and the app.

    Returns:
        Number of new rows (rows that already exist are skipped).
    """
    if database.is_synthetic(db_path):
        logger.error("%s contains SYNTHETIC data – not mixing real history into it.", db_path)
        return 0
    table = history.rename(columns={"free": "free_spots", "total": "capacity"})
    table = table[["timestamp", "garage_id", "free_spots", "capacity"]].astype({"capacity": object})
    table["capacity"] = table["capacity"].where(table["capacity"].notna(), None)
    inserted = 0
    for first in range(0, len(table), batch_size):  # in batches to keep memory use low
        inserted += database.insert_records(table.iloc[first:first + batch_size].to_dict("records"), db_path)
    return inserted


def find_gaps(history: pd.DataFrame, gap_hours: float = GAP_HOURS) -> pd.DataFrame:
    """Garages with long pauses between two measurements.

    Returns:
        One row per garage with at least one gap: garage_name, gaps (count),
        longest_gap_days, longest_gap_from.
    """
    rows = []
    for garage_id, group in history.groupby("garage_id"):
        times = group["timestamp"].sort_values().reset_index(drop=True)
        pauses = times.diff()  # time since the previous measurement
        long_pauses = pauses[pauses > pd.Timedelta(hours=gap_hours)]
        if long_pauses.empty:
            continue
        longest = long_pauses.idxmax()  # position of the first measurement after the longest gap
        rows.append(
            {
                "garage_name": group["garage_name"].iloc[0],
                "gaps": len(long_pauses),
                "longest_gap_days": round(long_pauses.max() / pd.Timedelta(days=1), 1),
                "longest_gap_from": times[longest - 1],  # last measurement before the gap
            }
        )
    return pd.DataFrame(rows, columns=["garage_name", "gaps", "longest_gap_days", "longest_gap_from"])


def print_summary(history: pd.DataFrame, garages: pd.DataFrame, failed: list[str], missing: list[str] = ()) -> None:
    """Short report at the end of the download.

    ``missing`` are the ids of garages that have not been downloaded yet.
    """
    with_data = history["garage_id"].nunique()
    names = garages.set_index("id")["name"]
    missing_names = sorted(names[garage_id] for garage_id in missing)
    print("\n=== Summary ===")
    print(f"Garages:  {with_data} with data (of {len(garages)} in the list)")
    if not history.empty:
        print(f"Period:   {history['timestamp'].min():%Y-%m-%d %H:%M} – {history['timestamp'].max():%Y-%m-%d %H:%M} (Zurich time)")
    print(f"Rows:     {len(history):,}")
    if missing_names:
        print(f"Missing:  {', '.join(missing_names)} – not downloaded yet, run again to continue")
    without = sorted(set(garages["name"]) - set(history["garage_name"]) - set(missing_names))
    if without:
        print(f"No data:  {', '.join(without)}")
    gaps = find_gaps(history)
    print(f"Gaps (> {GAP_HOURS} h without data) in {len(gaps)} garages:")
    for row in gaps.sort_values("longest_gap_days", ascending=False).itertuples():
        print(f"  - {row.garage_name}: {row.gaps} gaps, longest {row.longest_gap_days} days from {row.longest_gap_from:%Y-%m-%d}")
    if failed:
        print(f"Failed requests: {len(failed)} (run again to retry), e.g. {', '.join(failed[:5])}")


# --- Command line ---------------------------------------------------------------------


def parse_day(text: str) -> datetime:
    """'2025-01-31' -> datetime at midnight UTC."""
    return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical occupancy of Zurich garages (ParkenDD).")
    parser.add_argument("--start", type=parse_day, help="first day YYYY-MM-DD (default: 12 months ago)")
    parser.add_argument("--end", type=parse_day, help="last day YYYY-MM-DD, inclusive (default: yesterday)")
    parser.add_argument("--garage", action="append", help="only this garage id (can be repeated)")
    parser.add_argument("--pause", type=float, default=PAUSE_S, help=f"seconds between requests (default {PAUSE_S})")
    parser.add_argument("--no-db", action="store_true", help="do not import into the SQLite database")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Default end: today 00:00 UTC – stays the same all day, so the cache can be reused.
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    end = args.end + timedelta(days=1) if args.end else today
    start = args.start or end - timedelta(days=DEFAULT_DAYS)
    if start >= end:
        raise SystemExit("--start must be before --end")

    all_garages = fetch_garage_list()  # garages.csv always lists every garage
    garages = all_garages[all_garages["id"].isin(args.garage)] if args.garage else all_garages
    print(f"Downloading {len(garages)} garages from {start:%Y-%m-%d} to {end:%Y-%m-%d} "
          f"({len(week_chunks(start, end))} weeks each) …")

    # The API works with UTC times without a time zone.
    api_start, api_end = start.replace(tzinfo=None), end.replace(tzinfo=None)
    failed = download(garages, api_start, api_end, PARTS_DIR, args.pause)  # one file per garage

    # Merge the files of all finished garages into one table.
    raw, missing = merge_parts(list(garages["id"]), api_start, api_end, PARTS_DIR)
    history = clean_history(raw, all_garages)
    save_csv(history, all_garages, HISTORY_CSV_PATH, GARAGES_CSV_PATH)
    print(f"Saved {HISTORY_CSV_PATH} and {GARAGES_CSV_PATH}")
    if not args.no_db:
        print(f"Imported {import_into_db(history, DB_PATH):,} new rows into {DB_PATH}")
    print_summary(history, garages, failed, missing)


if __name__ == "__main__":
    main()
