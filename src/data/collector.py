"""Log Zurich garage occupancy into SQLite and/or monthly CSV files.

Usage:
    python -m src.data.collector                    # fetch once -> SQLite
    python -m src.data.collector --loop             # fetch every COLLECTOR_INTERVAL_MIN minutes
    python -m src.data.collector --no-db --csv-dir data/occupancy   # used by GitHub Actions
    python -m src.data.collector --import-csv       # load data/occupancy/*.csv into SQLite

GitHub Actions (``.github/workflows/collect.yml``) runs this every 15 minutes
and commits the CSV files to the ``data`` branch. See README for how to pull
that data and import it locally.

An unreachable API is not an error: it is logged and nothing is written.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

from src.config import COLLECTOR_INTERVAL_MIN, DB_PATH, OCCUPANCY_CSV_DIR
from src.data import database, parking_api

logger = logging.getLogger(__name__)


def run_once(db_path: Path | None = DB_PATH, csv_dir: Path | None = None) -> int:
    """Fetch all garages once and store the measurements.

    Args:
        db_path: SQLite file to write to, or ``None`` to skip the database.
        csv_dir: If given, also append to ``<csv_dir>/<YYYY-MM>.csv``.

    Returns:
        Number of records written (new DB rows, or CSV rows if ``db_path`` is None).
    """
    records = parking_api.fetch_occupancy()
    if not records:
        logger.warning("No occupancy data received, nothing written")
        return 0

    written = 0
    if csv_dir is not None:
        written = append_csv(records, Path(csv_dir))
        logger.info("Appended %d rows to CSV in %s", written, csv_dir)
    if db_path is not None:
        written = database.insert_records(records, db_path)
        logger.info("Inserted %d new rows into %s", written, db_path)
    return written


def append_csv(records: list[dict[str, Any]], csv_dir: Path) -> int:
    """Append records to the monthly CSV file (one file per UTC month).

    Skips the batch if the file already ends with the same source timestamp,
    i.e. the source has not updated since the last run.

    Returns:
        Number of rows appended.
    """
    rows = [{**r, "timestamp": database.to_utc_iso(r["timestamp"])} for r in records]
    path = csv_dir / f"{rows[0]['timestamp'][:7]}.csv"
    if _last_timestamp(path) == rows[0]["timestamp"]:
        logger.info("Source not updated since last run (%s), skipping CSV", rows[0]["timestamp"])
        return 0

    csv_dir.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=database.COLUMNS, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def import_csv(csv_dir: Path = OCCUPANCY_CSV_DIR, db_path: Path = DB_PATH) -> int:
    """Load all ``*.csv`` files from ``csv_dir`` into SQLite (duplicates ignored).

    Returns:
        Number of newly inserted rows.
    """
    total = 0
    for path in sorted(Path(csv_dir).glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            inserted = database.insert_records(csv.DictReader(f), db_path)
        logger.info("%s: %d new rows", path.name, inserted)
        total += inserted
    return total


def run_loop(interval_min: int = COLLECTOR_INTERVAL_MIN, db_path: Path = DB_PATH) -> None:
    """Call ``run_once`` every ``interval_min`` minutes until interrupted."""
    logger.info("Collecting every %d min (Ctrl+C to stop)", interval_min)
    try:
        while True:
            started = time.monotonic()
            try:
                run_once(db_path=db_path)
            except Exception:  # keep the loop alive, e.g. on a locked DB file
                logger.exception("Collector run failed")
            time.sleep(max(0.0, interval_min * 60 - (time.monotonic() - started)))
    except KeyboardInterrupt:
        logger.info("Stopped")


def _last_timestamp(path: Path) -> str | None:
    """Timestamp of the last row in a CSV file, or ``None`` if there is none."""
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:  # only header
        return None
    return lines[-1].split(",", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Log Zurich parking garage occupancy.")
    parser.add_argument("--loop", action="store_true", help="run continuously")
    parser.add_argument("--csv-dir", type=Path, help="also append to monthly CSV files in this folder")
    parser.add_argument("--no-db", action="store_true", help="do not write to SQLite")
    parser.add_argument("--db", type=Path, default=DB_PATH, help=f"SQLite file (default: {DB_PATH})")
    parser.add_argument(
        "--import-csv",
        nargs="?",
        const=OCCUPANCY_CSV_DIR,
        type=Path,
        metavar="DIR",
        help=f"import CSV files into SQLite (default dir: {OCCUPANCY_CSV_DIR})",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.import_csv is not None:
        print(f"Imported {import_csv(args.import_csv, args.db)} new rows into {args.db}")
    elif args.loop:
        run_loop(db_path=args.db)
    else:
        written = run_once(db_path=None if args.no_db else args.db, csv_dir=args.csv_dir)
        print(f"Wrote {written} rows.")


if __name__ == "__main__":
    main()
