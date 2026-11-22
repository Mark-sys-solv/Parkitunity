"""SQLite storage for logged garage occupancy.

Timestamps are stored as UTC ISO 8601 strings (``2026-11-20T12:41:21Z``), so
they sort and compare correctly as text. ``(garage_id, timestamp)`` is the
primary key: inserting a measurement twice (e.g. when the downloaded history
is imported again) is silently ignored.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DB_PATH

COLUMNS = ["timestamp", "garage_id", "free_spots", "capacity"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS occupancy (
    timestamp   TEXT    NOT NULL,  -- UTC, ISO 8601
    garage_id   TEXT    NOT NULL,
    free_spots  INTEGER NOT NULL,
    capacity    INTEGER,           -- NULL if unknown
    PRIMARY KEY (garage_id, timestamp)
);
-- Key/value metadata about the dataset, e.g. synthetic = "true".
CREATE TABLE IF NOT EXISTS dataset_info (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a connection to the SQLite database (creates the folder if needed)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: Path = DB_PATH) -> None:
    """Create the tables if they do not exist yet (safe to call repeatedly)."""
    with closing(get_connection(db_path)) as conn, conn:
        conn.executescript(SCHEMA)


def insert_records(records: Iterable[Mapping[str, Any]], db_path: Path = DB_PATH) -> int:
    """Store occupancy measurements, skipping ones that already exist.

    Args:
        records: Dicts with at least ``timestamp`` (datetime or ISO string),
            ``garage_id``, ``free_spots`` and optionally ``capacity`` – e.g.
            the output of ``parking_api.fetch_occupancy()``. Extra keys are ignored.

    Returns:
        Number of newly inserted rows.
    """
    rows = [
        (to_utc_iso(r["timestamp"]), str(r["garage_id"]), int(r["free_spots"]), _optional_int(r.get("capacity")))
        for r in records
    ]
    if not rows:
        return 0
    with closing(get_connection(db_path)) as conn, conn:
        conn.executescript(SCHEMA)
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO occupancy (timestamp, garage_id, free_spots, capacity) VALUES (?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes - before


def load_history(
    garage_id: str | None = None,
    since: datetime | str | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Load occupancy history, sorted by garage and time.

    Args:
        garage_id: Only this garage (default: all).
        since: Only measurements at or after this time. Naive datetimes are
            treated as UTC.

    Returns:
        DataFrame with columns ``timestamp`` (datetime64, UTC), ``garage_id``,
        ``free_spots`` and ``capacity`` (nullable Int64).
    """
    query = f"SELECT {', '.join(COLUMNS)} FROM occupancy"
    clauses: list[str] = []
    params: list[Any] = []
    if garage_id is not None:
        clauses.append("garage_id = ?")
        params.append(garage_id)
    if since is not None:
        clauses.append("timestamp >= ?")
        params.append(to_utc_iso(since))
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY garage_id, timestamp"

    init_db(db_path)
    with closing(get_connection(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["free_spots"] = df["free_spots"].astype("Int64")
    df["capacity"] = df["capacity"].astype("Int64")
    return df


def set_dataset_info(info: Mapping[str, str], db_path: Path = DB_PATH) -> None:
    """Store metadata about the dataset (e.g. ``{"synthetic": "true"}``)."""
    with closing(get_connection(db_path)) as conn, conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT OR REPLACE INTO dataset_info (key, value) VALUES (?, ?)",
            [(k, str(v)) for k, v in info.items()],
        )


def get_dataset_info(db_path: Path = DB_PATH) -> dict[str, str]:
    """Return the dataset metadata (empty dict if none or the DB does not exist)."""
    if not Path(db_path).exists():
        return {}
    init_db(db_path)
    with closing(get_connection(db_path)) as conn:
        return dict(conn.execute("SELECT key, value FROM dataset_info").fetchall())


def is_synthetic(db_path: Path = DB_PATH) -> bool:
    """True if the database was filled by the synthetic data generator."""
    return get_dataset_info(db_path).get("synthetic") == "true"


def to_utc_iso(value: datetime | str) -> str:
    """Normalise a datetime or ISO string to ``YYYY-MM-DDTHH:MM:SSZ`` (UTC)."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _optional_int(value: Any) -> int | None:
    """Convert to int; ``None``/empty/NaN (e.g. from CSV) become ``None``."""
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return None
    return int(value)
