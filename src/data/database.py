"""SQLite storage for logged garage occupancy."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS occupancy (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    garage_id    TEXT    NOT NULL,
    timestamp    TEXT    NOT NULL,  -- ISO 8601
    free_spots   INTEGER NOT NULL,
    capacity     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_occupancy_garage_time ON occupancy (garage_id, timestamp);
"""


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open a connection to the SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they do not exist yet."""
    # TODO: add a table for weather snapshots if needed.
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def insert_occupancy(
    garage_id: str,
    timestamp: datetime,
    free_spots: int,
    capacity: int | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """Store one occupancy measurement."""
    # TODO: implement (parameterised INSERT; consider executemany for batches).
    raise NotImplementedError


def load_occupancy(
    garage_id: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Load occupancy history as a DataFrame.

    Returns:
        Columns: garage_id, timestamp (datetime64), free_spots, capacity.
    """
    # TODO: implement with pd.read_sql_query and optional filters.
    raise NotImplementedError
