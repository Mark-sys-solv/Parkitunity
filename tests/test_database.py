from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data import database

T0 = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    database.init_db(path)
    return path


def _record(garage_id: str, ts: datetime, free: int, capacity: int | None = 100) -> dict:
    return {"garage_id": garage_id, "timestamp": ts, "free_spots": free, "capacity": capacity}


def test_insert_and_load(db: Path) -> None:
    records = [_record("a", T0, 10), _record("b", T0, 20, None), _record("a", T0 + timedelta(minutes=15), 5)]
    assert database.insert_records(records, db) == 3

    df = database.load_history(db_path=db)
    assert list(df.columns) == ["timestamp", "garage_id", "free_spots", "capacity"]
    assert len(df) == 3
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df.loc[df["garage_id"] == "b", "capacity"].isna().all()


def test_duplicates_are_ignored(db: Path) -> None:
    database.insert_records([_record("a", T0, 10)], db)
    assert database.insert_records([_record("a", T0, 10), _record("a", T0 + timedelta(minutes=15), 8)], db) == 1
    assert len(database.load_history(db_path=db)) == 2


def test_timestamps_are_normalised_to_utc(db: Path) -> None:
    # Same instant given as UTC+1 string -> counts as a duplicate.
    database.insert_records([_record("a", T0, 10)], db)
    assert database.insert_records([_record("a", "2026-11-20T13:00:00+01:00", 10)], db) == 0


def test_filters(db: Path) -> None:
    database.insert_records(
        [_record("a", T0 + timedelta(hours=h), h) for h in range(4)] + [_record("b", T0, 1)], db
    )
    assert set(database.load_history("a", db_path=db)["garage_id"]) == {"a"}

    since = database.load_history(since=T0 + timedelta(hours=2), db_path=db)
    assert len(since) == 2
    assert (since["timestamp"] >= pd.Timestamp(T0 + timedelta(hours=2))).all()


def test_empty_db_returns_empty_frame(tmp_path: Path) -> None:
    df = database.load_history(db_path=tmp_path / "new.db")
    assert df.empty
    assert list(df.columns) == ["timestamp", "garage_id", "free_spots", "capacity"]
