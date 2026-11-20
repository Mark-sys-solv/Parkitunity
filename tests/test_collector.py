from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.data import collector, database, parking_api

TS = datetime(2026, 11, 20, 12, 41, 21, tzinfo=timezone.utc)
RECORDS = [
    {"garage_id": "a", "name": "A", "free_spots": 10, "capacity": 100, "timestamp": TS, "lat": 47.0, "lon": 8.0},
    {"garage_id": "b", "name": "B", "free_spots": 3, "capacity": None, "timestamp": TS, "lat": None, "lon": None},
]


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parking_api, "fetch_occupancy", lambda **kw: RECORDS)


def test_run_once_writes_to_db(fake_api: None, tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    assert collector.run_once(db_path=db) == 2
    assert collector.run_once(db_path=db) == 0  # same source timestamp -> no duplicates
    assert len(database.load_history(db_path=db)) == 2


def test_csv_append_and_import(fake_api: None, tmp_path: Path) -> None:
    csv_dir = tmp_path / "occupancy"
    assert collector.run_once(db_path=None, csv_dir=csv_dir) == 2
    assert collector.run_once(db_path=None, csv_dir=csv_dir) == 0  # unchanged source -> skipped

    csv_file = csv_dir / "2026-11.csv"
    lines = csv_file.read_text().splitlines()
    assert lines[0] == "timestamp,garage_id,free_spots,capacity"
    assert lines[1] == "2026-11-20T12:41:21Z,a,10,100"
    assert lines[2] == "2026-11-20T12:41:21Z,b,3,"

    db = tmp_path / "imported.db"
    assert collector.import_csv(csv_dir, db) == 2
    df = database.load_history(db_path=db)
    assert df.set_index("garage_id")["capacity"].isna().to_dict() == {"a": False, "b": True}


def test_api_failure_does_not_crash(tmp_path: Path) -> None:
    # Real fetch_occupancy with blocked network (see conftest).
    assert collector.run_once(db_path=tmp_path / "test.db", csv_dir=tmp_path / "csv") == 0
    assert not (tmp_path / "csv").exists()
