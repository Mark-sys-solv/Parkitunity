"""Tests for src/data/download_history.py with a simulated ParkenDD API."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data import database
from src.data import download_history as dh

FMT = dh.API_TIME_FORMAT
START = datetime(2026, 11, 2)  # naive UTC, like the API
LOTS = [
    {"id": "a", "name": "A", "coords": {"lat": 47.37, "lng": 8.54}, "total": 100, "free": 5, "state": "open"},
    {"id": "b", "name": "B", "coords": None, "total": 0, "free": 0, "state": "nodata"},  # no coords, unknown size
]
# One measurement every 5 minutes (7 s after the slot) for three weeks.
TIMES = [START + timedelta(minutes=5 * k, seconds=7) for k in range(3 * 7 * 288)]


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch):
    """Simulated API. Call ``api(limit=, fail=, missing=)``; returns the list of calls."""

    def configure(limit: int = 10_000, fail=lambda garage, frm: False, missing=lambda garage, t: False):
        calls: list[tuple[str, dict | None]] = []

        def fake_get_json(url: str, params: dict | None = None, **kwargs):
            calls.append((url, params))
            if url == dh.PARKENDD_URL:
                return {"lots": LOTS}
            garage = url.split("/")[-2]
            frm, to = datetime.strptime(params["from"], FMT), datetime.strptime(params["to"], FMT)
            if fail(garage, frm):
                return None
            rows = [{"timestamp": t.strftime(FMT), "free": k % 100}
                    for k, t in enumerate(TIMES) if frm <= t <= to and not missing(garage, t)]
            return {"data": rows[:limit]}  # the real API cuts long answers off

        monkeypatch.setattr(dh, "get_json", fake_get_json)
        return calls

    return configure


def download_raw(garages: pd.DataFrame, start: datetime, end: datetime, folder: Path) -> pd.DataFrame:
    """Download and merge, like main() does."""
    dh.download(garages, start, end, folder, pause_s=0)
    raw, _ = dh.merge_parts(list(garages["id"]), start, end, folder)
    return raw


def test_week_chunks() -> None:
    chunks = dh.week_chunks(START, START + timedelta(days=20))
    assert [(a.day, b.day) for a, b in chunks] == [(2, 9), (9, 16), (16, 22)]


def test_garage_list(api) -> None:
    api()
    garages = dh.fetch_garage_list()
    assert list(garages.columns) == ["id", "name", "lat", "lon", "total"]
    assert garages.loc[0, "total"] == 100
    assert pd.isna(garages.loc[1, "total"]) and pd.isna(garages.loc[1, "lat"])  # 0 / missing -> empty


def test_cut_off_answers_are_completed(api, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = api(limit=100)
    monkeypatch.setattr(dh, "ROW_LIMIT", 100)
    rows = dh.fetch_timespan("a", START, START + timedelta(days=1), pause_s=0)
    assert len(rows) == 288  # one day, all 5-minute measurements
    assert len({r["timestamp"] for r in rows}) == 288  # no duplicates
    assert len(calls) == 3


def test_failed_request_keeps_partial_data(api, monkeypatch: pytest.MonkeyPatch) -> None:
    api(limit=100, fail=lambda garage, frm: frm > START)  # every follow-up request fails
    monkeypatch.setattr(dh, "ROW_LIMIT", 100)
    assert len(dh.fetch_timespan("a", START, START + timedelta(days=1), pause_s=0)) == 100
    api(fail=lambda garage, frm: True)
    assert dh.fetch_timespan("a", START, START + timedelta(days=1), pause_s=0) is None


def test_each_garage_is_saved_and_skipped_next_time(api, tmp_path: Path) -> None:
    garages = pd.DataFrame({"id": ["a", "b"], "name": ["A", "B"]})
    end = START + timedelta(days=14)
    api(fail=lambda garage, frm: garage == "b" and frm >= START + timedelta(days=7))  # 2nd week of "b" fails
    failed = dh.download(garages, START, end, tmp_path, pause_s=0)
    assert failed == ["b 2026-11-09"]
    assert [f.name for f in tmp_path.iterdir()] == ["history_a.csv"]  # incomplete "b" is not saved, no temp files
    assert len(pd.read_csv(tmp_path / "history_a.csv")) == 14 * 288

    calls = api()  # second run: "a" is skipped, only "b" is downloaded again
    assert dh.download(garages, START, end, tmp_path, pause_s=0) == []
    assert calls and all("/b/" in url for url, _ in calls)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["history_a.csv", "history_b.csv"]


def test_ctrl_c_keeps_finished_garages(api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api()
    garages = pd.DataFrame({"id": ["a", "b"], "name": ["A", "B"]})
    real_fetch = dh.fetch_timespan

    def fetch_until_ctrl_c(garage_id, start, end, pause_s):
        if garage_id == "b":
            raise KeyboardInterrupt  # the user presses Ctrl+C during garage "b"
        return real_fetch(garage_id, start, end, pause_s)

    monkeypatch.setattr(dh, "fetch_timespan", fetch_until_ctrl_c)
    assert dh.download(garages, START, START + timedelta(days=1), tmp_path, pause_s=0) == []
    assert [f.name for f in tmp_path.iterdir()] == ["history_a.csv"]


def test_stops_when_the_api_is_down(api, tmp_path: Path) -> None:
    garages = pd.DataFrame({"id": ["a", "b", "c"], "name": ["A", "B", "C"]})
    end = START + timedelta(days=21)
    calls = api(fail=lambda garage, frm: garage != "a")  # the API breaks down after garage "a"
    assert dh.download(garages, START, end, tmp_path, pause_s=0) == []
    assert [f.name for f in tmp_path.iterdir()] == ["history_a.csv"]  # "a" is saved
    assert not any("/c/" in url for url, _ in calls)  # stopped during "b", "c" was never asked
    assert sum("/b/" in url for url, _ in calls) == dh.MAX_FAILURES_IN_A_ROW


def test_single_failed_weeks_do_not_stop_the_download(api, tmp_path: Path) -> None:
    garages = pd.DataFrame({"id": ["a", "b"], "name": ["A", "B"]})

    def every_other_week(garage, frm):  # weeks 1 and 3 of "a" fail, never two in a row
        return garage == "a" and (frm - START).days // 7 % 2 == 0

    api(fail=every_other_week)
    failed = dh.download(garages, START, START + timedelta(days=28), tmp_path, pause_s=0)
    assert failed == ["a 2026-11-02", "a 2026-11-16"]  # "a" is incomplete, but the download goes on
    assert [f.name for f in tmp_path.iterdir()] == ["history_b.csv"]


def test_merge_parts(tmp_path: Path) -> None:
    pd.DataFrame({"timestamp": ["2026-11-01T23:55:07", "2026-11-02T00:00:07"], "garage_id": "a", "free": [1, 2]}).to_csv(
        tmp_path / "history_a.csv", index=False)
    pd.DataFrame(columns=dh.RAW_COLUMNS).to_csv(tmp_path / "history_b.csv", index=False)  # garage without data
    raw, missing = dh.merge_parts(["a", "b", "c"], START, START + timedelta(days=1), tmp_path)
    assert list(raw.columns) == dh.RAW_COLUMNS
    assert list(raw["free"]) == [2]  # the measurement before START is outside the period
    assert missing == ["c"]  # "c" has no file yet


def test_clean_history(api) -> None:
    api()
    garages = dh.fetch_garage_list()
    raw = pd.DataFrame(
        {
            "timestamp": ["2026-11-02T10:05:07", "2026-11-02T10:00:07", "2026-11-02T10:00:07"],  # unsorted + duplicate
            "garage_id": ["a", "a", "a"],
            "free": [4, 5, 5],
        }
    )
    history = dh.clean_history(raw, garages)
    assert list(history.columns) == ["timestamp", "garage_id", "garage_name", "free", "total"]
    assert len(history) == 2
    assert history["timestamp"].iloc[0].isoformat() == "2026-11-02T11:00:07+01:00"  # Zurich time
    assert list(history["free"]) == [5, 4]
    assert (history["garage_name"] == "A").all() and (history["total"] == 100).all()


def test_csv_files(api, tmp_path: Path) -> None:
    api()
    garages = dh.fetch_garage_list()
    raw = download_raw(garages.head(1), START, START + timedelta(days=1), tmp_path / "raw")
    history = dh.clean_history(raw, garages)
    dh.save_csv(history, garages, tmp_path / "history.csv", tmp_path / "garages.csv")

    lines = (tmp_path / "history.csv").read_text().splitlines()
    assert lines[0] == "timestamp,garage_id,garage_name,free,total"
    assert lines[1] == "2026-11-02T01:00:07+01:00,a,A,0,100"
    garage_lines = (tmp_path / "garages.csv").read_text().splitlines()
    assert garage_lines[0] == "id,name,lat,lon,total"
    assert garage_lines[1:] == ["a,A,47.37,8.54,100", "b,B,,,"]


def test_import_into_db(api, tmp_path: Path) -> None:
    api()
    garages = dh.fetch_garage_list()
    raw = download_raw(garages, START, START + timedelta(days=1), tmp_path / "raw")
    history = dh.clean_history(raw, garages)
    db = tmp_path / "real.db"
    assert dh.import_into_db(history, db, batch_size=100) == 2 * 288
    assert dh.import_into_db(history, db) == 0  # importing again adds nothing
    loaded = database.load_history("b", db_path=db)
    assert len(loaded) == 288 and loaded["capacity"].isna().all()

    synthetic = tmp_path / "synthetic.db"
    database.set_dataset_info({"synthetic": "true"}, synthetic)
    assert dh.import_into_db(history, synthetic) == 0  # never mix real and synthetic data


def test_find_gaps(api, tmp_path: Path) -> None:
    api(missing=lambda garage, t: garage == "b" and START + timedelta(days=3) <= t < START + timedelta(days=5))
    garages = dh.fetch_garage_list()
    raw = download_raw(garages, START, START + timedelta(days=7), tmp_path)
    gaps = dh.find_gaps(dh.clean_history(raw, garages))
    assert list(gaps["garage_name"]) == ["B"]
    assert gaps.loc[0, "gaps"] == 1
    assert gaps.loc[0, "longest_gap_days"] == 2.0
    assert gaps.loc[0, "longest_gap_from"].date().isoformat() == "2026-11-05"  # Zurich: 4 Nov 23:55 UTC -> 5 Nov


def test_main(api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    calls = api()
    monkeypatch.setattr(dh, "HISTORY_CSV_PATH", tmp_path / "processed" / "history.csv")
    monkeypatch.setattr(dh, "GARAGES_CSV_PATH", tmp_path / "garages.csv")
    monkeypatch.setattr(dh, "DB_PATH", tmp_path / "parkitunity.db")
    monkeypatch.setattr(dh, "PARTS_DIR", tmp_path / "raw")
    monkeypatch.setattr(sys, "argv", ["download_history", "--start", "2026-11-02", "--end", "2026-11-03", "--pause", "0"])
    dh.main()

    out = capsys.readouterr().out
    assert "Garages:  2 with data (of 2 in the list)" in out
    assert f"Rows:     {2 * 2 * 288:,}" in out
    assert "Missing" not in out
    assert sorted(f.name for f in (tmp_path / "raw").iterdir()) == ["history_a.csv", "history_b.csv"]
    first_csv = (tmp_path / "processed" / "history.csv").read_text()
    assert len(pd.read_csv(tmp_path / "processed" / "history.csv")) == 2 * 2 * 288
    assert len(database.load_history(db_path=tmp_path / "parkitunity.db")) == 2 * 2 * 288

    # Second run: both garages are already saved -> no history requests, same result.
    calls.clear()
    dh.main()
    assert all(url == dh.PARKENDD_URL for url, _ in calls)  # only the list of garages
    assert (tmp_path / "processed" / "history.csv").read_text() == first_csv


def test_summary_names_missing_garages(api, capsys: pytest.CaptureFixture) -> None:
    api()
    garages = dh.fetch_garage_list()
    history = dh.clean_history(pd.DataFrame(columns=dh.RAW_COLUMNS), garages)
    dh.print_summary(history, garages, failed=[], missing=["b"])
    out = capsys.readouterr().out
    assert "Missing:  B – not downloaded yet, run again to continue" in out
    assert "No data:  A" in out
