from datetime import datetime, timezone

import pytest

from src.data import parking_api
from tests.conftest import load_fixture

EXPECTED_KEYS = {"garage_id", "name", "free_spots", "capacity", "timestamp", "lat", "lon"}


@pytest.fixture
def parkendd_response(monkeypatch: pytest.MonkeyPatch) -> dict:
    data = load_fixture("parkendd_zuerich.json")
    monkeypatch.setattr(parking_api, "get_json", lambda *a, **kw: data)
    return data


def test_fetch_occupancy_parses_records(parkendd_response: dict) -> None:
    records = parking_api.fetch_occupancy()
    by_id = {r["garage_id"]: r for r in records}

    assert all(set(r) == EXPECTED_KEYS for r in records)
    assert "zuerichparkhauspwest" not in by_id  # state "nodata" is skipped
    assert len(records) == len(parkendd_response["lots"]) - 1

    accu = by_id["zuerichparkhausaccu"]
    assert (accu["free_spots"], accu["capacity"]) == (162, 194)
    assert accu["lat"] == pytest.approx(47.414848)
    assert accu["timestamp"] == datetime(2026, 11, 20, 12, 41, 21, tzinfo=timezone.utc)


def test_feed_quirks_are_cleaned(parkendd_response: dict) -> None:
    by_id = {r["garage_id"]: r for r in parking_api.fetch_occupancy()}
    # Feed reports 69 free of 59 total -> clipped to capacity.
    assert by_id["zuerichparkhausmaxbillplatz"]["free_spots"] == 59
    # total == 0 means unknown, missing coords stay None.
    hafenenge = by_id["zuerichparkhaushafenenge"]
    assert hafenenge["capacity"] is None
    assert hafenenge["lat"] is None and hafenenge["lon"] is None


def test_fetch_occupancy_returns_empty_list_when_api_down() -> None:
    # Network is blocked by conftest -> ConnectionError is handled.
    assert parking_api.fetch_occupancy() == []


def test_fetch_garages_skips_incomplete_lots(parkendd_response: dict) -> None:
    ids = {g.id for g in parking_api.fetch_garages()}
    assert "zuerichparkhausaccu" in ids
    assert "zuerichparkhaushafenenge" not in ids  # no coordinates / capacity
    assert "zuerichparkhauscrowneplaza" not in ids  # no coordinates


def test_fetch_garages_falls_back_to_dummy_data() -> None:
    garages = parking_api.fetch_garages()
    assert [g.id for g in garages] == [g.id for g in parking_api._DUMMY_GARAGES]


@pytest.mark.live
def test_live_parkendd() -> None:
    records = parking_api.fetch_occupancy()
    assert len(records) > 10
    assert all(r["free_spots"] >= 0 for r in records)
