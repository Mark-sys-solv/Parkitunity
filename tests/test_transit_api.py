from datetime import datetime, timedelta, timezone

import pytest

from src.data import http_utils, transit_api
from src.models import Location
from tests.conftest import load_fixture

DEPARTURE = datetime(2026, 11, 20, 14, 0)


@pytest.fixture
def api_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Serve the recorded API response and record the query params of each call."""
    calls: list[dict] = []
    data = load_fixture("transport_connections.json")

    def fake_get_json(url: str, params: dict | None = None, **kwargs: object) -> dict:
        calls.append(params or {})
        return data

    monkeypatch.setattr(transit_api, "get_json", fake_get_json)
    return calls


def test_parse_duration() -> None:
    assert transit_api.parse_duration("00d00:23:00") == 23
    assert transit_api.parse_duration("01d02:03:30") == 1440 + 123.5


def test_returns_connection_with_earliest_arrival(api_calls: list[dict]) -> None:
    result = transit_api.get_transit_options("Zürich Oerlikon", "Zürich Paradeplatz", DEPARTURE)

    assert result is not None
    assert set(result) == {"duration_min", "transfers", "departure", "arrival", "sections"}
    # The 2nd recorded connection (S19 + tram 7, arr. 14:19) beats the 1st (arr. 14:24).
    assert result["duration_min"] == 17
    assert result["transfers"] == 1
    assert result["arrival"] == datetime(2026, 11, 20, 13, 19, tzinfo=timezone.utc)

    modes = [(s["mode"], s["line"]) for s in result["sections"]]
    assert modes == [("walk", None), ("transit", "S19"), ("walk", None), ("transit", "T7")]
    assert result["sections"][1]["summary"] == "S19 Zürich Oerlikon → Zürich HB (5 min)"
    assert result["sections"][0]["duration_min"] == 2


def test_query_params_and_coordinates(api_calls: list[dict]) -> None:
    transit_api.get_transit_options((47.4111, 8.5442), Location(47.3697, 8.5389), DEPARTURE)
    params = api_calls[0]
    assert params["from"] == "47.4111,8.5442"
    assert params["to"] == "47.3697,8.5389"
    assert (params["date"], params["time"]) == ("2026-11-20", "14:00")


def test_aware_departure_is_converted_to_zurich_time(api_calls: list[dict]) -> None:
    transit_api.get_transit_options("A", "B", datetime(2026, 11, 20, 13, 0, tzinfo=timezone.utc))
    assert api_calls[0]["time"] == "14:00"  # UTC+1 in November


def test_same_request_is_cached_for_five_minutes(api_calls: list[dict], monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(http_utils, "_now", lambda: clock[0])

    transit_api.get_transit_options("A", "B", DEPARTURE)
    transit_api.get_transit_options("A", "B", DEPARTURE + timedelta(seconds=30))  # same minute
    assert len(api_calls) == 1

    transit_api.get_transit_options("A", "C", DEPARTURE)  # different request
    assert len(api_calls) == 2

    clock[0] += 301  # cache expired
    transit_api.get_transit_options("A", "B", DEPARTURE)
    assert len(api_calls) == 3


def test_api_failure_returns_none_and_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(transit_api, "get_json", lambda *a, **kw: calls.append(1))  # returns None
    assert transit_api.get_transit_options("A", "B", DEPARTURE) is None
    assert transit_api.get_transit_options("A", "B", DEPARTURE) is None
    assert len(calls) == 2


def test_network_error_is_handled() -> None:
    # Real get_json with blocked network (see conftest).
    assert transit_api.get_transit_options("A", "B", DEPARTURE) is None


@pytest.mark.live
def test_live_transport_opendata() -> None:
    result = transit_api.get_transit_options((47.4111, 8.5442), "Zürich Paradeplatz")
    assert result is not None
    assert result["duration_min"] > 0
    assert result["sections"]
