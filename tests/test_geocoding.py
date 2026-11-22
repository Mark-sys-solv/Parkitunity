"""Tests for src/data/geocoding.py (Nominatim is mocked)."""

import pytest
import requests

from src import config
from src.data import geocoding, routing


class FakeResponse:
    def __init__(self, payload, status: int = 200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


@pytest.fixture
def nominatim(monkeypatch: pytest.MonkeyPatch):
    """Mock requests.get; call ``nominatim(payload)`` and get the list of recorded calls."""

    def configure(payload=None, error: Exception | None = None):
        calls: list[dict] = []

        def fake_get(url, params=None, headers=None, timeout=None):
            calls.append({"url": url, "params": params, "headers": headers})
            if error is not None:
                raise error
            return FakeResponse(payload if payload is not None else [])

        monkeypatch.setattr(geocoding.requests, "get", fake_get)
        return calls

    return configure


def test_found(nominatim) -> None:
    calls = nominatim([{"lat": "47.3703", "lon": "8.5483", "display_name": "Kunsthaus Zürich"}])
    assert geocoding.geocode("Kunsthaus Zürich") == (47.3703, 8.5483)
    request = calls[0]
    assert request["url"] == config.NOMINATIM_URL
    assert request["params"]["countrycodes"] == "ch"  # only Switzerland
    assert request["params"]["viewbox"] == config.NOMINATIM_VIEWBOX  # Zurich preferred
    assert request["params"]["limit"] == 1
    assert request["headers"]["User-Agent"].startswith("Parkitunity/")


def test_user_agent_contains_contact(nominatim, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "NOMINATIM_CONTACT", "team@example.org")
    calls = nominatim([{"lat": "1", "lon": "2"}])
    geocoding.geocode("Paradeplatz")
    assert "team@example.org" in calls[0]["headers"]["User-Agent"]


def test_not_found_and_empty_input(nominatim) -> None:
    calls = nominatim([])
    assert geocoding.geocode("Atlantis") is None
    assert geocoding.geocode("   ") is None
    assert len(calls) == 1  # empty input never reaches the API


def test_coordinates_need_no_request(nominatim) -> None:
    calls = nominatim([])
    assert geocoding.geocode("47.3697, 8.5391") == (47.3697, 8.5391)
    assert geocoding.parse_coordinates("200, 8") is None  # not a valid latitude
    assert calls == []


def test_results_are_cached(nominatim) -> None:
    calls = nominatim([{"lat": "47.37", "lon": "8.54"}])
    geocoding.geocode("Paradeplatz, Zürich")
    geocoding.geocode("paradeplatz,   zürich")  # same address, other spelling
    assert len(calls) == 1


def test_network_errors_are_not_cached(nominatim) -> None:
    calls = nominatim(error=requests.ConnectionError("offline"))
    assert geocoding.geocode("Paradeplatz") is None
    assert geocoding.geocode("Paradeplatz") is None
    assert len(calls) == 2  # asked again, because errors are not cached


def test_at_most_one_request_per_second(nominatim, monkeypatch: pytest.MonkeyPatch) -> None:
    clock, waits = [100.0], []
    monkeypatch.setattr(geocoding, "_now", lambda: clock[0])
    monkeypatch.setattr(geocoding, "_sleep", lambda s: waits.append(s) or clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(geocoding, "_last_request", 0.0)
    nominatim([{"lat": "1", "lon": "2"}])
    geocoding.geocode("Adresse A")
    clock[0] += 0.3  # the next search follows 0.3 s later
    geocoding.geocode("Adresse B")
    assert waits == [pytest.approx(0.7)]


def test_routing_geocode_returns_location_or_none(nominatim) -> None:
    nominatim([{"lat": "47.4115", "lon": "8.5441"}])
    location = routing.geocode("Bahnhof Oerlikon")
    assert (location.lat, location.lon, location.name) == (47.4115, 8.5441, "Bahnhof Oerlikon")
    nominatim([])
    assert routing.geocode("Nirgendwo") is None
