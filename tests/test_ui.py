from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config
from src.data import geocoding
from src.models import Garage, Location
from src.ui import backend
from src.ui.charts import mode_chart_data, occupancy_chart_data
from src.ui.map_view import DESTINATION_COLOR, OVER_BUDGET_COLOR, garage_table, map_data
from src.ui.navigation import google_maps_url, recommended_route
from src.ui.recommendation_card import build_recommendation

ARRIVAL = datetime(2026, 11, 22, 14, 30, tzinfo=backend.TZ)
DEST = Location(47.3697, 8.5391, "Paradeplatz")


def _ranked(free: int = 45, rate: float | None = 4.5, garage_id: str = "zuerichparkhausurania", name: str = "Urania",
            drive: float = 12.0, search: float = 0.0) -> dict:
    """An entry in the format of ``rank_garages``."""
    garage = Garage(garage_id, name, Location(47.3744, 8.5396), 600, free, rate)
    walk = 4.2
    return {"garage": garage, "predicted_free": free, "drive_min": drive, "search_min": search,
            "walking_min": walk, "total_min": drive + search + walk, "arrival_at_garage": ARRIVAL,
            "parking_cost_chf": rate * 2 if rate else None, "score": 0.7, "score_parts": {}}


def _comparison(car_min: float | None, transit_min: float | None, car_cost: float = 12.0,
                transit_cost: float = 9.2, recommendation: str | None = None, reason: str = "Weil.") -> dict:
    """A result in the format of ``compare_modes`` (None = option unavailable)."""
    car_ok, transit_ok = car_min is not None, transit_min is not None
    if recommendation is None and (car_ok or transit_ok):
        recommendation = "transit" if transit_ok and (not car_ok or transit_min <= car_min) else "car"
    return {
        "car": {"available": car_ok, "total_min": car_min, "drive_min": None, "search_min": None,
                "walk_min": None, "cost_chf": car_cost if car_ok else None, "parking_cost_chf": None,
                "km_cost_chf": None, "parking_garage_id": "zuerichparkhausurania" if car_ok else None,
                "predicted_free": 45 if car_ok else None, "arrival": None},
        "transit": {"available": transit_ok, "total_min": transit_min, "wait_min": 3 if transit_ok else None,
                    "duration_min": transit_min, "cost_chf": transit_cost if transit_ok else None,
                    "transfers": 1 if transit_ok else None, "departure": None, "arrival": None},
        "recommendation": recommendation,
        "reason": reason,
    }


# --- backend helpers --------------------------------------------------------------


def test_geocode_accepts_coordinates() -> None:
    assert backend.geocode(" 47.39, 8.51 ") == Location(47.39, 8.51, "47.39, 8.51")


def test_geocode_unknown_address_offline() -> None:
    assert backend.geocode("Bahnhofstrasse 1, Zürich") is None  # network is blocked in tests


@pytest.mark.parametrize(
    "free, capacity, status",
    [(200, 600, "good"), (60, 600, "tight"), (15, 600, "full"), (15, 100, "tight"), (3, 600, "full"), (25, 30, "good"), (10, None, "tight")],
)
def test_availability(free: int, capacity: int | None, status: str) -> None:
    assert backend.availability(free, capacity) == status


def test_display_name() -> None:
    assert backend.display_name(_ranked()["garage"]) == "Parkhaus Urania"
    assert backend.display_name(_ranked(garage_id="zuerichparkplatztheater11", name="Theater 11")["garage"]) == "Parkplatz Theater 11"
    assert backend.display_name(_ranked(name="Parkhaus Urania")["garage"]) == "Parkhaus Urania"


# --- recommendation card -------------------------------------------------------------


def test_card_transit_faster() -> None:
    rec = build_recommendation(_ranked(), _comparison(car_min=30, transit_min=22, reason="Der ÖV ist 8 min schneller."))
    assert rec.headline == "Nimm den ÖV – 8 min schneller"
    assert rec.reason == "Der ÖV ist 8 min schneller."
    assert any("davon 3 min warten" in d for d in rec.details)
    assert any("Parkhaus Urania" in d for d in rec.details)


def test_card_transit_slightly_slower_but_cheaper() -> None:
    rec = build_recommendation(_ranked(), _comparison(car_min=20, transit_min=22, recommendation="transit"))
    assert rec.headline == "Nimm den ÖV – CHF 2.80 günstiger"


def test_card_car_shows_time_breakdown_and_reason() -> None:
    rec = build_recommendation(_ranked(free=45, drive=12, search=5), _comparison(car_min=21, transit_min=40, reason="Das Auto ist schneller."))
    assert rec.headline == "Fahr zu Parkhaus Urania – ~45 Plätze frei"
    assert rec.reason == "Das Auto ist schneller."
    assert "12 min Fahrt + 5 min Suche + 4 min zu Fuss = 21 min" in rec.details
    assert "Ankunft am Parkhaus ca. 14:30" in rec.details


def test_card_without_comparison_and_without_garage() -> None:
    rec = build_recommendation(_ranked(drive=0), None)
    assert rec.icon == "🅿️" and rec.reason == ""
    assert "4 min zu Fuss" in rec.details  # no drive time without a start
    assert build_recommendation(None, None).headline == "Kein passendes Parkhaus gefunden"


def test_card_transit_api_down() -> None:
    rec = build_recommendation(_ranked(), _comparison(car_min=15, transit_min=None, reason="Nur Auto-Empfehlung: …"))
    assert rec.icon == "🅿️"
    assert rec.reason.startswith("Nur Auto-Empfehlung")


def test_card_nothing_available() -> None:
    rec = build_recommendation(None, _comparison(car_min=None, transit_min=None, reason="Keine Empfehlung möglich: …"))
    assert rec.headline == "Keine Empfehlung möglich"


def test_card_all_garages_full_transit_only() -> None:
    rec = build_recommendation(None, _comparison(car_min=None, transit_min=30))
    assert rec.headline == "Nimm den ÖV"
    assert not any(d.startswith("Auto") for d in rec.details)


# --- map --------------------------------------------------------------------------------


def test_map_data_colours_sizes_and_destination() -> None:
    ranked = [
        _ranked(free=200, garage_id="a", name="A"),
        _ranked(free=3, garage_id="b", name="B"),
        _ranked(free=200, rate=7.0, garage_id="c", name="C"),  # over budget
    ]
    data = map_data(ranked, DEST, recommended_id="a", max_price_chf=5.0)
    assert list(data.columns) == ["lat", "lon", "color", "size"]
    assert list(data["color"]) == [backend.STATUS_COLORS["good"], backend.STATUS_COLORS["full"], OVER_BUDGET_COLOR, DESTINATION_COLOR]
    assert data["size"].iloc[0] > data["size"].iloc[1]  # the recommendation is bigger


def test_map_data_without_destination_or_garages() -> None:
    assert len(map_data([_ranked()], None, None, 8)) == 1  # no destination dot
    empty = map_data([], None, None, 8)
    assert len(empty) == 1 and empty["color"].iloc[0] == "#00000000"  # invisible dot centres the map


def test_garage_table() -> None:
    ranked = [_ranked(free=200, garage_id="a", name="A"), _ranked(free=50, rate=7.0, garage_id="c", name="C")]
    table = garage_table(ranked, recommended_id="a", max_price_chf=5.0)
    assert list(table["Parkhaus"]) == ["Parkhaus A", "Parkhaus C"]
    assert list(table["Hinweis"]) == ["★ Empfehlung", "über Preislimit"]
    assert table["Verfügbarkeit"].iloc[0] == "Gut verfügbar"


# --- charts ---------------------------------------------------------------------------------


def test_occupancy_chart_data() -> None:
    now = ARRIVAL - timedelta(hours=1)
    history = pd.DataFrame({"time": pd.date_range(now - timedelta(hours=3), now, freq="15min"), "free_spots": 100})
    forecast = pd.DataFrame({"time": pd.date_range(now + timedelta(minutes=15), now + timedelta(hours=3), freq="15min"), "free_spots": 80})
    table = occupancy_chart_data(history, forecast)
    assert list(table.columns) == ["Gemessen", "Prognose"]
    assert table.index.tz is None  # Zurich clock time, no time zone shift in the chart
    assert table.loc[now.replace(tzinfo=None), "Prognose"] == 100  # forecast starts at the last measurement
    assert table["Gemessen"].notna().sum() == len(history)

    only_forecast = occupancy_chart_data(history.iloc[0:0], forecast)
    assert only_forecast["Prognose"].notna().sum() == len(forecast)


def test_mode_chart_data_handles_missing_options() -> None:
    table = mode_chart_data(_comparison(car_min=15, transit_min=None))
    assert list(table.index) == ["Auto", "ÖV"]
    assert table.loc["Auto", "Minuten"] == 15 and table.loc["Auto", "CHF"] == 12.0
    assert np.isnan(table.loc["ÖV", "Minuten"]) and np.isnan(table.loc["ÖV", "CHF"])


def test_forecast_is_empty_for_unknown_garage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "none.db")
    st.cache_data.clear()
    start, end = backend.forecast_window(backend.now())
    assert backend.occupancy_forecast("no-such-garage", start.isoformat(), end.isoformat()).empty
    # Known (dummy) garage while the API is down: a real forecast.
    assert len(backend.occupancy_forecast("urania", start.isoformat(), end.isoformat())) > 20


# --- Google Maps link ---------------------------------------------------------------------


def test_google_maps_url() -> None:
    url = google_maps_url(Location(47.37, 8.54), "transit", origin=Location(47.41, 8.54))
    query = parse_qs(urlparse(url).query)
    assert url.startswith("https://www.google.com/maps/dir/?api=1&")
    assert query == {"api": ["1"], "origin": ["47.41,8.54"], "destination": ["47.37,8.54"], "travelmode": ["transit"]}
    assert "origin" not in parse_qs(urlparse(google_maps_url(DEST, "driving")).query)  # Maps uses the current position


def test_route_drives_to_the_recommended_garage() -> None:
    best = _ranked(garage_id="g", name="Urania")
    url, caption = recommended_route(Location(47.41, 8.54), DEST, best, _comparison(car_min=15, transit_min=40))
    query = parse_qs(urlparse(url).query)
    assert query["travelmode"] == ["driving"]
    assert query["destination"] == [f"{best['garage'].location.lat},{best['garage'].location.lon}"]  # the garage, not the destination
    assert "Parkhaus Urania" in caption and "4 min zu Fuss" in caption


def test_route_uses_transit_when_recommended() -> None:
    url, caption = recommended_route(None, DEST, _ranked(), _comparison(car_min=30, transit_min=20))
    query = parse_qs(urlparse(url).query)
    assert query["travelmode"] == ["transit"]
    assert query["destination"] == [f"{DEST.lat},{DEST.lon}"]


def test_no_route_without_recommendation() -> None:
    assert recommended_route(None, DEST, None, None) is None
    assert recommended_route(None, DEST, None, _comparison(car_min=None, transit_min=None)) is None


# --- whole app (offline: dummy garages, no transit connection) -----------------------------


KNOWN_PLACES = {"bahnhof oerlikon, zürich": (47.4115, 8.5441), "paradeplatz, zürich": (47.3697, 8.5391)}


@pytest.fixture
def offline_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "none.db")
    monkeypatch.setattr(geocoding, "_lookup", lambda query: KNOWN_PLACES.get(query))  # fake Nominatim
    st.cache_data.clear()
    return AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=60)


def test_app_runs_offline(offline_app: AppTest) -> None:
    at = offline_app.run()
    assert not at.exception
    headlines = [m.value for m in at.markdown if m.value.startswith("###")]
    assert headlines and headlines[0].startswith("### 🅿️ Fahr zu Parkhaus")
    assert [s.value for s in at.subheader][:2] == ["Belegung", "Auto oder ÖV?"]


def test_app_price_filter_and_unknown_address(offline_app: AppTest) -> None:
    at = offline_app.run()
    at.slider[0].set_value(1.0).run()  # all dummy garages cost more
    assert any("Keine Empfehlung möglich" in m.value for m in at.markdown)  # no garage, transit offline
    at.text_input(key="destination_query").set_value("Atlantis").run()
    assert any("„Atlantis“ wurde nicht gefunden" in e.value for e in at.sidebar.error)
    assert not at.exception


def test_app_accepts_coordinates_as_destination(offline_app: AppTest) -> None:
    at = offline_app.run()
    at.text_input(key="destination_query").set_value("47.39, 8.515").run()
    assert not at.exception
    assert not at.sidebar.error
    assert any(m.value.startswith("### ") for m in at.markdown)


def test_app_shows_google_maps_button(offline_app: AppTest) -> None:
    at = offline_app.run()
    buttons = [b for b in at.get("link_button")]
    assert len(buttons) == 1
    assert "travelmode=driving" in buttons[0].proto.url  # transit is offline in tests -> car
    assert "origin=47.4115%2C8.5441" in buttons[0].proto.url
