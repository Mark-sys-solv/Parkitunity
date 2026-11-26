from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import config
from src.data import geocoding, transit_api
from src.models import Location
from src.recommendation.mode_compare import NO_TRANSIT, compare_modes
from src.ui import backend
from src.ui.charts import mode_chart_data
from src.ui.map_view import (
    DESTINATION_MARKER,
    MAX_ZOOM,
    OVER_BUDGET_COLOR,
    START_MARKER,
    fit_zoom,
    garage_table,
    hex_to_rgba,
    map_data,
)
from src.ui.navigation import google_maps_url

ARRIVAL = datetime(2026, 11, 22, 14, 30, tzinfo=backend.TZ)
DEST = Location(47.3697, 8.5391, "Paradeplatz")


def _ranked(free: int = 45, rate: float = 4.5, garage_id: str = "zuerichparkhausurania", name: str = "Urania",
            drive: float = 12.0) -> dict:
    """A row of the ranking as dict."""
    return {"garage_id": garage_id, "name": name, "lat": 47.3744, "lon": 8.5396, "capacity": 600, "free_now": 50,
            "drive_min": drive, "drive_km": 5.0, "walk_min": 4, "price_per_hour": rate,
            "arrival_time": ARRIVAL, "free_at_arrival": free, "score": 0.7, "rank": 1}


def _table(*rows: dict) -> pd.DataFrame:
    """Several rows as a ranking table."""
    return pd.DataFrame(list(rows), columns=list(_ranked()))


# --- backend helpers --------------------------------------------------------------


def test_geocode_accepts_coordinates() -> None:
    assert backend.geocode(" 47.39, 8.51 ") == Location(47.39, 8.51, "47.39, 8.51")


def test_geocode_unknown_address_offline() -> None:
    assert backend.geocode("Bahnhofstrasse 1, Zürich") is None  # network is blocked in tests


@pytest.mark.parametrize("free, status", [(200, "good"), (50, "good"), (49, "tight"), (10, "tight"), (9, "full"), (0, "full")])
def test_availability(free: int, status: str) -> None:
    assert backend.availability(free) == status


def test_display_name() -> None:
    assert backend.display_name("Urania", "zuerichparkhausurania") == "Parkhaus Urania"
    assert backend.display_name("Theater 11", "zuerichparkplatztheater11") == "Parkplatz Theater 11"
    assert backend.display_name("Parkhaus Urania", "x") == "Parkhaus Urania"


# --- map --------------------------------------------------------------------------------


def test_map_data_colours_sizes_start_and_destination() -> None:
    ranked = _table(
        _ranked(free=200, garage_id="a", name="A"),
        _ranked(free=3, garage_id="b", name="B"),
        _ranked(free=200, rate=7.0, garage_id="c", name="C"),  # over budget
    )
    origin = Location(47.4115, 8.5441, "Bahnhof Oerlikon")
    data = map_data(ranked, DEST, recommended_id="a", max_price_chf=5.0, origin=origin)
    assert list(data.columns) == ["lat", "lon", "color", "size", "name", "garage_id"]
    garages, markers = data.iloc[:3], data.iloc[3:]
    assert list(garages["name"]) == ["Parkhaus A", "Parkhaus B", "Parkhaus C"]  # shown when pointing at a dot
    assert list(garages["garage_id"]) == ["a", "b", "c"]  # returned when a dot is clicked
    assert set(markers["name"]) == {"Startpunkt", "Ziel"} and set(markers["garage_id"]) == {""}
    assert list(garages["color"]) == [backend.STATUS_COLORS["good"], backend.STATUS_COLORS["full"], OVER_BUDGET_COLOR]
    assert garages["size"].iloc[0] > garages["size"].iloc[1]  # the recommendation is bigger
    # Start and destination come last (drawn on top), each as dots from big to small.
    expected = [(origin.lat, c, s) for c, s in START_MARKER] + [(DEST.lat, c, s) for c, s in DESTINATION_MARKER]
    assert list(zip(markers["lat"], markers["color"], markers["size"])) == expected
    assert list(markers["size"].iloc[:len(START_MARKER)]) == sorted(markers["size"].iloc[:len(START_MARKER)], reverse=True)


def test_map_data_without_destination_or_garages() -> None:
    assert len(map_data(_table(_ranked()), None, None, 8)) == 1  # no start, no destination
    assert len(map_data(_table(), None, None, 8, origin=DEST)) == len(START_MARKER)  # only the start point
    empty = map_data(_table(), None, None, 8)
    assert len(empty) == 1 and empty["color"].iloc[0] == "#00000000"  # invisible dot centres the map


def test_hex_to_rgba() -> None:
    assert hex_to_rgba("#2563eb") == [37, 99, 235, 255]
    assert hex_to_rgba("#00000000") == [0, 0, 0, 0]  # invisible


def test_fit_zoom_keeps_all_dots_on_the_map() -> None:
    city = pd.DataFrame({"lat": [47.356, 47.415], "lon": [8.510, 8.554]})  # spread of the Zurich garages
    far_start = pd.concat([city, pd.DataFrame({"lat": [47.50], "lon": [8.72]})])  # start in Winterthur
    assert 11 <= fit_zoom(city) <= 12.5
    assert fit_zoom(far_start) < fit_zoom(city) - 1  # zoomed out to show Winterthur as well
    assert fit_zoom(pd.DataFrame({"lat": [47.37], "lon": [8.54]})) == MAX_ZOOM  # one dot: not too close


def test_garage_table() -> None:
    ranked = _table(_ranked(free=200, garage_id="a", name="A"), _ranked(free=50, rate=7.0, garage_id="c", name="C"))
    table = garage_table(ranked, recommended_id="a", max_price_chf=5.0)
    assert list(table["Parkhaus"]) == ["Parkhaus A", "Parkhaus C"]
    assert list(table["Hinweis"]) == ["★ Empfehlung", "über Preislimit"]
    assert table["Verfügbarkeit"].iloc[0] == "Gut verfügbar"


# --- charts ---------------------------------------------------------------------------------


def test_mode_chart_data_handles_missing_options() -> None:
    table = mode_chart_data(compare_modes(_ranked(), NO_TRANSIT, 2, "balanced"))
    assert list(table.index) == ["Auto", "ÖV"]
    assert table.loc["Auto", "Minuten"] == 16 and table.loc["Auto", "CHF"] > 0
    assert np.isnan(table.loc["ÖV", "Minuten"]) and np.isnan(table.loc["ÖV", "CHF"])


# --- Google Maps link ---------------------------------------------------------------------


def test_google_maps_url() -> None:
    url = google_maps_url(Location(47.37, 8.54), "transit", origin=Location(47.41, 8.54))
    query = parse_qs(urlparse(url).query)
    assert url.startswith("https://www.google.com/maps/dir/?api=1&")
    assert query == {"api": ["1"], "origin": ["47.41,8.54"], "destination": ["47.37,8.54"], "travelmode": ["transit"]}
    assert "origin" not in parse_qs(urlparse(google_maps_url(DEST, "driving")).query)  # Maps uses the current position


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
    assert [s.value for s in at.main.subheader] == [
        "Die besten Parkhäuser für deine Fahrt", "Wie kommst du am besten ans Ziel?", "Auto oder ÖV?"
    ]


def test_app_price_filter_and_unknown_address(offline_app: AppTest) -> None:
    at = offline_app.run()
    at.slider[0].set_value(1.0).run()  # all dummy garages cost more
    assert any("Kein passendes Parkhaus gefunden" in m.value for m in at.markdown)
    at.text_input(key="destination_query").set_value("Atlantis").run()
    assert any("„Atlantis“ wurde nicht gefunden" in e.value for e in at.sidebar.error)
    assert not at.exception


def test_app_accepts_coordinates_as_destination(offline_app: AppTest) -> None:
    at = offline_app.run()
    at.text_input(key="destination_query").set_value("47.39, 8.515").run()
    assert not at.exception
    assert not at.sidebar.error
    assert not at.session_state["ranking"].empty


def test_app_priority_radio_changes_the_ranking(offline_app: AppTest) -> None:
    at = offline_app.run()
    assert at.session_state["priority"] == "balanced"
    assert len(at.radio[0].options) == 4
    winners = {}
    for key in config.PRIORITY_WEIGHTS:
        at.radio[0].set_value(key).run()
        assert not at.exception
        assert at.session_state["priority"] == key
        winners[key] = at.session_state["ranking"].iloc[0]
    ranking = at.session_state["ranking"]
    assert winners["cheapest"]["price_per_hour"] == ranking["price_per_hour"].min()
    assert winners["most_spots"]["free_at_arrival"] == ranking["free_at_arrival"].max()


def test_app_walk_slider_changes_the_ranking(offline_app: AppTest) -> None:
    at = offline_app.run()
    walk = next(s for s in at.slider if s.label == "Max. Fussweg zum Ziel")
    assert walk.value == config.DEFAULT_MAX_WALK_MIN
    walk.set_value(10).run()
    assert not at.exception
    assert at.session_state["ranking"]["walk_min"].iloc[0] <= 10
    next(s for s in at.slider if s.label == "Max. Fussweg zum Ziel").set_value(5).run()
    assert any("min zu Fuss vom Ziel" in w.value for w in at.warning)  # every dummy garage is further away


def test_app_cards_and_chevron(offline_app: AppTest) -> None:
    at = offline_app.run()
    ranking = at.session_state["ranking"]
    cards = [m.value for m in at.main.markdown if "pk-rank" in m.value]
    assert len(cards) == 3 and "Beste Gesamtoption" in cards[0]
    assert any("Beispielwerte" in w.value for w in at.warning)  # live data is down in tests
    at.button(key="chevron_2").click().run()
    assert not at.exception
    assert at.session_state["selected_garage"] == ranking["garage_id"].iloc[1]
    at.radio[0].set_value("cheapest").run()  # the cards follow the priority at once
    cards = [m.value for m in at.main.markdown if "pk-rank" in m.value]
    assert "Günstigste Option" in cards[0]


def test_app_without_suitable_garage_shows_a_message(offline_app: AppTest) -> None:
    at = offline_app.run()
    at.slider[0].set_value(1.0).run()  # every garage is too expensive
    assert any("keine passenden Parkhäuser" in i.value for i in at.info)
    assert not at.exception


def test_app_mode_cards_without_transit(offline_app: AppTest) -> None:
    at = offline_app.run()
    texts = [m.value for m in at.main.markdown]
    assert any("Bahnhof Oerlikon → Paradeplatz" in t for t in texts)
    assert any("Keine ÖV-Daten verfügbar" in t for t in texts)  # transit is offline in tests
    assert not any("pk-check" in t or "Deine Ersparnis" in t for t in texts)  # nothing is recommended
    car, transit = at.get("link_button")
    assert "travelmode=driving" in car.proto.url and "origin=47.4115%2C8.5441" in car.proto.url
    assert "travelmode=transit" in transit.proto.url
    assert {car.proto.type, transit.proto.type} == {"secondary"}


def test_app_mode_cards_highlight_the_winner(offline_app: AppTest, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake(origin, destination, departure):
        calls.append(departure)
        return {"duration_min": 5, "transfers": 0, "departure": departure, "arrival": departure + timedelta(minutes=5),
                "sections": []}

    monkeypatch.setattr(transit_api, "get_transit_options", fake)
    at = offline_app.run()
    assert not at.exception
    at.radio[0].set_value("fastest").run()
    texts = [m.value for m in at.main.markdown]
    assert any("Deine Ersparnis mit dem ÖV" in t for t in texts) and any("Direkt" in t for t in texts)
    car, transit = at.get("link_button")
    assert (car.proto.type, transit.proto.type) == ("secondary", "primary")
    assert len(calls) == 1  # cached
    # Parking for longer costs more.
    before = at.main.markdown
    slider = next(s for s in at.slider if s.label == "Parkdauer")
    assert slider.value == 2.0
    slider.set_value(8.0).run()
    assert [m.value for m in before] != [m.value for m in at.main.markdown]
    assert any("Parkdauer von 8 Stunden" in c.value for c in at.caption)


def test_app_shows_the_same_forecast_everywhere(offline_app: AppTest) -> None:
    at = offline_app.run()
    for key in config.PRIORITY_WEIGHTS:
        at.radio[0].set_value(key).run()
        assert not at.exception
        free = at.session_state["ranking"]["free_at_arrival"].iloc[0]
        assert at.session_state["selected_garage"] == at.session_state["ranking"]["garage_id"].iloc[0]
        numbers = [m.value for m in at.main.markdown if "Frei bei Ankunft" in m.value]
        assert f">{free}<" in numbers[0]  # first card of the list
        assert f">{free}<" in numbers[-1]  # forecast
        assert any(f"{free} freie Plätze erwartet" in m.value for m in at.main.markdown)  # car card
    # Example garages have no live measurement.
    assert any("–" in m.value and "Freie Plätze aktuell" in m.value for m in at.main.markdown)
    assert not any("Aktuell gemessen" in m.value for m in at.main.markdown)
