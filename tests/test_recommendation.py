"""Tests for the comparison car / public transport (transit is mocked)."""

from datetime import datetime, timedelta

import pytest

from src import config
from src.data import transit_api
from src.models import CarOption, Location, TransitOption
from src.recommendation.mode_compare import (
    NO_TRANSIT,
    car_option,
    compare_modes,
    decide,
    savings,
    transit_fare,
    transit_option,
)
from src.ui.backend import TZ
from src.ui.mode_cards import footnote, savings_items, short_place, transfers_text

DEPARTURE = datetime(2026, 11, 24, 14, 0, tzinfo=TZ)
DEST = Location(47.3697, 8.5391, "Paradeplatz")
ORIGIN = Location(47.4115, 8.5441, "Oerlikon")
# Rank 1 of the ranking.
BEST = {"garage_id": "near", "drive_min": 15.4, "drive_km": 5.0, "walk_min": 4, "price_per_hour": 4.0, "free_at_arrival": 80}


def _car(minutes: int, chf: float) -> CarOption:
    """An invented car option."""
    return CarOption(available=True, total_min=minutes, cost_chf=chf, garage=BEST)


def _transit(minutes: int, chf: float) -> TransitOption:
    """An invented public transport option."""
    return TransitOption(available=True, total_min=minutes, wait_min=0, cost_chf=chf, transfers=1)


# --- options -----------------------------------------------------------------------


def test_car_time_and_costs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "COST_PER_KM_CHF", 0.5)
    car = car_option(BEST, parking_hours=3)
    assert car["total_min"] == 19  # 15.4 + 4
    assert car["cost_chf"] == 2 * 5.0 * 0.5 + 4.0 * 3  # km there and back + parking
    assert car["garage"] is BEST


def test_car_unknown_price_and_no_garage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "COST_PER_KM_CHF", 0)
    unknown = car_option({**BEST, "price_per_hour": float("nan")}, 2)
    assert unknown["cost_chf"] == 2 * config.ASSUMED_RATE_IF_UNKNOWN_CHF_PER_H
    assert car_option(None, 2)["available"] is False


def test_transit_time_includes_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(origin, destination, departure):
        start = departure + timedelta(minutes=10)
        return {"duration_min": 12, "transfers": 1, "departure": start, "arrival": start + timedelta(minutes=12), "sections": []}

    monkeypatch.setattr(transit_api, "get_transit_options", fake)
    option = transit_option(ORIGIN, DEST, DEPARTURE)
    assert (option["wait_min"], option["total_min"], option["transfers"]) == (10, 22, 1)
    assert option["cost_chf"] == 2 * 4.60  # ~4.6 km -> city ticket, there and back


def test_transit_api_down() -> None:
    assert transit_option(ORIGIN, DEST, DEPARTURE) == NO_TRANSIT  # network is blocked in tests


@pytest.mark.parametrize("km, price", [(1.0, 2.80), (5, 4.60), (15, 7.00), (25, 9.20), (100, 13.80)])
def test_transit_fare(km: float, price: float) -> None:
    assert transit_fare(km) == price


# --- decision ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "priority, winner",
    [("fastest", "car"), ("cheapest", "transit"), ("balanced", "transit"), ("most_spots", "transit")],
)
def test_decision_depends_on_the_priority(priority: str, winner: str) -> None:
    # The car is a little faster, but costs twice as much.
    assert decide(_car(20, 20.0), _transit(22, 10.0), priority) == winner


def test_ties_and_missing_data_have_no_winner() -> None:
    assert decide(_car(20, 5.0), _transit(20, 9.0), "fastest") is None
    assert decide(_car(20, 9.0), _transit(30, 9.0), "cheapest") is None
    assert decide(_car(20, 10.0), _transit(10, 20.0), "balanced") is None  # twice as fast, but twice as expensive
    for priority in config.PRIORITY_WEIGHTS:
        assert decide(_car(20, 5.0), NO_TRANSIT, priority) is None
        assert decide(car_option(None, 2), _transit(20, 5.0), priority) is None


def test_every_priority_has_mode_weights() -> None:
    assert set(config.MODE_WEIGHTS) == set(config.PRIORITY_WEIGHTS)


# --- savings -----------------------------------------------------------------------


def test_savings_of_the_winner() -> None:
    saved = savings({"car": _car(20, 11.35), "transit": _transit(14, 9.20), "recommendation": "transit"})
    assert saved == {"minutes": 6, "chf": 2.15, "percent": 30}  # 6 / 20
    assert savings_items(saved) == [
        ("6 Min.", "Zeitersparnis", True), ("CHF 2.15", "Kosteneinsparung", True), ("30 %", "Kürzere Reisezeit", True),
    ]


def test_negative_savings_are_shown_as_extra_costs_or_time() -> None:
    faster = savings({"car": _car(20, 12.0), "transit": _transit(30, 9.20), "recommendation": "car"})
    assert savings_items(faster)[1] == ("CHF 2.80", "Mehrkosten", False)
    cheaper = savings({"car": _car(20, 12.0), "transit": _transit(30, 9.20), "recommendation": "transit"})
    assert savings_items(cheaper)[0] == ("10 Min.", "Längere Reisezeit", False)
    assert savings_items(cheaper)[2] == ("33 %", "Längere Reisezeit", False)  # 10 / 30


def test_no_savings_without_recommendation() -> None:
    assert savings(compare_modes(BEST, NO_TRANSIT, 2, "balanced")) is None


# --- texts -------------------------------------------------------------------------


def test_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    assert [transfers_text(n) for n in (0, 1, 2)] == ["Direkt", "1 Umstieg", "2 Umstiege"]
    assert short_place("Bahnhof Oerlikon, Zürich") == "Bahnhof Oerlikon"
    assert short_place("47.39, 8.515") == "47.39, 8.515"
    assert "Parkdauer von 2.5 Stunden" in footnote(2.5) and "vorläufige Platzhalter" in footnote(2.5)
    monkeypatch.setattr(config, "USE_PLACEHOLDER_PRICES", False)
    assert "Platzhalter" not in footnote(2)
