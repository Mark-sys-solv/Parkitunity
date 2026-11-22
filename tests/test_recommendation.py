"""Tests for src/recommendation (garages, predictions and transit are mocked)."""

from datetime import datetime, timedelta

import pytest

from src import config
from src.data import parking_api, routing, transit_api
from src.models import Garage, Location
from src.recommendation import mode_compare, parking_score
from src.recommendation.mode_compare import compare_modes, transit_fare
from src.recommendation.parking_score import rank_garages, search_penalty

TZ = mode_compare._TZ
DEPARTURE = datetime(2026, 11, 24, 14, 0, tzinfo=TZ)
DEST = Location(47.3697, 8.5391, "Paradeplatz")
ORIGIN = Location(47.4115, 8.5441, "Oerlikon")
KM_LAT = 1 / 111.2  # degrees latitude per km


def _garage(gid: str, km_north: float, rate: float | None) -> Garage:
    return Garage(gid, gid.title(), Location(DEST.lat + km_north * KM_LAT, DEST.lon), 300, None, rate)


GARAGES = [
    _garage("near", 0.2, 4.0),  # close, plenty of space
    _garage("far", -1.5, 2.0),  # cheap but a long walk
    _garage("pricey", 0.1, 7.5),  # closest, expensive
    _garage("tight", 0.3, 3.0),  # almost full -> search penalty
    _garage("full", 0.05, 3.0),  # no free spots -> excluded
    _garage("noprice", 0.4, None),  # unknown price
]
FREE = {"near": 80, "far": 200, "pricey": 100, "tight": 8, "full": 0, "noprice": 60}


@pytest.fixture
def predictions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, datetime]]:
    """Mock garages and predict(); returns the recorded predict calls."""
    calls: list[tuple[str, datetime]] = []

    def fake_predict(garage_id: str, when: datetime) -> int:
        calls.append((garage_id, when))
        return FREE[garage_id]

    monkeypatch.setattr(parking_api, "fetch_garages", lambda: GARAGES)
    monkeypatch.setattr(parking_score, "predict", fake_predict)
    return calls


def _ids(ranked: list[dict]) -> list[str]:
    return [r["garage"].id for r in ranked]


# --- rank_garages ---------------------------------------------------------------


@pytest.mark.parametrize("free, minutes", [(200, 0), (50, 0), (30, 5), (10, 10), (1, 10)])
def test_search_penalty(free: int, minutes: float) -> None:
    assert search_penalty(free) == pytest.approx(minutes)


def test_full_garages_are_excluded_and_list_is_sorted(predictions) -> None:
    ranked = rank_garages(DEST, DEPARTURE, origin=ORIGIN)
    assert "full" not in _ids(ranked)
    assert len(ranked) == len(GARAGES) - 1
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_prediction_is_made_for_arrival_at_the_garage(predictions) -> None:
    rank_garages(DEST, DEPARTURE, origin=ORIGIN)
    by_id = {g.id: g for g in GARAGES}
    for garage_id, when in predictions:
        drive = routing.get_drive_time(ORIGIN, by_id[garage_id].location)
        assert when == DEPARTURE + timedelta(minutes=drive)
    assert drive > 10  # Oerlikon -> city centre takes a while


def test_without_origin_there_is_no_drive_time(predictions) -> None:
    ranked = rank_garages(DEST, DEPARTURE)
    assert all(r["drive_min"] == 0 for r in ranked)
    assert {when for _, when in predictions} == {DEPARTURE}


def test_entry_fields_and_total_time(predictions) -> None:
    near = next(r for r in rank_garages(DEST, DEPARTURE, origin=ORIGIN) if r["garage"].id == "near")
    assert near["walking_min"] == pytest.approx(0.2 / 5 * 60, abs=0.05)  # 200 m at 5 km/h
    assert near["search_min"] == 0
    assert near["total_min"] == pytest.approx(near["drive_min"] + near["search_min"] + near["walking_min"])
    assert near["parking_cost_chf"] == 4.0 * config.PARKING_DURATION_H
    assert near["arrival_at_garage"] == DEPARTURE + timedelta(minutes=near["drive_min"])
    assert set(near["score_parts"]) == {"time", "availability", "price"}


def test_ranking_balances_time_availability_and_price(predictions) -> None:
    order = _ids(rank_garages(DEST, DEPARTURE, origin=ORIGIN))
    assert order.index("near") < order.index("far")  # 18 min walk is too long
    assert order.index("near") < order.index("tight")  # 8 free spots -> +10 min search
    assert order.index("near") < order.index("pricey")  # 7.50 CHF/h outweighs 100 m


def test_max_price_drops_expensive_but_keeps_unknown_prices(predictions) -> None:
    ids = _ids(rank_garages(DEST, DEPARTURE, max_price=5.0, origin=ORIGIN))
    assert "pricey" not in ids
    assert "noprice" in ids


def test_score_does_not_depend_on_other_garages(predictions) -> None:
    def near_score(**kw) -> float:
        return next(r["score"] for r in rank_garages(DEST, DEPARTURE, origin=ORIGIN, **kw) if r["garage"].id == "near")

    assert near_score() == near_score(max_price=4.0)


def test_weights_come_from_config(predictions, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SCORE_WEIGHTS", {"time": 1, "availability": 0, "price": 0})
    ranked = rank_garages(DEST, DEPARTURE, origin=ORIGIN)
    totals = [r["total_min"] for r in ranked]
    assert totals == sorted(totals)

    monkeypatch.setattr(config, "SCORE_WEIGHTS", {"time": 0, "availability": 0, "price": 5})
    assert _ids(rank_garages(DEST, DEPARTURE, origin=ORIGIN))[0] == "far"  # cheapest


def test_all_garages_full(monkeypatch: pytest.MonkeyPatch, predictions) -> None:
    monkeypatch.setattr(parking_score, "predict", lambda gid, when: 0)
    assert rank_garages(DEST, DEPARTURE, origin=ORIGIN) == []


# --- compare_modes ------------------------------------------------------------------


@pytest.fixture
def transit(monkeypatch: pytest.MonkeyPatch):
    """Mock the transit API; call ``transit(wait=, duration=, transfers=)`` or ``transit(None)``."""
    calls: list[tuple] = []

    def configure(wait: float | None = 2, duration: float = 12, transfers: int = 1):
        def fake(origin, destination, departure_time):
            calls.append((origin, destination, departure_time))
            if wait is None:
                return None
            dep = departure_time + timedelta(minutes=wait)
            return {"duration_min": duration, "transfers": transfers, "departure": dep,
                    "arrival": dep + timedelta(minutes=duration), "sections": []}

        monkeypatch.setattr(transit_api, "get_transit_options", fake)
        return calls

    return configure


def test_transit_faster_and_cheaper(predictions, transit) -> None:
    transit(wait=2, duration=12)
    result = compare_modes(ORIGIN, DEST, DEPARTURE)
    assert result["recommendation"] == "transit"
    assert result["transit"]["total_min"] == 14
    assert result["car"]["total_min"] > 14
    assert "schneller" in result["reason"] and "günstiger" in result["reason"]
    assert result["reason"].startswith("Der ÖV")


def test_car_wins_when_transit_is_slow(predictions, transit) -> None:
    transit(wait=5, duration=55, transfers=3)
    result = compare_modes(ORIGIN, DEST, DEPARTURE)
    assert result["recommendation"] == "car"
    assert result["reason"].startswith("Das Auto ist")
    assert result["car"]["parking_garage_id"] == "near"


def test_waiting_time_counts_for_transit(predictions, transit) -> None:
    transit(wait=10, duration=12)
    t = compare_modes(ORIGIN, DEST, DEPARTURE)["transit"]
    assert (t["wait_min"], t["duration_min"], t["total_min"]) == (10, 12, 22)


def test_transit_api_down_gives_car_only_recommendation(predictions, transit) -> None:
    transit(wait=None)
    result = compare_modes(ORIGIN, DEST, DEPARTURE)
    assert result["recommendation"] == "car"
    assert result["transit"]["available"] is False
    assert result["transit"]["total_min"] is None
    assert result["reason"].startswith("Nur Auto-Empfehlung")
    assert result["car"]["available"] is True


def test_all_garages_full_recommends_transit(predictions, transit, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parking_score, "predict", lambda gid, when: 0)
    transit(wait=2, duration=40)
    result = compare_modes(ORIGIN, DEST, DEPARTURE)
    assert result["recommendation"] == "transit"
    assert result["car"]["available"] is False
    assert result["car"]["parking_garage_id"] is None
    assert "Kein passendes Parkhaus" in result["reason"]


def test_all_full_and_transit_down(predictions, transit, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parking_score, "predict", lambda gid, when: 0)
    transit(wait=None)
    result = compare_modes(ORIGIN, DEST, DEPARTURE)
    assert result["recommendation"] is None
    assert result["reason"].startswith("Keine Empfehlung möglich")


def test_costs_round_trip(predictions, transit) -> None:
    transit()
    result = compare_modes(ORIGIN, DEST, DEPARTURE)
    car, near = result["car"], next(g for g in GARAGES if g.id == "near")
    km = routing.get_drive_distance_km(ORIGIN, near.location)
    assert car["parking_cost_chf"] == 4.0 * config.PARKING_DURATION_H
    assert car["km_cost_chf"] == pytest.approx(2 * km * config.CAR_COST_PER_KM_CHF, abs=0.01)
    assert car["cost_chf"] == pytest.approx(car["parking_cost_chf"] + car["km_cost_chf"], abs=0.01)
    assert result["transit"]["cost_chf"] == 2 * 4.60  # ~4.6 km -> city ticket, there and back


def test_costs_one_way(predictions, transit, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "ROUND_TRIP", False)
    monkeypatch.setattr(config, "INCLUDE_CAR_KM_COSTS", False)
    transit()
    result = compare_modes(ORIGIN, DEST, DEPARTURE)
    assert result["transit"]["cost_chf"] == 4.60
    assert result["car"]["km_cost_chf"] == 0
    assert result["car"]["cost_chf"] == result["car"]["parking_cost_chf"]


@pytest.mark.parametrize("km, price", [(1.0, 2.80), (5, 4.60), (15, 7.00), (25, 9.20), (100, 13.80)])
def test_transit_fare(km: float, price: float) -> None:
    assert transit_fare(km) == price


def test_max_price_is_respected_for_the_car(predictions, transit) -> None:
    transit(wait=5, duration=55)
    result = compare_modes(ORIGIN, DEST, DEPARTURE, max_price=3.0)
    assert result["car"]["parking_garage_id"] in {"far", "tight", "noprice"}
    assert result["car"]["parking_garage_id"] != "noprice" or result["car"]["parking_cost_chf"] == (
        config.ASSUMED_RATE_IF_UNKNOWN_CHF_PER_H * config.PARKING_DURATION_H
    )


def test_near_tie_goes_to_transit(predictions, transit, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "DECISION_TIE_CHF", 1_000)
    transit(wait=5, duration=55)
    result = compare_modes(ORIGIN, DEST, DEPARTURE)
    assert result["recommendation"] == "transit"
    assert "gleich gut" in result["reason"]


def test_place_formats(predictions, transit, monkeypatch: pytest.MonkeyPatch) -> None:
    real_geocode = routing.geocode  # still parses coordinate strings without network
    monkeypatch.setattr(routing, "geocode", lambda text: ORIGIN if text == "Zürich Oerlikon" else real_geocode(text))
    calls = transit()
    a = compare_modes("47.4115,8.5441", DEST, DEPARTURE)
    b = compare_modes((47.4115, 8.5441), DEST, DEPARTURE)
    assert a["car"] == b["car"]
    compare_modes("Zürich Oerlikon", DEST, DEPARTURE)
    assert calls[-1][0] == "Zürich Oerlikon"  # station names go to the transit API as typed
    assert isinstance(calls[0][0], Location)  # coordinates are passed as a Location


def test_naive_departure_is_zurich_time(predictions, transit) -> None:
    calls = transit()
    compare_modes(ORIGIN, DEST, DEPARTURE.replace(tzinfo=None))
    assert calls[0][2] == DEPARTURE


@pytest.mark.parametrize(
    "minutes, chf, expected",
    [
        (8, 3.4, "Der ÖV ist 8 min schneller und CHF 3.40 günstiger."),
        (8, 0.2, "Der ÖV ist 8 min schneller bei etwa gleichen Kosten."),
        (8, -2.0, "Der ÖV ist 8 min schneller – die CHF 2.00 Mehrkosten lohnen sich bei einem Zeitwert von 25 CHF/h."),
        (-3, 6.0, "Der ÖV ist CHF 6.00 günstiger und nur 3 min langsamer."),
        (0, 6.0, "Der ÖV ist CHF 6.00 günstiger."),
    ],
)
def test_reason_texts(minutes: float, chf: float, expected: str) -> None:
    assert mode_compare._reason("Der ÖV", minutes, chf) == expected
