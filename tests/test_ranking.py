"""Tests for build_candidates and rank_garages with invented data."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import parking_api, routing
from src.models import Garage, Location
from src.recommendation import candidates as candidates_module
from src.recommendation.candidates import COLUMNS, build_candidates
from src.recommendation.parking_score import best_garage, normalise, rank_garages


def _candidates(*rows: tuple) -> pd.DataFrame:
    """Invented table: (garage_id, drive_min, walk_min, price_per_hour, free_at_arrival)."""
    return pd.DataFrame(rows, columns=["garage_id", "drive_min", "walk_min", "price_per_hour", "free_at_arrival"])


CANDIDATES = _candidates(
    ("quick", 5, 2, 6.0, 40),
    ("cheap", 20, 10, 1.0, 40),
    ("roomy", 18, 12, 5.0, 400),
    ("allround", 9, 4, 2.5, 100),
)


@pytest.mark.parametrize(
    "priority, winner",
    [("fastest", "quick"), ("cheapest", "cheap"), ("most_spots", "roomy"), ("balanced", "allround")],
)
def test_every_priority_has_the_expected_winner(priority: str, winner: str) -> None:
    ranking = rank_garages(CANDIDATES, priority)
    assert ranking["garage_id"].iloc[0] == winner
    assert list(ranking["rank"]) == [1, 2, 3, 4]
    assert ranking["score"].is_monotonic_decreasing and ranking["score"].between(0, 1).all()


def test_default_priority_is_balanced_and_input_is_not_changed() -> None:
    before = CANDIDATES.copy()
    pd.testing.assert_frame_equal(rank_garages(CANDIDATES), rank_garages(CANDIDATES, "balanced"))
    pd.testing.assert_frame_equal(CANDIDATES, before)


def test_weights_sum_to_one() -> None:
    for weights in config.PRIORITY_WEIGHTS.values():
        assert sum(weights.values()) == pytest.approx(1)


def test_garages_with_few_spots_come_last() -> None:
    limit = config.MIN_FREE_AT_ARRIVAL
    table = _candidates(("perfect but full", 1, 1, 1.0, limit - 1), ("slow", 30, 20, 8.0, limit), ("ok", 10, 5, 4.0, 50))
    for priority in config.PRIORITY_WEIGHTS:
        assert rank_garages(table, priority)["garage_id"].iloc[-1] == "perfect but full"


def test_too_far_garages_come_after_the_others() -> None:
    table = _candidates(("far", 1, 21, 1.0, 300), ("near", 30, 20, 8.0, 20), ("near but full", 2, 1, 1.0, 3))
    for priority in config.PRIORITY_WEIGHTS:
        ranking = rank_garages(table, priority, max_walk_min=20)
        assert list(ranking["garage_id"]) == ["near", "far", "near but full"]
    assert rank_garages(table, "cheapest", max_walk_min=25)["garage_id"].iloc[0] == "far"  # the user's choice


def test_unsuitable_garages_do_not_change_the_scores_of_the_others() -> None:
    extra = _candidates(("far away", 1, 90, 0.5, 900), ("full", 1, 1, 0.5, 0))
    for priority in config.PRIORITY_WEIGHTS:
        alone = rank_garages(CANDIDATES, priority)
        together = rank_garages(pd.concat([CANDIDATES, extra], ignore_index=True), priority)
        pd.testing.assert_frame_equal(alone, together.head(len(alone)))


def test_nothing_suitable_still_gives_a_ranking() -> None:
    table = _candidates(("farther", 10, 60, 3.0, 50), ("far", 10, 30, 3.0, 50))
    assert list(rank_garages(table, "fastest")["garage_id"]) == ["far", "farther"]


def test_more_than_enough_spots_make_no_difference() -> None:
    table = _candidates(("huge", 10, 10, 3.0, 999), ("enough", 10, 10, 3.0, 200), ("small", 10, 10, 3.0, 100))
    score = rank_garages(table, "most_spots", enough_spots=200).set_index("garage_id")["score"]
    assert score["huge"] == score["enough"] > score["small"]
    score = rank_garages(table, "most_spots", enough_spots=500).set_index("garage_id")["score"]
    assert score["huge"] > score["enough"]


def test_all_values_equal_does_not_divide_by_zero() -> None:
    ranking = rank_garages(_candidates(("a", 10, 5, 3.0, 50), ("b", 10, 5, 3.0, 50)), "fastest")
    assert ranking["score"].notna().all()
    assert list(normalise(pd.Series([3, 3]), True, pd.Series([3, 3]))) == [1, 1]


def test_one_garage_empty_table_and_unknown_price() -> None:
    assert rank_garages(_candidates(("a", 10, 5, np.nan, 50)))["rank"].iloc[0] == 1
    empty = rank_garages(pd.DataFrame(columns=COLUMNS))
    assert empty.empty and {"score", "rank"} <= set(empty.columns)
    assert best_garage(empty) is None


def test_best_garage_is_rank_1() -> None:
    assert best_garage(rank_garages(CANDIDATES, "fastest"))["garage_id"] == "quick"


# --- build_candidates ---------------------------------------------------------------

ZIEL = Location(47.3697, 8.5391)
START = Location(47.4115, 8.5441)
GARAGES = [
    Garage("near", "Near", Location(47.3715, 8.5391), 300, 111, 4.0),
    Garage("noprice", "No price", Location(47.38, 8.53), 200, 22, None),
]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Mock garages and predict_free(); returns the time predict_free was asked for."""
    recorded = {}

    def fake_predict_free(garage_id, when):
        recorded[garage_id] = when
        return np.array([77])

    monkeypatch.setattr(parking_api, "fetch_garages", lambda: GARAGES)
    monkeypatch.setattr(candidates_module, "predict_free", fake_predict_free)
    return recorded


def test_candidates_columns_and_values(calls: dict) -> None:
    departure = datetime(2026, 11, 24, 14, 0, tzinfo=timezone.utc)
    table = build_candidates(START, ZIEL, departure)
    assert list(table.columns) == COLUMNS
    near = table.iloc[0]
    assert (near["garage_id"], near["capacity"], near["free_now"], near["free_at_arrival"]) == ("near", 300, 111, 77)
    assert near["drive_min"] == routing.get_drive_time(START, GARAGES[0].location) > 0
    assert near["drive_km"] == routing.get_drive_distance_km(START, GARAGES[0].location)
    assert near["arrival_time"] == departure + timedelta(minutes=near["drive_min"])
    assert calls["near"] == near["arrival_time"]  # forecast for the arrival at THIS garage
    # About 200 m: 0.2 km x 1.3 / 4.5 km/h = 3.5 min -> rounded up.
    assert near["walk_min"] == 4
    assert np.isnan(table["price_per_hour"].iloc[1])


def test_without_start_there_is_no_drive_and_default_departure_is_now(calls: dict) -> None:
    table = build_candidates(None, ZIEL)
    assert (table["drive_min"] == 0).all() and (table["drive_km"] == 0).all()
    assert abs(calls["near"] - datetime.now(timezone.utc)) < timedelta(seconds=30)
