"""Contract tests: the module interfaces must keep these shapes."""

from datetime import datetime

import numpy as np

from src.ml.predict import predict_free
from src.models import CarOption, Location, TransitOption
from src.recommendation.candidates import COLUMNS, build_candidates
from src.recommendation.mode_compare import NO_TRANSIT, compare_modes
from src.recommendation.parking_score import best_garage, rank_garages

TIME = datetime(2026, 11, 20, 14, 0)
DEST = Location(47.3697, 8.5391)


def test_predict_free_returns_an_array() -> None:
    result = predict_free("urania", TIME)
    assert isinstance(result, np.ndarray) and len(result) == 1
    assert result[0] >= 0
    assert len(predict_free("urania", [TIME, TIME])) == 2


def test_build_candidates_and_rank_garages_shape() -> None:
    candidates = build_candidates(Location(47.4115, 8.5441), DEST, TIME)
    assert list(candidates.columns) == COLUMNS
    ranking = rank_garages(candidates, "balanced")
    assert list(ranking.columns) == [*COLUMNS, "score", "rank"]
    assert list(ranking["rank"]) == list(range(1, len(candidates) + 1))


def test_compare_modes_shape() -> None:
    best = best_garage(rank_garages(build_candidates(None, DEST, TIME)))
    result = compare_modes(best, NO_TRANSIT, 2.0, "balanced")
    assert set(result) == {"car", "transit", "recommendation"}
    assert set(result["car"]) == set(CarOption.__annotations__)
    assert set(result["transit"]) == set(TransitOption.__annotations__)
    assert result["recommendation"] in ("car", "transit", None)
