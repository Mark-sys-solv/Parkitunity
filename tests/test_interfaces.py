"""Contract tests: the module interfaces must keep these shapes."""

from datetime import datetime

from src.ml.predict import predict
from src.recommendation.mode_compare import compare_modes
from src.recommendation.parking_score import rank_garages
from src.data import routing


def test_predict_returns_non_negative_int() -> None:
    result = predict("urania", datetime(2026, 11, 20, 14, 0))
    assert isinstance(result, int)
    assert result >= 0


def test_compare_modes_shape() -> None:
    result = compare_modes("Zürich Oerlikon", "Zürich Paradeplatz", datetime(2026, 11, 20, 14, 0))
    assert set(result) == {"car", "transit", "recommendation"}
    assert set(result["car"]) == {"duration_min", "cost_chf", "parking_garage_id"}
    assert set(result["transit"]) == {"duration_min", "cost_chf", "transfers"}
    assert result["recommendation"] in ("car", "transit")


def test_rank_garages_sorted_by_score() -> None:
    ranked = rank_garages(routing.geocode("Paradeplatz"), datetime(2026, 11, 20, 14, 0))
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
