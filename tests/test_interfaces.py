"""Contract tests: the module interfaces must keep these shapes."""

from datetime import datetime

import pytest

from src.ml.predict import predict
from src.models import CarOption, Location, TransitOption
from src.recommendation.mode_compare import compare_modes
from src.recommendation.parking_score import rank_garages


def test_predict_returns_non_negative_int() -> None:
    result = predict("urania", datetime(2026, 11, 20, 14, 0))
    assert isinstance(result, int)
    assert result >= 0


def test_compare_modes_shape() -> None:
    result = compare_modes("47.4115,8.5441", "47.3697,8.5391", datetime(2026, 11, 20, 14, 0))
    assert set(result) == {"car", "transit", "recommendation", "reason"}
    assert set(result["car"]) == set(CarOption.__annotations__)
    assert set(result["transit"]) == set(TransitOption.__annotations__)
    assert result["recommendation"] in ("car", "transit", None)
    assert isinstance(result["reason"], str) and result["reason"]


def test_rank_garages_sorted_by_score() -> None:
    ranked = rank_garages(Location(47.3697, 8.5391), datetime(2026, 11, 20, 14, 0))
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_compare_modes_rejects_unknown_address() -> None:
    with pytest.raises(ValueError, match="Address not found"):
        compare_modes("Nirgendwo 123", "47.3697,8.5391", datetime(2026, 11, 20, 14, 0))  # offline -> not found
