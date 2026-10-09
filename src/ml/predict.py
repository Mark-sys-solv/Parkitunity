"""Occupancy prediction: the interface used by the recommendation layer."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import MODEL_PATH


def load_model(path: Path = MODEL_PATH) -> Any | None:
    """Load the trained model, or return ``None`` if none exists yet."""
    # TODO: joblib.load(path) with caching (functools.lru_cache).
    return None


def predict(garage_id: str, arrival_time: datetime) -> int:
    """Predict the number of free spots in a garage at a given time.

    Args:
        garage_id: Stable garage id (same as ``Garage.id``).
        arrival_time: Expected arrival time at the garage.

    Returns:
        Predicted number of free spots (>= 0, <= capacity).
    """
    # TODO: model = load_model(); X = features.features_for(garage_id, arrival_time)
    # TODO: clip prediction to [0, capacity] and round to int.
    # Dummy: current free spots, reduced during rush hours.
    from src.data.parking_api import get_garage

    garage = get_garage(garage_id)
    if garage is None or garage.free_spots is None:
        return 0
    rush_hour = arrival_time.hour in (7, 8, 17, 18)
    return max(0, int(garage.free_spots * (0.5 if rush_hour else 1.0)))
