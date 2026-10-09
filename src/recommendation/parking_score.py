"""Rank parking garages for a given destination and arrival time."""

from __future__ import annotations

from datetime import datetime

from src.config import SCORE_WEIGHTS
from src.data import parking_api, routing
from src.ml.predict import predict
from src.models import Garage, Location


def score_garage(
    predicted_free: int,
    capacity: int,
    walking_min: float,
    hourly_rate_chf: float | None,
) -> float:
    """Combine availability, walking time and price into a score in [0, 1].

    Higher is better. Weights come from ``config.SCORE_WEIGHTS``.
    """
    # TODO: tune normalisation (e.g. 15 min walk = 0, 5 CHF/h = 0) and weights.
    free_score = min(predicted_free / max(capacity * 0.2, 1), 1.0)
    walk_score = max(0.0, 1 - walking_min / 15)
    price_score = max(0.0, 1 - (hourly_rate_chf or 5.0) / 6)
    return (
        SCORE_WEIGHTS["free_spots"] * free_score
        + SCORE_WEIGHTS["walking_distance"] * walk_score
        + SCORE_WEIGHTS["price"] * price_score
    )


def rank_garages(destination: Location, arrival_time: datetime, top_n: int = 5) -> list[dict]:
    """Return the best garages for a destination, best first.

    Returns:
        List of dicts with keys: garage (Garage), predicted_free (int),
        walking_min (float), score (float).
    """
    # TODO: filter out garages beyond a max walking distance.
    results: list[dict] = []
    for garage in parking_api.fetch_garages():
        free = predict(garage.id, arrival_time)
        walk = routing.get_walking_time(garage.location, destination)
        results.append(
            {
                "garage": garage,
                "predicted_free": free,
                "walking_min": walk,
                "score": score_garage(free, garage.capacity, walk, garage.hourly_rate_chf),
            }
        )
    return sorted(results, key=lambda r: r["score"], reverse=True)[:top_n]


def best_garage(destination: Location, arrival_time: datetime) -> Garage | None:
    """Convenience wrapper returning only the top-ranked garage."""
    ranked = rank_garages(destination, arrival_time, top_n=1)
    return ranked[0]["garage"] if ranked else None
