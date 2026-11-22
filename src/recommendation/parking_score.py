"""Rank parking garages for a trip.

For every garage we estimate the whole trip as seen by the driver:

    depart ──drive──▶ garage ──search──▶ park ──walk──▶ destination

- drive:  ``routing.get_drive_time(origin, garage)`` (0 if no origin is given)
- free:   ``predict(garage, departure + drive)`` – free spots when you get there
- search: extra minutes when few spots are free (``search_penalty``)
- walk:   straight line from the garage to the destination (``routing.get_walking_time``)

Full garages (0 predicted free spots) and garages above ``max_price`` are
dropped; the rest is sorted by a score combining total time, availability and
price. All assumptions live in ``src/config.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from src import config
from src.data import parking_api, routing
from src.ml.predict import predict
from src.models import Garage, Location


def search_penalty(predicted_free: int) -> float:
    """Extra minutes needed to find a spot.

    0 with more than ``SEARCH_PENALTY_HIGH_FREE`` free spots, the maximum with
    ``SEARCH_PENALTY_LOW_FREE`` or fewer, linear in between.
    """
    low, high = config.SEARCH_PENALTY_LOW_FREE, config.SEARCH_PENALTY_HIGH_FREE
    if predicted_free >= high:
        return 0.0
    if predicted_free <= low:
        return config.SEARCH_PENALTY_MAX_MIN
    return config.SEARCH_PENALTY_MAX_MIN * (high - predicted_free) / (high - low)


def score_parts(total_min: float, predicted_free: int, hourly_rate_chf: float | None) -> dict[str, float]:
    """The three sub-scores in [0, 1] (higher is better).

    Absolute scales (see config) keep a garage's score independent of the
    other garages in the list.
    """
    rate = config.ASSUMED_RATE_IF_UNKNOWN_CHF_PER_H if hourly_rate_chf is None else hourly_rate_chf
    return {
        "time": max(0.0, 1 - total_min / config.SCORE_TIME_WORST_MIN),
        "availability": min(predicted_free / config.SCORE_FREE_SATURATION, 1.0),
        "price": max(0.0, 1 - rate / config.SCORE_PRICE_WORST_CHF_PER_H),
    }


def score_garage(total_min: float, predicted_free: int, hourly_rate_chf: float | None) -> float:
    """Weighted sum of ``score_parts`` with ``config.SCORE_WEIGHTS`` (normalised to sum 1)."""
    parts = score_parts(total_min, predicted_free, hourly_rate_chf)
    return _weighted(parts)


def rank_garages(
    destination: Location,
    departure_time: datetime,
    max_price: float | None = None,
    origin: Location | None = None,
) -> list[dict]:
    """Return the usable garages for a trip, best first.

    Args:
        destination: Where the driver wants to go.
        departure_time: When the driver sets off (naive = Zurich time).
        max_price: Drop garages whose hourly rate is above this (CHF/h).
            Garages without a known price are kept.
        origin: Start of the trip. Without it, drive times are 0 and the
            prediction is made for ``departure_time``.

    Returns:
        One dict per garage, sorted by ``score`` (descending): ``garage``
        (Garage), ``predicted_free`` (int, > 0), ``drive_min``, ``search_min``,
        ``walking_min``, ``total_min`` (floats, minutes), ``arrival_at_garage``
        (datetime), ``parking_cost_chf`` (fee for ``PARKING_DURATION_H``, or
        None if the price is unknown), ``score`` and ``score_parts``.
        Empty list if every garage is full or too expensive.
    """
    candidates: list[dict] = []
    for garage in parking_api.fetch_garages():
        rate = garage.hourly_rate_chf
        if max_price is not None and rate is not None and rate > max_price:
            continue
        drive = routing.get_drive_time(origin, garage.location) if origin is not None else 0.0
        at_garage = departure_time + timedelta(minutes=drive)
        free = predict(garage.id, at_garage)
        if free <= 0:
            continue
        walk = routing.get_walking_time(garage.location, destination)
        search = search_penalty(free)
        total = drive + search + walk
        parts = score_parts(total, free, rate)
        candidates.append(
            {
                "garage": garage,
                "predicted_free": free,
                "drive_min": drive,
                "search_min": search,
                "walking_min": walk,
                "total_min": total,
                "arrival_at_garage": at_garage,
                "parking_cost_chf": rate * config.PARKING_DURATION_H if rate is not None else None,
                "score": _weighted(parts),
                "score_parts": parts,
            }
        )
    # Ties: the faster garage first.
    return sorted(candidates, key=lambda c: (-c["score"], c["total_min"]))


def best_garage(
    destination: Location,
    departure_time: datetime,
    max_price: float | None = None,
    origin: Location | None = None,
) -> Garage | None:
    """Convenience wrapper returning only the top-ranked garage."""
    ranked = rank_garages(destination, departure_time, max_price, origin)
    return ranked[0]["garage"] if ranked else None


def _weighted(parts: dict[str, float]) -> float:
    weights = config.SCORE_WEIGHTS
    total_weight = sum(weights.values())
    return sum(weights[name] * parts[name] for name in weights) / total_weight
