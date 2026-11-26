# AI citation: this file was written with Claude (Anthropic).
"""Compare driving + parking with public transport for one trip.

- Car: rank 1 of the ranking; time = drive + walk,
  costs = 2 x drive_km x costs per km + hourly rate x length of stay.
- Public transport: the next connection; time = waiting + travel time,
  costs = simplified ticket there and back.
- Decision: depends on the priority, see ``config.MODE_WEIGHTS``.

Everything except ``transit_option`` (asks the API) is pure calculation.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src import config
from src.data import routing, transit_api
from src.models import CarOption, Location, ModeComparison, TransitOption

NO_TRANSIT = TransitOption(available=False, total_min=None, wait_min=None, cost_chf=None, transfers=None)


def car_option(best: dict | None, parking_hours: float) -> CarOption:
    """Time and costs for driving to the garage ``best`` (a row of the ranking as dict)."""
    if best is None:
        return CarOption(available=False, total_min=None, cost_chf=None, garage=None)
    rate = best["price_per_hour"]
    if pd.isna(rate):
        rate = config.ASSUMED_RATE_IF_UNKNOWN_CHF_PER_H
    cost = 2 * best["drive_km"] * config.COST_PER_KM_CHF + rate * parking_hours
    return CarOption(
        available=True,
        total_min=round(best["drive_min"] + best["walk_min"]),
        cost_chf=round(cost, 2),
        garage=best,
    )


def transit_option(origin: Location, destination: Location, departure: datetime) -> TransitOption:
    """Next connection departing at or after ``departure``; ``NO_TRANSIT`` if the API fails."""
    connection = transit_api.get_transit_options(origin, destination, departure)
    if connection is None:
        return NO_TRANSIT
    wait = max(0.0, (connection["departure"] - departure).total_seconds() / 60)
    total = max(connection["duration_min"], (connection["arrival"] - departure).total_seconds() / 60)
    return TransitOption(
        available=True,
        total_min=round(total),
        wait_min=round(wait),
        cost_chf=round(2 * transit_fare(routing.haversine_km(origin, destination)), 2),
        transfers=connection["transfers"],
    )


def transit_fare(distance_km: float) -> float:
    """Simplified single ticket for a straight-line distance (placeholder, see config)."""
    for max_km, price in config.TRANSIT_FARES_CHF:
        if distance_km <= max_km:
            return price
    return config.TRANSIT_FARES_CHF[-1][1]


def decide(car: CarOption, transit: TransitOption, priority: str) -> str | None:
    """The better mode for the priority: "car", "transit" or ``None`` (tie or missing data).

    Time and costs have different units, so each is turned into a share: e.g.
    20 min (car) and 30 min (transit) become 0.4 and 0.6. Less is better.
    """
    if not car["available"] or not transit["available"]:
        return None
    weights = config.MODE_WEIGHTS[priority]
    car_share = (
        weights["time"] * _share(car["total_min"], transit["total_min"])
        + weights["cost"] * _share(car["cost_chf"], transit["cost_chf"])
    )
    # The two shares sum to 1, so 0.5 is a tie (rounded, because of floating point errors).
    if round(car_share, 6) == 0.5:
        return None
    return "car" if car_share < 0.5 else "transit"


def _share(value: float, other: float) -> float:
    """``value`` as a share of both values together (0.5 if both are 0)."""
    total = value + other
    return 0.5 if total == 0 else value / total


def compare_modes(best: dict | None, transit: TransitOption, parking_hours: float, priority: str) -> ModeComparison:
    """Compare car and public transport.

    Args:
        best: Rank 1 of the ranking (``parking_score.best_garage``), None = no garage.
        transit: Output of ``transit_option``.
        parking_hours: Length of stay in hours.
        priority: Key of ``config.MODE_WEIGHTS``.
    """
    car = car_option(best, parking_hours)
    return ModeComparison(car=car, transit=transit, recommendation=decide(car, transit, priority))


def savings(comparison: ModeComparison) -> dict[str, float] | None:
    """What the recommended mode saves compared to the other one (negative = it is worse).

    Returns:
        ``minutes``, ``chf`` and ``percent`` (time difference / longer travel
        time), or ``None`` without a recommendation.
    """
    winner = comparison["recommendation"]
    if winner is None:
        return None
    best, other = comparison[winner], comparison["transit" if winner == "car" else "car"]
    minutes = other["total_min"] - best["total_min"]
    longer = max(other["total_min"], best["total_min"])
    return {
        "minutes": minutes,
        "chf": round(other["cost_chf"] - best["cost_chf"], 2),
        "percent": round(minutes / longer * 100) if longer else 0,
    }
