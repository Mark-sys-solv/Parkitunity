"""Compare driving + parking against public transport."""

from __future__ import annotations

from datetime import datetime

from src.config import CAR_COST_PER_KM_CHF, DEFAULT_TRANSIT_TICKET_CHF
from src.data import routing, transit_api
from src.models import ModeComparison
from src.recommendation.parking_score import rank_garages

DEFAULT_PARKING_HOURS: float = 2.0


def compare_modes(origin: str, destination: str, time: datetime) -> ModeComparison:
    """Compare car (incl. parking) and public transport for a trip.

    Args:
        origin: Start address or station.
        destination: Destination address or station.
        time: Planned departure time.

    Returns:
        ``{"car": {"duration_min", "cost_chf", "parking_garage_id"},
           "transit": {"duration_min", "cost_chf", "transfers"},
           "recommendation": "car" | "transit"}``
    """
    origin_loc = routing.geocode(origin)
    dest_loc = routing.geocode(destination)

    # --- Car: drive to best garage + walk to destination -------------------
    # TODO: drive time should go to the garage, not the destination.
    # TODO: make parking duration a user input.
    ranked = rank_garages(dest_loc, time, top_n=1)
    best = ranked[0] if ranked else None
    drive_min = routing.get_drive_time(origin_loc, dest_loc)
    distance_km = routing.get_drive_distance_km(origin_loc, dest_loc)
    walk_min = best["walking_min"] if best else 0.0
    rate = (best["garage"].hourly_rate_chf or 0.0) if best else 0.0
    car_duration = drive_min + walk_min + 5  # +5 min searching / entering garage
    car_cost = distance_km * CAR_COST_PER_KM_CHF + rate * DEFAULT_PARKING_HOURS

    # --- Transit -----------------------------------------------------------
    # TODO: compute ticket price from ZVV zones.
    transit = transit_api.get_transit_options(origin, destination, time)
    # No connection / API down -> infinite duration, so the car is recommended.
    transit_duration = transit["duration_min"] if transit else float("inf")

    # TODO: smarter decision rule (weight time vs. cost, weather, user prefs).
    recommendation = "transit" if transit_duration <= car_duration * 1.2 else "car"

    return {
        "car": {
            "duration_min": round(car_duration, 1),
            "cost_chf": round(car_cost, 2),
            "parking_garage_id": best["garage"].id if best else None,
        },
        "transit": {
            "duration_min": round(transit_duration, 1),
            "cost_chf": DEFAULT_TRANSIT_TICKET_CHF,
            "transfers": transit["transfers"] if transit else 0,
        },
        "recommendation": recommendation,
    }
