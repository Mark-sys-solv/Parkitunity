# AI citation: this file was written with Claude (Anthropic).
"""Build the table of all garages that could be used for a trip."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src import config
from src.data import parking_api, routing
from src.ml.predict import predict_free
from src.models import Location

COLUMNS = [
    "garage_id", "name", "lat", "lon", "capacity", "free_now", "drive_min", "drive_km",
    "walk_min", "price_per_hour", "arrival_time", "free_at_arrival",
]


def build_candidates(start: Location | None, ziel: Location, departure: datetime | None = None) -> pd.DataFrame:
    """One row per garage with everything the ranking needs.

    Args:
        start: Where the trip starts. ``None``: unknown, driving takes 0 minutes.
        ziel: Where the driver wants to go.
        departure: When the trip starts (naive = Zurich time). ``None``: now.

    Returns:
        DataFrame with the columns ``COLUMNS``. ``free_now`` comes from the
        live API, ``free_at_arrival`` from ``predict_free``. ``price_per_hour``
        is NaN if no price is known (see ``config.USE_PLACEHOLDER_PRICES``).
    """
    tz = ZoneInfo(config.ZURICH_TZ)
    if departure is None:
        departure = datetime.now(tz)
    elif departure.tzinfo is None:
        departure = departure.replace(tzinfo=tz)

    rows = []
    for garage in parking_api.fetch_garages():
        drive_min = routing.get_drive_time(start, garage.location) if start is not None else 0.0
        drive_km = routing.get_drive_distance_km(start, garage.location) if start is not None else 0.0
        arrival_time = departure + timedelta(minutes=drive_min)  # different for every garage
        rows.append({
            "garage_id": garage.id,
            "name": garage.name,
            "lat": garage.location.lat,
            "lon": garage.location.lon,
            "capacity": garage.capacity,
            "free_now": garage.free_spots,
            "drive_min": drive_min,
            "drive_km": drive_km,
            "walk_min": routing.get_walking_time(garage.location, ziel),
            "price_per_hour": garage.hourly_rate_chf,
            "arrival_time": arrival_time,
            "free_at_arrival": predict_free(garage.id, arrival_time)[0],
        })
    table = pd.DataFrame(rows, columns=COLUMNS)
    table["price_per_hour"] = table["price_per_hour"].astype(float)  # None -> NaN
    return table
