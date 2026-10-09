"""Geocoding, driving times and walking distances."""

from __future__ import annotations

import math
from datetime import datetime

from src.config import ZURICH_CENTER
from src.models import Location

ROAD_FACTOR: float = 1.4  # straight-line distance -> approx. road distance
AVG_CITY_SPEED_KMH: float = 25.0
WALKING_SPEED_KMH: float = 4.8


def geocode(address: str) -> Location:
    """Convert an address or place name into coordinates.

    Args:
        address: Free-text address, e.g. "Bahnhofstrasse 1, Zürich".
    """
    # TODO: use a geocoding service (e.g. Nominatim or transport.opendata.ch /locations).
    lat, lon = ZURICH_CENTER
    return Location(lat, lon, address)


def haversine_km(a: Location, b: Location) -> float:
    """Straight-line distance between two points in kilometres."""
    r = 6371.0
    dlat = math.radians(b.lat - a.lat)
    dlon = math.radians(b.lon - a.lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(a.lat)) * math.cos(math.radians(b.lat)) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def get_drive_distance_km(origin: Location, destination: Location) -> float:
    """Estimated road distance in km."""
    # TODO: replace with a routing API (e.g. OpenRouteService, key in .env).
    return haversine_km(origin, destination) * ROAD_FACTOR


def get_drive_time(origin: Location, destination: Location, time: datetime) -> float:
    """Estimated driving time in minutes.

    Args:
        time: Departure time (later: account for rush hour).
    """
    # TODO: use routing API with traffic; add rush-hour penalty.
    return get_drive_distance_km(origin, destination) / AVG_CITY_SPEED_KMH * 60


def get_walking_time(origin: Location, destination: Location) -> float:
    """Estimated walking time in minutes (e.g. garage -> destination)."""
    # TODO: use routing API footpaths instead of straight line.
    return haversine_km(origin, destination) * 1.2 / WALKING_SPEED_KMH * 60
