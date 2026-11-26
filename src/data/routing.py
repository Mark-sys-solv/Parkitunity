"""Geocoding, driving times and walking times.

Driving times come from a pluggable ``RoutingProvider``. The default
``HaversineProvider`` uses straight-line distance × road factor at an average
city speed. To use a real routing API later:

1. Write a class implementing ``drive_distance_km`` and ``drive_time_min``.
2. Register it in ``_PROVIDERS`` below.
3. Set ``PARKITUNITY_ROUTING_PROVIDER=<name>`` in ``.env``.

If the configured provider fails, we log the error and fall back to the
haversine estimate, so callers always get a number.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Protocol

import numpy as np

from src.config import (
    AVG_CITY_SPEED_KMH,
    ROAD_FACTOR,
    ROUTING_PROVIDER,
    WALKING_DETOUR_FACTOR,
    WALKING_SPEED_KMH,
)
from src.data.geocoding import geocode as find_coordinates
from src.models import Location

logger = logging.getLogger(__name__)

Coordinates = tuple[float, float]  # (lat, lon)
Point = Location | Coordinates

EARTH_RADIUS_KM = 6371.0


class RoutingProvider(Protocol):
    """Interface every routing backend must implement."""

    def drive_distance_km(self, origin: Coordinates, destination: Coordinates) -> float:
        """Length of the drive in km."""

    def drive_time_min(self, origin: Coordinates, destination: Coordinates) -> float:
        """Duration of the drive in minutes."""


class HaversineProvider:
    """Estimate driving from straight-line distance (no API, no traffic)."""

    def __init__(self, road_factor: float = ROAD_FACTOR, speed_kmh: float = AVG_CITY_SPEED_KMH) -> None:
        """Remember the road factor and the average speed."""
        self.road_factor = road_factor
        self.speed_kmh = speed_kmh

    def drive_distance_km(self, origin: Coordinates, destination: Coordinates) -> float:
        """Straight line x road factor."""
        return haversine_km(origin, destination) * self.road_factor

    def drive_time_min(self, origin: Coordinates, destination: Coordinates) -> float:
        """Distance divided by the average speed."""
        return self.drive_distance_km(origin, destination) / self.speed_kmh * 60


# Name -> factory. Add real routing APIs here (e.g. "openrouteservice").
_PROVIDERS: dict[str, Callable[[], RoutingProvider]] = {
    "haversine": HaversineProvider,
}
_FALLBACK = HaversineProvider()
_provider: RoutingProvider | None = None


def get_provider() -> RoutingProvider:
    """Return the configured routing provider (created on first use)."""
    global _provider
    if _provider is None:
        factory = _PROVIDERS.get(ROUTING_PROVIDER)
        if factory is None:
            logger.warning("Unknown routing provider %r, using haversine", ROUTING_PROVIDER)
            factory = HaversineProvider
        _provider = factory()
    return _provider


def set_provider(provider: RoutingProvider | None) -> None:
    """Override the routing provider (``None`` resets to the configured one)."""
    global _provider
    _provider = provider


def get_drive_time(origin: Point, destination: Point) -> float:
    """Estimated driving time in minutes.

    Args:
        origin: ``Location`` or (lat, lon).
        destination: ``Location`` or (lat, lon).
    """
    a, b = _to_coords(origin), _to_coords(destination)
    try:
        return get_provider().drive_time_min(a, b)
    except Exception:  # any provider failure -> estimate instead of crashing
        logger.exception("Routing provider failed, falling back to haversine estimate")
        return _FALLBACK.drive_time_min(a, b)


def get_drive_distance_km(origin: Point, destination: Point) -> float:
    """Estimated road distance in km (same provider/fallback as drive time)."""
    a, b = _to_coords(origin), _to_coords(destination)
    try:
        return get_provider().drive_distance_km(a, b)
    except Exception:
        logger.exception("Routing provider failed, falling back to haversine estimate")
        return _FALLBACK.drive_distance_km(a, b)


def get_walking_time(origin: Point, destination: Point) -> int:
    """Estimated walking time in whole minutes, rounded up (e.g. garage -> destination).

    Straight-line distance × ``WALKING_DETOUR_FACTOR`` at ``WALKING_SPEED_KMH`` (config).
    """
    minutes = haversine_km(origin, destination) * WALKING_DETOUR_FACTOR / WALKING_SPEED_KMH * 60
    return math.ceil(minutes)


def haversine_km(a: Point, b: Point) -> float:
    """Straight-line (great-circle) distance between two points in km."""
    (lat1, lon1), (lat2, lon2) = _to_coords(a), _to_coords(b)
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    # Haversine formula: distance on a sphere.
    h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h)))


def geocode(address: str) -> Location | None:
    """Convert an address or place name into a ``Location``.

    Uses OpenStreetMap Nominatim (see ``src/data/geocoding.py``); coordinate
    strings like ``"47.3769,8.5417"`` are parsed directly.

    Args:
        address: Free-text address, e.g. "Bahnhofstrasse 1, Zürich".

    Returns:
        The location, or ``None`` if the address was not found.
    """
    coords = find_coordinates(address)
    return Location(*coords, name=address) if coords is not None else None


def _to_coords(point: Point) -> Coordinates:
    """(lat, lon) of a Location or tuple."""
    if isinstance(point, Location):
        return (point.lat, point.lon)
    lat, lon = point
    return (float(lat), float(lon))
