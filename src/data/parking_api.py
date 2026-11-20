"""Live occupancy of Zurich parking garages via the ParkenDD API.

ParkenDD (https://api.parkendd.de) mirrors the official feed of
PLS Parkleitsystem Zürich AG (CC-0 licence).

Two entry points:
- ``fetch_occupancy()``: raw records for the collector/database (always fresh).
- ``fetch_garages()``:   ``Garage`` objects for the app (cached, dummy fallback).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src import config
from src.data.http_utils import get_json, ttl_cache
from src.models import Garage, Location

logger = logging.getLogger(__name__)

# ParkenDD lot states that carry no usable occupancy value.
_STATES_WITHOUT_DATA = {"nodata", "unknown"}

# Dummy garages so the app still works when the API is down.
# Coordinates are approximate; capacities/rates are placeholders.
_DUMMY_GARAGES: list[Garage] = [
    Garage("urania", "Parkhaus Urania", Location(47.3744, 8.5396, "Urania"), 600, 120, 4.0),
    Garage("jelmoli", "Parkhaus Jelmoli", Location(47.3737, 8.5373, "Jelmoli"), 220, 15, 5.0),
    Garage("hb", "Parkhaus Hauptbahnhof", Location(47.3797, 8.5388, "Hauptbahnhof"), 180, 40, 4.5),
    Garage("opera", "Parkhaus Opéra", Location(47.3650, 8.5470, "Opernhaus"), 300, 90, 4.0),
    Garage("hohe-promenade", "Parkhaus Hohe Promenade", Location(47.3685, 8.5480, "Hohe Promenade"), 550, 210, 3.5),
]


def fetch_occupancy(retries: int = 2) -> list[dict[str, Any]]:
    """Fetch the current occupancy of all Zurich garages.

    Args:
        retries: Extra attempts if the API fails (it is occasionally flaky).

    Returns:
        One dict per garage with keys ``garage_id`` (str), ``name`` (str),
        ``free_spots`` (int), ``capacity`` (int | None), ``timestamp``
        (timezone-aware UTC datetime of the last source update), ``lat`` and
        ``lon`` (float | None). Empty list if the API is unavailable.
    """
    data = get_json(config.PARKENDD_URL, timeout=config.PARKING_TIMEOUT_S, retries=retries)
    if not isinstance(data, dict) or not isinstance(data.get("lots"), list):
        logger.error("No usable parking data from %s", config.PARKENDD_URL)
        return []
    return parse_parkendd(data)


def parse_parkendd(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a ParkenDD city response into occupancy records.

    Cleans up known quirks of the feed: lots without data are skipped,
    a capacity of 0 is treated as unknown, and ``free_spots`` is clipped to
    ``[0, capacity]`` (the feed sometimes reports more free spots than exist).
    """
    timestamp = _parse_timestamp(data.get("last_updated"))
    records: list[dict[str, Any]] = []
    for lot in data.get("lots", []):
        try:
            if lot.get("state") in _STATES_WITHOUT_DATA or lot.get("free") is None:
                continue
            capacity = int(lot["total"]) if lot.get("total") else None
            free = max(0, int(lot["free"]))
            if lot.get("state") == "closed":
                free = 0
            if capacity is not None:
                free = min(free, capacity)
            coords = lot.get("coords") or {}
            records.append(
                {
                    "garage_id": str(lot["id"]),
                    "name": lot.get("name") or str(lot["id"]),
                    "free_spots": free,
                    "capacity": capacity,
                    "timestamp": timestamp,
                    "lat": coords.get("lat"),
                    "lon": coords.get("lng"),
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed lot %r: %s", lot.get("id"), exc)
    return records


def _parse_timestamp(value: Any) -> datetime:
    """Parse ParkenDD's ``last_updated`` (naive ISO string in UTC)."""
    try:
        ts = datetime.fromisoformat(str(value))
        return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
    except ValueError:
        logger.warning("Invalid last_updated %r, using current time", value)
        return datetime.now(timezone.utc).replace(microsecond=0)


@ttl_cache(config.PARKING_CACHE_TTL_S)
def fetch_garages() -> list[Garage]:
    """Return garages for the app, cached for ``PARKING_CACHE_TTL_S`` seconds.

    Only garages with coordinates and a known capacity are returned, since the
    map and ranking need both. Falls back to dummy garages if live data is
    unavailable (or ``USE_DUMMY_DATA`` is set), so the result is never empty
    and a failing API is queried at most once per cache period.
    """
    if config.USE_DUMMY_DATA:
        return list(_DUMMY_GARAGES)
    garages = [
        Garage(
            id=r["garage_id"],
            name=r["name"],
            location=Location(r["lat"], r["lon"], r["name"]),
            capacity=r["capacity"],
            free_spots=r["free_spots"],
            hourly_rate_chf=None,  # not provided by ParkenDD
            updated_at=r["timestamp"],
        )
        for r in fetch_occupancy(retries=1)
        if r["lat"] is not None and r["lon"] is not None and r["capacity"]
    ]
    if not garages:
        logger.warning("Live parking data unavailable, using dummy garages")
        return list(_DUMMY_GARAGES)
    return garages


def get_garage(garage_id: str) -> Garage | None:
    """Return a single garage by id, or ``None`` if unknown."""
    return next((g for g in fetch_garages() if g.id == garage_id), None)
