"""Public transport connections via transport.opendata.ch.

API docs: https://transport.opendata.ch/docs.html
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from src.config import TRANSIT_API_URL, TRANSIT_CACHE_TTL_S, ZURICH_TZ
from src.data.http_utils import get_json, ttl_cache
from src.models import Location

logger = logging.getLogger(__name__)

# A place is an address/station name, a (lat, lon) tuple or a Location.
Place = str | tuple[float, float] | Location

_TZ = ZoneInfo(ZURICH_TZ)
_MAX_CONNECTIONS = 4


def get_transit_options(
    origin: Place,
    destination: Place,
    departure_time: datetime | None = None,
) -> dict[str, Any] | None:
    """Return the best public transport connection for a trip.

    "Best" means the earliest arrival among the next few connections departing
    at or after ``departure_time``. Identical requests within
    ``TRANSIT_CACHE_TTL_S`` seconds are served from an in-memory cache.

    Args:
        origin: Address, station name or (lat, lon) coordinates.
        destination: Address, station name or (lat, lon) coordinates.
        departure_time: Earliest departure. Naive datetimes are interpreted as
            Zurich local time. Defaults to now.

    Returns:
        ``{"duration_min": float, "transfers": int, "departure": datetime,
        "arrival": datetime, "sections": list[dict]}`` – see ``parse_section``
        for the section format. ``None`` if no connection was found or the
        API failed.
    """
    when = _to_zurich(departure_time or datetime.now(_TZ)).replace(second=0, microsecond=0)
    return _fetch_best_connection(_format_place(origin), _format_place(destination), when)


@ttl_cache(TRANSIT_CACHE_TTL_S)
def _fetch_best_connection(origin: str, destination: str, when: datetime) -> dict[str, Any] | None:
    """Query the API and pick the connection with the earliest arrival (cached)."""
    params = {
        "from": origin,
        "to": destination,
        "date": when.strftime("%Y-%m-%d"),
        "time": when.strftime("%H:%M"),
        "limit": _MAX_CONNECTIONS,
    }
    data = get_json(TRANSIT_API_URL, params=params)
    if not isinstance(data, dict) or not data.get("connections"):
        logger.warning("No transit connection found from %r to %r at %s", origin, destination, when)
        return None
    connections = [c for c in map(parse_connection, data["connections"]) if c is not None]
    if not connections:
        return None
    return min(connections, key=lambda c: c["arrival"])


def parse_connection(conn: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one API connection into our result format (``None`` if malformed)."""
    try:
        departure = _parse_ts(conn["from"]["departure"])
        arrival = _parse_ts(conn["to"]["arrival"])
        duration = (
            parse_duration(conn["duration"])
            if conn.get("duration")
            else (arrival - departure).total_seconds() / 60
        )
        return {
            "duration_min": duration,
            "transfers": int(conn.get("transfers") or 0),
            "departure": departure,
            "arrival": arrival,
            "sections": [parse_section(s) for s in conn.get("sections") or []],
        }
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Skipping malformed connection: %s", exc)
        return None


def parse_section(section: dict[str, Any]) -> dict[str, Any]:
    """Summarise one leg of a connection.

    Returns:
        ``{"mode": "walk" | "transit", "line": str | None, "from": str,
        "to": str, "departure": datetime, "arrival": datetime,
        "duration_min": float, "summary": str}``, e.g. summary
        ``"S19 Zürich Oerlikon → Zürich HB (5 min)"``.
    """
    dep, arr = section["departure"], section["arrival"]
    departure = _parse_ts(dep["departure"])
    arrival = _parse_ts(arr["arrival"])
    duration = round((arrival - departure).total_seconds() / 60, 1)
    from_name, to_name = dep["station"]["name"], arr["station"]["name"]

    journey = section.get("journey")
    if journey:
        mode = "transit"
        line = f"{journey.get('category') or ''}{journey.get('number') or ''}" or None
        label = line or "Transit"
    else:
        mode, line, label = "walk", None, "Walk"

    return {
        "mode": mode,
        "line": line,
        "from": from_name,
        "to": to_name,
        "departure": departure,
        "arrival": arrival,
        "duration_min": duration,
        "summary": f"{label} {from_name} → {to_name} ({duration:.0f} min)",
    }


def parse_duration(duration: str) -> float:
    """Convert an API duration like ``"00d00:23:00"`` to minutes."""
    days, clock = duration.split("d")
    hours, minutes, seconds = (int(x) for x in clock.split(":"))
    return int(days) * 1440 + hours * 60 + minutes + seconds / 60


def _format_place(place: Place) -> str:
    """Turn a place into the string the API expects (coordinates as "lat,lon")."""
    if isinstance(place, Location):
        return f"{place.lat},{place.lon}"
    if isinstance(place, tuple):
        lat, lon = place
        return f"{float(lat)},{float(lon)}"
    return place.strip()


def _parse_ts(value: str) -> datetime:
    """Parse API timestamps like ``"2026-11-20T14:02:00+0100"``."""
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")


def _to_zurich(dt: datetime) -> datetime:
    """Interpret naive datetimes as Zurich time; convert aware ones to it."""
    return dt.replace(tzinfo=_TZ) if dt.tzinfo is None else dt.astimezone(_TZ)
