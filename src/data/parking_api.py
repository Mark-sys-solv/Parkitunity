"""Live parking garage data for Zurich (Parkleitsystem Zürich)."""

from __future__ import annotations

from datetime import datetime

from src.models import Garage, Location

# Dummy garages so the app runs without network access.
# Coordinates are approximate; capacities/rates are placeholders.
_DUMMY_GARAGES: list[Garage] = [
    Garage("urania", "Parkhaus Urania", Location(47.3744, 8.5396, "Urania"), 600, 120, 4.0),
    Garage("jelmoli", "Parkhaus Jelmoli", Location(47.3737, 8.5373, "Jelmoli"), 220, 15, 5.0),
    Garage("hb", "Parkhaus Hauptbahnhof", Location(47.3797, 8.5388, "Hauptbahnhof"), 180, 40, 4.5),
    Garage("opera", "Parkhaus Opéra", Location(47.3650, 8.5470, "Opernhaus"), 300, 90, 4.0),
    Garage("hohe-promenade", "Parkhaus Hohe Promenade", Location(47.3685, 8.5480, "Hohe Promenade"), 550, 210, 3.5),
]


def fetch_garages() -> list[Garage]:
    """Return all garages with their current number of free spots.

    Returns:
        List of ``Garage`` objects with ``free_spots`` and ``updated_at`` set.
    """
    # TODO: fetch and parse config.PARKING_FEED_URL (RSS) with requests.
    # TODO: map feed entries to Garage objects (ids must be stable for the DB/ML).
    # TODO: fall back to dummy data if config.USE_DUMMY_DATA or on network error.
    now = datetime.now()
    return [
        Garage(g.id, g.name, g.location, g.capacity, g.free_spots, g.hourly_rate_chf, now)
        for g in _DUMMY_GARAGES
    ]


def get_garage(garage_id: str) -> Garage | None:
    """Return a single garage by id, or ``None`` if unknown."""
    # TODO: cache fetch_garages() results to avoid repeated requests.
    return next((g for g in fetch_garages() if g.id == garage_id), None)


def parse_feed(raw_xml: str) -> list[Garage]:
    """Parse the raw RSS feed into ``Garage`` objects.

    Args:
        raw_xml: Response body of the parking feed.
    """
    # TODO: implement (xml.etree.ElementTree or feedparser).
    raise NotImplementedError
