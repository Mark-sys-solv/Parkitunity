"""Public transport connections via transport.opendata.ch."""

from __future__ import annotations

from datetime import datetime, timedelta

from src.models import TransitConnection


def get_connections(
    origin: str,
    destination: str,
    time: datetime,
    is_arrival_time: bool = False,
    limit: int = 4,
) -> list[TransitConnection]:
    """Return public transport connections between two places.

    Args:
        origin: Station name or address, e.g. "Zürich Oerlikon".
        destination: Station name or address.
        time: Desired departure (or arrival, see ``is_arrival_time``) time.
        is_arrival_time: If True, ``time`` is the latest arrival time.
        limit: Maximum number of connections.

    Returns:
        Connections sorted by departure time.
    """
    # TODO: GET config.TRANSIT_API_URL with params from/to/date/time/isArrivalTime/limit.
    # TODO: parse "duration" (format "00d00:23:00") with parse_duration().
    # TODO: fall back to dummy data on error.
    return [
        TransitConnection(
            departure=time,
            arrival=time + timedelta(minutes=24),
            duration_min=24.0,
            transfers=1,
            products=["S12", "T4"],
        )
    ]


def parse_duration(duration: str) -> float:
    """Convert a transport.opendata.ch duration like "00d00:23:00" to minutes."""
    # TODO: implement.
    raise NotImplementedError
