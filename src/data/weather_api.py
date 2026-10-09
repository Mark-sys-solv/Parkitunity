"""Weather data (used as an ML feature and for the mode recommendation)."""

from __future__ import annotations

from datetime import datetime

from src.models import Location


def get_weather(time: datetime, location: Location | None = None) -> dict[str, float]:
    """Return the weather forecast for a time and place.

    Args:
        time: Point in time of interest.
        location: Defaults to Zurich centre (config.ZURICH_CENTER).

    Returns:
        Dict with keys ``temperature_c``, ``precipitation_mm``.
    """
    # TODO: query config.WEATHER_API_URL (Open-Meteo, no key needed) with hourly data.
    # TODO: pick the hour closest to ``time``.
    return {"temperature_c": 12.0, "precipitation_mm": 0.0}


def get_historical_weather(start: datetime, end: datetime) -> list[dict[str, float]]:
    """Return hourly historical weather for building the training set."""
    # TODO: implement via Open-Meteo archive API.
    raise NotImplementedError
