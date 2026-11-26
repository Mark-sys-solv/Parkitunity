"""Hourly weather for Zurich from Open-Meteo (free, no API key).

Used as optional ML features (temperature, precipitation). The forecast
endpoint covers roughly the last 3 months and the next 16 days; older data
comes from the archive endpoint.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.config import (
    REQUEST_TIMEOUT_S,
    WEATHER_API_URL,
    WEATHER_ARCHIVE_URL,
    ZURICH_CENTER,
)
from src.data.http_utils import get_json, ttl_cache
from src.models import Location

logger = logging.getLogger(__name__)

WEATHER_COLUMNS = ["time", "temperature_c", "precipitation_mm"]
_FORECAST_PAST_DAYS = 85  # stay safely inside the forecast endpoint's ~92-day window
_FORECAST_FUTURE_DAYS = 15


def get_hourly_weather(
    start: datetime,
    end: datetime,
    location: Location | None = None,
) -> pd.DataFrame:
    """Return hourly weather between ``start`` and ``end`` (past or forecast).

    Args:
        start: First hour of interest. Naive datetimes are treated as UTC.
        end: Last hour of interest.
        location: Defaults to Zurich centre.

    Returns:
        DataFrame with ``time`` (hourly, UTC), ``temperature_c`` and
        ``precipitation_mm``. Empty DataFrame if the API is unavailable.
    """
    lat, lon = (location.lat, location.lon) if location else ZURICH_CENTER
    start_utc, end_utc = _as_utc(start), _as_utc(end)
    today = datetime.now(timezone.utc).date()
    boundary = today - timedelta(days=_FORECAST_PAST_DAYS)

    # (url, first day, last day) – split between archive and forecast endpoint.
    requests_: list[tuple[str, date, date]] = []
    if start_utc.date() < boundary:
        requests_.append((WEATHER_ARCHIVE_URL, start_utc.date(), min(end_utc.date(), boundary - timedelta(days=1))))
    if end_utc.date() >= boundary:
        last_day = min(end_utc.date(), today + timedelta(days=_FORECAST_FUTURE_DAYS))
        requests_.append((WEATHER_API_URL, max(start_utc.date(), boundary), last_day))

    frames = []
    for url, first, last in requests_:
        if first > last:
            continue
        data = _fetch(url, lat, lon, first.isoformat(), last.isoformat())
        if data:
            frames.append(_parse(data))
    if not frames:
        return pd.DataFrame(columns=WEATHER_COLUMNS)

    df = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")
    mask = (df["time"] >= pd.Timestamp(start_utc).floor("h")) & (df["time"] <= pd.Timestamp(end_utc))
    return df[mask].reset_index(drop=True)


def get_weather(time: datetime, location: Location | None = None) -> dict[str, float] | None:
    """Return the weather for the hour containing ``time``.

    Naive datetimes are treated as UTC.

    Returns:
        ``{"temperature_c": ..., "precipitation_mm": ...}`` or ``None`` if unavailable.
    """
    hour = pd.Timestamp(_as_utc(time)).floor("h")
    df = get_hourly_weather(hour.to_pydatetime(), hour.to_pydatetime(), location)
    if df.empty:
        return None
    row = df.iloc[0]
    return {"temperature_c": float(row["temperature_c"]), "precipitation_mm": float(row["precipitation_mm"])}


@ttl_cache(3600)
def _fetch(url: str, lat: float, lon: float, start_date: str, end_date: str) -> dict[str, Any] | None:
    """Call Open-Meteo (cached for 1 h; failures are logged by get_json)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation",
        "timezone": "UTC",
        "start_date": start_date,
        "end_date": end_date,
    }
    data = get_json(url, params=params, timeout=REQUEST_TIMEOUT_S)
    if not isinstance(data, dict) or "hourly" not in data:
        return None
    return data


def _parse(data: dict[str, Any]) -> pd.DataFrame:
    """Convert the JSON answer into a table."""
    hourly = data["hourly"]
    return pd.DataFrame(
        {
            "time": pd.to_datetime(hourly["time"], utc=True),
            "temperature_c": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
            "precipitation_mm": pd.to_numeric(hourly["precipitation"], errors="coerce"),
        }
    )


def _as_utc(dt: datetime) -> datetime:
    """Convert to UTC (naive = UTC)."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
