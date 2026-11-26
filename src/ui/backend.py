"""UI data layer: cached backend calls.

Every API/model call of the app goes through this module, so that Streamlit
caches results across reruns (``st.cache_data``). Placeholder prices are
switched in ``config.USE_PLACEHOLDER_PRICES``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src import config
from src.data import database, geocoding
from src.ml import predict
from src.models import Location, TransitOption
from src.recommendation.candidates import build_candidates
from src.recommendation.mode_compare import NO_TRANSIT, transit_option

TZ = ZoneInfo(config.ZURICH_TZ)

Status = Literal["good", "tight", "full"]
STATUS_LABELS: dict[Status, str] = {"good": "Gut verfügbar", "tight": "Wird knapp", "full": "Fast voll"}
# Fixed status colours (always shown together with the label, never alone).
STATUS_COLORS: dict[Status, str] = {"good": "#0ca30c", "tight": "#fab219", "full": "#d03b3b"}


# --- Time & formatting helpers -------------------------------------------------


def now() -> datetime:
    """Current time in Zurich (the server itself may run in UTC)."""
    return datetime.now(TZ)


def display_name(name: str, garage_id: str) -> str:
    """"Parkhaus Urania" instead of ParkenDD's bare "Urania"."""
    if name.lower().startswith("park"):
        return name
    return f"{'Parkplatz' if 'parkplatz' in garage_id else 'Parkhaus'} {name}"


def availability(free: int) -> Status:
    """Traffic-light status of a number of free spots (thresholds: config)."""
    if free >= config.FREE_GREEN_FROM:
        return "good"
    return "tight" if free >= config.FREE_ORANGE_FROM else "full"


def prices_are_placeholders() -> bool:
    """True while the hourly prices are only guesses."""
    return config.USE_PLACEHOLDER_PRICES


def uses_synthetic_db() -> bool:
    """True if the database contains invented data."""
    db = Path(config.DB_PATH)
    return db.exists() and database.is_synthetic(db)


# --- Geocoding -----------------------------------------------------------------


def geocode(query: str) -> Location | None:
    """Resolve an address, place name or ``"lat, lon"`` (Nominatim, cached); ``None`` if unknown."""
    coords = geocoding.geocode(query)
    return Location(*coords, name=query.strip()) if coords is not None else None


# --- Cached backend calls --------------------------------------------------------


@st.cache_data(ttl=60, show_spinner=False)
def candidates(
    dest_lat: float,
    dest_lon: float,
    departure_iso: str,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> pd.DataFrame:
    """All garages for the trip (``build_candidates``, without price limit)."""
    origin = Location(origin_lat, origin_lon) if origin_lat is not None and origin_lon is not None else None
    return build_candidates(origin, Location(dest_lat, dest_lon), datetime.fromisoformat(departure_iso))


class TransitUnavailable(Exception):
    """No connection found. Raised so that Streamlit does not cache the failure."""


@st.cache_data(ttl=config.TRANSIT_CACHE_TTL_S, show_spinner=False)
def _transit(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float, departure_iso: str) -> TransitOption:
    """Cached request to the transit API."""
    option = transit_option(
        Location(origin_lat, origin_lon), Location(dest_lat, dest_lon), datetime.fromisoformat(departure_iso)
    )
    if not option["available"]:
        raise TransitUnavailable
    return option


def transit(origin: Location, destination: Location, departure: datetime) -> TransitOption:
    """Next public transport connection (``NO_TRANSIT`` if the API fails)."""
    try:
        return _transit(origin.lat, origin.lon, destination.lat, destination.lon, departure.isoformat())
    except TransitUnavailable:
        return NO_TRANSIT


@st.cache_data(ttl=config.PARKING_CACHE_TTL_S, show_spinner=False)
def forecast_curve(garage_id: str, hours: int) -> pd.DataFrame:
    """Cached forecast (``predict.forecast_curve``)."""
    return predict.forecast_curve(garage_id, hours)
