"""UI data layer: cached backend calls.

Every API/model call of the app goes through this module, so that Streamlit
caches results across reruns (``st.cache_data``). Placeholder prices are
switched in ``config.USE_PLACEHOLDER_PRICES``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src import config
from src.data import database, geocoding, parking_api
from src.ml.predict import predict
from src.models import Garage, Location, ModeComparison
from src.recommendation.mode_compare import compare_modes
from src.recommendation.parking_score import rank_garages

TZ = ZoneInfo(config.ZURICH_TZ)

Status = Literal["good", "tight", "full"]
STATUS_LABELS: dict[Status, str] = {"good": "Gut verfügbar", "tight": "Knapp", "full": "Fast voll"}
# Fixed status colours (always shown together with the label, never alone).
STATUS_COLORS: dict[Status, str] = {"good": "#0ca30c", "tight": "#fab219", "full": "#d03b3b"}


# --- Time & formatting helpers -------------------------------------------------


def now() -> datetime:
    """Current time in Zurich (the server itself may run in UTC)."""
    return datetime.now(TZ)


def display_name(garage: Garage) -> str:
    """"Parkhaus Urania" instead of ParkenDD's bare "Urania"."""
    if garage.name.lower().startswith("park"):
        return garage.name
    return f"{'Parkplatz' if 'parkplatz' in garage.id else 'Parkhaus'} {garage.name}"


def availability(free: int, capacity: int | None) -> Status:
    """Traffic-light status of a predicted number of free spots."""
    ratio = free / capacity if capacity else None
    if free < 5 or (ratio is not None and ratio < 0.05):
        return "full"
    if free < 20 or (ratio is not None and ratio < 0.15):
        return "tight"
    return "good"


def prices_are_placeholders() -> bool:
    return config.USE_PLACEHOLDER_PRICES


def uses_synthetic_db() -> bool:
    db = Path(config.DB_PATH)
    return db.exists() and database.is_synthetic(db)


# --- Geocoding -----------------------------------------------------------------


def geocode(query: str) -> Location | None:
    """Resolve an address, place name or ``"lat, lon"`` (Nominatim, cached); ``None`` if unknown."""
    coords = geocoding.geocode(query)
    return Location(*coords, name=query.strip()) if coords is not None else None


# --- Cached backend calls --------------------------------------------------------


@st.cache_data(ttl=60, show_spinner=False)
def ranked_garages(
    dest_lat: float,
    dest_lon: float,
    departure_iso: str,
    origin_lat: float | None = None,
    origin_lon: float | None = None,
) -> list[dict]:
    """All usable garages for the trip (``rank_garages`` without price limit).

    The price filter is applied by the caller so that over-budget garages can
    still be shown (greyed out) on the map. Scores do not depend on the other
    garages, so the order is the same as with ``max_price``.
    """
    origin = Location(origin_lat, origin_lon) if origin_lat is not None and origin_lon is not None else None
    return rank_garages(Location(dest_lat, dest_lon), datetime.fromisoformat(departure_iso), origin=origin)


@st.cache_data(ttl=300, show_spinner=False)
def mode_comparison(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    departure_iso: str,
    max_price: float | None = None,
) -> ModeComparison:
    """Car vs. public transport (``compare_modes``)."""
    return compare_modes(
        Location(origin_lat, origin_lon),
        Location(dest_lat, dest_lon),
        datetime.fromisoformat(departure_iso),
        max_price=max_price,
    )


@st.cache_data(ttl=300, show_spinner=False)
def occupancy_history(garage_id: str, hours: int = 24) -> pd.DataFrame:
    """Measured free spots of the last ``hours`` (database + live value).

    Returns:
        DataFrame with ``time`` (Zurich time) and ``free_spots``; may be empty.
    """
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    frames = []
    db = Path(config.DB_PATH)
    if db.exists():
        frames.append(database.load_history(garage_id, since=since.to_pydatetime(), db_path=db))
    # The live value would not fit synthetic history (see predict._history).
    garage = None if uses_synthetic_db() else parking_api.get_garage(garage_id)
    if garage is not None and garage.free_spots is not None and garage.updated_at is not None:
        frames.append(pd.DataFrame({"timestamp": [pd.Timestamp(garage.updated_at)], "free_spots": [garage.free_spots]}))
    frames = [f[["timestamp", "free_spots"]] for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame({"time": pd.Series(dtype=f"datetime64[ns, {config.ZURICH_TZ}]"), "free_spots": []})
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(TZ)
    df = df[df["time"] >= since.tz_convert(TZ)]
    return df.drop_duplicates("time").sort_values("time")[["time", "free_spots"]].reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def occupancy_forecast(garage_id: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    """``predict()`` every 15 minutes between start and end.

    Returns:
        DataFrame with ``time`` (Zurich time) and ``free_spots``; empty if the
        garage is unknown (``predict`` would return a misleading 0 for it, e.g.
        when ParkenDD switches between live and dummy garages).
    """
    if parking_api.get_garage(garage_id) is None and occupancy_history(garage_id).empty:
        return pd.DataFrame({"time": pd.Series(dtype=f"datetime64[ns, {config.ZURICH_TZ}]"), "free_spots": []})
    start = pd.Timestamp(start_iso).tz_convert(TZ).ceil("15min")
    times = pd.date_range(start, pd.Timestamp(end_iso).tz_convert(TZ), freq="15min")
    return pd.DataFrame({"time": times, "free_spots": [predict(garage_id, t.to_pydatetime()) for t in times]})


def forecast_window(arrival: datetime) -> tuple[datetime, datetime]:
    """From now until at least 6 h ahead and 1 h after arriving at the garage (max. 24 h)."""
    start = now()
    end = min(max(start + timedelta(hours=6), arrival + timedelta(hours=1)), start + timedelta(hours=24))
    return start, end
