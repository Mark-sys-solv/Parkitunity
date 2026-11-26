# AI citation: this file was written with Claude (Anthropic), following the group's document
# "Parkitunity: The Machine Learning Pattern" (FCS Assignments 10 and 11).
"""Occupancy prediction: the only place where forecasts are calculated.

The model answers one question: "how many spaces are free in 1 hour?". The app
needs other arrival times too, so ``predict_free(garage_id, timestamps)`` does this
(provisional - the group still has to decide how to handle other arrival times):

1. Now: the latest measurement.
2. Within the next hour: on the straight line from the measurement (now) to the
   model's forecast (in 1 hour). So the forecast always starts exactly at the
   live value, and the correction towards it fades out linearly within 60
   minutes. After exactly 1 hour only the model counts.
3. Further ahead: the average free spaces at the same weekday and time over the
   last weeks ("typical week"), corrected by how much the model's forecast
   differs from normal. This correction fades out (half-life ~1 h).

The three parts meet in the same points, so the curve has no jumps. (An earlier
version went BACKWARDS from the model's forecast along the typical week. While
a garage was filling up, this gave far too many free spaces for the first
minutes, e.g. 130 measured -> 168 -> 129.)

Without model or history, the current live value is returned.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    DB_PATH,
    FEATURE_FREQ_MIN,
    FORECAST_CURVE_HOURS,
    FORECAST_HORIZON_MIN,
    MODEL_PATH,
    PARKING_CACHE_TTL_S,
    ZURICH_TZ,
)
from src.data import database, parking_api, weather_api
from src.data.http_utils import ttl_cache
from src.ml.features import add_features, zurich_holidays

logger = logging.getLogger(__name__)

FREQ = f"{FEATURE_FREQ_MIN}min"
PROFILE_WEEKS = 4  # history used for the typical week
DEVIATION_DECAY_MIN = 90  # e-folding time of the "current deviation" (half-life ~1 h)
NOW_TOLERANCE = pd.Timedelta(minutes=5)
MAX_AGE = pd.Timedelta(minutes=30)  # older measurements no longer count as "now"
HORIZON = pd.Timedelta(minutes=FORECAST_HORIZON_MIN)
_BUNDLE_KEYS = {"model", "features", "garages", "capacities", "medians", "info"}


def load_model(path: Path | None = None) -> dict[str, Any] | None:
    """Load the trained model bundle, or ``None`` if there is none (yet).

    The file is read once and cached; a retrained model (newer file) is picked
    up automatically on the next call.
    """
    path = Path(path or MODEL_PATH)
    if not path.exists():
        return None
    return _load_bundle(str(path), path.stat().st_mtime)


@st.cache_resource(show_spinner=False, max_entries=2)
def _load_bundle(path: str, mtime: float) -> dict[str, Any] | None:
    """Read the model file once (``mtime``: a newer file is read again)."""
    try:
        bundle = joblib.load(path)  # written by train.py
    except Exception:  # corrupt file, incompatible sklearn version, ...
        logger.exception("Could not load model from %s", path)
        return None
    if not isinstance(bundle, dict) or not _BUNDLE_KEYS <= bundle.keys():
        logger.error("%s is not a valid model bundle", path)
        return None
    return bundle


def model_info() -> dict[str, Any] | None:
    """Metadata of the current model (data source, metrics, ...), if any."""
    bundle = load_model()
    return bundle["info"] if bundle else None


def predict_free(garage_id: str, timestamps: datetime | Iterable[datetime]) -> np.ndarray:
    """Predicted free spots of one garage - the only source of forecasts in the app.

    Args:
        garage_id: Stable garage id (same as ``Garage.id``).
        timestamps: One time or several. Naive datetimes are Zurich local time.

    Returns:
        Array of whole numbers in ``[0, capacity]``, one per timestamp
        (also an array for a single time). 0 if the garage is completely unknown.
    """
    times = [timestamps] if isinstance(timestamps, datetime) else list(timestamps)
    garage = parking_api.get_garage(garage_id)
    history = _history(garage_id)[0]

    if history.empty:
        # No history at all (e.g. dummy garages while the API is down): the live value.
        live = garage.free_spots if garage is not None and garage.free_spots is not None else 0
        values = [float(live)] * len(times)
        capacity = None
    else:
        now = _now()
        forecast = _forecast_in_1h(history, garage_id, now)  # the model runs only once for all times
        values = [_value_at(_to_utc(time), now, history, forecast) for time in times]
        capacity = history["capacity"].iloc[-1]

    # Limit to [0, capacity] and round. The live capacity is the most recent one.
    if garage is not None and garage.capacity:
        capacity = garage.capacity
    upper = math.inf if not capacity or math.isnan(capacity) else float(capacity)
    return np.array([round(min(max(value, 0.0), upper)) for value in values], dtype=int)


def forecast_curve(garage_id: str, hours: float = FORECAST_CURVE_HOURS) -> pd.DataFrame:
    """Forecast from now until ``hours`` ahead, every 15 minutes (via ``predict_free``).

    Returns:
        DataFrame with ``timestamp`` (Zurich time) and ``free_pred``.
    """
    start = _now().tz_convert(ZURICH_TZ).floor("min")
    times = pd.date_range(start, start + pd.Timedelta(hours=hours), freq=FREQ)
    return pd.DataFrame({"timestamp": times, "free_pred": predict_free(garage_id, times)})


def _now() -> pd.Timestamp:
    """Current time in UTC (separate function, so tests can fake the time)."""
    return pd.Timestamp.now(tz="UTC")


def _value_at(
    target: pd.Timestamp,
    now: pd.Timestamp,
    history: pd.DataFrame,
    forecast: tuple[float, float] | None,
) -> float:
    """Unlimited estimate for one time, see the steps in the module docstring.

    Args:
        forecast: Output of ``_forecast_in_1h``.
    """
    # A time in the past (a few minutes ago still counts as now): simply what was measured then.
    if target < now - NOW_TOLERANCE:
        known = history[history["timestamp"] <= target]
        if not known.empty and target - known["timestamp"].iloc[-1] <= MAX_AGE:
            return float(known["free_spots"].iloc[-1])

    # Steps 1 and 2: line from the measurement (now) to the model's forecast (in 1 hour).
    if forecast is not None:
        free_now, free_in_1h = forecast
        share = max(0.0, (target - now) / HORIZON)  # 0 = now, 1 = in 1 hour
        if share <= 1:
            return free_now + share * (free_in_1h - free_now)
        anchor_time, anchor = now + HORIZON, free_in_1h
    else:  # no model or no fresh value: the latest measurement is the best we have
        anchor_time, anchor = history["timestamp"].iloc[-1], float(history["free_spots"].iloc[-1])

    # Step 3: typical week + fading difference between the anchor and normal.
    typical, typical_at_anchor = _typical_value(history, target), _typical_value(history, anchor_time)
    if typical is None:
        return anchor
    if typical_at_anchor is None:
        return typical
    minutes_ahead = max(0.0, (target - anchor_time).total_seconds() / 60)
    return typical + (anchor - typical_at_anchor) * math.exp(-minutes_ahead / DEVIATION_DECAY_MIN)


def _forecast_in_1h(
    history: pd.DataFrame, garage_id: str, now: pd.Timestamp
) -> tuple[float, float] | None:
    """Ask the model: (free spaces now, predicted free spaces in 1 hour).

    A measurement of the last 30 minutes counts as "now".

    ``None`` without a model, for a garage the model does not know, or if the
    latest measurement is too old.
    """
    bundle = load_model()
    known = history[history["timestamp"] <= now + NOW_TOLERANCE]
    if bundle is None or garage_id not in bundle["garages"] or known.empty:
        return None
    latest = known.iloc[-1]
    if now - latest["timestamp"] > MAX_AGE:
        return None

    weather = None
    if bundle["info"]["uses_weather"]:
        current = weather_api.get_weather(latest["timestamp"].to_pydatetime())
        hour = latest["timestamp"].floor("h")
        # Without an answer from the weather API, the medians from training are used.
        weather = pd.DataFrame([{"time": hour, "temperature_c": None, "precipitation_mm": None, **(current or {})}])

    # One row in the same format as the training table.
    row = pd.DataFrame({
        "time": [latest["timestamp"].tz_convert(ZURICH_TZ)],
        "garage": [garage_id],
        "capacity": [latest["capacity"]],
        "free": [latest["free_spots"]],
    })
    row = add_features(row, bundle["garages"], weather)
    x = row[bundle["features"]].astype(float)
    x["capacity"] = x["capacity"].fillna(bundle["capacities"].get(garage_id))
    x = x.fillna(bundle["medians"])
    return float(latest["free_spots"]), float(bundle["model"].predict(x)[0])


def _typical_value(history: pd.DataFrame, when: pd.Timestamp) -> float | None:
    """Mean free spots at the same weekday and time of day (±15 min).

    Public holidays count as Sundays.
    """
    tz = ZoneInfo(ZURICH_TZ)
    local = history["timestamp"].dt.tz_convert(tz)
    when_local = when.tz_convert(tz)
    holidays = zurich_holidays(tuple(sorted({*local.dt.year, when_local.year})))

    day_type = local.dt.day_of_week.where(~local.dt.date.isin(holidays), 6)
    when_type = 6 if when_local.date() in holidays else when_local.day_of_week
    minute = local.dt.hour * 60 + local.dt.minute
    when_minute = when_local.hour * 60 + when_local.minute
    diff = (minute - when_minute).abs()
    mask = (day_type == when_type) & (diff.where(diff <= 720, 1440 - diff) <= 15)
    values = history.loc[mask, "free_spots"]
    return float(values.mean()) if len(values) else None


@ttl_cache(PARKING_CACHE_TTL_S)
def _history(garage_id: str) -> tuple[pd.DataFrame]:
    """Measurements of the last weeks from the database plus the current live value.

    Returned in a tuple, because ``ttl_cache`` cannot check if a table is empty.

    The live value is skipped for a synthetic database: mixing invented
    history with a real current value would produce meaningless forecasts.
    """
    since = _now() - pd.Timedelta(weeks=PROFILE_WEEKS)
    frames = []
    db_path = Path(DB_PATH)
    synthetic = db_path.exists() and _is_synthetic(str(db_path), db_path.stat().st_mtime)
    if db_path.exists():
        frames.append(database.load_history(garage_id, since=since.to_pydatetime(), db_path=db_path))
    garage = None if synthetic else parking_api.get_garage(garage_id)
    if garage is not None and garage.free_spots is not None and garage.updated_at is not None:
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp(garage.updated_at).tz_convert("UTC")],
                    "garage_id": [garage_id],
                    "free_spots": [garage.free_spots],
                    "capacity": [garage.capacity],
                }
            )
        )
    frames = [f for f in frames if not f.empty]
    if not frames:
        return (pd.DataFrame(columns=database.COLUMNS),)
    history = pd.concat(frames, ignore_index=True).astype({"free_spots": float, "capacity": float})
    return (history.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True),)


@functools.lru_cache(maxsize=4)
def _is_synthetic(db_path: str, mtime: float) -> bool:
    """Cached check of the synthetic flag (re-read when the DB file changes)."""
    return database.is_synthetic(Path(db_path))


def _to_utc(dt: datetime) -> pd.Timestamp:
    """Convert to UTC (naive = Zurich time)."""
    ts = pd.Timestamp(dt)
    return ts.tz_localize(ZURICH_TZ).tz_convert("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

