"""Occupancy prediction: the interface used by the recommendation layer and UI.

``predict(garage_id, arrival_time)`` combines three sources:

0. Arrival now: simply the latest measurement.
1. **ML model** (arrival up to 30 min ahead = the horizon it was trained for):
   features are built from the latest measurements (database + live value).
2. **Typical week** (arrival further ahead): the average free spots at the
   same weekday and time of day over the last weeks. The model cannot know
   what happens in between, so we start from what is normal for that time...
3. **...corrected by the current situation**: if the garage is fuller than
   usual right now, it will probably stay fuller for a while. That deviation
   is added on top of the typical value and fades out (half-life ~1 h).

Without model or history, the current live value is returned.
"""

from __future__ import annotations

import functools
import gzip
import logging
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import DB_PATH, FEATURE_FREQ_MIN, MODEL_PATH, ZURICH_TZ
from src.data import database, parking_api, weather_api
from src.data.http_utils import ttl_cache
from src.ml.features import build_features, zurich_holidays

logger = logging.getLogger(__name__)

FREQ = f"{FEATURE_FREQ_MIN}min"
PROFILE_WEEKS = 4  # history used for the typical week
DEVIATION_DECAY_MIN = 90  # e-folding time of the "current deviation" (half-life ~1 h)
NOW_TOLERANCE = pd.Timedelta(minutes=5)
_LAG_COLUMNS = ["lag_15min", "lag_1h", "rolling_mean_1h", "lag_1w"]
_BUNDLE_KEYS = {"model", "feature_columns", "fill_values", "garage_codes", "info"}


def load_model(path: Path | None = None) -> dict[str, Any] | None:
    """Load the trained model bundle, or ``None`` if there is none (yet).

    The file is read once and cached; a retrained model (newer file) is picked
    up automatically on the next call.
    """
    path = Path(path or MODEL_PATH)
    if not path.exists():
        return None
    return _load_bundle(str(path), path.stat().st_mtime)


@functools.lru_cache(maxsize=2)
def _load_bundle(path: str, mtime: float) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rb") as f:  # written by train.py (pickle + gzip)
            bundle = pickle.load(f)
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


def predict(garage_id: str, arrival_time: datetime) -> int:
    """Predict the number of free spots in a garage at a given time.

    Args:
        garage_id: Stable garage id (same as ``Garage.id``).
        arrival_time: Expected arrival. Naive datetimes are Zurich local time.

    Returns:
        Predicted number of free spots, clipped to ``[0, capacity]``.
        0 if the garage is completely unknown.
    """
    target = _to_utc(arrival_time).floor(FREQ)
    garage = parking_api.get_garage(garage_id)
    value, capacity = _estimate(garage_id, target)
    if garage is not None and garage.capacity:
        capacity = garage.capacity
    if value is None or math.isnan(value):
        # No history at all (e.g. dummy garages while the API is down).
        if garage is None or garage.free_spots is None:
            return 0
        value = float(garage.free_spots)
    value = max(0.0, value)
    if capacity and not math.isnan(capacity):
        value = min(value, float(capacity))
    return int(round(value))


@ttl_cache(300)
def _estimate(garage_id: str, target: pd.Timestamp) -> tuple[float | None, float | None]:
    """(unclipped estimate, capacity from history) for one garage and 15-min slot.

    Cached for 5 minutes per garage and slot.
    """
    history = _history(
        garage_id,
        since=min(
            target - pd.Timedelta(days=7, hours=2),
            pd.Timestamp.now(tz="UTC") - pd.Timedelta(weeks=PROFILE_WEEKS),
        ),
    )
    if history.empty:
        return None, None
    return _estimate_from_history(history, garage_id, target), history["capacity"].iloc[-1]


def _estimate_from_history(history: pd.DataFrame, garage_id: str, target: pd.Timestamp) -> float | None:
    """Apply the strategy described in the module docstring."""
    now = pd.Timestamp.now(tz="UTC")
    bundle = load_model()
    horizon = pd.Timedelta(minutes=bundle["info"]["horizon_min"]) if bundle else pd.Timedelta(0)

    # 0. Arrival now (or in the past): a fresh measurement beats any forecast.
    if target <= now:
        known = history[history["timestamp"] <= target + pd.Timedelta(FREQ)]
        if not known.empty and target - known["timestamp"].iloc[-1] <= pd.Timedelta(minutes=30):
            return float(known["free_spots"].iloc[-1])

    # 1. Within the model's horizon: ask the model.
    if bundle and target - horizon <= now + NOW_TOLERANCE:
        value = _model_prediction(bundle, history, garage_id, target)
        if value is not None:
            return value

    # 2. Further ahead: typical week + fading current deviation.
    typical = _typical_value(history, target)
    anchor_time = (now + horizon).ceil(FREQ)
    anchor = _model_prediction(bundle, history, garage_id, anchor_time) if bundle else None
    if anchor is None:  # no model: the latest measurement is our best "now"
        anchor_time, anchor = history["timestamp"].iloc[-1], float(history["free_spots"].iloc[-1])
    if typical is None:
        return anchor
    typical_at_anchor = _typical_value(history, anchor_time)
    if typical_at_anchor is None:
        return typical
    minutes_ahead = max(0.0, (target - anchor_time).total_seconds() / 60)
    return typical + (anchor - typical_at_anchor) * math.exp(-minutes_ahead / DEVIATION_DECAY_MIN)


def _model_prediction(bundle: dict[str, Any], history: pd.DataFrame, garage_id: str, target: pd.Timestamp) -> float | None:
    """Run the model for ``target``; ``None`` if there is no recent measurement."""
    horizon_min = bundle["info"]["horizon_min"]
    reference = target - pd.Timedelta(minutes=horizon_min)

    weather = None
    if bundle["info"].get("uses_weather"):
        w = weather_api.get_weather(target.to_pydatetime())
        if w:
            weather = pd.DataFrame([{"time": target.floor("h"), **w}])

    feats = build_features(history, horizon_min=horizon_min, weather=weather, until=reference)
    row = feats[feats["timestamp"] == reference].copy()
    if row.empty or pd.isna(row["free_now"].iloc[0]):
        return None  # latest data older than ~30 min before the reference time

    row["garage_code"] = bundle["garage_codes"].get(garage_id, -1)
    # Missing lags (short history): assume "no change" rather than a global median.
    for column in _LAG_COLUMNS:
        row[column] = row[column].fillna(row["free_now"])
    x = row.reindex(columns=bundle["feature_columns"]).fillna(bundle["fill_values"]).astype(float)
    return float(bundle["model"].predict(x)[0])


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


def _history(garage_id: str, since: pd.Timestamp) -> pd.DataFrame:
    """Measurements from the database plus the current live value.

    The live value is skipped for a synthetic database: mixing invented
    history with a real current value would produce meaningless forecasts.
    """
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
        return pd.DataFrame(columns=database.COLUMNS)
    history = pd.concat(frames, ignore_index=True).astype({"free_spots": float, "capacity": float})
    return history.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


@functools.lru_cache(maxsize=4)
def _is_synthetic(db_path: str, mtime: float) -> bool:
    """Cached check of the synthetic flag (re-read when the DB file changes)."""
    return database.is_synthetic(Path(db_path))


def _to_utc(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    return ts.tz_localize(ZURICH_TZ).tz_convert("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

