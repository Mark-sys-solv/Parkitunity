"""Feature engineering for the occupancy forecast.

Every output row describes one garage at one *reference time* ``t`` (a slot on
a regular 15-minute grid) and is used to predict the free spots at the
*target time* ``T = t + horizon`` (30 minutes by default):

- calendar features describe ``T`` (that's when the driver arrives),
- state features (current value, lags, rolling mean) describe what is known at ``t``,
- ``lag_1w`` is the value one week before ``T`` (this is also the baseline),
- ``target`` is the actually observed value at ``T`` (NaN if unknown).

Training (``train.py``) and prediction (``predict.py``) both use
``build_features`` so the features are guaranteed to be computed identically.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from src.config import FEATURE_FREQ_MIN, FORECAST_HORIZON_MIN, ZURICH_TZ

FREQ = f"{FEATURE_FREQ_MIN}min"
MAX_FILL_SLOTS = 2  # carry the last value forward over gaps of up to 30 min

CALENDAR_FEATURES = ["hour", "weekday", "is_weekend", "is_holiday", "month"]
STATE_FEATURES = [
    "free_now",
    "capacity",
    "occupancy_rate",
    "rolling_mean_1h",
    "lag_15min",
    "lag_1h",
    "lag_1w",
]
WEATHER_FEATURES = ["temperature_c", "precipitation_mm"]
TARGET_COLUMN = "target"

# Human-readable names for plots and the video.
FEATURE_LABELS = {
    "hour": "Hour of day",
    "weekday": "Weekday",
    "is_weekend": "Weekend",
    "is_holiday": "Public holiday",
    "month": "Month",
    "free_now": "Free spots now",
    "capacity": "Capacity",
    "occupancy_rate": "Occupancy rate now",
    "rolling_mean_1h": "Mean of last hour",
    "lag_15min": "Free spots 15 min ago",
    "lag_1h": "Free spots 1 h ago",
    "lag_1w": "Same time last week",
    "temperature_c": "Temperature",
    "precipitation_mm": "Precipitation",
    "garage_code": "Garage",
}


def build_features(
    history: pd.DataFrame,
    horizon_min: int = FORECAST_HORIZON_MIN,
    weather: pd.DataFrame | None = None,
    until: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Turn raw occupancy history into one feature row per garage and 15-min slot.

    Args:
        history: Output of ``database.load_history`` (columns ``timestamp``,
            ``garage_id``, ``free_spots``, ``capacity``).
        horizon_min: How far ahead the target lies (multiple of 15).
        weather: Optional hourly weather (``weather_api.get_hourly_weather``);
            adds ``temperature_c`` / ``precipitation_mm`` for the target hour.
        until: Extend each garage's grid up to this time (UTC), e.g. "now"
            when predicting and the latest measurement is a few minutes old.

    Returns:
        DataFrame with ``timestamp`` (reference time t, UTC), ``target_time``,
        ``garage_id``, all ``CALENDAR_FEATURES`` + ``STATE_FEATURES``
        (+ ``WEATHER_FEATURES`` if weather given) and ``TARGET_COLUMN``.
        Lags that reach before the start of the data are NaN.
    """
    if horizon_min % FEATURE_FREQ_MIN:
        raise ValueError(f"horizon_min must be a multiple of {FEATURE_FREQ_MIN}")
    if history.empty:
        return pd.DataFrame(columns=["timestamp", "target_time", "garage_id", *CALENDAR_FEATURES, *STATE_FEATURES, TARGET_COLUMN])

    grid = _to_grid(history, until)
    h = horizon_min // FEATURE_FREQ_MIN
    week = 7 * 24 * 60 // FEATURE_FREQ_MIN
    by_garage = grid.groupby("garage_id", sort=False)

    # --- What we know at reference time t ------------------------------------
    grid["occupancy_rate"] = 1 - grid["free_now"] / grid["capacity"]
    grid["lag_15min"] = by_garage["free_now"].shift(1)
    grid["lag_1h"] = by_garage["free_now"].shift(4)
    # Slots t-45min .. t, i.e. the last hour.
    grid["rolling_mean_1h"] = by_garage["free_now"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    # Same time as the target, one week earlier: T - 7 days = t - (7 days - horizon).
    grid["lag_1w"] = by_garage["free_now"].shift(week - h)

    # --- Target: the real (not forward-filled) value at T ---------------------
    grid[TARGET_COLUMN] = by_garage["free_raw"].shift(-h)

    # --- Calendar of the target time (local Zurich time) ----------------------
    grid["target_time"] = grid["timestamp"] + pd.Timedelta(minutes=horizon_min)
    grid = grid.join(calendar_features(grid["target_time"]))

    if weather is not None:
        grid = _join_weather(grid, weather)

    return grid.drop(columns=["free_raw"])


def calendar_features(times: pd.Series) -> pd.DataFrame:
    """Calendar features for UTC timestamps, evaluated in Zurich local time."""
    local = times.dt.tz_convert(ZURICH_TZ)
    holiday_dates = zurich_holidays(tuple(sorted(set(local.dt.year))))
    weekday = local.dt.day_of_week
    return pd.DataFrame(
        {
            "hour": local.dt.hour,
            "weekday": weekday,
            "is_weekend": (weekday >= 5).astype(int),
            "is_holiday": local.dt.date.isin(holiday_dates).astype(int),
            "month": local.dt.month,
        },
        index=times.index,
    )


def easter_sunday(year: int) -> date:
    """Date of Easter Sunday in the Gregorian calendar.

    Uses the well-known "anonymous Gregorian algorithm" (Meeus/Jones/Butcher):
    only integer arithmetic, valid for all years after 1583.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    weekday_shift = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * weekday_shift) // 451
    month = (h + weekday_shift - 7 * m + 114) // 31
    day = (h + weekday_shift - 7 * m + 114) % 31 + 1
    return date(year, month, day)


@lru_cache(maxsize=16)
def zurich_holidays(years: tuple[int, ...]) -> frozenset[date]:
    """Public holidays in the canton of Zurich, plus Berchtoldstag (2 January).

    Berchtoldstag is not an official cantonal holiday, but most businesses in
    Zurich are closed, so it behaves like one for parking demand.
    """
    days = set()
    for year in years:
        # Fixed dates (month, day).
        for month, day in [(1, 1), (1, 2), (5, 1), (8, 1), (12, 25), (12, 26)]:
            days.add(date(year, month, day))  # New Year, Berchtoldstag, Labour Day, National Day, Christmas, St Stephen
        # Movable holidays: a fixed number of days before/after Easter Sunday.
        easter = easter_sunday(year)
        for offset in (-2, 1, 39, 50):  # Good Friday, Easter Monday, Ascension, Whit Monday
            days.add(easter + timedelta(days=offset))
    return frozenset(days)


def _to_grid(history: pd.DataFrame, until: pd.Timestamp | None) -> pd.DataFrame:
    """Resample each garage onto a regular 15-min grid.

    ``free_raw`` holds the last measurement inside each slot (NaN if none),
    ``free_now`` additionally carries values forward over short gaps.
    """
    df = history.dropna(subset=["free_spots"]).copy()
    df["slot"] = df["timestamp"].dt.floor(FREQ)
    per_slot = df.sort_values("timestamp").groupby(["garage_id", "slot"], as_index=False).last()

    frames = []
    for garage_id, g in per_slot.groupby("garage_id", sort=False):
        end = max(g["slot"].max(), until.floor(FREQ)) if until is not None else g["slot"].max()
        slots = pd.date_range(g["slot"].min(), end, freq=FREQ)
        g = g.set_index("slot").reindex(slots)
        free_raw = g["free_spots"].astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": slots,
                    "garage_id": garage_id,
                    "free_raw": free_raw.to_numpy(),
                    "free_now": free_raw.ffill(limit=MAX_FILL_SLOTS).to_numpy(),
                    "capacity": g["capacity"].astype(float).ffill().bfill().to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _join_weather(grid: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Add the weather of the target hour."""
    w = weather.rename(columns={"time": "_weather_hour"})[["_weather_hour", *WEATHER_FEATURES]]
    grid["_weather_hour"] = grid["target_time"].dt.floor("h")
    return grid.merge(w, on="_weather_hour", how="left").drop(columns=["_weather_hour"])
