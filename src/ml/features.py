# AI citation: this file was written with Claude (Anthropic), following the group's document
# "Parkitunity: The Machine Learning Pattern" (FCS Assignments 10 and 11).
"""Build the table the model learns from: one row per garage per 15 minutes.

Columns: ``time`` (Zurich time), ``garage``, ``capacity``, ``free``. On top come

- the target ``free_in_1h``: free spaces in the same garage 1 hour later, and
- the features: only things that are known at ``time`` (no future values!).

Training and prediction both use ``add_features``, so a feature always means
the same thing.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import pandas as pd

from src.config import FEATURE_FREQ_MIN, FORECAST_HORIZON_MIN, ZURICH_TZ

FREQ = f"{FEATURE_FREQ_MIN}min"
STEPS_AHEAD = FORECAST_HORIZON_MIN // FEATURE_FREQ_MIN  # 1 hour = 4 rows of 15 minutes

TARGET = "free_in_1h"
FEATURES = ["free", "hour", "weekday", "garage_id", "capacity"]
WEATHER_FEATURES = ["temperature_c", "rain_mm"]  # optional
HOLIDAY_FEATURES = ["is_holiday"]  # optional

# Human-readable names for the report and the video.
FEATURE_LABELS = {
    "free": "Free spaces now",
    "hour": "Hour of day",
    "weekday": "Weekday",
    "garage_id": "Garage",
    "capacity": "Capacity",
    "temperature_c": "Temperature",
    "rain_mm": "Rain",
    "is_holiday": "Public holiday",
}


def build_table(history: pd.DataFrame) -> pd.DataFrame:
    """Turn raw measurements into one row per garage per 15 minutes, plus the target.

    Args:
        history: Output of ``database.load_history`` (columns ``timestamp``,
            ``garage_id``, ``free_spots``, ``capacity``).

    Returns:
        DataFrame with ``time``, ``garage``, ``capacity``, ``free`` and
        ``free_in_1h``. Slots without a measurement are NaN.
    """
    parts = []
    for garage, rows in history.groupby("garage_id"):
        # The last measurement in every 15-minute slot. Empty slots stay as NaN rows, so
        # "4 rows later" is always exactly 1 hour later, even next to a data gap.
        slots = rows.set_index("timestamp")[["free_spots", "capacity"]].astype(float).resample(FREQ).last()
        part = pd.DataFrame({
            "time": slots.index.tz_convert(ZURICH_TZ),
            "garage": garage,
            "capacity": slots["capacity"].ffill().bfill().to_numpy(),
            "free": slots["free_spots"].to_numpy(),
        })
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["time", "garage", "capacity", "free", TARGET])

    table = pd.concat(parts, ignore_index=True).sort_values(["garage", "time"]).reset_index(drop=True)
    # Target: free spaces 1 hour later in the same garage.
    table[TARGET] = table.groupby("garage")["free"].shift(-STEPS_AHEAD)
    return table


def add_features(table: pd.DataFrame, garages: list[str], weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add the feature columns to a table with ``time``, ``garage``, ``capacity`` and ``free``.

    Args:
        garages: All garage names; the position in this list is the ``garage_id`` number.
        weather: Optional hourly weather (``weather_api.get_hourly_weather``).
    """
    table = table.copy()
    table["hour"] = table["time"].dt.hour
    table["weekday"] = table["time"].dt.day_of_week  # 0 = Monday
    table["garage_id"] = table["garage"].map({name: number for number, name in enumerate(garages)})
    holidays = zurich_holidays(tuple(sorted(set(table["time"].dt.year))))
    table["is_holiday"] = table["time"].dt.date.isin(holidays).astype(int)
    if weather is not None:
        hourly = weather.rename(columns={"time": "hour_time", "precipitation_mm": "rain_mm"})
        table["hour_time"] = table["time"].dt.tz_convert("UTC").dt.floor("h")  # 14:15 -> 14:00
        table = table.merge(hourly[["hour_time", *WEATHER_FEATURES]], on="hour_time", how="left")
    return table


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
