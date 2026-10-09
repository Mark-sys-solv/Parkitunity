"""Feature engineering for the occupancy prediction model."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

FEATURE_COLUMNS: list[str] = [
    "garage_id",
    "hour",
    "weekday",
    "is_weekend",
    "is_holiday",
    "temperature_c",
    "precipitation_mm",
]
TARGET_COLUMN: str = "free_spots"


def build_features(occupancy: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Turn raw occupancy history into a training table.

    Args:
        occupancy: Output of ``database.load_occupancy``.
        weather: Optional hourly weather, joined on the rounded timestamp.

    Returns:
        DataFrame with ``FEATURE_COLUMNS`` + ``TARGET_COLUMN``.
    """
    # TODO: extract hour/weekday/is_weekend from timestamp.
    # TODO: add Zurich public holidays (e.g. `holidays` package, canton ZH).
    # TODO: join weather data.
    raise NotImplementedError


def features_for(garage_id: str, arrival_time: datetime) -> pd.DataFrame:
    """Build a single-row feature frame for a prediction request.

    Must produce exactly the same columns as ``build_features``.
    """
    # TODO: fetch weather forecast via weather_api.get_weather(arrival_time).
    raise NotImplementedError
