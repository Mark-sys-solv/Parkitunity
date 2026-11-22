from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.config import WEATHER_API_URL, WEATHER_ARCHIVE_URL
from src.data import weather_api


@pytest.fixture
def api_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """Fake Open-Meteo: hourly data for the requested days, temperature = hour of day."""
    calls: list[tuple[str, dict]] = []

    def fake_get_json(url: str, params: dict | None = None, **kwargs: object) -> dict:
        calls.append((url, params))
        hours = pd.date_range(params["start_date"], f"{params['end_date']} 23:00", freq="h")
        return {
            "hourly": {
                "time": [h.strftime("%Y-%m-%dT%H:%M") for h in hours],
                "temperature_2m": [float(h.hour) for h in hours],
                "precipitation": [0.5] * len(hours),
            }
        }

    monkeypatch.setattr(weather_api, "get_json", fake_get_json)
    return calls


def test_hourly_weather_for_recent_period(api_calls) -> None:
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    df = weather_api.get_hourly_weather(start, start + timedelta(hours=5))
    assert list(df.columns) == weather_api.WEATHER_COLUMNS
    assert len(df) == 6
    assert str(df["time"].dt.tz) == "UTC"
    assert [url for url, _ in api_calls] == [WEATHER_API_URL]


def test_old_periods_use_archive(api_calls) -> None:
    end = datetime.now(timezone.utc) - timedelta(days=80)
    weather_api.get_hourly_weather(end - timedelta(days=20), end)
    assert [url for url, _ in api_calls] == [WEATHER_ARCHIVE_URL, WEATHER_API_URL]


def test_get_weather_returns_the_hour(api_calls) -> None:
    when = datetime.now(timezone.utc).replace(hour=14, minute=37)
    assert weather_api.get_weather(when) == {"temperature_c": 14.0, "precipitation_mm": 0.5}


def test_api_failure() -> None:
    # Network blocked (conftest) -> empty result, no exception.
    now = datetime.now(timezone.utc)
    assert weather_api.get_hourly_weather(now, now + timedelta(hours=2)).empty
    assert weather_api.get_weather(now) is None
