from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ml.features import CALENDAR_FEATURES, STATE_FEATURES, TARGET_COLUMN, build_features, easter_sunday, zurich_holidays

START = pd.Timestamp("2026-12-17 23:00", tz="UTC")  # = 18 Dec 00:00 in Zurich


def make_history(days: int = 9, garage_id: str = "g1", capacity: int = 1000) -> pd.DataFrame:
    """Regular 15-min measurements where free_spots = slot index (easy to check lags)."""
    slots = pd.date_range(START, periods=days * 96, freq="15min")
    return pd.DataFrame(
        {
            "timestamp": slots + pd.Timedelta(minutes=3),  # measurements a few minutes after each slot
            "garage_id": garage_id,
            "free_spots": np.arange(len(slots)) % capacity,
            "capacity": capacity,
        }
    )


def test_columns_and_one_row_per_slot() -> None:
    feats = build_features(make_history())
    for column in ["timestamp", "target_time", "garage_id", *CALENDAR_FEATURES, *STATE_FEATURES, TARGET_COLUMN]:
        assert column in feats.columns
    assert len(feats) == 9 * 96


def test_lags_rolling_mean_and_target() -> None:
    feats = build_features(make_history(), horizon_min=30).set_index("timestamp")
    t = START + pd.Timedelta(days=8)  # slot index 768
    row = feats.loc[t]
    assert row["free_now"] == 768
    assert row["lag_15min"] == 767
    assert row["lag_1h"] == 764
    assert row["rolling_mean_1h"] == pytest.approx((765 + 766 + 767 + 768) / 4)
    assert row["target"] == 770  # value 30 min later
    # Same time as the target (t + 30 min), one week earlier: index 770 - 672.
    assert row["lag_1w"] == 770 - 672
    assert row["occupancy_rate"] == pytest.approx(1 - 768 / 1000)


def test_first_week_has_no_weekly_lag_and_last_rows_no_target() -> None:
    feats = build_features(make_history())
    assert feats["lag_1w"].iloc[:600].isna().all()
    assert feats[TARGET_COLUMN].iloc[-2:].isna().all()


def test_calendar_uses_zurich_time_of_target() -> None:
    feats = build_features(make_history(), horizon_min=30).set_index("timestamp")
    # Reference 24 Dec 22:30 UTC -> target 23:00 UTC = 25 Dec 00:00 Zurich (Christmas).
    row = feats.loc[pd.Timestamp("2026-12-24 22:30", tz="UTC")]
    assert (row["hour"], row["weekday"], row["month"], row["is_holiday"]) == (0, 4, 12, 1)
    assert row["is_weekend"] == 0


@pytest.mark.parametrize("year, expected", [(2024, date(2024, 3, 31)), (2025, date(2025, 4, 20)),
                                            (2026, date(2026, 4, 5)), (2027, date(2027, 3, 28))])
def test_easter_sunday(year: int, expected: date) -> None:
    assert easter_sunday(year) == expected


def test_zurich_holidays_2026() -> None:
    expected = {date(2026, m, d) for m, d in [(1, 1), (1, 2), (4, 3), (4, 6), (5, 1), (5, 14), (5, 25), (8, 1), (12, 25), (12, 26)]}
    assert zurich_holidays((2026,)) == expected


def test_zurich_holidays_include_berchtoldstag() -> None:
    days = zurich_holidays((2027,))
    assert date(2027, 1, 2) in days  # Berchtoldstag (added manually)
    assert date(2027, 8, 1) in days  # national day
    assert date(2027, 3, 26) in days  # Good Friday


def test_short_gaps_are_filled_but_target_stays_raw() -> None:
    history = make_history(days=2)
    gap = START + pd.Timedelta(hours=10)
    history = history[history["timestamp"].dt.floor("15min") != gap]
    feats = build_features(history).set_index("timestamp")
    assert feats.loc[gap, "free_now"] == feats.loc[gap - pd.Timedelta("15min"), "free_now"]  # carried forward
    assert np.isnan(feats.loc[gap - pd.Timedelta("30min"), TARGET_COLUMN])  # no fake target


def test_long_gaps_are_not_filled() -> None:
    history = make_history(days=2)
    gap = (history["timestamp"] > START + pd.Timedelta(hours=10)) & (history["timestamp"] < START + pd.Timedelta(hours=12))
    feats = build_features(history[~gap]).set_index("timestamp")
    assert np.isnan(feats.loc[START + pd.Timedelta(hours=11), "free_now"])


def test_until_extends_grid() -> None:
    history = make_history(days=1)
    until = history["timestamp"].max() + pd.Timedelta(minutes=40)
    feats = build_features(history, until=until)
    assert feats["timestamp"].max() == until.floor("15min")


def test_multiple_garages_do_not_mix() -> None:
    history = pd.concat([make_history(days=1, garage_id="a"), make_history(days=1, garage_id="b", capacity=50)])
    feats = build_features(history)
    b = feats[feats["garage_id"] == "b"]
    assert b["free_now"].max() < 50
    assert np.isnan(b["lag_15min"].iloc[0])  # no value leaking over from garage "a"


def test_weather_is_joined_on_target_hour() -> None:
    history = make_history(days=1)
    weather = pd.DataFrame(
        {
            "time": pd.date_range(START, periods=30, freq="h"),
            "temperature_c": np.arange(30, dtype=float),
            "precipitation_mm": 0.0,
        }
    )
    feats = build_features(history, horizon_min=30, weather=weather).set_index("timestamp")
    # Reference 01:45 UTC -> target 02:15 -> weather hour 02:00 = START + 3 h -> temperature 3.
    assert feats.loc[START + pd.Timedelta(hours=2, minutes=45), "temperature_c"] == 3


def test_validation_and_empty_input() -> None:
    with pytest.raises(ValueError):
        build_features(make_history(days=1), horizon_min=20)
    empty = build_features(make_history(days=1).iloc[0:0])
    assert empty.empty and TARGET_COLUMN in empty.columns
