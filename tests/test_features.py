from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ml.features import FEATURES, TARGET, add_features, build_table, easter_sunday, zurich_holidays

START = pd.Timestamp("2026-12-17 23:00", tz="UTC")  # = 18 Dec 00:00 in Zurich


def make_history(days: int = 2, garage_id: str = "g1", capacity: int = 1000) -> pd.DataFrame:
    """Regular 15-min measurements where free_spots = slot number (easy to check)."""
    slots = pd.date_range(START, periods=days * 96, freq="15min")
    return pd.DataFrame({
        "timestamp": slots + pd.Timedelta(minutes=3),  # a few minutes after each slot starts
        "garage_id": garage_id,
        "free_spots": np.arange(len(slots)) % capacity,
        "capacity": capacity,
    })


def test_one_row_per_slot_and_target_is_1_hour_later() -> None:
    table = build_table(make_history())
    assert list(table.columns) == ["time", "garage", "capacity", "free", TARGET]
    assert len(table) == 2 * 96
    assert table["free"].iloc[10] == 10 and table[TARGET].iloc[10] == 14  # 4 x 15 min later
    assert table[TARGET].iloc[-4:].isna().all()  # the future is unknown


def test_garages_do_not_mix() -> None:
    table = build_table(pd.concat([make_history(garage_id="a"), make_history(garage_id="b", capacity=50)]))
    assert table.loc[table["garage"] == "a", TARGET].iloc[-4:].isna().all()  # not the first values of "b"


def test_target_is_never_taken_across_a_gap() -> None:
    history = make_history()
    gap = (history["timestamp"] > START + pd.Timedelta(hours=10)) & (history["timestamp"] < START + pd.Timedelta(hours=12))
    table = build_table(history[~gap]).dropna(subset=["free", TARGET])
    assert ((table[TARGET] - table["free"]) == 4).all()  # always exactly 1 hour apart
    assert not table["time"].between(START + pd.Timedelta(hours=10), START + pd.Timedelta(hours=11, minutes=59)).any()


def test_features_use_zurich_time_of_the_row() -> None:
    history = make_history(days=9)
    rows = add_features(build_table(history), ["g0", "g1"]).set_index("time")
    row = rows.loc[pd.Timestamp("2026-12-24 23:00", tz="UTC")]  # = 25 Dec 00:00 in Zurich
    assert (row["hour"], row["weekday"], row["is_holiday"], row["garage_id"]) == (0, 4, 1, 1)
    assert set(FEATURES) <= set(rows.columns)


def test_weather_is_merged_by_hour_and_keeps_every_row() -> None:
    table = build_table(make_history(days=1))
    weather = pd.DataFrame({
        "time": pd.date_range(START, periods=10, freq="h"),  # shorter than the table
        "temperature_c": np.arange(10.0),
        "precipitation_mm": 0.5,
    })
    rows = add_features(table, ["g1"], weather)
    assert len(rows) == len(table)
    assert rows["temperature_c"].iloc[9] == 2  # 02:15 -> weather of 02:00
    assert rows["rain_mm"].iloc[0] == 0.5 and np.isnan(rows["rain_mm"].iloc[-1])


def test_empty_input() -> None:
    table = build_table(make_history().iloc[0:0])
    assert table.empty and TARGET in table.columns


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
