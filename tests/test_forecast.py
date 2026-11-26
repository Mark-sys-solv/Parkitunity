"""Tests for the forecast component (invented data)."""

import pandas as pd
import pytest

from src import config
from src.ui.forecast import chart_data, curve_hours, forecast_chart, spots_text

START = pd.Timestamp("2026-11-25 11:30", tz=config.ZURICH_TZ)
CURVE = pd.DataFrame({"timestamp": pd.date_range(START, periods=25, freq="15min"), "free_pred": range(130, 105, -1)})


def test_arrival_is_a_point_of_the_line_and_times_are_zurich_clock_time() -> None:
    data, visible = chart_data(CURVE, START + pd.Timedelta(minutes=14), 129)
    assert visible and len(data) == 26
    assert list(data["time"].iloc[:3]) == ["2026-11-25T11:30:00", "2026-11-25T11:44:00", "2026-11-25T11:45:00"]
    assert data["free"].iloc[1] == 129


def test_arrival_exactly_on_a_step_is_not_doubled() -> None:
    data, _ = chart_data(CURVE, START + pd.Timedelta(minutes=15), 129)
    assert len(data) == 25


@pytest.mark.parametrize("minutes", [-10, 7 * 60])
def test_arrival_outside_of_the_chart(minutes: int) -> None:
    data, visible = chart_data(CURVE, START + pd.Timedelta(minutes=minutes), 1)
    assert not visible and len(data) == 25


@pytest.mark.parametrize("minutes, hours", [(14, 6), (5 * 60, 6), (8 * 60 + 5, 10), (3 * 24 * 60, 24)])
def test_chart_gets_longer_for_a_late_arrival(minutes: int, hours: int) -> None:
    assert curve_hours(START + pd.Timedelta(minutes=minutes), START) == hours


def test_blue_dot_only_with_live_data() -> None:
    arrival = START + pd.Timedelta(minutes=14)
    with_live, _ = forecast_chart(CURVE, 130, arrival, 129)
    without_live, _ = forecast_chart(CURVE, None, arrival, 129)
    assert len(with_live.layer) == len(without_live.layer) + 1
    assert "Deine Ankunft" in with_live.to_json()


def test_spots_text() -> None:
    assert (spots_text(194), spots_text(1)) == ("194 freie Plätze", "1 freier Platz")
