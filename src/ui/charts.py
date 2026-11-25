"""Charts with Streamlit's built-in ``st.line_chart`` and ``st.vega_lite_chart``.

The ``*_data`` functions prepare a small pandas table (easy to test), the
``render_*`` functions show it.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from src.models import ModeComparison

MEASURED_COLOR = "#2a78d6"  # blue
FORECAST_COLOR = "#86b6ef"  # light blue
BAR_COLOR = "#2a78d6"  # one colour for both bar charts: the axis names the mode


# --- Occupancy: last 24 h + forecast ------------------------------------------------


def occupancy_chart_data(history: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Table for ``st.line_chart``: one row per time, columns "Gemessen" and "Prognose".

    Args:
        history: Columns ``time`` and ``free_spots`` (measurements, may be empty).
        forecast: Columns ``time`` and ``free_spots`` (predictions from now on).
    """
    measured = history.set_index("time")["free_spots"].rename("Gemessen")
    predicted = forecast.set_index("time")["free_spots"].rename("Prognose")
    if not measured.empty and not predicted.empty:
        # Start the forecast at the last measurement, so the two lines touch.
        predicted = pd.concat([measured.tail(1).rename("Prognose"), predicted])
        predicted = predicted[~predicted.index.duplicated(keep="last")]

    table = pd.concat([measured, predicted], axis=1).sort_index()
    if isinstance(table.index, pd.DatetimeIndex) and table.index.tz is not None:
        table.index = table.index.tz_localize(None)  # keep the Zurich clock time in the chart
    table.index.name = "Zeit"
    return table


def render_occupancy_chart(history: pd.DataFrame, forecast: pd.DataFrame, arrival: datetime, arrival_free: int) -> None:
    """Line chart (measured + forecast) and a caption for the arrival time."""
    table = occupancy_chart_data(history, forecast)
    if table.empty:
        st.info("Für dieses Parkhaus gibt es noch keine Daten.")
        return
    colors = [MEASURED_COLOR if column == "Gemessen" else FORECAST_COLOR for column in table.columns]
    st.line_chart(table, y_label="Freie Plätze", x_label="", color=colors, height=300)
    st.caption(f"Ankunft am Parkhaus ca. {arrival:%H:%M} Uhr: ~{arrival_free} Plätze frei (Prognose).")


# --- Car vs. public transport -----------------------------------------------------------


def mode_chart_data(comparison: ModeComparison) -> pd.DataFrame:
    """One row per mode: total time (one way, minutes) and cost (round trip, CHF).

    Unavailable options (no garage / no connection) get NaN, so no bar is drawn.
    """
    rows = {}
    for label, option in (("Auto", comparison["car"]), ("ÖV", comparison["transit"])):
        rows[label] = {
            "Minuten": option["total_min"] if option["available"] else np.nan,
            "CHF": option["cost_chf"] if option["available"] else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _render_bars(table: pd.DataFrame, column: str) -> None:
    """Upright bar chart of one column.

    Not ``st.bar_chart``: that one turns the labels "Auto" and "ÖV" by 90 degrees.
    """
    data = table[[column]].rename_axis("Option").reset_index()
    spec = {
        "mark": {"type": "bar", "color": BAR_COLOR},
        "encoding": {
            "x": {"field": "Option", "type": "nominal", "title": None, "axis": {"labelAngle": 0}},
            "y": {"field": column, "type": "quantitative", "title": None},
            "tooltip": [{"field": "Option", "type": "nominal"}, {"field": column, "type": "quantitative"}],
        },
    }
    st.vega_lite_chart(data, spec, height=250)


def render_mode_comparison(comparison: ModeComparison) -> None:
    """Two small bar charts (time and cost) plus the exact numbers as text."""
    table = mode_chart_data(comparison)
    time_col, cost_col = st.columns(2)
    with time_col:
        st.caption("Gesamtzeit (einfach), min")
        _render_bars(table, "Minuten")
    with cost_col:
        st.caption("Kosten (hin & zurück), CHF")
        _render_bars(table, "CHF")

    lines = []
    for label, missing in (("Auto", "kein Parkhaus"), ("ÖV", "keine Verbindung")):
        minutes, chf = table.loc[label, "Minuten"], table.loc[label, "CHF"]
        lines.append(f"**{label}:** {missing}" if np.isnan(minutes) else f"**{label}:** {minutes:.0f} min · CHF {chf:.2f}")
    st.markdown(" &nbsp;|&nbsp; ".join(lines))
