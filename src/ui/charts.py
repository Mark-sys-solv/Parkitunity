"""Bar charts car vs. public transport (``st.vega_lite_chart``).

The ``*_data`` functions prepare a small pandas table (easy to test), the
``render_*`` functions show it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.models import ModeComparison

BAR_COLOR = "#2a78d6"  # one colour for both bar charts: the axis names the mode


# --- Car vs. public transport -----------------------------------------------------------


def mode_chart_data(comparison: ModeComparison) -> pd.DataFrame:
    """One row per mode: total time (one way, minutes) and cost (there and back, CHF).

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
