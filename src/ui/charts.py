"""Charts for occupancy forecasts and the mode comparison."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from src.ml.predict import predict
from src.models import ModeComparison


def render_occupancy_chart(garage_id: str, start: datetime, hours: int = 12) -> None:
    """Line chart of predicted free spots over the next ``hours`` hours."""
    # TODO: overlay historical average from the database.
    times = [start + timedelta(hours=h) for h in range(hours)]
    df = pd.DataFrame({"time": times, "free spots": [predict(garage_id, t) for t in times]})
    st.line_chart(df, x="time", y="free spots")


def render_mode_comparison(comparison: ModeComparison) -> None:
    """Show car vs. transit as metrics and a bar chart."""
    # TODO: nicer chart (altair/plotly), highlight recommended mode.
    car, transit = comparison["car"], comparison["transit"]
    col1, col2 = st.columns(2)
    col1.metric("🚗 Car + parking", f"{car['duration_min']:.0f} min", f"CHF {car['cost_chf']:.2f}", delta_color="off")
    col2.metric("🚆 Public transport", f"{transit['duration_min']:.0f} min", f"CHF {transit['cost_chf']:.2f}", delta_color="off")
    df = pd.DataFrame(
        {"duration (min)": [car["duration_min"], transit["duration_min"]]},
        index=["Car", "Transit"],
    )
    st.bar_chart(df)
