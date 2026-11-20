"""Parkitunity – Streamlit entry point.

Run with:  streamlit run app.py
This file only composes the UI; all logic lives in ``src/``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data import routing
from src.recommendation.mode_compare import compare_modes
from src.recommendation.parking_score import rank_garages
from src.ui.charts import render_mode_comparison, render_occupancy_chart
from src.ui.map_view import render_map
from src.ui.sidebar import render_sidebar

st.set_page_config(page_title="Parkitunity", page_icon="🅿️", layout="wide")
st.title("🅿️ Parkitunity")
st.caption("Which garage in Zurich should you head for, and would public transport be better?")

inputs = render_sidebar()
destination = routing.geocode(inputs["destination"])
ranked = rank_garages(destination, inputs["time"])
comparison = compare_modes(inputs["origin"], inputs["destination"], inputs["time"])

left, right = st.columns([3, 2])
with left:
    st.subheader("Recommended garages")
    render_map(ranked, destination)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Garage": r["garage"].name,
                    "Predicted free": r["predicted_free"],
                    "Walk (min)": round(r["walking_min"], 1),
                    "CHF/h": r["garage"].hourly_rate_chf,
                    "Score": round(r["score"], 2),
                }
                for r in ranked
            ]
        ),
        hide_index=True,
        width="stretch",
    )

with right:
    st.subheader("Car or public transport?")
    winner = "🚆 Public transport" if comparison["recommendation"] == "transit" else "🚗 Car"
    st.success(f"Recommendation: **{winner}**")
    render_mode_comparison(comparison)
    if ranked:
        st.subheader(f"Forecast: {ranked[0]['garage'].name}")
        render_occupancy_chart(ranked[0]["garage"].id, inputs["time"])

st.info(
    "⚠️ Prototype – parking occupancy and transit connections are live; "
    "geocoding, prices and occupancy forecasts are still placeholders."
)
