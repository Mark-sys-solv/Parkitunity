"""Sidebar with user inputs."""

from __future__ import annotations

from datetime import datetime

import streamlit as st


def render_sidebar() -> dict:
    """Render the input widgets.

    Returns:
        Dict with keys ``origin`` (str), ``destination`` (str), ``time`` (datetime).
    """
    # TODO: add parking duration, preference sliders (time vs. cost).
    # TODO: address autocomplete via transport.opendata.ch /locations.
    st.sidebar.header("Trip")
    origin = st.sidebar.text_input("Start", "Zürich Oerlikon")
    destination = st.sidebar.text_input("Destination", "Zürich Paradeplatz")
    now = datetime.now()
    date = st.sidebar.date_input("Date", now.date())
    time = st.sidebar.time_input("Departure time", now.time().replace(second=0, microsecond=0))
    return {
        "origin": origin,
        "destination": destination,
        "time": datetime.combine(date, time),
    }
