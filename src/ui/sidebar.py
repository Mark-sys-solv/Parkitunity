"""Sidebar with the trip inputs."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TypedDict

import streamlit as st

from src.models import Location
from src.ui import backend

_DEFAULTS = {"origin_query": "Bahnhof Oerlikon, Zürich", "destination_query": "Paradeplatz, Zürich"}
MAX_PRICE_CHF = 8.0


class TripInputs(TypedDict):
    origin_query: str  # text typed as start point
    destination_query: str  # text typed as destination
    origin: Location | None  # geocoded start point (None = not found / empty)
    destination: Location | None  # geocoded destination (None = not found / empty)
    departure_time: datetime  # when the trip starts; timezone-aware, Europe/Zurich
    max_price_chf: float
    show_transit: bool


def render_sidebar() -> TripInputs:
    """Render all inputs and return them."""
    _init_state()
    with st.sidebar:
        st.header("Deine Fahrt")
        st.text_input("Startpunkt", key="origin_query", placeholder="z.B. Bahnhof Oerlikon, Zürich")
        origin = _find("origin_query", "den Startpunkt")
        st.text_input("Ziel", key="destination_query", placeholder="z.B. Kunsthaus Zürich")
        destination = _find("destination_query", "das Ziel")
        st.caption("Adresse, Ort oder Koordinaten (z.B. 47.3697, 8.5391)")

        st.subheader("Abfahrt")
        date_col, time_col = st.columns(2)
        date_col.date_input("Datum", key="departure_date", format="DD.MM.YYYY")
        time_col.time_input("Uhrzeit", key="departure_clock", step=timedelta(minutes=5))
        st.button("↺ Jetzt", on_click=reset_departure_time, type="tertiary")
        departure = datetime.combine(
            st.session_state["departure_date"], st.session_state["departure_clock"], tzinfo=backend.TZ
        )

        st.subheader("Parkieren")
        max_price = st.slider(
            "Max. Preis pro Stunde",
            min_value=1.0,
            max_value=MAX_PRICE_CHF,
            value=MAX_PRICE_CHF,
            step=0.5,
            format="CHF %.2f",
        )
        if backend.prices_are_placeholders():
            st.caption("Preise sind vorläufige Platzhalter.")

        st.subheader("Vergleich")
        show_transit = st.toggle("ÖV-Vergleich anzeigen", value=True)

    return TripInputs(
        origin_query=st.session_state["origin_query"],
        destination_query=st.session_state["destination_query"],
        origin=origin,
        destination=destination,
        departure_time=departure,
        max_price_chf=max_price,
        show_transit=show_transit,
    )


def _find(key: str, what: str) -> Location | None:
    """Geocode the text of input ``key``; show a clear message below it if that fails."""
    text = st.session_state[key].strip()
    if not text:
        st.warning(f"Bitte gib {what} ein.", icon="📍")
        return None
    with st.spinner("Suche Adresse …"):
        location = backend.geocode(text)
    if location is None:
        st.error(
            f"„{text}“ wurde nicht gefunden. Prüfe die Schreibweise oder ergänze den Ort, "
            "z.B. „Bahnhofstrasse 1, Zürich“.",
            icon="📍",
        )
    return location


def reset_departure_time() -> None:
    """Set the departure to now, rounded up to the next 5 minutes."""
    current = backend.now()
    rounded = (current + timedelta(minutes=4, seconds=59)).replace(second=0, microsecond=0)
    rounded -= timedelta(minutes=rounded.minute % 5)
    st.session_state["departure_date"] = rounded.date()
    st.session_state["departure_clock"] = rounded.time().replace(tzinfo=None)


def _init_state() -> None:
    """Defaults live in session state so callbacks can change widget values."""
    for key, value in _DEFAULTS.items():
        st.session_state.setdefault(key, value)
    if "departure_date" not in st.session_state:
        reset_departure_time()
