"""Parkitunity – Streamlit entry point.

Run with:  streamlit run app.py
This file only composes the UI; logic lives in ``src/``, cached data access
and dummy data in ``src/ui/backend.py``.
"""

from __future__ import annotations

import streamlit as st

from src.config import PARKING_DURATION_H
from src.ml.predict import model_info
from src.ui import backend
from src.ui.charts import render_mode_comparison, render_occupancy_chart
from src.ui.map_view import render_garage_info, render_garage_table, render_legend, render_map
from src.ui.navigation import recommended_route
from src.ui.recommendation_card import build_recommendation, render_recommendation
from src.ui.sidebar import render_sidebar

st.set_page_config(page_title="Parkitunity", page_icon="🅿️", layout="wide")

inputs = render_sidebar()
departure = inputs["departure_time"]

st.title("🅿️ Parkitunity")
st.caption("Welches Parkhaus in Zürich – oder doch lieber mit dem ÖV?")

model = model_info()
if model and model["data_source"] == "synthetic":
    st.warning("Die Prognosen stammen von einem Modell, das mit **synthetischen** Daten trainiert wurde – nur zur Entwicklung.", icon="⚠️")

# --- Inputs (start point and destination are geocoded in the sidebar) --------------
origin, destination = inputs["origin"], inputs["destination"]
if destination is None:
    st.info("Gib in der Seitenleiste ein Ziel ein, das gefunden wird – dann erscheint die Empfehlung.", icon="📍")

with st.spinner("Berechne Prognosen …"):
    ranked = backend.ranked_garages(
        destination.lat, destination.lon, departure.isoformat(),
        origin.lat if origin else None, origin.lon if origin else None,
    ) if destination else []
# Same garage as compare_modes picks for the car (scores do not depend on the filter).
eligible = [r for r in ranked if r["garage"].hourly_rate_chf is None or r["garage"].hourly_rate_chf <= inputs["max_price_chf"]]
best = eligible[0] if eligible else None
best_id = best["garage"].id if best else None

comparison = None
if inputs["show_transit"] and destination and origin:
    with st.spinner("Vergleiche mit dem ÖV …"):
        comparison = backend.mode_comparison(
            origin.lat, origin.lon, destination.lat, destination.lon, departure.isoformat(), inputs["max_price_chf"]
        )

# --- Top: recommendation ----------------------------------------------------------
if destination:
    render_recommendation(build_recommendation(best, comparison))
    route = recommended_route(origin, destination, best, comparison)
    if route is not None:
        url, caption = route
        st.link_button("Route in Google Maps öffnen", url, icon="🗺️")
        st.caption(caption)

# --- Middle: map + table ------------------------------------------------------------
clicked_id = render_map(ranked, destination, best_id, inputs["max_price_chf"], origin)
render_legend()
by_id = {r["garage"].id: r for r in ranked}
if clicked_id in by_id:
    render_garage_info(by_id[clicked_id], best_id, inputs["max_price_chf"])
if ranked:
    with st.expander("Alle Parkhäuser als Tabelle"):
        render_garage_table(ranked, best_id, inputs["max_price_chf"])

# --- Bottom: charts -------------------------------------------------------------------
if ranked:
    # Follow the recommendation whenever the trip changes; keep a manual choice otherwise.
    ids = [r["garage"].id for r in ranked]
    context = (destination.lat, destination.lon, departure.isoformat(), inputs["max_price_chf"], inputs["origin_query"])
    if st.session_state.get("_selection_context") != context or st.session_state.get("selected_garage_id") not in ids:
        st.session_state["selected_garage_id"] = best_id or ids[0]
        st.session_state["_selection_context"] = context
    # A newly clicked garage on the map is also shown in the occupancy chart.
    if clicked_id in by_id and clicked_id != st.session_state.get("_last_map_click"):
        st.session_state["selected_garage_id"] = clicked_id
    st.session_state["_last_map_click"] = clicked_id

    left, right = st.columns([3, 2], gap="large") if comparison else (st.container(), None)
    with left:
        st.subheader("Belegung")
        selected_id = st.selectbox(
            "Parkhaus",
            options=ids,
            key="selected_garage_id",
            format_func=lambda gid: backend.display_name(by_id[gid]["garage"]),
            label_visibility="collapsed",
        )
        selected = by_id[selected_id]
        at_garage = selected["arrival_at_garage"]
        start, end = backend.forecast_window(at_garage)
        history = backend.occupancy_history(selected_id)
        forecast = backend.occupancy_forecast(selected_id, start.isoformat(), end.isoformat())
        render_occupancy_chart(history, forecast, at_garage, selected["predicted_free"])
        if not forecast.empty and history.empty:
            st.caption("Für die letzten 24 h sind noch keine Messwerte gespeichert – nur die Prognose wird gezeigt.")
    if right is not None:
        with right:
            st.subheader("Auto oder ÖV?")
            render_mode_comparison(comparison)
            st.caption(f"Auto: Fahrt + Parkplatzsuche + Fussweg; Kosten: {PARKING_DURATION_H:g} h Parkieren "
                       f"+ km-Kosten. ÖV: inkl. Wartezeit; Ticketpreis vereinfacht (ZVV).")

# --- Footer -----------------------------------------------------------------------------
notes = ["Belegung: PLS Zürich via ParkenDD (CC-0)", "ÖV: transport.opendata.ch"]
if backend.prices_are_placeholders():
    notes.append("Preise: Platzhalter")
notes.append("Adresssuche: © OpenStreetMap-Mitwirkende (Nominatim)")
if backend.uses_synthetic_db():
    notes.append("Messverlauf: synthetische Daten")
st.divider()
st.caption(" · ".join(notes))
