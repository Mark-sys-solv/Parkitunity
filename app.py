"""Parkitunity – Streamlit entry point.

Run with:  streamlit run app.py
This file only composes the UI; logic lives in ``src/``, cached data access
and dummy data in ``src/ui/backend.py``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ml.predict import model_info
from src.recommendation.candidates import COLUMNS
from src.recommendation.mode_compare import compare_modes
from src.recommendation.parking_score import best_garage, rank_garages
from src.ui import backend
from src.ui.charts import render_mode_comparison
from src.ui.forecast import render_forecast
from src.ui.garage_cards import render_garage_cards
from src.ui.map_view import render_garage_info, render_garage_table, render_legend, render_map
from src.ui.mode_cards import render_mode_cards
from src.ui.priority import render_priority
from src.ui.sidebar import render_sidebar
from src.ui.styles import inject_styles

st.set_page_config(page_title="Parkitunity", page_icon="🅿️", layout="wide")
inject_styles()

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

priority = render_priority()

with st.spinner("Berechne Prognosen …"):
    candidates = backend.candidates(
        destination.lat, destination.lon, departure.isoformat(),
        origin.lat if origin else None, origin.lon if origin else None,
    ) if destination else pd.DataFrame(columns=COLUMNS)
# Garages above the price limit are not ranked; they are only shown in grey.
too_expensive = candidates["price_per_hour"] > inputs["max_price_chf"]
ranking = rank_garages(candidates[~too_expensive], priority, inputs["max_walk_min"], inputs["enough_spots"])
st.session_state["ranking"] = ranking
best = best_garage(ranking)
best_id = best["garage_id"] if best else None
# All garages to show: the ranking first, then the ones above the price limit.
shown = pd.concat([ranking, candidates[too_expensive]], ignore_index=True) if too_expensive.any() else ranking

comparison = None
if inputs["show_transit"] and destination and origin:
    with st.spinner("Vergleiche mit dem ÖV …"):
        transit = backend.transit(origin, destination, departure)
    comparison = compare_modes(best, transit, inputs["parking_hours"], priority)

# --- Top: best garages, car or public transport ------------------------------------
if destination:
    if best and best["walk_min"] > inputs["max_walk_min"]:
        st.warning(f"Kein Parkhaus mit genügend freien Plätzen liegt höchstens {inputs['max_walk_min']} min zu Fuss vom Ziel – "
                   f"der Vorschlag ist {best['walk_min']} min entfernt.", icon="🚶")
    render_garage_cards()
if comparison:
    render_mode_cards(comparison, origin, destination, inputs["origin_query"], inputs["destination_query"],
                      inputs["parking_hours"])

# --- Middle: map + table ------------------------------------------------------------
clicked_id = render_map(shown, destination, best_id, inputs["max_price_chf"], origin)
render_legend()
by_id = {g["garage_id"]: g for g in shown.to_dict("records")}
if clicked_id in by_id:
    render_garage_info(by_id[clicked_id], best_id, inputs["max_price_chf"])
if by_id:
    with st.expander("Alle Parkhäuser als Tabelle"):
        render_garage_table(shown, best_id, inputs["max_price_chf"])

# --- Bottom: charts -------------------------------------------------------------------
if by_id:
    # Follow the recommendation whenever the trip or the priority changes; keep a manual choice otherwise.
    ids = list(by_id)
    context = (destination.lat, destination.lon, departure.isoformat(), inputs["max_price_chf"],
               inputs["origin_query"], priority, inputs["max_walk_min"], inputs["enough_spots"])
    if st.session_state.get("_selection_context") != context or st.session_state.get("selected_garage") not in ids:
        st.session_state["selected_garage"] = best_id or ids[0]
        st.session_state["_selection_context"] = context
    # A newly clicked garage on the map is also shown in the occupancy chart.
    if clicked_id in by_id and clicked_id != st.session_state.get("_last_map_click"):
        st.session_state["selected_garage"] = clicked_id
    st.session_state["_last_map_click"] = clicked_id

    render_forecast(by_id)
if comparison:
    st.subheader("Auto oder ÖV?")
    render_mode_comparison(comparison)

# --- Footer -----------------------------------------------------------------------------
notes = ["Belegung: PLS Zürich via ParkenDD (CC-0)", "ÖV: transport.opendata.ch"]
if backend.prices_are_placeholders():
    notes.append("Preise: Platzhalter")
notes.append("Adresssuche: © OpenStreetMap-Mitwirkende (Nominatim)")
if backend.uses_synthetic_db():
    notes.append("Messverlauf: synthetische Daten")
st.divider()
st.caption(" · ".join(notes))
