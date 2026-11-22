"""Map of Zurich with all garages (``st.map``) and a table with the details.

``st.map`` draws one coloured dot per row of a DataFrame. It has no popups,
so the details of each garage are shown in a table below the map instead.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import ZURICH_CENTER
from src.models import Garage, Location
from src.ui import backend

OVER_BUDGET_COLOR = "#898781"  # grey: more expensive than the user's limit
DESTINATION_COLOR = "#0b0b0b"  # black dot: the destination
# Dot sizes are radii in metres on the map.
GARAGE_SIZE_M = 40
RECOMMENDED_SIZE_M = 90  # the recommended garage gets a bigger dot
DESTINATION_SIZE_M = 50


def is_over_budget(garage: Garage, max_price_chf: float) -> bool:
    """True if the garage's hourly rate is known and above the user's limit."""
    return garage.hourly_rate_chf is not None and garage.hourly_rate_chf > max_price_chf


def map_data(
    ranked: list[dict],
    destination: Location | None,
    recommended_id: str | None,
    max_price_chf: float,
) -> pd.DataFrame:
    """One row per dot with the columns ``st.map`` needs: lat, lon, color, size.

    Args:
        ranked: Output of ``rank_garages`` (garage, predicted_free, ...).
        destination: Drawn as a black dot (None = no valid destination).
        recommended_id: This garage gets a bigger dot.
        max_price_chf: Garages above this hourly rate are grey.
    """
    rows = []
    for r in ranked:
        garage = r["garage"]
        status = backend.availability(r["predicted_free"], garage.capacity)
        color = OVER_BUDGET_COLOR if is_over_budget(garage, max_price_chf) else backend.STATUS_COLORS[status]
        size = RECOMMENDED_SIZE_M if garage.id == recommended_id else GARAGE_SIZE_M
        rows.append({"lat": garage.location.lat, "lon": garage.location.lon, "color": color, "size": size})
    if destination is not None:
        rows.append({"lat": destination.lat, "lon": destination.lon, "color": DESTINATION_COLOR, "size": DESTINATION_SIZE_M})
    if not rows:
        # Nothing to show: one invisible dot keeps the map centred on Zurich.
        rows.append({"lat": ZURICH_CENTER[0], "lon": ZURICH_CENTER[1], "color": "#00000000", "size": 1})
    return pd.DataFrame(rows, columns=["lat", "lon", "color", "size"])


def render_map(ranked: list[dict], destination: Location | None, recommended_id: str | None, max_price_chf: float) -> None:
    """Show the map (Streamlit zooms to fit all dots)."""
    data = map_data(ranked, destination, recommended_id, max_price_chf)
    st.map(data, latitude="lat", longitude="lon", color="color", size="size", zoom=13)


def garage_table(ranked: list[dict], recommended_id: str | None, max_price_chf: float) -> pd.DataFrame:
    """The details that used to be in map popups: one row per garage, best first."""
    rows = []
    for r in ranked:
        garage = r["garage"]
        status = backend.availability(r["predicted_free"], garage.capacity)
        if garage.id == recommended_id:
            note = "★ Empfehlung"
        elif is_over_budget(garage, max_price_chf):
            note = "über Preislimit"
        else:
            note = ""
        rows.append(
            {
                "Parkhaus": backend.display_name(garage),
                "Verfügbarkeit": backend.STATUS_LABELS[status],
                "Frei (Prognose)": r["predicted_free"],
                "CHF/h": garage.hourly_rate_chf,
                "Fussweg (min)": round(r["walking_min"]),
                "Gesamt (min)": round(r["total_min"]),
                "Hinweis": note,
            }
        )
    return pd.DataFrame(rows)


def render_garage_table(ranked: list[dict], recommended_id: str | None, max_price_chf: float) -> None:
    st.dataframe(garage_table(ranked, recommended_id, max_price_chf), hide_index=True, width="stretch")


def render_legend() -> None:
    """Colour legend below the map (labels, so colour never carries meaning alone)."""
    dot = '<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{};margin:0 6px 0 14px;vertical-align:middle"></span>'
    items = [dot.format(backend.STATUS_COLORS[s]) + backend.STATUS_LABELS[s] for s in ("good", "tight", "full")]
    items.append(dot.format(OVER_BUDGET_COLOR) + "Über Preislimit")
    items.append(dot.format(DESTINATION_COLOR) + "Ziel")
    items.append('<span style="margin-left:14px">Grosser Punkt = Empfehlung</span>')
    st.markdown(
        f'<div style="font-size:0.85rem;opacity:0.8;margin:-0.5rem 0 0.5rem -14px">{"".join(items)}</div>',
        unsafe_allow_html=True,
    )
