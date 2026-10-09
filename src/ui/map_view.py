"""Map of recommended garages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.models import Location


def render_map(ranked_garages: list[dict], destination: Location | None = None) -> None:
    """Show garages on a map.

    Args:
        ranked_garages: Output of ``parking_score.rank_garages``.
        destination: Optional destination marker.
    """
    # TODO: switch to pydeck for coloured markers (by free spots) and tooltips.
    rows = [
        {"lat": r["garage"].location.lat, "lon": r["garage"].location.lon}
        for r in ranked_garages
    ]
    if destination is not None:
        rows.append({"lat": destination.lat, "lon": destination.lon})
    st.map(pd.DataFrame(rows), zoom=13)
