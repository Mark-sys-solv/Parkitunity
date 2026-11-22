"""Map of Zurich with all garages, the start point and the destination (``st.map``).

``st.map`` can only draw coloured dots (one per row of a DataFrame) – no icons,
no popups. Therefore:
- the details of each garage are shown in a table below the map, and
- start and destination are drawn as several dots on the same spot, biggest
  first. The smaller dots lie on top, so it looks like a ring or a target.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.config import ZURICH_CENTER
from src.models import Garage, Location
from src.ui import backend

OVER_BUDGET_COLOR = "#898781"  # grey: more expensive than the user's limit
START_COLOR = "#2563eb"  # blue: start point
DESTINATION_COLOR = "#0b0b0b"  # black: destination
RING_COLOR = "#ffffff"  # white ring, so start and destination stand out on light and dark maps

# Dot sizes are radii in screen pixels (converted to metres for st.map in render_map).
GARAGE_SIZE_PX = 4
RECOMMENDED_SIZE_PX = 8  # the recommended garage gets a bigger dot
# Start and destination as (colour, radius in pixels), from the outside in.
START_MARKER = [(RING_COLOR, 10), (START_COLOR, 7)]  # blue dot with a white ring
DESTINATION_MARKER = [(DESTINATION_COLOR, 10), (RING_COLOR, 7), (DESTINATION_COLOR, 4)]  # target

# Used to choose a zoom level at which all dots fit on the map.
MAP_HEIGHT_PX = 500  # default height of st.map
MAP_WIDTH_PX = 600  # the real width depends on the screen – a careful guess
MARGIN_PX = 30  # free space around the outermost dots
EARTH_CIRCUMFERENCE_M = 40_075_016
METRES_PER_DEGREE = 111_320  # length of one degree of latitude
MIN_ZOOM, MAX_ZOOM = 8, 14  # 8 ≈ whole canton, 14 ≈ a few streets


def is_over_budget(garage: Garage, max_price_chf: float) -> bool:
    """True if the garage's hourly rate is known and above the user's limit."""
    return garage.hourly_rate_chf is not None and garage.hourly_rate_chf > max_price_chf


def marker_rows(location: Location, layers: list[tuple[str, int]]) -> list[dict]:
    """Several dots on the same spot (biggest first), e.g. a dot with a ring."""
    return [{"lat": location.lat, "lon": location.lon, "color": color, "size": size} for color, size in layers]


def map_data(
    ranked: list[dict],
    destination: Location | None,
    recommended_id: str | None,
    max_price_chf: float,
    origin: Location | None = None,
) -> pd.DataFrame:
    """One row per dot with the columns lat, lon, color, size (radius in pixels).

    Args:
        ranked: Output of ``rank_garages`` (garage, predicted_free, ...).
        destination: Drawn as a black-and-white target (None = no valid destination).
        recommended_id: This garage gets a bigger dot.
        max_price_chf: Garages above this hourly rate are grey.
        origin: Start point, drawn as a blue dot with a white ring (None = unknown).
    """
    rows = []
    for r in ranked:
        garage = r["garage"]
        status = backend.availability(r["predicted_free"], garage.capacity)
        color = OVER_BUDGET_COLOR if is_over_budget(garage, max_price_chf) else backend.STATUS_COLORS[status]
        size = RECOMMENDED_SIZE_PX if garage.id == recommended_id else GARAGE_SIZE_PX
        rows.append({"lat": garage.location.lat, "lon": garage.location.lon, "color": color, "size": size})
    # Start and destination come last, so they are drawn on top of the garages.
    if origin is not None:
        rows.extend(marker_rows(origin, START_MARKER))
    if destination is not None:
        rows.extend(marker_rows(destination, DESTINATION_MARKER))
    if not rows:
        # Nothing to show: one invisible dot keeps the map centred on Zurich.
        rows.append({"lat": ZURICH_CENTER[0], "lon": ZURICH_CENTER[1], "color": "#00000000", "size": 1})
    return pd.DataFrame(rows, columns=["lat", "lon", "color", "size"])


def metres_per_pixel(zoom: float, lat: float) -> float:
    """How many metres one screen pixel covers at this zoom level.

    st.map draws with deck.gl: there the whole world is 512 pixels wide at
    zoom 0, and twice as wide with every zoom level.
    """
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / (512 * 2**zoom)


def fit_zoom(data: pd.DataFrame) -> float:
    """Zoom level at which all dots fit on the map.

    st.map puts the middle of all dots in the centre of the map; the zoom we
    choose ourselves, so that start, destination and all garages are visible.
    """
    lat = data["lat"].mean()
    height_m = (data["lat"].max() - data["lat"].min()) * METRES_PER_DEGREE
    width_m = (data["lon"].max() - data["lon"].min()) * METRES_PER_DEGREE * math.cos(math.radians(lat))
    # Metres per pixel we need so that everything fits (at least 1, for a single dot).
    needed = max(height_m / (MAP_HEIGHT_PX - 2 * MARGIN_PX), width_m / (MAP_WIDTH_PX - 2 * MARGIN_PX), 1.0)
    zoom = math.log2(metres_per_pixel(0, lat) / needed)
    return round(min(max(zoom, MIN_ZOOM), MAX_ZOOM), 1)


def render_map(
    ranked: list[dict],
    destination: Location | None,
    recommended_id: str | None,
    max_price_chf: float,
    origin: Location | None = None,
) -> None:
    """Show the map with all garages, the start point and the destination."""
    data = map_data(ranked, destination, recommended_id, max_price_chf, origin)
    zoom = fit_zoom(data)
    # st.map wants radii in metres: convert, so the dots have the same size at every zoom.
    data["size"] = data["size"] * metres_per_pixel(zoom, data["lat"].mean())
    st.map(data, latitude="lat", longitude="lon", color="color", size="size", zoom=zoom)


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


def _swatch(background: str, size_px: int = 10) -> str:
    """A small round colour sample for the legend."""
    return (f'<span style="display:inline-block;width:{size_px}px;height:{size_px}px;border-radius:50%;'
            f'background:{background};margin:0 6px 0 14px;vertical-align:middle"></span>')


def render_legend() -> None:
    """Colour legend below the map (labels, so colour never carries meaning alone)."""
    items = [_swatch(backend.STATUS_COLORS[s]) + backend.STATUS_LABELS[s] for s in ("good", "tight", "full")]
    items.append(_swatch(OVER_BUDGET_COLOR) + "Über Preislimit")
    items.append(_swatch(START_COLOR, 12) + "Startpunkt")
    target = f"radial-gradient(circle, {DESTINATION_COLOR} 0 30%, {RING_COLOR} 31% 62%, {DESTINATION_COLOR} 63%)"
    items.append(_swatch(target, 12) + "Ziel")
    items.append('<span style="margin-left:14px">Grosser Punkt = Empfehlung</span>')
    st.markdown(
        f'<div style="font-size:0.85rem;opacity:0.8;margin:-0.5rem 0 0.5rem -14px">{"".join(items)}</div>',
        unsafe_allow_html=True,
    )
