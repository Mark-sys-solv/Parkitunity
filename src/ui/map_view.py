"""Map of Zurich with all garages, the start point and the destination (``st.pydeck_chart``).

The map draws coloured dots (one per row of a DataFrame):
- pointing at a dot shows its name (tooltip); clicking a garage selects it,
- start and destination are drawn as several dots on the same spot, biggest
  first. The smaller dots lie on top, so it looks like a ring or a target.

pydeck is installed automatically together with Streamlit, so it is not an
extra entry in requirements.txt.
"""

from __future__ import annotations

import math

import pandas as pd
import pydeck as pdk
import streamlit as st

from src.config import ZURICH_CENTER
from src.models import Location
from src.ui import backend

OVER_BUDGET_COLOR = "#898781"  # grey: more expensive than the user's limit
START_COLOR = "#2563eb"  # blue: start point
DESTINATION_COLOR = "#0b0b0b"  # black: destination
RING_COLOR = "#ffffff"  # white ring, so start and destination stand out on light and dark maps

# Dot sizes are radii in screen pixels.
GARAGE_SIZE_PX = 4
RECOMMENDED_SIZE_PX = 8  # the recommended garage gets a bigger dot
# Start and destination as (colour, radius in pixels), from the outside in.
START_MARKER = [(RING_COLOR, 10), (START_COLOR, 7)]  # blue dot with a white ring
DESTINATION_MARKER = [(DESTINATION_COLOR, 10), (RING_COLOR, 7), (DESTINATION_COLOR, 4)]  # target

# Used to choose a zoom level at which all dots fit on the map.
MAP_HEIGHT_PX = 500  # height of the map
MAP_WIDTH_PX = 600  # the real width depends on the screen – a careful guess
MARGIN_PX = 30  # free space around the outermost dots
EARTH_CIRCUMFERENCE_M = 40_075_016
METRES_PER_DEGREE = 111_320  # length of one degree of latitude
MIN_ZOOM, MAX_ZOOM = 8, 14  # 8 ≈ whole canton, 14 ≈ a few streets
MAP_KEY = "garage_map"  # widget key of the map (it remembers the clicked garage)


def is_over_budget(price_per_hour: float, max_price_chf: float) -> bool:
    """True if the hourly price is above the user's limit (an unknown price, NaN, is not)."""
    return price_per_hour > max_price_chf


def marker_rows(location: Location, layers: list[tuple[str, int]], name: str) -> list[dict]:
    """Several dots on the same spot (biggest first), e.g. a dot with a ring."""
    return [
        {"lat": location.lat, "lon": location.lon, "color": color, "size": size, "name": name, "garage_id": ""}
        for color, size in layers
    ]


MAP_COLUMNS = ["lat", "lon", "color", "size", "name", "garage_id"]


def hex_to_rgba(color: str) -> list[int]:
    """"#rrggbb" or "#rrggbbaa" -> [r, g, b, a] (pydeck wants numbers, not hex)."""
    digits = color.lstrip("#")
    if len(digits) == 6:
        digits += "ff"  # fully opaque
    return [int(digits[i:i + 2], 16) for i in (0, 2, 4, 6)]


def map_data(
    garages: pd.DataFrame,
    destination: Location | None,
    recommended_id: str | None,
    max_price_chf: float,
    origin: Location | None = None,
) -> pd.DataFrame:
    """One row per dot: lat, lon, color, size (radius in pixels), name (tooltip), garage_id.

    Args:
        garages: Rows of ``build_candidates`` or ``rank_garages``.
        destination: Drawn as a black-and-white target (None = no valid destination).
        recommended_id: This garage gets a bigger dot.
        max_price_chf: Garages above this hourly rate are grey.
        origin: Start point, drawn as a blue dot with a white ring (None = unknown).
    """
    rows = []
    for g in garages.to_dict("records"):
        status = backend.availability(g["free_at_arrival"])
        over_budget = is_over_budget(g["price_per_hour"], max_price_chf)
        rows.append({
            "lat": g["lat"],
            "lon": g["lon"],
            "color": OVER_BUDGET_COLOR if over_budget else backend.STATUS_COLORS[status],
            "size": RECOMMENDED_SIZE_PX if g["garage_id"] == recommended_id else GARAGE_SIZE_PX,
            "name": backend.display_name(g["name"], g["garage_id"]),
            "garage_id": g["garage_id"],
        })
    # Start and destination come last, so they are drawn on top of the garages.
    if origin is not None:
        rows.extend(marker_rows(origin, START_MARKER, "Startpunkt"))
    if destination is not None:
        rows.extend(marker_rows(destination, DESTINATION_MARKER, "Ziel"))
    if not rows:
        # Nothing to show: one invisible dot keeps the map centred on Zurich.
        rows.append({"lat": ZURICH_CENTER[0], "lon": ZURICH_CENTER[1], "color": "#00000000", "size": 1,
                     "name": "", "garage_id": ""})
    return pd.DataFrame(rows, columns=MAP_COLUMNS)


def metres_per_pixel(zoom: float, lat: float) -> float:
    """How many metres one screen pixel covers at this zoom level.

    The map is drawn with deck.gl: there the whole world is 512 pixels wide at
    zoom 0, and twice as wide with every zoom level.
    """
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(lat)) / (512 * 2**zoom)


def fit_zoom(data: pd.DataFrame) -> float:
    """Zoom level at which all dots fit on the map.

    The middle of all dots is the centre of the map; the zoom is chosen so
    that start, destination and all garages are visible.
    """
    lat = data["lat"].mean()
    height_m = (data["lat"].max() - data["lat"].min()) * METRES_PER_DEGREE
    width_m = (data["lon"].max() - data["lon"].min()) * METRES_PER_DEGREE * math.cos(math.radians(lat))
    # Metres per pixel we need so that everything fits (at least 1, for a single dot).
    needed = max(height_m / (MAP_HEIGHT_PX - 2 * MARGIN_PX), width_m / (MAP_WIDTH_PX - 2 * MARGIN_PX), 1.0)
    zoom = math.log2(metres_per_pixel(0, lat) / needed)
    return round(min(max(zoom, MIN_ZOOM), MAX_ZOOM), 1)


def render_map(
    garages: pd.DataFrame,
    destination: Location | None,
    recommended_id: str | None,
    max_price_chf: float,
    origin: Location | None = None,
) -> str | None:
    """Show the map with all garages, the start point and the destination.

    Returns the id of the garage the user clicked on (None = no garage clicked).
    """
    data = map_data(garages, destination, recommended_id, max_price_chf, origin)
    view = pdk.ViewState(
        latitude=(data["lat"].min() + data["lat"].max()) / 2,
        longitude=(data["lon"].min() + data["lon"].max()) / 2,
        zoom=fit_zoom(data),
    )
    data["rgba"] = data["color"].apply(hex_to_rgba)
    is_garage = data["garage_id"] != ""

    def dots(layer_id: str, rows: pd.DataFrame) -> pdk.Layer:
        """One map layer with a dot per row."""
        return pdk.Layer(
            "ScatterplotLayer",
            id=layer_id,
            data=rows,
            get_position=["lon", "lat"],
            get_fill_color="rgba",
            get_radius="size",
            radius_units="'pixels'",  # inner quotes: a fixed value, not a column name; same size at every zoom
            pickable=True,  # needed for the tooltip and for clicks
        )

    # Two layers, so a click can only select a garage, not the start or the destination.
    deck = pdk.Deck(
        layers=[dots("garages", data[is_garage]), dots("markers", data[~is_garage])],
        initial_view_state=view,
        map_style=None,  # Streamlit's default map, light or dark like the app
        tooltip={"text": "{name}"},
    )
    event = st.pydeck_chart(deck, height=MAP_HEIGHT_PX, on_select="rerun", selection_mode="single-object", key=MAP_KEY)
    clicked = event.selection.objects.get("garages", [])
    return clicked[0]["garage_id"] if clicked else None


def render_garage_info(garage: dict, recommended_id: str | None, max_price_chf: float) -> None:
    """Name and key numbers of the garage that was clicked on the map."""
    row = garage_table(pd.DataFrame([garage]), recommended_id, max_price_chf).iloc[0]
    parts = [f"**{row['Parkhaus']}**", row["Verfügbarkeit"], f"{row['Frei (Prognose)']} frei (Prognose)"]
    if not pd.isna(row["CHF/h"]):
        parts.append(f"CHF {row['CHF/h']:.2f}/h")
    parts.append(f"{row['Fussweg (min)']} min zu Fuss")
    if row["Hinweis"]:
        parts.append(row["Hinweis"])
    st.info(" · ".join(parts), icon="📍")


def garage_table(garages: pd.DataFrame, recommended_id: str | None, max_price_chf: float) -> pd.DataFrame:
    """The details of the garages for the table below the map, in the given order."""
    rows = []
    for g in garages.to_dict("records"):
        if g["garage_id"] == recommended_id:
            note = "★ Empfehlung"
        elif is_over_budget(g["price_per_hour"], max_price_chf):
            note = "über Preislimit"
        else:
            note = ""
        rows.append({
            "Parkhaus": backend.display_name(g["name"], g["garage_id"]),
            "Verfügbarkeit": backend.STATUS_LABELS[backend.availability(g["free_at_arrival"])],
            "Frei (Prognose)": g["free_at_arrival"],
            "CHF/h": g["price_per_hour"],
            "Fussweg (min)": g["walk_min"],
            "Gesamt (min)": round(g["drive_min"] + g["walk_min"]),
            "Hinweis": note,
        })
    return pd.DataFrame(rows)


def render_garage_table(garages: pd.DataFrame, recommended_id: str | None, max_price_chf: float) -> None:
    """Show the table of all garages."""
    st.dataframe(garage_table(garages, recommended_id, max_price_chf), hide_index=True, width="stretch")


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
