# AI citation: this file was written with Claude (Anthropic).
"""Component "Parkhausprognose": what our ML model expects for one garage.

All forecast numbers come from ``predict_free`` (via the ranking and
``forecast_curve``), so list, car card and chart always show the same value.
"""

from __future__ import annotations

import html
import math

import altair as alt
import pandas as pd
import streamlit as st

from src import config
from src.data import parking_api
from src.ui import backend
from src.ui.garage_cards import render_number
from src.ui.styles import BLUE, GREEN, free_color_class

CHART_HEIGHT_PX = 340
LABEL_Y_PX = 26  # vertical centre of the green label
HEADROOM = 1.35  # the y axis ends 35 % above the highest value, so there is space for the label
# Outline of the green label (rounded rectangle, 2 units wide); "size" scales it.
LABEL_SHAPE = "M-0.9,-0.32H0.9Q1,-0.32,1,-0.22V0.22Q1,0.32,0.9,0.32H-0.9Q-1,0.32,-1,0.22V-0.22Q-1,-0.32,-0.9,-0.32Z"
LABEL_SIZE = 18_000
LABEL_SHIFT_PX = 80  # the label stands next to the arrival line


def curve_hours(arrival: pd.Timestamp, now: pd.Timestamp) -> int:
    """Length of the chart: 6 h, or longer so that a later arrival is still visible (max. 24 h)."""
    needed = math.ceil((arrival - now) / pd.Timedelta(hours=1) + 1)  # 1 h of space after the arrival
    return min(max(config.FORECAST_CURVE_HOURS, needed), config.FORECAST_CURVE_MAX_HOURS)


def chart_data(curve: pd.DataFrame, arrival: pd.Timestamp, arrival_free: int) -> tuple[pd.DataFrame, bool]:
    """Table for the chart: the curve plus the point of the arrival.

    The arrival usually lies between two 15-minute steps. Its value is added
    to the line, so the green dot lies exactly on it.

    Returns:
        (columns ``time`` and ``free``, is the arrival inside the chart?).
        ``time`` is Zurich clock time as text without time zone: otherwise
        the browser would convert it to the viewer's own time zone.
    """
    visible = curve["timestamp"].iloc[0] <= arrival <= curve["timestamp"].iloc[-1]
    table = curve
    if visible:
        point = pd.DataFrame({"timestamp": [arrival], "free_pred": [arrival_free]})
        others = curve[curve["timestamp"] != arrival]
        table = pd.concat([others, point], ignore_index=True).sort_values("timestamp")
    return pd.DataFrame({"time": clock_text(table["timestamp"]), "free": table["free_pred"]}), visible


def clock_text(times: pd.Series) -> pd.Series:
    """Zurich clock time as text, e.g. "2026-11-25T11:30:00"."""
    return times.dt.tz_convert(config.ZURICH_TZ).dt.strftime("%Y-%m-%dT%H:%M:%S")


def forecast_chart(
    curve: pd.DataFrame, free_now: int | None, arrival: pd.Timestamp, arrival_free: int
) -> tuple[alt.LayerChart, bool]:
    """Build the Altair chart, layer by layer (the last layer is drawn on top).

    Args:
        curve: Output of ``forecast_curve``.
        free_now: Live measurement, ``None`` = no live data (no blue dot).
        arrival: When the driver gets to the garage.
        arrival_free: Forecast for the arrival (``free_at_arrival`` of the ranking).

    Returns:
        (chart, is the arrival inside the chart?)
    """
    data, visible = chart_data(curve, arrival, arrival_free)
    top = max(data["free"].max(), free_now or 0, 10) * HEADROOM
    x = alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M", grid=False, tickCount=6, labelOverlap=True))
    y = alt.Y("free:Q", title=None, scale=alt.Scale(domain=[0, top]), axis=alt.Axis(gridOpacity=0.4, tickCount=5, domain=False))
    tooltip = [
        alt.Tooltip("time:T", title="Uhrzeit", format="%H:%M"),
        alt.Tooltip("free:Q", title="Freie Plätze (Prognose)"),
    ]

    # 1. Forecast: light blue area and dashed blue line.
    base = alt.Chart(data).encode(x=x, y=y)
    layers = [
        base.mark_area(color=BLUE, opacity=0.12),
        base.mark_line(color=BLUE, strokeDash=[6, 4], strokeWidth=2.5),
    ]
    # 2. Tooltip: the nearest point of the line reacts to the mouse.
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["time"], empty=False)
    layers.append(
        base.mark_point(color=BLUE, filled=True, size=60)
        .encode(tooltip=tooltip, opacity=alt.condition(hover, alt.value(1), alt.value(0)))
        .add_params(hover)
    )
    # 3. Blue dot: what is measured now.
    if free_now is not None:
        measured = pd.DataFrame({"time": data["time"].iloc[:1], "free": [free_now]})
        layers.append(
            alt.Chart(measured).mark_point(color=BLUE, filled=True, size=130, opacity=1)
            .encode(x=x, y=y, tooltip=[tooltip[0], alt.Tooltip("free:Q", title="Freie Plätze (gemessen)")])
        )
    # 4. Arrival: green line, green dot and green label.
    if visible:
        row = pd.DataFrame({
            "time": clock_text(pd.Series([arrival])),
            "free": [arrival_free],
            "label": [f"Deine Ankunft\n{spots_text(arrival_free)}"],
        })
        arrival_chart = alt.Chart(row)
        # The label stands on the right of the line, or on the left if the arrival is late in the chart.
        share = (arrival - curve["timestamp"].iloc[0]) / (curve["timestamp"].iloc[-1] - curve["timestamp"].iloc[0])
        shift = LABEL_SHIFT_PX if share < 0.5 else -LABEL_SHIFT_PX
        label = arrival_chart.encode(x=x, y=alt.value(LABEL_Y_PX))
        layers += [
            arrival_chart.mark_rule(color=GREEN, strokeDash=[4, 4], strokeWidth=1.5).encode(x=x),
            arrival_chart.mark_point(color=GREEN, filled=True, size=130, opacity=1).encode(x=x, y=y, tooltip=tooltip),
            label.mark_point(shape=LABEL_SHAPE, size=LABEL_SIZE, filled=True, color=GREEN, opacity=1, xOffset=shift),
            label.mark_text(color="white", fontWeight="bold", fontSize=13, lineBreak="\n", lineHeight=17,
                            align="center", baseline="middle", dx=shift, dy=-8).encode(text="label:N"),
        ]
    return alt.layer(*layers).properties(height=CHART_HEIGHT_PX).configure_view(strokeWidth=0), visible


def spots_text(free: int) -> str:
    """"194 freie Plätze" or "1 freier Platz"."""
    return "1 freier Platz" if free == 1 else f"{free} freie Plätze"


def render_header(free_at_arrival: int) -> None:
    """Title, and on the right the status as a coloured pill."""
    status = backend.STATUS_LABELS[backend.availability(free_at_arrival)]
    st.markdown(
        '<div class="pk-forecast-head"><h3 style="padding:0">Parkhausprognose</h3>'
        f'<span class="pk-pill {free_color_class(free_at_arrival)}">{html.escape(status)}</span></div>',
        unsafe_allow_html=True,
    )


def render_legend(has_measurement: bool) -> None:
    """Legend below the chart."""
    measured = '<span><span class="pk-blue">●</span> Aktuell gemessen</span>' if has_measurement else ""
    st.markdown(
        f'<div class="pk-legend">{measured}<span><span class="pk-blue">- - -</span> ML-Prognose</span></div>',
        unsafe_allow_html=True,
    )


def render_forecast(garages: dict[str, dict]) -> None:
    """Show the whole component.

    Args:
        garages: Garage id -> row of the ranking. ``st.session_state["selected_garage"]``
            must be one of the ids (app.py sets it to rank 1 by default).
    """
    header = st.container()  # filled below, when the selected garage is known
    selected_id = st.selectbox(
        ":material/directions_car: Parkhaus",
        options=list(garages),
        key="selected_garage",  # the choice is written back to the session state
        format_func=lambda garage_id: backend.display_name(garages[garage_id]["name"], garage_id),
    )
    garage = garages[selected_id]
    free_at_arrival = int(garage["free_at_arrival"])
    arrival = pd.Timestamp(garage["arrival_time"]).tz_convert(config.ZURICH_TZ)
    # Example garages (API down) have no measurement.
    no_live_data = pd.isna(garage["free_now"]) or parking_api.is_dummy(selected_id)
    free_now = None if no_live_data else int(garage["free_now"])

    with header:
        render_header(free_at_arrival)
    with st.container(key="forecast_numbers"):
        now_col, arrival_col, time_col = st.columns(3)
        with now_col:
            color = "" if free_now is None else free_color_class(free_now)
            render_number("–" if free_now is None else str(free_now), "Freie Plätze aktuell", color)
        with arrival_col:
            render_number(str(free_at_arrival), "Frei bei Ankunft", free_color_class(free_at_arrival))
        with time_col:
            render_number(f"{arrival:%H:%M}", "Ankunft")

    hours = curve_hours(arrival, pd.Timestamp(backend.now()))
    chart, visible = forecast_chart(backend.forecast_curve(selected_id, hours), free_now, arrival, free_at_arrival)
    st.altair_chart(chart, width="stretch")
    render_legend(has_measurement=free_now is not None)
    if not visible:
        st.caption(f"Deine Ankunft ({arrival:%d.%m. %H:%M}) liegt ausserhalb des gezeigten Zeitraums.")
