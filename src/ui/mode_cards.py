# AI citation: this file was written with Claude (Anthropic).
"""Component "Wie kommst du am besten ans Ziel?": car and public transport side by side."""

from __future__ import annotations

import html

import streamlit as st

from src import config
from src.data.geocoding import parse_coordinates
from src.models import CarOption, Location, ModeComparison, TransitOption
from src.recommendation.mode_compare import savings
from src.ui import backend
from src.ui.garage_cards import render_number
from src.ui.navigation import google_maps_url
from src.ui.styles import free_color_class


def short_place(text: str) -> str:
    """"Bahnhof Oerlikon, Zürich" -> "Bahnhof Oerlikon" (coordinates stay as they are)."""
    return text.strip() if parse_coordinates(text) else text.split(",")[0].strip()


def transfers_text(transfers: int) -> str:
    """"Direkt", "1 Umstieg", "2 Umstiege"."""
    if transfers == 0:
        return "Direkt"
    return "1 Umstieg" if transfers == 1 else f"{transfers} Umstiege"


def savings_items(saved: dict[str, float]) -> list[tuple[str, str, bool]]:
    """(number, label, is it an advantage?) for the three values of the savings row."""
    minutes, chf, percent = saved["minutes"], saved["chf"], saved["percent"]
    return [
        (f"{abs(minutes)} Min.", "Zeitersparnis" if minutes >= 0 else "Längere Reisezeit", minutes > 0),
        (f"CHF {abs(chf):.2f}", "Kosteneinsparung" if chf >= 0 else "Mehrkosten", chf > 0),
        (f"{abs(percent)} %", "Kürzere Reisezeit" if minutes >= 0 else "Längere Reisezeit", minutes > 0),
    ]


def _text(text: str, css_class: str) -> None:
    """One line of escaped text with a CSS class."""
    st.markdown(f'<div class="{css_class}">{html.escape(text)}</div>', unsafe_allow_html=True)


def _header(icon: str, title: str, recommended: bool) -> None:
    """Icon and title, with a green check mark on the right if recommended."""
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.markdown(f"{icon} **{title}**")
        if recommended:
            st.markdown('<div class="pk-check" title="Empfohlen">✓</div>', unsafe_allow_html=True)


def _time_and_cost(option: CarOption | TransitOption, recommended: bool) -> None:
    """The two large numbers of a card (green if recommended)."""
    color = "pk-green" if recommended else ""
    time_col, cost_col = st.columns(2)
    with time_col:
        render_number(f"{option['total_min']} Min.", "Reisezeit", color)
    with cost_col:
        render_number(f"CHF {option['cost_chf']:.2f}", "Gesamtkosten", color)


def _card(mode: str, recommended: bool):
    """Empty card container; the key tells the CSS which card is recommended."""
    return st.container(border=True, height="stretch", key=f"mode_card_{mode}" + ("_best" if recommended else ""))


def _route_button(label: str, url: str, recommended: bool) -> None:
    """Link to Google Maps: black if recommended, otherwise only an outline."""
    st.link_button(label, url, type="primary" if recommended else "secondary", width="stretch")


def _car_card(car: CarOption, origin: Location, recommended: bool) -> None:
    """Card for driving to the recommended garage."""
    with _card("car", recommended):
        _header(":material/directions_car:", "Auto", recommended)
        if not car["available"]:
            _text("Kein passendes Parkhaus gefunden", "pk-muted")
            return
        _time_and_cost(car, recommended)
        st.divider()
        garage = car["garage"]
        free = int(garage["free_at_arrival"])
        _text("Empfohlenes Parkhaus", "pk-muted")
        _text(backend.display_name(garage["name"], garage["garage_id"]), "pk-name")
        # Green, but orange or red if only few spots are expected (same colours as in the garage cards).
        _text(f"{free} freie Plätze erwartet" if free != 1 else "1 freier Platz erwartet", free_color_class(free))
        place = Location(garage["lat"], garage["lon"])
        _route_button("Autoroute öffnen", google_maps_url(place, "driving", origin), recommended)


def _transit_card(transit: TransitOption, origin: Location, destination: Location, recommended: bool) -> None:
    """Card for public transport."""
    with _card("transit", recommended):
        _header(":material/train:", "ÖV", recommended)
        if not transit["available"]:
            _text("Keine ÖV-Daten verfügbar", "pk-muted")
        else:
            _time_and_cost(transit, recommended)
            st.divider()
            _text("Verbindung", "pk-muted")
            _text(transfers_text(transit["transfers"]), "pk-name")
            _text("Kein Parkplatz erforderlich", "pk-green")
        _route_button("ÖV-Route öffnen", google_maps_url(destination, "transit", origin), recommended)


def _savings_row(comparison: ModeComparison) -> None:
    """What the recommended mode saves (nothing without a recommendation)."""
    saved = savings(comparison)
    if saved is None:
        return
    st.divider()
    mode = "ÖV" if comparison["recommendation"] == "transit" else "Auto"
    with st.container(key="mode_savings"):
        _text(f"Deine Ersparnis mit dem {mode}", "pk-title")
        for column, (number, label, good) in zip(st.columns(3), savings_items(saved)):
            with column:
                render_number(number, label, "pk-green" if good else "pk-muted-number")


def footnote(parking_hours: float) -> str:
    """Small print below the comparison."""
    text = (f"Reisezeit pro Weg; Kosten hin und zurück inklusive angenommener Parkdauer von {parking_hours:g} Stunden. "
            "ÖV-Reisezeit inklusive Wartezeit.")
    if config.USE_PLACEHOLDER_PRICES:
        text += " Preise sind vorläufige Platzhalter."
    return text


def render_mode_cards(
    comparison: ModeComparison,
    origin: Location,
    destination: Location,
    origin_text: str,
    destination_text: str,
    parking_hours: float,
) -> None:
    """Show the whole component."""
    st.subheader("Wie kommst du am besten ans Ziel?")
    _text(f"{short_place(origin_text)} → {short_place(destination_text)}", "pk-muted pk-route")
    winner = comparison["recommendation"]
    car_col, transit_col = st.columns(2)
    with car_col:
        _car_card(comparison["car"], origin, winner == "car")
    with transit_col:
        _transit_card(comparison["transit"], origin, destination, winner == "transit")
    _savings_row(comparison)
    st.caption(footnote(parking_hours))
