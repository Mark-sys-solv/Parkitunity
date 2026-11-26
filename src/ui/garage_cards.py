# AI citation: this file was written with Claude (Anthropic).
"""Component "Die besten Parkhäuser für deine Fahrt": the top garages as cards.

Only shows ``st.session_state["ranking"]``; nothing is calculated again.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from src import config
from src.data import parking_api
from src.ui import backend
from src.ui.styles import free_color_class

# Priority -> subtitle of the garage on rank 1.
PRIORITY_TITLES: dict[str, str] = {
    "fastest": "Schnellste Option",
    "cheapest": "Günstigste Option",
    "most_spots": "Meiste freie Plätze",
    "balanced": "Beste Gesamtoption",
}
OTHER_TITLE = "Weitere Alternative"


def card_labels(top: pd.DataFrame, priority: str) -> list[str]:
    """Subtitles of the cards, in the order of ``top`` (the best rows of the ranking).

    Rank 1 gets the title of the chosen priority. Another garage gets a title
    if it alone is the best of the shown garages in that category and the title
    is still free.
    """
    # For every category: a number per garage, smaller = better.
    categories = {
        "fastest": top["drive_min"] + top["walk_min"],
        "cheapest": top["price_per_hour"],
        "most_spots": -top["free_at_arrival"],
    }
    labels: list[str] = []
    for position in range(len(top)):
        if position == 0:
            labels.append(PRIORITY_TITLES[priority])
            continue
        label = OTHER_TITLE
        for key, values in categories.items():
            # A tie (e.g. two garages with the same price) has no winner.
            is_best = values.iloc[position] == values.min() and (values == values.min()).sum() == 1
            if is_best and PRIORITY_TITLES[key] not in labels:
                label = PRIORITY_TITLES[key]
                break
        labels.append(label)
    return labels


def select_garage(garage_id: str) -> None:
    """Show this garage in the forecast (callback of the chevron button)."""
    st.session_state["selected_garage"] = garage_id


def render_number(text: str, label: str, css_class: str = "") -> None:
    """A large number with a small grey label below."""
    st.markdown(
        f'<div class="pk-number {css_class}">{html.escape(text)}</div><div class="pk-muted">{html.escape(label)}</div>',
        unsafe_allow_html=True,
    )


def render_garage_card(row: dict, label: str) -> None:
    """One card: rank, name, subtitle, chevron and the three key numbers.

    Args:
        row: A row of the ranking as dict.
        label: Subtitle, see ``card_labels``.
    """
    rank = row["rank"]
    name = backend.display_name(row["name"], row["garage_id"])
    with st.container(border=True, key=f"garage_card_{rank}"):
        with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
            st.markdown(
                f'<div class="pk-head"><div class="pk-rank">{rank}</div><div>'
                f'<div class="pk-name">{html.escape(name)}</div>'
                f'<div class="pk-muted">{html.escape(label)}</div></div></div>',
                unsafe_allow_html=True,
            )
            st.button(
                ":material/chevron_right:",
                key=f"chevron_{rank}",
                type="tertiary",
                help=f"Prognose für {name} anzeigen",
                on_click=select_garage,
                args=(row["garage_id"],),
            )
        spots, walk, price = st.columns(3)
        with spots:
            free = int(row["free_at_arrival"])
            render_number(str(free), "Frei bei Ankunft", free_color_class(free))
        with walk:
            render_number(f"{int(row['walk_min'])} Min.", "Fussweg")
        with price:
            known = not pd.isna(row["price_per_hour"])
            render_number(f"CHF {row['price_per_hour']:.2f}" if known else "–", "Pro Stunde")


def render_garage_cards() -> None:
    """Show the best garages of ``st.session_state["ranking"]``."""
    st.subheader("Die besten Parkhäuser für deine Fahrt")
    ranking = st.session_state.get("ranking")
    if ranking is None or ranking.empty:
        st.info("Im Moment gibt es keine passenden Parkhäuser. Erhöhe das Preislimit oder versuche es später noch einmal.",
                icon=":material/info:")
        return
    top = ranking.head(config.CARDS_SHOWN)
    if any(parking_api.is_dummy(garage_id) for garage_id in top["garage_id"]):
        st.warning("Die Live-Daten der Parkhäuser sind gerade nicht erreichbar. Du siehst Beispielwerte, keine echten Zahlen.",
                   icon=":material/cloud_off:")
    for row, label in zip(top.to_dict("records"), card_labels(top, st.session_state["priority"])):
        render_garage_card(row, label)
