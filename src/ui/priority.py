# AI citation: this file was written with Claude (Anthropic).
"""Component "Was ist dir wichtig?": the driver chooses the priority of the ranking."""

from __future__ import annotations

import streamlit as st

from src import config

# Key (used everywhere in the code) -> label (only shown on the page).
PRIORITY_LABELS: dict[str, str] = {
    "fastest": ":material/schedule: Möglichst schnell ans Ziel",
    "cheapest": ":material/account_balance_wallet: Möglichst günstig parkieren",
    "most_spots": ":material/local_parking: Möglichst viele freie Parkplätze",
    "balanced": ":material/balance: Beste Gesamtoption",
}


def render_priority() -> str:
    """Show the four options; the chosen key is in ``st.session_state["priority"]`` and returned."""
    st.session_state.setdefault("priority", config.DEFAULT_PRIORITY)
    return st.radio(
        "Was ist dir wichtig?",
        options=list(PRIORITY_LABELS),
        format_func=lambda key: PRIORITY_LABELS[key],
        key="priority",
    )
