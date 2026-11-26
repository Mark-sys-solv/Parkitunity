# AI citation: this file was written with Claude (Anthropic).
"""All custom CSS of the app in one place (only typography, colours and spacing)."""

from __future__ import annotations

import streamlit as st

from src.ui import backend

# Text colours of the free spots; readable on the light and the dark theme.
GREEN, ORANGE, RED = "#16a34a", "#d97706", "#dc2626"
BLUE = "#2a78d6"  # measurements and forecast in the chart

# Streamlit adds the class "st-key-<key>" to containers, so the cards can be found by key.
CSS = f"""
<style>
[class*="st-key-garage_card_"] {{
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
}}
/* Streamlit pulls text up by 1rem at the bottom; inside the cards we want the full padding. */
[class*="st-key-garage_card_"] [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
[class*="st-key-garage_card_"] button p {{ font-size: 1.4rem; }}
/* Keep the three numbers side by side on narrow screens, too. */
[class*="st-key-garage_card_"] [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap; }}
[class*="st-key-garage_card_"] [data-testid="stColumn"] {{ min-width: 0; }}
.pk-head {{ display: flex; align-items: center; gap: 0.9rem; }}
.pk-rank {{
    flex: none; width: 2.4rem; height: 2.4rem; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: rgba(128, 128, 128, 0.15); font-weight: 600;
}}
.pk-name {{ font-weight: 700; font-size: 1.05rem; line-height: 1.3; }}
.pk-muted {{ font-size: 0.85rem; line-height: 1.3; opacity: 0.65; }}
.pk-number {{ font-weight: 700; font-size: 1.5rem; line-height: 1.3; white-space: nowrap; }}
.pk-green {{ color: {GREEN}; }}
.pk-orange {{ color: {ORANGE}; }}
.pk-red {{ color: {RED}; }}
/* Comparison car / public transport. The recommended card has a key that ends with "_best". */
[class*="st-key-mode_card_"] {{ border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 12px; padding: 1.25rem 1.5rem; }}
[class*="st-key-mode_card_"][class*="_best"] {{ border: 2px solid {GREEN}; }}
[class*="st-key-mode_card_"] [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
[class*="st-key-mode_card_"] hr {{ margin: 0.25rem 0; }}
[class*="st-key-mode_card_"][class*="_best"] a[data-testid^="stBaseLinkButton"] {{
    background: #111111; color: #ffffff; border-color: rgba(128, 128, 128, 0.6);
}}
.st-key-mode_savings [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
.pk-muted-number {{ opacity: 0.55; }}
[class*="st-key-mode_card_"] [data-testid="stMarkdownContainer"] p {{ margin: 0; }}
.pk-route {{ margin-bottom: 1rem; }}
.pk-forecast-head {{ display: flex; align-items: center; justify-content: space-between; gap: 1rem; }}
.pk-pill {{ border-radius: 999px; padding: 0.25rem 0.9rem; font-size: 0.85rem; font-weight: 600; white-space: nowrap; }}
.pk-pill.pk-green {{ background: {GREEN}22; }}
.pk-pill.pk-orange {{ background: {ORANGE}22; }}
.pk-pill.pk-red {{ background: {RED}22; }}
.pk-legend {{ display: flex; gap: 1.5rem; justify-content: center; font-size: 0.85rem; }}
.pk-blue {{ color: {BLUE}; font-weight: 700; }}
.st-key-forecast_numbers [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap; }}
.st-key-forecast_numbers [data-testid="stColumn"] {{ min-width: 0; }}
.pk-title {{ font-weight: 700; font-size: 1.15rem; }}
.pk-check {{
    flex: none; width: 1.6rem; height: 1.6rem; border-radius: 50%; background: {GREEN}; color: #ffffff;
    display: flex; align-items: center; justify-content: center; font-size: 1rem; font-weight: 700;
}}
@media (max-width: 640px) {{ .pk-number {{ font-size: 1.15rem; }} }}
</style>
"""


def inject_styles() -> None:
    """Add the CSS to the page (once per run, at the top of app.py)."""
    st.html(CSS)


def free_color_class(free: int) -> str:
    """CSS class for a number of free spots (thresholds: config)."""
    return {"good": "pk-green", "tight": "pk-orange", "full": "pk-red"}[backend.availability(free)]
