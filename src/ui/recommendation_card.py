"""The headline card at the top: "Take public transport" or "Drive to garage X"."""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from src.models import ModeComparison
from src.ui import backend


@dataclass
class Recommendation:
    icon: str
    headline: str
    reason: str = ""  # one sentence, shown below the headline
    details: list[str] = field(default_factory=list)  # short facts


def build_recommendation(best: dict | None, comparison: ModeComparison | None) -> Recommendation:
    """Decide what the card says.

    Args:
        best: Top entry of the price-filtered ``rank_garages`` output, or None.
            It is the same garage ``compare_modes`` uses for the car.
        comparison: ``compare_modes`` result, or None if the comparison is off.
    """
    if comparison is not None and comparison["recommendation"] is None:
        return Recommendation("⚠️", "Keine Empfehlung möglich", comparison["reason"])
    if comparison is not None and comparison["recommendation"] == "transit":
        return _transit_card(comparison, best)
    if best is None:
        return Recommendation(
            "⚠️",
            "Kein passendes Parkhaus gefunden",
            "Alle Parkhäuser sind voraussichtlich voll oder über deinem Preislimit.",
            ["Tipp: Preislimit erhöhen oder eine andere Abfahrtszeit wählen."],
        )
    return _car_card(best, comparison)


def render_recommendation(rec: Recommendation) -> None:
    with st.container(border=True):
        st.markdown(f"### {rec.icon} {rec.headline}")
        if rec.reason:
            st.markdown(rec.reason)
        if rec.details:
            st.caption(" &nbsp;·&nbsp; ".join(rec.details))


def _transit_card(comparison: ModeComparison, best: dict | None) -> Recommendation:
    car, transit = comparison["car"], comparison["transit"]
    headline = "Nimm den ÖV"
    if car["available"]:
        faster = car["total_min"] - transit["total_min"]
        cheaper = car["cost_chf"] - transit["cost_chf"]
        if faster >= 1:
            headline += f" – {faster:.0f} min schneller"
        elif cheaper >= 1:
            headline += f" – CHF {cheaper:.2f} günstiger"
    wait = f" (davon {transit['wait_min']:.0f} min warten)" if transit["wait_min"] >= 1 else ""
    transfers = transit["transfers"]
    details = [
        f"ÖV {transit['total_min']:.0f} min{wait}, {transfers} {'Umstieg' if transfers == 1 else 'Umstiege'}, "
        f"CHF {transit['cost_chf']:.2f} hin & zurück",
    ]
    if car["available"]:
        details.append(f"Auto {car['total_min']:.0f} min, CHF {car['cost_chf']:.2f}")
    if best is not None:
        details.append(f"Falls doch Auto: {backend.display_name(best['garage'])} (~{best['predicted_free']} frei)")
    return Recommendation("🚆", headline, comparison["reason"], details)


def _car_card(best: dict, comparison: ModeComparison | None) -> Recommendation:
    garage = best["garage"]
    headline = f"Fahr zu {backend.display_name(garage)} – ~{best['predicted_free']} Plätze frei"
    parts = []
    if best["drive_min"] >= 1:
        parts.append(f"{best['drive_min']:.0f} min Fahrt")
    if best["search_min"] >= 1:
        parts.append(f"{best['search_min']:.0f} min Suche")
    parts.append(f"{best['walking_min']:.0f} min zu Fuss")
    details = [" + ".join(parts) + (f" = {best['total_min']:.0f} min" if len(parts) > 1 else "")]
    if garage.hourly_rate_chf is not None:
        suffix = " (Platzhalter)" if backend.prices_are_placeholders() else ""
        details.append(f"CHF {garage.hourly_rate_chf:.2f}/h{suffix}")
    details.append(f"Ankunft am Parkhaus ca. {best['arrival_at_garage']:%H:%M}")
    reason = comparison["reason"] if comparison is not None else ""
    return Recommendation("🅿️", headline, reason, details)
