"""Google Maps route links for the recommendation.

Uses the documented "Maps URLs" format, which needs no API key:
https://developers.google.com/maps/documentation/urls/get-started#directions-action
"""

from __future__ import annotations

from urllib.parse import urlencode

from src.models import Location, ModeComparison
from src.ui import backend

GOOGLE_MAPS_DIRECTIONS = "https://www.google.com/maps/dir/"


def google_maps_url(destination: Location, travelmode: str, origin: Location | None = None) -> str:
    """Directions link, e.g. ``…/maps/dir/?api=1&origin=47.41,8.54&destination=…&travelmode=transit``.

    Args:
        destination: Where the route ends.
        travelmode: "driving", "transit", "walking" or "bicycling".
        origin: Start of the route. Without it, Google Maps starts at the
            user's current position.
    """
    params = {"api": "1"}
    if origin is not None:
        params["origin"] = f"{origin.lat},{origin.lon}"
    params["destination"] = f"{destination.lat},{destination.lon}"
    params["travelmode"] = travelmode
    return GOOGLE_MAPS_DIRECTIONS + "?" + urlencode(params)


def recommended_route(
    origin: Location | None,
    destination: Location,
    best: dict | None,
    comparison: ModeComparison | None,
) -> tuple[str, str] | None:
    """Link and short explanation for the recommended way to travel.

    - Public transport recommended -> transit route to the destination.
    - Otherwise -> driving route to the recommended garage.

    Returns:
        ``(url, caption)``, or ``None`` if there is nothing to navigate to.
    """
    if comparison is not None and comparison["recommendation"] == "transit":
        return google_maps_url(destination, "transit", origin), "Mit dem ÖV direkt zum Ziel."
    if best is None or (comparison is not None and comparison["recommendation"] is None):
        return None
    garage = best["garage"]
    caption = (f"Mit dem Auto zu {backend.display_name(garage)} – "
               f"von dort ca. {best['walking_min']:.0f} min zu Fuss zum Ziel.")
    return google_maps_url(garage.location, "driving", origin), caption
