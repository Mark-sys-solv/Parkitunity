"""Google Maps route links.

Uses the documented "Maps URLs" format, which needs no API key:
https://developers.google.com/maps/documentation/urls/get-started#directions-action
"""

from __future__ import annotations

from urllib.parse import urlencode

from src.models import Location

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
