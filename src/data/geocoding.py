"""Address search (geocoding) with OpenStreetMap Nominatim.

Nominatim usage policy – please read before changing anything here:
https://operations.osmfoundation.org/policies/nominatim/

How we follow it:
- We identify ourselves with our own User-Agent: project name + a contact
  address from ``.env`` (``NOMINATIM_CONTACT``).
- At most 1 request per second (``_wait_for_rate_limit``).
- Results are cached (``st.cache_data``), so each address is looked up only once.
- No search-as-you-type: Streamlit text fields only send the text when the
  user presses Enter or leaves the field.

Search results are © OpenStreetMap contributors (ODbL); the app shows this
attribution in its footer.
"""

from __future__ import annotations

import logging
import threading
import time

import requests
import streamlit as st

from src import config

logger = logging.getLogger(__name__)

MIN_SECONDS_BETWEEN_REQUESTS = 1.0  # policy: "an absolute maximum of 1 request per second"
_rate_lock = threading.Lock()  # several app users may search at the same time
_last_request = 0.0  # time of our last request (time.monotonic)


def geocode(address: str) -> tuple[float, float] | None:
    """Find the coordinates of an address or place in Switzerland.

    Results in the Zurich area are preferred. Coordinates typed as
    ``"47.3697, 8.5391"`` are returned directly, without asking Nominatim.

    Args:
        address: e.g. "Bahnhofstrasse 1, Zürich" or "Kunsthaus Zürich".

    Returns:
        ``(lat, lon)``, or ``None`` if nothing was found or the service is
        not reachable (the error is logged).
    """
    text = " ".join(address.split())  # remove extra spaces
    if not text:
        return None
    coords = parse_coordinates(text)
    if coords is not None:
        return coords
    try:
        return _lookup(text.lower())  # lower case: "Paradeplatz" and "paradeplatz" share a cache entry
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.warning("Address search for %r failed: %s", text, exc)
        return None


@st.cache_data(ttl=config.GEOCODING_CACHE_TTL_S, show_spinner=False)
def _lookup(query: str) -> tuple[float, float] | None:
    """Ask Nominatim (cached).

    "Not found" (``None``) is cached; network errors raise an exception and
    are therefore NOT cached, so the next attempt asks again.
    """
    _wait_for_rate_limit()
    response = requests.get(
        config.NOMINATIM_URL,
        params={
            "q": query,
            "format": "jsonv2",
            "limit": 1,
            "countrycodes": "ch",  # only Switzerland
            "viewbox": config.NOMINATIM_VIEWBOX,  # prefer the Zurich area ...
            "bounded": 0,  # ... but also accept places outside of it
        },
        headers={"User-Agent": user_agent(), "Accept-Language": "de"},
        timeout=config.REQUEST_TIMEOUT_S,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"])


def user_agent() -> str:
    """Our own User-Agent: project name and (if configured) a contact address."""
    contact = config.NOMINATIM_CONTACT.strip()
    return f"Parkitunity/0.1 (university project{'; ' + contact if contact else ''})"


def parse_coordinates(text: str) -> tuple[float, float] | None:
    """Parse ``"lat,lon"`` into a tuple, or return ``None`` if it is not coordinates."""
    parts = text.split(",")
    if len(parts) != 2:
        return None
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return (lat, lon)


def _wait_for_rate_limit() -> None:
    """Sleep so that at least one second lies between two requests."""
    global _last_request
    with _rate_lock:
        wait = MIN_SECONDS_BETWEEN_REQUESTS - (_now() - _last_request)
        if wait > 0:
            _sleep(wait)
        _last_request = _now()


def _now() -> float:
    """Monotonic clock (separate function so tests can fake the time)."""
    return time.monotonic()


def _sleep(seconds: float) -> None:
    """Sleep (separate function so tests can skip waiting)."""
    time.sleep(seconds)
