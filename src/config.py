"""Central configuration: paths, API endpoints and constants.

Secrets are read from a local ``.env`` file (see ``.env.example``).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> None:
    """Read ``KEY=VALUE`` lines from a ``.env`` file into the environment.

    Lines starting with ``#`` and empty lines are skipped. Variables that are
    already set (e.g. in the terminal) are not overwritten.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(ROOT_DIR / ".env")

# --- Paths -----------------------------------------------------------------
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
DB_PATH: Path = Path(os.getenv("PARKITUNITY_DB_PATH", DATA_DIR / "parkitunity.db"))
MODEL_PATH: Path = MODELS_DIR / "model.pkl"  # saved with joblib
DOCS_DIR: Path = ROOT_DIR / "docs"
# Synthetic development data (see notebooks/generate_synthetic_data.py). Never mix with real data.
SYNTHETIC_DB_PATH: Path = DATA_DIR / "synthetic.db"

# Historical occupancy from ParkenDD (written by src/data/download_history.py).
HISTORY_CSV_PATH: Path = PROCESSED_DATA_DIR / "history.csv"
GARAGES_CSV_PATH: Path = DATA_DIR / "garages.csv"

# --- External APIs -----------------------------------------------------------
# ParkenDD mirrors the official PLS Zürich feed (https://www.pls-zh.ch/plsFeed/rss, CC-0).
PARKENDD_URL: str = "https://api.parkendd.de/Zuerich"
TRANSIT_API_URL: str = "https://transport.opendata.ch/v1/connections"
# Address search (geocoding) with OpenStreetMap Nominatim. Usage policy:
# https://operations.osmfoundation.org/policies/nominatim/
NOMINATIM_URL: str = "https://nominatim.openstreetmap.org/search"
# Contact for the User-Agent (required by the policy), e.g. an e-mail address or project URL.
NOMINATIM_CONTACT: str = os.getenv("NOMINATIM_CONTACT", "")
# Prefer results in the Zurich area: lon_left, lat_top, lon_right, lat_bottom.
NOMINATIM_VIEWBOX: str = "8.35,47.50,8.75,47.25"
GEOCODING_CACHE_TTL_S: int = 7 * 24 * 3600  # addresses rarely move – cache for a week
WEATHER_API_URL: str = "https://api.open-meteo.com/v1/forecast"  # ~3 months back to 16 days ahead
WEATHER_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"  # older data
ROUTING_API_KEY: str | None = os.getenv("ROUTING_API_KEY")
ROUTING_PROVIDER: str = os.getenv("PARKITUNITY_ROUTING_PROVIDER", "haversine")

# --- HTTP / caching ------------------------------------------------------------
REQUEST_TIMEOUT_S: int = 10
PARKING_TIMEOUT_S: int = 15  # ParkenDD regularly needs ~5 s to respond
PARKING_CACHE_TTL_S: int = 60  # live data in the app is refreshed at most once a minute
TRANSIT_CACHE_TTL_S: int = 300
USER_AGENT: str = "Parkitunity/0.1 (university project)"

# --- Domain constants --------------------------------------------------------
ZURICH_CENTER: tuple[float, float] = (47.3769, 8.5417)  # (lat, lon)
ZURICH_TZ: str = "Europe/Zurich"

# Drive-time heuristic used by routing.HaversineProvider.
ROAD_FACTOR: float = 1.4  # straight-line distance -> approx. road distance
AVG_CITY_SPEED_KMH: float = 25.0

# Walking from the garage to the destination: straight-line distance x detour factor, rounded up.
WALKING_SPEED_KMH: float = 4.5
WALKING_DETOUR_FACTOR: float = 1.3  # streets are longer than the straight line

# --- Machine learning ----------------------------------------------------------
FORECAST_HORIZON_MIN: int = 60  # the model predicts free spots this many minutes ahead
FORECAST_CURVE_HOURS: int = 6  # length of the forecast chart
FORECAST_CURVE_MAX_HOURS: int = 24  # ... if the arrival is later than that
FEATURE_FREQ_MIN: int = 15  # time grid of the features (history is resampled to 15 min)
VALIDATION_DAYS: int = 14  # the 2 weeks before the test period: compare the models
TEST_DAYS: int = 14  # the newest 2 weeks: final result of the best model, used once

# --- Recommendation: assumptions -------------------------------------------------
# All numbers below are rough, documented assumptions. Change them here – the
# recommendation code (src/recommendation/) reads them from this file only.

# Ranking: weights of the three criteria for every priority ("Was ist dir wichtig?").
# travel = drive + walk (less is better), price = CHF/h (less is better),
# spots = free spots at arrival (more is better). The weights of a priority sum to 1.
PRIORITY_WEIGHTS: dict[str, dict[str, float]] = {
    "fastest": {"travel": 0.7, "price": 0.15, "spots": 0.15},
    "cheapest": {"travel": 0.15, "price": 0.7, "spots": 0.15},
    "most_spots": {"travel": 0.15, "price": 0.15, "spots": 0.7},
    "balanced": {"travel": 1 / 3, "price": 1 / 3, "spots": 1 / 3},
}
DEFAULT_PRIORITY: str = "balanced"
MIN_FREE_AT_ARRIVAL: int = 10  # garages with fewer expected free spots go to the end of the ranking
# Defaults and ranges (min, max, step) of the two sliders in the sidebar.
DEFAULT_MAX_WALK_MIN: int = 20  # garages with a longer walk go to the end of the ranking
MAX_WALK_RANGE: tuple[int, int, int] = (5, 60, 5)
DEFAULT_ENOUGH_SPOTS: int = 200  # more free spots than this do not improve the score
ENOUGH_SPOTS_RANGE: tuple[int, int, int] = (50, 500, 50)

# Traffic light of the free spots (cards, map and forecast).
FREE_GREEN_FROM: int = 50  # green from this number on
FREE_ORANGE_FROM: int = 10  # orange from this number on, red below
CARDS_SHOWN: int = 3  # how many garages are shown as cards

ASSUMED_RATE_IF_UNKNOWN_CHF_PER_H: float = 5.0  # used for ranking/costs if a garage has no price

# Costs of the trip there AND back (times are always one way).
# Car: 2 x km x COST_PER_KM_CHF + hourly rate x length of stay.
# PLACEHOLDER: rough guess for fuel and wear. 0 = only the parking fee counts.
COST_PER_KM_CHF: float = 0.20
DEFAULT_PARKING_HOURS: float = 2.0  # assumed length of stay
PARKING_HOURS_RANGE: tuple[float, float, float] = (0.5, 8.0, 0.5)  # slider: min, max, step

# PLACEHOLDER – public transport: SIMPLIFIED ZVV single tickets (adult, 2nd class, approx.).
# Real fares depend on zones, not distance – verify on zvv.ch before relying on them.
# (max. straight-line distance in km, price in CHF), checked in order.
TRANSIT_FARES_CHF: list[tuple[float, float]] = [
    (1.5, 2.80),  # short trip ("Kurzstrecke")
    (10.0, 4.60),  # within the city of Zurich (zone 110)
    (20.0, 7.00),  # ~3 zones
    (30.0, 9.20),  # ~4 zones
    (float("inf"), 13.80),  # further: rough upper value
]

# Decision car vs. public transport: weights of travel time and costs per priority
# (1/0 = only the time counts, 0/1 = only the costs count).
MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "fastest": {"time": 1.0, "cost": 0.0},
    "cheapest": {"time": 0.0, "cost": 1.0},
    "most_spots": {"time": 0.5, "cost": 0.5},
    "balanced": {"time": 0.5, "cost": 0.5},
}

# ParkenDD provides no prices. Until a real price source exists, garages get
# PLACEHOLDER hourly rates (PLACEHOLDER_RATES_CHF below). Set to false
# once real prices are available.
USE_PLACEHOLDER_PRICES: bool = True
# PLACEHOLDER hourly rates in CHF – rough guesses, NOT real prices.
PLACEHOLDER_RATES_CHF: dict[str, float] = {
    "zuerichparkhausaccu": 2.0,
    "zuerichparkhausalbisriederplatz": 2.5,
    "zuerichparkhausbleicherweg": 4.5,
    "zuerichparkhauscentereleven": 2.0,
    "zuerichparkhauscityparking": 4.0,
    "zuerichparkhauscityport": 3.0,
    "zuerichparkhauscrowneplaza": 3.5,
    "zuerichparkhausdorflinde": 2.0,
    "zuerichparkhausfeldegg": 3.5,
    "zuerichparkhausglobus": 5.0,
    "zuerichparkhaushardauii": 2.0,
    "zuerichparkhaushauptbahnhof": 6.0,
    "zuerichparkhaushohepromenade": 4.0,
    "zuerichparkhausjelmoli": 5.0,
    "zuerichparkhausjungholz": 2.0,
    "zuerichparkhausmaxbillplatz": 2.5,
    "zuerichparkhausmessezuerichag": 3.0,
    "zuerichparkhausnordhaus": 2.5,
    "zuerichparkhausoctavo": 2.5,
    "zuerichparkhausopéra": 4.5,
    "zuerichparkhauspwest": 2.5,
    "zuerichparkhausparkhyatt": 5.0,
    "zuerichparkhausparkside": 3.0,
    "zuerichparkhauspfingstweid": 2.5,
    "zuerichparkhausstampfenbach": 3.5,
    "zuerichparkhaustalgarten": 4.0,
    "zuerichparkhaususznord": 3.5,
    "zuerichparkhausuniirchel": 2.0,
    "zuerichparkhausurania": 4.5,
    "zuerichparkhausutoquai": 4.0,
    "zuerichparkhauszueri11shopping": 2.0,
    "zuerichparkhauszuerichhorn": 3.5,
    "zuerichparkplatztheater11": 2.5,
    "zuerichparkplatzuszsued": 3.5,
}
PLACEHOLDER_DEFAULT_RATE_CHF: float = 3.5  # garages that are not in the list

# If true, the app skips live parking data entirely. If false (default), live
# data is used and dummy data only serves as a fallback when the API fails.
USE_DUMMY_DATA: bool = os.getenv("PARKITUNITY_USE_DUMMY_DATA", "false").lower() == "true"
