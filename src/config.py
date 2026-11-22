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
MODEL_PATH: Path = MODELS_DIR / "occupancy_model.pkl.gz"  # pickle, gzip-compressed
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

# Walking from the garage to the destination: straight-line distance at 5 km/h.
WALKING_SPEED_KMH: float = 5.0
WALKING_DETOUR_FACTOR: float = 1.0  # 1.0 = straight line; ~1.3 would model detours along streets

# --- Machine learning ----------------------------------------------------------
FORECAST_HORIZON_MIN: int = 30  # the model predicts free spots this many minutes ahead
FEATURE_FREQ_MIN: int = 15  # time grid of the features (history is resampled to 15 min)
TEST_FRACTION: float = 0.2  # last 20 % of the time range is used for testing

# --- Recommendation: assumptions -------------------------------------------------
# All numbers below are rough, documented assumptions. Change them here – the
# recommendation code (src/recommendation/) reads them from this file only.

# Search penalty: extra minutes to find a spot when a garage is nearly full.
# 0 min with more than HIGH free spots, MAX min with LOW or fewer, linear in between.
SEARCH_PENALTY_MAX_MIN: float = 10.0
SEARCH_PENALTY_LOW_FREE: int = 10
SEARCH_PENALTY_HIGH_FREE: int = 50

# Garage score = weighted sum of three sub-scores in [0, 1] (higher is better).
# Weights are normalised to sum to 1. With the defaults below, roughly
# 1 CHF/h more ≈ 3 min longer ≈ 7 fewer free spots.
SCORE_WEIGHTS: dict[str, float] = {
    "time": 0.5,  # total time = drive + search + walk
    "availability": 0.3,  # predicted free spots
    "price": 0.2,  # hourly rate
}
# Sub-scores use absolute scales, so a garage's score does not depend on which
# other garages are in the list (e.g. after applying the price filter).
SCORE_TIME_WORST_MIN: float = 60.0  # total time at which the time score reaches 0
SCORE_FREE_SATURATION: int = 50  # more free spots than this do not improve the score
SCORE_PRICE_WORST_CHF_PER_H: float = 8.0  # hourly rate at which the price score reaches 0
ASSUMED_RATE_IF_UNKNOWN_CHF_PER_H: float = 5.0  # used for scoring/costs if a garage has no price

# Car costs.
PARKING_DURATION_H: float = 2.0  # assumed length of stay for the parking fee
INCLUDE_CAR_KM_COSTS: bool = True
# Variable costs per km (fuel, wear) – what matters when you already own a car.
# The full TCS cost incl. depreciation and insurance would be much higher (~0.70 CHF/km).
CAR_COST_PER_KM_CHF: float = 0.20
# Costs cover the way there AND back (parking implies returning; transit needs a
# return ticket). Times are always one way.
ROUND_TRIP: bool = True

# Public transport: SIMPLIFIED ZVV single tickets (adult, 2nd class, approx.).
# Real fares depend on zones, not distance – verify on zvv.ch before relying on them.
# (max. straight-line distance in km, price in CHF), checked in order.
TRANSIT_FARES_CHF: list[tuple[float, float]] = [
    (1.5, 2.80),  # short trip ("Kurzstrecke")
    (10.0, 4.60),  # within the city of Zurich (zone 110)
    (20.0, 7.00),  # ~3 zones
    (30.0, 9.20),  # ~4 zones
    (float("inf"), 13.80),  # further: rough upper value
]

# Decision car vs. public transport: time is converted to money and added to
# the costs ("generalised costs"); the cheaper mode wins.
VALUE_OF_TIME_CHF_PER_H: float = 25.0  # rough assumption for leisure trips
DECISION_TIE_CHF: float = 1.0  # smaller differences count as a tie -> public transport (no parking stress)

# ParkenDD provides no prices. Until a real price source exists, garages get
# PLACEHOLDER hourly rates (parking_api._PLACEHOLDER_RATES_CHF). Set to false
# once real prices are available.
USE_PLACEHOLDER_PRICES: bool = True

# If true, the app skips live parking data entirely. If false (default), live
# data is used and dummy data only serves as a fallback when the API fails.
USE_DUMMY_DATA: bool = os.getenv("PARKITUNITY_USE_DUMMY_DATA", "false").lower() == "true"
