"""Central configuration: paths, API endpoints and constants.

Secrets are read from a local ``.env`` file (see ``.env.example``).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional for running the dummy app
    pass

# --- Paths -----------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
DB_PATH: Path = Path(os.getenv("PARKITUNITY_DB_PATH", DATA_DIR / "parkitunity.db"))
MODEL_PATH: Path = MODELS_DIR / "occupancy_model.joblib"

OCCUPANCY_CSV_DIR: Path = DATA_DIR / "occupancy"  # filled by GitHub Actions (branch "data")

# --- External APIs -----------------------------------------------------------
# ParkenDD mirrors the official PLS Zürich feed (https://www.pls-zh.ch/plsFeed/rss, CC-0).
PARKENDD_URL: str = "https://api.parkendd.de/Zuerich"
TRANSIT_API_URL: str = "https://transport.opendata.ch/v1/connections"
WEATHER_API_URL: str = "https://api.open-meteo.com/v1/forecast"
ROUTING_API_KEY: str | None = os.getenv("ROUTING_API_KEY")
ROUTING_PROVIDER: str = os.getenv("PARKITUNITY_ROUTING_PROVIDER", "haversine")

# --- HTTP / caching ------------------------------------------------------------
REQUEST_TIMEOUT_S: int = 10
PARKING_TIMEOUT_S: int = 15  # ParkenDD regularly needs ~5 s to respond
PARKING_CACHE_TTL_S: int = 60  # app only; the collector always fetches fresh data
TRANSIT_CACHE_TTL_S: int = 300
USER_AGENT: str = "Parkitunity/0.1 (university project)"

# --- Domain constants --------------------------------------------------------
ZURICH_CENTER: tuple[float, float] = (47.3769, 8.5417)  # (lat, lon)
ZURICH_TZ: str = "Europe/Zurich"
COLLECTOR_INTERVAL_MIN: int = 15

# Drive-time heuristic used by routing.HaversineProvider.
ROAD_FACTOR: float = 1.4  # straight-line distance -> approx. road distance
AVG_CITY_SPEED_KMH: float = 25.0
WALKING_SPEED_KMH: float = 4.8

# Cost assumptions for the car vs. transit comparison (TODO: refine / source).
CAR_COST_PER_KM_CHF: float = 0.70
DEFAULT_TRANSIT_TICKET_CHF: float = 4.60

# Weights for the parking score (must sum to 1.0).
SCORE_WEIGHTS: dict[str, float] = {
    "free_spots": 0.4,
    "walking_distance": 0.4,
    "price": 0.2,
}

# If true, the app skips live parking data entirely. If false (default), live
# data is used and dummy data only serves as a fallback when the API fails.
USE_DUMMY_DATA: bool = os.getenv("PARKITUNITY_USE_DUMMY_DATA", "false").lower() == "true"
