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

# --- External APIs -----------------------------------------------------------
# TODO: verify the current URL of the Zurich parking guidance system feed.
PARKING_FEED_URL: str = "https://www.pls-zh.ch/plsFeed/rss"
TRANSIT_API_URL: str = "https://transport.opendata.ch/v1/connections"
WEATHER_API_URL: str = "https://api.open-meteo.com/v1/forecast"
ROUTING_API_KEY: str | None = os.getenv("ROUTING_API_KEY")

# --- Domain constants --------------------------------------------------------
ZURICH_CENTER: tuple[float, float] = (47.3769, 8.5417)  # (lat, lon)
REQUEST_TIMEOUT_S: int = 10
COLLECTOR_INTERVAL_MIN: int = 10

# Cost assumptions for the car vs. transit comparison (TODO: refine / source).
CAR_COST_PER_KM_CHF: float = 0.70
DEFAULT_TRANSIT_TICKET_CHF: float = 4.60

# Weights for the parking score (must sum to 1.0).
SCORE_WEIGHTS: dict[str, float] = {
    "free_spots": 0.4,
    "walking_distance": 0.4,
    "price": 0.2,
}

USE_DUMMY_DATA: bool = os.getenv("PARKITUNITY_USE_DUMMY_DATA", "true").lower() == "true"
