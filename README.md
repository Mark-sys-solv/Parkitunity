# 🅿️ Parkitunity

Group 9.6 – University project.

Parkitunity is a Streamlit app for Zurich that recommends **which parking garage to head for**
(based on predicted free spots, walking distance and price) and compares whether
**public transport would be faster or cheaper**.

## Setup

```bash
git clone <repo-url> && cd Parkitunity
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # adjust values if needed
streamlit run app.py
```

No API keys needed. Parking occupancy ([ParkenDD](https://api.parkendd.de/Zuerich)) and
transit connections ([transport.opendata.ch](https://transport.opendata.ch)) are live; if the
parking API is down, the app falls back to built-in dummy garages.

### Other commands

```bash
pytest                                   # run all tests offline (APIs are mocked)
RUN_LIVE_TESTS=1 pytest -m live          # additionally check the real APIs
python -m src.data.collector             # log garage occupancy once into SQLite
python -m src.data.collector --loop      # log every 15 minutes
python -m src.ml.train                   # train the occupancy model
```

## Collected data

A GitHub Actions workflow ([.github/workflows/collect.yml](.github/workflows/collect.yml)) logs the
occupancy of all garages every 15 minutes. It appends to monthly CSV files
(`data/occupancy/YYYY-MM.csv`) and commits them to the separate branch **`data`**, so `main`
stays free of bot commits. Load the data into your local SQLite database like this:

```bash
git fetch origin data
git restore --source origin/data --worktree -- data/occupancy   # copies the CSVs, stages nothing
python -m src.data.collector --import-csv                       # -> data/parkitunity.db
```

`data/occupancy/` is git-ignored on `main`. Importing is idempotent, so you can repeat these
steps anytime to get the latest data. The workflow can also be started manually from the
**Actions** tab ("Run workflow").

Things to know about GitHub Actions:

- Scheduled runs often start a few minutes late.
- They only run from the default branch, and GitHub disables them after 60 days without
  repository activity.
- On **private** repos every run counts as at least 1 billed minute (~2,900 min/month at 15-min
  intervals; the free plan includes 2,000). Public repos are free.

## Project structure

```
app.py                  Streamlit entry point (UI composition only)
src/config.py           Paths, API URLs, constants
src/models.py           Shared data types (Garage, Location, ModeComparison, …)
src/data/               External APIs, SQLite database, occupancy collector
src/ml/                 Features, training, prediction
src/recommendation/     Garage ranking, car vs. transit comparison
src/ui/                 Sidebar, map, charts
models/                 Trained model files
data/raw/               Raw data (not in git)
data/processed/         Processed datasets
notebooks/              Exploration
tests/                  Tests
docs/                   Documentation, reports
```

## Key interfaces

| Function | Module | Returns |
|---|---|---|
| `predict(garage_id: str, arrival_time: datetime) -> int` | `src/ml/predict.py` | Predicted free spots |
| `compare_modes(origin: str, destination: str, time: datetime) -> dict` | `src/recommendation/mode_compare.py` | `{"car": {...}, "transit": {...}, "recommendation": "car" \| "transit"}` |
| `rank_garages(destination: Location, arrival_time: datetime) -> list[dict]` | `src/recommendation/parking_score.py` | Garages sorted by score |
| `fetch_occupancy() -> list[dict]` | `src/data/parking_api.py` | Raw occupancy per garage: `garage_id`, `name`, `free_spots`, `capacity`, `timestamp` (UTC), `lat`, `lon` |
| `fetch_garages() -> list[Garage]` | `src/data/parking_api.py` | Garages for the app (cached 60 s, dummy fallback) |
| `get_transit_options(origin, destination, departure_time) -> dict \| None` | `src/data/transit_api.py` | Best ÖV connection: `duration_min`, `transfers`, `departure`, `arrival`, `sections` (cached 5 min) |
| `get_drive_time(origin, destination) -> float` | `src/data/routing.py` | Driving minutes (haversine estimate; pluggable provider) |
| `load_history(garage_id=None, since=None) -> DataFrame` | `src/data/database.py` | Logged occupancy from SQLite |

Places can be given as an address string, a `(lat, lon)` tuple or a `Location`.

Changing one of these signatures? Tell the team and update `tests/test_interfaces.py`.

## Responsibilities

| Area | Folder(s) | Responsible |
|---|---|---|
| Data sources & APIs | `src/data/parking_api.py`, `transit_api.py`, `weather_api.py`, `routing.py` | Person A |
| Database & data collection | `src/data/database.py`, `collector.py`, `data/` | Person B |
| Machine learning | `src/ml/`, `models/`, `notebooks/` | Person C |
| Recommendation logic | `src/recommendation/` | Person D |
| UI & app | `app.py`, `src/ui/` | Person E |
| Tests & docs | `tests/`, `docs/`, `README.md` | Everyone |
