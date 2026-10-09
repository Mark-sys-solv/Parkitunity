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

The app currently runs entirely on **dummy data** – no API keys needed.

### Other commands

```bash
pytest                                   # run interface tests
python -m src.data.collector             # log garage occupancy once
python -m src.data.collector --loop      # log every 10 minutes
python -m src.ml.train                   # train the occupancy model
```

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
| `fetch_garages() -> list[Garage]` | `src/data/parking_api.py` | Current garage occupancy |
| `get_connections(origin, destination, time) -> list[TransitConnection]` | `src/data/transit_api.py` | ÖV connections |

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
