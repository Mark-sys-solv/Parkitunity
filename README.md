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
pip install -r requirements.txt     # only streamlit, pandas, numpy, requests, scikit-learn
cp .env.example .env               # then set NOMINATIM_CONTACT (see "Data sources")
streamlit run app.py
```

No API keys needed. Parking occupancy ([ParkenDD](https://api.parkendd.de/Zuerich)) and
transit connections ([transport.opendata.ch](https://transport.opendata.ch)) are live; if the
parking API is down, the app falls back to built-in dummy garages.

### Other commands

```bash
pip install -r requirements-dev.txt      # once: adds pytest (not needed to run the app)
pytest                                   # run all tests offline (APIs are mocked)
RUN_LIVE_TESTS=1 pytest -m live          # additionally check the real APIs
python -m src.data.download_history     # download historical occupancy (see "Data sources")
python -m src.ml.train                   # train the occupancy model (see below)
```

## App (UI)

```
Sidebar: start point · destination (addresses, found via Nominatim) · departure time · max. price/h ·
         public transport toggle
Top:     "Was ist dir wichtig?" (priority) · top 3 garages as cards · car and public transport side by side with
         savings and buttons to Google Maps (driving to the garage, transit to the destination)
Middle:  map (st.pydeck_chart) – garages coloured by predicted availability, recommendation = bigger dot,
         destination = black dot; details of all garages in a table below
Bottom:  forecast for the selected garage (next 6 h, with the arrival) · car vs. public transport as bars
```

| File | Content |
|---|---|
| `app.py` | Layout only |
| `src/ui/backend.py` | All backend calls, cached with `st.cache_data` |
| `src/ui/sidebar.py`, `priority.py`, `map_view.py`, `charts.py`, `mode_cards.py`, `forecast.py` | UI components (`st.radio`, `st.pydeck_chart`, `st.altair_chart`, `st.vega_lite_chart`; pydeck and Altair come with Streamlit) |

**Placeholders still in use** (each is one switch, the return formats are final):

| What | Why | Switch |
|---|---|---|
| Hourly prices | ParkenDD has no prices → rough guesses per garage | `USE_PLACEHOLDER_PRICES` in `src/config.py` |

## Recommendation logic

The time a user enters is the **departure time**. `build_candidates` makes one row per garage:

```
arrival_time    = departure + drive_min                        (different for every garage)
free_at_arrival = predict_free(garage, arrival_time)
walk_min        = straight line x 1.3 at 4.5 km/h, rounded up
```

`rank_garages` scales travel time (drive + walk), price per hour and free spots to 0–1 (min-max) and
adds them up with the weights of the chosen priority ("Was ist dir wichtig?": fastest, cheapest,
most_spots, balanced). Two sliders: garages further than the longest acceptable walk (default 20 min) come
near the end, and more than "enough" free spots (default 200) do not improve the score. Garages with fewer than 10 expected free spots come last. The 0–1 scale only uses
suitable garages (close enough, enough spots). Garages above the
price limit are not ranked and only shown in grey.

`compare_modes` compares rank 1 of this ranking with the next public transport connection:

```
car:     time = drive + walk             costs = 2 x km x cost per km + price per hour x length of stay
transit: time = waiting + travel time    costs = 2 x ticket (simplified)
```

The priority decides: fastest = shorter time, cheapest = lower costs, otherwise both count half (as shares
of the sum). With a tie or without transit data nothing is recommended. **All assumptions** (speeds, weights, fares, cost per km, …) are commented
constants in [src/config.py](src/config.py) under "Recommendation: assumptions".

## Occupancy forecast (ML)

The model answers: **how many spaces will be free in this garage in 1 hour?** (regression, target
`free_in_1h`). The table has one row per garage per 15 minutes; features are `free`, `hour`, `weekday`,
`garage_id` and `capacity` (optional: weather, holidays). Results: [docs/model_report.md](docs/model_report.md).

The model is trained on **historical data only** (see "Data sources"): download it once, then train.

```bash
python -m src.data.download_history             # -> history.csv + data/parkitunity.db (several hours)
python -m src.ml.train                          # train on the real history (data/parkitunity.db)
python -m src.ml.train --weather --holiday      # also use weather (Open-Meteo) and public holidays
```

For quick experiments without downloading, there is **synthetic development data**
(clearly marked, stored in a separate file, never real results):

```bash
python notebooks/generate_synthetic_data.py     # -> data/synthetic.db (8 weeks, real garage ids)
python -m src.ml.train --db data/synthetic.db   # -> models/model.pkl + docs/
PARKITUNITY_DB_PATH=data/synthetic.db streamlit run app.py   # app shows a "synthetic" warning
```

The data is split **by time** (train = oldest, validation = 2 weeks, test = newest 2 weeks; never
shuffled). Five kinds of scikit-learn regressors are compared on the validation set by MAE; only the
best one is tested, once, next to the baseline "same as now". The tested model is saved with `joblib`
to `models/` (git-ignored, training takes about 1 min), the report to `docs/`. `joblib` is installed
together with scikit-learn.

For arrivals other than 1 hour ahead, `predict_free` uses provisional rules (see the docstring of
`src/ml/predict.py`); the group still has to decide on this.

## Data sources

| Data | Source | Used for |
|---|---|---|
| Historical occupancy (every ~5 min since March 2021) | [ParkenDD API](https://api.parkendd.de/Zuerich), documented by [Open Data Zürich](https://opendatazurich.github.io/parkendd-api) – data from PLS Parkleitsystem Zürich AG (CC-0) | **Training the model** |
| Current occupancy | same API, `/Zuerich` | Input for the forecast in the app |
| Public transport | [transport.opendata.ch](https://transport.opendata.ch) | Car vs. public transport |
| Weather (optional) | [Open-Meteo](https://open-meteo.com) | Optional model features |
| Address search | [Nominatim](https://nominatim.openstreetmap.org) – © OpenStreetMap contributors (ODbL) | Start point and destination → coordinates |
| Navigation | [Google Maps URLs](https://developers.google.com/maps/documentation/urls/get-started) (link only, no API key) | "Autoroute öffnen" / "ÖV-Route öffnen" |

**Address search (Nominatim):** we follow the
[Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/) – our own
User-Agent with project name and contact, at most 1 request per second, results cached for a week,
search limited to Switzerland (Zurich preferred), attribution in the app footer. Put a contact
(e-mail or project URL) into `.env` as `NOMINATIM_CONTACT`; see `src/data/geocoding.py`.

The forecast model is trained **only on historical data** – there is no ongoing data collection.
`python -m src.data.download_history` fetches the history of all Zurich garages (default: last
12 months; `--start` / `--end` to change) and writes:

- `data/processed/history.csv` – `timestamp` (Zurich time with UTC offset), `garage_id`,
  `garage_name`, `free`, `total` (~150 MB, git-ignored)
- `data/garages.csv` – `id`, `name`, `lat`, `lon`, `total`
- the same history into `data/parkitunity.db`, which training and the app read

Good to know about the API: at most 7 days per request and ~10 s per request, so 12 months for
all garages take **several hours**. That is why every finished garage is saved right away in
`data/raw/history_<garage_id>.csv`. If the download stops, just start it again: garages that
already have a file are skipped, and at the end all files are merged into `history.csv`.
To download a different period, delete `data/raw/history_*.csv` first.
There is no data from 18 July to 18 October 2026.
Run the download again now and then, so that the app's "typical week" stays up to date.

## Project structure

```
app.py                  Streamlit entry point (UI composition only)
src/config.py           Paths, API URLs, constants
src/models.py           Shared data types (Garage, Location, ModeComparison, …)
src/data/               External APIs, SQLite database, history download
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
| `predict_free(garage_id, timestamps) -> array` | `src/ml/predict.py` | Predicted free spots, the only source of forecasts (model for 1 h ahead, line from now until then, typical week beyond) |
| `forecast_curve(garage_id) -> DataFrame` | `src/ml/predict.py` | `timestamp`, `free_pred` for the next 6 h in 15-min steps (via `predict_free`) |
| `build_table(history) -> DataFrame` | `src/ml/features.py` | One row per garage per 15 min, with target `free_in_1h` |
| `add_features(table, garages) -> DataFrame` | `src/ml/features.py` | Adds the features (shared by training and prediction) |
| `get_hourly_weather(start, end) -> DataFrame` | `src/data/weather_api.py` | Hourly temperature/precipitation (past and forecast) |
| `compare_modes(best, transit, parking_hours, priority) -> dict` | `src/recommendation/mode_compare.py` | `{"car": {...}, "transit": {...}, "recommendation": "car" \| "transit" \| None}` (fields: `src/models.py`) |
| `build_candidates(start, ziel, departure=None) -> DataFrame` | `src/recommendation/candidates.py` | One row per garage: live data, travel times, price, forecast at arrival |
| `rank_garages(candidates, priority) -> DataFrame` | `src/recommendation/parking_score.py` | Same rows sorted, with `score` and `rank` |
| `fetch_occupancy() -> list[dict]` | `src/data/parking_api.py` | Raw occupancy per garage: `garage_id`, `name`, `free_spots`, `capacity`, `timestamp` (UTC), `lat`, `lon` |
| `fetch_garages() -> list[Garage]` | `src/data/parking_api.py` | Garages for the app (cached 60 s, dummy fallback) |
| `get_transit_options(origin, destination, departure_time) -> dict \| None` | `src/data/transit_api.py` | Best ÖV connection: `duration_min`, `transfers`, `departure`, `arrival`, `sections` (cached 5 min) |
| `get_drive_time(origin, destination) -> float` | `src/data/routing.py` | Driving minutes (haversine estimate; pluggable provider) |
| `geocode(address) -> (lat, lon) \| None` | `src/data/geocoding.py` | Coordinates of an address in Switzerland (Nominatim, cached, max. 1 request/s) |
| `load_history(garage_id=None, since=None) -> DataFrame` | `src/data/database.py` | Historical occupancy from SQLite |

For transit connections, places can be given as an address string, a `(lat, lon)` tuple or a `Location`.

Changing one of these signatures? Tell the team and update `tests/test_interfaces.py`.

## Responsibilities

| Area | Folder(s) | Responsible |
|---|---|---|
| Data sources & APIs | `src/data/parking_api.py`, `transit_api.py`, `weather_api.py`, `routing.py`, `geocoding.py` | Person A |
| Database & historical data | `src/data/database.py`, `download_history.py`, `data/` | Person B |
| Machine learning | `src/ml/`, `models/`, `notebooks/` | Person C |
| Recommendation logic | `src/recommendation/` | Person D |
| UI & app | `app.py`, `src/ui/` | Person E |
| Tests & docs | `tests/`, `docs/`, `README.md` | Everyone |
