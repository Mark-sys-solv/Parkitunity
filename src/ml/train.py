# AI citation: this file was written with Claude (Anthropic), following the group's document
# "Parkitunity: The Machine Learning Pattern" (FCS Assignments 10 and 11).
"""Train the model: how many spaces will be free in 1 hour?

Usage:
    python -m src.ml.train                          # real data (config.DB_PATH)
    python -m src.ml.train --db data/synthetic.db   # development with synthetic data
    python -m src.ml.train --weather --holiday      # optional extra features

Steps (the Assignment 11 pattern):
1. Build the table: one row per garage per 15 minutes, target ``free_in_1h``.
2. Split by TIME, never randomly: oldest = train, next = validation, newest = test.
   (Rows 15 minutes apart are nearly identical - shuffling would leak the answer.)
3. Train several models and compare them on the validation set with the MAE.
4. Report only the best model, once, on the test set - next to the baseline
   "in 1 hour it will be the same as now".
5. Save the model with joblib, so the app does not have to train.
"""

from __future__ import annotations

import argparse
import html
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import DB_PATH, DOCS_DIR, FORECAST_HORIZON_MIN, MODEL_PATH, SYNTHETIC_DB_PATH, TEST_DAYS, VALIDATION_DAYS
from src.data import database, weather_api
from src.ml.features import (
    FEATURE_LABELS,
    FEATURES,
    HOLIDAY_FEATURES,
    TARGET,
    WEATHER_FEATURES,
    add_features,
    build_table,
)

logger = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 1_000
MIN_EVAL_ROWS = 200  # for validation and test
SAMPLE_ROWS = 20_000  # rows used for the train MAE and the feature importance (to save time)
BASELINE = "Baseline: same as now"


def make_models() -> dict[str, Any]:
    """The models we compare. The numbers in the names are hyperparameters.

    Only k-NN and linear regression need scaling. ``make_pipeline`` puts the
    scaler and the model together: the scaler is fitted on the same data as
    the model, so on the training data only.
    """
    return {
        "Linear regression": make_pipeline(RobustScaler(), LinearRegression()),
        "k-NN (k=5)": make_pipeline(RobustScaler(), KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
        "k-NN (k=20)": make_pipeline(RobustScaler(), KNeighborsRegressor(n_neighbors=20, n_jobs=-1)),
        "Decision tree (depth 5)": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Decision tree (depth 8)": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Decision tree (depth 12)": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random forest (depth 8)": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        "Random forest (depth 12)": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient boosting": GradientBoostingRegressor(random_state=42),
    }


def feature_columns(use_weather: bool = False, use_holiday: bool = False) -> list[str]:
    """Names of the columns the model may look at."""
    columns = list(FEATURES)
    if use_weather:
        columns += WEATHER_FEATURES
    if use_holiday:
        columns += HOLIDAY_FEATURES
    return columns


def load_dataset(db_path: Path, use_weather: bool = False) -> pd.DataFrame:
    """Load the history and return the finished table (rows without free or target are dropped)."""
    history = database.load_history(db_path=db_path)
    if history.empty:
        raise SystemExit(f"No occupancy data in {db_path}. {_DATA_HINT}")

    weather = None
    if use_weather:
        weather = weather_api.get_hourly_weather(
            history["timestamp"].min().to_pydatetime(), history["timestamp"].max().to_pydatetime()
        )
        if weather.empty:
            raise SystemExit("Weather data unavailable - run again later or without --weather.")

    table = build_table(history).dropna(subset=["free", TARGET])
    garages = sorted(table["garage"].unique())
    return add_features(table, garages, weather).reset_index(drop=True)


def time_split(
    data: pd.DataFrame, validation_days: int = VALIDATION_DAYS, test_days: int = TEST_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time into train (oldest), validation and test (newest).

    A row also needs its answer (1 hour later) inside its own period, so the
    last hour before each border is left out.
    """
    answer_time = data["time"] + pd.Timedelta(minutes=FORECAST_HORIZON_MIN)
    test_start = data["time"].max() - pd.Timedelta(days=test_days)
    validation_start = test_start - pd.Timedelta(days=validation_days)
    train = data[answer_time < validation_start]
    validation = data[(data["time"] >= validation_start) & (answer_time < test_start)]
    test = data[data["time"] >= test_start]
    if len(train) < MIN_TRAIN_ROWS or min(len(validation), len(test)) < MIN_EVAL_ROWS:
        raise SystemExit(
            f"Not enough data to train (train={len(train)}, validation={len(validation)}, test={len(test)} rows). {_DATA_HINT}"
        )
    return train, validation, test


def learn_fill_values(train: pd.DataFrame, columns: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """What to use for missing values, learned from the TRAINING data only.

    Returns:
        (capacity per garage, median per column). A garage without a known
        capacity gets the largest number of free spaces we have seen there.
    """
    per_garage = train.groupby("garage").agg(capacity=("capacity", "last"), most_free=("free", "max"))
    capacities = per_garage["capacity"].fillna(per_garage["most_free"]).to_dict()
    return capacities, train[columns].median().to_dict()


def prepare(data: pd.DataFrame, columns: list[str], capacities: dict[str, float], medians: dict[str, float]) -> pd.DataFrame:
    """The feature columns (X) without missing values."""
    x = data[columns].copy()
    x["capacity"] = x["capacity"].fillna(data["garage"].map(capacities))
    return x.fillna(medians).astype(float)


def compare_models(
    models: dict[str, Any], x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series
) -> pd.DataFrame:
    """Train every model and compare them on the validation set.

    Returns:
        Table with the MAE on train and validation, best model first. A train
        MAE far below the validation MAE is the sign of overfitting.
    """
    sample = x_train.sample(min(SAMPLE_ROWS, len(x_train)), random_state=42)
    rows = {}
    for name, model in models.items():
        logger.info("Training %s ...", name)
        model.fit(x_train, y_train)
        rows[name] = {
            "Train MAE": mean_absolute_error(y_train.loc[sample.index], model.predict(sample)),
            "Validation MAE": mean_absolute_error(y_val, model.predict(x_val)),
        }
    return pd.DataFrame(rows).T.sort_values("Validation MAE")


def feature_importance(model: Any, x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """How much the MAE grows when the values of one feature are shuffled (biggest first)."""
    sample = x.sample(min(SAMPLE_ROWS, len(x)), random_state=42)
    result = permutation_importance(
        model, sample, y.loc[sample.index], scoring="neg_mean_absolute_error", n_repeats=5, random_state=42
    )
    return pd.Series(result.importances_mean, index=x.columns).clip(lower=0).sort_values(ascending=False)


def _format_spots(value: float) -> str:
    """E.g. "+12", "+5.7" or "+0.34"."""
    return f"+{value:.0f}" if value >= 10 else f"+{value:.1f}" if value >= 1 else f"+{value:.2f}"


def save_importance_svg(importance: pd.Series, title: str, subtitle: str, path: Path) -> None:
    """Save the feature importance as a horizontal bar chart (SVG file).

    SVG is plain text, so we can write the chart ourselves without a plotting
    library: one ``<rect>`` per bar plus ``<text>`` labels.
    """
    label_width, bar_area, bar_height, row_height, top = 200, 420, 16, 26, 70
    width = label_width + bar_area + 70
    height = top + row_height * len(importance) + 40
    biggest = max(importance.max(), 1e-9)  # avoid dividing by zero

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'font-family="sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="#fcfcfb"/>',
        f'<text x="10" y="26" font-size="16" font-weight="bold" fill="#0b0b0b">{html.escape(title)}</text>',
        f'<text x="10" y="46" fill="#52514e">{html.escape(subtitle)}</text>',
    ]
    for row, (feature, value) in enumerate(importance.items()):
        y = top + row * row_height
        bar = value / biggest * bar_area
        label = html.escape(FEATURE_LABELS.get(feature, feature))
        parts.append(f'<text x="{label_width - 8}" y="{y + 12}" text-anchor="end" fill="#52514e">{label}</text>')
        parts.append(f'<rect x="{label_width}" y="{y}" width="{bar:.1f}" height="{bar_height}" rx="3" fill="#2a78d6"/>')
        parts.append(f'<text x="{label_width + bar + 6:.1f}" y="{y + 12}" fill="#52514e">{_format_spots(value)}</text>')
    parts.append(f'<text x="{label_width}" y="{height - 12}" fill="#898781">'
                 "Extra error (MAE, free spots) when the feature is shuffled</text>")
    parts.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_report(results: pd.DataFrame, importance: pd.Series, info: dict[str, Any], path: Path) -> None:
    """Write the results as a Markdown report."""
    lines = ["# Occupancy model - evaluation", ""]
    if info["data_source"] == "synthetic":
        lines += ["> ⚠️ **Trained on SYNTHETIC data** - for development only, not real results.", ""]
    lines += [
        f"- Target: free spaces **{info['horizon_min']} min** ahead, per garage (regression)",
        f"- Data: {info['data_source']}, {info['n_garages']} garages, one row per garage per 15 minutes",
        f"- Features: {', '.join(info['features'])}",
        "- Split by time (no shuffling):",
        *[f"  - {name}: {first} to {last}, {rows:,} rows" for name, (first, last, rows) in info["periods"].items()],
        f"- Trained: {info['trained_at']}",
        "",
        "## Comparison on the validation set (MAE in spaces, lower is better)",
        "",
        "| Model | Train MAE | Validation MAE |",
        "|---|---:|---:|",
        *[f"| {name} | {r['Train MAE']:.2f} | {r['Validation MAE']:.2f} |" for name, r in results.iterrows()],
        f"| {BASELINE} | {info['baseline_train_mae']:.2f} | {info['baseline_val_mae']:.2f} |",
        "",
        "## Final result on the test set (used once, best model only)",
        "",
        "| | Test MAE |",
        "|---|---:|",
        f"| **{info['model_name']}** | **{info['test_mae']:.2f}** |",
        f"| {BASELINE} | {info['baseline_test_mae']:.2f} |",
        "",
        f"The model is off by {info['test_mae']:.1f} spaces on average, the baseline by {info['baseline_test_mae']:.1f}.",
        "",
        "## Feature importance",
        "",
        "How much larger the MAE gets on the validation set when a feature is shuffled.",
        "",
        "![Feature importance](feature_importance.svg)",
        "",
        "| Feature | Extra MAE when shuffled |",
        "|---|---:|",
        *[f"| {FEATURE_LABELS[c]} | {_format_spots(v)} |" for c, v in importance.items()],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the whole training (see the steps at the top of the file)."""
    parser = argparse.ArgumentParser(description="Train the parking occupancy forecast.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help=f"SQLite file (default {DB_PATH})")
    parser.add_argument("--weather", action="store_true", help="add temperature and rain as features")
    parser.add_argument("--holiday", action="store_true", help="add public holidays as a feature")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    synthetic = database.is_synthetic(args.db)
    if synthetic:
        print("⚠️  Training on SYNTHETIC data - results are for development only!\n")

    # 1. Table and 2. split by time
    data = load_dataset(args.db, args.weather)
    columns = feature_columns(args.weather, args.holiday)
    train, validation, test = time_split(data)
    capacities, medians = learn_fill_values(train, columns)
    x_train, y_train = prepare(train, columns, capacities, medians), train[TARGET]
    x_val, y_val = prepare(validation, columns, capacities, medians), validation[TARGET]
    x_test, y_test = prepare(test, columns, capacities, medians), test[TARGET]

    # 3. Compare the models on the validation set
    models = make_models()
    results = compare_models(models, x_train, y_train, x_val, y_val)
    best_name = results.index[0]
    best = models[best_name]
    baseline_val_mae = mean_absolute_error(y_val, validation["free"])
    print(f"\nMAE in spaces, {FORECAST_HORIZON_MIN} min ahead:\n")
    print(results.to_string(float_format="{:.2f}".format))
    print(f"{BASELINE}: validation MAE {baseline_val_mae:.2f}")
    if results["Validation MAE"].iloc[0] >= baseline_val_mae:
        print("⚠️  No model beats the baseline!")

    # 4. Only the best model is tested, and only once
    test_mae = mean_absolute_error(y_test, best.predict(x_test))
    baseline_test_mae = mean_absolute_error(y_test, test["free"])
    print(f"\nBest model: {best_name}")
    print(f"Test MAE:          {test_mae:.2f} spaces")
    print(f"Test MAE baseline: {baseline_test_mae:.2f} spaces")

    info: dict[str, Any] = {
        "model_name": best_name,
        "data_source": "synthetic" if synthetic else "real",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon_min": FORECAST_HORIZON_MIN,
        "features": columns,
        "uses_weather": args.weather,
        "n_garages": data["garage"].nunique(),
        "periods": {
            name: (f"{part['time'].min():%Y-%m-%d}", f"{part['time'].max():%Y-%m-%d}", len(part))
            for name, part in [("Train", train), ("Validation", validation), ("Test", test)]
        },
        "test_mae": float(test_mae),
        "baseline_test_mae": float(baseline_test_mae),
        "baseline_val_mae": float(baseline_val_mae),
        "baseline_train_mae": float(mean_absolute_error(y_train, train["free"])),
    }

    # 5. Save the tested model, so the test MAE describes exactly the model in the app
    bundle = {
        "model": best,
        "features": columns,
        "garages": sorted(data["garage"].unique()),  # position = garage_id
        "capacities": capacities,
        "medians": medians,
        "info": info,
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_path, compress=3)
    print(f"Saved model to {args.model_path}")

    importance = feature_importance(best, x_val, y_val)
    subtitle = f"{best_name}, forecast {FORECAST_HORIZON_MIN} min ahead, validation set · " + (
        "SYNTHETIC DATA – development only" if synthetic else "real data"
    )
    save_importance_svg(importance, "What the model looks at", subtitle, args.docs_dir / "feature_importance.svg")
    write_report(results, importance, info, args.docs_dir / "model_report.md")
    print(f"Saved report to {args.docs_dir}/")


_DATA_HINT = (
    f"The model needs more than {VALIDATION_DAYS + TEST_DAYS} days of data (validation + test period). "
    "Download the history first: python -m src.data.download_history - or, for quick development, "
    "use synthetic data: python notebooks/generate_synthetic_data.py && "
    f"python -m src.ml.train --db {SYNTHETIC_DB_PATH.relative_to(SYNTHETIC_DB_PATH.parents[1])}"
)


if __name__ == "__main__":
    main()
