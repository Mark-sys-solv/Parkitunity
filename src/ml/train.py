"""Train the occupancy forecast: free spots per garage in 30 minutes.

Usage:
    python -m src.ml.train                          # real data (config.DB_PATH)
    python -m src.ml.train --db data/synthetic.db   # development with synthetic data
    python -m src.ml.train --weather                # add temperature/precipitation features

Steps:
1. Load the occupancy history and build features (``features.build_features``).
2. Split by time: the last ``TEST_FRACTION`` of the period is the test set.
   Training rows only use targets *before* the test period starts, so the
   model never sees the future (no random split!).
3. Compare two baselines and two models on the test set (MAE, RMSE):
   - Baseline "same time last week" (the value 7 days before the target time)
   - Baseline "stays the same" (the current value; a stricter benchmark)
   - RandomForestRegressor and GradientBoostingRegressor
4. Save the (permutation) feature importance of the best model as SVG chart to ``docs/``.
5. Refit the best model on all data and save it (pickle + gzip) to ``models/``.
"""

from __future__ import annotations

import argparse
import gzip
import html
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance

from src.config import (
    DB_PATH,
    DOCS_DIR,
    FEATURE_FREQ_MIN,
    FORECAST_HORIZON_MIN,
    MODEL_PATH,
    SYNTHETIC_DB_PATH,
    TEST_FRACTION,
)
from src.data import database, weather_api
from src.ml.features import (
    CALENDAR_FEATURES,
    FEATURE_LABELS,
    STATE_FEATURES,
    TARGET_COLUMN,
    WEATHER_FEATURES,
    build_features,
)

logger = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 1_000
MIN_TEST_ROWS = 200
BASELINE = "Baseline: same time last week"
PERSISTENCE = "Baseline: stays the same"


def make_models(random_state: int = 42) -> dict[str, RegressorMixin]:
    """The candidate models (hyperparameters chosen for accuracy vs. file size)."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_leaf=10,
            max_features=0.6,
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            random_state=random_state,
        ),
    }


def feature_columns(use_weather: bool) -> list[str]:
    """Columns the model is trained on (``garage_code`` identifies the garage)."""
    return CALENDAR_FEATURES + STATE_FEATURES + (WEATHER_FEATURES if use_weather else []) + ["garage_code"]


def load_dataset(db_path: Path, use_weather: bool = False) -> pd.DataFrame:
    """Load history from ``db_path`` and return usable feature rows.

    Rows without a target, current value or last-week value are dropped
    (the first week of data therefore never appears in training).
    """
    history = database.load_history(db_path=db_path)
    if history.empty:
        raise SystemExit(f"No occupancy data in {db_path}. {_DATA_HINT}")

    weather = None
    if use_weather:
        weather = weather_api.get_hourly_weather(
            history["timestamp"].min().to_pydatetime(),
            (history["timestamp"].max() + pd.Timedelta(minutes=FORECAST_HORIZON_MIN)).to_pydatetime(),
        )
        if weather.empty:
            raise SystemExit("Weather data unavailable – run again later or without --weather.")

    data = build_features(history, weather=weather)
    return data.dropna(subset=[TARGET_COLUMN, "free_now", "lag_1w"]).reset_index(drop=True)


def time_split(data: pd.DataFrame, test_fraction: float = TEST_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: training targets end where the test period begins."""
    times = np.sort(data["timestamp"].unique())
    cutoff = times[int(len(times) * (1 - test_fraction))]
    train = data[data["target_time"] <= cutoff]
    test = data[data["timestamp"] >= cutoff]
    return train, test


def metrics(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """Mean absolute error and root mean squared error (in free spots)."""
    error = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {"MAE": float(np.mean(np.abs(error))), "RMSE": float(np.sqrt(np.mean(error**2)))}


def train_and_evaluate(
    data: pd.DataFrame,
    columns: list[str],
    models: dict[str, RegressorMixin],
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, dict[str, RegressorMixin], pd.DataFrame]:
    """Fit all models on the training period and score everything on the test period.

    Returns:
        (results table sorted by MAE, fitted models by name, test rows with
        missing values filled – used for the feature importance)
    """
    train, test = time_split(data, test_fraction)
    if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
        days = (data["timestamp"].max() - data["timestamp"].min()).days if len(data) else 0
        raise SystemExit(
            f"Not enough data to train (train={len(train)}, test={len(test)} rows, ~{days} usable days). {_DATA_HINT}"
        )
    logger.info("Train: %s – %s (%d rows)", train["timestamp"].min(), train["timestamp"].max(), len(train))
    logger.info("Test:  %s – %s (%d rows)", test["timestamp"].min(), test["timestamp"].max(), len(test))

    fill = train[columns].median()
    x_train, x_test = train[columns].fillna(fill), test[columns].fillna(fill)
    y_test = test[TARGET_COLUMN]

    rows = {BASELINE: metrics(y_test, test["lag_1w"]), PERSISTENCE: metrics(y_test, test["free_now"])}
    fitted: dict[str, RegressorMixin] = {}
    for name, model in models.items():
        logger.info("Training %s ...", name)
        model.fit(x_train, train[TARGET_COLUMN])
        fitted[name] = model
        rows[name] = metrics(y_test, model.predict(x_test))

    results = pd.DataFrame(rows).T
    results["MAE vs. baseline"] = (results["MAE"] / results.loc[BASELINE, "MAE"] - 1).map("{:+.0%}".format)
    test_filled = x_test.assign(**{TARGET_COLUMN: y_test})
    return results.sort_values("MAE"), fitted, test_filled


def feature_importance(model: RegressorMixin, test: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Permutation importance on the test set, sorted descending.

    For each feature, its values are shuffled (so the model can no longer use
    it) and we measure how much the MAE grows – in free spots. Unlike the
    built-in impurity importance this is measured on unseen data and is not
    biased towards features with many distinct values.
    """
    result = permutation_importance(
        model,
        test[columns],
        test[TARGET_COLUMN],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=columns).clip(lower=0).sort_values(ascending=False)


def _format_spots(value: float) -> str:
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
    """Write metrics and feature importance as a Markdown report."""
    lines = [
        "# Occupancy model – evaluation",
        "",
        "> ⚠️ **Trained on SYNTHETIC data** – for development only, not real results." if info["data_source"] == "synthetic" else "",
        "",
        f"- Target: free spots **{info['horizon_min']} min** ahead, per garage",
        f"- Data: {info['data_source']} ({info['period'][0]} – {info['period'][1]}, {info['n_rows']:,} rows, {info['n_garages']} garages)",
        f"- Test set: last {TEST_FRACTION:.0%} of the period (time-based split)",
        f"- Best model: **{info['model_name']}** (refit on all data and saved)",
        f"- Trained: {info['trained_at']}",
        "",
        "## Metrics on the test set (in free spots, lower is better)",
        "",
        "| Model | MAE | RMSE | MAE vs. baseline |",
        "|---|---:|---:|---:|",
        *[f"| {name} | {r['MAE']:.2f} | {r['RMSE']:.2f} | {r['MAE vs. baseline']} |" for name, r in results.iterrows()],
        "",
        "## Feature importance",
        "",
        "Permutation importance on the test set: how much larger the MAE gets (in free spots)",
        "when a feature's values are shuffled, i.e. when the model can no longer use it.",
        "Features that do not vary in the test period (e.g. no holiday in it) show 0.",
        "",
        "![Feature importance](feature_importance.svg)",
        "",
        "| Feature | Extra MAE when shuffled |",
        "|---|---:|",
        *[f"| {FEATURE_LABELS.get(c, c)} | {_format_spots(v)} |" for c, v in importance.items()],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the parking occupancy forecast.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help=f"SQLite file (default {DB_PATH})")
    parser.add_argument("--weather", action="store_true", help="use temperature/precipitation features")
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    synthetic = database.is_synthetic(args.db)
    if synthetic:
        print("⚠️  Training on SYNTHETIC data – results are for development only!\n")

    data = load_dataset(args.db, args.weather)
    garage_codes = {g: i for i, g in enumerate(sorted(data["garage_id"].unique()))}
    data["garage_code"] = data["garage_id"].map(garage_codes)
    columns = feature_columns(args.weather)

    results, fitted, test = train_and_evaluate(data, columns, make_models(), args.test_fraction)
    best_name = next(name for name in results.index if name in fitted)

    print(f"\nTest-set errors in free spots (forecast {FORECAST_HORIZON_MIN} min ahead):\n")
    print(results.to_string(float_format="{:.2f}".format))
    print(f"\nBest model: {best_name}")

    # Refit the winner on all data so the saved model knows the most recent weeks.
    final = clone(fitted[best_name])
    fill = data[columns].median()
    final.fit(data[columns].fillna(fill), data[TARGET_COLUMN])

    info: dict[str, Any] = {
        "model_name": best_name,
        "data_source": "synthetic" if synthetic else "real",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon_min": FORECAST_HORIZON_MIN,
        "freq_min": FEATURE_FREQ_MIN,
        "uses_weather": args.weather,
        "period": (f"{data['timestamp'].min():%Y-%m-%d}", f"{data['timestamp'].max():%Y-%m-%d}"),
        "n_rows": len(data),
        "n_garages": len(garage_codes),
        "metrics": results[["MAE", "RMSE"]].to_dict("index"),
    }
    bundle = {
        "model": final,
        "feature_columns": columns,
        "fill_values": fill.to_dict(),
        "garage_codes": garage_codes,
        "info": info,
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.model_path, "wb") as f:  # compressed: a random forest is large
        pickle.dump(bundle, f)
    print(f"Saved model to {args.model_path}")

    importance = feature_importance(fitted[best_name], test, columns)
    subtitle = f"{best_name}, forecast {FORECAST_HORIZON_MIN} min ahead, test set · " + (
        "SYNTHETIC DATA – development only" if synthetic else f"real data {info['period'][0]} – {info['period'][1]}"
    )
    save_importance_svg(importance, "What the model looks at", subtitle, args.docs_dir / "feature_importance.svg")
    write_report(results, importance, info, args.docs_dir / "model_report.md")
    print(f"Saved feature importance plot and report to {args.docs_dir}/")


_DATA_HINT = (
    "The model needs at least ~2 weeks of data (1 week for the 'last week' feature + test period). "
    "Download the history first: python -m src.data.download_history – or, for quick development, "
    "use synthetic data: "
    f"python notebooks/generate_synthetic_data.py && python -m src.ml.train --db {SYNTHETIC_DB_PATH.relative_to(SYNTHETIC_DB_PATH.parents[1])}"
)


if __name__ == "__main__":
    main()
