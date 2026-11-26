import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from notebooks import generate_synthetic_data as gen
from src.data import database
from src.ml import train
from src.ml.features import FEATURES, TARGET

END = pd.Timestamp("2026-11-20 12:00", tz="UTC")
GARAGES = ["zuerichparkhausjelmoli", "zuerichparkhaushardauii", "zuerichparkhausopéra"]


def tiny_models() -> dict:
    return {
        "Linear regression": LinearRegression(),
        "Decision tree": DecisionTreeRegressor(max_depth=8, random_state=0),
    }


@pytest.fixture(scope="module")
def synthetic_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    db = tmp_path_factory.mktemp("data") / "synthetic.db"
    df = gen.generate(weeks=6, seed=1, end=END)
    gen.write_db(df[df["garage_id"].isin(GARAGES)], db, seed=1)
    return db


@pytest.fixture(scope="module")
def dataset(synthetic_db: Path) -> pd.DataFrame:
    return train.load_dataset(synthetic_db)


def test_split_by_time(dataset: pd.DataFrame) -> None:
    train_rows, validation, test = train.time_split(dataset)
    hour = pd.Timedelta(hours=1)
    # Even the answers (1 hour later) of one period lie before the next period.
    assert train_rows["time"].max() + hour < validation["time"].min() + pd.Timedelta(minutes=15)
    assert validation["time"].max() + hour < test["time"].min() + pd.Timedelta(minutes=15)
    assert (test["time"].max() - test["time"].min()).days == 14


def test_at_least_five_kinds_of_models() -> None:
    kinds = {name.split(" (")[0] for name in train.make_models()}
    assert kinds == {"Linear regression", "k-NN", "Decision tree", "Random forest", "Gradient boosting"}


def test_feature_columns() -> None:
    assert train.feature_columns() == FEATURES == ["free", "hour", "weekday", "garage_id", "capacity"]
    assert train.feature_columns(use_weather=True, use_holiday=True)[5:] == ["temperature_c", "rain_mm", "is_holiday"]


def test_missing_capacity_is_learned_from_training_data_only(dataset: pd.DataFrame) -> None:
    data = dataset.copy()
    data.loc[data["garage"] == GARAGES[0], "capacity"] = None
    train_rows, validation, _ = train.time_split(data)
    capacities, medians = train.learn_fill_values(train_rows, FEATURES)
    assert capacities[GARAGES[0]] == train_rows.loc[train_rows["garage"] == GARAGES[0], "free"].max()
    assert not train.prepare(validation, FEATURES, capacities, medians).isna().any().any()


def test_compare_models_sorts_by_validation_mae(dataset: pd.DataFrame) -> None:
    train_rows, validation, _ = train.time_split(dataset)
    results = train.compare_models(tiny_models(), train_rows[FEATURES], train_rows[TARGET],
                                   validation[FEATURES], validation[TARGET])
    assert set(results.index) == set(tiny_models())
    assert list(results["Validation MAE"]) == sorted(results["Validation MAE"])


def test_main_saves_model_plot_and_report(
    synthetic_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    model_path, docs = tmp_path / "model.pkl", tmp_path / "docs"
    monkeypatch.setattr(train, "make_models", tiny_models)
    monkeypatch.setattr(
        sys, "argv", ["train", "--db", str(synthetic_db), "--model-path", str(model_path), "--docs-dir", str(docs)]
    )
    train.main()

    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "Test MAE baseline" in out
    bundle = joblib.load(model_path)
    assert bundle["info"]["data_source"] == "synthetic"
    assert bundle["info"]["horizon_min"] == 60
    assert bundle["features"] == FEATURES and bundle["garages"] == sorted(GARAGES)
    assert bundle["info"]["test_mae"] > 0 and bundle["info"]["baseline_test_mae"] > 0
    svg = (docs / "feature_importance.svg").read_text()
    assert svg.startswith("<svg") and svg.count("<rect") == 1 + len(FEATURES)
    report = (docs / "model_report.md").read_text()
    assert "SYNTHETIC" in report and train.BASELINE in report


def test_weather_features_are_added(synthetic_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_weather(start, end, location=None):
        hours = pd.date_range(pd.Timestamp(start).floor("h"), end, freq="h")
        return pd.DataFrame({"time": hours, "temperature_c": 5.0, "precipitation_mm": 0.2})

    monkeypatch.setattr(train.weather_api, "get_hourly_weather", fake_weather)
    data = train.load_dataset(synthetic_db, use_weather=True)
    assert data["rain_mm"].notna().mean() > 0.99


def test_too_little_data_gives_a_helpful_error(tmp_path: Path) -> None:
    db = tmp_path / "small.db"
    gen.write_db(gen.generate(weeks=1, seed=1, end=END), db, seed=1)
    with pytest.raises(SystemExit, match="synthetic"):
        train.time_split(train.load_dataset(db))


def test_empty_database(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    database.init_db(db)
    with pytest.raises(SystemExit, match="No occupancy data"):
        train.load_dataset(db)
