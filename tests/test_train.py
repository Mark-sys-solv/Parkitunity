import gzip
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from notebooks import generate_synthetic_data as gen
from src.data import database
from src.ml import train
from src.ml.features import STATE_FEATURES

END = pd.Timestamp("2026-11-20 12:00", tz="UTC")
GARAGES = ["zuerichparkhausjelmoli", "zuerichparkhaushardauii", "zuerichparkhausopéra"]


def tiny_models() -> dict:
    return {
        "Random Forest": RandomForestRegressor(n_estimators=20, max_depth=8, random_state=0),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=0),
    }


@pytest.fixture(scope="module")
def synthetic_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    db = tmp_path_factory.mktemp("data") / "synthetic.db"
    df = gen.generate(weeks=3, seed=1, end=END)
    gen.write_db(df[df["garage_id"].isin(GARAGES)], db, seed=1)
    return db


@pytest.fixture(scope="module")
def dataset(synthetic_db: Path) -> pd.DataFrame:
    data = train.load_dataset(synthetic_db)
    data["garage_code"] = data["garage_id"].map({g: i for i, g in enumerate(sorted(GARAGES))})
    return data


def test_time_split_never_trains_on_the_test_period(dataset: pd.DataFrame) -> None:
    train_rows, test_rows = train.time_split(dataset, 0.2)
    assert train_rows["target_time"].max() <= test_rows["timestamp"].min()
    assert len(test_rows) / len(dataset) == pytest.approx(0.2, abs=0.03)


def test_models_beat_weekly_baseline(dataset: pd.DataFrame) -> None:
    results, fitted, test = train.train_and_evaluate(dataset, train.feature_columns(False), tiny_models())
    assert set(results.index) == {train.BASELINE, train.PERSISTENCE, "Random Forest", "Gradient Boosting"}
    assert {"MAE", "RMSE"} <= set(results.columns)
    for name in fitted:
        assert results.loc[name, "MAE"] < results.loc[train.BASELINE, "MAE"]
    assert not test[train.feature_columns(False)].isna().any().any()


def test_feature_importance_covers_all_features(dataset: pd.DataFrame) -> None:
    columns = train.feature_columns(False)
    _, fitted, test = train.train_and_evaluate(dataset, columns, tiny_models())
    importance = train.feature_importance(fitted["Random Forest"], test, columns)
    assert set(importance.index) == set(columns)
    assert importance.index[0] in STATE_FEATURES  # recent occupancy matters most for 30 min ahead
    assert (importance >= 0).all()


def test_main_saves_model_plot_and_report(
    synthetic_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    model_path, docs = tmp_path / "model.pkl.gz", tmp_path / "docs"
    monkeypatch.setattr(train, "make_models", tiny_models)
    monkeypatch.setattr(
        sys, "argv", ["train", "--db", str(synthetic_db), "--model-path", str(model_path), "--docs-dir", str(docs)]
    )
    train.main()

    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "MAE" in out
    with gzip.open(model_path, "rb") as f:
        bundle = pickle.load(f)
    assert {"model", "feature_columns", "fill_values", "garage_codes", "info"} <= bundle.keys()
    assert bundle["info"]["data_source"] == "synthetic"
    assert bundle["info"]["horizon_min"] == 30
    svg = (docs / "feature_importance.svg").read_text()
    assert svg.startswith("<svg") and svg.count("<rect") > 10  # background + one bar per feature
    assert "SYNTHETIC" in (docs / "model_report.md").read_text()


def test_weather_features_are_added(synthetic_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_weather(start, end, location=None):
        hours = pd.date_range(pd.Timestamp(start).floor("h"), end, freq="h")
        return pd.DataFrame({"time": hours, "temperature_c": 5.0, "precipitation_mm": 0.2})

    monkeypatch.setattr(train.weather_api, "get_hourly_weather", fake_weather)
    data = train.load_dataset(synthetic_db, use_weather=True)
    assert data["temperature_c"].notna().mean() > 0.99
    assert "temperature_c" in train.feature_columns(True)


def test_too_little_data_gives_a_helpful_error(tmp_path: Path) -> None:
    db = tmp_path / "small.db"
    gen.write_db(gen.generate(weeks=1, seed=1, end=END), db, seed=1)
    data = train.load_dataset(db)
    data["garage_code"] = 0
    with pytest.raises(SystemExit, match="synthetic"):
        train.train_and_evaluate(data, train.feature_columns(False), tiny_models())


def test_empty_database(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    database.init_db(db)
    with pytest.raises(SystemExit, match="No occupancy data"):
        train.load_dataset(db)
