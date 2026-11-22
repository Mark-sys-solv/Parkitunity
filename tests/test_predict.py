import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from notebooks import generate_synthetic_data as gen
from src.data import database, parking_api
from src.ml import predict, train
from src.models import Garage, Location
from tests.test_train import GARAGES, tiny_models

GARAGE = "zuerichparkhausjelmoli"
CAPACITY = 222


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Synthetic DB with data up to now + a small model trained on it."""
    folder = tmp_path_factory.mktemp("predict")
    db, model_path = folder / "synthetic.db", folder / "model.pkl.gz"
    # End 10 min ago: measurements get up to 5 min random offset and must never lie
    # after "now" (otherwise they would beat the simulated live value).
    df = gen.generate(weeks=3, seed=2, end=datetime.now(timezone.utc) - timedelta(minutes=10))
    gen.write_db(df[df["garage_id"].isin(GARAGES)], db, seed=2)

    mp = pytest.MonkeyPatch()
    mp.setattr(train, "make_models", tiny_models)
    mp.setattr(sys, "argv", ["train", "--db", str(db), "--model-path", str(model_path), "--docs-dir", str(folder / "docs")])
    train.main()
    mp.undo()
    return db, model_path


@pytest.fixture
def use_trained(trained: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    db, model_path = trained
    monkeypatch.setattr(predict, "DB_PATH", db)
    monkeypatch.setattr(predict, "MODEL_PATH", model_path)
    return trained


def _latest(db: Path, garage_id: str = GARAGE) -> float:
    history = database.load_history(garage_id, db_path=db)
    cutoff = pd.Timestamp.now(tz="UTC").floor("15min") + pd.Timedelta("15min")
    return float(history.loc[history["timestamp"] <= cutoff, "free_spots"].iloc[-1])


def test_model_info(use_trained) -> None:
    info = predict.model_info()
    assert info["data_source"] == "synthetic"
    assert info["horizon_min"] == 30


def test_arrival_now_returns_latest_measurement(use_trained) -> None:
    db, _ = use_trained
    assert predict.predict(GARAGE, datetime.now()) == _latest(db)


@pytest.mark.parametrize("minutes", [15, 30, 60, 240, 24 * 60, 6 * 24 * 60])
def test_predictions_are_ints_within_capacity(use_trained, minutes: int) -> None:
    value = predict.predict(GARAGE, datetime.now() + timedelta(minutes=minutes))
    assert isinstance(value, int)
    assert 0 <= value <= CAPACITY


def test_model_is_used_within_its_horizon(use_trained, monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = dict(predict.load_model())

    class FixedModel:
        def predict(self, x):
            assert list(x.columns) == bundle["feature_columns"]
            assert not x.isna().any().any()
            return np.array([42.0])

    bundle["model"] = FixedModel()
    monkeypatch.setattr(predict, "load_model", lambda path=None: bundle)
    assert predict.predict(GARAGE, datetime.now() + timedelta(minutes=30)) == 42


def test_far_future_uses_typical_week(use_trained, monkeypatch: pytest.MonkeyPatch) -> None:
    db, _ = use_trained
    target = pd.Timestamp.now(tz="UTC").floor("15min") + pd.Timedelta(days=3)
    history = database.load_history(GARAGE, db_path=db)
    typical = predict._typical_value(history, target)
    # Three days ahead the current deviation has faded completely.
    assert predict.predict(GARAGE, target.to_pydatetime()) == pytest.approx(typical, abs=1)


def test_works_without_model(use_trained, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "missing.pkl.gz")
    assert predict.model_info() is None
    value = predict.predict(GARAGE, datetime.now() + timedelta(hours=2))
    assert 0 <= value <= CAPACITY


def test_unknown_garage_returns_zero(use_trained) -> None:
    assert predict.predict("does-not-exist", datetime.now()) == 0


def test_dummy_garage_without_history_returns_live_value() -> None:
    # Network blocked -> dummy garages; "urania" has 120 free spots and no history.
    assert predict.predict("urania", datetime.now()) == 120


def _live_garage(free_spots: int) -> Garage:
    return Garage(GARAGE, "Jelmoli", Location(47.37, 8.54), CAPACITY, free_spots, None, datetime.now(timezone.utc))


def test_synthetic_db_ignores_real_live_value(use_trained, monkeypatch: pytest.MonkeyPatch) -> None:
    db, _ = use_trained
    monkeypatch.setattr(parking_api, "get_garage", lambda gid: _live_garage(1))
    assert predict.predict(GARAGE, datetime.now()) == _latest(db)


def test_real_db_uses_live_value(use_trained, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db, _ = use_trained
    real_db = tmp_path / "real.db"
    database.insert_records(database.load_history(GARAGE, db_path=db).to_dict("records"), real_db)
    monkeypatch.setattr(predict, "DB_PATH", real_db)
    monkeypatch.setattr(parking_api, "get_garage", lambda gid: _live_garage(1))
    assert predict.predict(GARAGE, datetime.now()) == 1


def test_model_is_cached_and_reloaded_after_retraining(use_trained) -> None:
    _, model_path = use_trained
    first = predict.load_model()
    assert predict.load_model() is first
    stat = model_path.stat()
    os.utime(model_path, (stat.st_atime, stat.st_mtime + 10))  # simulate a newly trained file
    assert predict.load_model() is not first
