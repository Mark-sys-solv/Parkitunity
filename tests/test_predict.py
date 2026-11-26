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
    db, model_path = folder / "synthetic.db", folder / "model.pkl"
    # End 10 min ago: measurements get up to 5 min random offset and must never lie
    # after "now" (otherwise they would beat the simulated live value).
    df = gen.generate(weeks=6, seed=2, end=datetime.now(timezone.utc) - timedelta(minutes=10))
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
    assert info["horizon_min"] == 60


def test_arrival_now_returns_latest_measurement(use_trained) -> None:
    db, _ = use_trained
    assert predict.predict_free(GARAGE, datetime.now())[0] == _latest(db)


@pytest.mark.parametrize("minutes", [15, 30, 60, 240, 24 * 60, 6 * 24 * 60])
def test_predictions_are_ints_within_capacity(use_trained, minutes: int) -> None:
    value = predict.predict_free(GARAGE, datetime.now() + timedelta(minutes=minutes))[0]
    assert isinstance(value, np.integer)
    assert 0 <= value <= CAPACITY


def _fixed_model(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    """Replace the model by one that always predicts ``value``."""
    bundle = dict(predict.load_model())

    class FixedModel:
        def predict(self, x):
            assert list(x.columns) == bundle["features"]  # same features in the same order as in training
            assert not x.isna().any().any()  # the missing capacity is filled in
            return np.array([value])

    bundle["model"] = FixedModel()
    monkeypatch.setattr(predict, "load_model", lambda path=None: bundle)


def _fake_measurement(monkeypatch: pytest.MonkeyPatch, free: float, minutes_old: int = 0) -> pd.Timestamp:
    """Only one measurement, no other history. Returns the (fixed) current time."""
    now = pd.Timestamp("2026-11-25 10:30", tz="UTC")
    history = pd.DataFrame({"timestamp": [now - pd.Timedelta(minutes=minutes_old)], "garage_id": [GARAGE],
                            "free_spots": [free], "capacity": [np.nan]})
    monkeypatch.setattr(predict, "_now", lambda: now)
    monkeypatch.setattr(predict, "_history", lambda gid: (history,))
    return now


def test_forecast_starts_at_the_measurement_and_reaches_the_model(use_trained, monkeypatch: pytest.MonkeyPatch) -> None:
    _fixed_model(monkeypatch, 40.0)
    now = _fake_measurement(monkeypatch, 100.0)
    minutes = [0, 14, 30, 60]  # 14: not rounded to a quarter of an hour
    values = predict.predict_free(GARAGE, [now + pd.Timedelta(minutes=m) for m in minutes])
    assert list(values) == [100, 86, 70, 40]


def test_no_forecast_for_old_measurements_and_unknown_garages(use_trained, monkeypatch: pytest.MonkeyPatch) -> None:
    now = _fake_measurement(monkeypatch, 100.0, minutes_old=120)
    old = predict._history(GARAGE)[0]
    assert predict._forecast_in_1h(old, GARAGE, now) is None
    fresh = old.assign(timestamp=now)
    assert predict._forecast_in_1h(fresh, GARAGE, now) is not None
    assert predict._forecast_in_1h(fresh, "unknown", now) is None  # the model never saw this garage


def test_no_jump_while_a_garage_is_filling_up(use_trained, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: 130 measured -> 194 after 14 minutes -> 130 was shown for Urania."""
    _fixed_model(monkeypatch, 60.0)
    now = _fake_measurement(monkeypatch, 130.0)
    # A typical day: the number of free spaces falls by 2 per minute.
    monkeypatch.setattr(predict, "_typical_value", lambda history, when: 300 - 2 * (when - now).total_seconds() / 60)
    curve = predict.forecast_curve(GARAGE, hours=2)
    assert curve["free_pred"].iloc[0] == 130
    assert curve["free_pred"].is_monotonic_decreasing  # never up again
    assert curve["free_pred"].diff().abs().max() <= 50


def test_forecast_curve_has_15_minute_steps_over_6_hours(use_trained) -> None:
    curve = predict.forecast_curve(GARAGE)
    assert list(curve.columns) == ["timestamp", "free_pred"]
    assert len(curve) == 6 * 4 + 1 and (curve["timestamp"].diff().dropna() == pd.Timedelta("15min")).all()
    assert str(curve["timestamp"].dt.tz) == "Europe/Zurich"
    assert list(curve["free_pred"]) == list(predict.predict_free(GARAGE, curve["timestamp"]))


def test_training_and_prediction_use_the_same_features_in_zurich_time() -> None:
    from src.ml.features import FEATURES, add_features, build_table

    history = pd.DataFrame({"timestamp": pd.to_datetime(["2026-11-20 10:30", "2026-07-03 10:30"], utc=True),
                            "garage_id": GARAGE, "free_spots": [50, 60], "capacity": 222})
    table = add_features(build_table(history).dropna(subset=["free"]), [GARAGE])
    winter = table[table["free"] == 50][FEATURES].iloc[0]
    summer = table[table["free"] == 60][FEATURES].iloc[0]
    assert (winter["hour"], winter["weekday"]) == (11, 4)  # 10:30 UTC = 11:30 in winter
    assert (summer["hour"], summer["weekday"]) == (12, 4)  # ... and 12:30 in summer


def test_far_future_uses_typical_week(use_trained, monkeypatch: pytest.MonkeyPatch) -> None:
    db, _ = use_trained
    target = pd.Timestamp.now(tz="UTC").floor("15min") + pd.Timedelta(days=3)
    since = pd.Timestamp.now(tz="UTC") - pd.Timedelta(weeks=predict.PROFILE_WEEKS)  # same period as predict()
    history = database.load_history(GARAGE, since=since.to_pydatetime(), db_path=db)
    typical = predict._typical_value(history, target)
    # Three days ahead the current deviation has faded completely.
    assert predict.predict_free(GARAGE, target.to_pydatetime())[0] == pytest.approx(typical, abs=1)


def test_works_without_model(use_trained, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "missing.pkl")
    assert predict.model_info() is None
    value = predict.predict_free(GARAGE, datetime.now() + timedelta(hours=2))[0]
    assert 0 <= value <= CAPACITY


def test_unknown_garage_returns_zero(use_trained) -> None:
    assert predict.predict_free("does-not-exist", datetime.now())[0] == 0


def test_dummy_garage_without_history_returns_live_value() -> None:
    # Network blocked -> dummy garages; "urania" has 120 free spots and no history.
    assert predict.predict_free("urania", datetime.now())[0] == 120


def test_several_timestamps_give_the_same_values_as_single_ones(use_trained) -> None:
    times = [datetime.now() + timedelta(minutes=m) for m in (30, 60, 180)]
    values = predict.predict_free(GARAGE, times)
    assert isinstance(values, np.ndarray) and len(values) == 3
    assert list(values) == [predict.predict_free(GARAGE, t)[0] for t in times]


@pytest.mark.parametrize("estimate, expected", [(-5.0, 0), (9999.0, CAPACITY), (12.6, 13)])
def test_result_is_limited_to_capacity_and_rounded(use_trained, monkeypatch: pytest.MonkeyPatch, estimate: float, expected: int) -> None:
    monkeypatch.setattr(predict, "_value_at", lambda *args: estimate)
    assert predict.predict_free(GARAGE, datetime.now())[0] == expected


def _live_garage(free_spots: int) -> Garage:
    return Garage(GARAGE, "Jelmoli", Location(47.37, 8.54), CAPACITY, free_spots, None, datetime.now(timezone.utc))


def test_synthetic_db_ignores_real_live_value(use_trained, monkeypatch: pytest.MonkeyPatch) -> None:
    db, _ = use_trained
    monkeypatch.setattr(parking_api, "get_garage", lambda gid: _live_garage(1))
    assert predict.predict_free(GARAGE, datetime.now())[0] == _latest(db)


def test_real_db_uses_live_value(use_trained, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db, _ = use_trained
    real_db = tmp_path / "real.db"
    database.insert_records(database.load_history(GARAGE, db_path=db).to_dict("records"), real_db)
    monkeypatch.setattr(predict, "DB_PATH", real_db)
    monkeypatch.setattr(parking_api, "get_garage", lambda gid: _live_garage(1))
    assert predict.predict_free(GARAGE, datetime.now())[0] == 1


def test_model_is_cached_and_reloaded_after_retraining(use_trained) -> None:
    _, model_path = use_trained
    first = predict.load_model()
    assert predict.load_model() is first
    stat = model_path.stat()
    os.utime(model_path, (stat.st_atime, stat.st_mtime + 10))  # simulate a newly trained file
    assert predict.load_model() is not first
