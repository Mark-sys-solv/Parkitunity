from pathlib import Path

import pandas as pd
import pytest

from notebooks import generate_synthetic_data as gen
from src.data import database

END = pd.Timestamp("2026-11-20 12:00", tz="UTC")


def test_generate_is_deterministic_and_plausible() -> None:
    a = gen.generate(weeks=1, seed=7, end=END)
    b = gen.generate(weeks=1, seed=7, end=END)
    pd.testing.assert_frame_equal(a, b)

    assert set(a.columns) == {"timestamp", "garage_id", "free_spots", "capacity"}
    assert a["garage_id"].nunique() == len(gen.GARAGES)
    assert (a["free_spots"] >= 0).all() and (a["free_spots"] <= a["capacity"]).all()
    assert a["timestamp"].max() <= END + pd.Timedelta(minutes=5)
    # Some measurements are missing on purpose (like in the real feed).
    assert len(a) < len(gen.GARAGES) * (7 * 96 + 1)


def test_shopping_garage_is_fuller_on_saturday_afternoon_than_sunday() -> None:
    df = gen.generate(weeks=2, seed=3, end=END)
    jelmoli = df[df["garage_id"] == "zuerichparkhausjelmoli"].copy()
    local = jelmoli["timestamp"].dt.tz_convert("Europe/Zurich")
    afternoon = local.dt.hour.between(13, 16)
    sat = jelmoli.loc[afternoon & (local.dt.day_of_week == 5), "free_spots"].mean()
    sun = jelmoli.loc[afternoon & (local.dt.day_of_week == 6), "free_spots"].mean()
    assert sat < sun  # fewer free spots on Saturday


def test_write_db_marks_data_as_synthetic(tmp_path: Path) -> None:
    db = tmp_path / "synthetic.db"
    df = gen.generate(weeks=1, seed=1, end=END)
    assert gen.write_db(df, db, seed=1) == len(df)
    assert database.is_synthetic(db)
    # Regenerating replaces the old synthetic file.
    assert gen.write_db(df.head(10), db, seed=1) == 10


def test_write_db_refuses_to_overwrite_real_data(tmp_path: Path) -> None:
    real = tmp_path / "parkitunity.db"
    database.insert_records([{"timestamp": END, "garage_id": "x", "free_spots": 1, "capacity": 2}], real)
    with pytest.raises(SystemExit):
        gen.write_db(gen.generate(weeks=1, seed=1, end=END), real, seed=1)
    assert len(database.load_history(db_path=real)) == 1
