"""
############################################################################
#  SYNTHETIC DATA – FOR DEVELOPMENT ONLY                                   #
#                                                                          #
#  This script INVENTS parking occupancy data so the ML pipeline can be    #
#  developed and tested without downloading the real history.            #
#  Never report results from this data as real findings, and never mix     #
#  it with the real database.                                              #
############################################################################

The data imitates real patterns (daily/weekly rhythm, shopping Saturdays,
office hours, events, holidays, sensor gaps and noise), using the real Zurich
garage ids and capacities so the app can use a model trained on it.

Output: a separate SQLite file (default ``data/synthetic.db``) marked with
``dataset_info.synthetic = "true"``. The training script stores this flag in
the model, and the app shows a warning while such a model is in use.

Usage:
    python notebooks/generate_synthetic_data.py                 # 8 weeks up to now
    python notebooks/generate_synthetic_data.py --weeks 12 --seed 1
    python -m src.ml.train --db data/synthetic.db
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python notebooks/generate_synthetic_data.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SYNTHETIC_DB_PATH, ZURICH_TZ  # noqa: E402
from src.data import database  # noqa: E402
from src.ml.features import zurich_holidays  # noqa: E402

# Real ParkenDD garage ids and capacities (Nov 2026) with an assumed usage profile.
GARAGES: list[tuple[str, str, int, str]] = [
    ("zuerichparkhausaccu", "Accu", 194, "shopping"),
    ("zuerichparkhausalbisriederplatz", "Albisriederplatz", 66, "business"),
    ("zuerichparkhausbleicherweg", "Bleicherweg", 275, "business"),
    ("zuerichparkhauscentereleven", "Center Eleven", 342, "shopping"),
    ("zuerichparkhauscityparking", "City Parking", 620, "shopping"),
    ("zuerichparkhauscityport", "Cityport", 153, "business"),
    ("zuerichparkhauscrowneplaza", "Crowne Plaza", 520, "business"),
    ("zuerichparkhausdorflinde", "Dorflinde", 98, "business"),
    ("zuerichparkhausfeldegg", "Feldegg", 346, "leisure"),
    ("zuerichparkhausglobus", "Globus", 178, "shopping"),
    ("zuerichparkhaushardauii", "Hardau II", 982, "business"),
    ("zuerichparkhaushauptbahnhof", "Hauptbahnhof", 176, "shopping"),
    ("zuerichparkhaushohepromenade", "Hohe Promenade", 556, "shopping"),
    ("zuerichparkhausjelmoli", "Jelmoli", 222, "shopping"),
    ("zuerichparkhausjungholz", "Jungholz", 124, "business"),
    ("zuerichparkhausmaxbillplatz", "Max-Bill-Platz", 59, "business"),
    ("zuerichparkhausmessezuerichag", "Messe Zürich AG", 2000, "event"),
    ("zuerichparkhausnordhaus", "Nordhaus", 175, "business"),
    ("zuerichparkhausoctavo", "Octavo", 123, "business"),
    ("zuerichparkhausopéra", "Opéra", 299, "leisure"),
    ("zuerichparkhauspwest", "P West", 1000, "event"),
    ("zuerichparkhausparkhyatt", "Park Hyatt", 267, "business"),
    ("zuerichparkhausparkside", "Parkside", 38, "leisure"),
    ("zuerichparkhauspfingstweid", "Pfingstweid", 276, "business"),
    ("zuerichparkhausstampfenbach", "Stampfenbach", 237, "business"),
    ("zuerichparkhaustalgarten", "Talgarten", 110, "shopping"),
    ("zuerichparkhaususznord", "USZ Nord", 90, "business"),
    ("zuerichparkhausuniirchel", "Uni Irchel", 1227, "business"),
    ("zuerichparkhausurania", "Urania", 607, "shopping"),
    ("zuerichparkhausutoquai", "Utoquai", 175, "leisure"),
    ("zuerichparkhauszueri11shopping", "Züri 11 Shopping", 60, "shopping"),
    ("zuerichparkhauszuerichhorn", "Zürichhorn", 245, "leisure"),
    ("zuerichparkplatztheater11", "Theater 11", 188, "leisure"),
    ("zuerichparkplatzuszsued", "USZ Süd", 80, "business"),
]


def _bump(hour: np.ndarray, center: float, width: float) -> np.ndarray:
    """Smooth bell-shaped bump (peak 1 at ``center``)."""
    return np.exp(-0.5 * ((hour - center) / width) ** 2)


def _plateau(hour: np.ndarray, start: float, end: float, softness: float = 0.7) -> np.ndarray:
    """Smooth step up at ``start`` and down at ``end``."""
    return 1 / (1 + np.exp(-(hour - start) / softness)) - 1 / (1 + np.exp(-(hour - end) / softness))


def slow_noise(rng: np.random.Generator, n: int, keep: float = 0.92, step: float = 0.012) -> np.ndarray:
    """Random noise that drifts slowly instead of jumping (an "AR(1) process").

    Each value keeps 92 % of the previous one and adds a small random step,
    so neighbouring 15-minute slots stay similar.
    """
    steps = rng.normal(0, step, n)
    noise = np.zeros(n)
    noise[0] = steps[0]
    for t in range(1, n):
        noise[t] = keep * noise[t - 1] + steps[t]
    return noise


def occupancy_curve(profile: str, hour: np.ndarray, weekday: np.ndarray, closed_day: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Expected occupancy rate (0..1) for a usage profile, without noise.

    ``closed_day`` is True on Sundays and public holidays (shops closed).
    """
    sat = weekday == 5
    workday = ~closed_day & ~sat
    if profile == "shopping":
        rate = 0.12 + workday * (0.55 * _bump(hour, 14.5, 3.0))
        rate += (workday & (weekday == 3)) * 0.2 * _bump(hour, 18.5, 1.2)  # Thursday late shopping
        rate += sat * 0.78 * _bump(hour, 14.0, 2.6)
        rate += sat * (month == 12) * 0.12 * _bump(hour, 14.0, 3.0)  # Advent Saturdays
        rate += closed_day * 0.10 * _bump(hour, 15, 3)
    elif profile == "business":
        rate = 0.08 + workday * 0.75 * _plateau(hour, 7.5, 17.5)
        rate += (sat | closed_day) * 0.12 * _bump(hour, 13, 4)
    elif profile == "leisure":
        evening = ((weekday == 4) | sat) * 0.6 + (~closed_day & (weekday < 4)) * 0.35
        rate = 0.1 + evening * _bump(hour, 20.5, 1.8)
        rate += (sat | closed_day) * 0.35 * _bump(hour, 15, 2.5)
    elif profile == "event":
        rate = 0.05 + workday * 0.1 * _plateau(hour, 8, 18)
    else:
        raise ValueError(profile)
    return rate


def generate(weeks: int = 8, seed: int = 42, end: datetime | None = None) -> pd.DataFrame:
    """Create synthetic occupancy measurements for all garages.

    Returns:
        DataFrame with ``timestamp`` (UTC), ``garage_id``, ``free_spots``, ``capacity``.
    """
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end or datetime.now(timezone.utc)).floor("15min")
    slots = pd.date_range(end - pd.Timedelta(weeks=weeks), end, freq="15min")
    local = slots.tz_convert(ZURICH_TZ)
    hour = (local.hour + local.minute / 60).to_numpy()
    weekday = local.day_of_week.to_numpy()
    month = local.month.to_numpy()
    holiday_dates = zurich_holidays(tuple(sorted(set(local.year))))
    closed_day = (weekday == 6) | np.isin(local.date, list(holiday_dates))
    days = pd.Index(local.date)
    unique_days = days.unique()

    frames = []
    for garage_id, _name, capacity, profile in GARAGES:
        rate = occupancy_curve(profile, hour, weekday, closed_day, month)
        rate = rate * rng.uniform(0.75, 1.1) + rng.uniform(-0.03, 0.05)  # garage-specific level

        # Events (concerts, fairs, games): unpredictable from the calendar alone.
        event_prob = {"event": 0.18, "leisure": 0.05}.get(profile, 0.0)
        for day in unique_days[rng.random(len(unique_days)) < event_prob]:
            start_h = rng.choice([9.0, 18.5]) if profile == "event" else 19.0
            strength = rng.uniform(0.5, 0.9)
            mask = days == day
            rate[mask] += strength * _plateau(hour[mask], start_h, start_h + rng.uniform(3, 6), 0.4)

        # Day-to-day variation + slowly drifting noise (AR(1)) + measurement noise.
        day_effect = pd.Series(rng.normal(0, 0.04, len(unique_days)), index=unique_days)
        rate += day_effect.reindex(days).to_numpy()
        rate += slow_noise(rng, len(slots))
        rate += rng.normal(0, 0.008, len(slots))
        rate = np.clip(rate, 0.0, 0.995)

        free = np.round(capacity * (1 - rate)).astype(int)
        # Like real measurements: a few minutes after each slot.
        timestamps = slots + pd.to_timedelta(rng.integers(0, 300, len(slots)), unit="s")
        frames.append(pd.DataFrame({"timestamp": timestamps, "garage_id": garage_id, "free_spots": free, "capacity": capacity}))

    df = pd.concat(frames, ignore_index=True)

    # Missing measurements like in the real feed: random drops + a few outages.
    keep = rng.random(len(df)) > 0.03
    for _ in range(max(1, weeks // 3)):
        outage_start = slots[rng.integers(0, len(slots) - 24)]
        outage_end = outage_start + pd.Timedelta(hours=float(rng.uniform(1, 6)))
        keep &= ~df["timestamp"].between(outage_start, outage_end)
    return df[keep].sort_values(["garage_id", "timestamp"]).reset_index(drop=True)


def write_db(df: pd.DataFrame, db_path: Path, seed: int) -> int:
    """Write synthetic data into its own DB file, refusing to touch real data."""
    db_path = Path(db_path)
    if db_path.exists():
        if not database.is_synthetic(db_path):
            raise SystemExit(f"Refusing to overwrite {db_path}: it is not marked as synthetic (real data?).")
        db_path.unlink()
    database.init_db(db_path)
    database.set_dataset_info(
        {
            "synthetic": "true",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generator": "notebooks/generate_synthetic_data.py",
            "seed": str(seed),
        },
        db_path,
    )
    return database.insert_records(df.to_dict("records"), db_path)


def main() -> None:
    """Generate the synthetic database."""
    parser = argparse.ArgumentParser(description="Generate SYNTHETIC parking data (development only).")
    parser.add_argument("--weeks", type=int, default=8, help="weeks of data up to now (default 8)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--db", type=Path, default=SYNTHETIC_DB_PATH, help=f"output DB (default {SYNTHETIC_DB_PATH})")
    args = parser.parse_args()

    df = generate(args.weeks, args.seed)
    rows = write_db(df, args.db, args.seed)
    print("⚠️  SYNTHETIC DATA – for development only, not real measurements!")
    print(f"Wrote {rows:,} rows for {df['garage_id'].nunique()} garages "
          f"({df['timestamp'].min():%Y-%m-%d} – {df['timestamp'].max():%Y-%m-%d}) to {args.db}")


if __name__ == "__main__":
    main()
