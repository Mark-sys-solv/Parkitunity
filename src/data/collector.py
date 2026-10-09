"""Periodically log garage occupancy into the database.

Usage:
    python -m src.data.collector            # log once
    python -m src.data.collector --loop     # log every COLLECTOR_INTERVAL_MIN minutes

For long-term collection, run it via cron, e.g.:
    */10 * * * * cd /path/to/Parkitunity && .venv/bin/python -m src.data.collector
"""

from __future__ import annotations

import argparse

from src.config import COLLECTOR_INTERVAL_MIN


def run_once() -> int:
    """Fetch all garages and store one measurement per garage.

    Returns:
        Number of rows written.
    """
    # TODO: database.init_db(); garages = parking_api.fetch_garages()
    # TODO: database.insert_occupancy(...) for each garage with free_spots set.
    # TODO: log errors instead of crashing (the collector must keep running).
    raise NotImplementedError


def run_loop(interval_min: int = COLLECTOR_INTERVAL_MIN) -> None:
    """Call ``run_once`` forever, every ``interval_min`` minutes."""
    # TODO: implement with time.sleep; handle KeyboardInterrupt.
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Log Zurich parking garage occupancy.")
    parser.add_argument("--loop", action="store_true", help="run continuously")
    args = parser.parse_args()
    if args.loop:
        run_loop()
    else:
        print(f"Wrote {run_once()} rows.")


if __name__ == "__main__":
    main()
