# AI citation: this file was written with Claude (Anthropic).
"""Rank parking garages by what matters to the driver (pure functions, no Streamlit).

Every criterion is scaled to 0 (worst garage) ... 1 (best garage) and the
score is the weighted sum. The weights per priority are in ``config.PRIORITY_WEIGHTS``.
"""

from __future__ import annotations

import pandas as pd

from src import config


def normalise(values: pd.Series, more_is_better: bool, reference: pd.Series) -> pd.Series:
    """Min-max normalisation: best value = 1, worst value = 0.

    Args:
        values: The numbers to scale.
        more_is_better: False if a small number is good (e.g. the price).
        reference: The scale goes from the smallest to the largest of these
            values. Values outside of it get 0 or 1.

    If all reference values are equal, the criterion cannot tell the garages
    apart: everyone gets 1 (and we avoid dividing by 0).
    """
    spread = reference.max() - reference.min()
    if len(reference) == 0 or spread == 0:
        return pd.Series(1.0, index=values.index)
    scaled = ((values - reference.min()) / spread).clip(0, 1)
    return scaled if more_is_better else 1 - scaled


def rank_garages(
    candidates: pd.DataFrame,
    priority: str = config.DEFAULT_PRIORITY,
    max_walk_min: int = config.DEFAULT_MAX_WALK_MIN,
    enough_spots: int = config.DEFAULT_ENOUGH_SPOTS,
) -> pd.DataFrame:
    """Sort the garages, best first.

    Args:
        candidates: Output of ``build_candidates`` (not changed).
        priority: Key of ``config.PRIORITY_WEIGHTS``.
        max_walk_min: Garages with a longer walk to the destination come after the others.
        enough_spots: More free spots than this count the same, so that one huge
            garage does not make all others look bad.

    Returns:
        Copy of the table with the extra columns ``score`` (0 ... 1) and
        ``rank`` (1 = best). Whatever their score, garages that are too far to
        walk come near the end, and those with fewer than ``MIN_FREE_AT_ARRIVAL``
        expected free spots come last.
    """
    weights = config.PRIORITY_WEIGHTS[priority]
    ranked = candidates.copy()
    travel_min = ranked["drive_min"] + ranked["walk_min"]
    price = ranked["price_per_hour"].fillna(config.ASSUMED_RATE_IF_UNKNOWN_CHF_PER_H)
    spots = ranked["free_at_arrival"].clip(upper=enough_spots)

    # Garages that do not really come into question; they are sorted to the end.
    too_few = ranked["free_at_arrival"] < config.MIN_FREE_AT_ARRIVAL
    too_far = ranked["walk_min"] > max_walk_min
    # They must not stretch the scale either (e.g. a garage 87 minutes away).
    suitable = ~(too_few | too_far)
    if not suitable.any():
        suitable[:] = True  # nothing suitable: compare all garages with each other

    ranked["score"] = (
        weights["travel"] * normalise(travel_min, False, travel_min[suitable])
        + weights["price"] * normalise(price, False, price[suitable])
        + weights["spots"] * normalise(spots, True, spots[suitable])
    )
    # Helper columns, only for sorting: too few spots last, then too far, then score, then the faster garage.
    ranked["_too_few"], ranked["_too_far"] = too_few, too_far
    ranked["_travel"] = travel_min
    ranked = ranked.sort_values(["_too_few", "_too_far", "score", "_travel"], ascending=[True, True, False, True])
    ranked = ranked.drop(columns=["_too_few", "_too_far", "_travel"]).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def best_garage(ranking: pd.DataFrame) -> dict | None:
    """Rank 1 as a dict (``None`` if the ranking is empty)."""
    return None if ranking.empty else ranking.iloc[0].to_dict()
