"""Tests for the garage cards (invented data)."""

import pandas as pd
import pytest

from src import config
from src.data import parking_api
from src.ui.garage_cards import OTHER_TITLE, card_labels
from src.ui.styles import free_color_class


def _top(*rows: tuple) -> pd.DataFrame:
    """(drive_min, walk_min, price_per_hour, free_at_arrival) per garage."""
    return pd.DataFrame(rows, columns=["drive_min", "walk_min", "price_per_hour", "free_at_arrival"])


def test_every_category_winner_gets_its_label() -> None:
    top = _top((10, 5, 4.0, 100), (5, 5, 5.0, 50), (20, 5, 2.0, 20))
    assert card_labels(top, "balanced") == ["Beste Gesamtoption", "Schnellste Option", "Günstigste Option"]
    top["free_at_arrival"] = [10, 20, 300]
    top["price_per_hour"] = [1.0, 2.0, 3.0]
    assert card_labels(top, "cheapest") == ["Günstigste Option", "Schnellste Option", "Meiste freie Plätze"]


def test_a_label_is_used_only_once() -> None:
    # Rank 2 is the fastest, but rank 1 already has that label.
    top = _top((10, 5, 1.0, 300), (5, 5, 5.0, 50), (20, 5, 3.0, 20))
    assert card_labels(top, "fastest") == ["Schnellste Option", OTHER_TITLE, OTHER_TITLE]


def test_ties_and_unknown_prices_have_no_winner() -> None:
    assert card_labels(_top((5, 5, 4.0, 50), (5, 5, 4.0, 50)), "balanced")[1] == OTHER_TITLE
    assert card_labels(_top((5, 5, float("nan"), 90), (9, 9, float("nan"), 50)), "balanced")[1] == OTHER_TITLE
    assert card_labels(_top((5, 5, 4.0, 50)), "most_spots") == ["Meiste freie Plätze"]


@pytest.mark.parametrize("free, css_class", [(50, "pk-green"), (49, "pk-orange"), (10, "pk-orange"), (9, "pk-red"), (0, "pk-red")])
def test_free_color(free: int, css_class: str) -> None:
    assert free_color_class(free) == css_class


def test_is_dummy() -> None:
    assert parking_api.is_dummy("urania")
    assert not parking_api.is_dummy("zuerichparkhausurania")
    assert config.CARDS_SHOWN == 3
