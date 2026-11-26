"""Shared data types used as the contract between modules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypedDict


@dataclass(frozen=True)
class Location:
    """A geographic point, optionally with a human-readable name."""

    lat: float
    lon: float
    name: str = ""


@dataclass
class Garage:
    """A parking garage and its (last known) occupancy."""

    id: str
    name: str
    location: Location
    capacity: int
    free_spots: int | None = None
    hourly_rate_chf: float | None = None
    updated_at: datetime | None = None


class CarOption(TypedDict):
    """Car part of ``compare_modes``. Numbers are ``None`` if ``available`` is False."""

    available: bool  # False: there is no garage in the ranking
    total_min: int | None  # drive + walk, one way
    cost_chf: float | None  # km costs + parking fee, there and back
    garage: dict | None  # the recommended garage: row 1 of the ranking


class TransitOption(TypedDict):
    """Public transport part of ``compare_modes``. Numbers are ``None`` if not available."""

    available: bool  # False: API down or no connection found
    total_min: int | None  # waiting for the first departure + travel time, one way
    wait_min: int | None
    cost_chf: float | None  # tickets there and back
    transfers: int | None


class ModeComparison(TypedDict):
    """Return type of ``recommendation.mode_compare.compare_modes``."""

    car: CarOption
    transit: TransitOption
    recommendation: Literal["car", "transit"] | None  # None: tie or an option is missing
