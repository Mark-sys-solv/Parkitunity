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

    available: bool  # False: no garage with free spots (within the price limit)
    total_min: float | None  # drive + search + walk
    drive_min: float | None
    search_min: float | None
    walk_min: float | None
    cost_chf: float | None  # parking fee + km costs (round trip if config.ROUND_TRIP)
    parking_cost_chf: float | None
    km_cost_chf: float | None
    parking_garage_id: str | None
    predicted_free: int | None
    arrival: datetime | None  # at the destination


class TransitOption(TypedDict):
    """Public transport part of ``compare_modes``. Numbers are ``None`` if not available."""

    available: bool  # False: API down or no connection found
    total_min: float | None  # waiting for the first departure + travel time
    wait_min: float | None
    duration_min: float | None  # travel time of the connection itself
    cost_chf: float | None  # simplified ZVV fare (round trip if config.ROUND_TRIP)
    transfers: int | None
    departure: datetime | None
    arrival: datetime | None


class ModeComparison(TypedDict):
    """Return type of ``recommendation.mode_compare.compare_modes``."""

    car: CarOption
    transit: TransitOption
    recommendation: Literal["car", "transit"] | None  # None: neither option available
    reason: str  # short German explanation for the UI
