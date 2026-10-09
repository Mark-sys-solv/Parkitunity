"""Shared data types used as the contract between modules."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class TransitConnection:
    """One public-transport connection from transport.opendata.ch."""

    departure: datetime
    arrival: datetime
    duration_min: float
    transfers: int
    products: list[str] = field(default_factory=list)  # e.g. ["S12", "T4"]


class CarOption(TypedDict):
    duration_min: float
    cost_chf: float
    parking_garage_id: str | None


class TransitOption(TypedDict):
    duration_min: float
    cost_chf: float
    transfers: int


class ModeComparison(TypedDict):
    """Return type of ``recommendation.mode_compare.compare_modes``."""

    car: CarOption
    transit: TransitOption
    recommendation: Literal["car", "transit"]
