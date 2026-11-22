"""Compare driving + parking with public transport for one trip.

- Car: the best garage from ``rank_garages``; total time = drive + search + walk,
  cost = parking fee (+ variable km costs).
- Public transport: the best connection from ``transit_api``; total time =
  waiting for the first departure + travel time, cost = simplified ZVV fare.
- Decision: time is converted to money (``VALUE_OF_TIME_CHF_PER_H``) and added
  to the costs; the mode with the lower total wins, near-ties go to public
  transport. A short German reason is returned for the UI.

If the transit API does not answer, only the car is evaluated (with a note).
All assumptions live in ``src/config.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src import config
from src.data import routing, transit_api
from src.data.geocoding import parse_coordinates
from src.models import CarOption, Location, ModeComparison, TransitOption
from src.recommendation.parking_score import rank_garages

Place = str | Location | tuple[float, float]
_TZ = ZoneInfo(config.ZURICH_TZ)


def compare_modes(
    origin: Place,
    destination: Place,
    departure_time: datetime,
    max_price: float | None = None,
) -> ModeComparison:
    """Compare car (incl. parking) and public transport.

    Args:
        origin: Start – address/station name, ``"lat,lon"``, (lat, lon) or Location.
        destination: Destination, same formats.
        departure_time: When the trip starts (naive = Zurich time).
        max_price: Maximum hourly parking rate the driver accepts (CHF/h).

    Returns:
        ``{"car": CarOption, "transit": TransitOption,
        "recommendation": "car" | "transit" | None, "reason": str}`` –
        see ``src/models.py`` for the fields. ``recommendation`` is None only
        if neither a garage nor a connection is available.

    Raises:
        ValueError: if ``origin`` or ``destination`` is an address that cannot be found.
    """
    departure = _aware(departure_time)
    origin_loc, dest_loc = _to_location(origin), _to_location(destination)

    ranked = rank_garages(dest_loc, departure, max_price=max_price, origin=origin_loc)
    car = car_option(ranked[0] if ranked else None, origin_loc, departure)
    transit = transit_option(_transit_place(origin, origin_loc), _transit_place(destination, dest_loc),
                             departure, routing.haversine_km(origin_loc, dest_loc))
    recommendation, reason = decide(car, transit)
    return {"car": car, "transit": transit, "recommendation": recommendation, "reason": reason}


def car_option(best: dict | None, origin: Location, departure: datetime) -> CarOption:
    """Times and costs for driving to the ``best`` garage (an entry of ``rank_garages``)."""
    if best is None:
        return CarOption(
            available=False, total_min=None, drive_min=None, search_min=None, walk_min=None,
            cost_chf=None, parking_cost_chf=None, km_cost_chf=None, parking_garage_id=None,
            predicted_free=None, arrival=None,
        )
    garage = best["garage"]
    trips = 2 if config.ROUND_TRIP else 1
    rate = garage.hourly_rate_chf if garage.hourly_rate_chf is not None else config.ASSUMED_RATE_IF_UNKNOWN_CHF_PER_H
    parking = rate * config.PARKING_DURATION_H
    km = routing.get_drive_distance_km(origin, garage.location) * trips if config.INCLUDE_CAR_KM_COSTS else 0.0
    km_cost = km * config.CAR_COST_PER_KM_CHF
    return CarOption(
        available=True,
        total_min=round(best["total_min"], 1),
        drive_min=round(best["drive_min"], 1),
        search_min=round(best["search_min"], 1),
        walk_min=round(best["walking_min"], 1),
        cost_chf=round(parking + km_cost, 2),
        parking_cost_chf=round(parking, 2),
        km_cost_chf=round(km_cost, 2),
        parking_garage_id=garage.id,
        predicted_free=best["predicted_free"],
        arrival=departure + timedelta(minutes=best["total_min"]),
    )


def transit_option(origin: Place, destination: Place, departure: datetime, distance_km: float) -> TransitOption:
    """Best connection departing at or after ``departure``; unavailable if the API fails."""
    conn = transit_api.get_transit_options(origin, destination, departure)
    if conn is None:
        return TransitOption(
            available=False, total_min=None, wait_min=None, duration_min=None,
            cost_chf=None, transfers=None, departure=None, arrival=None,
        )
    wait = max(0.0, (conn["departure"] - departure).total_seconds() / 60)
    total = max(conn["duration_min"], (conn["arrival"] - departure).total_seconds() / 60)
    trips = 2 if config.ROUND_TRIP else 1
    return TransitOption(
        available=True,
        total_min=round(total, 1),
        wait_min=round(wait, 1),
        duration_min=round(conn["duration_min"], 1),
        cost_chf=round(transit_fare(distance_km) * trips, 2),
        transfers=conn["transfers"],
        departure=conn["departure"],
        arrival=conn["arrival"],
    )


def transit_fare(distance_km: float) -> float:
    """Simplified ZVV single ticket for a straight-line distance (see config)."""
    for max_km, price in config.TRANSIT_FARES_CHF:
        if distance_km <= max_km:
            return price
    return config.TRANSIT_FARES_CHF[-1][1]


def decide(car: CarOption, transit: TransitOption) -> tuple[str | None, str]:
    """Pick the mode with the lower generalised cost and explain why (German)."""
    if not car["available"] and not transit["available"]:
        return None, ("Keine Empfehlung möglich: kein Parkhaus mit freien Plätzen gefunden "
                      "und keine ÖV-Verbindung abrufbar.")
    if not transit["available"]:
        return "car", "Nur Auto-Empfehlung: Die ÖV-Verbindungen konnten gerade nicht abgefragt werden."
    if not car["available"]:
        return "transit", "Kein passendes Parkhaus mit freien Plätzen – der ÖV ist die einzige Option."

    car_total, transit_total = generalised_cost(car), generalised_cost(transit)
    if abs(car_total - transit_total) < config.DECISION_TIE_CHF:
        return "transit", "Beide etwa gleich gut – der ÖV erspart dir die Parkplatzsuche."

    winner = "transit" if transit_total < car_total else "car"
    loser_opt, winner_opt = (car, transit) if winner == "transit" else (transit, car)
    return winner, _reason(
        "Der ÖV" if winner == "transit" else "Das Auto",
        minutes_saved=loser_opt["total_min"] - winner_opt["total_min"],
        chf_saved=loser_opt["cost_chf"] - winner_opt["cost_chf"],
    )


def generalised_cost(option: CarOption | TransitOption) -> float:
    """Costs + time converted to CHF.

    Costs cover the round trip if ``ROUND_TRIP``, so the (one-way) time is
    counted twice as well – both directions are assumed to take about as long.
    """
    trips = 2 if config.ROUND_TRIP else 1
    return option["cost_chf"] + trips * option["total_min"] / 60 * config.VALUE_OF_TIME_CHF_PER_H


def _reason(subject: str, minutes_saved: float, chf_saved: float) -> str:
    """One sentence from the winner's point of view (minutes one way, CHF as compared)."""
    faster, cheaper = round(minutes_saved), round(chf_saved, 2)
    if faster >= 1:
        if cheaper >= 0.5:
            return f"{subject} ist {faster} min schneller und CHF {cheaper:.2f} günstiger."
        if cheaper > -0.5:
            return f"{subject} ist {faster} min schneller bei etwa gleichen Kosten."
        return (f"{subject} ist {faster} min schneller – die CHF {-cheaper:.2f} Mehrkosten lohnen sich "
                f"bei einem Zeitwert von {config.VALUE_OF_TIME_CHF_PER_H:.0f} CHF/h.")
    if cheaper >= 0.5:
        slower = f" und nur {-faster} min langsamer" if faster <= -1 else ""
        return f"{subject} ist CHF {cheaper:.2f} günstiger{slower}."
    return f"{subject} ist insgesamt die bessere Wahl."


def _to_location(place: Place) -> Location:
    if isinstance(place, Location):
        return place
    if isinstance(place, tuple):
        return Location(float(place[0]), float(place[1]))
    location = routing.geocode(place)
    if location is None:
        raise ValueError(f"Address not found: {place!r}")
    return location


def _transit_place(original: Place, resolved: Location) -> Place:
    """Station names go to the transit API as typed (it knows them better); else coordinates."""
    if isinstance(original, str) and parse_coordinates(original) is None:
        return original
    return resolved


def _aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=_TZ) if dt.tzinfo is None else dt
