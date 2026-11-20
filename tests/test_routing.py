import pytest

from src.data import routing
from src.models import Location

HB = (47.3779, 8.5403)
PARADEPLATZ = (47.3697, 8.5389)


@pytest.fixture(autouse=True)
def _reset_provider():
    yield
    routing.set_provider(None)


def test_haversine_distance_is_plausible() -> None:
    # Zürich HB -> Paradeplatz is roughly 0.9 km in a straight line.
    assert routing.haversine_km(HB, PARADEPLATZ) == pytest.approx(0.92, abs=0.05)
    assert routing.haversine_km(HB, HB) == 0


def test_drive_time_formula() -> None:
    expected = routing.haversine_km(HB, PARADEPLATZ) * 1.4 / 25 * 60
    assert routing.get_drive_time(HB, PARADEPLATZ) == pytest.approx(expected)
    assert routing.get_drive_time(HB, HB) == 0


def test_accepts_location_and_tuple() -> None:
    assert routing.get_drive_time(Location(*HB), Location(*PARADEPLATZ)) == pytest.approx(
        routing.get_drive_time(HB, PARADEPLATZ)
    )


def test_custom_provider_is_used() -> None:
    class FixedProvider:
        def drive_distance_km(self, origin, destination):
            return 5.0

        def drive_time_min(self, origin, destination):
            return 12.0

    routing.set_provider(FixedProvider())
    assert routing.get_drive_time(HB, PARADEPLATZ) == 12.0
    assert routing.get_drive_distance_km(HB, PARADEPLATZ) == 5.0


def test_failing_provider_falls_back_to_haversine() -> None:
    class BrokenProvider:
        def drive_distance_km(self, origin, destination):
            raise RuntimeError("API down")

        def drive_time_min(self, origin, destination):
            raise RuntimeError("API down")

    expected = routing.HaversineProvider().drive_time_min(HB, PARADEPLATZ)
    routing.set_provider(BrokenProvider())
    assert routing.get_drive_time(HB, PARADEPLATZ) == pytest.approx(expected)
