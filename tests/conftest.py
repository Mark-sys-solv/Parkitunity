"""Shared test setup.

By default tests run offline: real HTTP requests fail with a ConnectionError
(which our code handles like an unreachable API). Tests marked ``@pytest.mark.live``
hit the real APIs and only run with ``RUN_LIVE_TESTS=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import requests
import streamlit as st

from src.data import geocoding, http_utils, parking_api, transit_api, weather_api
from src.ml import predict

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "live: test calls real external APIs (set RUN_LIVE_TESTS=1)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.getenv("RUN_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="live API test, set RUN_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolate(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clear caches, skip retry sleeps, use temp model/DB paths and block the network."""
    parking_api.fetch_garages.cache_clear()
    transit_api._fetch_best_connection.cache_clear()
    weather_api._fetch.cache_clear()
    predict._history.cache_clear()
    predict._load_bundle.clear()
    predict._is_synthetic.cache_clear()
    st.cache_data.clear()  # e.g. geocoding results
    monkeypatch.setattr(http_utils, "_sleep", lambda s: None)
    monkeypatch.setattr(geocoding, "_sleep", lambda s: None)  # no 1-second waits in tests
    monkeypatch.setattr(parking_api.config, "USE_DUMMY_DATA", False)
    # Never read the real database or model in tests.
    monkeypatch.setattr(predict, "DB_PATH", tmp_path / "no-real-data.db")
    monkeypatch.setattr(predict, "MODEL_PATH", tmp_path / "no-model.pkl")
    if request.node.get_closest_marker("live") is None:

        def _blocked(*args: Any, **kwargs: Any) -> None:
            raise requests.ConnectionError("Network access is disabled in tests")

        monkeypatch.setattr(requests.Session, "request", _blocked)


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
