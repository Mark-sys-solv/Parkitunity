"""Shared HTTP helpers: JSON GET with timeout/retries and a small TTL cache.

All external API calls go through ``get_json`` so that timeouts, retries and
error logging are handled in one place. It never raises on network/API errors;
callers get ``None`` and decide on a fallback.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import requests

from src.config import REQUEST_TIMEOUT_S, USER_AGENT

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _now() -> float:
    """Monotonic clock (separate function so tests can fake the time)."""
    return time.monotonic()


def _sleep(seconds: float) -> None:
    """Sleep between retries (separate function so tests can skip it)."""
    time.sleep(seconds)


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = REQUEST_TIMEOUT_S,
    retries: int = 1,
    backoff_s: float = 1.0,
) -> Any | None:
    """GET ``url`` and return the decoded JSON body, or ``None`` on failure.

    Retries on timeouts, connection errors, 5xx responses and invalid/empty
    JSON (ParkenDD occasionally returns an empty body). 4xx errors are not
    retried because repeating the same request will not help.

    Args:
        url: Endpoint URL.
        params: Query parameters.
        timeout: Timeout per attempt in seconds.
        retries: Number of additional attempts after the first one.
        backoff_s: Base wait time between attempts (grows linearly).
    """
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            logger.warning("GET %s failed with HTTP %s (attempt %d/%d)", url, status, attempt, attempts)
            if status is not None and status < 500:
                break
        except (requests.RequestException, ValueError) as exc:
            # ValueError covers invalid JSON (requests' JSONDecodeError subclasses it).
            logger.warning("GET %s failed (attempt %d/%d): %s", url, attempt, attempts, exc)
        if attempt < attempts:
            _sleep(backoff_s * attempt)
    logger.error("Giving up on GET %s", url)
    return None


def ttl_cache(seconds: float) -> Callable[[F], F]:
    """Cache a function's results in memory for ``seconds``.

    Falsy results (``None``, ``[]``, ``{}``) are not cached, so a failed API
    call is retried on the next invocation. Arguments must be hashable.
    Cached objects are shared between callers – do not mutate them.
    The wrapped function gets a ``cache_clear()`` method.
    """

    def decorator(func: F) -> F:
        """Wrap ``func`` with its own cache."""
        cache: dict[Any, tuple[float, Any]] = {}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Return the cached result, or call the function."""
            key = (args, tuple(sorted(kwargs.items())))
            now = _now()
            hit = cache.get(key)
            if hit is not None and now - hit[0] < seconds:
                return hit[1]
            result = func(*args, **kwargs)
            if result:
                # Drop expired entries so the cache does not grow forever.
                for k in [k for k, (t, _) in cache.items() if now - t >= seconds]:
                    del cache[k]
                cache[key] = (now, result)
            return result

        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
