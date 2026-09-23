"""Shared HTTP retry helper with exponential back-off.

Consolidates the duplicated retry/back-off logic previously scattered across
``get_nvd_data``, ``get_vulncheck_data``, ``get_osv_data``, and other tool
modules into a single, configurable function.

Usage::

    from manus_agent.utils.http_retry import http_get_with_retry

    # Simple GET with defaults (3 retries, 1 s base delay, retry on 429/5xx)
    response = http_get_with_retry("https://api.example.com/data")

    # Customised per-service
    response = http_get_with_retry(
        "https://services.nvd.nist.gov/rest/json/cves/2.0",
        headers={"apiKey": "..."},
        max_retries=3,
        base_delay=2.0,
        timeout=15,
    )

Every parameter is optional — sensible defaults match the behaviour of the
previous per-module helpers.  The ``sleep_fn`` parameter exists solely for
test injection; production callers should never set it.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Default set of HTTP status codes worth retrying.
DEFAULT_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def http_get_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 15,
    max_retries: int = 3,
    base_delay: float = 1.0,
    retryable_statuses: frozenset[int] | set[int] | None = None,
    raise_for_status: bool = True,
    sleep_fn: Any | None = None,
) -> requests.Response:
    """HTTP GET with exponential back-off retry on transient errors.

    Parameters
    ----------
    url:
        The URL to request.
    headers:
        Optional request headers.
    params:
        Optional query parameters.
    timeout:
        Per-request timeout in seconds (default 15).
    max_retries:
        Maximum number of retry attempts *after* the first request
        (default 3, so up to 4 total attempts).
    base_delay:
        Base delay in seconds for exponential back-off (default 1.0).
        Delay for attempt *n* (1-indexed) is ``base_delay * 2^(n-1)``.
    retryable_statuses:
        Set of HTTP status codes to retry on.  Defaults to
        ``{429, 500, 502, 503, 504}``.
    raise_for_status:
        Whether to call ``response.raise_for_status()`` on non-retryable
        responses.  Set to ``False`` if the caller wants to inspect 404s
        or other non-retryable codes without an exception.
    sleep_fn:
        Injectable sleep function (for testing).  Defaults to
        :func:`time.sleep`.

    Returns
    -------
    requests.Response
        The successful response.

    Raises
    ------
    requests.exceptions.HTTPError
        On non-retryable HTTP errors, or after all retries are exhausted
        for retryable status codes.
    requests.exceptions.RequestException
        On network-level errors after all retries are exhausted.
    """
    if retryable_statuses is None:
        retryable_statuses = DEFAULT_RETRYABLE_STATUSES
    if sleep_fn is None:
        sleep_fn = time.sleep

    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        # Sleep before retry (not before the first attempt)
        if attempt > 0:
            delay = base_delay * (2 ** (attempt - 1))
            logger.debug(
                "Retry %d/%d for %s (sleeping %.1fs)",
                attempt,
                max_retries,
                url,
                delay,
            )
            sleep_fn(delay)

        try:
            response = requests.get(
                url,
                headers=headers or {},
                params=params or {},
                timeout=timeout,
            )

            if response.status_code in retryable_statuses:
                last_exc = requests.exceptions.HTTPError(
                    f"HTTP {response.status_code}",
                    response=response,
                )
                if attempt < max_retries:
                    continue  # retry
                # All retries exhausted — raise
                if raise_for_status:
                    raise last_exc
                return response

            # Non-retryable status — raise or return
            if raise_for_status:
                response.raise_for_status()
            return response

        except requests.exceptions.HTTPError as exc:
            # Non-retryable client errors (4xx other than those in
            # retryable_statuses) fail immediately.
            resp = getattr(exc, "response", None)
            if resp is not None and resp.status_code not in retryable_statuses:
                raise
            last_exc = exc
            if attempt < max_retries:
                continue
            raise

        except requests.exceptions.RequestException as exc:
            # Network-level errors (ConnectionError, Timeout, etc.)
            last_exc = exc
            if attempt < max_retries:
                continue
            raise

    # Should never reach here, but appease type checkers.
    if last_exc is not None:
        raise last_exc
    raise requests.exceptions.RequestException(  # pragma: no cover
        f"Request to {url} failed after {max_retries + 1} attempts"
    )
