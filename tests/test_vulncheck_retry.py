"""Tests for get_vulncheck_data retry/back-off logic (_vc_get_with_retry).

All HTTP calls are fully mocked — no real network requests are made.
The VULNCHECK_RETRY_BASE_DELAY env var is set to "0" in every test so
retry loops complete instantly without actual sleeping.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use(cve_id: str = "CVE-2024-3094", tool_use_id: str = "retry-test-001") -> dict:
    return {"toolUseId": tool_use_id, "input": {"cve_id": cve_id}}


def _make_response(status_code: int = 200, payload: dict | None = None) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload or {}
    if status_code < 400:
        resp.raise_for_status.return_value = None
    else:
        http_err = requests.exceptions.HTTPError(
            f"HTTP {status_code}",
            response=resp,
        )
        resp.raise_for_status.side_effect = http_err
    return resp


def _make_kev_hit_response() -> MagicMock:
    return _make_response(
        200,
        {
            "data": [
                {
                    "cveID": "CVE-2024-3094",
                    "dateAdded": "2024-03-29",
                    "sources": ["CISA KEV"],
                    "ransomwareUse": False,
                }
            ]
        },
    )


def _make_nvd2_hit_response() -> MagicMock:
    return _make_response(
        200,
        {
            "data": [
                {
                    "id": "CVE-2024-3094",
                    "published": "2024-03-29",
                    "descriptions": [{"lang": "en", "value": "Test CVE."}],
                    "metrics": {},
                    "configurations": [],
                }
            ]
        },
    )


def _make_429_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 429
    resp.json.return_value = {"error": "rate limited"}

    http_err = requests.exceptions.HTTPError("429 Too Many Requests", response=resp)
    resp.raise_for_status.side_effect = http_err
    return resp


def _make_5xx_response(status: int = 503) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"error": "server error"}
    http_err = requests.exceptions.HTTPError(f"{status} Server Error", response=resp)
    resp.raise_for_status.side_effect = http_err
    return resp


# ===========================================================================
# Module-level constants
# ===========================================================================


def test_retryable_statuses_includes_429():
    from manus_agent.tools.get_vulncheck_data import _VC_RETRYABLE_STATUSES

    assert 429 in _VC_RETRYABLE_STATUSES


def test_retryable_statuses_includes_5xx():
    from manus_agent.tools.get_vulncheck_data import _VC_RETRYABLE_STATUSES

    for code in (500, 502, 503, 504):
        assert code in _VC_RETRYABLE_STATUSES


def test_retryable_statuses_excludes_401():
    from manus_agent.tools.get_vulncheck_data import _VC_RETRYABLE_STATUSES

    assert 401 not in _VC_RETRYABLE_STATUSES


def test_retryable_statuses_excludes_404():
    from manus_agent.tools.get_vulncheck_data import _VC_RETRYABLE_STATUSES

    assert 404 not in _VC_RETRYABLE_STATUSES


def test_max_retries_default_is_3():
    """Default retry count should be 3 (env var unset)."""
    import sys

    # Re-import with env var cleared to check default.
    mod_name = "manus_agent.tools.get_vulncheck_data"
    saved = sys.modules.pop(mod_name, None)
    try:
        import os

        old_val = os.environ.pop("VULNCHECK_MAX_RETRIES", None)
        import manus_agent.tools.get_vulncheck_data as vc_mod

        assert vc_mod._VC_MAX_RETRIES == 3
    finally:
        if old_val is not None:
            os.environ["VULNCHECK_MAX_RETRIES"] = old_val
        sys.modules.pop(mod_name, None)
        if saved is not None:
            sys.modules[mod_name] = saved


def test_retry_base_delay_default_is_1():
    """Default base delay should be 1.0 s."""
    import os
    import sys

    mod_name = "manus_agent.tools.get_vulncheck_data"
    saved = sys.modules.pop(mod_name, None)
    try:
        old_val = os.environ.pop("VULNCHECK_RETRY_BASE_DELAY", None)
        import manus_agent.tools.get_vulncheck_data as vc_mod

        assert vc_mod._VC_RETRY_BASE_DELAY == 1.0
    finally:
        if old_val is not None:
            os.environ["VULNCHECK_RETRY_BASE_DELAY"] = old_val
        sys.modules.pop(mod_name, None)
        if saved is not None:
            sys.modules[mod_name] = saved


# ===========================================================================
# _vc_get_with_retry — success on first attempt
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_success_first_attempt_no_sleep(mock_sleep, mock_get, monkeypatch):
    """A 200 response on the first attempt must not sleep."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    mock_get.return_value = _make_response(200, {"data": []})

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    _vc_get_with_retry(
        "https://example.com",
        headers={"Authorization": "Bearer k"},
        params={"cve": "CVE-2024-1234"},
        timeout=5,
    )

    mock_sleep.assert_not_called()
    mock_get.assert_called_once()


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
def test_success_first_attempt_returns_response(mock_get, monkeypatch):
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    resp = _make_response(200, {"data": [{"id": "x"}]})
    mock_get.return_value = resp

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    result = _vc_get_with_retry(
        "https://example.com",
        headers={},
        params={},
        timeout=5,
    )
    assert result is resp


# ===========================================================================
# _vc_get_with_retry — retry on 429
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_retries_on_429_then_succeeds(mock_sleep, mock_get, monkeypatch):
    """One 429 followed by a 200 should result in 2 total attempts."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    mock_get.side_effect = [
        _make_429_response(),
        _make_response(200, {"data": []}),
    ]

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    resp = _vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)

    assert resp.status_code == 200
    assert mock_get.call_count == 2
    mock_sleep.assert_called_once()


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_retries_on_429_twice_then_succeeds(mock_sleep, mock_get, monkeypatch):
    """Two 429s followed by a 200 should result in 3 total attempts."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    mock_get.side_effect = [
        _make_429_response(),
        _make_429_response(),
        _make_response(200, {"data": []}),
    ]

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    resp = _vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)

    assert resp.status_code == 200
    assert mock_get.call_count == 3
    assert mock_sleep.call_count == 2


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_exhausts_retries_on_persistent_429(mock_sleep, mock_get, monkeypatch):
    """Persistent 429s should exhaust retries and raise."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    mock_get.return_value = _make_429_response()

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    with pytest.raises(requests.exceptions.RequestException):
        _vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)

    assert mock_get.call_count == 3


# ===========================================================================
# _vc_get_with_retry — retry on 5xx
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_retries_on_503_then_succeeds(mock_sleep, mock_get, monkeypatch):
    """503 Service Unavailable should be retried."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    mock_get.side_effect = [
        _make_5xx_response(503),
        _make_response(200, {"data": []}),
    ]

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    resp = _vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)
    assert resp.status_code == 200
    assert mock_get.call_count == 2


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_retries_on_500(mock_sleep, mock_get, monkeypatch):
    """500 Internal Server Error should also be retried."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "2")

    mock_get.side_effect = [
        _make_5xx_response(500),
        _make_response(200, {"data": []}),
    ]

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    resp = _vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)
    assert resp.status_code == 200


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_all_5xx_status_codes_are_retried(mock_sleep, mock_get, monkeypatch):
    """All five retryable 5xx codes should trigger retry."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "2")

    for status in (500, 502, 503, 504):
        mock_get.reset_mock()
        mock_get.side_effect = [
            _make_5xx_response(status),
            _make_response(200, {"data": []}),
        ]

        from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

        resp = _vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)
        assert resp.status_code == 200, f"{status} was not retried"
        assert mock_get.call_count == 2, f"{status} call count mismatch"


# ===========================================================================
# _vc_get_with_retry — non-retryable errors
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_401_not_retried(mock_sleep, mock_get, monkeypatch):
    """401 Unauthorized is non-retryable — should raise immediately."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    mock_get.return_value = _make_response(401)

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    with pytest.raises(requests.exceptions.RequestException):
        _vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)

    assert mock_get.call_count == 1
    mock_sleep.assert_not_called()


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_403_not_retried(mock_sleep, mock_get, monkeypatch):
    """403 Forbidden is non-retryable — should raise immediately."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    mock_get.return_value = _make_response(403)

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    with pytest.raises(requests.exceptions.RequestException):
        _vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)

    assert mock_get.call_count == 1
    mock_sleep.assert_not_called()


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_404_not_retried(mock_sleep, mock_get, monkeypatch):
    """404 Not Found is non-retryable — should raise immediately."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    mock_get.return_value = _make_response(404)

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    with pytest.raises(requests.exceptions.RequestException):
        _vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)

    assert mock_get.call_count == 1
    mock_sleep.assert_not_called()


# ===========================================================================
# _vc_get_with_retry — network errors
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_connection_error_retried(mock_sleep, mock_get, monkeypatch):
    """Connection errors (no HTTP response) should be retried."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    mock_get.side_effect = [
        requests.exceptions.ConnectionError("Connection refused"),
        _make_response(200, {"data": []}),
    ]

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    resp = _vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)
    assert resp.status_code == 200
    assert mock_get.call_count == 2


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_timeout_retried(mock_sleep, mock_get, monkeypatch):
    """Timeout errors should be retried."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    mock_get.side_effect = [
        requests.exceptions.Timeout("Read timeout"),
        _make_response(200, {"data": []}),
    ]

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    resp = _vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)
    assert resp.status_code == 200
    assert mock_get.call_count == 2


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_persistent_connection_error_raises(mock_sleep, mock_get, monkeypatch):
    """Persistent connection errors should exhaust retries and raise."""
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    mock_get.side_effect = requests.exceptions.ConnectionError("network failure")

    from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

    with pytest.raises(requests.exceptions.RequestException):
        _vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)

    assert mock_get.call_count == 3


# ===========================================================================
# _vc_get_with_retry — back-off timing
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
@patch("manus_agent.tools.get_vulncheck_data._VC_MAX_RETRIES", 3)
def test_backoff_delay_doubles_each_retry(mock_sleep, mock_get):
    """Sleep delays must double each attempt: base, 2*base, 4*base…"""
    base = 2.0

    mock_get.side_effect = [
        _make_429_response(),
        _make_429_response(),
        _make_response(200, {"data": []}),
    ]

    import manus_agent.tools.get_vulncheck_data as vc_mod

    orig_base = vc_mod._VC_RETRY_BASE_DELAY
    try:
        vc_mod._VC_RETRY_BASE_DELAY = base
        vc_mod._vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)
    finally:
        vc_mod._VC_RETRY_BASE_DELAY = orig_base

    assert mock_sleep.call_count == 2
    sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
    assert sleep_calls[0] == pytest.approx(base * 1)  # attempt 1 → sleep base * 2^0
    assert sleep_calls[1] == pytest.approx(base * 2)  # attempt 2 → sleep base * 2^1


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_no_sleep_after_final_retry(mock_sleep, mock_get):
    """No sleep should happen after the last failed attempt."""
    import manus_agent.tools.get_vulncheck_data as vc_mod

    orig_max = vc_mod._VC_MAX_RETRIES
    try:
        vc_mod._VC_MAX_RETRIES = 2
        mock_get.return_value = _make_429_response()

        with pytest.raises(requests.exceptions.RequestException):
            vc_mod._vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)
    finally:
        vc_mod._VC_MAX_RETRIES = orig_max

    # With 2 max retries, sleep only once (between attempt 1 and 2).
    assert mock_sleep.call_count == 1


# ===========================================================================
# Integration: get_vulncheck_data tool — retry behaviour visible at tool level
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_tool_retries_kev_on_429(mock_sleep, mock_get, monkeypatch):
    """get_vulncheck_data should retry KEV 429 transparently."""
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    mock_get.side_effect = [
        _make_429_response(),  # KEV attempt 1 → 429
        _make_kev_hit_response(),  # KEV attempt 2 → success
        _make_nvd2_hit_response(),  # NVD2 attempt 1 → success
    ]

    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())

    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["available"] is True
    assert payload["kev"]["in_kev"] is True
    # 3 total HTTP calls: 1 failed KEV + 1 successful KEV + 1 successful NVD2
    assert mock_get.call_count == 3


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_tool_retries_nvd2_on_503(mock_sleep, mock_get, monkeypatch):
    """get_vulncheck_data should retry NVD2 503 transparently."""
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    mock_get.side_effect = [
        _make_kev_hit_response(),  # KEV success
        _make_5xx_response(503),  # NVD2 attempt 1 → 503
        _make_nvd2_hit_response(),  # NVD2 attempt 2 → success
    ]

    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())

    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["nvd2"]["description"] == "Test CVE."
    assert payload["error"] is None


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_tool_records_error_after_kev_exhausted(mock_sleep, mock_get, monkeypatch):
    """When KEV retries are exhausted the tool records an error (does not raise)."""
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-key")

    import manus_agent.tools.get_vulncheck_data as vc_mod

    orig_max = vc_mod._VC_MAX_RETRIES
    orig_delay = vc_mod._VC_RETRY_BASE_DELAY
    try:
        vc_mod._VC_MAX_RETRIES = 2
        vc_mod._VC_RETRY_BASE_DELAY = 0.0

        mock_get.side_effect = [
            _make_429_response(),  # KEV attempt 1 → 429
            _make_429_response(),  # KEV attempt 2 → 429 (exhausted)
            _make_nvd2_hit_response(),  # NVD2 still attempted and succeeds
        ]

        result = vc_mod.get_vulncheck_data(_make_tool_use())
    finally:
        vc_mod._VC_MAX_RETRIES = orig_max
        vc_mod._VC_RETRY_BASE_DELAY = orig_delay

    # Tool should still return success (partial result).
    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["error"] is not None
    assert payload["kev"]["in_kev"] is False  # default fallback


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_tool_no_retry_on_401(mock_sleep, mock_get, monkeypatch):
    """401 must not be retried — wrong API key should surface immediately."""
    monkeypatch.setenv("VULNCHECK_API_KEY", "bad-key")
    monkeypatch.setenv("VULNCHECK_RETRY_BASE_DELAY", "0")
    monkeypatch.setenv("VULNCHECK_MAX_RETRIES", "3")

    # Only one call per endpoint because 401 is non-retryable.
    mock_get.side_effect = [
        _make_response(401),  # KEV 401 — should not retry
        _make_response(401),  # NVD2 401 — should not retry
    ]

    from manus_agent.tools.get_vulncheck_data import get_vulncheck_data

    result = get_vulncheck_data(_make_tool_use())

    # Tool returns success with error message (graceful degradation).
    assert result["status"] == "success"
    payload = result["content"][0]["json"]
    assert payload["error"] is not None
    mock_sleep.assert_not_called()


# ===========================================================================
# _vc_get_with_retry — env var overrides
# ===========================================================================


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_max_retries_module_var_one(mock_sleep, mock_get):
    """_VC_MAX_RETRIES=1 should mean no retries (single attempt)."""
    import manus_agent.tools.get_vulncheck_data as vc_mod

    orig_max = vc_mod._VC_MAX_RETRIES
    try:
        vc_mod._VC_MAX_RETRIES = 1
        mock_get.return_value = _make_429_response()

        with pytest.raises(requests.exceptions.RequestException):
            vc_mod._vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)
    finally:
        vc_mod._VC_MAX_RETRIES = orig_max

    assert mock_get.call_count == 1
    mock_sleep.assert_not_called()


@patch("manus_agent.tools.get_vulncheck_data.requests.get")
@patch("manus_agent.tools.get_vulncheck_data.time.sleep")
def test_retry_base_delay_zero_sleeps_zero(mock_sleep, mock_get):
    """_VC_RETRY_BASE_DELAY=0 should sleep 0 seconds (instant retry)."""
    import manus_agent.tools.get_vulncheck_data as vc_mod

    orig_max = vc_mod._VC_MAX_RETRIES
    orig_delay = vc_mod._VC_RETRY_BASE_DELAY
    try:
        vc_mod._VC_MAX_RETRIES = 2
        vc_mod._VC_RETRY_BASE_DELAY = 0.0

        mock_get.side_effect = [
            _make_429_response(),
            _make_response(200, {"data": []}),
        ]

        vc_mod._vc_get_with_retry("https://example.com", headers={}, params={}, timeout=5)
    finally:
        vc_mod._VC_MAX_RETRIES = orig_max
        vc_mod._VC_RETRY_BASE_DELAY = orig_delay

    mock_sleep.assert_called_once_with(0.0)
