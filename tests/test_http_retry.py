"""Comprehensive test suite for manus_agent.utils.http_retry.

100% mocked — no real HTTP calls. Tests cover:
- Successful first-attempt responses
- Exponential back-off retry on retryable status codes (429, 500, 502, 503, 504)
- Non-retryable 4xx fail-fast behaviour
- Network-level error retries (ConnectionError, Timeout)
- raise_for_status=False mode
- Custom headers, params, timeout passthrough
- Custom retryable_statuses override
- sleep_fn injection for deterministic testing
- Edge cases: max_retries=0, base_delay=0
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
import requests

from manus_agent.utils.http_retry import (
    DEFAULT_RETRYABLE_STATUSES,
    http_get_with_retry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    """Create a mock requests.Response with the given status code."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = ""
    resp.content = b""
    # raise_for_status should raise for 4xx/5xx
    if 400 <= status_code < 600:
        http_error = requests.exceptions.HTTPError(
            f"HTTP {status_code}",
            response=resp,
        )
        resp.raise_for_status.side_effect = http_error
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Default retryable statuses
# ---------------------------------------------------------------------------


class TestDefaultRetryableStatuses:
    """Verify the default set of retryable HTTP status codes."""

    def test_contains_429(self):
        assert 429 in DEFAULT_RETRYABLE_STATUSES

    def test_contains_500(self):
        assert 500 in DEFAULT_RETRYABLE_STATUSES

    def test_contains_502(self):
        assert 502 in DEFAULT_RETRYABLE_STATUSES

    def test_contains_503(self):
        assert 503 in DEFAULT_RETRYABLE_STATUSES

    def test_contains_504(self):
        assert 504 in DEFAULT_RETRYABLE_STATUSES

    def test_does_not_contain_400(self):
        assert 400 not in DEFAULT_RETRYABLE_STATUSES

    def test_does_not_contain_401(self):
        assert 401 not in DEFAULT_RETRYABLE_STATUSES

    def test_does_not_contain_403(self):
        assert 403 not in DEFAULT_RETRYABLE_STATUSES

    def test_does_not_contain_404(self):
        assert 404 not in DEFAULT_RETRYABLE_STATUSES

    def test_is_frozenset(self):
        assert isinstance(DEFAULT_RETRYABLE_STATUSES, frozenset)


# ---------------------------------------------------------------------------
# Successful first-attempt responses
# ---------------------------------------------------------------------------


class TestSuccessfulFirstAttempt:
    """Responses that succeed on the first try without any retries."""

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_200_ok(self, mock_get):
        resp = _mock_response(200, {"data": "ok"})
        mock_get.return_value = resp
        sleep_fn = MagicMock()

        result = http_get_with_retry("https://api.example.com/v1", sleep_fn=sleep_fn)

        assert result is resp
        mock_get.assert_called_once()
        sleep_fn.assert_not_called()

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_headers_passthrough(self, mock_get):
        resp = _mock_response(200)
        mock_get.return_value = resp

        http_get_with_retry(
            "https://api.example.com",
            headers={"Authorization": "Bearer tok123"},
            sleep_fn=MagicMock(),
        )

        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer tok123"

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_params_passthrough(self, mock_get):
        resp = _mock_response(200)
        mock_get.return_value = resp

        http_get_with_retry(
            "https://api.example.com",
            params={"cve": "CVE-2024-1234"},
            sleep_fn=MagicMock(),
        )

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["cve"] == "CVE-2024-1234"

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_custom_timeout_passthrough(self, mock_get):
        resp = _mock_response(200)
        mock_get.return_value = resp

        http_get_with_retry("https://api.example.com", timeout=30, sleep_fn=MagicMock())

        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] == 30

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_default_timeout_is_15(self, mock_get):
        resp = _mock_response(200)
        mock_get.return_value = resp

        http_get_with_retry("https://api.example.com", sleep_fn=MagicMock())

        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] == 15

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_default_empty_headers(self, mock_get):
        resp = _mock_response(200)
        mock_get.return_value = resp

        http_get_with_retry("https://api.example.com", sleep_fn=MagicMock())

        _, kwargs = mock_get.call_args
        assert kwargs["headers"] == {}

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_default_empty_params(self, mock_get):
        resp = _mock_response(200)
        mock_get.return_value = resp

        http_get_with_retry("https://api.example.com", sleep_fn=MagicMock())

        _, kwargs = mock_get.call_args
        assert kwargs["params"] == {}

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_201_accepted(self, mock_get):
        resp = _mock_response(201)
        mock_get.return_value = resp

        result = http_get_with_retry("https://api.example.com", sleep_fn=MagicMock())
        assert result.status_code == 201

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_304_not_modified(self, mock_get):
        resp = _mock_response(304)
        mock_get.return_value = resp

        result = http_get_with_retry("https://api.example.com", sleep_fn=MagicMock())
        assert result.status_code == 304


# ---------------------------------------------------------------------------
# Retryable status codes
# ---------------------------------------------------------------------------


class TestRetryableStatusCodes:
    """Retry behaviour on 429 and 5xx responses."""

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_429_retries_then_succeeds(self, mock_get):
        fail = _mock_response(429)
        success = _mock_response(200, {"ok": True})
        mock_get.side_effect = [fail, success]
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://api.example.com",
            max_retries=3,
            base_delay=1.0,
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 200
        assert mock_get.call_count == 2
        sleep_fn.assert_called_once_with(1.0)  # base_delay * 2^0

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_500_retries_then_succeeds(self, mock_get):
        fail = _mock_response(500)
        success = _mock_response(200)
        mock_get.side_effect = [fail, success]
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://api.example.com",
            max_retries=2,
            base_delay=0.5,
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 200
        sleep_fn.assert_called_once_with(0.5)

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_502_retries_then_succeeds(self, mock_get):
        fail = _mock_response(502)
        success = _mock_response(200)
        mock_get.side_effect = [fail, success]
        sleep_fn = MagicMock()

        result = http_get_with_retry("https://api.example.com", sleep_fn=sleep_fn)
        assert result.status_code == 200

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_503_retries_then_succeeds(self, mock_get):
        fail = _mock_response(503)
        success = _mock_response(200)
        mock_get.side_effect = [fail, success]
        sleep_fn = MagicMock()

        result = http_get_with_retry("https://api.example.com", sleep_fn=sleep_fn)
        assert result.status_code == 200

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_504_retries_then_succeeds(self, mock_get):
        fail = _mock_response(504)
        success = _mock_response(200)
        mock_get.side_effect = [fail, success]
        sleep_fn = MagicMock()

        result = http_get_with_retry("https://api.example.com", sleep_fn=sleep_fn)
        assert result.status_code == 200

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_multiple_retries_before_success(self, mock_get):
        """Three failures then success on attempt 4."""
        f1 = _mock_response(429)
        f2 = _mock_response(503)
        f3 = _mock_response(500)
        success = _mock_response(200)
        mock_get.side_effect = [f1, f2, f3, success]
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://api.example.com",
            max_retries=3,
            base_delay=1.0,
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 200
        assert mock_get.call_count == 4
        # Sleep calls: 1.0 (2^0), 2.0 (2^1), 4.0 (2^2)
        assert sleep_fn.call_args_list == [call(1.0), call(2.0), call(4.0)]

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_exponential_backoff_delays(self, mock_get):
        """Verify exact exponential back-off with base_delay=2.0."""
        f1 = _mock_response(429)
        f2 = _mock_response(429)
        f3 = _mock_response(429)
        success = _mock_response(200)
        mock_get.side_effect = [f1, f2, f3, success]
        sleep_fn = MagicMock()

        http_get_with_retry(
            "https://api.example.com",
            max_retries=3,
            base_delay=2.0,
            sleep_fn=sleep_fn,
        )

        # Delays: 2.0 * 2^0 = 2.0, 2.0 * 2^1 = 4.0, 2.0 * 2^2 = 8.0
        assert sleep_fn.call_args_list == [call(2.0), call(4.0), call(8.0)]


# ---------------------------------------------------------------------------
# Exhausted retries
# ---------------------------------------------------------------------------


class TestExhaustedRetries:
    """Behaviour when all retries are used up on retryable errors."""

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_all_429_raises_after_max_retries(self, mock_get):
        mock_get.return_value = _mock_response(429)
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            http_get_with_retry(
                "https://api.example.com",
                max_retries=2,
                sleep_fn=sleep_fn,
            )

        assert "429" in str(exc_info.value)
        assert mock_get.call_count == 3  # 1 initial + 2 retries

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_all_500_raises_after_max_retries(self, mock_get):
        mock_get.return_value = _mock_response(500)
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.HTTPError):
            http_get_with_retry(
                "https://api.example.com",
                max_retries=1,
                sleep_fn=sleep_fn,
            )

        assert mock_get.call_count == 2

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_exhausted_retries_with_raise_for_status_false(self, mock_get):
        """When raise_for_status=False, return the last response instead of raising."""
        resp = _mock_response(429)
        mock_get.return_value = resp
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://api.example.com",
            max_retries=2,
            raise_for_status=False,
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 429
        assert mock_get.call_count == 3


# ---------------------------------------------------------------------------
# Non-retryable HTTP errors (fail fast)
# ---------------------------------------------------------------------------


class TestNonRetryableErrors:
    """4xx errors not in the retryable set should fail immediately."""

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_401_fails_immediately(self, mock_get):
        mock_get.return_value = _mock_response(401)
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            http_get_with_retry("https://api.example.com", sleep_fn=sleep_fn)

        assert "401" in str(exc_info.value)
        mock_get.assert_called_once()
        sleep_fn.assert_not_called()

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_403_fails_immediately(self, mock_get):
        mock_get.return_value = _mock_response(403)
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.HTTPError):
            http_get_with_retry("https://api.example.com", sleep_fn=sleep_fn)

        mock_get.assert_called_once()
        sleep_fn.assert_not_called()

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_404_fails_immediately(self, mock_get):
        mock_get.return_value = _mock_response(404)
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.HTTPError):
            http_get_with_retry("https://api.example.com", sleep_fn=sleep_fn)

        mock_get.assert_called_once()

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_404_with_raise_for_status_false(self, mock_get):
        """When raise_for_status=False, 404 returns the response object."""
        resp = _mock_response(404)
        mock_get.return_value = resp
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://api.example.com",
            raise_for_status=False,
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 404
        mock_get.assert_called_once()

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_422_fails_immediately(self, mock_get):
        mock_get.return_value = _mock_response(422)
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.HTTPError):
            http_get_with_retry("https://api.example.com", sleep_fn=sleep_fn)

        mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# Network-level errors (ConnectionError, Timeout)
# ---------------------------------------------------------------------------


class TestNetworkErrors:
    """Retry on network-level exceptions (no HTTP response)."""

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_connection_error_retries_then_succeeds(self, mock_get):
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            _mock_response(200),
        ]
        sleep_fn = MagicMock()

        result = http_get_with_retry("https://api.example.com", sleep_fn=sleep_fn)

        assert result.status_code == 200
        assert mock_get.call_count == 2
        sleep_fn.assert_called_once()

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_timeout_error_retries_then_succeeds(self, mock_get):
        mock_get.side_effect = [
            requests.exceptions.Timeout("Request timed out"),
            _mock_response(200),
        ]
        sleep_fn = MagicMock()

        result = http_get_with_retry("https://api.example.com", sleep_fn=sleep_fn)

        assert result.status_code == 200

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_connection_error_exhausted_raises(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.ConnectionError):
            http_get_with_retry(
                "https://api.example.com",
                max_retries=2,
                sleep_fn=sleep_fn,
            )

        assert mock_get.call_count == 3

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_timeout_error_exhausted_raises(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.Timeout):
            http_get_with_retry(
                "https://api.example.com",
                max_retries=1,
                sleep_fn=sleep_fn,
            )

        assert mock_get.call_count == 2

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_mixed_network_and_status_errors(self, mock_get):
        """ConnectionError then 429 then success."""
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("DNS failure"),
            _mock_response(429),
            _mock_response(200),
        ]
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://api.example.com",
            max_retries=3,
            base_delay=1.0,
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 200
        assert mock_get.call_count == 3


# ---------------------------------------------------------------------------
# Custom retryable_statuses
# ---------------------------------------------------------------------------


class TestCustomRetryableStatuses:
    """Override the default set of retryable status codes."""

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_custom_retryable_includes_418(self, mock_get):
        """Retry on 418 when it's in the custom set."""
        fail = _mock_response(418)
        success = _mock_response(200)
        mock_get.side_effect = [fail, success]
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://api.example.com",
            retryable_statuses=frozenset({418}),
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 200
        assert mock_get.call_count == 2

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_429_not_retried_when_removed_from_set(self, mock_get):
        """429 fails immediately when it's not in the custom retryable set."""
        mock_get.return_value = _mock_response(429)
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.HTTPError):
            http_get_with_retry(
                "https://api.example.com",
                retryable_statuses=frozenset({500}),
                sleep_fn=sleep_fn,
            )

        mock_get.assert_called_once()

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_empty_retryable_set_no_retries_on_500(self, mock_get):
        """With an empty retryable set, even 500 fails immediately."""
        mock_get.return_value = _mock_response(500)
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.HTTPError):
            http_get_with_retry(
                "https://api.example.com",
                retryable_statuses=frozenset(),
                sleep_fn=sleep_fn,
            )

        mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case behaviour for parameter boundaries."""

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_max_retries_zero_no_retry(self, mock_get):
        """max_retries=0 means exactly one attempt, no retries."""
        mock_get.return_value = _mock_response(429)
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.HTTPError):
            http_get_with_retry(
                "https://api.example.com",
                max_retries=0,
                sleep_fn=sleep_fn,
            )

        mock_get.assert_called_once()
        sleep_fn.assert_not_called()

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_max_retries_zero_success(self, mock_get):
        """max_retries=0 still returns on first success."""
        mock_get.return_value = _mock_response(200)
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://api.example.com",
            max_retries=0,
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 200

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_base_delay_zero(self, mock_get):
        """base_delay=0 means sleep(0) between retries."""
        fail = _mock_response(429)
        success = _mock_response(200)
        mock_get.side_effect = [fail, success]
        sleep_fn = MagicMock()

        http_get_with_retry(
            "https://api.example.com",
            base_delay=0.0,
            sleep_fn=sleep_fn,
        )

        sleep_fn.assert_called_once_with(0.0)

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_large_max_retries(self, mock_get):
        """Many retries eventually succeed."""
        failures = [_mock_response(503)] * 9
        success = _mock_response(200)
        mock_get.side_effect = failures + [success]
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://api.example.com",
            max_retries=10,
            base_delay=0.0,
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 200
        assert mock_get.call_count == 10


# ---------------------------------------------------------------------------
# Integration-style: NVD-like configuration
# ---------------------------------------------------------------------------


class TestNvdLikeConfig:
    """Verify the utility works with NVD-like parameters (base_delay=2)."""

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_nvd_style_retry_with_api_key(self, mock_get):
        fail = _mock_response(429)
        success = _mock_response(200, {"vulnerabilities": [{"cve": "data"}]})
        mock_get.side_effect = [fail, success]
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=CVE-2024-3094",
            headers={"apiKey": "test-key"},
            max_retries=3,
            base_delay=2.0,
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 200
        sleep_fn.assert_called_once_with(2.0)
        _, kwargs = mock_get.call_args_list[0]
        assert kwargs["headers"]["apiKey"] == "test-key"


# ---------------------------------------------------------------------------
# Integration-style: VulnCheck-like configuration
# ---------------------------------------------------------------------------


class TestVulncheckLikeConfig:
    """Verify the utility works with VulnCheck-like parameters."""

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_vulncheck_style_with_bearer_token(self, mock_get):
        fail = _mock_response(429)
        success = _mock_response(200, {"data": [{"cve": "info"}]})
        mock_get.side_effect = [fail, success]
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://api.vulncheck.com/v3/index/vulncheck-kev",
            headers={"Authorization": "Bearer vc-token", "Accept": "application/json"},
            params={"cve": "CVE-2024-3094"},
            timeout=20,
            max_retries=3,
            base_delay=1.0,
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 200
        _, kwargs = mock_get.call_args_list[0]
        assert kwargs["params"]["cve"] == "CVE-2024-3094"
        assert kwargs["timeout"] == 20

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_vulncheck_401_fails_immediately(self, mock_get):
        """VulnCheck 401 (bad API key) should not retry."""
        mock_get.return_value = _mock_response(401)
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.HTTPError):
            http_get_with_retry(
                "https://api.vulncheck.com/v3/index/vulncheck-kev",
                headers={"Authorization": "Bearer bad-key"},
                sleep_fn=sleep_fn,
            )

        mock_get.assert_called_once()


# ---------------------------------------------------------------------------
# Integration-style: OSV-like configuration
# ---------------------------------------------------------------------------


class TestOsvLikeConfig:
    """Verify the utility works with OSV-like parameters (raise_for_status=False)."""

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_osv_style_404_returned_not_raised(self, mock_get):
        """OSV uses raise_for_status=False to handle 404 for unknown IDs."""
        resp = _mock_response(404)
        mock_get.return_value = resp
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://api.osv.dev/v1/vulns/CVE-9999-0000",
            headers={"Accept": "application/json"},
            raise_for_status=False,
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 404
        mock_get.assert_called_once()

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_osv_style_429_still_retried_with_raise_for_status_false(self, mock_get):
        """Even with raise_for_status=False, retryable codes are retried."""
        fail = _mock_response(429)
        success = _mock_response(200, {"id": "CVE-2024-3094"})
        mock_get.side_effect = [fail, success]
        sleep_fn = MagicMock()

        result = http_get_with_retry(
            "https://api.osv.dev/v1/vulns/CVE-2024-3094",
            raise_for_status=False,
            sleep_fn=sleep_fn,
        )

        assert result.status_code == 200
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# Sleep function injection
# ---------------------------------------------------------------------------


class TestSleepFnInjection:
    """Verify the sleep_fn parameter is used correctly."""

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_sleep_fn_called_with_correct_delays(self, mock_get):
        f1 = _mock_response(429)
        f2 = _mock_response(503)
        success = _mock_response(200)
        mock_get.side_effect = [f1, f2, success]
        sleep_fn = MagicMock()

        http_get_with_retry(
            "https://api.example.com",
            max_retries=3,
            base_delay=0.5,
            sleep_fn=sleep_fn,
        )

        # Delays: 0.5 * 2^0 = 0.5, 0.5 * 2^1 = 1.0
        assert sleep_fn.call_args_list == [call(0.5), call(1.0)]

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_default_sleep_fn_is_time_sleep(self, mock_get):
        """When sleep_fn is not provided, time.sleep is used."""
        fail = _mock_response(429)
        success = _mock_response(200)
        mock_get.side_effect = [fail, success]

        with patch("manus_agent.utils.http_retry.time.sleep") as mock_sleep:
            http_get_with_retry(
                "https://api.example.com",
                max_retries=1,
                base_delay=1.0,
            )

            mock_sleep.assert_called_once_with(1.0)


# ---------------------------------------------------------------------------
# Refactored tool modules still work
# ---------------------------------------------------------------------------


class TestRefactoredModulesImport:
    """Verify the refactored tool modules can import the shared utility."""

    def test_nvd_data_imports_http_retry(self):
        from manus_agent.tools.get_nvd_data import _nvd_get_with_retry

        assert callable(_nvd_get_with_retry)

    def test_vulncheck_data_imports_http_retry(self):
        from manus_agent.tools.get_vulncheck_data import _vc_get_with_retry

        assert callable(_vc_get_with_retry)

    def test_osv_data_imports_http_retry(self):
        from manus_agent.tools.get_osv_data import _osv_get_with_retry

        assert callable(_osv_get_with_retry)

    def test_http_retry_importable(self):
        from manus_agent.utils.http_retry import http_get_with_retry

        assert callable(http_get_with_retry)

    def test_default_statuses_importable(self):
        from manus_agent.utils.http_retry import DEFAULT_RETRYABLE_STATUSES

        assert isinstance(DEFAULT_RETRYABLE_STATUSES, frozenset)


# ---------------------------------------------------------------------------
# Regression: HTTPError with response attribute
# ---------------------------------------------------------------------------


class TestHttpErrorResponseAttribute:
    """Ensure HTTPError exceptions carry the response object for callers."""

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_retryable_status_exception_has_response(self, mock_get):
        resp = _mock_response(429)
        mock_get.return_value = resp
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            http_get_with_retry(
                "https://api.example.com",
                max_retries=0,
                sleep_fn=sleep_fn,
            )

        assert exc_info.value.response is not None
        assert exc_info.value.response.status_code == 429

    @patch("manus_agent.utils.http_retry.requests.get")
    def test_non_retryable_exception_has_response(self, mock_get):
        resp = _mock_response(403)
        mock_get.return_value = resp
        sleep_fn = MagicMock()

        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            http_get_with_retry("https://api.example.com", sleep_fn=sleep_fn)

        assert exc_info.value.response is not None
        assert exc_info.value.response.status_code == 403
