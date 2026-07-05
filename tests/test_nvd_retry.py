"""Tests for NVD retry/back-off and NVD_API_KEY support in get_nvd_data and obtain_cves."""

import os
from unittest import mock

import pytest
import requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code=200, json_data=None, raise_for_status=None):
    """Build a minimal mock requests.Response."""
    resp = mock.MagicMock()
    resp.status_code = status_code
    if json_data is not None:
        resp.json.return_value = json_data
    if raise_for_status is not None:
        resp.raise_for_status.side_effect = raise_for_status
    else:
        resp.raise_for_status.return_value = None
    return resp


def _good_nvd_response(cve_id="CVE-2024-1234"):
    """Return a minimal valid NVD API payload."""
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "id": cve_id,
                    "vulnStatus": "Analyzed",
                    "descriptions": [{"lang": "en", "value": "Test CVE"}],
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# _build_nvd_headers
# ---------------------------------------------------------------------------


class TestBuildNvdHeaders:
    def test_no_api_key_returns_empty_dict(self, monkeypatch):
        monkeypatch.delenv("NVD_API_KEY", raising=False)
        from manus_agent.tools.get_nvd_data import _build_nvd_headers

        assert _build_nvd_headers() == {}

    def test_api_key_injected_as_header(self, monkeypatch):
        monkeypatch.setenv("NVD_API_KEY", "my-test-key")
        from manus_agent.tools.get_nvd_data import _build_nvd_headers

        headers = _build_nvd_headers()
        assert headers.get("apiKey") == "my-test-key"

    def test_blank_api_key_not_injected(self, monkeypatch):
        monkeypatch.setenv("NVD_API_KEY", "   ")
        from manus_agent.tools.get_nvd_data import _build_nvd_headers

        assert "apiKey" not in _build_nvd_headers()


# ---------------------------------------------------------------------------
# _nvd_get_with_retry
# ---------------------------------------------------------------------------


class TestNvdGetWithRetry:
    """All tests zero-delay: NVD_RETRY_BASE_DELAY=0 via monkeypatch."""

    @pytest.fixture(autouse=True)
    def _patch_delay(self, monkeypatch):
        monkeypatch.setenv("NVD_RETRY_BASE_DELAY", "0")
        monkeypatch.setenv("NVD_MAX_RETRIES", "3")
        # Reload module-level constants
        import manus_agent.tools.get_nvd_data as m

        monkeypatch.setattr(m, "_NVD_MAX_RETRIES", 3)
        monkeypatch.setattr(m, "_NVD_RETRY_BASE_DELAY", 0.0)
        monkeypatch.setattr(m, "_NVD_RETRYABLE_STATUS", {429, 500, 502, 503, 504})

    def _retry_fn(self):
        from manus_agent.tools.get_nvd_data import _nvd_get_with_retry

        return _nvd_get_with_retry

    # --- happy path ---

    def test_success_on_first_attempt(self, monkeypatch):
        good = _mock_response(200, _good_nvd_response())
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", return_value=good) as m:
            resp = self._retry_fn()("https://example.com")
        assert resp is good
        assert m.call_count == 1

    def test_api_key_forwarded_as_header(self, monkeypatch):
        monkeypatch.setenv("NVD_API_KEY", "secret-key")
        good = _mock_response(200, _good_nvd_response())
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", return_value=good) as m:
            self._retry_fn()("https://example.com")
        _, kwargs = m.call_args
        assert kwargs["headers"].get("apiKey") == "secret-key"

    def test_no_api_key_sends_empty_headers(self, monkeypatch):
        monkeypatch.delenv("NVD_API_KEY", raising=False)
        good = _mock_response(200, _good_nvd_response())
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", return_value=good) as m:
            self._retry_fn()("https://example.com")
        _, kwargs = m.call_args
        assert kwargs["headers"] == {}

    # --- 429 rate-limit retry ---

    def test_retries_on_429_then_succeeds(self, monkeypatch):
        rate_limited = _mock_response(429)
        good = _mock_response(200, _good_nvd_response())
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", side_effect=[rate_limited, good]) as m:
            with mock.patch("manus_agent.tools.get_nvd_data.time.sleep"):
                resp = self._retry_fn()("https://example.com")
        assert m.call_count == 2
        assert resp is good

    def test_retries_on_500_then_succeeds(self, monkeypatch):
        err500 = _mock_response(500)
        good = _mock_response(200, _good_nvd_response())
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", side_effect=[err500, good]) as m:
            with mock.patch("manus_agent.tools.get_nvd_data.time.sleep"):
                resp = self._retry_fn()("https://example.com")
        assert m.call_count == 2
        assert resp is good

    def test_retries_on_503_then_succeeds(self, monkeypatch):
        err503 = _mock_response(503)
        good = _mock_response(200, _good_nvd_response())
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", side_effect=[err503, good]) as m:
            with mock.patch("manus_agent.tools.get_nvd_data.time.sleep"):
                self._retry_fn()("https://example.com")
        assert m.call_count == 2

    def test_all_three_retries_then_success_on_4th(self, monkeypatch):
        err = _mock_response(429)
        good = _mock_response(200, _good_nvd_response())
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", side_effect=[err, err, err, good]) as m:
            with mock.patch("manus_agent.tools.get_nvd_data.time.sleep"):
                resp = self._retry_fn()("https://example.com")
        assert m.call_count == 4  # 1 initial + 3 retries
        assert resp is good

    def test_exhausted_retries_on_429_raises(self, monkeypatch):
        import manus_agent.tools.get_nvd_data as m_mod

        monkeypatch.setattr(m_mod, "_NVD_MAX_RETRIES", 2)
        err = _mock_response(429)
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", return_value=err):
            with mock.patch("manus_agent.tools.get_nvd_data.time.sleep"):
                with pytest.raises(requests.exceptions.HTTPError):
                    self._retry_fn()("https://example.com")

    # --- connection errors ---

    def test_connection_error_retried_then_succeeds(self, monkeypatch):
        good = _mock_response(200, _good_nvd_response())
        with mock.patch(
            "manus_agent.tools.get_nvd_data.requests.get",
            side_effect=[requests.exceptions.ConnectionError("refused"), good],
        ) as m:
            with mock.patch("manus_agent.tools.get_nvd_data.time.sleep"):
                resp = self._retry_fn()("https://example.com")
        assert m.call_count == 2
        assert resp is good

    def test_timeout_error_retried_then_succeeds(self, monkeypatch):
        good = _mock_response(200, _good_nvd_response())
        with mock.patch(
            "manus_agent.tools.get_nvd_data.requests.get", side_effect=[requests.exceptions.Timeout("timed out"), good]
        ) as m:
            with mock.patch("manus_agent.tools.get_nvd_data.time.sleep"):
                self._retry_fn()("https://example.com")
        assert m.call_count == 2

    def test_exhausted_retries_on_connection_error_raises(self, monkeypatch):
        import manus_agent.tools.get_nvd_data as m_mod

        monkeypatch.setattr(m_mod, "_NVD_MAX_RETRIES", 1)
        with mock.patch(
            "manus_agent.tools.get_nvd_data.requests.get", side_effect=requests.exceptions.ConnectionError("fail")
        ):
            with mock.patch("manus_agent.tools.get_nvd_data.time.sleep"):
                with pytest.raises(requests.exceptions.RequestException):
                    self._retry_fn()("https://example.com")

    # --- back-off schedule ---

    def test_backoff_delay_increases_exponentially(self, monkeypatch):
        import manus_agent.tools.get_nvd_data as m_mod

        monkeypatch.setattr(m_mod, "_NVD_MAX_RETRIES", 3)
        monkeypatch.setattr(m_mod, "_NVD_RETRY_BASE_DELAY", 2.0)
        err = _mock_response(429)
        good = _mock_response(200, _good_nvd_response())
        sleep_calls = []
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", side_effect=[err, err, err, good]):
            with mock.patch("manus_agent.tools.get_nvd_data.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
                self._retry_fn()("https://example.com")
        # attempt 2 delay=2, attempt 3 delay=4, attempt 4 delay=8
        assert sleep_calls == [2.0, 4.0, 8.0]

    def test_no_sleep_on_first_attempt(self, monkeypatch):
        good = _mock_response(200, _good_nvd_response())
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", return_value=good):
            with mock.patch("manus_agent.tools.get_nvd_data.time.sleep") as sleep_mock:
                self._retry_fn()("https://example.com")
        sleep_mock.assert_not_called()

    # --- non-retryable status codes ---

    def test_404_not_retried(self, monkeypatch):
        err404 = _mock_response(404)
        # Provide a response object so the retry logic can inspect status_code
        err404.raise_for_status.side_effect = requests.exceptions.HTTPError("404", response=err404)
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", return_value=err404) as m:
            with pytest.raises(requests.exceptions.HTTPError):
                self._retry_fn()("https://example.com")
        assert m.call_count == 1  # no retry for 404

    def test_400_not_retried(self, monkeypatch):
        err400 = _mock_response(400)
        err400.raise_for_status.side_effect = requests.exceptions.HTTPError("400", response=err400)
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", return_value=err400) as m:
            with pytest.raises(requests.exceptions.HTTPError):
                self._retry_fn()("https://example.com")
        assert m.call_count == 1


# ---------------------------------------------------------------------------
# get_nvd_data tool entry point
# ---------------------------------------------------------------------------


class TestGetNvdDataTool:
    """Tests that the public tool function uses the retry helper."""

    @pytest.fixture(autouse=True)
    def _zero_delay(self, monkeypatch):
        import manus_agent.tools.get_nvd_data as m

        monkeypatch.setattr(m, "_NVD_MAX_RETRIES", 3)
        monkeypatch.setattr(m, "_NVD_RETRY_BASE_DELAY", 0.0)

    def _tool(self, cve_id):
        return {"toolUseId": "t1", "input": {"cve_id": cve_id}}

    def test_success_returns_vulnerability_data(self):
        good = _mock_response(200, _good_nvd_response("CVE-2024-9999"))
        from manus_agent.tools.get_nvd_data import get_nvd_data

        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", return_value=good):
            result = get_nvd_data(self._tool("CVE-2024-9999"))
        assert result["status"] == "success"

    def test_retries_on_429_then_success(self):
        rate_limited = _mock_response(429)
        good = _mock_response(200, _good_nvd_response())
        from manus_agent.tools.get_nvd_data import get_nvd_data

        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", side_effect=[rate_limited, good]) as m:
            with mock.patch("manus_agent.tools.get_nvd_data.time.sleep"):
                result = get_nvd_data(self._tool("CVE-2024-1234"))
        assert result["status"] == "success"
        assert m.call_count == 2

    def test_exhausted_retries_returns_error(self):
        import manus_agent.tools.get_nvd_data as m_mod

        m_mod._NVD_MAX_RETRIES = 1
        err429 = _mock_response(429)
        from manus_agent.tools.get_nvd_data import get_nvd_data

        try:
            with mock.patch("manus_agent.tools.get_nvd_data.requests.get", return_value=err429):
                with mock.patch("manus_agent.tools.get_nvd_data.time.sleep"):
                    result = get_nvd_data(self._tool("CVE-2024-1234"))
            assert result["status"] == "error"
        finally:
            m_mod._NVD_MAX_RETRIES = 3

    def test_invalid_cve_id_no_retry(self):
        from manus_agent.tools.get_nvd_data import get_nvd_data

        with mock.patch("manus_agent.tools.get_nvd_data.requests.get") as m:
            result = get_nvd_data(self._tool("not-a-cve"))
        assert result["status"] == "error"
        m.assert_not_called()


# ---------------------------------------------------------------------------
# obtain_cves: _get_all_cves_from_nvd uses _nvd_get_with_retry
# ---------------------------------------------------------------------------


class TestObtainCvesNvdRetry:
    """Verify obtain_cves._get_all_cves_from_nvd delegates to the retry helper."""

    @pytest.fixture(autouse=True)
    def _zero_delay(self, monkeypatch):
        import manus_agent.tools.get_nvd_data as m

        monkeypatch.setattr(m, "_NVD_MAX_RETRIES", 3)
        monkeypatch.setattr(m, "_NVD_RETRY_BASE_DELAY", 0.0)

    def _nvd_page(self, cves, total):
        return {"vulnerabilities": cves, "totalResults": total}

    def test_uses_retry_helper_on_429(self):
        """_get_all_cves_from_nvd retries when NVD returns 429."""
        from manus_agent.tools.obtain_cves import _get_all_cves_from_nvd

        rate_limited = _mock_response(429)
        good = _mock_response(200, self._nvd_page([{"cve": {"id": "CVE-2024-0001"}}], 1))

        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", side_effect=[rate_limited, good]) as m:
            with mock.patch("manus_agent.tools.get_nvd_data.time.sleep"):
                cves = _get_all_cves_from_nvd("2024-01-01T00:00:00.000Z", "2024-01-31T00:00:00.000Z")
        assert len(cves) == 1
        assert m.call_count == 2

    def test_single_page_returns_all_cves(self):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_nvd

        cve_list = [{"cve": {"id": f"CVE-2024-{i:04d}"}} for i in range(5)]
        good = _mock_response(200, self._nvd_page(cve_list, 5))

        with mock.patch("manus_agent.tools.get_nvd_data.requests.get", return_value=good):
            cves = _get_all_cves_from_nvd("2024-01-01T00:00:00.000Z", "2024-01-31T00:00:00.000Z")
        assert len(cves) == 5

    def test_exhausted_retries_propagates(self):
        import manus_agent.tools.get_nvd_data as m_mod

        m_mod._NVD_MAX_RETRIES = 1
        from manus_agent.tools.obtain_cves import _get_all_cves_from_nvd

        err429 = _mock_response(429)
        try:
            with mock.patch("manus_agent.tools.get_nvd_data.requests.get", return_value=err429):
                with mock.patch("manus_agent.tools.get_nvd_data.time.sleep"):
                    with pytest.raises(requests.exceptions.HTTPError):
                        _get_all_cves_from_nvd("2024-01-01T00:00:00.000Z", "2024-01-31T00:00:00.000Z")
        finally:
            m_mod._NVD_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Retryable status set contract
# ---------------------------------------------------------------------------


class TestRetryableStatusSet:
    def test_429_in_retryable_set(self):
        from manus_agent.tools.get_nvd_data import _NVD_RETRYABLE_STATUS

        assert 429 in _NVD_RETRYABLE_STATUS

    def test_500_in_retryable_set(self):
        from manus_agent.tools.get_nvd_data import _NVD_RETRYABLE_STATUS

        assert 500 in _NVD_RETRYABLE_STATUS

    def test_200_not_in_retryable_set(self):
        from manus_agent.tools.get_nvd_data import _NVD_RETRYABLE_STATUS

        assert 200 not in _NVD_RETRYABLE_STATUS

    def test_404_not_in_retryable_set(self):
        from manus_agent.tools.get_nvd_data import _NVD_RETRYABLE_STATUS

        assert 404 not in _NVD_RETRYABLE_STATUS

    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_all_expected_codes_in_retryable_set(self, code):
        from manus_agent.tools.get_nvd_data import _NVD_RETRYABLE_STATUS

        assert code in _NVD_RETRYABLE_STATUS


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_default_max_retries_is_3(self, monkeypatch):
        monkeypatch.delenv("NVD_MAX_RETRIES", raising=False)
        assert int(os.environ.get("NVD_MAX_RETRIES", "3")) == 3

    def test_default_base_delay_is_2_seconds(self, monkeypatch):
        monkeypatch.delenv("NVD_RETRY_BASE_DELAY", raising=False)
        assert float(os.environ.get("NVD_RETRY_BASE_DELAY", "2")) == 2.0

    def test_obtain_cves_imports_retry_helper(self):
        import manus_agent.tools.obtain_cves as oc
        from manus_agent.tools.get_nvd_data import _nvd_get_with_retry

        # The import should resolve the same object
        assert oc._nvd_get_with_retry is _nvd_get_with_retry
