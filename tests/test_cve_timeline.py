"""Comprehensive test suite for get_cve_timeline tool and CLI subcommand."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.get_cve_timeline import (
    _fetch_epss_events,
    _fetch_github_advisory_events,
    _fetch_kev_events,
    _fetch_nvd_events,
    _fetch_vulncheck_kev_events,
    _get_with_retry,
    build_timeline,
    handler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CVE_ID = "CVE-2021-44228"


@pytest.fixture
def nvd_response():
    """Minimal NVD API response for Log4Shell."""
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "id": CVE_ID,
                    "published": "2021-12-10T10:15:00.000",
                    "lastModified": "2023-11-07T03:38:00.000",
                    "metrics": {
                        "cvssMetricV31": [
                            {
                                "cvssData": {
                                    "baseScore": 10.0,
                                    "baseSeverity": "CRITICAL",
                                }
                            }
                        ]
                    },
                }
            }
        ]
    }


@pytest.fixture
def epss_response():
    """Minimal EPSS time-series response."""
    return {
        "data": [
            {
                "cve": CVE_ID,
                "epss": "0.97547",
                "percentile": "0.99998",
                "date": "2024-06-15",
                "time-series": [
                    {"date": "2022-01-01", "epss": "0.85000", "percentile": "0.98000"},
                    {"date": "2022-06-01", "epss": "0.92000", "percentile": "0.99000"},
                    {"date": "2024-06-15", "epss": "0.97547", "percentile": "0.99998"},
                ],
            }
        ]
    }


@pytest.fixture
def kev_response():
    """Minimal CISA KEV response containing the CVE."""
    return {
        "vulnerabilities": [
            {
                "cveID": CVE_ID,
                "dateAdded": "2021-12-10",
                "dueDate": "2021-12-24",
                "shortDescription": "Apache Log4j2 Remote Code Execution Vulnerability",
            },
            {
                "cveID": "CVE-2022-9999",
                "dateAdded": "2022-05-01",
                "dueDate": "2022-05-15",
                "shortDescription": "Other vulnerability",
            },
        ]
    }


@pytest.fixture
def github_advisory_response():
    """Minimal GitHub advisory response."""
    return [
        {
            "ghsa_id": "GHSA-jfh8-c2jp-5v3q",
            "published_at": "2021-12-10T00:00:00Z",
            "updated_at": "2024-01-15T12:00:00Z",
            "severity": "critical",
        }
    ]


@pytest.fixture
def vulncheck_kev_response():
    """Minimal VulnCheck KEV response."""
    return {
        "data": [
            {
                "cve": CVE_ID,
                "date_added": "2021-12-11T00:00:00Z",
            }
        ]
    }


# ---------------------------------------------------------------------------
# Tests: _get_with_retry
# ---------------------------------------------------------------------------


class TestGetWithRetry:
    """Tests for the HTTP retry helper."""

    @patch("manus_agent.tools.get_cve_timeline.requests.get")
    def test_success_first_attempt(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _get_with_retry("https://example.com/api")
        assert result == mock_resp
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.get_cve_timeline.time.sleep")
    @patch("manus_agent.tools.get_cve_timeline.requests.get")
    def test_retry_on_429(self, mock_get, mock_sleep):
        fail_resp = MagicMock()
        fail_resp.status_code = 429
        fail_resp.raise_for_status = MagicMock(side_effect=Exception("Rate limited"))

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [fail_resp, ok_resp]
        result = _get_with_retry("https://example.com/api")
        assert result == ok_resp
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    @patch("manus_agent.tools.get_cve_timeline._MAX_RETRIES", 2)
    @patch("manus_agent.tools.get_cve_timeline.time.sleep")
    @patch("manus_agent.tools.get_cve_timeline.requests.get")
    def test_all_retries_exhausted_raises(self, mock_get, mock_sleep):
        import requests as req

        fail_resp = MagicMock()
        fail_resp.status_code = 503
        mock_get.return_value = fail_resp
        fail_resp.raise_for_status.side_effect = req.exceptions.HTTPError(response=fail_resp)

        with pytest.raises(req.exceptions.HTTPError):
            _get_with_retry("https://example.com/api")
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.get_cve_timeline._MAX_RETRIES", 2)
    @patch("manus_agent.tools.get_cve_timeline.time.sleep")
    @patch("manus_agent.tools.get_cve_timeline.requests.get")
    def test_connection_error_retries(self, mock_get, mock_sleep):
        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("timeout")

        with pytest.raises(req.exceptions.ConnectionError):
            _get_with_retry("https://example.com/api")
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.get_cve_timeline.requests.get")
    def test_custom_headers_and_params(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        _get_with_retry(
            "https://example.com/api",
            headers={"X-Custom": "test"},
            params={"q": "value"},
            timeout=30,
        )
        mock_get.assert_called_once_with(
            "https://example.com/api",
            headers={"X-Custom": "test"},
            params={"q": "value"},
            timeout=30,
        )


# ---------------------------------------------------------------------------
# Tests: _fetch_nvd_events
# ---------------------------------------------------------------------------


class TestFetchNvdEvents:
    """Tests for NVD event fetcher."""

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_full_nvd_response(self, mock_get, nvd_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = nvd_response
        mock_get.return_value = mock_resp

        events = _fetch_nvd_events(CVE_ID)
        assert len(events) == 3  # published, modified, CVSS
        assert events[0]["event"] == "CVE published"
        assert events[0]["date"] == "2021-12-10"
        assert events[0]["source"] == "NVD"
        assert events[1]["event"] == "NVD record updated"
        assert events[1]["date"] == "2023-11-07"
        assert events[2]["event"] == "CVSS 3.1 assigned"
        assert "10.0" in events[2]["detail"]
        assert "CRITICAL" in events[2]["detail"]

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_nvd_published_only_no_modified_dup(self, mock_get):
        """When lastModified == published date, skip duplicate."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": CVE_ID,
                        "published": "2021-12-10T10:15:00.000",
                        "lastModified": "2021-12-10T10:15:00.000",
                        "metrics": {},
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_nvd_events(CVE_ID)
        assert len(events) == 1
        assert events[0]["event"] == "CVE published"

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_nvd_no_vulnerabilities(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        events = _fetch_nvd_events(CVE_ID)
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_nvd_api_failure_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        events = _fetch_nvd_events(CVE_ID)
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_nvd_cvss_v2_fallback(self, mock_get):
        """Falls back to CVSS 2.0 when 3.x not available."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": CVE_ID,
                        "published": "2015-01-01T00:00:00.000",
                        "lastModified": "2015-01-01T00:00:00.000",
                        "metrics": {
                            "cvssMetricV2": [
                                {
                                    "cvssData": {
                                        "baseScore": 7.5,
                                        "baseSeverity": "HIGH",
                                    }
                                }
                            ]
                        },
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_nvd_events(CVE_ID)
        cvss_events = [e for e in events if "CVSS" in e["event"]]
        assert len(cvss_events) == 1
        assert "2.0" in cvss_events[0]["event"]
        assert "7.5" in cvss_events[0]["detail"]

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    @patch.dict("os.environ", {"NVD_API_KEY": "test-key-123"})
    def test_nvd_uses_api_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        _fetch_nvd_events(CVE_ID)
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["apiKey"] == "test-key-123"

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    @patch.dict("os.environ", {}, clear=True)
    def test_nvd_no_api_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        _fetch_nvd_events(CVE_ID)
        call_kwargs = mock_get.call_args[1]
        assert "apiKey" not in call_kwargs.get("headers", {})


# ---------------------------------------------------------------------------
# Tests: _fetch_epss_events
# ---------------------------------------------------------------------------


class TestFetchEpssEvents:
    """Tests for EPSS event fetcher."""

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_full_epss_response(self, mock_get, epss_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = epss_response
        mock_get.return_value = mock_resp

        events = _fetch_epss_events(CVE_ID)
        assert len(events) == 2  # first seen + current
        assert events[0]["event"] == "EPSS tracking started"
        assert events[0]["date"] == "2022-01-01"
        assert "0.8500" in events[0]["detail"]
        assert events[1]["event"] == "EPSS current score"
        assert events[1]["date"] == "2024-06-15"

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_epss_single_datapoint(self, mock_get):
        """Single data point means only first-seen, no current."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "cve": CVE_ID,
                    "epss": "0.50000",
                    "percentile": "0.90000",
                    "date": "2024-01-01",
                    "time-series": [
                        {"date": "2024-01-01", "epss": "0.50000", "percentile": "0.90000"},
                    ],
                }
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_epss_events(CVE_ID)
        assert len(events) == 1
        assert events[0]["event"] == "EPSS tracking started"

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_epss_no_data(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        events = _fetch_epss_events(CVE_ID)
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_epss_empty_time_series(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "cve": CVE_ID,
                    "epss": "0.5",
                    "percentile": "0.9",
                    "date": "2024-01-01",
                    "time-series": [],
                }
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_epss_events(CVE_ID)
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_epss_api_failure_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("Timeout")
        events = _fetch_epss_events(CVE_ID)
        assert events == []


# ---------------------------------------------------------------------------
# Tests: _fetch_kev_events
# ---------------------------------------------------------------------------


class TestFetchKevEvents:
    """Tests for CISA KEV event fetcher."""

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_cve_in_kev(self, mock_get, kev_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = kev_response
        mock_get.return_value = mock_resp

        events = _fetch_kev_events(CVE_ID)
        assert len(events) == 2  # dateAdded + dueDate
        assert events[0]["event"] == "Added to CISA KEV"
        assert events[0]["date"] == "2021-12-10"
        assert "Apache Log4j2" in events[0]["detail"]
        assert events[1]["event"] == "KEV remediation deadline"
        assert events[1]["date"] == "2021-12-24"

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_cve_not_in_kev(self, mock_get, kev_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = kev_response
        mock_get.return_value = mock_resp

        events = _fetch_kev_events("CVE-2099-0001")
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_kev_case_insensitive_match(self, mock_get, kev_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = kev_response
        mock_get.return_value = mock_resp

        events = _fetch_kev_events("cve-2021-44228")
        assert len(events) == 2

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_kev_no_due_date(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cveID": CVE_ID,
                    "dateAdded": "2021-12-10",
                    "dueDate": "",
                    "shortDescription": "Test",
                }
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_kev_events(CVE_ID)
        assert len(events) == 1
        assert events[0]["event"] == "Added to CISA KEV"

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_kev_api_failure_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        events = _fetch_kev_events(CVE_ID)
        assert events == []


# ---------------------------------------------------------------------------
# Tests: _fetch_github_advisory_events
# ---------------------------------------------------------------------------


class TestFetchGitHubAdvisoryEvents:
    """Tests for GitHub advisory event fetcher."""

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_full_advisory(self, mock_get, github_advisory_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = github_advisory_response
        mock_get.return_value = mock_resp

        events = _fetch_github_advisory_events(CVE_ID)
        assert len(events) == 2  # published + updated
        assert events[0]["event"] == "GHSA published"
        assert events[0]["date"] == "2021-12-10"
        assert "GHSA-jfh8-c2jp-5v3q" in events[0]["detail"]
        assert "critical" in events[0]["detail"]
        assert events[1]["event"] == "GHSA updated"
        assert events[1]["date"] == "2024-01-15"

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_advisory_same_publish_update_date(self, mock_get):
        """No duplicate event when published_at == updated_at."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "ghsa_id": "GHSA-xxxx-yyyy-zzzz",
                "published_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z",
                "severity": "high",
            }
        ]
        mock_get.return_value = mock_resp

        events = _fetch_github_advisory_events(CVE_ID)
        assert len(events) == 1
        assert events[0]["event"] == "GHSA published"

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_no_advisories(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        events = _fetch_github_advisory_events(CVE_ID)
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_advisory_not_a_list(self, mock_get):
        """Handle unexpected response format gracefully."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": "Not Found"}
        mock_get.return_value = mock_resp

        events = _fetch_github_advisory_events(CVE_ID)
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_advisory_api_failure_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("403 Forbidden")
        events = _fetch_github_advisory_events(CVE_ID)
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    def test_github_token_sent_in_header(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        _fetch_github_advisory_events(CVE_ID)
        call_kwargs = mock_get.call_args[1]
        assert "Bearer ghp_test123" in call_kwargs["headers"]["Authorization"]

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    @patch.dict("os.environ", {}, clear=True)
    def test_no_github_token(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        _fetch_github_advisory_events(CVE_ID)
        call_kwargs = mock_get.call_args[1]
        assert "Authorization" not in call_kwargs["headers"]


# ---------------------------------------------------------------------------
# Tests: _fetch_vulncheck_kev_events
# ---------------------------------------------------------------------------


class TestFetchVulnCheckKevEvents:
    """Tests for VulnCheck KEV event fetcher."""

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    @patch.dict("os.environ", {"VULNCHECK_API_KEY": "vc-test-key"})
    def test_vulncheck_with_data(self, mock_get, vulncheck_kev_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = vulncheck_kev_response
        mock_get.return_value = mock_resp

        events = _fetch_vulncheck_kev_events(CVE_ID)
        assert len(events) == 1
        assert events[0]["event"] == "VulnCheck KEV entry"
        assert events[0]["date"] == "2021-12-11"
        assert events[0]["source"] == "VulnCheck KEV"

    @patch.dict("os.environ", {}, clear=True)
    def test_vulncheck_no_api_key_returns_empty(self):
        events = _fetch_vulncheck_kev_events(CVE_ID)
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    @patch.dict("os.environ", {"VULNCHECK_API_KEY": "vc-test-key"})
    def test_vulncheck_no_data(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        events = _fetch_vulncheck_kev_events(CVE_ID)
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    @patch.dict("os.environ", {"VULNCHECK_API_KEY": "vc-test-key"})
    def test_vulncheck_api_failure_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("Timeout")
        events = _fetch_vulncheck_kev_events(CVE_ID)
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    @patch.dict("os.environ", {"VULNCHECK_API_KEY": "vc-test-key"})
    def test_vulncheck_date_without_time_suffix(self, mock_get):
        """date_added without T is handled."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"cve": CVE_ID, "date_added": "2022-03-15"}]}
        mock_get.return_value = mock_resp

        events = _fetch_vulncheck_kev_events(CVE_ID)
        assert len(events) == 1
        assert events[0]["date"] == "2022-03-15"
        assert "T00:00:00Z" in events[0]["timestamp"]


# ---------------------------------------------------------------------------
# Tests: build_timeline (integration)
# ---------------------------------------------------------------------------


class TestBuildTimeline:
    """Tests for the main timeline assembly function."""

    @patch("manus_agent.tools.get_cve_timeline._fetch_vulncheck_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_full_timeline_assembly(self, mock_nvd, mock_epss, mock_kev, mock_gh, mock_vc):
        mock_nvd.return_value = [
            {
                "date": "2021-12-10",
                "timestamp": "2021-12-10T10:15:00Z",
                "source": "NVD",
                "event": "CVE published",
                "detail": "test",
            }
        ]
        mock_epss.return_value = [
            {
                "date": "2022-01-01",
                "timestamp": "2022-01-01T00:00:00Z",
                "source": "EPSS",
                "event": "EPSS tracking started",
                "detail": "test",
            }
        ]
        mock_kev.return_value = [
            {
                "date": "2021-12-10",
                "timestamp": "2021-12-10T00:00:00Z",
                "source": "CISA KEV",
                "event": "Added to CISA KEV",
                "detail": "test",
            }
        ]
        mock_gh.return_value = [
            {
                "date": "2021-12-10",
                "timestamp": "2021-12-10T00:00:00Z",
                "source": "GitHub Advisory",
                "event": "GHSA published",
                "detail": "test",
            }
        ]
        mock_vc.return_value = []

        result = build_timeline(CVE_ID)
        assert result["cve_id"] == CVE_ID
        assert result["event_count"] == 4
        assert len(result["events"]) == 4
        assert "NVD" in result["sources_with_data"]
        assert "EPSS" in result["sources_with_data"]
        assert "CISA KEV" in result["sources_with_data"]
        assert "GitHub Advisory" in result["sources_with_data"]
        assert "VulnCheck KEV" not in result["sources_with_data"]
        assert result["sources_queried"] == ["NVD", "EPSS", "CISA KEV", "GitHub Advisory", "VulnCheck KEV"]

    @patch("manus_agent.tools.get_cve_timeline._fetch_vulncheck_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_events_sorted_chronologically(self, mock_nvd, mock_epss, mock_kev, mock_gh, mock_vc):
        mock_nvd.return_value = [
            {
                "date": "2022-03-01",
                "timestamp": "2022-03-01T00:00:00Z",
                "source": "NVD",
                "event": "CVE published",
                "detail": "",
            }
        ]
        mock_epss.return_value = [
            {
                "date": "2022-04-01",
                "timestamp": "2022-04-01T00:00:00Z",
                "source": "EPSS",
                "event": "EPSS tracking started",
                "detail": "",
            }
        ]
        mock_kev.return_value = [
            {
                "date": "2022-01-01",
                "timestamp": "2022-01-01T00:00:00Z",
                "source": "CISA KEV",
                "event": "Added to CISA KEV",
                "detail": "",
            }
        ]
        mock_gh.return_value = []
        mock_vc.return_value = []

        result = build_timeline(CVE_ID)
        dates = [e["date"] for e in result["events"]]
        assert dates == sorted(dates)

    @patch("manus_agent.tools.get_cve_timeline._fetch_vulncheck_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_same_date_sorted_by_source_priority(self, mock_nvd, mock_epss, mock_kev, mock_gh, mock_vc):
        """Events on the same date are ordered by source priority."""
        mock_nvd.return_value = [
            {
                "date": "2022-01-01",
                "timestamp": "2022-01-01T00:00:00Z",
                "source": "NVD",
                "event": "CVE published",
                "detail": "",
            }
        ]
        mock_epss.return_value = [
            {
                "date": "2022-01-01",
                "timestamp": "2022-01-01T00:00:00Z",
                "source": "EPSS",
                "event": "EPSS tracking started",
                "detail": "",
            }
        ]
        mock_kev.return_value = [
            {
                "date": "2022-01-01",
                "timestamp": "2022-01-01T00:00:00Z",
                "source": "CISA KEV",
                "event": "Added to CISA KEV",
                "detail": "",
            }
        ]
        mock_gh.return_value = []
        mock_vc.return_value = []

        result = build_timeline(CVE_ID)
        sources = [e["source"] for e in result["events"]]
        assert sources == ["NVD", "EPSS", "CISA KEV"]

    @patch("manus_agent.tools.get_cve_timeline._fetch_vulncheck_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_span_days_calculation(self, mock_nvd, mock_epss, mock_kev, mock_gh, mock_vc):
        mock_nvd.return_value = [
            {"date": "2022-01-01", "timestamp": "", "source": "NVD", "event": "CVE published", "detail": ""}
        ]
        mock_epss.return_value = [
            {"date": "2022-01-31", "timestamp": "", "source": "EPSS", "event": "EPSS tracking started", "detail": ""}
        ]
        mock_kev.return_value = []
        mock_gh.return_value = []
        mock_vc.return_value = []

        result = build_timeline(CVE_ID)
        assert result["span_days"] == 30

    @patch("manus_agent.tools.get_cve_timeline._fetch_vulncheck_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_no_events_returns_empty(self, mock_nvd, mock_epss, mock_kev, mock_gh, mock_vc):
        mock_nvd.return_value = []
        mock_epss.return_value = []
        mock_kev.return_value = []
        mock_gh.return_value = []
        mock_vc.return_value = []

        result = build_timeline(CVE_ID)
        assert result["events"] == []
        assert result["event_count"] == 0
        assert result["span_days"] is None
        assert result["sources_with_data"] == []

    @patch("manus_agent.tools.get_cve_timeline._fetch_vulncheck_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_cve_id_normalised_to_uppercase(self, mock_nvd, mock_epss, mock_kev, mock_gh, mock_vc):
        mock_nvd.return_value = []
        mock_epss.return_value = []
        mock_kev.return_value = []
        mock_gh.return_value = []
        mock_vc.return_value = []

        result = build_timeline("  cve-2021-44228  ")
        assert result["cve_id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.get_cve_timeline._fetch_vulncheck_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_single_event_span_is_none(self, mock_nvd, mock_epss, mock_kev, mock_gh, mock_vc):
        mock_nvd.return_value = [
            {"date": "2022-01-01", "timestamp": "", "source": "NVD", "event": "CVE published", "detail": ""}
        ]
        mock_epss.return_value = []
        mock_kev.return_value = []
        mock_gh.return_value = []
        mock_vc.return_value = []

        result = build_timeline(CVE_ID)
        assert result["span_days"] is None


# ---------------------------------------------------------------------------
# Tests: handler (Strands tool interface)
# ---------------------------------------------------------------------------


class TestHandler:
    """Tests for the Strands SDK tool handler."""

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_handler_success(self, mock_build):
        mock_build.return_value = {
            "cve_id": CVE_ID,
            "events": [{"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": "test"}],
            "sources_queried": ["NVD"],
            "sources_with_data": ["NVD"],
            "span_days": None,
            "event_count": 1,
        }

        tool_use = {"toolUseId": "test-123", "input": {"cve_id": CVE_ID}}
        result = handler(tool_use)
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"
        content_text = result["content"][0]["text"]
        parsed = json.loads(content_text)
        assert parsed["cve_id"] == CVE_ID
        assert len(parsed["events"]) == 1

    def test_handler_missing_cve_id(self):
        tool_use = {"toolUseId": "test-456", "input": {"cve_id": ""}}
        result = handler(tool_use)
        assert result["status"] == "error"
        assert "required" in result["content"][0]["text"]

    def test_handler_no_cve_id_key(self):
        tool_use = {"toolUseId": "test-789", "input": {}}
        result = handler(tool_use)
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_handler_exception(self, mock_build):
        mock_build.side_effect = RuntimeError("Something broke")
        tool_use = {"toolUseId": "test-err", "input": {"cve_id": CVE_ID}}
        result = handler(tool_use)
        assert result["status"] == "error"
        assert "Something broke" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Tests: CLI subcommand (_run_cve_timeline)
# ---------------------------------------------------------------------------


class TestCLISubcommand:
    """Tests for the cve-timeline CLI subcommand."""

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_cli_text_output(self, mock_build, capsys):
        mock_build.return_value = {
            "cve_id": CVE_ID,
            "events": [
                {"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": "Record created"},
                {"date": "2022-01-01", "source": "EPSS", "event": "EPSS tracking started", "detail": "Score: 0.85"},
            ],
            "sources_queried": ["NVD", "EPSS", "CISA KEV", "GitHub Advisory", "VulnCheck KEV"],
            "sources_with_data": ["NVD", "EPSS"],
            "span_days": 22,
            "event_count": 2,
        }

        from manus_agent.cli import _run_cve_timeline

        exit_code = _run_cve_timeline([CVE_ID])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert CVE_ID in captured.out
        assert "NVD" in captured.out
        assert "EPSS" in captured.out
        assert "22 days" in captured.out

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_cli_json_output(self, mock_build, capsys):
        mock_build.return_value = {
            "cve_id": CVE_ID,
            "events": [
                {"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": "test"},
            ],
            "sources_queried": ["NVD"],
            "sources_with_data": ["NVD"],
            "span_days": None,
            "event_count": 1,
        }

        from manus_agent.cli import _run_cve_timeline

        exit_code = _run_cve_timeline([CVE_ID, "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["cve_id"] == CVE_ID
        assert len(parsed["events"]) == 1

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_cli_no_events_returns_error(self, mock_build, capsys):
        mock_build.return_value = {
            "cve_id": CVE_ID,
            "events": [],
            "sources_queried": ["NVD"],
            "sources_with_data": [],
            "span_days": None,
            "event_count": 0,
        }

        from manus_agent.cli import _run_cve_timeline

        exit_code = _run_cve_timeline([CVE_ID])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "No timeline events found" in captured.err

    def test_cli_missing_cve_id_exits(self):
        from manus_agent.cli import _run_cve_timeline

        with pytest.raises(SystemExit) as exc_info:
            _run_cve_timeline([])
        assert exc_info.value.code == 2  # argparse error

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_cli_exception_returns_error(self, mock_build, capsys):
        mock_build.side_effect = RuntimeError("Connection failed")

        from manus_agent.cli import _run_cve_timeline

        exit_code = _run_cve_timeline([CVE_ID])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Timeline assembly failed" in captured.err

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_cli_no_span_days(self, mock_build, capsys):
        """When span_days is None, no span line is printed."""
        mock_build.return_value = {
            "cve_id": CVE_ID,
            "events": [
                {"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": "test"},
            ],
            "sources_queried": ["NVD"],
            "sources_with_data": ["NVD"],
            "span_days": None,
            "event_count": 1,
        }

        from manus_agent.cli import _run_cve_timeline

        exit_code = _run_cve_timeline([CVE_ID])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Timeline span" not in captured.out


# ---------------------------------------------------------------------------
# Tests: TOOL_SPEC structure
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Tests for the TOOL_SPEC definition."""

    def test_tool_spec_name(self):
        from manus_agent.tools.get_cve_timeline import TOOL_SPEC

        assert TOOL_SPEC["name"] == "get_cve_timeline"

    def test_tool_spec_has_description(self):
        from manus_agent.tools.get_cve_timeline import TOOL_SPEC

        assert len(TOOL_SPEC["description"]) > 50

    def test_tool_spec_input_schema(self):
        from manus_agent.tools.get_cve_timeline import TOOL_SPEC

        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "cve_id" in schema["properties"]
        assert schema["required"] == ["cve_id"]

    def test_tool_spec_cve_id_is_string(self):
        from manus_agent.tools.get_cve_timeline import TOOL_SPEC

        assert TOOL_SPEC["inputSchema"]["json"]["properties"]["cve_id"]["type"] == "string"


# ---------------------------------------------------------------------------
# Tests: Edge cases and integration details
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge case tests."""

    @patch("manus_agent.tools.get_cve_timeline._fetch_vulncheck_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_partial_source_failure_still_returns_data(self, mock_nvd, mock_epss, mock_kev, mock_gh, mock_vc):
        """Even if some sources fail, we still get data from working ones."""
        mock_nvd.return_value = [
            {"date": "2022-01-01", "timestamp": "", "source": "NVD", "event": "CVE published", "detail": ""}
        ]
        mock_epss.return_value = []  # failed or no data
        mock_kev.return_value = []  # not in KEV
        mock_gh.return_value = []  # no advisory
        mock_vc.return_value = []  # no key

        result = build_timeline(CVE_ID)
        assert result["event_count"] == 1
        assert result["sources_with_data"] == ["NVD"]

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_nvd_cvss_v30_when_v31_missing(self, mock_get):
        """Uses CVSS 3.0 when 3.1 is unavailable."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": CVE_ID,
                        "published": "2020-01-01T00:00:00.000",
                        "lastModified": "2020-01-01T00:00:00.000",
                        "metrics": {
                            "cvssMetricV30": [
                                {
                                    "cvssData": {
                                        "baseScore": 9.8,
                                        "baseSeverity": "CRITICAL",
                                    }
                                }
                            ]
                        },
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_nvd_events(CVE_ID)
        cvss_events = [e for e in events if "CVSS" in e["event"]]
        assert len(cvss_events) == 1
        assert "3.0" in cvss_events[0]["event"]

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_kev_long_description_truncated_in_detail(self, mock_get):
        """Long shortDescription is truncated."""
        long_desc = "A" * 200
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cveID": CVE_ID,
                    "dateAdded": "2022-01-01",
                    "dueDate": "",
                    "shortDescription": long_desc,
                }
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_kev_events(CVE_ID)
        assert len(events) == 1
        # shortDescription[:120] is used in detail
        assert len(events[0]["detail"]) < 200

    @patch("manus_agent.tools.get_cve_timeline._fetch_vulncheck_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_events_contain_required_keys(self, mock_nvd, mock_epss, mock_kev, mock_gh, mock_vc):
        mock_nvd.return_value = [
            {
                "date": "2022-01-01",
                "timestamp": "2022-01-01T00:00:00Z",
                "source": "NVD",
                "event": "CVE published",
                "detail": "test detail",
            }
        ]
        mock_epss.return_value = []
        mock_kev.return_value = []
        mock_gh.return_value = []
        mock_vc.return_value = []

        result = build_timeline(CVE_ID)
        for event in result["events"]:
            assert "date" in event
            assert "source" in event
            assert "event" in event
            assert "detail" in event
            assert "timestamp" in event
