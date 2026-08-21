"""
Comprehensive test suite for the cve_timeline tool.

All tests are fully mocked — no real HTTP calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.cve_timeline import (
    TOOL_SPEC,
    _days_between,
    _extract_date_from_url,
    _fetch_epss_history,
    _fetch_kev_date,
    _fetch_nvd_dates,
    _parse_date,
    build_timeline,
    format_timeline_json,
    format_timeline_text,
    handler,
)

ToolUse = dict  # type alias for test typing

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nvd_response_log4j():
    """NVD API response fixture for CVE-2021-44228."""
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "id": "CVE-2021-44228",
                    "published": "2021-12-10T10:15:00.000",
                    "lastModified": "2023-11-07T03:39:00.000",
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
                    "references": [
                        {
                            "url": "https://github.com/apache/logging-log4j2/commit/abc123",
                            "tags": ["Patch"],
                        },
                        {
                            "url": "https://logging.apache.org/log4j/2.x/security.html",
                            "tags": ["Vendor Advisory"],
                        },
                        {
                            "url": "https://www.kb.cert.org/vuls/id/930724",
                            "tags": ["Third Party Advisory"],
                        },
                    ],
                }
            }
        ]
    }


@pytest.fixture
def epss_response():
    """EPSS time-series response fixture."""
    return {
        "data": [
            {
                "cve": "CVE-2021-44228",
                "time-series": [
                    {"date": "2021-12-10", "epss": "0.0100", "percentile": "0.50"},
                    {"date": "2021-12-11", "epss": "0.0500", "percentile": "0.70"},
                    {"date": "2021-12-12", "epss": "0.2000", "percentile": "0.90"},
                    {"date": "2021-12-13", "epss": "0.5000", "percentile": "0.95"},
                    {"date": "2021-12-14", "epss": "0.9700", "percentile": "0.99"},
                ],
            }
        ]
    }


@pytest.fixture
def kev_response():
    """CISA KEV catalog response fixture."""
    return {
        "vulnerabilities": [
            {
                "cveID": "CVE-2021-44228",
                "dateAdded": "2021-12-10",
                "dueDate": "2021-12-24",
                "knownRansomwareCampaignUse": "Known",
                "vendorProject": "Apache",
                "product": "Log4j",
                "shortDescription": "Apache Log4j2 RCE",
            },
            {
                "cveID": "CVE-2022-99999",
                "dateAdded": "2022-05-01",
                "dueDate": "2022-05-15",
                "knownRansomwareCampaignUse": "Unknown",
                "vendorProject": "Example",
                "product": "Widget",
                "shortDescription": "Example vuln",
            },
        ]
    }


# ---------------------------------------------------------------------------
# TOOL_SPEC contract tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Verify TOOL_SPEC follows Strands conventions."""

    def test_has_name(self):
        assert TOOL_SPEC["name"] == "cve_timeline"

    def test_has_description(self):
        assert isinstance(TOOL_SPEC["description"], str)
        assert len(TOOL_SPEC["description"]) > 20

    def test_has_input_schema(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_cve_id_property(self):
        prop = TOOL_SPEC["inputSchema"]["json"]["properties"]["cve_id"]
        assert prop["type"] == "string"
        assert "description" in prop


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test input validation in build_timeline."""

    def test_invalid_cve_format_no_prefix(self):
        result = build_timeline("2021-44228")
        assert "error" in result
        assert "Invalid CVE ID" in result["error"]

    def test_invalid_cve_format_wrong_prefix(self):
        result = build_timeline("VUL-2021-44228")
        assert "error" in result

    def test_invalid_cve_format_short_number(self):
        result = build_timeline("CVE-2021-12")
        assert "error" in result

    def test_empty_cve_id(self):
        result = build_timeline("")
        assert "error" in result

    def test_valid_cve_format_accepted(self):
        """Valid format should not return a format error (may fail on network)."""
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"vulnerabilities": []}
            mock_get.return_value = mock_resp
            result = build_timeline("CVE-2021-44228")
            # Should not have a format error
            assert "Invalid CVE ID" not in result.get("error", "")

    def test_case_insensitive(self):
        """CVE ID should be normalized to uppercase."""
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"vulnerabilities": []}
            mock_get.return_value = mock_resp
            result = build_timeline("cve-2021-44228")
            assert result["cve_id"] == "CVE-2021-44228"

    def test_whitespace_stripped(self):
        """CVE ID with whitespace should be cleaned."""
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"vulnerabilities": []}
            mock_get.return_value = mock_resp
            result = build_timeline("  CVE-2021-44228  ")
            assert result["cve_id"] == "CVE-2021-44228"


# ---------------------------------------------------------------------------
# NVD fetch tests
# ---------------------------------------------------------------------------


class TestFetchNvdDates:
    """Test _fetch_nvd_dates function."""

    def test_successful_fetch(self, nvd_response_log4j):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = nvd_response_log4j
            mock_get.return_value = mock_resp

            result = _fetch_nvd_dates("CVE-2021-44228")
            assert result["published"] == "2021-12-10"
            assert result["last_modified"] == "2023-11-07"
            assert result["cvss_score"] == 10.0
            assert result["cvss_severity"] == "CRITICAL"

    def test_references_extracted(self, nvd_response_log4j):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = nvd_response_log4j
            mock_get.return_value = mock_resp

            result = _fetch_nvd_dates("CVE-2021-44228")
            assert len(result["references"]) == 3
            assert result["references"][0]["type"] == "patch"
            assert result["references"][1]["type"] == "advisory"

    def test_cve_not_found(self):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"vulnerabilities": []}
            mock_get.return_value = mock_resp

            result = _fetch_nvd_dates("CVE-9999-99999")
            assert "error" in result
            assert "not found" in result["error"]

    def test_network_error(self):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            import requests

            mock_get.side_effect = requests.exceptions.ConnectionError("timeout")
            result = _fetch_nvd_dates("CVE-2021-44228")
            assert "error" in result

    def test_nvd_api_key_used(self, nvd_response_log4j):
        with (
            patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get,
            patch.dict("os.environ", {"NVD_API_KEY": "test-key-123"}),
        ):
            mock_resp = MagicMock()
            mock_resp.json.return_value = nvd_response_log4j
            mock_get.return_value = mock_resp

            _fetch_nvd_dates("CVE-2021-44228")
            call_kwargs = mock_get.call_args[1]
            assert call_kwargs["headers"]["apiKey"] == "test-key-123"

    def test_cvss_v2_fallback(self):
        """Falls back to CVSS v2 when v3.1 is not available."""
        nvd_data = {
            "vulnerabilities": [
                {
                    "cve": {
                        "published": "2020-01-01T00:00:00.000",
                        "lastModified": "2020-01-02T00:00:00.000",
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
                        "references": [],
                    }
                }
            ]
        }
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = nvd_data
            mock_get.return_value = mock_resp

            result = _fetch_nvd_dates("CVE-2020-0001")
            assert result["cvss_score"] == 7.5


# ---------------------------------------------------------------------------
# EPSS fetch tests
# ---------------------------------------------------------------------------


class TestFetchEpssHistory:
    """Test _fetch_epss_history function."""

    def test_successful_fetch(self, epss_response):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = epss_response
            mock_get.return_value = mock_resp

            result = _fetch_epss_history("CVE-2021-44228")
            assert result["current_score"] == 0.97
            assert result["first_seen"] == "2021-12-10"
            assert result["last_seen"] == "2021-12-14"
            assert result["history_points"] == 5

    def test_spike_detection(self, epss_response):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = epss_response
            mock_get.return_value = mock_resp

            result = _fetch_epss_history("CVE-2021-44228")
            # 0.01→0.05 = +0.04 (no spike)
            # 0.05→0.20 = +0.15 (spike!)
            # 0.20→0.50 = +0.30 (spike!)
            # 0.50→0.97 = +0.47 (spike!)
            assert len(result["spikes"]) == 3
            assert result["spikes"][0]["date"] == "2021-12-12"
            assert result["spikes"][0]["jump"] == 0.15

    def test_no_data(self):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"data": []}
            mock_get.return_value = mock_resp

            result = _fetch_epss_history("CVE-9999-99999")
            assert "error" in result

    def test_single_point_response(self):
        """Handle case where API returns a single data point (no time-series)."""
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"data": [{"cve": "CVE-2024-1234", "epss": "0.05", "date": "2024-06-01"}]}
            mock_get.return_value = mock_resp

            result = _fetch_epss_history("CVE-2024-1234")
            assert result["current_score"] == 0.05
            assert result["first_seen"] == "2024-06-01"

    def test_network_error(self):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            import requests

            mock_get.side_effect = requests.exceptions.Timeout("timeout")
            result = _fetch_epss_history("CVE-2021-44228")
            assert "error" in result

    def test_no_spikes_in_flat_series(self):
        """No spikes when EPSS scores are stable."""
        flat_data = {
            "data": [
                {
                    "cve": "CVE-2024-0001",
                    "time-series": [
                        {"date": "2024-01-01", "epss": "0.01", "percentile": "0.30"},
                        {"date": "2024-01-02", "epss": "0.02", "percentile": "0.31"},
                        {"date": "2024-01-03", "epss": "0.02", "percentile": "0.31"},
                    ],
                }
            ]
        }
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = flat_data
            mock_get.return_value = mock_resp

            result = _fetch_epss_history("CVE-2024-0001")
            assert result["spikes"] == []


# ---------------------------------------------------------------------------
# KEV fetch tests
# ---------------------------------------------------------------------------


class TestFetchKevDate:
    """Test _fetch_kev_date function."""

    def test_cve_in_kev(self, kev_response):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = kev_response
            mock_get.return_value = mock_resp

            result = _fetch_kev_date("CVE-2021-44228")
            assert result["in_kev"] is True
            assert result["date_added"] == "2021-12-10"
            assert result["due_date"] == "2021-12-24"
            assert result["known_ransomware"] == "Known"
            assert result["vendor"] == "Apache"
            assert result["product"] == "Log4j"

    def test_cve_not_in_kev(self, kev_response):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = kev_response
            mock_get.return_value = mock_resp

            result = _fetch_kev_date("CVE-2024-0001")
            assert result["in_kev"] is False

    def test_case_insensitive_match(self, kev_response):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = kev_response
            mock_get.return_value = mock_resp

            result = _fetch_kev_date("cve-2021-44228")
            assert result["in_kev"] is True

    def test_network_error(self):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            import requests

            mock_get.side_effect = requests.exceptions.ConnectionError("fail")
            result = _fetch_kev_date("CVE-2021-44228")
            assert "error" in result


# ---------------------------------------------------------------------------
# HTTP retry tests
# ---------------------------------------------------------------------------


class TestRetry:
    """Test _get_with_retry retry logic."""

    def test_retry_on_429(self):
        from manus_agent.tools.cve_timeline import _get_with_retry

        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429
        mock_resp_success = MagicMock()
        mock_resp_success.status_code = 200
        mock_resp_success.raise_for_status = MagicMock()

        with patch("manus_agent.tools.cve_timeline.requests.get") as mock_get:
            with patch("manus_agent.tools.cve_timeline.time.sleep"):
                mock_get.side_effect = [mock_resp_429, mock_resp_success]
                result = _get_with_retry("https://example.com")
                assert result == mock_resp_success
                assert mock_get.call_count == 2

    def test_retry_exhausted(self):
        import requests

        from manus_agent.tools.cve_timeline import _get_with_retry

        mock_resp_500 = MagicMock()
        mock_resp_500.status_code = 500

        with patch("manus_agent.tools.cve_timeline.requests.get") as mock_get:
            with patch("manus_agent.tools.cve_timeline.time.sleep"):
                mock_get.return_value = mock_resp_500
                with pytest.raises(requests.exceptions.HTTPError):
                    _get_with_retry("https://example.com")
                assert mock_get.call_count == 4  # 1 + 3 retries

    def test_no_retry_on_404(self):
        import requests

        from manus_agent.tools.cve_timeline import _get_with_retry

        mock_resp_404 = MagicMock()
        mock_resp_404.status_code = 404
        mock_resp_404.raise_for_status.side_effect = requests.exceptions.HTTPError("404")

        with patch("manus_agent.tools.cve_timeline.requests.get") as mock_get:
            mock_get.return_value = mock_resp_404
            with pytest.raises(requests.exceptions.HTTPError):
                _get_with_retry("https://example.com")
            assert mock_get.call_count == 1

    def test_retry_on_connection_error(self):
        import requests

        from manus_agent.tools.cve_timeline import _get_with_retry

        mock_resp_success = MagicMock()
        mock_resp_success.status_code = 200
        mock_resp_success.raise_for_status = MagicMock()

        with patch("manus_agent.tools.cve_timeline.requests.get") as mock_get:
            with patch("manus_agent.tools.cve_timeline.time.sleep"):
                mock_get.side_effect = [
                    requests.exceptions.ConnectionError("fail"),
                    mock_resp_success,
                ]
                result = _get_with_retry("https://example.com")
                assert result == mock_resp_success


# ---------------------------------------------------------------------------
# Date utility tests
# ---------------------------------------------------------------------------


class TestDateUtils:
    """Test date parsing and calculation utilities."""

    def test_parse_date_valid(self):
        dt = _parse_date("2021-12-10")
        assert dt is not None
        assert dt.year == 2021
        assert dt.month == 12
        assert dt.day == 10

    def test_parse_date_none(self):
        assert _parse_date(None) is None

    def test_parse_date_invalid(self):
        assert _parse_date("not-a-date") is None

    def test_parse_date_empty(self):
        assert _parse_date("") is None

    def test_parse_date_truncates(self):
        """Handles full ISO timestamps by taking first 10 chars."""
        dt = _parse_date("2021-12-10T10:15:00.000")
        assert dt is not None
        assert dt.day == 10

    def test_days_between_same_day(self):
        assert _days_between("2021-12-10", "2021-12-10") == 0

    def test_days_between_normal(self):
        assert _days_between("2021-12-10", "2021-12-24") == 14

    def test_days_between_reversed(self):
        """Order doesn't matter — returns absolute difference."""
        assert _days_between("2021-12-24", "2021-12-10") == 14

    def test_days_between_none_input(self):
        assert _days_between(None, "2021-12-10") is None
        assert _days_between("2021-12-10", None) is None

    def test_days_between_invalid(self):
        assert _days_between("invalid", "2021-12-10") is None


# ---------------------------------------------------------------------------
# URL date extraction tests
# ---------------------------------------------------------------------------


class TestExtractDateFromUrl:
    """Test _extract_date_from_url function."""

    def test_iso_date_in_url(self):
        url = "https://advisory.example.com/2021-12-10/fix"
        assert _extract_date_from_url(url) == "2021-12-10"

    def test_slash_date_in_url(self):
        url = "https://blog.example.com/2021/12/10/security-advisory"
        assert _extract_date_from_url(url) == "2021-12-10"

    def test_no_date_in_url(self):
        url = "https://github.com/apache/logging-log4j2/commit/abc123"
        assert _extract_date_from_url(url) is None

    def test_invalid_month_rejected(self):
        url = "https://example.com/2021-13-10/fix"
        assert _extract_date_from_url(url) is None

    def test_invalid_day_rejected(self):
        url = "https://example.com/2021-12-32/fix"
        assert _extract_date_from_url(url) is None


# ---------------------------------------------------------------------------
# Timeline assembly tests
# ---------------------------------------------------------------------------


class TestBuildTimeline:
    """Test the full build_timeline assembly function."""

    def test_full_timeline(self, nvd_response_log4j, epss_response, kev_response):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            # NVD → EPSS → KEV (three calls)
            mock_nvd = MagicMock()
            mock_nvd.json.return_value = nvd_response_log4j
            mock_epss = MagicMock()
            mock_epss.json.return_value = epss_response
            mock_kev = MagicMock()
            mock_kev.json.return_value = kev_response
            mock_get.side_effect = [mock_nvd, mock_epss, mock_kev]

            result = build_timeline("CVE-2021-44228")

            assert result["cve_id"] == "CVE-2021-44228"
            assert result["event_count"] > 0
            assert "events" in result
            assert "summary" in result
            assert "sources" in result

    def test_events_sorted_chronologically(self, nvd_response_log4j, epss_response, kev_response):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_nvd = MagicMock()
            mock_nvd.json.return_value = nvd_response_log4j
            mock_epss = MagicMock()
            mock_epss.json.return_value = epss_response
            mock_kev = MagicMock()
            mock_kev.json.return_value = kev_response
            mock_get.side_effect = [mock_nvd, mock_epss, mock_kev]

            result = build_timeline("CVE-2021-44228")
            dates = [e["date"] for e in result["events"]]
            assert dates == sorted(dates)

    def test_summary_includes_time_deltas(self, nvd_response_log4j, epss_response, kev_response):
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_nvd = MagicMock()
            mock_nvd.json.return_value = nvd_response_log4j
            mock_epss = MagicMock()
            mock_epss.json.return_value = epss_response
            mock_kev = MagicMock()
            mock_kev.json.return_value = kev_response
            mock_get.side_effect = [mock_nvd, mock_epss, mock_kev]

            result = build_timeline("CVE-2021-44228")
            summary = result["summary"]
            # Published 2021-12-10, KEV added 2021-12-10 → 0 days
            assert summary["days_publish_to_exploit"] == 0

    def test_graceful_degradation_nvd_fails(self, epss_response, kev_response):
        """Timeline still works if NVD fails."""
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            import requests

            mock_epss = MagicMock()
            mock_epss.json.return_value = epss_response
            mock_kev = MagicMock()
            mock_kev.json.return_value = kev_response
            mock_get.side_effect = [
                requests.exceptions.Timeout("nvd timeout"),
                mock_epss,
                mock_kev,
            ]

            result = build_timeline("CVE-2021-44228")
            assert "error" not in result
            assert result["sources"]["nvd"].startswith("error:")
            assert result["sources"]["epss"] == "ok"
            assert result["sources"]["kev"] == "ok"

    def test_graceful_degradation_epss_fails(self, nvd_response_log4j, kev_response):
        """Timeline still works if EPSS fails."""
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            import requests

            mock_nvd = MagicMock()
            mock_nvd.json.return_value = nvd_response_log4j
            mock_kev = MagicMock()
            mock_kev.json.return_value = kev_response
            mock_get.side_effect = [
                mock_nvd,
                requests.exceptions.ConnectionError("epss fail"),
                mock_kev,
            ]

            result = build_timeline("CVE-2021-44228")
            assert "error" not in result
            assert result["sources"]["nvd"] == "ok"
            assert result["sources"]["epss"].startswith("error:")

    def test_graceful_degradation_kev_fails(self, nvd_response_log4j, epss_response):
        """Timeline still works if KEV fails."""
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            import requests

            mock_nvd = MagicMock()
            mock_nvd.json.return_value = nvd_response_log4j
            mock_epss = MagicMock()
            mock_epss.json.return_value = epss_response
            mock_get.side_effect = [
                mock_nvd,
                mock_epss,
                requests.exceptions.Timeout("kev timeout"),
            ]

            result = build_timeline("CVE-2021-44228")
            assert "error" not in result
            assert result["sources"]["kev"].startswith("error:")

    def test_all_sources_fail(self):
        """Timeline returns partial result even when all sources fail."""
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            import requests

            mock_get.side_effect = requests.exceptions.Timeout("all fail")

            result = build_timeline("CVE-2021-44228")
            assert "error" not in result  # Not a fatal error, just empty
            assert result["event_count"] == 0
            assert all(v.startswith("error:") for v in result["sources"].values())

    def test_kev_not_present(self, nvd_response_log4j, epss_response):
        """CVE not in KEV should reflect in summary."""
        kev_data = {"vulnerabilities": []}
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_nvd = MagicMock()
            mock_nvd.json.return_value = nvd_response_log4j
            mock_epss = MagicMock()
            mock_epss.json.return_value = epss_response
            mock_kev = MagicMock()
            mock_kev.json.return_value = kev_data
            mock_get.side_effect = [mock_nvd, mock_epss, mock_kev]

            result = build_timeline("CVE-2021-44228")
            assert result["summary"]["in_kev"] is False


# ---------------------------------------------------------------------------
# Text formatting tests
# ---------------------------------------------------------------------------


class TestFormatTimelineText:
    """Test text output formatting."""

    def test_basic_formatting(self):
        result = {
            "cve_id": "CVE-2021-44228",
            "events": [
                {
                    "date": "2021-12-10",
                    "event_type": "nvd_published",
                    "description": "CVE published in NVD",
                },
            ],
            "summary": {
                "cve_id": "CVE-2021-44228",
                "cvss_score": 10.0,
                "cvss_severity": "CRITICAL",
                "in_kev": True,
                "epss_current": 0.97,
            },
            "sources": {"nvd": "ok", "epss": "ok", "kev": "ok"},
            "event_count": 1,
        }
        text = format_timeline_text(result)
        assert "CVE-2021-44228" in text
        assert "10.0" in text
        assert "CRITICAL" in text
        assert "CISA KEV" in text
        assert "2021-12-10" in text

    def test_error_formatting(self):
        result = {"error": "Something went wrong"}
        text = format_timeline_text(result)
        assert "Error:" in text
        assert "Something went wrong" in text

    def test_no_events_formatting(self):
        result = {
            "cve_id": "CVE-2024-0001",
            "events": [],
            "summary": {"cve_id": "CVE-2024-0001", "in_kev": False},
            "sources": {"nvd": "error: not found"},
            "event_count": 0,
        }
        text = format_timeline_text(result)
        assert "No timeline events found" in text

    def test_key_intervals_shown(self):
        result = {
            "cve_id": "CVE-2021-44228",
            "events": [],
            "summary": {
                "cve_id": "CVE-2021-44228",
                "days_publish_to_exploit": 0,
                "days_publish_to_epss_spike": 2,
                "in_kev": True,
            },
            "sources": {},
            "event_count": 0,
        }
        text = format_timeline_text(result)
        assert "Key Intervals" in text
        assert "0 days" in text
        assert "2 days" in text

    def test_not_in_kev_text(self):
        result = {
            "cve_id": "CVE-2024-0001",
            "events": [],
            "summary": {"cve_id": "CVE-2024-0001", "in_kev": False},
            "sources": {},
            "event_count": 0,
        }
        text = format_timeline_text(result)
        assert "Not in CISA KEV" in text


# ---------------------------------------------------------------------------
# JSON formatting tests
# ---------------------------------------------------------------------------


class TestFormatTimelineJson:
    """Test JSON output formatting."""

    def test_valid_json(self):
        result = {
            "cve_id": "CVE-2021-44228",
            "events": [],
            "summary": {},
            "sources": {},
            "event_count": 0,
        }
        output = format_timeline_json(result)
        parsed = json.loads(output)
        assert parsed["cve_id"] == "CVE-2021-44228"

    def test_preserves_all_fields(self):
        result = {
            "cve_id": "CVE-2021-44228",
            "events": [{"date": "2021-12-10", "event_type": "nvd_published", "description": "x"}],
            "summary": {"cvss_score": 10.0},
            "sources": {"nvd": "ok"},
            "event_count": 1,
        }
        output = format_timeline_json(result)
        parsed = json.loads(output)
        assert parsed["events"][0]["event_type"] == "nvd_published"
        assert parsed["summary"]["cvss_score"] == 10.0


# ---------------------------------------------------------------------------
# Strands handler tests
# ---------------------------------------------------------------------------


class TestHandler:
    """Test the Strands tool handler."""

    def test_handler_success(self):
        with patch("manus_agent.tools.cve_timeline.build_timeline") as mock_build:
            mock_build.return_value = {
                "cve_id": "CVE-2021-44228",
                "events": [],
                "summary": {},
                "sources": {},
                "event_count": 0,
            }
            tool_use: ToolUse = {
                "toolUseId": "test-1",
                "name": "cve_timeline",
                "input": {"cve_id": "CVE-2021-44228"},
            }
            result = handler(tool_use)
            assert result["status"] == "success"
            assert len(result["content"]) == 1

    def test_handler_error_missing_cve_id(self):
        tool_use: ToolUse = {
            "toolUseId": "test-2",
            "name": "cve_timeline",
            "input": {},
        }
        result = handler(tool_use)
        assert result["status"] == "error"
        assert "required" in result["content"][0]["text"].lower()

    def test_handler_error_invalid_cve(self):
        tool_use: ToolUse = {
            "toolUseId": "test-3",
            "name": "cve_timeline",
            "input": {"cve_id": "INVALID"},
        }
        result = handler(tool_use)
        assert result["status"] == "error"

    def test_handler_returns_json(self):
        with patch("manus_agent.tools.cve_timeline.build_timeline") as mock_build:
            mock_build.return_value = {
                "cve_id": "CVE-2021-44228",
                "events": [
                    {
                        "date": "2021-12-10",
                        "event_type": "nvd_published",
                        "description": "test",
                    }
                ],
                "summary": {},
                "sources": {},
                "event_count": 1,
            }
            tool_use: ToolUse = {
                "toolUseId": "test-4",
                "name": "cve_timeline",
                "input": {"cve_id": "CVE-2021-44228"},
            }
            result = handler(tool_use)
            content_text = result["content"][0]["text"]
            # Should be valid JSON
            parsed = json.loads(content_text)
            assert parsed["event_count"] == 1


# ---------------------------------------------------------------------------
# CLI subcommand tests
# ---------------------------------------------------------------------------


class TestCliSubcommand:
    """Test CLI dispatch for cve-timeline."""

    def test_parser_creation(self):
        from manus_agent.cli import _build_cve_timeline_parser

        parser = _build_cve_timeline_parser()
        args = parser.parse_args(["CVE-2021-44228"])
        assert args.cve_id == "CVE-2021-44228"
        assert args.output == "text"

    def test_parser_json_output(self):
        from manus_agent.cli import _build_cve_timeline_parser

        parser = _build_cve_timeline_parser()
        args = parser.parse_args(["CVE-2021-44228", "--output", "json"])
        assert args.output == "json"

    def test_run_text_output(self, capsys):
        from manus_agent.cli import _run_cve_timeline

        with patch("manus_agent.tools.cve_timeline.build_timeline") as mock_build:
            mock_build.return_value = {
                "cve_id": "CVE-2021-44228",
                "events": [
                    {
                        "date": "2021-12-10",
                        "event_type": "nvd_published",
                        "description": "CVE published in NVD",
                    }
                ],
                "summary": {"cve_id": "CVE-2021-44228", "in_kev": False},
                "sources": {"nvd": "ok"},
                "event_count": 1,
            }
            exit_code = _run_cve_timeline(["CVE-2021-44228"])
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "CVE-2021-44228" in captured.out

    def test_run_json_output(self, capsys):
        from manus_agent.cli import _run_cve_timeline

        with patch("manus_agent.tools.cve_timeline.build_timeline") as mock_build:
            mock_build.return_value = {
                "cve_id": "CVE-2021-44228",
                "events": [],
                "summary": {},
                "sources": {},
                "event_count": 0,
            }
            exit_code = _run_cve_timeline(["CVE-2021-44228", "--output", "json"])
            assert exit_code == 0
            captured = capsys.readouterr()
            parsed = json.loads(captured.out)
            assert parsed["cve_id"] == "CVE-2021-44228"

    def test_run_error_exit_code(self, capsys):
        from manus_agent.cli import _run_cve_timeline

        with patch("manus_agent.tools.cve_timeline.build_timeline") as mock_build:
            mock_build.return_value = {"error": "Invalid CVE ID format: BAD"}
            exit_code = _run_cve_timeline(["BAD"])
            assert exit_code == 1

    def test_subcommand_registered(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "cve-timeline" in _SUBCOMMANDS

    def test_dispatch_in_main(self):
        """Verify cve-timeline is dispatched in main()."""
        import inspect

        import manus_agent.cli as cli_module

        source = inspect.getsource(cli_module.main)
        assert "cve-timeline" in source
        assert "_run_cve_timeline" in source


# ---------------------------------------------------------------------------
# Event icon tests
# ---------------------------------------------------------------------------


class TestEventIcons:
    """Test _event_icon function."""

    def test_known_icons(self):
        from manus_agent.tools.cve_timeline import _event_icon

        assert _event_icon("nvd_published") == "📋"
        assert _event_icon("kev_added") == "🚨"
        assert _event_icon("epss_spike") == "📈"
        assert _event_icon("patch_released") == "🩹"

    def test_unknown_icon_fallback(self):
        from manus_agent.tools.cve_timeline import _event_icon

        assert _event_icon("unknown_event") == "•"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_cve_with_five_digit_number(self):
        """CVE-2024-12345 format should be valid."""
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"vulnerabilities": []}
            mock_get.return_value = mock_resp
            result = build_timeline("CVE-2024-12345")
            assert result["cve_id"] == "CVE-2024-12345"

    def test_nvd_no_metrics(self):
        """NVD entry with no CVSS metrics should still work."""
        nvd_data = {
            "vulnerabilities": [
                {
                    "cve": {
                        "published": "2024-01-01T00:00:00.000",
                        "lastModified": "2024-01-01T00:00:00.000",
                        "metrics": {},
                        "references": [],
                    }
                }
            ]
        }
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = nvd_data
            mock_get.return_value = mock_resp
            result = _fetch_nvd_dates("CVE-2024-0001")
            assert "cvss_score" not in result
            assert result["published"] == "2024-01-01"

    def test_same_publish_and_modify_date(self):
        """Same publish/modify date shouldn't create duplicate events."""
        nvd_data = {
            "vulnerabilities": [
                {
                    "cve": {
                        "published": "2024-01-01T00:00:00.000",
                        "lastModified": "2024-01-01T00:00:00.000",
                        "metrics": {},
                        "references": [],
                    }
                }
            ]
        }
        epss_data = {"data": []}
        kev_data = {"vulnerabilities": []}

        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_nvd = MagicMock()
            mock_nvd.json.return_value = nvd_data
            mock_epss = MagicMock()
            mock_epss.json.return_value = epss_data
            mock_kev = MagicMock()
            mock_kev.json.return_value = kev_data
            mock_get.side_effect = [mock_nvd, mock_epss, mock_kev]

            result = build_timeline("CVE-2024-0001")
            # Should only have one event (published), not two
            nvd_events = [e for e in result["events"] if e["event_type"] in ("nvd_published", "nvd_modified")]
            assert len(nvd_events) == 1

    def test_epss_alternative_key_format(self):
        """Handle EPSS API using 'timeSeries' key instead of 'time-series'."""
        epss_data = {
            "data": [
                {
                    "cve": "CVE-2024-0001",
                    "timeSeries": [
                        {"date": "2024-01-01", "epss": "0.05", "percentile": "0.60"},
                        {"date": "2024-01-02", "epss": "0.06", "percentile": "0.61"},
                    ],
                }
            ]
        }
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = epss_data
            mock_get.return_value = mock_resp

            result = _fetch_epss_history("CVE-2024-0001")
            assert result["current_score"] == 0.06
            assert result["history_points"] == 2

    def test_large_cve_number(self):
        """CVE with very long number should be accepted."""
        with patch("manus_agent.tools.cve_timeline._get_with_retry") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"vulnerabilities": []}
            mock_get.return_value = mock_resp
            result = build_timeline("CVE-2024-1234567")
            assert result["cve_id"] == "CVE-2024-1234567"
