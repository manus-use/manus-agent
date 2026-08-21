"""Comprehensive test suite for get_exposure_window tool and CLI subcommand.

100% mocked — no real HTTP calls.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
import requests

from manus_agent.tools.get_exposure_window import (
    TOOL_SPEC,
    _compute_risk_label,
    _days_between,
    _extract_patch_date_from_refs,
    _format_json,
    _format_text,
    _parse_date,
    compute_exposure_window,
    get_exposure_window,
)

# ---------------------------------------------------------------------------
# TOOL_SPEC contract tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Verify TOOL_SPEC adheres to Strands module-based tool specification."""

    def test_has_name(self):
        assert TOOL_SPEC["name"] == "get_exposure_window"

    def test_has_description(self):
        assert "exposure window" in TOOL_SPEC["description"].lower()

    def test_has_input_schema(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "cve_id" in schema["properties"]
        assert schema["required"] == ["cve_id"]

    def test_cve_id_is_string(self):
        prop = TOOL_SPEC["inputSchema"]["json"]["properties"]["cve_id"]
        assert prop["type"] == "string"


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test input validation in compute_exposure_window."""

    def test_invalid_cve_format_no_prefix(self):
        result = compute_exposure_window("not-a-cve")
        assert "error" in result
        assert "Invalid CVE ID" in result["error"]

    def test_invalid_cve_format_missing_numbers(self):
        result = compute_exposure_window("CVE-2024")
        assert "error" in result

    def test_invalid_cve_format_empty(self):
        result = compute_exposure_window("")
        assert "error" in result

    def test_valid_format_accepted(self):
        with patch("manus_agent.tools.get_exposure_window._fetch_nvd_data", return_value=None):
            result = compute_exposure_window("CVE-2021-44228")
            assert "error" in result
            assert "not found" in result["error"]

    def test_case_insensitive(self):
        with patch("manus_agent.tools.get_exposure_window._fetch_nvd_data", return_value=None):
            result = compute_exposure_window("cve-2021-44228")
            assert "not found" in result["error"]

    def test_whitespace_stripped(self):
        with patch("manus_agent.tools.get_exposure_window._fetch_nvd_data", return_value=None):
            result = compute_exposure_window("  CVE-2021-44228  ")
            assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# Date parsing tests
# ---------------------------------------------------------------------------


class TestParseDateHelper:
    """Test _parse_date with various formats."""

    def test_iso_datetime_with_ms(self):
        assert _parse_date("2021-12-10T02:30:17.123") == date(2021, 12, 10)

    def test_iso_datetime_no_ms(self):
        assert _parse_date("2024-03-29T13:00:00") == date(2024, 3, 29)

    def test_iso_date_only(self):
        assert _parse_date("2023-06-15") == date(2023, 6, 15)

    def test_with_z_suffix(self):
        assert _parse_date("2022-01-01T00:00:00Z") == date(2022, 1, 1)

    def test_with_timezone_offset(self):
        assert _parse_date("2022-06-01T12:00:00+05:30") == date(2022, 6, 1)

    def test_none_returns_none(self):
        assert _parse_date(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_date("") is None

    def test_garbage_returns_none(self):
        assert _parse_date("not-a-date") is None

    def test_partial_date_returns_none(self):
        assert _parse_date("2024-13") is None


# ---------------------------------------------------------------------------
# days_between tests
# ---------------------------------------------------------------------------


class TestDaysBetween:
    """Test _days_between helper."""

    def test_same_day(self):
        d = date(2024, 1, 1)
        assert _days_between(d, d) == 0

    def test_positive_days(self):
        d1 = date(2024, 1, 1)
        d2 = date(2024, 1, 31)
        assert _days_between(d1, d2) == 30

    def test_negative_days(self):
        d1 = date(2024, 6, 1)
        d2 = date(2024, 1, 1)
        assert _days_between(d1, d2) == -152

    def test_none_first(self):
        assert _days_between(None, date(2024, 1, 1)) is None

    def test_none_second(self):
        assert _days_between(date(2024, 1, 1), None) is None

    def test_both_none(self):
        assert _days_between(None, None) is None


# ---------------------------------------------------------------------------
# Patch date extraction from NVD references
# ---------------------------------------------------------------------------


class TestExtractPatchDate:
    """Test _extract_patch_date_from_refs."""

    def test_no_references(self):
        record = {"references": []}
        assert _extract_patch_date_from_refs(record) is None

    def test_no_patch_tags(self):
        record = {
            "references": [
                {"url": "https://example.com", "tags": ["Third Party Advisory"]},
            ]
        }
        assert _extract_patch_date_from_refs(record) is None

    def test_patch_tag_returns_last_modified(self):
        record = {
            "references": [
                {"url": "https://github.com/fix/commit", "tags": ["Patch"]},
            ],
            "lastModified": "2024-01-15T10:00:00",
        }
        assert _extract_patch_date_from_refs(record) == "2024-01-15T10:00:00"

    def test_multiple_patch_refs_uses_last_modified(self):
        record = {
            "references": [
                {"url": "https://github.com/a", "tags": ["Patch"]},
                {"url": "https://github.com/b", "tags": ["Patch", "Vendor Advisory"]},
            ],
            "lastModified": "2024-02-20T08:00:00",
        }
        assert _extract_patch_date_from_refs(record) == "2024-02-20T08:00:00"

    def test_none_tags_field(self):
        record = {
            "references": [
                {"url": "https://example.com", "tags": None},
            ]
        }
        assert _extract_patch_date_from_refs(record) is None

    def test_empty_tags_field(self):
        record = {
            "references": [
                {"url": "https://example.com"},
            ]
        }
        assert _extract_patch_date_from_refs(record) is None


# ---------------------------------------------------------------------------
# Risk label computation tests
# ---------------------------------------------------------------------------


class TestComputeRiskLabel:
    """Test _compute_risk_label logic."""

    def test_kev_unpatched_is_critical(self):
        assert _compute_risk_label(10, 0.1, is_in_kev=True, is_patched=False) == "critical"

    def test_kev_patched_moderate_exposure(self):
        # KEV + patched with moderate days doesn't auto-upgrade
        label = _compute_risk_label(20, 0.05, is_in_kev=True, is_patched=True)
        assert label in ("moderate", "low")

    def test_unpatched_high_epss(self):
        assert _compute_risk_label(5, 0.6, is_in_kev=False, is_patched=False) == "critical"

    def test_unpatched_moderate_epss(self):
        assert _compute_risk_label(5, 0.35, is_in_kev=False, is_patched=False) == "high"

    def test_unpatched_low_epss_long_days(self):
        assert _compute_risk_label(10, 0.05, is_in_kev=False, is_patched=False) == "high"

    def test_unpatched_low_epss_short_days(self):
        assert _compute_risk_label(3, 0.05, is_in_kev=False, is_patched=False) == "moderate"

    def test_patched_critical_exposure(self):
        assert _compute_risk_label(120, 0.4, is_in_kev=False, is_patched=True) == "critical"

    def test_patched_high_days(self):
        assert _compute_risk_label(45, 0.15, is_in_kev=False, is_patched=True) == "high"

    def test_patched_moderate_days(self):
        assert _compute_risk_label(10, 0.12, is_in_kev=False, is_patched=True) == "moderate"

    def test_patched_low_everything(self):
        assert _compute_risk_label(3, 0.02, is_in_kev=False, is_patched=True) == "low"

    def test_none_exposure_days(self):
        label = _compute_risk_label(None, 0.01, is_in_kev=False, is_patched=True)
        assert label == "low"

    def test_none_epss(self):
        label = _compute_risk_label(5, None, is_in_kev=False, is_patched=True)
        assert label == "low"

    def test_both_none(self):
        label = _compute_risk_label(None, None, is_in_kev=False, is_patched=True)
        assert label == "low"


# ---------------------------------------------------------------------------
# NVD data fetcher tests
# ---------------------------------------------------------------------------


class TestFetchNvdData:
    """Test _fetch_nvd_data with mocked HTTP."""

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_success(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_nvd_data

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [{"cve": {"id": "CVE-2021-44228", "published": "2021-12-10"}}]
        }
        mock_get.return_value = mock_resp

        result = _fetch_nvd_data("CVE-2021-44228")
        assert result is not None
        assert result["id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_empty_vulnerabilities(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_nvd_data

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        assert _fetch_nvd_data("CVE-9999-0001") is None

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_network_error(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_nvd_data

        mock_get.side_effect = RuntimeError("All retries exhausted")
        assert _fetch_nvd_data("CVE-2021-44228") is None


# ---------------------------------------------------------------------------
# EPSS fetcher tests
# ---------------------------------------------------------------------------


class TestFetchEpss:
    """Test _fetch_epss with mocked HTTP."""

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_success(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_epss

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"epss": "0.97156"}]}
        mock_get.return_value = mock_resp

        result = _fetch_epss("CVE-2021-44228")
        assert result == pytest.approx(0.97156)

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_no_data(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_epss

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        assert _fetch_epss("CVE-9999-0001") is None

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_network_error(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_epss

        mock_get.side_effect = RuntimeError("timeout")
        assert _fetch_epss("CVE-2021-44228") is None


# ---------------------------------------------------------------------------
# KEV fetcher tests
# ---------------------------------------------------------------------------


class TestFetchKevDate:
    """Test _fetch_kev_date with mocked HTTP."""

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_found_in_kev(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_kev_date

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cveID": "CVE-2021-44228", "dateAdded": "2021-12-10"},
                {"cveID": "CVE-2024-3094", "dateAdded": "2024-03-29"},
            ]
        }
        mock_get.return_value = mock_resp

        assert _fetch_kev_date("CVE-2021-44228") == "2021-12-10"

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_not_in_kev(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_kev_date

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cveID": "CVE-2024-3094", "dateAdded": "2024-03-29"},
            ]
        }
        mock_get.return_value = mock_resp

        assert _fetch_kev_date("CVE-2021-00001") is None

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_network_error(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_kev_date

        mock_get.side_effect = RuntimeError("network")
        assert _fetch_kev_date("CVE-2021-44228") is None

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_case_insensitive_match(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_kev_date

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cveID": "CVE-2021-44228", "dateAdded": "2021-12-10"},
            ]
        }
        mock_get.return_value = mock_resp

        assert _fetch_kev_date("cve-2021-44228") == "2021-12-10"


# ---------------------------------------------------------------------------
# GHSA fetcher tests
# ---------------------------------------------------------------------------


class TestFetchGhsaPatchDate:
    """Test _fetch_ghsa_patch_date with mocked HTTP."""

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_single_advisory(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_ghsa_patch_date

        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"published_at": "2021-12-12T00:00:00Z", "updated_at": "2021-12-15T00:00:00Z"}]
        mock_get.return_value = mock_resp

        assert _fetch_ghsa_patch_date("CVE-2021-44228") == "2021-12-12T00:00:00Z"

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_multiple_advisories_picks_earliest(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_ghsa_patch_date

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"published_at": "2021-12-15T00:00:00Z"},
            {"published_at": "2021-12-12T00:00:00Z"},
            {"published_at": "2021-12-20T00:00:00Z"},
        ]
        mock_get.return_value = mock_resp

        assert _fetch_ghsa_patch_date("CVE-2021-44228") == "2021-12-12T00:00:00Z"

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_empty_list(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_ghsa_patch_date

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        assert _fetch_ghsa_patch_date("CVE-9999-0001") is None

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_not_a_list(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_ghsa_patch_date

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": "not found"}
        mock_get.return_value = mock_resp

        assert _fetch_ghsa_patch_date("CVE-9999-0001") is None

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_network_error(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_ghsa_patch_date

        mock_get.side_effect = RuntimeError("timeout")
        assert _fetch_ghsa_patch_date("CVE-2021-44228") is None

    @patch("manus_agent.tools.get_exposure_window._get_with_retry")
    def test_fallback_to_updated_at(self, mock_get):
        from manus_agent.tools.get_exposure_window import _fetch_ghsa_patch_date

        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"published_at": None, "updated_at": "2022-01-05T12:00:00Z"}]
        mock_get.return_value = mock_resp

        assert _fetch_ghsa_patch_date("CVE-2022-0001") == "2022-01-05T12:00:00Z"


# ---------------------------------------------------------------------------
# HTTP retry helper tests
# ---------------------------------------------------------------------------


class TestGetWithRetry:
    """Test _get_with_retry exponential back-off."""

    @patch("manus_agent.tools.get_exposure_window.time.sleep")
    @patch("manus_agent.tools.get_exposure_window.requests.get")
    def test_retries_on_429(self, mock_get, mock_sleep):
        from manus_agent.tools.get_exposure_window import _get_with_retry

        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.raise_for_status = MagicMock()
        mock_get.side_effect = [resp_429, resp_200]

        result = _get_with_retry("https://example.com")
        assert result == resp_200
        assert mock_sleep.call_count == 1

    @patch("manus_agent.tools.get_exposure_window.time.sleep")
    @patch("manus_agent.tools.get_exposure_window.requests.get")
    def test_retries_on_503(self, mock_get, mock_sleep):
        from manus_agent.tools.get_exposure_window import _get_with_retry

        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.raise_for_status = MagicMock()
        mock_get.side_effect = [resp_503, resp_503, resp_200]

        result = _get_with_retry("https://example.com")
        assert result == resp_200
        assert mock_sleep.call_count == 2

    @patch("manus_agent.tools.get_exposure_window.time.sleep")
    @patch("manus_agent.tools.get_exposure_window.requests.get")
    def test_all_retries_exhausted(self, mock_get, mock_sleep):
        from manus_agent.tools.get_exposure_window import _get_with_retry

        resp_500 = MagicMock()
        resp_500.status_code = 500
        mock_get.return_value = resp_500

        with pytest.raises((RuntimeError, requests.HTTPError)):
            _get_with_retry("https://example.com")

    @patch("manus_agent.tools.get_exposure_window.time.sleep")
    @patch("manus_agent.tools.get_exposure_window.requests.get")
    def test_retries_on_connection_error(self, mock_get, mock_sleep):
        import requests as _req

        from manus_agent.tools.get_exposure_window import _get_with_retry

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.raise_for_status = MagicMock()
        mock_get.side_effect = [_req.ConnectionError("fail"), resp_200]

        result = _get_with_retry("https://example.com")
        assert result == resp_200

    @patch("manus_agent.tools.get_exposure_window.time.sleep")
    @patch("manus_agent.tools.get_exposure_window.requests.get")
    def test_retries_on_timeout(self, mock_get, mock_sleep):
        import requests as _req

        from manus_agent.tools.get_exposure_window import _get_with_retry

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.raise_for_status = MagicMock()
        mock_get.side_effect = [_req.Timeout("timeout"), resp_200]

        result = _get_with_retry("https://example.com")
        assert result == resp_200


# ---------------------------------------------------------------------------
# Core compute_exposure_window integration tests
# ---------------------------------------------------------------------------


class TestComputeExposureWindow:
    """Test compute_exposure_window end-to-end with mocked fetchers."""

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_patched_cve(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = {
            "published": "2021-12-10T02:30:17.000",
            "lastModified": "2021-12-20T10:00:00.000",
            "references": [
                {"url": "https://github.com/fix", "tags": ["Patch"]},
            ],
            "descriptions": [{"value": "Log4Shell RCE via JNDI"}],
        }
        mock_epss.return_value = 0.975
        mock_kev.return_value = "2021-12-10"
        mock_ghsa.return_value = "2021-12-12T00:00:00Z"

        result = compute_exposure_window("CVE-2021-44228")

        assert result["cve_id"] == "CVE-2021-44228"
        assert result["status"] == "patched"
        assert result["disclosure_date"] == "2021-12-10"
        assert result["patch_date"] is not None
        assert result["exposure_days"] is not None
        assert result["exposure_days"] >= 0
        assert result["kev_date"] == "2021-12-10"
        assert result["current_epss"] == pytest.approx(0.975)
        assert result["risk_label"] in ("critical", "high", "moderate", "low")

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_unpatched_cve(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = {
            "published": "2024-06-01T00:00:00.000",
            "lastModified": "2024-06-05T00:00:00.000",
            "references": [
                {"url": "https://advisory.example.com", "tags": ["Third Party Advisory"]},
            ],
            "descriptions": [{"value": "Unpatched vuln"}],
        }
        mock_epss.return_value = 0.45
        mock_kev.return_value = None
        mock_ghsa.return_value = None

        result = compute_exposure_window("CVE-2024-12345")

        assert result["status"] == "unpatched"
        assert result["patch_date"] is None
        assert result["patch_source"] is None
        assert result["exposure_days"] is not None
        assert result["exposure_days"] > 0

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_nvd_patch_preferred_over_ghsa_when_earlier(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = {
            "published": "2023-01-01T00:00:00.000",
            "lastModified": "2023-01-05T00:00:00.000",
            "references": [{"url": "https://patch.com", "tags": ["Patch"]}],
            "descriptions": [{"value": "Test"}],
        }
        mock_epss.return_value = 0.1
        mock_kev.return_value = None
        mock_ghsa.return_value = "2023-01-10T00:00:00Z"

        result = compute_exposure_window("CVE-2023-00001")

        assert result["patch_source"] == "nvd_reference"
        assert result["patch_date"] == "2023-01-05"

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_ghsa_patch_preferred_when_earlier(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = {
            "published": "2023-01-01T00:00:00.000",
            "lastModified": "2023-01-20T00:00:00.000",
            "references": [{"url": "https://patch.com", "tags": ["Patch"]}],
            "descriptions": [{"value": "Test"}],
        }
        mock_epss.return_value = 0.1
        mock_kev.return_value = None
        mock_ghsa.return_value = "2023-01-08T00:00:00Z"

        result = compute_exposure_window("CVE-2023-00002")

        assert result["patch_source"] == "ghsa"
        assert result["patch_date"] == "2023-01-08"

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_kev_exposure_days_computed(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = {
            "published": "2024-01-01T00:00:00.000",
            "lastModified": "2024-01-15T00:00:00.000",
            "references": [{"url": "https://fix.com", "tags": ["Patch"]}],
            "descriptions": [{"value": "Test"}],
        }
        mock_epss.return_value = 0.8
        mock_kev.return_value = "2024-01-05"
        mock_ghsa.return_value = None

        result = compute_exposure_window("CVE-2024-00001")

        assert result["kev_date"] == "2024-01-05"
        assert result["kev_exposure_days"] == 10  # Jan 5 to Jan 15

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_no_epss_data(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = {
            "published": "2024-01-01T00:00:00.000",
            "lastModified": "2024-01-10T00:00:00.000",
            "references": [],
            "descriptions": [{"value": "Test"}],
        }
        mock_epss.return_value = None
        mock_kev.return_value = None
        mock_ghsa.return_value = None

        result = compute_exposure_window("CVE-2024-00002")

        assert result["current_epss"] is None
        assert result["risk_label"] in ("critical", "high", "moderate", "low")

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_nvd_not_found(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = None
        result = compute_exposure_window("CVE-9999-99999")
        assert "error" in result
        assert "not found" in result["error"]

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_description_included(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = {
            "published": "2024-01-01T00:00:00.000",
            "lastModified": "2024-01-01T00:00:00.000",
            "references": [],
            "descriptions": [{"value": "A critical vuln in libfoo"}],
        }
        mock_epss.return_value = 0.5
        mock_kev.return_value = None
        mock_ghsa.return_value = None

        result = compute_exposure_window("CVE-2024-00003")
        assert "libfoo" in result["description"]


# ---------------------------------------------------------------------------
# Text formatting tests
# ---------------------------------------------------------------------------


class TestFormatText:
    """Test _format_text output."""

    def test_patched_output(self):
        result = {
            "cve_id": "CVE-2021-44228",
            "status": "patched",
            "risk_label": "critical",
            "disclosure_date": "2021-12-10",
            "patch_date": "2021-12-20",
            "patch_source": "nvd_reference",
            "exposure_days": 10,
            "kev_date": "2021-12-10",
            "kev_exposure_days": 10,
            "current_epss": 0.975,
            "description": "Log4Shell",
        }
        text = _format_text(result)
        assert "CVE-2021-44228" in text
        assert "PATCHED" in text
        assert "CRITICAL" in text
        assert "10 days (closed)" in text
        assert "0.9750" in text

    def test_unpatched_output(self):
        result = {
            "cve_id": "CVE-2024-12345",
            "status": "unpatched",
            "risk_label": "high",
            "disclosure_date": "2024-06-01",
            "patch_date": None,
            "patch_source": None,
            "exposure_days": 100,
            "kev_date": None,
            "kev_exposure_days": None,
            "current_epss": 0.45,
            "description": "Test vuln",
        }
        text = _format_text(result)
        assert "UNPATCHED" in text
        assert "STILL OPEN" in text
        assert "No patch detected" in text

    def test_error_output(self):
        result = {"error": "Invalid CVE ID format: bad"}
        text = _format_text(result)
        assert "Error:" in text

    def test_no_kev_section(self):
        result = {
            "cve_id": "CVE-2024-00001",
            "status": "patched",
            "risk_label": "low",
            "disclosure_date": "2024-01-01",
            "patch_date": "2024-01-05",
            "patch_source": "ghsa",
            "exposure_days": 4,
            "kev_date": None,
            "kev_exposure_days": None,
            "current_epss": 0.01,
            "description": "Minor issue",
        }
        text = _format_text(result)
        assert "KEV" not in text

    def test_long_description_truncated(self):
        result = {
            "cve_id": "CVE-2024-00001",
            "status": "patched",
            "risk_label": "low",
            "disclosure_date": "2024-01-01",
            "patch_date": "2024-01-02",
            "patch_source": "nvd_reference",
            "exposure_days": 1,
            "kev_date": None,
            "kev_exposure_days": None,
            "current_epss": 0.01,
            "description": "A" * 300,
        }
        text = _format_text(result)
        assert "..." in text

    def test_none_exposure_days(self):
        result = {
            "cve_id": "CVE-2024-00001",
            "status": "unpatched",
            "risk_label": "moderate",
            "disclosure_date": None,
            "patch_date": None,
            "patch_source": None,
            "exposure_days": None,
            "kev_date": None,
            "kev_exposure_days": None,
            "current_epss": None,
            "description": "",
        }
        text = _format_text(result)
        assert "Unknown" in text


# ---------------------------------------------------------------------------
# JSON formatting tests
# ---------------------------------------------------------------------------


class TestFormatJson:
    """Test _format_json output."""

    def test_valid_json(self):
        result = {
            "cve_id": "CVE-2021-44228",
            "status": "patched",
            "exposure_days": 10,
        }
        output = _format_json(result)
        parsed = json.loads(output)
        assert parsed["cve_id"] == "CVE-2021-44228"
        assert parsed["exposure_days"] == 10

    def test_none_values_serialized(self):
        result = {
            "cve_id": "CVE-2024-00001",
            "patch_date": None,
            "current_epss": None,
        }
        output = _format_json(result)
        parsed = json.loads(output)
        assert parsed["patch_date"] is None


# ---------------------------------------------------------------------------
# Strands tool handler tests
# ---------------------------------------------------------------------------


class TestGetExposureWindowHandler:
    """Test get_exposure_window Strands handler."""

    @patch("manus_agent.tools.get_exposure_window.compute_exposure_window")
    def test_success(self, mock_compute):
        mock_compute.return_value = {
            "cve_id": "CVE-2021-44228",
            "status": "patched",
            "risk_label": "critical",
            "disclosure_date": "2021-12-10",
            "patch_date": "2021-12-20",
            "patch_source": "nvd_reference",
            "exposure_days": 10,
            "kev_date": "2021-12-10",
            "kev_exposure_days": 10,
            "current_epss": 0.975,
            "description": "Log4Shell",
        }

        tool_use = {"toolUseId": "test-123", "input": {"cve_id": "CVE-2021-44228"}}
        result = get_exposure_window(tool_use)

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"
        assert "CVE-2021-44228" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_exposure_window.compute_exposure_window")
    def test_error_result(self, mock_compute):
        mock_compute.return_value = {"error": "CVE not found in NVD: CVE-9999-0001"}

        tool_use = {"toolUseId": "test-456", "input": {"cve_id": "CVE-9999-0001"}}
        result = get_exposure_window(tool_use)

        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_missing_cve_id(self):
        tool_use = {"toolUseId": "test-789", "input": {}}
        result = get_exposure_window(tool_use)
        assert result["status"] == "error"
        assert "Missing" in result["content"][0]["text"]

    def test_empty_cve_id(self):
        tool_use = {"toolUseId": "test-000", "input": {"cve_id": ""}}
        result = get_exposure_window(tool_use)
        assert result["status"] == "error"

    def test_non_string_cve_id(self):
        tool_use = {"toolUseId": "test-111", "input": {"cve_id": 12345}}
        result = get_exposure_window(tool_use)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# CLI subcommand tests
# ---------------------------------------------------------------------------


class TestCliExposureWindow:
    """Test _run_exposure_window CLI dispatch."""

    @patch("manus_agent.tools.get_exposure_window.compute_exposure_window")
    def test_text_output(self, mock_compute, capsys):
        from manus_agent.cli import _run_exposure_window

        mock_compute.return_value = {
            "cve_id": "CVE-2021-44228",
            "status": "patched",
            "risk_label": "critical",
            "disclosure_date": "2021-12-10",
            "patch_date": "2021-12-20",
            "patch_source": "nvd_reference",
            "exposure_days": 10,
            "kev_date": "2021-12-10",
            "kev_exposure_days": 10,
            "current_epss": 0.975,
            "description": "Log4Shell",
        }

        exit_code = _run_exposure_window(["CVE-2021-44228"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "CVE-2021-44228" in captured.out

    @patch("manus_agent.tools.get_exposure_window.compute_exposure_window")
    def test_json_output(self, mock_compute, capsys):
        from manus_agent.cli import _run_exposure_window

        mock_compute.return_value = {
            "cve_id": "CVE-2021-44228",
            "status": "patched",
            "risk_label": "high",
            "disclosure_date": "2021-12-10",
            "patch_date": "2021-12-20",
            "patch_source": "nvd_reference",
            "exposure_days": 10,
            "kev_date": None,
            "kev_exposure_days": None,
            "current_epss": 0.5,
            "description": "Test",
        }

        exit_code = _run_exposure_window(["CVE-2021-44228", "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.split("Computing")[0] or captured.out.split("\n", 1)[-1])
        assert parsed["cve_id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.get_exposure_window.compute_exposure_window")
    def test_error_returns_1(self, mock_compute, capsys):
        from manus_agent.cli import _run_exposure_window

        mock_compute.return_value = {"error": "CVE not found in NVD: CVE-9999-0001"}

        exit_code = _run_exposure_window(["CVE-9999-0001"])
        assert exit_code == 1

    def test_help_flag(self):
        from manus_agent.cli import _run_exposure_window

        with pytest.raises(SystemExit) as exc_info:
            _run_exposure_window(["--help"])
        assert exc_info.value.code == 0

    def test_no_args_shows_error(self):
        from manus_agent.cli import _run_exposure_window

        with pytest.raises(SystemExit) as exc_info:
            _run_exposure_window([])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test various edge cases."""

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_no_descriptions(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = {
            "published": "2024-01-01T00:00:00.000",
            "lastModified": "2024-01-01T00:00:00.000",
            "references": [],
            "descriptions": [],
        }
        mock_epss.return_value = None
        mock_kev.return_value = None
        mock_ghsa.return_value = None

        result = compute_exposure_window("CVE-2024-00001")
        assert result["description"] == ""

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_kev_without_patch(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = {
            "published": "2024-01-01T00:00:00.000",
            "lastModified": "2024-01-05T00:00:00.000",
            "references": [],
            "descriptions": [{"value": "Actively exploited"}],
        }
        mock_epss.return_value = 0.9
        mock_kev.return_value = "2024-01-03"
        mock_ghsa.return_value = None

        result = compute_exposure_window("CVE-2024-00099")
        assert result["status"] == "unpatched"
        assert result["kev_date"] == "2024-01-03"
        assert result["kev_exposure_days"] is None
        assert result["risk_label"] == "critical"

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_same_day_patch(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = {
            "published": "2024-03-01T00:00:00.000",
            "lastModified": "2024-03-01T12:00:00.000",
            "references": [{"url": "https://fix", "tags": ["Patch"]}],
            "descriptions": [{"value": "Same-day fix"}],
        }
        mock_epss.return_value = 0.05
        mock_kev.return_value = None
        mock_ghsa.return_value = None

        result = compute_exposure_window("CVE-2024-00100")
        assert result["status"] == "patched"
        assert result["exposure_days"] == 0

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_nvd_record_missing_published(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = {
            "published": None,
            "lastModified": "2024-01-05T00:00:00.000",
            "references": [],
            "descriptions": [{"value": "No publish date"}],
        }
        mock_epss.return_value = 0.2
        mock_kev.return_value = None
        mock_ghsa.return_value = None

        result = compute_exposure_window("CVE-2024-00101")
        assert result["disclosure_date"] is None
        # exposure_days will be None when disclosure_date is None and unpatched
        # Actually our code tries (today - disclosure_date) when unpatched, but disclosure_date is None
        # so exposure_days remains None

    @patch("manus_agent.tools.get_exposure_window._fetch_ghsa_patch_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_kev_date")
    @patch("manus_agent.tools.get_exposure_window._fetch_epss")
    @patch("manus_agent.tools.get_exposure_window._fetch_nvd_data")
    def test_all_sources_fail_gracefully(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        """When NVD returns data but all enrichment sources fail, still produce output."""
        mock_nvd.return_value = {
            "published": "2024-01-01T00:00:00.000",
            "lastModified": "2024-01-01T00:00:00.000",
            "references": [],
            "descriptions": [{"value": "Minimal data"}],
        }
        mock_epss.return_value = None
        mock_kev.return_value = None
        mock_ghsa.return_value = None

        result = compute_exposure_window("CVE-2024-00102")
        assert "error" not in result
        assert result["status"] == "unpatched"
        assert result["current_epss"] is None
        assert result["kev_date"] is None


# ---------------------------------------------------------------------------
# NVD API key header tests
# ---------------------------------------------------------------------------


class TestNvdHeaders:
    """Test _build_nvd_headers API key injection."""

    @patch.dict("os.environ", {"NVD_API_KEY": "test-key-123"})
    def test_api_key_included(self):
        from manus_agent.tools.get_exposure_window import _build_nvd_headers

        headers = _build_nvd_headers()
        assert headers["apiKey"] == "test-key-123"

    @patch.dict("os.environ", {"NVD_API_KEY": ""})
    def test_empty_api_key_excluded(self):
        from manus_agent.tools.get_exposure_window import _build_nvd_headers

        headers = _build_nvd_headers()
        assert "apiKey" not in headers

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key(self):
        from manus_agent.tools.get_exposure_window import _build_nvd_headers

        headers = _build_nvd_headers()
        assert "apiKey" not in headers


# ---------------------------------------------------------------------------
# GitHub headers tests
# ---------------------------------------------------------------------------


class TestGithubHeaders:
    """Test _build_github_headers token injection."""

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    def test_token_included(self):
        from manus_agent.tools.get_exposure_window import _build_github_headers

        headers = _build_github_headers()
        assert headers["Authorization"] == "Bearer ghp_test123"

    @patch.dict("os.environ", {"GITHUB_TOKEN": ""})
    def test_empty_token_excluded(self):
        from manus_agent.tools.get_exposure_window import _build_github_headers

        headers = _build_github_headers()
        assert "Authorization" not in headers

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_token(self):
        from manus_agent.tools.get_exposure_window import _build_github_headers

        headers = _build_github_headers()
        assert "Authorization" not in headers
