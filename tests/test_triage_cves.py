"""
Comprehensive test suite for triage_cves tool.

100% mocked — no real HTTP calls.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.triage_cves import (
    TOOL_SPEC,
    _compute_triage_score,
    _fetch_epss_batch,
    _fetch_kev_catalogue,
    _fetch_nvd_cvss,
    _fetch_vulncheck_kev,
    _format_text,
    _get_with_retry,
    _severity_label,
    triage_cves,
    triage_cves_handler,
)

# ---------------------------------------------------------------------------
# TOOL_SPEC contract tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Verify TOOL_SPEC structure follows Strands conventions."""

    def test_has_name(self):
        assert TOOL_SPEC["name"] == "triage_cves"

    def test_has_description(self):
        assert "batch" in TOOL_SPEC["description"].lower()
        assert "triage" in TOOL_SPEC["description"].lower()

    def test_has_input_schema(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "cve_ids" in schema["properties"]
        assert "output_format" in schema["properties"]

    def test_cve_ids_is_required(self):
        assert "cve_ids" in TOOL_SPEC["inputSchema"]["json"]["required"]

    def test_cve_ids_is_array(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert props["cve_ids"]["type"] == "array"
        assert props["cve_ids"]["items"]["type"] == "string"

    def test_output_format_enum(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert set(props["output_format"]["enum"]) == {"text", "json"}


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test input validation logic."""

    def test_empty_cve_list(self):
        result = triage_cves([])
        assert "error" in result
        assert result["results"] == []

    def test_too_many_cves(self):
        cve_ids = [f"CVE-2024-{i:04d}" for i in range(51)]
        result = triage_cves(cve_ids)
        assert "error" in result
        assert "50" in result["error"]

    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    def test_all_invalid_cve_ids(self, mock_nvd, mock_vc, mock_kev, mock_epss):
        result = triage_cves(["not-a-cve", "also-bad", "123"])
        assert "error" in result
        assert "invalid" in result["error"].lower()

    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    def test_mixed_valid_and_invalid(self, mock_nvd, mock_vc, mock_kev, mock_epss):
        mock_epss.return_value = {"CVE-2024-1234": {"epss": 0.5, "percentile": 0.9}}
        mock_kev.return_value = set()
        mock_vc.return_value = set()
        mock_nvd.return_value = {"cvss_score": 7.5, "cvss_vector": "AV:N", "severity": "HIGH"}

        result = triage_cves(["CVE-2024-1234", "bad-id"])
        assert len(result["results"]) == 1
        assert result["summary"]["invalid_ids"] == ["bad-id"]

    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    def test_duplicate_cves_deduplicated(self, mock_nvd, mock_vc, mock_kev, mock_epss):
        mock_epss.return_value = {"CVE-2024-1234": {"epss": 0.3, "percentile": 0.7}}
        mock_kev.return_value = set()
        mock_vc.return_value = set()
        mock_nvd.return_value = {"cvss_score": 5.0, "cvss_vector": "AV:L", "severity": "MEDIUM"}

        result = triage_cves(["CVE-2024-1234", "CVE-2024-1234", "cve-2024-1234"])
        assert len(result["results"]) == 1

    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    def test_case_insensitive_normalization(self, mock_nvd, mock_vc, mock_kev, mock_epss):
        mock_epss.return_value = {"CVE-2024-5678": {"epss": 0.1, "percentile": 0.5}}
        mock_kev.return_value = set()
        mock_vc.return_value = set()
        mock_nvd.return_value = {"cvss_score": 4.0, "cvss_vector": "AV:L", "severity": "MEDIUM"}

        result = triage_cves(["cve-2024-5678"])
        assert result["results"][0]["cve_id"] == "CVE-2024-5678"


# ---------------------------------------------------------------------------
# Triage scoring tests
# ---------------------------------------------------------------------------


class TestTriageScoring:
    """Test the composite scoring logic."""

    def test_all_zeros(self):
        score = _compute_triage_score(cvss_score=None, epss_score=None, in_kev=False)
        assert score == 0.0

    def test_max_score(self):
        score = _compute_triage_score(cvss_score=10.0, epss_score=1.0, in_kev=True)
        assert score == 100.0

    def test_cvss_only(self):
        score = _compute_triage_score(cvss_score=10.0, epss_score=0.0, in_kev=False)
        # (10/10 * 0.30 + 0 * 0.40 + 0 * 0.30) * 100 = 30.0
        assert score == 30.0

    def test_epss_only(self):
        score = _compute_triage_score(cvss_score=0.0, epss_score=1.0, in_kev=False)
        # (0 * 0.30 + 1.0 * 0.40 + 0 * 0.30) * 100 = 40.0
        assert score == 40.0

    def test_kev_only(self):
        score = _compute_triage_score(cvss_score=0.0, epss_score=0.0, in_kev=True)
        # (0 * 0.30 + 0 * 0.40 + 1.0 * 0.30) * 100 = 30.0
        assert score == 30.0

    def test_typical_high_severity(self):
        # CVSS 9.8, EPSS 0.95, in KEV
        score = _compute_triage_score(cvss_score=9.8, epss_score=0.95, in_kev=True)
        expected = round((9.8 / 10 * 0.30 + 0.95 * 0.40 + 1.0 * 0.30) * 100, 1)
        assert score == expected

    def test_none_cvss_treated_as_zero(self):
        score = _compute_triage_score(cvss_score=None, epss_score=0.5, in_kev=False)
        expected = round((0.0 * 0.30 + 0.5 * 0.40 + 0.0 * 0.30) * 100, 1)
        assert score == expected

    def test_none_epss_treated_as_zero(self):
        score = _compute_triage_score(cvss_score=7.0, epss_score=None, in_kev=True)
        expected = round((7.0 / 10 * 0.30 + 0.0 * 0.40 + 1.0 * 0.30) * 100, 1)
        assert score == expected


# ---------------------------------------------------------------------------
# Severity label tests
# ---------------------------------------------------------------------------


class TestSeverityLabel:
    """Test urgency label mapping."""

    def test_critical(self):
        assert _severity_label(80.0) == "CRITICAL"
        assert _severity_label(100.0) == "CRITICAL"

    def test_high(self):
        assert _severity_label(60.0) == "HIGH"
        assert _severity_label(79.9) == "HIGH"

    def test_medium(self):
        assert _severity_label(40.0) == "MEDIUM"
        assert _severity_label(59.9) == "MEDIUM"

    def test_low(self):
        assert _severity_label(20.0) == "LOW"
        assert _severity_label(39.9) == "LOW"

    def test_info(self):
        assert _severity_label(0.0) == "INFO"
        assert _severity_label(19.9) == "INFO"


# ---------------------------------------------------------------------------
# HTTP retry tests
# ---------------------------------------------------------------------------


class TestGetWithRetry:
    """Test the retry/back-off HTTP helper."""

    @patch("manus_agent.tools.triage_cves.requests.get")
    def test_success_on_first_try(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves.requests.get")
    def test_retry_on_429(self, mock_get, mock_sleep):
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_get.side_effect = [mock_429, mock_200]

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves.requests.get")
    def test_retry_on_500(self, mock_get, mock_sleep):
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_get.side_effect = [mock_500, mock_200]

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves.requests.get")
    def test_max_retries_exhausted(self, mock_get, mock_sleep):
        mock_500 = MagicMock()
        mock_500.status_code = 500
        mock_get.return_value = mock_500

        result = _get_with_retry("https://example.com")
        assert result.status_code == 500
        assert mock_get.call_count == 3

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves.requests.get")
    def test_retry_on_request_exception(self, mock_get, mock_sleep):
        import requests as req

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_get.side_effect = [req.ConnectionError("timeout"), mock_200]

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves.requests.get")
    def test_all_retries_fail_with_exception(self, mock_get, mock_sleep):
        import requests as req

        mock_get.side_effect = req.ConnectionError("down")
        with pytest.raises(req.ConnectionError):
            _get_with_retry("https://example.com")


# ---------------------------------------------------------------------------
# NVD CVSS fetcher tests
# ---------------------------------------------------------------------------


class TestFetchNvdCvss:
    """Test NVD CVSS data fetching."""

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_cvss_v31_extraction(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "metrics": {
                            "cvssMetricV31": [
                                {
                                    "cvssData": {
                                        "baseScore": 9.8,
                                        "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                                        "baseSeverity": "CRITICAL",
                                    }
                                }
                            ]
                        }
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_nvd_cvss("CVE-2024-3094")
        assert result["cvss_score"] == 9.8
        assert result["severity"] == "CRITICAL"
        assert "CVSS:3.1" in result["cvss_vector"]

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_cvss_v30_fallback(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "metrics": {
                            "cvssMetricV30": [
                                {
                                    "cvssData": {
                                        "baseScore": 7.5,
                                        "vectorString": "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H",
                                        "baseSeverity": "HIGH",
                                    }
                                }
                            ]
                        }
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_nvd_cvss("CVE-2020-1234")
        assert result["cvss_score"] == 7.5
        assert result["severity"] == "HIGH"

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_cvss_v2_fallback(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "metrics": {
                            "cvssMetricV2": [
                                {
                                    "cvssData": {
                                        "baseScore": 7.5,
                                        "vectorString": "AV:N/AC:L/Au:N/C:P/I:P/A:P",
                                    }
                                }
                            ]
                        }
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_nvd_cvss("CVE-2015-1234")
        assert result["cvss_score"] == 7.5
        assert result["severity"] == "HIGH"

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_no_vulnerabilities(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        result = _fetch_nvd_cvss("CVE-9999-0001")
        assert result["cvss_score"] is None

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = _fetch_nvd_cvss("CVE-2024-0000")
        assert result["cvss_score"] is None
        assert result["severity"] is None

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_exception_graceful_degradation(self, mock_get):
        mock_get.side_effect = Exception("network error")
        result = _fetch_nvd_cvss("CVE-2024-0001")
        assert result["cvss_score"] is None

    @patch.dict("os.environ", {"NVD_API_KEY": "test-key-123"})
    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_uses_api_key_header(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        _fetch_nvd_cvss("CVE-2024-1111")
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["headers"]["apiKey"] == "test-key-123"


# ---------------------------------------------------------------------------
# EPSS batch fetcher tests
# ---------------------------------------------------------------------------


class TestFetchEpssBatch:
    """Test EPSS batch API fetching."""

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_single_cve(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"cve": "CVE-2024-3094", "epss": "0.97565", "percentile": "0.99988"}]}
        mock_get.return_value = mock_resp

        result = _fetch_epss_batch(["CVE-2024-3094"])
        assert result["CVE-2024-3094"]["epss"] == pytest.approx(0.97565)
        assert result["CVE-2024-3094"]["percentile"] == pytest.approx(0.99988)

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_multiple_cves(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"cve": "CVE-2024-3094", "epss": "0.97565", "percentile": "0.99988"},
                {"cve": "CVE-2021-44228", "epss": "0.97500", "percentile": "0.99950"},
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_epss_batch(["CVE-2024-3094", "CVE-2021-44228"])
        assert len(result) == 2
        assert "CVE-2024-3094" in result
        assert "CVE-2021-44228" in result

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_missing_cve_in_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"cve": "CVE-2024-3094", "epss": "0.5", "percentile": "0.8"}]}
        mock_get.return_value = mock_resp

        result = _fetch_epss_batch(["CVE-2024-3094", "CVE-9999-0001"])
        assert result["CVE-2024-3094"]["epss"] == 0.5
        assert result["CVE-9999-0001"]["epss"] is None

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_http_error_returns_none_for_all(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        result = _fetch_epss_batch(["CVE-2024-1111", "CVE-2024-2222"])
        assert result["CVE-2024-1111"]["epss"] is None
        assert result["CVE-2024-2222"]["epss"] is None

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_exception_graceful(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        result = _fetch_epss_batch(["CVE-2024-3094"])
        assert result["CVE-2024-3094"]["epss"] is None

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_batching_over_100(self, mock_get):
        """CVEs > 100 should be split into multiple API calls."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        cve_ids = [f"CVE-2024-{i:04d}" for i in range(150)]
        _fetch_epss_batch(cve_ids)
        # Should make 2 calls: 100 + 50
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# KEV catalogue tests
# ---------------------------------------------------------------------------


class TestFetchKevCatalogue:
    """Test CISA KEV catalogue fetching."""

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cveID": "CVE-2021-44228"},
                {"cveID": "CVE-2024-3094"},
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_kev_catalogue()
        assert "CVE-2021-44228" in result
        assert "CVE-2024-3094" in result

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_http_error_returns_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_get.return_value = mock_resp

        result = _fetch_kev_catalogue()
        assert result == set()

    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_exception_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("network error")
        result = _fetch_kev_catalogue()
        assert result == set()


# ---------------------------------------------------------------------------
# VulnCheck KEV tests
# ---------------------------------------------------------------------------


class TestFetchVulncheckKev:
    """Test VulnCheck KEV enrichment."""

    @patch.dict("os.environ", {}, clear=True)
    def test_no_api_key_returns_empty(self):
        result = _fetch_vulncheck_kev(["CVE-2024-3094"])
        assert result == set()

    @patch.dict("os.environ", {"VULNCHECK_API_KEY": "vc-test-key"})
    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_found_in_vulncheck(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"cve": "CVE-2024-3094"}]}
        mock_get.return_value = mock_resp

        result = _fetch_vulncheck_kev(["CVE-2024-3094"])
        assert "CVE-2024-3094" in result

    @patch.dict("os.environ", {"VULNCHECK_API_KEY": "vc-test-key"})
    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_not_found_in_vulncheck(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        result = _fetch_vulncheck_kev(["CVE-9999-0001"])
        assert result == set()

    @patch.dict("os.environ", {"VULNCHECK_API_KEY": "vc-test-key"})
    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_api_error_graceful(self, mock_get):
        mock_get.side_effect = Exception("down")
        result = _fetch_vulncheck_kev(["CVE-2024-3094"])
        assert result == set()

    @patch.dict("os.environ", {"VULNCHECK_API_KEY": "vc-test-key"})
    @patch("manus_agent.tools.triage_cves._get_with_retry")
    def test_auth_header_sent(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        _fetch_vulncheck_kev(["CVE-2024-1111"])
        call_kwargs = mock_get.call_args
        assert "Bearer vc-test-key" in str(call_kwargs)


# ---------------------------------------------------------------------------
# Core triage logic integration tests
# ---------------------------------------------------------------------------


class TestTriageCves:
    """Test the main triage_cves function end-to-end."""

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    def test_basic_triage_two_cves(self, mock_epss, mock_kev, mock_vc, mock_nvd, mock_sleep):
        mock_epss.return_value = {
            "CVE-2024-3094": {"epss": 0.975, "percentile": 0.999},
            "CVE-2024-0001": {"epss": 0.01, "percentile": 0.3},
        }
        mock_kev.return_value = {"CVE-2024-3094"}
        mock_vc.return_value = set()
        mock_nvd.side_effect = [
            {"cvss_score": 9.8, "cvss_vector": "CVSS:3.1/...", "severity": "CRITICAL"},
            {"cvss_score": 4.0, "cvss_vector": "CVSS:3.1/...", "severity": "MEDIUM"},
        ]

        result = triage_cves(["CVE-2024-3094", "CVE-2024-0001"])
        assert len(result["results"]) == 2
        # First should be the higher-scored one
        assert result["results"][0]["cve_id"] == "CVE-2024-3094"
        assert result["results"][0]["triage_score"] > result["results"][1]["triage_score"]

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    def test_json_output_format(self, mock_epss, mock_kev, mock_vc, mock_nvd, mock_sleep):
        mock_epss.return_value = {"CVE-2024-1111": {"epss": 0.5, "percentile": 0.8}}
        mock_kev.return_value = set()
        mock_vc.return_value = set()
        mock_nvd.return_value = {"cvss_score": 7.0, "cvss_vector": "AV:N", "severity": "HIGH"}

        result = triage_cves(["CVE-2024-1111"], output_format="json")
        # Should not have 'formatted' key in json mode
        assert "formatted" not in result
        assert result["results"][0]["cve_id"] == "CVE-2024-1111"

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    def test_text_output_has_formatted(self, mock_epss, mock_kev, mock_vc, mock_nvd, mock_sleep):
        mock_epss.return_value = {"CVE-2024-1111": {"epss": 0.5, "percentile": 0.8}}
        mock_kev.return_value = set()
        mock_vc.return_value = set()
        mock_nvd.return_value = {"cvss_score": 7.0, "cvss_vector": "AV:N", "severity": "HIGH"}

        result = triage_cves(["CVE-2024-1111"], output_format="text")
        assert "formatted" in result
        assert "TRIAGE REPORT" in result["formatted"]

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    def test_summary_counts(self, mock_epss, mock_kev, mock_vc, mock_nvd, mock_sleep):
        mock_epss.return_value = {
            "CVE-2024-0001": {"epss": 0.975, "percentile": 0.999},
            "CVE-2024-0002": {"epss": 0.5, "percentile": 0.8},
            "CVE-2024-0003": {"epss": 0.01, "percentile": 0.1},
        }
        mock_kev.return_value = {"CVE-2024-0001"}
        mock_vc.return_value = set()
        mock_nvd.side_effect = [
            {"cvss_score": 9.8, "cvss_vector": "...", "severity": "CRITICAL"},
            {"cvss_score": 6.0, "cvss_vector": "...", "severity": "MEDIUM"},
            {"cvss_score": 2.0, "cvss_vector": "...", "severity": "LOW"},
        ]

        result = triage_cves(["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"])
        summary = result["summary"]
        assert summary["total_cves"] == 3
        assert summary["kev_count"] == 1

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    def test_results_sorted_descending(self, mock_epss, mock_kev, mock_vc, mock_nvd, mock_sleep):
        mock_epss.return_value = {
            "CVE-2024-0001": {"epss": 0.1, "percentile": 0.3},
            "CVE-2024-0002": {"epss": 0.9, "percentile": 0.99},
        }
        mock_kev.return_value = set()
        mock_vc.return_value = set()
        mock_nvd.side_effect = [
            {"cvss_score": 3.0, "cvss_vector": "...", "severity": "LOW"},
            {"cvss_score": 9.0, "cvss_vector": "...", "severity": "CRITICAL"},
        ]

        result = triage_cves(["CVE-2024-0001", "CVE-2024-0002"])
        scores = [r["triage_score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    def test_vulncheck_kev_boosts_score(self, mock_epss, mock_kev, mock_vc, mock_nvd, mock_sleep):
        mock_epss.return_value = {"CVE-2024-5555": {"epss": 0.5, "percentile": 0.8}}
        mock_kev.return_value = set()  # Not in CISA KEV
        mock_vc.return_value = {"CVE-2024-5555"}  # But in VulnCheck KEV
        mock_nvd.return_value = {"cvss_score": 7.0, "cvss_vector": "...", "severity": "HIGH"}

        result = triage_cves(["CVE-2024-5555"])
        rec = result["results"][0]
        assert rec["in_vulncheck_kev"] is True
        assert rec["in_cisa_kev"] is False
        # Score should include KEV boost
        score_with_kev = _compute_triage_score(cvss_score=7.0, epss_score=0.5, in_kev=True)
        assert rec["triage_score"] == score_with_kev


# ---------------------------------------------------------------------------
# Text formatting tests
# ---------------------------------------------------------------------------


class TestFormatText:
    """Test text report formatting."""

    def test_header_present(self):
        records = [
            {
                "cve_id": "CVE-2024-3094",
                "cvss_score": 9.8,
                "cvss_severity": "CRITICAL",
                "epss_score": 0.975,
                "epss_percentile": 0.999,
                "in_cisa_kev": True,
                "in_vulncheck_kev": False,
                "triage_score": 95.4,
                "urgency": "CRITICAL",
            }
        ]
        summary = {
            "total_cves": 1,
            "critical_count": 1,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "info_count": 0,
            "kev_count": 1,
            "invalid_ids": None,
        }
        text = _format_text(records, summary, [])
        assert "TRIAGE REPORT" in text
        assert "CVE-2024-3094" in text
        assert "CRITICAL" in text

    def test_kev_yes_label(self):
        records = [
            {
                "cve_id": "CVE-2021-44228",
                "cvss_score": 10.0,
                "cvss_severity": "CRITICAL",
                "epss_score": 0.975,
                "epss_percentile": 0.999,
                "in_cisa_kev": True,
                "in_vulncheck_kev": False,
                "triage_score": 100.0,
                "urgency": "CRITICAL",
            }
        ]
        summary = {
            "total_cves": 1,
            "critical_count": 1,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "info_count": 0,
            "kev_count": 1,
            "invalid_ids": None,
        }
        text = _format_text(records, summary, [])
        assert "YES" in text

    def test_invalid_ids_shown(self):
        records = []
        summary = {
            "total_cves": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "info_count": 0,
            "kev_count": 0,
            "invalid_ids": ["bad-1"],
        }
        text = _format_text(records, summary, ["bad-1"])
        assert "bad-1" in text

    def test_none_cvss_shows_na(self):
        records = [
            {
                "cve_id": "CVE-2024-9999",
                "cvss_score": None,
                "cvss_severity": None,
                "epss_score": None,
                "epss_percentile": None,
                "in_cisa_kev": False,
                "in_vulncheck_kev": False,
                "triage_score": 0.0,
                "urgency": "INFO",
            }
        ]
        summary = {
            "total_cves": 1,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "info_count": 1,
            "kev_count": 0,
            "invalid_ids": None,
        }
        text = _format_text(records, summary, [])
        assert "N/A" in text

    def test_methodology_section(self):
        records = []
        summary = {
            "total_cves": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "info_count": 0,
            "kev_count": 0,
            "invalid_ids": None,
        }
        text = _format_text(records, summary, [])
        assert "SCORING METHODOLOGY" in text
        assert "NVD" in text
        assert "EPSS" in text


# ---------------------------------------------------------------------------
# Strands handler tests
# ---------------------------------------------------------------------------


class TestTriageCvesHandler:
    """Test the Strands tool handler wrapper."""

    @patch("manus_agent.tools.triage_cves.triage_cves")
    def test_success_text_format(self, mock_triage):
        mock_triage.return_value = {
            "results": [{"cve_id": "CVE-2024-3094", "triage_score": 95.0}],
            "summary": {"total_cves": 1},
            "formatted": "Report text here",
        }
        tool_use = {"toolUseId": "test-123", "input": {"cve_ids": ["CVE-2024-3094"]}}
        result = triage_cves_handler(tool_use)
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"
        assert "Report text here" in result["content"][0]["text"]

    @patch("manus_agent.tools.triage_cves.triage_cves")
    def test_success_json_format(self, mock_triage):
        mock_triage.return_value = {
            "results": [{"cve_id": "CVE-2024-3094", "triage_score": 95.0}],
            "summary": {"total_cves": 1},
        }
        tool_use = {
            "toolUseId": "test-456",
            "input": {"cve_ids": ["CVE-2024-3094"], "output_format": "json"},
        }
        result = triage_cves_handler(tool_use)
        assert result["status"] == "success"
        # Should be valid JSON
        parsed = json.loads(result["content"][0]["text"])
        assert parsed["results"][0]["cve_id"] == "CVE-2024-3094"

    @patch("manus_agent.tools.triage_cves.triage_cves")
    def test_error_result(self, mock_triage):
        mock_triage.return_value = {
            "error": "No valid CVE IDs found.",
            "results": [],
            "summary": {},
        }
        tool_use = {"toolUseId": "test-789", "input": {"cve_ids": ["bad"]}}
        result = triage_cves_handler(tool_use)
        assert result["status"] == "error"
        assert "Error" in result["content"][0]["text"]

    @patch("manus_agent.tools.triage_cves.triage_cves")
    def test_default_output_format(self, mock_triage):
        mock_triage.return_value = {
            "results": [{"cve_id": "CVE-2024-1111"}],
            "summary": {},
            "formatted": "text output",
        }
        tool_use = {"toolUseId": "test-000", "input": {"cve_ids": ["CVE-2024-1111"]}}
        triage_cves_handler(tool_use)
        mock_triage.assert_called_once_with(["CVE-2024-1111"], output_format="text")


# ---------------------------------------------------------------------------
# CLI subcommand tests
# ---------------------------------------------------------------------------


class TestCliTriage:
    """Test the CLI dispatch for triage subcommand."""

    @patch("manus_agent.tools.triage_cves.triage_cves")
    def test_cli_triage_from_args(self, mock_triage):
        import io

        mock_triage.return_value = {
            "results": [{"cve_id": "CVE-2024-3094", "triage_score": 95.0}],
            "summary": {"total_cves": 1},
            "formatted": "Report output\n",
        }
        from manus_agent.cli import _run_triage

        with patch("sys.stderr", new=io.StringIO()), patch("sys.stdout", new=io.StringIO()):
            rc = _run_triage(["CVE-2024-3094"])
        assert rc == 0

    @patch("manus_agent.tools.triage_cves.triage_cves")
    def test_cli_triage_json_output(self, mock_triage):
        import io

        mock_triage.return_value = {
            "results": [{"cve_id": "CVE-2024-3094", "triage_score": 95.0}],
            "summary": {"total_cves": 1},
        }
        from manus_agent.cli import _run_triage

        mock_stdout = io.StringIO()
        with patch("sys.stderr", new=io.StringIO()), patch("sys.stdout", new=mock_stdout):
            rc = _run_triage(["CVE-2024-3094", "--output", "json"])
        assert rc == 0
        output = mock_stdout.getvalue()
        parsed = json.loads(output)
        assert parsed["results"][0]["cve_id"] == "CVE-2024-3094"

    @patch("manus_agent.tools.triage_cves.triage_cves")
    def test_cli_triage_error(self, mock_triage):
        import io

        mock_triage.return_value = {"error": "No valid CVE IDs found.", "results": [], "summary": {}}
        from manus_agent.cli import _run_triage

        with patch("sys.stderr", new=io.StringIO()), patch("sys.stdout", new=io.StringIO()):
            rc = _run_triage(["bad-id"])
        assert rc == 1

    def test_cli_triage_from_file(self, tmp_path):
        import io

        cve_file = tmp_path / "cves.txt"
        cve_file.write_text("CVE-2024-3094\n# comment\nCVE-2021-44228\n")

        with patch("manus_agent.tools.triage_cves.triage_cves") as mock_triage:
            mock_triage.return_value = {
                "results": [
                    {"cve_id": "CVE-2024-3094", "triage_score": 95.0},
                    {"cve_id": "CVE-2021-44228", "triage_score": 90.0},
                ],
                "summary": {"total_cves": 2},
                "formatted": "Report\n",
            }
            from manus_agent.cli import _run_triage

            with patch("sys.stderr", new=io.StringIO()), patch("sys.stdout", new=io.StringIO()):
                rc = _run_triage(["--file", str(cve_file)])

        assert rc == 0
        call_args = mock_triage.call_args[0][0]
        assert "CVE-2024-3094" in call_args
        assert "CVE-2021-44228" in call_args

    def test_cli_triage_missing_file(self):
        import io

        from manus_agent.cli import _run_triage

        with patch("sys.stderr", new=io.StringIO()), patch("sys.stdout", new=io.StringIO()):
            rc = _run_triage(["--file", "/nonexistent/path.txt"])
        assert rc == 1


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    def test_all_data_missing(self, mock_epss, mock_kev, mock_vc, mock_nvd, mock_sleep):
        """Triage still works when all external sources return nothing."""
        mock_epss.return_value = {"CVE-2024-9999": {"epss": None, "percentile": None}}
        mock_kev.return_value = set()
        mock_vc.return_value = set()
        mock_nvd.return_value = {"cvss_score": None, "cvss_vector": None, "severity": None}

        result = triage_cves(["CVE-2024-9999"])
        assert len(result["results"]) == 1
        assert result["results"][0]["triage_score"] == 0.0
        assert result["results"][0]["urgency"] == "INFO"

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    def test_whitespace_in_cve_ids(self, mock_epss, mock_kev, mock_vc, mock_nvd, mock_sleep):
        mock_epss.return_value = {"CVE-2024-1234": {"epss": 0.5, "percentile": 0.8}}
        mock_kev.return_value = set()
        mock_vc.return_value = set()
        mock_nvd.return_value = {"cvss_score": 5.0, "cvss_vector": "...", "severity": "MEDIUM"}

        result = triage_cves(["  CVE-2024-1234  "])
        assert len(result["results"]) == 1
        assert result["results"][0]["cve_id"] == "CVE-2024-1234"

    def test_exactly_50_cves_allowed(self):
        """50 CVEs should not trigger the limit error."""
        cve_ids = [f"CVE-2024-{i:04d}" for i in range(50)]

        with (
            patch("manus_agent.tools.triage_cves._fetch_epss_batch") as mock_epss,
            patch("manus_agent.tools.triage_cves._fetch_kev_catalogue") as mock_kev,
            patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev") as mock_vc,
            patch("manus_agent.tools.triage_cves._fetch_nvd_cvss") as mock_nvd,
            patch("manus_agent.tools.triage_cves.time.sleep"),
        ):
            mock_epss.return_value = {cid.upper(): {"epss": 0.1, "percentile": 0.5} for cid in cve_ids}
            mock_kev.return_value = set()
            mock_vc.return_value = set()
            mock_nvd.return_value = {"cvss_score": 5.0, "cvss_vector": "...", "severity": "MEDIUM"}

            result = triage_cves(cve_ids)
            assert "error" not in result
            assert len(result["results"]) == 50

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    def test_cvss_v2_severity_mapping(self, mock_epss, mock_kev, mock_vc, mock_nvd, mock_sleep):
        """V2 scores map to severity labels correctly."""
        mock_epss.return_value = {"CVE-2015-0001": {"epss": 0.1, "percentile": 0.5}}
        mock_kev.return_value = set()
        mock_vc.return_value = set()
        mock_nvd.return_value = {"cvss_score": 3.5, "cvss_vector": "AV:N/...", "severity": "LOW"}

        result = triage_cves(["CVE-2015-0001"])
        assert result["results"][0]["cvss_severity"] == "LOW"


# ---------------------------------------------------------------------------
# NVD rate limiting tests
# ---------------------------------------------------------------------------


class TestNvdRateLimiting:
    """Test NVD rate limiting behavior."""

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    def test_sleep_between_nvd_calls_without_api_key(self, mock_epss, mock_kev, mock_vc, mock_nvd, mock_sleep):
        mock_epss.return_value = {
            "CVE-2024-0001": {"epss": 0.1, "percentile": 0.5},
            "CVE-2024-0002": {"epss": 0.2, "percentile": 0.6},
        }
        mock_kev.return_value = set()
        mock_vc.return_value = set()
        mock_nvd.return_value = {"cvss_score": 5.0, "cvss_vector": "...", "severity": "MEDIUM"}

        with patch.dict("os.environ", {}, clear=True):
            triage_cves(["CVE-2024-0001", "CVE-2024-0002"])

        # Should have called sleep at least once (between NVD requests)
        assert mock_sleep.call_count >= 1

    @patch("manus_agent.tools.triage_cves.time.sleep")
    @patch("manus_agent.tools.triage_cves._fetch_nvd_cvss")
    @patch("manus_agent.tools.triage_cves._fetch_vulncheck_kev")
    @patch("manus_agent.tools.triage_cves._fetch_kev_catalogue")
    @patch("manus_agent.tools.triage_cves._fetch_epss_batch")
    @patch.dict("os.environ", {"NVD_API_KEY": "test-key"})
    def test_shorter_sleep_with_api_key(self, mock_epss, mock_kev, mock_vc, mock_nvd, mock_sleep):
        mock_epss.return_value = {
            "CVE-2024-0001": {"epss": 0.1, "percentile": 0.5},
            "CVE-2024-0002": {"epss": 0.2, "percentile": 0.6},
        }
        mock_kev.return_value = set()
        mock_vc.return_value = set()
        mock_nvd.return_value = {"cvss_score": 5.0, "cvss_vector": "...", "severity": "MEDIUM"}

        triage_cves(["CVE-2024-0001", "CVE-2024-0002"])

        # With API key, sleep should use shorter delay (0.15)
        if mock_sleep.call_count > 0:
            call_arg = mock_sleep.call_args_list[0][0][0]
            assert call_arg <= 0.15
