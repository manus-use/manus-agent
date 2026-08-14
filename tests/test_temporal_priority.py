"""
Comprehensive test suite for the get_temporal_priority tool module.

Tests cover: TOOL_SPEC contract, input validation, HTTP retry logic,
NVD data fetching, EPSS data fetching, KEV status checking, CVSS extraction,
patch detection, spike analysis, score computation, age calculation,
urgency labels, text rendering, handler integration, CLI subcommand, and edge cases.

All tests are 100% mocked — no real HTTP calls.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.get_temporal_priority import (
    TOOL_SPEC,
    _analyse_spike,
    _check_patch_refs,
    _compute_age_days,
    _extract_cvss,
    _http_get,
    _safe_float,
    _urgency_label,
    compute_score,
    compute_temporal_priority,
    fetch_epss_data,
    fetch_kev_status,
    fetch_nvd_data,
    get_temporal_priority,
    render_text,
)

# ===========================================================================
# TOOL_SPEC contract tests
# ===========================================================================


class TestToolSpec:
    """Verify TOOL_SPEC follows Strands SDK conventions."""

    def test_has_name(self):
        assert TOOL_SPEC["name"] == "get_temporal_priority"

    def test_has_description(self):
        assert "temporal priority" in TOOL_SPEC["description"].lower()
        assert len(TOOL_SPEC["description"]) > 50

    def test_input_schema_has_cve_id(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_input_schema_type_object(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"


# ===========================================================================
# Input validation tests
# ===========================================================================


class TestInputValidation:
    """Test handler input validation."""

    def test_empty_cve_id(self):
        tool_use = {"toolUseId": "test-1", "input": {"cve_id": ""}}
        result = get_temporal_priority(tool_use)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_none_cve_id(self):
        tool_use = {"toolUseId": "test-2", "input": {"cve_id": None}}
        result = get_temporal_priority(tool_use)
        assert result["status"] == "error"

    def test_missing_cve_id(self):
        tool_use = {"toolUseId": "test-3", "input": {}}
        result = get_temporal_priority(tool_use)
        assert result["status"] == "error"

    def test_invalid_format(self):
        tool_use = {"toolUseId": "test-4", "input": {"cve_id": "not-a-cve"}}
        result = get_temporal_priority(tool_use)
        assert result["status"] == "error"
        assert "Invalid CVE format" in result["content"][0]["text"]

    def test_numeric_only(self):
        tool_use = {"toolUseId": "test-5", "input": {"cve_id": "12345"}}
        result = get_temporal_priority(tool_use)
        assert result["status"] == "error"

    def test_too_short_number(self):
        tool_use = {"toolUseId": "test-6", "input": {"cve_id": "CVE-2024-12"}}
        result = get_temporal_priority(tool_use)
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_temporal_priority.compute_temporal_priority")
    def test_valid_cve_id(self, mock_compute):
        mock_compute.return_value = {"cve_id": "CVE-2024-3094", "score": 75.0, "label": "HIGH"}
        tool_use = {"toolUseId": "test-7", "input": {"cve_id": "CVE-2024-3094"}}
        result = get_temporal_priority(tool_use)
        assert result["status"] == "success"


# ===========================================================================
# HTTP retry tests
# ===========================================================================


class TestHttpRetry:
    """Test HTTP retry/back-off logic."""

    @patch("manus_agent.tools.get_temporal_priority.time.sleep")
    @patch("manus_agent.tools.get_temporal_priority.requests.get")
    def test_success_on_first_try(self, mock_get, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        result = _http_get("https://example.com")
        assert result == mock_resp
        mock_sleep.assert_not_called()

    @patch("manus_agent.tools.get_temporal_priority.time.sleep")
    @patch("manus_agent.tools.get_temporal_priority.requests.get")
    def test_retry_on_429(self, mock_get, mock_sleep):
        resp_429 = MagicMock()
        resp_429.status_code = 429

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.raise_for_status = MagicMock()

        mock_get.side_effect = [resp_429, resp_200]
        result = _http_get("https://example.com")
        assert result == resp_200
        assert mock_sleep.call_count == 1

    @patch("manus_agent.tools.get_temporal_priority._MAX_RETRIES", 2)
    @patch("manus_agent.tools.get_temporal_priority.time.sleep")
    @patch("manus_agent.tools.get_temporal_priority.requests.get")
    def test_all_retries_exhausted(self, mock_get, mock_sleep):
        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("timeout")
        with pytest.raises(req.exceptions.ConnectionError):
            _http_get("https://example.com")
        assert mock_sleep.call_count == 1  # retries - 1

    @patch("manus_agent.tools.get_temporal_priority.time.sleep")
    @patch("manus_agent.tools.get_temporal_priority.requests.get")
    def test_retry_on_connection_error(self, mock_get, mock_sleep):
        import requests as req

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.raise_for_status = MagicMock()
        mock_get.side_effect = [req.exceptions.ConnectionError("fail"), resp_ok]
        result = _http_get("https://example.com")
        assert result == resp_ok

    @patch("manus_agent.tools.get_temporal_priority.time.sleep")
    @patch("manus_agent.tools.get_temporal_priority.requests.get")
    def test_passes_params_and_headers(self, mock_get, mock_sleep):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp
        _http_get("https://example.com/api", params={"q": "x"}, headers={"X-Key": "abc"})
        mock_get.assert_called_once_with(
            "https://example.com/api",
            params={"q": "x"},
            headers={"X-Key": "abc"},
            timeout=20,
        )


# ===========================================================================
# NVD data fetching tests
# ===========================================================================


class TestFetchNvdData:
    """Test NVD data fetching and parsing."""

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    def test_successful_fetch(self, mock_http):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 9.8}}]},
                        "published": "2024-03-29T00:00:00.000",
                        "references": [{"url": "https://patch.example.com", "tags": ["Patch"]}],
                    }
                }
            ]
        }
        mock_http.return_value = mock_resp
        result = fetch_nvd_data("CVE-2024-3094")
        assert result["cvss_score"] == 9.8
        assert result["published_date"] == "2024-03-29T00:00:00.000"
        assert result["has_patch"] is True

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    def test_no_vulnerabilities(self, mock_http):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_http.return_value = mock_resp
        result = fetch_nvd_data("CVE-9999-0000")
        assert result["cvss_score"] is None
        assert result["published_date"] is None
        assert result["has_patch"] is None

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    def test_http_error_graceful(self, mock_http):
        import requests as req

        mock_http.side_effect = req.exceptions.ConnectionError("timeout")
        result = fetch_nvd_data("CVE-2024-3094")
        assert result["cvss_score"] is None
        assert result["has_patch"] is None

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    def test_no_patch_refs(self, mock_http):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 7.5}}]},
                        "published": "2024-01-01T00:00:00.000",
                        "references": [{"url": "https://example.com", "tags": ["Third Party Advisory"]}],
                    }
                }
            ]
        }
        mock_http.return_value = mock_resp
        result = fetch_nvd_data("CVE-2024-1234")
        assert result["has_patch"] is False

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    @patch.dict("os.environ", {"NVD_API_KEY": "test-key-123"})
    def test_uses_api_key(self, mock_http):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_http.return_value = mock_resp
        fetch_nvd_data("CVE-2024-3094")
        call_kwargs = mock_http.call_args
        assert call_kwargs[1]["headers"]["apiKey"] == "test-key-123"


# ===========================================================================
# CVSS extraction tests
# ===========================================================================


class TestExtractCvss:
    """Test CVSS score extraction from NVD metrics."""

    def test_v31_preferred(self):
        cve_item = {
            "metrics": {
                "cvssMetricV31": [{"cvssData": {"baseScore": 9.8}}],
                "cvssMetricV30": [{"cvssData": {"baseScore": 8.5}}],
            }
        }
        assert _extract_cvss(cve_item) == 9.8

    def test_v30_fallback(self):
        cve_item = {"metrics": {"cvssMetricV30": [{"cvssData": {"baseScore": 7.2}}]}}
        assert _extract_cvss(cve_item) == 7.2

    def test_v2_fallback(self):
        cve_item = {"metrics": {"cvssMetricV2": [{"cvssData": {"baseScore": 6.0}}]}}
        assert _extract_cvss(cve_item) == 6.0

    def test_no_metrics(self):
        cve_item = {"metrics": {}}
        assert _extract_cvss(cve_item) is None

    def test_empty_cve_item(self):
        assert _extract_cvss({}) is None


# ===========================================================================
# Patch reference detection tests
# ===========================================================================


class TestCheckPatchRefs:
    """Test patch reference detection in NVD references."""

    def test_patch_found(self):
        cve_item = {"references": [{"url": "https://github.com/fix", "tags": ["Patch"]}]}
        assert _check_patch_refs(cve_item) is True

    def test_no_patch(self):
        cve_item = {"references": [{"url": "https://blog.example.com", "tags": ["Third Party Advisory"]}]}
        assert _check_patch_refs(cve_item) is False

    def test_no_references(self):
        cve_item = {"references": []}
        assert _check_patch_refs(cve_item) is False

    def test_empty_cve_item(self):
        assert _check_patch_refs({}) is False

    def test_multiple_refs_one_patch(self):
        cve_item = {
            "references": [
                {"url": "https://blog.example.com", "tags": ["Third Party Advisory"]},
                {"url": "https://github.com/commit/abc", "tags": ["Patch", "Third Party Advisory"]},
            ]
        }
        assert _check_patch_refs(cve_item) is True


# ===========================================================================
# EPSS data fetching tests
# ===========================================================================


class TestFetchEpssData:
    """Test EPSS data fetching and spike analysis."""

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    def test_successful_fetch(self, mock_http):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "epss": "0.95",
                    "time-series": [
                        {"date": "2024-03-01", "epss": "0.10", "percentile": "0.80"},
                        {"date": "2024-03-02", "epss": "0.95", "percentile": "0.99"},
                    ],
                }
            ]
        }
        mock_http.return_value = mock_resp
        result = fetch_epss_data("CVE-2024-3094")
        assert result["current_epss"] == 0.95
        assert result["spike_detected"] is True
        assert result["max_jump"] == pytest.approx(0.85, abs=0.01)

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    def test_no_data(self, mock_http):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_http.return_value = mock_resp
        result = fetch_epss_data("CVE-9999-0000")
        assert result["current_epss"] is None
        assert result["spike_detected"] is None

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    def test_http_error_graceful(self, mock_http):
        import requests as req

        mock_http.side_effect = req.exceptions.Timeout("timeout")
        result = fetch_epss_data("CVE-2024-3094")
        assert result["current_epss"] is None

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    def test_no_spike(self, mock_http):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "epss": "0.05",
                    "time-series": [
                        {"date": "2024-03-01", "epss": "0.04", "percentile": "0.50"},
                        {"date": "2024-03-02", "epss": "0.05", "percentile": "0.51"},
                    ],
                }
            ]
        }
        mock_http.return_value = mock_resp
        result = fetch_epss_data("CVE-2024-1234")
        assert result["current_epss"] == 0.05
        assert result["spike_detected"] is False


# ===========================================================================
# Spike analysis tests
# ===========================================================================


class TestAnalyseSpike:
    """Test EPSS spike detection logic."""

    def test_no_series(self):
        result = _analyse_spike([])
        assert result["spike_detected"] is False
        assert result["max_jump"] == 0.0

    def test_single_point(self):
        result = _analyse_spike([{"date": "2024-01-01", "epss": "0.5"}])
        assert result["spike_detected"] is False

    def test_large_spike(self):
        series = [
            {"date": "2024-01-01", "epss": "0.01"},
            {"date": "2024-01-02", "epss": "0.50"},
        ]
        result = _analyse_spike(series)
        assert result["spike_detected"] is True
        assert result["max_jump"] == pytest.approx(0.49, abs=0.01)

    def test_below_threshold(self):
        series = [
            {"date": "2024-01-01", "epss": "0.01"},
            {"date": "2024-01-02", "epss": "0.02"},
        ]
        result = _analyse_spike(series)
        assert result["spike_detected"] is False
        assert result["max_jump"] == pytest.approx(0.01, abs=0.001)

    def test_days_since_spike_computed(self):
        # Use a date far in the past
        series = [
            {"date": "2020-01-01", "epss": "0.01"},
            {"date": "2020-01-02", "epss": "0.80"},
        ]
        result = _analyse_spike(series)
        assert result["spike_detected"] is True
        assert result["days_since_spike"] is not None
        assert result["days_since_spike"] > 1000

    def test_descending_series_no_spike(self):
        series = [
            {"date": "2024-01-01", "epss": "0.90"},
            {"date": "2024-01-02", "epss": "0.80"},
            {"date": "2024-01-03", "epss": "0.70"},
        ]
        result = _analyse_spike(series)
        assert result["spike_detected"] is False
        assert result["max_jump"] == 0.0


# ===========================================================================
# KEV status tests
# ===========================================================================


class TestFetchKevStatus:
    """Test CISA KEV catalog lookup."""

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    def test_cve_in_kev(self, mock_http):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cveID": "CVE-2024-3094", "dateAdded": "2024-03-30"},
                {"cveID": "CVE-2021-44228", "dateAdded": "2021-12-10"},
            ]
        }
        mock_http.return_value = mock_resp
        result = fetch_kev_status("CVE-2024-3094")
        assert result["in_kev"] is True
        assert result["date_added"] == "2024-03-30"

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    def test_cve_not_in_kev(self, mock_http):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [{"cveID": "CVE-2021-44228", "dateAdded": "2021-12-10"}]}
        mock_http.return_value = mock_resp
        result = fetch_kev_status("CVE-2024-9999")
        assert result["in_kev"] is False
        assert result["date_added"] is None

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    def test_http_error_graceful(self, mock_http):
        import requests as req

        mock_http.side_effect = req.exceptions.ConnectionError("fail")
        result = fetch_kev_status("CVE-2024-3094")
        assert result["in_kev"] is None

    @patch("manus_agent.tools.get_temporal_priority._http_get")
    def test_case_insensitive_match(self, mock_http):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [{"cveID": "cve-2024-3094", "dateAdded": "2024-03-30"}]}
        mock_http.return_value = mock_resp
        result = fetch_kev_status("CVE-2024-3094")
        assert result["in_kev"] is True


# ===========================================================================
# Score computation tests
# ===========================================================================


class TestComputeScore:
    """Test the core scoring logic."""

    def test_all_signals_max(self):
        """All signals at maximum = CRITICAL."""
        result = compute_score(
            cvss_score=10.0,
            current_epss=1.0,
            spike_detected=True,
            days_since_spike=0,
            in_kev=True,
            has_patch=False,
            published_date=datetime.now(timezone.utc).isoformat(),
        )
        assert result["score"] >= 85
        assert result["label"] == "CRITICAL"
        assert result["signals_available"] == 6

    def test_all_signals_min(self):
        """All signals at minimum = low score."""
        result = compute_score(
            cvss_score=0.0,
            current_epss=0.0,
            spike_detected=False,
            days_since_spike=None,
            in_kev=False,
            has_patch=True,
            published_date="2010-01-01T00:00:00.000",
        )
        assert result["score"] < 30
        assert result["label"] in ("LOW", "INFORMATIONAL")

    def test_kev_adds_significant_weight(self):
        """KEV membership should significantly increase score."""
        base = compute_score(
            cvss_score=7.0,
            current_epss=0.3,
            spike_detected=False,
            days_since_spike=None,
            in_kev=False,
            has_patch=False,
            published_date="2024-01-01T00:00:00.000",
        )
        with_kev = compute_score(
            cvss_score=7.0,
            current_epss=0.3,
            spike_detected=False,
            days_since_spike=None,
            in_kev=True,
            has_patch=False,
            published_date="2024-01-01T00:00:00.000",
        )
        assert with_kev["score"] > base["score"]
        assert with_kev["score"] - base["score"] > 10

    def test_patch_reduces_urgency(self):
        """Having a patch should reduce the score."""
        unpatched = compute_score(
            cvss_score=8.0,
            current_epss=0.5,
            spike_detected=False,
            days_since_spike=None,
            in_kev=False,
            has_patch=False,
            published_date="2024-06-01T00:00:00.000",
        )
        patched = compute_score(
            cvss_score=8.0,
            current_epss=0.5,
            spike_detected=False,
            days_since_spike=None,
            in_kev=False,
            has_patch=True,
            published_date="2024-06-01T00:00:00.000",
        )
        assert unpatched["score"] > patched["score"]

    def test_no_signals_available(self):
        """All signals unavailable should still produce a valid score."""
        result = compute_score(
            cvss_score=None,
            current_epss=None,
            spike_detected=None,
            days_since_spike=None,
            in_kev=None,
            has_patch=None,
            published_date=None,
        )
        assert 0 <= result["score"] <= 100
        assert result["signals_available"] == 0
        assert result["signals_total"] == 6

    def test_score_clamped_0_100(self):
        """Score should always be between 0 and 100."""
        result = compute_score(
            cvss_score=10.0,
            current_epss=1.0,
            spike_detected=True,
            days_since_spike=0,
            in_kev=True,
            has_patch=False,
            published_date=datetime.now(timezone.utc).isoformat(),
        )
        assert 0 <= result["score"] <= 100

    def test_spike_with_recent_date(self):
        """Recent spike should contribute more than old spike."""
        recent = compute_score(
            cvss_score=7.0,
            current_epss=0.5,
            spike_detected=True,
            days_since_spike=1,
            in_kev=False,
            has_patch=False,
            published_date="2024-06-01T00:00:00.000",
        )
        old = compute_score(
            cvss_score=7.0,
            current_epss=0.5,
            spike_detected=True,
            days_since_spike=90,
            in_kev=False,
            has_patch=False,
            published_date="2024-06-01T00:00:00.000",
        )
        assert recent["score"] > old["score"]

    def test_breakdown_structure(self):
        """Breakdown should have all 6 signal keys."""
        result = compute_score(
            cvss_score=7.5,
            current_epss=0.3,
            spike_detected=False,
            days_since_spike=None,
            in_kev=False,
            has_patch=False,
            published_date="2024-01-01T00:00:00.000",
        )
        breakdown = result["breakdown"]
        assert "cvss" in breakdown
        assert "epss" in breakdown
        assert "spike" in breakdown
        assert "kev" in breakdown
        assert "patch" in breakdown
        assert "age" in breakdown


# ===========================================================================
# Age computation tests
# ===========================================================================


class TestComputeAgeDays:
    """Test CVE age calculation."""

    def test_recent_date(self):
        recent = datetime.now(timezone.utc).isoformat()
        days = _compute_age_days(recent)
        assert days is not None
        assert days <= 1

    def test_old_date(self):
        days = _compute_age_days("2020-01-01T00:00:00.000")
        assert days is not None
        assert days > 1000

    def test_none_date(self):
        assert _compute_age_days(None) is None

    def test_empty_string(self):
        assert _compute_age_days("") is None

    def test_invalid_format(self):
        assert _compute_age_days("not-a-date") is None

    def test_with_z_suffix(self):
        days = _compute_age_days("2023-06-15T12:00:00Z")
        assert days is not None
        assert days > 300


# ===========================================================================
# Urgency label tests
# ===========================================================================


class TestUrgencyLabel:
    """Test urgency label mapping."""

    def test_critical(self):
        assert _urgency_label(90) == "CRITICAL"
        assert _urgency_label(85) == "CRITICAL"

    def test_high(self):
        assert _urgency_label(75) == "HIGH"
        assert _urgency_label(70) == "HIGH"

    def test_medium(self):
        assert _urgency_label(60) == "MEDIUM"
        assert _urgency_label(50) == "MEDIUM"

    def test_low(self):
        assert _urgency_label(40) == "LOW"
        assert _urgency_label(30) == "LOW"

    def test_informational(self):
        assert _urgency_label(20) == "INFORMATIONAL"
        assert _urgency_label(0) == "INFORMATIONAL"


# ===========================================================================
# Text rendering tests
# ===========================================================================


class TestRenderText:
    """Test human-readable text output."""

    def test_renders_score(self):
        result = {
            "cve_id": "CVE-2024-3094",
            "score": 85.5,
            "label": "CRITICAL",
            "signals_available": 6,
            "signals_total": 6,
            "breakdown": {
                "cvss": {"raw": 9.8, "points": 24.5, "max": 25},
                "epss": {"raw": 0.95, "points": 24.4, "max": 25},
                "spike": {"detected": True, "days_since": 2, "points": 12.0, "max": 15},
                "kev": {"in_kev": True, "points": 20.0, "max": 20},
                "patch": {"has_patch": False, "points": 10.0, "max": 10},
                "age": {"days": 30, "points": 4.5, "max": 5},
            },
        }
        text = render_text(result)
        assert "CVE-2024-3094" in text
        assert "85.5" in text
        assert "CRITICAL" in text
        assert "9.8" in text

    def test_renders_unavailable_signals(self):
        result = {
            "cve_id": "CVE-2024-0000",
            "score": 10.0,
            "label": "INFORMATIONAL",
            "signals_available": 0,
            "signals_total": 6,
            "breakdown": {
                "cvss": {"raw": None, "points": 0, "max": 25, "note": "unavailable"},
                "epss": {"raw": None, "points": 0, "max": 25, "note": "unavailable"},
                "spike": {"detected": None, "points": 0, "max": 15, "note": "unavailable"},
                "kev": {"in_kev": None, "points": 0, "max": 20, "note": "unavailable"},
                "patch": {"has_patch": None, "points": 0, "max": 10, "note": "unavailable"},
                "age": {"days": None, "points": 0, "max": 5, "note": "unavailable"},
            },
        }
        text = render_text(result)
        assert "unavailable" in text
        assert "0/6" in text

    def test_renders_kev_alert(self):
        result = {
            "cve_id": "CVE-2024-3094",
            "score": 80.0,
            "label": "HIGH",
            "signals_available": 4,
            "signals_total": 6,
            "breakdown": {
                "cvss": {"raw": 9.8, "points": 24.5, "max": 25},
                "epss": {"raw": None, "points": 0, "max": 25, "note": "unavailable"},
                "spike": {"detected": None, "points": 0, "max": 15, "note": "unavailable"},
                "kev": {"in_kev": True, "points": 20.0, "max": 20},
                "patch": {"has_patch": True, "points": -10.0, "max": 10},
                "age": {"days": 100, "points": 3.4, "max": 5},
            },
        }
        text = render_text(result)
        assert "IN CATALOG" in text

    def test_renders_progress_bar(self):
        result = {
            "cve_id": "CVE-2024-1234",
            "score": 50.0,
            "label": "MEDIUM",
            "signals_available": 6,
            "signals_total": 6,
            "breakdown": {
                "cvss": {"raw": 5.0, "points": 12.5, "max": 25},
                "epss": {"raw": 0.1, "points": 7.9, "max": 25},
                "spike": {"detected": False, "days_since": None, "points": 0, "max": 15},
                "kev": {"in_kev": False, "points": 0, "max": 20},
                "patch": {"has_patch": False, "points": 10, "max": 10},
                "age": {"days": 60, "points": 3.9, "max": 5},
            },
        }
        text = render_text(result)
        assert "█" in text
        assert "░" in text


# ===========================================================================
# Integration: compute_temporal_priority
# ===========================================================================


class TestComputeTemporalPriority:
    """Test the full orchestration pipeline."""

    @patch("manus_agent.tools.get_temporal_priority.fetch_kev_status")
    @patch("manus_agent.tools.get_temporal_priority.fetch_epss_data")
    @patch("manus_agent.tools.get_temporal_priority.fetch_nvd_data")
    def test_full_pipeline(self, mock_nvd, mock_epss, mock_kev):
        mock_nvd.return_value = {
            "cvss_score": 9.8,
            "published_date": "2024-03-29T00:00:00.000",
            "has_patch": False,
        }
        mock_epss.return_value = {
            "current_epss": 0.95,
            "spike_detected": True,
            "max_jump": 0.8,
            "days_since_spike": 2,
        }
        mock_kev.return_value = {"in_kev": True, "date_added": "2024-03-30"}

        result = compute_temporal_priority("CVE-2024-3094")
        assert result["cve_id"] == "CVE-2024-3094"
        assert result["score"] >= 80
        assert result["label"] in ("CRITICAL", "HIGH")
        assert result["signals_available"] == 6

    @patch("manus_agent.tools.get_temporal_priority.fetch_kev_status")
    @patch("manus_agent.tools.get_temporal_priority.fetch_epss_data")
    @patch("manus_agent.tools.get_temporal_priority.fetch_nvd_data")
    def test_all_sources_fail(self, mock_nvd, mock_epss, mock_kev):
        mock_nvd.return_value = {"cvss_score": None, "published_date": None, "has_patch": None}
        mock_epss.return_value = {
            "current_epss": None,
            "spike_detected": None,
            "max_jump": None,
            "days_since_spike": None,
        }
        mock_kev.return_value = {"in_kev": None, "date_added": None}

        result = compute_temporal_priority("CVE-9999-0000")
        assert result["cve_id"] == "CVE-9999-0000"
        assert 0 <= result["score"] <= 100
        assert result["signals_available"] == 0

    @patch("manus_agent.tools.get_temporal_priority.fetch_kev_status")
    @patch("manus_agent.tools.get_temporal_priority.fetch_epss_data")
    @patch("manus_agent.tools.get_temporal_priority.fetch_nvd_data")
    def test_case_normalization(self, mock_nvd, mock_epss, mock_kev):
        mock_nvd.return_value = {"cvss_score": 5.0, "published_date": None, "has_patch": None}
        mock_epss.return_value = {
            "current_epss": None,
            "spike_detected": None,
            "max_jump": None,
            "days_since_spike": None,
        }
        mock_kev.return_value = {"in_kev": None, "date_added": None}

        result = compute_temporal_priority("  cve-2024-3094  ")
        assert result["cve_id"] == "CVE-2024-3094"


# ===========================================================================
# Handler integration tests
# ===========================================================================


class TestHandler:
    """Test the Strands tool handler."""

    @patch("manus_agent.tools.get_temporal_priority.compute_temporal_priority")
    def test_success_response(self, mock_compute):
        mock_compute.return_value = {
            "cve_id": "CVE-2024-3094",
            "score": 92.0,
            "label": "CRITICAL",
            "signals_available": 6,
            "signals_total": 6,
            "breakdown": {},
        }
        tool_use = {"toolUseId": "h-1", "input": {"cve_id": "CVE-2024-3094"}}
        result = get_temporal_priority(tool_use)
        assert result["status"] == "success"
        assert result["toolUseId"] == "h-1"
        data = json.loads(result["content"][0]["text"])
        assert data["score"] == 92.0

    @patch("manus_agent.tools.get_temporal_priority.compute_temporal_priority")
    def test_whitespace_handling(self, mock_compute):
        mock_compute.return_value = {"cve_id": "CVE-2024-3094", "score": 50.0}
        tool_use = {"toolUseId": "h-2", "input": {"cve_id": "  CVE-2024-3094  "}}
        result = get_temporal_priority(tool_use)
        assert result["status"] == "success"

    def test_integer_input_rejected(self):
        tool_use = {"toolUseId": "h-3", "input": {"cve_id": 12345}}
        result = get_temporal_priority(tool_use)
        assert result["status"] == "error"


# ===========================================================================
# CLI subcommand tests
# ===========================================================================


class TestCliSubcommand:
    """Test the manus-agent temporal-priority CLI dispatch."""

    @patch("manus_agent.tools.get_temporal_priority.compute_temporal_priority")
    def test_text_output(self, mock_compute, capsys):
        from manus_agent.cli import _run_temporal_priority

        mock_compute.return_value = {
            "cve_id": "CVE-2024-3094",
            "score": 85.0,
            "label": "CRITICAL",
            "signals_available": 6,
            "signals_total": 6,
            "breakdown": {
                "cvss": {"raw": 9.8, "points": 24.5, "max": 25},
                "epss": {"raw": 0.95, "points": 24.4, "max": 25},
                "spike": {"detected": True, "days_since": 2, "points": 12.0, "max": 15},
                "kev": {"in_kev": True, "points": 20.0, "max": 20},
                "patch": {"has_patch": False, "points": 10.0, "max": 10},
                "age": {"days": 30, "points": 4.5, "max": 5},
            },
        }
        rc = _run_temporal_priority(["CVE-2024-3094"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "CVE-2024-3094" in captured.out
        assert "85.0" in captured.out

    @patch("manus_agent.tools.get_temporal_priority.compute_temporal_priority")
    def test_json_output(self, mock_compute, capsys):
        from manus_agent.cli import _run_temporal_priority

        mock_compute.return_value = {
            "cve_id": "CVE-2024-3094",
            "score": 85.0,
            "label": "CRITICAL",
        }
        rc = _run_temporal_priority(["CVE-2024-3094", "--output", "json"])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["score"] == 85.0

    def test_invalid_cve_format(self, capsys):
        from manus_agent.cli import _run_temporal_priority

        rc = _run_temporal_priority(["not-a-cve"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Invalid CVE format" in captured.err

    def test_empty_cve_triggers_error(self):
        from manus_agent.cli import _run_temporal_priority

        with pytest.raises(SystemExit):
            _run_temporal_priority([])

    def test_subcommand_registered(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "temporal-priority" in _SUBCOMMANDS


# ===========================================================================
# Safe float helper tests
# ===========================================================================


class TestSafeFloat:
    """Test the _safe_float helper."""

    def test_valid_string(self):
        assert _safe_float("0.95") == 0.95

    def test_valid_int(self):
        assert _safe_float(5) == 5.0

    def test_none(self):
        assert _safe_float(None) is None

    def test_invalid_string(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_cvss_zero(self):
        result = compute_score(
            cvss_score=0.0,
            current_epss=None,
            spike_detected=None,
            days_since_spike=None,
            in_kev=None,
            has_patch=None,
            published_date=None,
        )
        assert result["breakdown"]["cvss"]["points"] == 0

    def test_cvss_ten(self):
        result = compute_score(
            cvss_score=10.0,
            current_epss=None,
            spike_detected=None,
            days_since_spike=None,
            in_kev=None,
            has_patch=None,
            published_date=None,
        )
        assert result["breakdown"]["cvss"]["points"] == 25.0

    def test_epss_at_one(self):
        result = compute_score(
            cvss_score=None,
            current_epss=1.0,
            spike_detected=None,
            days_since_spike=None,
            in_kev=None,
            has_patch=None,
            published_date=None,
        )
        assert result["breakdown"]["epss"]["points"] == 25.0

    def test_epss_above_one_clamped(self):
        """EPSS > 1.0 should be clamped."""
        result = compute_score(
            cvss_score=None,
            current_epss=1.5,
            spike_detected=None,
            days_since_spike=None,
            in_kev=None,
            has_patch=None,
            published_date=None,
        )
        assert result["breakdown"]["epss"]["points"] == 25.0

    def test_spike_without_days(self):
        """Spike detected but days_since unknown → 75% points."""
        result = compute_score(
            cvss_score=None,
            current_epss=None,
            spike_detected=True,
            days_since_spike=None,
            in_kev=None,
            has_patch=None,
            published_date=None,
        )
        expected = 0.75 * 15  # _W_SPIKE default is 15
        assert result["breakdown"]["spike"]["points"] == pytest.approx(expected, abs=0.1)

    def test_very_old_cve_low_age_score(self):
        """Very old CVE should have minimal age contribution."""
        result = compute_score(
            cvss_score=None,
            current_epss=None,
            spike_detected=None,
            days_since_spike=None,
            in_kev=None,
            has_patch=None,
            published_date="2000-01-01T00:00:00.000",
        )
        assert result["breakdown"]["age"]["points"] < 0.1
