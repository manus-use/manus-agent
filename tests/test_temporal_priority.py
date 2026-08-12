"""
Comprehensive test suite for temporal_priority tool.

Tests cover:
- TOOL_SPEC contract (Strands SDK interface)
- Input validation
- HTTP helper retry/back-off
- Individual signal scoring functions
- Composite scoring
- Tool entry point
- CLI subcommand
- Edge cases
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.temporal_priority import (
    _SPIKE_THRESHOLD,
    TOOL_SPEC,
    _get_with_retry,
    compute_temporal_priority,
    score_age,
    score_cvss,
    score_epss_current,
    score_epss_spike,
    score_kev,
    score_patch_availability,
    temporal_priority,
)

# ---------------------------------------------------------------------------
# TOOL_SPEC contract tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Verify the TOOL_SPEC satisfies Strands SDK requirements."""

    def test_has_name(self):
        assert TOOL_SPEC["name"] == "temporal_priority"

    def test_has_description(self):
        assert isinstance(TOOL_SPEC["description"], str)
        assert len(TOOL_SPEC["description"]) > 20

    def test_has_input_schema(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_cve_id_property_type(self):
        prop = TOOL_SPEC["inputSchema"]["json"]["properties"]["cve_id"]
        assert prop["type"] == "string"
        assert "description" in prop


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Verify the tool entry point validates input."""

    def test_empty_cve_id_returns_error(self):
        tool_use = {"toolUseId": "test-1", "input": {"cve_id": ""}}
        result = temporal_priority(tool_use)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_none_cve_id_returns_error(self):
        tool_use = {"toolUseId": "test-2", "input": {"cve_id": None}}
        result = temporal_priority(tool_use)
        assert result["status"] == "error"

    def test_whitespace_only_cve_id_returns_error(self):
        tool_use = {"toolUseId": "test-3", "input": {"cve_id": "   "}}
        result = temporal_priority(tool_use)
        assert result["status"] == "error"

    def test_missing_cve_id_returns_error(self):
        tool_use = {"toolUseId": "test-4", "input": {}}
        result = temporal_priority(tool_use)
        assert result["status"] == "error"

    def test_non_string_cve_id_returns_error(self):
        tool_use = {"toolUseId": "test-5", "input": {"cve_id": 12345}}
        result = temporal_priority(tool_use)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# HTTP helper tests
# ---------------------------------------------------------------------------


class TestGetWithRetry:
    """Test the _get_with_retry HTTP helper."""

    @patch("manus_agent.tools.temporal_priority.requests.get")
    def test_success_first_try(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _get_with_retry("https://example.com/api")
        assert result == {"result": "ok"}
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.temporal_priority.time.sleep")
    @patch("manus_agent.tools.temporal_priority.requests.get")
    def test_retry_on_429(self, mock_get, mock_sleep):
        fail_resp = MagicMock()
        fail_resp.status_code = 429

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"data": "yes"}
        ok_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [fail_resp, ok_resp]
        result = _get_with_retry("https://example.com/api")
        assert result == {"data": "yes"}
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    @patch("manus_agent.tools.temporal_priority.time.sleep")
    @patch("manus_agent.tools.temporal_priority.requests.get")
    def test_retry_on_503(self, mock_get, mock_sleep):
        fail_resp = MagicMock()
        fail_resp.status_code = 503

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"ok": True}
        ok_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [fail_resp, ok_resp]
        result = _get_with_retry("https://example.com/api")
        assert result == {"ok": True}

    @patch("manus_agent.tools.temporal_priority._MAX_RETRIES", 2)
    @patch("manus_agent.tools.temporal_priority.time.sleep")
    @patch("manus_agent.tools.temporal_priority.requests.get")
    def test_exhausted_retries_raises(self, mock_get, mock_sleep):
        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("timeout")
        with pytest.raises(req.exceptions.ConnectionError):
            _get_with_retry("https://example.com/api")
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.temporal_priority._MAX_RETRIES", 3)
    @patch("manus_agent.tools.temporal_priority.time.sleep")
    @patch("manus_agent.tools.temporal_priority.requests.get")
    def test_non_retryable_status_exhausts_retries(self, mock_get, mock_sleep):
        """Non-retryable HTTP errors caught by the except block still retry."""
        import requests as req

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = req.exceptions.HTTPError("404")
        mock_get.return_value = mock_resp

        with pytest.raises(req.exceptions.HTTPError):
            _get_with_retry("https://example.com/api")
        # The HTTPError is caught by the generic except, so all retries are used
        assert mock_get.call_count == 3


# ---------------------------------------------------------------------------
# Signal scoring tests — CVSS
# ---------------------------------------------------------------------------


class TestScoreCvss:
    """Test CVSS signal scoring."""

    def test_cvss31_extraction(self):
        nvd = {"vulnerabilities": [{"cve": {"metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 9.8}}]}}}]}
        norm, raw = score_cvss(nvd)
        assert raw == 9.8
        assert norm == pytest.approx(0.98, abs=0.001)

    def test_cvss30_fallback(self):
        nvd = {"vulnerabilities": [{"cve": {"metrics": {"cvssMetricV30": [{"cvssData": {"baseScore": 7.5}}]}}}]}
        norm, raw = score_cvss(nvd)
        assert raw == 7.5
        assert norm == pytest.approx(0.75, abs=0.001)

    def test_cvss2_fallback(self):
        nvd = {"vulnerabilities": [{"cve": {"metrics": {"cvssMetricV2": [{"cvssData": {"baseScore": 5.0}}]}}}]}
        norm, raw = score_cvss(nvd)
        assert raw == 5.0
        assert norm == pytest.approx(0.5, abs=0.001)

    def test_empty_vulnerabilities(self):
        norm, raw = score_cvss({"vulnerabilities": []})
        assert norm == 0.0
        assert raw is None

    def test_no_metrics(self):
        nvd = {"vulnerabilities": [{"cve": {"metrics": {}}}]}
        norm, raw = score_cvss(nvd)
        assert norm == 0.0
        assert raw is None

    def test_perfect_10(self):
        nvd = {"vulnerabilities": [{"cve": {"metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 10.0}}]}}}]}
        norm, raw = score_cvss(nvd)
        assert norm == 1.0
        assert raw == 10.0


# ---------------------------------------------------------------------------
# Signal scoring tests — EPSS current
# ---------------------------------------------------------------------------


class TestScoreEpssCurrent:
    """Test EPSS current score extraction."""

    def test_top_level_epss(self):
        data = {"data": [{"epss": "0.95432"}]}
        norm, raw = score_epss_current(data)
        assert raw == pytest.approx(0.95432, abs=0.0001)
        assert norm == pytest.approx(0.95432, abs=0.0001)

    def test_fallback_to_timeseries(self):
        data = {"data": [{"time-series": [{"epss": "0.123", "date": "2025-01-01"}]}]}
        norm, raw = score_epss_current(data)
        assert raw == pytest.approx(0.123, abs=0.001)

    def test_empty_data(self):
        norm, raw = score_epss_current({"data": []})
        assert norm == 0.0
        assert raw is None

    def test_no_data_key(self):
        norm, raw = score_epss_current({})
        assert norm == 0.0
        assert raw is None

    def test_zero_epss(self):
        data = {"data": [{"epss": "0.0"}]}
        norm, raw = score_epss_current(data)
        assert norm == 0.0
        assert raw == 0.0


# ---------------------------------------------------------------------------
# Signal scoring tests — EPSS spike
# ---------------------------------------------------------------------------


class TestScoreEpssSpike:
    """Test EPSS spike detection and scoring."""

    def test_no_data_returns_zero(self):
        score, details = score_epss_spike({"data": []})
        assert score == 0.0
        assert details["spike_detected"] is False

    def test_single_point_no_spike(self):
        data = {"data": [{"time-series": [{"epss": "0.5", "date": "2025-01-01"}]}]}
        score, details = score_epss_spike(data)
        assert score == 0.0
        assert details["spike_detected"] is False

    def test_no_significant_jump(self):
        data = {
            "data": [
                {
                    "time-series": [
                        {"epss": "0.01", "date": "2025-01-01"},
                        {"epss": "0.02", "date": "2025-01-02"},
                        {"epss": "0.025", "date": "2025-01-03"},
                    ]
                }
            ]
        }
        score, details = score_epss_spike(data)
        assert score == 0.0
        assert details["spike_detected"] is False
        assert details["max_jump"] < _SPIKE_THRESHOLD

    @patch("manus_agent.tools.temporal_priority.datetime")
    def test_recent_spike_high_score(self, mock_dt):
        # Simulate a spike that happened "today"
        mock_dt.now.return_value = datetime(2025, 7, 10, tzinfo=timezone.utc)
        mock_dt.strptime = datetime.strptime

        data = {
            "data": [
                {
                    "time-series": [
                        {"epss": "0.01", "date": "2025-07-08"},
                        {"epss": "0.02", "date": "2025-07-09"},
                        {"epss": "0.12", "date": "2025-07-10"},  # +0.10 spike
                    ]
                }
            ]
        }
        score, details = score_epss_spike(data)
        assert details["spike_detected"] is True
        assert details["max_jump"] == pytest.approx(0.10, abs=0.001)
        # Recent spike → high score (close to 1.0)
        assert score > 0.8

    @patch("manus_agent.tools.temporal_priority.datetime")
    def test_old_spike_decayed(self, mock_dt):
        # Simulate a spike from 60 days ago
        mock_dt.now.return_value = datetime(2025, 9, 8, tzinfo=timezone.utc)
        mock_dt.strptime = datetime.strptime

        data = {
            "data": [
                {
                    "time-series": [
                        {"epss": "0.01", "date": "2025-07-09"},
                        {"epss": "0.12", "date": "2025-07-10"},  # +0.11 spike, 60 days ago
                    ]
                }
            ]
        }
        score, details = score_epss_spike(data)
        assert details["spike_detected"] is True
        # 60 days with 14-day half-life → very decayed
        assert score < 0.1


# ---------------------------------------------------------------------------
# Signal scoring tests — KEV
# ---------------------------------------------------------------------------


class TestScoreKev:
    """Test CISA KEV membership scoring."""

    def test_in_kev(self):
        score, in_kev = score_kev("CVE-2024-3094", {"CVE-2024-3094", "CVE-2021-44228"})
        assert score == 1.0
        assert in_kev is True

    def test_not_in_kev(self):
        score, in_kev = score_kev("CVE-2099-9999", {"CVE-2024-3094"})
        assert score == 0.0
        assert in_kev is False

    def test_case_insensitive(self):
        score, in_kev = score_kev("cve-2024-3094", {"CVE-2024-3094"})
        assert score == 1.0
        assert in_kev is True

    def test_empty_kev_set(self):
        score, in_kev = score_kev("CVE-2024-3094", set())
        assert score == 0.0
        assert in_kev is False


# ---------------------------------------------------------------------------
# Signal scoring tests — Patch availability
# ---------------------------------------------------------------------------


class TestScorePatchAvailability:
    """Test patch availability scoring."""

    def test_patch_available(self):
        nvd = {
            "vulnerabilities": [
                {
                    "cve": {
                        "references": [
                            {"url": "https://example.com/fix", "tags": ["Patch"]},
                        ]
                    }
                }
            ]
        }
        score, has_patch = score_patch_availability(nvd)
        assert score == 0.0  # patch present → lower urgency
        assert has_patch is True

    def test_no_patch(self):
        nvd = {
            "vulnerabilities": [
                {
                    "cve": {
                        "references": [
                            {"url": "https://example.com/advisory", "tags": ["Third Party Advisory"]},
                        ]
                    }
                }
            ]
        }
        score, has_patch = score_patch_availability(nvd)
        assert score == 1.0  # no patch → higher urgency
        assert has_patch is False

    def test_empty_references(self):
        nvd = {"vulnerabilities": [{"cve": {"references": []}}]}
        score, has_patch = score_patch_availability(nvd)
        assert score == 1.0
        assert has_patch is False

    def test_empty_vulnerabilities_returns_middle(self):
        score, has_patch = score_patch_availability({"vulnerabilities": []})
        assert score == 0.5
        assert has_patch is False

    def test_multiple_refs_one_patch(self):
        nvd = {
            "vulnerabilities": [
                {
                    "cve": {
                        "references": [
                            {"url": "https://example.com/cve", "tags": ["Vendor Advisory"]},
                            {"url": "https://github.com/fix", "tags": ["Patch", "Vendor Advisory"]},
                        ]
                    }
                }
            ]
        }
        score, has_patch = score_patch_availability(nvd)
        assert score == 0.0
        assert has_patch is True


# ---------------------------------------------------------------------------
# Signal scoring tests — Age
# ---------------------------------------------------------------------------


class TestScoreAge:
    """Test CVE age scoring."""

    @patch("manus_agent.tools.temporal_priority.datetime")
    def test_brand_new_cve(self, mock_dt):
        mock_dt.now.return_value = datetime(2025, 7, 10, 12, 0, tzinfo=timezone.utc)
        mock_dt.fromisoformat = datetime.fromisoformat

        nvd = {"vulnerabilities": [{"cve": {"published": "2025-07-10T10:00:00.000"}}]}
        score, age_days = score_age(nvd)
        assert age_days == 0
        assert score == pytest.approx(1.0, abs=0.01)

    @patch("manus_agent.tools.temporal_priority.datetime")
    def test_90_day_old_cve(self, mock_dt):
        mock_dt.now.return_value = datetime(2025, 10, 8, 12, 0, tzinfo=timezone.utc)
        mock_dt.fromisoformat = datetime.fromisoformat

        nvd = {"vulnerabilities": [{"cve": {"published": "2025-07-10T12:00:00.000"}}]}
        score, age_days = score_age(nvd)
        assert age_days == pytest.approx(90, abs=2)
        # 90-day half-life → ~0.5
        assert score == pytest.approx(0.5, abs=0.05)

    @patch("manus_agent.tools.temporal_priority.datetime")
    def test_year_old_cve(self, mock_dt):
        mock_dt.now.return_value = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
        mock_dt.fromisoformat = datetime.fromisoformat

        nvd = {"vulnerabilities": [{"cve": {"published": "2025-07-10T12:00:00.000"}}]}
        score, age_days = score_age(nvd)
        assert age_days == pytest.approx(365, abs=2)
        # Very old → very low
        assert score < 0.1

    def test_no_published_date(self):
        nvd = {"vulnerabilities": [{"cve": {}}]}
        score, age_days = score_age(nvd)
        assert score == 0.5
        assert age_days is None

    def test_empty_vulnerabilities(self):
        score, age_days = score_age({"vulnerabilities": []})
        assert score == 0.5
        assert age_days is None


# ---------------------------------------------------------------------------
# Composite scoring tests
# ---------------------------------------------------------------------------


class TestComputeTemporalPriority:
    """Test the composite temporal priority scorer."""

    def test_critical_cve_all_signals_high(self):
        """A CVE with high CVSS, high EPSS, in KEV, no patch, new → critical."""
        nvd = {
            "vulnerabilities": [
                {
                    "cve": {
                        "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 10.0}}]},
                        "references": [],
                        "published": datetime.now(timezone.utc).isoformat(),
                    }
                }
            ]
        }
        epss = {"data": [{"epss": "0.97", "time-series": []}]}
        kev_set = {"CVE-2024-9999"}

        result = compute_temporal_priority("CVE-2024-9999", nvd_data=nvd, epss_data=epss, kev_set=kev_set)
        assert result["score"] >= 80
        assert result["label"] == "CRITICAL"
        assert result["cve_id"] == "CVE-2024-9999"

    def test_low_risk_cve(self):
        """A CVE with low CVSS, low EPSS, not in KEV, patch available, old → low."""
        nvd = {
            "vulnerabilities": [
                {
                    "cve": {
                        "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 2.0}}]},
                        "references": [{"url": "https://fix.com", "tags": ["Patch"]}],
                        "published": "2020-01-01T00:00:00.000",
                    }
                }
            ]
        }
        epss = {"data": [{"epss": "0.001", "time-series": []}]}
        kev_set = set()

        result = compute_temporal_priority("CVE-2020-0001", nvd_data=nvd, epss_data=epss, kev_set=kev_set)
        assert result["score"] < 30
        assert result["label"] in ("LOW", "INFORMATIONAL")

    def test_medium_risk_cve(self):
        """A CVE with moderate signals → medium."""
        nvd = {
            "vulnerabilities": [
                {
                    "cve": {
                        "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 7.0}}]},
                        "references": [],
                        "published": "2025-03-01T00:00:00.000",
                    }
                }
            ]
        }
        epss = {"data": [{"epss": "0.3", "time-series": []}]}
        kev_set = set()

        result = compute_temporal_priority("CVE-2025-1234", nvd_data=nvd, epss_data=epss, kev_set=kev_set)
        assert 30 <= result["score"] <= 70

    def test_score_bounded_0_100(self):
        """Score should always be 0–100."""
        nvd = {"vulnerabilities": []}
        epss = {"data": []}
        kev_set = set()

        result = compute_temporal_priority("CVE-0000-0000", nvd_data=nvd, epss_data=epss, kev_set=kev_set)
        assert 0 <= result["score"] <= 100

    def test_kev_membership_significant_boost(self):
        """Being in KEV should significantly boost the score."""
        nvd = {
            "vulnerabilities": [
                {
                    "cve": {
                        "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 5.0}}]},
                        "references": [],
                        "published": "2024-06-01T00:00:00.000",
                    }
                }
            ]
        }
        epss = {"data": [{"epss": "0.1", "time-series": []}]}

        result_no_kev = compute_temporal_priority("CVE-2024-5555", nvd_data=nvd, epss_data=epss, kev_set=set())
        result_with_kev = compute_temporal_priority(
            "CVE-2024-5555", nvd_data=nvd, epss_data=epss, kev_set={"CVE-2024-5555"}
        )

        assert result_with_kev["score"] > result_no_kev["score"]
        # KEV weight is 0.20 → at least 15-point boost
        assert result_with_kev["score"] - result_no_kev["score"] >= 15

    def test_result_has_required_keys(self):
        nvd = {"vulnerabilities": []}
        epss = {"data": []}
        result = compute_temporal_priority("CVE-2024-0001", nvd_data=nvd, epss_data=epss, kev_set=set())

        assert "cve_id" in result
        assert "score" in result
        assert "label" in result
        assert "signals" in result
        assert "interpretation" in result

    def test_signals_have_weights(self):
        nvd = {"vulnerabilities": []}
        epss = {"data": []}
        result = compute_temporal_priority("CVE-2024-0001", nvd_data=nvd, epss_data=epss, kev_set=set())

        signals = result["signals"]
        for key in ("cvss", "epss_current", "epss_spike", "cisa_kev", "patch_availability", "age"):
            assert key in signals
            assert "weight" in signals[key]
            assert "normalised" in signals[key]

    def test_cve_id_uppercased(self):
        nvd = {"vulnerabilities": []}
        epss = {"data": []}
        result = compute_temporal_priority("cve-2024-0001", nvd_data=nvd, epss_data=epss, kev_set=set())
        assert result["cve_id"] == "CVE-2024-0001"


# ---------------------------------------------------------------------------
# Tool entry point tests
# ---------------------------------------------------------------------------


class TestToolEntryPoint:
    """Test the Strands tool entry point function."""

    @patch("manus_agent.tools.temporal_priority.compute_temporal_priority")
    def test_success(self, mock_compute):
        mock_compute.return_value = {
            "cve_id": "CVE-2024-3094",
            "score": 85.5,
            "label": "CRITICAL",
            "signals": {},
            "interpretation": "test",
        }
        tool_use = {"toolUseId": "t1", "input": {"cve_id": "CVE-2024-3094"}}
        result = temporal_priority(tool_use)
        assert result["status"] == "success"
        assert result["toolUseId"] == "t1"
        content = json.loads(result["content"][0]["text"])
        assert content["score"] == 85.5

    @patch("manus_agent.tools.temporal_priority.compute_temporal_priority")
    def test_exception_returns_error(self, mock_compute):
        mock_compute.side_effect = RuntimeError("API down")
        tool_use = {"toolUseId": "t2", "input": {"cve_id": "CVE-2024-3094"}}
        result = temporal_priority(tool_use)
        assert result["status"] == "error"
        assert "API down" in result["content"][0]["text"]

    @patch("manus_agent.tools.temporal_priority.compute_temporal_priority")
    def test_tool_use_id_preserved(self, mock_compute):
        mock_compute.return_value = {"cve_id": "X", "score": 50, "label": "MEDIUM", "signals": {}, "interpretation": ""}
        tool_use = {"toolUseId": "unique-id-123", "input": {"cve_id": "CVE-2024-1111"}}
        result = temporal_priority(tool_use)
        assert result["toolUseId"] == "unique-id-123"

    @patch("manus_agent.tools.temporal_priority.compute_temporal_priority")
    def test_cve_id_trimmed(self, mock_compute):
        mock_compute.return_value = {"cve_id": "X", "score": 50, "label": "MEDIUM", "signals": {}, "interpretation": ""}
        tool_use = {"toolUseId": "t3", "input": {"cve_id": "  CVE-2024-1111  "}}
        result = temporal_priority(tool_use)
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# CLI subcommand tests
# ---------------------------------------------------------------------------


class TestCliSubcommand:
    """Test the temporal-priority CLI subcommand."""

    @patch("manus_agent.tools.temporal_priority.compute_temporal_priority")
    def test_text_output(self, mock_compute, capsys):
        from manus_agent.cli import _run_temporal_priority

        mock_compute.return_value = {
            "cve_id": "CVE-2024-3094",
            "score": 72.3,
            "label": "HIGH",
            "signals": {
                "cvss": {"normalised": 0.98, "raw_score": 9.8, "weight": 0.25},
                "epss_current": {"normalised": 0.5, "raw_epss": 0.5, "weight": 0.25},
                "epss_spike": {"normalised": 0.0, "spike_detected": False, "max_jump": 0, "weight": 0.15},
                "cisa_kev": {"normalised": 1.0, "in_kev": True, "weight": 0.20},
                "patch_availability": {"normalised": 0.0, "has_patch": True, "weight": 0.05},
                "age": {"normalised": 0.8, "age_days": 30, "weight": 0.10},
            },
            "interpretation": "CVE-2024-3094 has a temporal priority of 72.3/100 (HIGH).",
        }

        rc = _run_temporal_priority(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "72.3/100" in out
        assert "HIGH" in out
        assert "CVE-2024-3094" in out

    @patch("manus_agent.tools.temporal_priority.compute_temporal_priority")
    def test_json_output(self, mock_compute, capsys):
        from manus_agent.cli import _run_temporal_priority

        mock_compute.return_value = {
            "cve_id": "CVE-2024-3094",
            "score": 72.3,
            "label": "HIGH",
            "signals": {},
            "interpretation": "test",
        }

        rc = _run_temporal_priority(["CVE-2024-3094", "--output", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["score"] == 72.3

    @patch("manus_agent.tools.temporal_priority.compute_temporal_priority")
    def test_exception_returns_nonzero(self, mock_compute, capsys):
        from manus_agent.cli import _run_temporal_priority

        mock_compute.side_effect = RuntimeError("Network error")
        rc = _run_temporal_priority(["CVE-2024-3094"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Network error" in err

    def test_help_flag(self, capsys):
        from manus_agent.cli import _run_temporal_priority

        with pytest.raises(SystemExit) as exc_info:
            _run_temporal_priority(["--help"])
        assert exc_info.value.code == 0

    def test_no_args_errors(self):
        from manus_agent.cli import _run_temporal_priority

        with pytest.raises(SystemExit) as exc_info:
            _run_temporal_priority([])
        assert exc_info.value.code != 0

    def test_subcommand_registered(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "temporal-priority" in _SUBCOMMANDS


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_all_signals_zero(self):
        """When all data is missing, score should be bounded and not crash."""
        result = compute_temporal_priority("CVE-0000-0000", nvd_data={}, epss_data={}, kev_set=set())
        assert 0 <= result["score"] <= 100
        assert result["label"] in ("INFORMATIONAL", "LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_nvd_missing_cve_key(self):
        """NVD response with malformed structure."""
        nvd = {"vulnerabilities": [{"cve": {}}]}
        result = compute_temporal_priority("CVE-2024-0001", nvd_data=nvd, epss_data={"data": []}, kev_set=set())
        assert 0 <= result["score"] <= 100

    def test_epss_data_with_empty_timeseries(self):
        data = {"data": [{"epss": "0.5", "time-series": []}]}
        score, details = score_epss_spike(data)
        assert score == 0.0
        assert details["spike_detected"] is False

    def test_score_labels_thresholds(self):
        """Verify label thresholds match documented behaviour."""
        nvd = {"vulnerabilities": []}
        epss = {"data": []}

        # Force various score ranges by manipulating weights
        with patch("manus_agent.tools.temporal_priority._W_CVSS", 0):
            with patch("manus_agent.tools.temporal_priority._W_EPSS", 0):
                with patch("manus_agent.tools.temporal_priority._W_SPIKE", 0):
                    with patch("manus_agent.tools.temporal_priority._W_KEV", 0):
                        with patch("manus_agent.tools.temporal_priority._W_PATCH", 0):
                            with patch("manus_agent.tools.temporal_priority._W_AGE", 0):
                                result = compute_temporal_priority(
                                    "CVE-0000-0000", nvd_data=nvd, epss_data=epss, kev_set=set()
                                )
                                assert result["score"] == 0.0
                                assert result["label"] == "INFORMATIONAL"

    @patch("manus_agent.tools.temporal_priority._fetch_nvd")
    @patch("manus_agent.tools.temporal_priority._fetch_epss")
    @patch("manus_agent.tools.temporal_priority._fetch_kev")
    def test_fetch_failures_graceful(self, mock_kev, mock_epss, mock_nvd):
        """If all fetches fail, the function still returns a valid result."""
        mock_nvd.return_value = {}
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        result = compute_temporal_priority("CVE-2024-9999")
        assert 0 <= result["score"] <= 100
        assert "cve_id" in result

    def test_interpretation_contains_cve_id(self):
        nvd = {"vulnerabilities": []}
        epss = {"data": []}
        result = compute_temporal_priority("CVE-2024-1234", nvd_data=nvd, epss_data=epss, kev_set=set())
        assert "CVE-2024-1234" in result["interpretation"]
