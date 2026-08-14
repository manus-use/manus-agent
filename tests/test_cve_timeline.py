"""Comprehensive test suite for get_cve_timeline module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.get_cve_timeline import (
    TOOL_SPEC,
    _compute_summary,
    _days_between,
    _extract_cvss,
    _extract_patch_references,
    _fetch_epss_events,
    _fetch_github_advisory_dates,
    _fetch_kev_date,
    _fetch_nvd_events,
    _get_with_retry,
    _parse_iso,
    build_timeline,
    format_timeline_json,
    format_timeline_text,
    handler,
)

# ---------------------------------------------------------------------------
# TOOL_SPEC contract tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Verify TOOL_SPEC conforms to Strands SDK expectations."""

    def test_has_name(self):
        assert TOOL_SPEC["name"] == "get_cve_timeline"

    def test_has_description(self):
        assert "timeline" in TOOL_SPEC["description"].lower()

    def test_input_schema_requires_cve_id(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_cve_id_is_string(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["properties"]["cve_id"]["type"] == "string"


# ---------------------------------------------------------------------------
# _parse_iso tests
# ---------------------------------------------------------------------------


class TestParseIso:
    """Test date parsing utility."""

    def test_simple_date(self):
        assert _parse_iso("2021-12-09") == "2021-12-09"

    def test_iso_with_z(self):
        assert _parse_iso("2021-12-09T14:30:00Z") == "2021-12-09"

    def test_iso_with_microseconds_z(self):
        assert _parse_iso("2021-12-09T14:30:00.123456Z") == "2021-12-09"

    def test_iso_with_timezone(self):
        assert _parse_iso("2021-12-09T14:30:00+00:00") == "2021-12-09"

    def test_iso_no_z(self):
        assert _parse_iso("2021-12-09T14:30:00") == "2021-12-09"

    def test_empty_string(self):
        assert _parse_iso("") == ""

    def test_fallback_first_10(self):
        assert _parse_iso("2021-12-09-some-garbage") == "2021-12-09"

    def test_unparseable(self):
        result = _parse_iso("not-a-date")
        assert result == "not-a-date"


# ---------------------------------------------------------------------------
# _days_between tests
# ---------------------------------------------------------------------------


class TestDaysBetween:
    """Test day-counting helper."""

    def test_same_date(self):
        assert _days_between("2021-12-09", "2021-12-09") == 0

    def test_one_day_forward(self):
        assert _days_between("2021-12-09", "2021-12-10") == 1

    def test_one_day_backward(self):
        assert _days_between("2021-12-10", "2021-12-09") == 1

    def test_large_delta(self):
        assert _days_between("2021-01-01", "2021-12-31") == 364

    def test_invalid_date(self):
        assert _days_between("not-a-date", "2021-12-09") == -1

    def test_none_input(self):
        assert _days_between(None, "2021-12-09") == -1


# ---------------------------------------------------------------------------
# _extract_cvss tests
# ---------------------------------------------------------------------------


class TestExtractCvss:
    """Test CVSS score extraction from NVD data."""

    def test_cvss_v31(self):
        cve_data = {
            "metrics": {
                "cvssMetricV31": [{"cvssData": {"baseScore": 10.0}}],
            }
        }
        score, version = _extract_cvss(cve_data)
        assert score == 10.0
        assert version == "3.1"

    def test_cvss_v30_fallback(self):
        cve_data = {
            "metrics": {
                "cvssMetricV30": [{"cvssData": {"baseScore": 7.5}}],
            }
        }
        score, version = _extract_cvss(cve_data)
        assert score == 7.5
        assert version == "3.0"

    def test_cvss_v2_fallback(self):
        cve_data = {
            "metrics": {
                "cvssMetricV2": [{"cvssData": {"baseScore": 5.0}}],
            }
        }
        score, version = _extract_cvss(cve_data)
        assert score == 5.0
        assert version == "2.0"

    def test_no_metrics(self):
        cve_data = {"metrics": {}}
        score, version = _extract_cvss(cve_data)
        assert score is None
        assert version == ""

    def test_prefers_v31_over_v2(self):
        cve_data = {
            "metrics": {
                "cvssMetricV31": [{"cvssData": {"baseScore": 9.8}}],
                "cvssMetricV2": [{"cvssData": {"baseScore": 7.0}}],
            }
        }
        score, version = _extract_cvss(cve_data)
        assert score == 9.8
        assert version == "3.1"

    def test_empty_metric_list(self):
        cve_data = {"metrics": {"cvssMetricV31": []}}
        score, version = _extract_cvss(cve_data)
        assert score is None


# ---------------------------------------------------------------------------
# _extract_patch_references tests
# ---------------------------------------------------------------------------


class TestExtractPatchReferences:
    """Test patch reference extraction from NVD references."""

    def test_commit_url(self):
        cve_data = {"references": [{"url": "https://github.com/org/repo/commit/abc123def", "tags": []}]}
        events = _extract_patch_references(cve_data)
        assert len(events) == 1
        assert events[0]["event"] == "Patch commit published"

    def test_release_url(self):
        cve_data = {"references": [{"url": "https://github.com/org/repo/releases/tag/v1.2.3", "tags": []}]}
        events = _extract_patch_references(cve_data)
        assert len(events) == 1
        assert events[0]["event"] == "Release published"

    def test_patch_tag(self):
        cve_data = {"references": [{"url": "https://example.com/fix.patch", "tags": ["Patch"]}]}
        events = _extract_patch_references(cve_data)
        assert len(events) == 1
        assert events[0]["event"] == "Patch published"

    def test_deduplicates_urls(self):
        cve_data = {
            "references": [
                {"url": "https://github.com/org/repo/commit/abc123", "tags": ["Patch"]},
                {"url": "https://github.com/org/repo/commit/abc123", "tags": []},
            ]
        }
        events = _extract_patch_references(cve_data)
        assert len(events) == 1

    def test_no_patch_references(self):
        cve_data = {"references": [{"url": "https://example.com/advisory", "tags": ["Vendor Advisory"]}]}
        events = _extract_patch_references(cve_data)
        assert len(events) == 0

    def test_empty_references(self):
        cve_data = {"references": []}
        events = _extract_patch_references(cve_data)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# _get_with_retry tests
# ---------------------------------------------------------------------------


class TestGetWithRetry:
    """Test HTTP retry logic."""

    @patch("manus_agent.tools.get_cve_timeline.requests.get")
    def test_success_on_first_try(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _get_with_retry("https://example.com")
        assert result == mock_resp
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.get_cve_timeline.time.sleep")
    @patch("manus_agent.tools.get_cve_timeline.requests.get")
    def test_retries_on_429(self, mock_get, mock_sleep):
        mock_fail = MagicMock()
        mock_fail.status_code = 429
        mock_fail.raise_for_status.side_effect = Exception("429")

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_fail, mock_ok]
        result = _get_with_retry("https://example.com", max_retries=2, backoff_base=1.0)
        assert result == mock_ok

    @patch("manus_agent.tools.get_cve_timeline.time.sleep")
    @patch("manus_agent.tools.get_cve_timeline.requests.get")
    def test_retries_on_500(self, mock_get, mock_sleep):
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_fail.raise_for_status.side_effect = Exception("500")

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_fail, mock_ok]
        result = _get_with_retry("https://example.com", max_retries=2, backoff_base=1.0)
        assert result == mock_ok

    @patch("manus_agent.tools.get_cve_timeline.time.sleep")
    @patch("manus_agent.tools.get_cve_timeline.requests.get")
    def test_raises_after_max_retries(self, mock_get, mock_sleep):
        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("timeout")

        with pytest.raises(req.exceptions.ConnectionError):
            _get_with_retry("https://example.com", max_retries=2, backoff_base=1.0)

        assert mock_get.call_count == 3  # initial + 2 retries

    @patch("manus_agent.tools.get_cve_timeline.requests.get")
    def test_passes_headers_and_params(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        _get_with_retry(
            "https://example.com",
            params={"key": "val"},
            headers={"Auth": "token"},
        )
        mock_get.assert_called_once_with(
            "https://example.com",
            params={"key": "val"},
            headers={"Auth": "token"},
            timeout=20,
        )


# ---------------------------------------------------------------------------
# _fetch_nvd_events tests
# ---------------------------------------------------------------------------


class TestFetchNvdEvents:
    """Test NVD event extraction."""

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_extracts_published_date(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "published": "2021-12-10T10:15:00.000Z",
                        "lastModified": "2022-01-15T10:00:00.000Z",
                        "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 10.0}}]},
                        "references": [],
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_nvd_events("CVE-2021-44228")
        event_types = [e["event"] for e in events]
        assert "CVE published" in event_types
        assert "NVD record updated" in event_types
        assert "CVSS 3.1 score assigned" in event_types

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_extracts_kev_from_nvd(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "published": "2021-12-10T10:15:00.000Z",
                        "lastModified": "2021-12-10T10:15:00.000Z",
                        "cisaExploitAdd": "2021-12-10",
                        "metrics": {},
                        "references": [],
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_nvd_events("CVE-2021-44228")
        kev_events = [e for e in events if "KEV" in e["event"]]
        assert len(kev_events) == 1

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_handles_empty_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        events = _fetch_nvd_events("CVE-9999-9999")
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_handles_network_error(self, mock_get):
        mock_get.side_effect = Exception("network error")
        events = _fetch_nvd_events("CVE-2021-44228")
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_no_duplicate_modified_when_same_as_published(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "published": "2021-12-10T10:15:00.000Z",
                        "lastModified": "2021-12-10T10:15:00.000Z",
                        "metrics": {},
                        "references": [],
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_nvd_events("CVE-2021-44228")
        modified_events = [e for e in events if e["event"] == "NVD record updated"]
        assert len(modified_events) == 0


# ---------------------------------------------------------------------------
# _fetch_epss_events tests
# ---------------------------------------------------------------------------


class TestFetchEpssEvents:
    """Test EPSS event extraction."""

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_extracts_first_score(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"epss": "0.001", "percentile": "0.3", "date": "2022-01-01"},
                {"epss": "0.005", "percentile": "0.5", "date": "2022-01-10"},
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_epss_events("CVE-2021-44228")
        first_events = [e for e in events if e["event"] == "First EPSS score"]
        assert len(first_events) == 1
        assert first_events[0]["date"] == "2022-01-01"

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_detects_spike(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"epss": "0.05", "percentile": "0.5", "date": "2022-01-01"},
                {"epss": "0.20", "percentile": "0.9", "date": "2022-01-02"},
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_epss_events("CVE-2021-44228")
        spike_events = [e for e in events if "spike" in e["event"].lower()]
        assert len(spike_events) == 1

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_no_spike_below_threshold(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"epss": "0.05", "percentile": "0.5", "date": "2022-01-01"},
                {"epss": "0.08", "percentile": "0.6", "date": "2022-01-02"},
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_epss_events("CVE-2021-44228")
        spike_events = [e for e in events if "spike" in e["event"].lower()]
        assert len(spike_events) == 0

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_handles_empty_data(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        events = _fetch_epss_events("CVE-2021-44228")
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_handles_network_error(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        events = _fetch_epss_events("CVE-2021-44228")
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_latest_score_event(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"epss": "0.001", "percentile": "0.3", "date": "2022-01-01"},
                {"epss": "0.050", "percentile": "0.8", "date": "2022-06-01"},
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_epss_events("CVE-2021-44228")
        latest_events = [e for e in events if e["event"] == "Latest EPSS score"]
        assert len(latest_events) == 1
        assert latest_events[0]["date"] == "2022-06-01"


# ---------------------------------------------------------------------------
# _fetch_kev_date tests
# ---------------------------------------------------------------------------


class TestFetchKevDate:
    """Test CISA KEV date fetching."""

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_finds_cve_in_kev(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cveID": "CVE-2021-44228",
                    "dateAdded": "2021-12-10",
                    "vulnerabilityName": "Apache Log4j RCE",
                    "requiredAction": "Apply updates",
                    "dueDate": "2021-12-24",
                }
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_kev_date("CVE-2021-44228")
        assert len(events) == 1
        assert events[0]["date"] == "2021-12-10"
        assert "Apache Log4j" in events[0]["detail"]

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_cve_not_in_kev(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [{"cveID": "CVE-2020-1234", "dateAdded": "2020-05-01"}]}
        mock_get.return_value = mock_resp

        events = _fetch_kev_date("CVE-2021-44228")
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_handles_network_error(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        events = _fetch_kev_date("CVE-2021-44228")
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_case_insensitive_matching(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cveID": "cve-2021-44228",
                    "dateAdded": "2021-12-10",
                    "vulnerabilityName": "Test",
                    "requiredAction": "Fix",
                    "dueDate": "2022-01-01",
                }
            ]
        }
        mock_get.return_value = mock_resp

        events = _fetch_kev_date("CVE-2021-44228")
        assert len(events) == 1


# ---------------------------------------------------------------------------
# _fetch_github_advisory_dates tests
# ---------------------------------------------------------------------------


class TestFetchGithubAdvisoryDates:
    """Test GitHub advisory date extraction."""

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_extracts_published_date(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "ghsa_id": "GHSA-jfh8-c2jp-5v3q",
                "published_at": "2021-12-10T00:00:00Z",
                "updated_at": "2022-01-05T00:00:00Z",
                "summary": "Remote code execution in Log4j",
            }
        ]
        mock_get.return_value = mock_resp

        events = _fetch_github_advisory_dates("CVE-2021-44228")
        assert len(events) == 2
        pub_events = [e for e in events if "published" in e["event"]]
        assert len(pub_events) == 1

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_no_advisory_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        events = _fetch_github_advisory_dates("CVE-9999-9999")
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_handles_network_error(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        events = _fetch_github_advisory_dates("CVE-2021-44228")
        assert events == []

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_no_duplicate_update_when_same_as_publish(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "ghsa_id": "GHSA-xxxx",
                "published_at": "2021-12-10T00:00:00Z",
                "updated_at": "2021-12-10T00:00:00Z",
                "summary": "Test",
            }
        ]
        mock_get.return_value = mock_resp

        events = _fetch_github_advisory_dates("CVE-2021-44228")
        assert len(events) == 1  # Only published, not updated


# ---------------------------------------------------------------------------
# _compute_summary tests
# ---------------------------------------------------------------------------


class TestComputeSummary:
    """Test summary computation."""

    def test_computes_publish_to_kev_delta(self):
        events = [
            {"date": "2021-12-10", "event": "CVE published", "source": "NVD"},
            {"date": "2021-12-15", "event": "Added to CISA KEV", "source": "CISA KEV"},
        ]
        summary = _compute_summary(events, "CVE-2021-44228")
        assert summary["days_publish_to_kev"] == 5

    def test_computes_publish_to_ghsa_delta(self):
        events = [
            {"date": "2021-12-10", "event": "CVE published", "source": "NVD"},
            {"date": "2021-12-12", "event": "GitHub advisory published", "source": "GitHub"},
        ]
        summary = _compute_summary(events, "CVE-2021-44228")
        assert summary["days_publish_to_ghsa"] == 2

    def test_counts_epss_spikes(self):
        events = [
            {"date": "2022-01-01", "event": "EPSS spike detected", "source": "EPSS"},
            {"date": "2022-02-01", "event": "EPSS spike detected", "source": "EPSS"},
        ]
        summary = _compute_summary(events, "CVE-2021-44228")
        assert summary["epss_spikes"] == 2

    def test_no_kev(self):
        events = [
            {"date": "2021-12-10", "event": "CVE published", "source": "NVD"},
        ]
        summary = _compute_summary(events, "CVE-2021-44228")
        assert summary["in_kev"] is False
        assert "days_publish_to_kev" not in summary

    def test_empty_events(self):
        summary = _compute_summary([], "CVE-2021-44228")
        assert summary["total_events"] == 0
        assert summary["in_kev"] is False


# ---------------------------------------------------------------------------
# build_timeline integration tests
# ---------------------------------------------------------------------------


class TestBuildTimeline:
    """Test the main timeline assembly function."""

    def test_invalid_cve_format(self):
        result = build_timeline("not-a-cve")
        assert "error" in result
        assert result["events"] == []

    def test_invalid_format_partial(self):
        result = build_timeline("CVE-2021")
        assert "error" in result

    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_dates")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_date")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_assembles_all_sources(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = [{"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": "test"}]
        mock_epss.return_value = [
            {"date": "2022-01-01", "source": "EPSS", "event": "First EPSS score", "detail": "0.5"}
        ]
        mock_kev.return_value = [
            {"date": "2021-12-15", "source": "CISA KEV", "event": "Added to CISA KEV", "detail": "test"}
        ]
        mock_ghsa.return_value = [
            {"date": "2021-12-11", "source": "GitHub Advisory", "event": "GitHub advisory published", "detail": "test"}
        ]

        result = build_timeline("CVE-2021-44228")
        assert result["cve_id"] == "CVE-2021-44228"
        assert len(result["events"]) == 4

    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_dates")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_date")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_sorts_chronologically(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = [{"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": ""}]
        mock_epss.return_value = [{"date": "2022-06-01", "source": "EPSS", "event": "First EPSS score", "detail": ""}]
        mock_kev.return_value = []
        mock_ghsa.return_value = [
            {"date": "2021-12-09", "source": "GitHub Advisory", "event": "GitHub advisory published", "detail": ""}
        ]

        result = build_timeline("CVE-2021-44228")
        dates = [e["date"] for e in result["events"]]
        assert dates == sorted(dates)

    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_dates")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_date")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_deduplicates_kev_events(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = [
            {"date": "2021-12-15", "source": "CISA KEV", "event": "Added to CISA KEV", "detail": "short"}
        ]
        mock_epss.return_value = []
        # KEV fetch should be skipped since NVD already gave us a KEV event
        mock_kev.return_value = []
        mock_ghsa.return_value = []

        result = build_timeline("CVE-2021-44228")
        kev_events = [e for e in result["events"] if "KEV" in e["event"]]
        assert len(kev_events) == 1

    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_dates")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_date")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_undated_references_separated(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = [
            {"date": None, "source": "NVD reference", "event": "Patch commit published", "detail": "url"},
            {"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": ""},
        ]
        mock_epss.return_value = []
        mock_kev.return_value = []
        mock_ghsa.return_value = []

        result = build_timeline("CVE-2021-44228")
        assert len(result["events"]) == 1  # Only dated events
        assert len(result["undated_references"]) == 1

    def test_normalizes_cve_id_uppercase(self):
        with (
            patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events") as mock_nvd,
            patch("manus_agent.tools.get_cve_timeline._fetch_epss_events") as mock_epss,
            patch("manus_agent.tools.get_cve_timeline._fetch_kev_date") as mock_kev,
            patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_dates") as mock_ghsa,
        ):
            mock_nvd.return_value = []
            mock_epss.return_value = []
            mock_kev.return_value = []
            mock_ghsa.return_value = []

            result = build_timeline("cve-2021-44228")
            assert result["cve_id"] == "CVE-2021-44228"

    def test_strips_whitespace(self):
        with (
            patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events") as mock_nvd,
            patch("manus_agent.tools.get_cve_timeline._fetch_epss_events") as mock_epss,
            patch("manus_agent.tools.get_cve_timeline._fetch_kev_date") as mock_kev,
            patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_dates") as mock_ghsa,
        ):
            mock_nvd.return_value = []
            mock_epss.return_value = []
            mock_kev.return_value = []
            mock_ghsa.return_value = []

            result = build_timeline("  CVE-2021-44228  ")
            assert result["cve_id"] == "CVE-2021-44228"


# ---------------------------------------------------------------------------
# format_timeline_text tests
# ---------------------------------------------------------------------------


class TestFormatTimelineText:
    """Test text output formatting."""

    def test_formats_events(self):
        result = {
            "cve_id": "CVE-2021-44228",
            "events": [
                {"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": "test detail"},
            ],
            "undated_references": [],
            "summary": {"total_events": 1, "in_kev": False, "epss_spikes": 0},
        }
        text = format_timeline_text(result)
        assert "CVE-2021-44228" in text
        assert "2021-12-10" in text
        assert "CVE published" in text

    def test_formats_error(self):
        result = {"error": "Invalid CVE ID", "events": []}
        text = format_timeline_text(result)
        assert "Error" in text

    def test_formats_no_events(self):
        result = {
            "cve_id": "CVE-9999-9999",
            "events": [],
            "undated_references": [],
            "summary": {"total_events": 0, "in_kev": False, "epss_spikes": 0},
        }
        text = format_timeline_text(result)
        assert "No timeline events found" in text

    def test_formats_kev_summary(self):
        result = {
            "cve_id": "CVE-2021-44228",
            "events": [
                {"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": ""},
                {"date": "2021-12-15", "source": "CISA KEV", "event": "Added to CISA KEV", "detail": ""},
            ],
            "undated_references": [],
            "summary": {
                "total_events": 2,
                "in_kev": True,
                "epss_spikes": 0,
                "days_publish_to_kev": 5,
            },
        }
        text = format_timeline_text(result)
        assert "KEV: 5 days" in text
        assert "In CISA KEV: Yes" in text

    def test_formats_undated_references(self):
        result = {
            "cve_id": "CVE-2021-44228",
            "events": [
                {"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": ""},
            ],
            "undated_references": [
                {"event": "Patch commit published", "detail": "https://github.com/org/repo/commit/abc"},
            ],
            "summary": {"total_events": 1, "in_kev": False, "epss_spikes": 0},
        }
        text = format_timeline_text(result)
        assert "Patch References (undated)" in text
        assert "github.com" in text


# ---------------------------------------------------------------------------
# format_timeline_json tests
# ---------------------------------------------------------------------------


class TestFormatTimelineJson:
    """Test JSON output formatting."""

    def test_valid_json(self):
        result = {
            "cve_id": "CVE-2021-44228",
            "events": [{"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": ""}],
            "undated_references": [],
            "summary": {"total_events": 1},
        }
        json_str = format_timeline_json(result)
        parsed = json.loads(json_str)
        assert parsed["cve_id"] == "CVE-2021-44228"

    def test_includes_all_fields(self):
        result = {
            "cve_id": "CVE-2021-44228",
            "events": [],
            "undated_references": [],
            "summary": {},
        }
        json_str = format_timeline_json(result)
        parsed = json.loads(json_str)
        assert "events" in parsed
        assert "summary" in parsed
        assert "undated_references" in parsed


# ---------------------------------------------------------------------------
# handler tests
# ---------------------------------------------------------------------------


class TestHandler:
    """Test the Strands tool handler entry point."""

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_success_response(self, mock_build):
        mock_build.return_value = {
            "cve_id": "CVE-2021-44228",
            "events": [{"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": ""}],
            "undated_references": [],
            "summary": {"total_events": 1, "in_kev": False, "epss_spikes": 0},
        }
        tool_use = {"toolUseId": "test-123", "input": {"cve_id": "CVE-2021-44228"}}
        result = handler(tool_use)
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"
        assert len(result["content"]) == 1

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_error_response(self, mock_build):
        mock_build.return_value = {"error": "Invalid CVE ID format: bad", "events": []}
        tool_use = {"toolUseId": "test-456", "input": {"cve_id": "bad"}}
        result = handler(tool_use)
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_calls_build_timeline_with_cve_id(self, mock_build):
        mock_build.return_value = {
            "cve_id": "CVE-2021-44228",
            "events": [],
            "undated_references": [],
            "summary": {"total_events": 0, "in_kev": False, "epss_spikes": 0},
        }
        tool_use = {"toolUseId": "test-789", "input": {"cve_id": "CVE-2021-44228"}}
        handler(tool_use)
        mock_build.assert_called_once_with("CVE-2021-44228")


# ---------------------------------------------------------------------------
# CLI subcommand tests
# ---------------------------------------------------------------------------


class TestCliSubcommand:
    """Test the CLI dispatch for cve-timeline."""

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_text_output(self, mock_build, capsys):
        from manus_agent.cli import _run_cve_timeline

        mock_build.return_value = {
            "cve_id": "CVE-2021-44228",
            "events": [{"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": "test"}],
            "undated_references": [],
            "summary": {"total_events": 1, "in_kev": False, "epss_spikes": 0},
        }
        exit_code = _run_cve_timeline(["CVE-2021-44228"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "CVE-2021-44228" in captured.out

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_json_output(self, mock_build, capsys):
        from manus_agent.cli import _run_cve_timeline

        mock_build.return_value = {
            "cve_id": "CVE-2021-44228",
            "events": [{"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": ""}],
            "undated_references": [],
            "summary": {"total_events": 1},
        }
        exit_code = _run_cve_timeline(["CVE-2021-44228", "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["cve_id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.get_cve_timeline.build_timeline")
    def test_error_returns_nonzero(self, mock_build, capsys):
        from manus_agent.cli import _run_cve_timeline

        mock_build.return_value = {"error": "Invalid CVE ID format: bad", "events": []}
        exit_code = _run_cve_timeline(["bad"])
        assert exit_code == 1

    def test_help_flag(self, capsys):
        from manus_agent.cli import _build_cve_timeline_parser

        parser = _build_cve_timeline_parser()
        # Just ensure it parses without error
        args = parser.parse_args(["CVE-2021-44228"])
        assert args.cve_id == "CVE-2021-44228"
        assert args.output == "text"

    def test_json_flag(self):
        from manus_agent.cli import _build_cve_timeline_parser

        parser = _build_cve_timeline_parser()
        args = parser.parse_args(["CVE-2021-44228", "--output", "json"])
        assert args.output == "json"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_cve_with_long_number(self):
        with (
            patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events") as mock_nvd,
            patch("manus_agent.tools.get_cve_timeline._fetch_epss_events") as mock_epss,
            patch("manus_agent.tools.get_cve_timeline._fetch_kev_date") as mock_kev,
            patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_dates") as mock_ghsa,
        ):
            mock_nvd.return_value = []
            mock_epss.return_value = []
            mock_kev.return_value = []
            mock_ghsa.return_value = []

            result = build_timeline("CVE-2024-123456")
            assert result["cve_id"] == "CVE-2024-123456"
            assert "error" not in result

    def test_cve_4_digit_suffix(self):
        with (
            patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events") as mock_nvd,
            patch("manus_agent.tools.get_cve_timeline._fetch_epss_events") as mock_epss,
            patch("manus_agent.tools.get_cve_timeline._fetch_kev_date") as mock_kev,
            patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_dates") as mock_ghsa,
        ):
            mock_nvd.return_value = []
            mock_epss.return_value = []
            mock_kev.return_value = []
            mock_ghsa.return_value = []

            result = build_timeline("CVE-2021-1234")
            assert "error" not in result

    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_dates")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_date")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_all_sources_fail_gracefully(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = []
        mock_epss.return_value = []
        mock_kev.return_value = []
        mock_ghsa.return_value = []

        result = build_timeline("CVE-2021-44228")
        assert result["events"] == []
        assert "error" not in result

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_nvd_api_key_in_headers(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {"NVD_API_KEY": "test-key-123"}):
            _fetch_nvd_events("CVE-2021-44228")

        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["headers"]["apiKey"] == "test-key-123"

    @patch("manus_agent.tools.get_cve_timeline._get_with_retry")
    def test_github_token_in_headers(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"}):
            _fetch_github_advisory_dates("CVE-2021-44228")

        call_kwargs = mock_get.call_args
        assert "Bearer ghp_test123" in call_kwargs[1]["headers"]["Authorization"]

    @patch("manus_agent.tools.get_cve_timeline._fetch_github_advisory_dates")
    @patch("manus_agent.tools.get_cve_timeline._fetch_kev_date")
    @patch("manus_agent.tools.get_cve_timeline._fetch_epss_events")
    @patch("manus_agent.tools.get_cve_timeline._fetch_nvd_events")
    def test_summary_present_in_result(self, mock_nvd, mock_epss, mock_kev, mock_ghsa):
        mock_nvd.return_value = [{"date": "2021-12-10", "source": "NVD", "event": "CVE published", "detail": ""}]
        mock_epss.return_value = []
        mock_kev.return_value = []
        mock_ghsa.return_value = []

        result = build_timeline("CVE-2021-44228")
        assert "summary" in result
        assert result["summary"]["nvd_published"] == "2021-12-10"
