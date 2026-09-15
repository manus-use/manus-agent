"""Comprehensive test suite for the obtain_cves tool module.

Tests cover all public and internal functions in
``src/manus_agent/tools/obtain_cves.py``:

- ``_get_all_cves_from_nvd``  — paginated NVD CVE fetcher with retry
- ``_get_all_cves_from_github`` — paginated GitHub Advisories fetcher
- ``_filter_cves_by_epss``    — EPSS score / percentile filtering
- ``_enrich_with_cisa_kev``   — CISA KEV enrichment
- ``_submit_in_batches``      — batch webhook submission
- ``obtain_cves``             — Strands tool handler

All HTTP calls are fully mocked — zero real network traffic.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from manus_agent.tools.obtain_cves import (
    _enrich_with_cisa_kev,
    _filter_cves_by_epss,
    _get_all_cves_from_github,
    _get_all_cves_from_nvd,
    _submit_in_batches,
    obtain_cves,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

START_DATE = "2026-09-01T00:00:00.000Z"
END_DATE = "2026-09-15T23:59:59.000Z"


def _make_nvd_cve(cve_id: str, base_score: float = 9.8) -> dict[str, Any]:
    """Build a minimal NVD-style CVE dict."""
    return {
        "cve": {
            "id": cve_id,
            "descriptions": [{"lang": "en", "value": f"Description for {cve_id}"}],
            "published": "2026-09-01T12:00:00.000",
            "metrics": {
                "cvssMetricV31": [
                    {
                        "cvssData": {
                            "baseScore": base_score,
                            "baseSeverity": "CRITICAL" if base_score >= 9.0 else "HIGH",
                        }
                    }
                ]
            },
            "weaknesses": [{"description": [{"value": "CWE-79"}]}],
            "configurations": [
                {
                    "nodes": [
                        {
                            "cpeMatch": [
                                {
                                    "vulnerable": True,
                                    "criteria": "cpe:2.3:a:vendor:product:1.0:*:*:*:*:*:*:*",
                                }
                            ]
                        }
                    ]
                }
            ],
            "references": [{"url": "https://example.com/advisory"}],
        }
    }


def _make_github_advisory(cve_id: str, summary: str = "Test advisory") -> dict[str, Any]:
    """Build a minimal GitHub Advisory response item."""
    return {
        "cve_id": cve_id,
        "summary": summary,
        "published_at": "2026-09-01T12:00:00Z",
        "cvss_severities": {"score": 9.1},
    }


def _make_nvd_page(cves: list[dict], total: int) -> MagicMock:
    """Build a mock requests.Response for an NVD API page."""
    resp = MagicMock()
    resp.json.return_value = {
        "vulnerabilities": cves,
        "totalResults": total,
    }
    resp.status_code = 200
    return resp


def _make_tool_use(start: str = START_DATE, end: str = END_DATE) -> dict[str, Any]:
    """Build a Strands ToolUse dict for obtain_cves."""
    return {
        "toolUseId": "test-tool-001",
        "input": {"start_date": start, "end_date": end},
    }


# =========================================================================
# _get_all_cves_from_nvd
# =========================================================================


class TestGetAllCvesFromNVD:
    """Tests for the NVD paginated CVE fetcher."""

    def test_single_page_returns_all_cves(self):
        cves = [_make_nvd_cve("CVE-2026-0001"), _make_nvd_cve("CVE-2026-0002")]
        mock_resp = _make_nvd_page(cves, total=2)

        with patch(
            "manus_agent.tools.obtain_cves._nvd_get_with_retry",
            return_value=mock_resp,
        ):
            result = _get_all_cves_from_nvd(START_DATE, END_DATE)

        assert len(result) == 2
        assert result[0]["cve"]["id"] == "CVE-2026-0001"
        assert result[1]["cve"]["id"] == "CVE-2026-0002"

    def test_pagination_fetches_all_pages(self):
        page1_cves = [_make_nvd_cve(f"CVE-2026-{i:04d}") for i in range(100)]
        page2_cves = [_make_nvd_cve(f"CVE-2026-{i:04d}") for i in range(100, 150)]

        call_count = 0

        def mock_nvd_get(url):
            nonlocal call_count
            call_count += 1
            if "startIndex=0" in url or call_count == 1:
                return _make_nvd_page(page1_cves, total=150)
            else:
                return _make_nvd_page(page2_cves, total=150)

        with patch(
            "manus_agent.tools.obtain_cves._nvd_get_with_retry",
            side_effect=mock_nvd_get,
        ):
            result = _get_all_cves_from_nvd(START_DATE, END_DATE)

        assert len(result) == 150

    def test_empty_response_returns_empty_list(self):
        mock_resp = _make_nvd_page([], total=0)

        with patch(
            "manus_agent.tools.obtain_cves._nvd_get_with_retry",
            return_value=mock_resp,
        ):
            result = _get_all_cves_from_nvd(START_DATE, END_DATE)

        assert result == []

    def test_url_contains_date_range(self):
        mock_resp = _make_nvd_page([], total=0)

        with patch(
            "manus_agent.tools.obtain_cves._nvd_get_with_retry",
            return_value=mock_resp,
        ) as mock_get:
            _get_all_cves_from_nvd(START_DATE, END_DATE)

        call_url = mock_get.call_args[0][0]
        assert START_DATE in call_url
        assert END_DATE in call_url

    def test_url_requests_high_and_critical_severity(self):
        mock_resp = _make_nvd_page([], total=0)

        with patch(
            "manus_agent.tools.obtain_cves._nvd_get_with_retry",
            return_value=mock_resp,
        ) as mock_get:
            _get_all_cves_from_nvd(START_DATE, END_DATE)

        call_url = mock_get.call_args[0][0]
        assert "cvssV3Severity=HIGH" in call_url
        assert "cvssV3Severity=CRITICAL" in call_url

    def test_url_requests_100_results_per_page(self):
        mock_resp = _make_nvd_page([], total=0)

        with patch(
            "manus_agent.tools.obtain_cves._nvd_get_with_retry",
            return_value=mock_resp,
        ) as mock_get:
            _get_all_cves_from_nvd(START_DATE, END_DATE)

        call_url = mock_get.call_args[0][0]
        assert "resultsPerPage=100" in call_url


# =========================================================================
# _get_all_cves_from_github
# =========================================================================


class TestGetAllCvesFromGitHub:
    """Tests for the GitHub Advisories paginated fetcher."""

    def test_single_page_returns_all_advisories(self):
        advisories = [
            _make_github_advisory("CVE-2026-1001"),
            _make_github_advisory("CVE-2026-1002"),
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = advisories
        mock_resp.links = {}
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _get_all_cves_from_github(START_DATE, END_DATE)

        assert len(result) == 2
        assert result[0]["cve"]["id"] == "CVE-2026-1001"
        assert result[1]["cve"]["id"] == "CVE-2026-1002"

    def test_pagination_follows_next_links(self):
        adv_page1 = [_make_github_advisory("CVE-2026-2001")]
        adv_page2 = [_make_github_advisory("CVE-2026-2002")]

        resp1 = MagicMock()
        resp1.json.return_value = adv_page1
        resp1.links = {"next": {"url": "https://api.github.com/advisories?page=2"}}
        resp1.raise_for_status = MagicMock()

        resp2 = MagicMock()
        resp2.json.return_value = adv_page2
        resp2.links = {}
        resp2.raise_for_status = MagicMock()

        with patch(
            "manus_agent.tools.obtain_cves.requests.get",
            side_effect=[resp1, resp2],
        ):
            result = _get_all_cves_from_github(START_DATE, END_DATE)

        assert len(result) == 2
        ids = {r["cve"]["id"] for r in result}
        assert ids == {"CVE-2026-2001", "CVE-2026-2002"}

    def test_skips_advisories_without_cve_id(self):
        advisories = [
            {"cve_id": "CVE-2026-3001", "summary": "Has CVE"},
            {"cve_id": None, "summary": "No CVE"},
            {"summary": "Missing field entirely"},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = advisories
        mock_resp.links = {}
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _get_all_cves_from_github(START_DATE, END_DATE)

        assert len(result) == 1
        assert result[0]["cve"]["id"] == "CVE-2026-3001"

    def test_maps_advisory_fields_to_cve_structure(self):
        advisories = [
            _make_github_advisory("CVE-2026-4001", summary="SQL injection in auth"),
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = advisories
        mock_resp.links = {}
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _get_all_cves_from_github(START_DATE, END_DATE)

        cve = result[0]["cve"]
        assert cve["id"] == "CVE-2026-4001"
        assert cve["descriptions"][0]["lang"] == "en"
        assert cve["descriptions"][0]["value"] == "SQL injection in auth"
        assert cve["summary"] == "SQL injection in auth"

    def test_request_error_returns_empty_list(self):
        import requests

        with patch(
            "manus_agent.tools.obtain_cves.requests.get",
            side_effect=requests.exceptions.ConnectionError("timeout"),
        ):
            result = _get_all_cves_from_github(START_DATE, END_DATE)

        assert result == []

    def test_http_error_returns_empty_list(self):
        import requests

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("403")

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _get_all_cves_from_github(START_DATE, END_DATE)

        assert result == []

    def test_date_range_in_url(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.links = {}
        mock_resp.raise_for_status = MagicMock()

        with patch(
            "manus_agent.tools.obtain_cves.requests.get",
            return_value=mock_resp,
        ) as mock_get:
            _get_all_cves_from_github(START_DATE, END_DATE)

        call_url = mock_get.call_args[0][0]
        # Dates should be truncated to YYYY-MM-DD
        assert "2026-09-01" in call_url
        assert "2026-09-15" in call_url

    def test_empty_advisories_returns_empty_list(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.links = {}
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _get_all_cves_from_github(START_DATE, END_DATE)

        assert result == []


# =========================================================================
# _filter_cves_by_epss
# =========================================================================


class TestFilterCvesByEpss:
    """Tests for the EPSS score / percentile filter."""

    def _mock_epss_response(self, epss_data: list[dict]) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = {"data": epss_data}
        resp.raise_for_status = MagicMock()
        return resp

    def test_filters_by_epss_score_above_threshold(self):
        cves = [_make_nvd_cve("CVE-2026-5001")]
        epss_data = [{"cve": "CVE-2026-5001", "epss": "0.15", "percentile": "0.3"}]
        mock_resp = self._mock_epss_response(epss_data)

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _filter_cves_by_epss(cves)

        assert len(result) == 1
        assert result[0]["epss_data"]["cve"] == "CVE-2026-5001"

    def test_filters_by_percentile_above_threshold(self):
        cves = [_make_nvd_cve("CVE-2026-5002")]
        epss_data = [{"cve": "CVE-2026-5002", "epss": "0.01", "percentile": "0.75"}]
        mock_resp = self._mock_epss_response(epss_data)

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _filter_cves_by_epss(cves)

        assert len(result) == 1

    def test_excludes_low_epss_and_low_percentile(self):
        cves = [_make_nvd_cve("CVE-2026-5003")]
        epss_data = [{"cve": "CVE-2026-5003", "epss": "0.01", "percentile": "0.1"}]
        mock_resp = self._mock_epss_response(epss_data)

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _filter_cves_by_epss(cves)

        assert len(result) == 0

    def test_mixed_results_filtered_correctly(self):
        cves = [
            _make_nvd_cve("CVE-2026-5004"),  # high epss
            _make_nvd_cve("CVE-2026-5005"),  # low epss, low percentile
            _make_nvd_cve("CVE-2026-5006"),  # low epss, high percentile
        ]
        epss_data = [
            {"cve": "CVE-2026-5004", "epss": "0.90", "percentile": "0.99"},
            {"cve": "CVE-2026-5005", "epss": "0.01", "percentile": "0.1"},
            {"cve": "CVE-2026-5006", "epss": "0.03", "percentile": "0.6"},
        ]
        mock_resp = self._mock_epss_response(epss_data)

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _filter_cves_by_epss(cves)

        ids = {r["cve"]["id"] for r in result}
        assert "CVE-2026-5004" in ids
        assert "CVE-2026-5005" not in ids
        assert "CVE-2026-5006" in ids

    def test_cve_not_in_epss_data_is_excluded(self):
        cves = [_make_nvd_cve("CVE-2026-5007")]
        # EPSS API returns data for a different CVE
        epss_data = [{"cve": "CVE-2026-9999", "epss": "0.90", "percentile": "0.99"}]
        mock_resp = self._mock_epss_response(epss_data)

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _filter_cves_by_epss(cves)

        assert len(result) == 0

    def test_empty_input_returns_empty_list(self):
        result = _filter_cves_by_epss([])
        assert result == []

    def test_epss_data_attached_to_result(self):
        cves = [_make_nvd_cve("CVE-2026-5008")]
        epss_data = [{"cve": "CVE-2026-5008", "epss": "0.50", "percentile": "0.85"}]
        mock_resp = self._mock_epss_response(epss_data)

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _filter_cves_by_epss(cves)

        assert result[0]["epss_data"]["epss"] == "0.50"
        assert result[0]["epss_data"]["percentile"] == "0.85"

    def test_boundary_epss_exactly_005_excluded(self):
        """EPSS exactly at 0.05 threshold should NOT pass (strictly greater than)."""
        cves = [_make_nvd_cve("CVE-2026-5009")]
        epss_data = [{"cve": "CVE-2026-5009", "epss": "0.05", "percentile": "0.4"}]
        mock_resp = self._mock_epss_response(epss_data)

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _filter_cves_by_epss(cves)

        assert len(result) == 0

    def test_boundary_percentile_exactly_05_excluded(self):
        """Percentile exactly at 0.5 threshold should NOT pass (strictly greater than)."""
        cves = [_make_nvd_cve("CVE-2026-5010")]
        epss_data = [{"cve": "CVE-2026-5010", "epss": "0.01", "percentile": "0.5"}]
        mock_resp = self._mock_epss_response(epss_data)

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _filter_cves_by_epss(cves)

        assert len(result) == 0

    def test_batch_cve_ids_joined_in_url(self):
        cves = [
            _make_nvd_cve("CVE-2026-5011"),
            _make_nvd_cve("CVE-2026-5012"),
        ]
        epss_data = []
        mock_resp = self._mock_epss_response(epss_data)

        with patch(
            "manus_agent.tools.obtain_cves.requests.get",
            return_value=mock_resp,
        ) as mock_get:
            _filter_cves_by_epss(cves)

        call_url = mock_get.call_args[0][0]
        assert "CVE-2026-5011" in call_url
        assert "CVE-2026-5012" in call_url
        assert "," in call_url  # IDs are comma-separated


# =========================================================================
# _enrich_with_cisa_kev
# =========================================================================


class TestEnrichWithCisaKev:
    """Tests for the CISA KEV enrichment function."""

    def _mock_kev_response(self, kev_cve_ids: list[str]) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = {"vulnerabilities": [{"cveID": cve_id} for cve_id in kev_cve_ids]}
        resp.raise_for_status = MagicMock()
        return resp

    def test_marks_kev_cves_as_true(self):
        cves = [_make_nvd_cve("CVE-2026-6001")]
        mock_resp = self._mock_kev_response(["CVE-2026-6001"])

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _enrich_with_cisa_kev(cves)

        assert result[0]["cisa_kev"] is True

    def test_marks_non_kev_cves_as_false(self):
        cves = [_make_nvd_cve("CVE-2026-6002")]
        mock_resp = self._mock_kev_response(["CVE-2026-9999"])

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _enrich_with_cisa_kev(cves)

        assert result[0]["cisa_kev"] is False

    def test_mixed_kev_and_non_kev(self):
        cves = [
            _make_nvd_cve("CVE-2026-6003"),
            _make_nvd_cve("CVE-2026-6004"),
            _make_nvd_cve("CVE-2026-6005"),
        ]
        mock_resp = self._mock_kev_response(["CVE-2026-6003", "CVE-2026-6005"])

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _enrich_with_cisa_kev(cves)

        assert result[0]["cisa_kev"] is True
        assert result[1]["cisa_kev"] is False
        assert result[2]["cisa_kev"] is True

    def test_empty_kev_feed(self):
        cves = [_make_nvd_cve("CVE-2026-6006")]
        mock_resp = self._mock_kev_response([])

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _enrich_with_cisa_kev(cves)

        assert result[0]["cisa_kev"] is False

    def test_empty_cve_list_returns_empty(self):
        mock_resp = self._mock_kev_response(["CVE-2026-9999"])

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _enrich_with_cisa_kev([])

        assert result == []

    def test_returns_mutated_input_list(self):
        """The function mutates and returns the same list objects."""
        cves = [_make_nvd_cve("CVE-2026-6007")]
        mock_resp = self._mock_kev_response([])

        with patch("manus_agent.tools.obtain_cves.requests.get", return_value=mock_resp):
            result = _enrich_with_cisa_kev(cves)

        assert result is cves


# =========================================================================
# _submit_in_batches
# =========================================================================


class TestSubmitInBatches:
    """Tests for the batch webhook submission function."""

    def test_submits_each_cve_individually(self):
        cves = [_make_nvd_cve("CVE-2026-7001"), _make_nvd_cve("CVE-2026-7002")]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.post", return_value=mock_resp) as mock_post:
            _submit_in_batches(cves)

        # Each CVE is posted individually within the batch
        assert mock_post.call_count == 2

    def test_formatted_cve_has_expected_fields(self):
        cve = _make_nvd_cve("CVE-2026-7003")
        cve["epss_data"] = {"epss": "0.5", "percentile": "0.85"}
        cve["cisa_kev"] = True

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.post", return_value=mock_resp) as mock_post:
            _submit_in_batches([cve])

        posted_json = mock_post.call_args[1]["json"]
        assert posted_json["cve_id"] == "CVE-2026-7003"
        assert posted_json["epss_score"] == "0.5"
        assert posted_json["epss_percentile"] == "0.85"
        assert posted_json["cisa_kev"] is True
        assert posted_json["exploited"] is True

    def test_cve_without_epss_data(self):
        cve = _make_nvd_cve("CVE-2026-7004")
        # No epss_data key
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.post", return_value=mock_resp) as mock_post:
            _submit_in_batches([cve])

        posted_json = mock_post.call_args[1]["json"]
        assert posted_json["epss_score"] is None
        assert posted_json["epss_percentile"] is None

    def test_cve_without_configurations(self):
        cve = _make_nvd_cve("CVE-2026-7005")
        del cve["cve"]["configurations"]

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.post", return_value=mock_resp) as mock_post:
            _submit_in_batches([cve])

        posted_json = mock_post.call_args[1]["json"]
        assert posted_json["cpe"] == ""
        assert posted_json["affected_products"] == ""

    def test_cve_without_metrics(self):
        cve = _make_nvd_cve("CVE-2026-7006")
        cve["cve"]["metrics"] = {}

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.post", return_value=mock_resp) as mock_post:
            _submit_in_batches([cve])

        posted_json = mock_post.call_args[1]["json"]
        assert posted_json["cvss_score"] == " ()"

    def test_empty_cves_no_posts(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.post", return_value=mock_resp) as mock_post:
            _submit_in_batches([])

        mock_post.assert_not_called()

    def test_batches_of_100(self):
        """CVEs should be chunked in groups of 100."""
        cves = [_make_nvd_cve(f"CVE-2026-{8000 + i}") for i in range(150)]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.post", return_value=mock_resp) as mock_post:
            _submit_in_batches(cves)

        # 150 individual posts (one per CVE, but in 2 batch loops)
        assert mock_post.call_count == 150

    def test_description_extraction(self):
        cve = _make_nvd_cve("CVE-2026-7007")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.post", return_value=mock_resp) as mock_post:
            _submit_in_batches([cve])

        posted_json = mock_post.call_args[1]["json"]
        assert posted_json["description"] == "Description for CVE-2026-7007"

    def test_cve_without_descriptions_uses_default(self):
        cve = _make_nvd_cve("CVE-2026-7008")
        cve["cve"]["descriptions"] = []

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.post", return_value=mock_resp) as mock_post:
            _submit_in_batches([cve])

        posted_json = mock_post.call_args[1]["json"]
        assert posted_json["description"] == "No description available."

    def test_affected_products_extracted_from_cpe(self):
        cve = _make_nvd_cve("CVE-2026-7009")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("manus_agent.tools.obtain_cves.requests.post", return_value=mock_resp) as mock_post:
            _submit_in_batches([cve])

        posted_json = mock_post.call_args[1]["json"]
        assert "vendor:product" in posted_json["affected_products"]
        assert "cpe:2.3:a:vendor:product" in posted_json["cpe"]


# =========================================================================
# obtain_cves (Strands tool handler)
# =========================================================================


class TestObtainCvesTool:
    """Tests for the main Strands tool handler."""

    def test_success_with_high_epss_cves(self):
        nvd_cves = [_make_nvd_cve("CVE-2026-8001"), _make_nvd_cve("CVE-2026-8002")]
        github_cves = []

        epss_data = [
            {"cve": "CVE-2026-8001", "epss": "0.80", "percentile": "0.95"},
            {"cve": "CVE-2026-8002", "epss": "0.02", "percentile": "0.1"},
        ]
        epss_resp = MagicMock()
        epss_resp.json.return_value = {"data": epss_data}
        epss_resp.raise_for_status = MagicMock()

        with (
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_nvd",
                return_value=nvd_cves,
            ),
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_github",
                return_value=github_cves,
            ),
            patch("manus_agent.tools.obtain_cves.requests.get", return_value=epss_resp),
        ):
            result = obtain_cves(_make_tool_use())

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-tool-001"
        # Should have text and json content blocks
        texts = [c for c in result["content"] if "text" in c]
        jsons = [c for c in result["content"] if "json" in c]
        assert len(texts) >= 1
        assert len(jsons) >= 1

    def test_success_returns_total_found_count(self):
        nvd_cves = [_make_nvd_cve("CVE-2026-8003")]
        epss_data = [{"cve": "CVE-2026-8003", "epss": "0.80", "percentile": "0.95"}]
        epss_resp = MagicMock()
        epss_resp.json.return_value = {"data": epss_data}
        epss_resp.raise_for_status = MagicMock()

        with (
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_nvd",
                return_value=nvd_cves,
            ),
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_github",
                return_value=[],
            ),
            patch("manus_agent.tools.obtain_cves.requests.get", return_value=epss_resp),
        ):
            result = obtain_cves(_make_tool_use())

        json_block = next(c["json"] for c in result["content"] if "json" in c)
        assert json_block["total_found"] == 1
        assert json_block["total_with_high_epss"] == 1

    def test_no_cves_found_returns_success_with_message(self):
        with (
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_nvd",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_github",
                return_value=[],
            ),
        ):
            result = obtain_cves(_make_tool_use())

        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "No new high/critical CVEs found" in text

    def test_deduplicates_nvd_and_github_cves(self):
        """Same CVE from both sources should be deduplicated."""
        nvd_cves = [_make_nvd_cve("CVE-2026-8004")]
        github_cves = [
            {
                "cve": {
                    "id": "CVE-2026-8004",
                    "descriptions": [{"lang": "en", "value": "From GitHub"}],
                    "summary": "From GitHub",
                    "published": "2026-09-01",
                    "cvss_score": None,
                }
            }
        ]

        epss_data = [{"cve": "CVE-2026-8004", "epss": "0.80", "percentile": "0.95"}]
        epss_resp = MagicMock()
        epss_resp.json.return_value = {"data": epss_data}
        epss_resp.raise_for_status = MagicMock()

        with (
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_nvd",
                return_value=nvd_cves,
            ),
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_github",
                return_value=github_cves,
            ),
            patch("manus_agent.tools.obtain_cves.requests.get", return_value=epss_resp),
        ):
            result = obtain_cves(_make_tool_use())

        json_block = next(c["json"] for c in result["content"] if "json" in c)
        assert json_block["total_found"] == 1

    def test_github_only_cve_added_when_not_in_nvd(self):
        nvd_cves = [_make_nvd_cve("CVE-2026-8005")]
        github_cves = [
            {
                "cve": {
                    "id": "CVE-2026-8006",
                    "descriptions": [{"lang": "en", "value": "GitHub only"}],
                    "summary": "GitHub only",
                    "published": "2026-09-01",
                    "cvss_score": None,
                }
            }
        ]

        epss_data = [
            {"cve": "CVE-2026-8005", "epss": "0.80", "percentile": "0.95"},
            {"cve": "CVE-2026-8006", "epss": "0.70", "percentile": "0.90"},
        ]
        epss_resp = MagicMock()
        epss_resp.json.return_value = {"data": epss_data}
        epss_resp.raise_for_status = MagicMock()

        with (
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_nvd",
                return_value=nvd_cves,
            ),
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_github",
                return_value=github_cves,
            ),
            patch("manus_agent.tools.obtain_cves.requests.get", return_value=epss_resp),
        ):
            result = obtain_cves(_make_tool_use())

        json_block = next(c["json"] for c in result["content"] if "json" in c)
        assert json_block["total_found"] == 2

    def test_exception_returns_error_status(self):
        with (
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_nvd",
                side_effect=Exception("NVD down"),
            ),
        ):
            result = obtain_cves(_make_tool_use())

        assert result["status"] == "error"
        assert "NVD down" in result["content"][0]["text"]

    def test_tool_use_id_echoed(self):
        with (
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_nvd",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_github",
                return_value=[],
            ),
        ):
            result = obtain_cves(_make_tool_use())

        assert result["toolUseId"] == "test-tool-001"

    def test_epss_filtering_in_chunks_of_100(self):
        """When there are >100 CVEs, EPSS filtering should batch by 100."""
        nvd_cves = [_make_nvd_cve(f"CVE-2026-{9000 + i}") for i in range(150)]
        epss_data = [{"cve": f"CVE-2026-{9000 + i}", "epss": "0.80", "percentile": "0.95"} for i in range(150)]
        epss_resp = MagicMock()
        epss_resp.json.return_value = {"data": epss_data}
        epss_resp.raise_for_status = MagicMock()

        with (
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_nvd",
                return_value=nvd_cves,
            ),
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_github",
                return_value=[],
            ),
            patch("manus_agent.tools.obtain_cves.requests.get", return_value=epss_resp),
        ):
            result = obtain_cves(_make_tool_use())

        json_block = next(c["json"] for c in result["content"] if "json" in c)
        assert json_block["total_found"] == 150

    def test_text_content_includes_date_range(self):
        nvd_cves = [_make_nvd_cve("CVE-2026-8010")]
        epss_data = [{"cve": "CVE-2026-8010", "epss": "0.80", "percentile": "0.95"}]
        epss_resp = MagicMock()
        epss_resp.json.return_value = {"data": epss_data}
        epss_resp.raise_for_status = MagicMock()

        with (
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_nvd",
                return_value=nvd_cves,
            ),
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_github",
                return_value=[],
            ),
            patch("manus_agent.tools.obtain_cves.requests.get", return_value=epss_resp),
        ):
            result = obtain_cves(_make_tool_use())

        text = result["content"][0]["text"]
        assert START_DATE in text
        assert END_DATE in text

    def test_json_contains_filtered_cves_list(self):
        nvd_cves = [_make_nvd_cve("CVE-2026-8011")]
        epss_data = [{"cve": "CVE-2026-8011", "epss": "0.80", "percentile": "0.95"}]
        epss_resp = MagicMock()
        epss_resp.json.return_value = {"data": epss_data}
        epss_resp.raise_for_status = MagicMock()

        with (
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_nvd",
                return_value=nvd_cves,
            ),
            patch(
                "manus_agent.tools.obtain_cves._get_all_cves_from_github",
                return_value=[],
            ),
            patch("manus_agent.tools.obtain_cves.requests.get", return_value=epss_resp),
        ):
            result = obtain_cves(_make_tool_use())

        json_block = next(c["json"] for c in result["content"] if "json" in c)
        assert "cves_with_high_epss" in json_block
        assert isinstance(json_block["cves_with_high_epss"], list)


# =========================================================================
# TOOL_SPEC validation
# =========================================================================


class TestToolSpec:
    """Tests for the tool specification metadata."""

    def test_tool_spec_name(self):
        from manus_agent.tools.obtain_cves import TOOL_SPEC

        assert TOOL_SPEC["name"] == "obtain_cves"

    def test_tool_spec_has_description(self):
        from manus_agent.tools.obtain_cves import TOOL_SPEC

        assert len(TOOL_SPEC["description"]) > 20

    def test_tool_spec_requires_start_and_end_date(self):
        from manus_agent.tools.obtain_cves import TOOL_SPEC

        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "start_date" in schema["properties"]
        assert "end_date" in schema["properties"]
        assert set(schema["required"]) == {"start_date", "end_date"}

    def test_tool_spec_dates_are_strings(self):
        from manus_agent.tools.obtain_cves import TOOL_SPEC

        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert props["start_date"]["type"] == "string"
        assert props["end_date"]["type"] == "string"


# =========================================================================
# Integration: vd_agent references obtain_cves
# =========================================================================


class TestVdAgentIntegration:
    """Verify that obtain_cves is referenced by vd_agent."""

    def test_vd_agent_imports_obtain_cves_module(self):
        import inspect

        from manus_agent.agents import vd_agent

        src = inspect.getsource(vd_agent)
        assert "obtain_cves" in src

    def test_obtain_cves_is_callable(self):
        assert callable(obtain_cves)
