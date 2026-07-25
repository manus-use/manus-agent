"""Comprehensive test suite for the obtain_cves tool module.

Tests cover:
- TOOL_SPEC structure and schema validation
- _get_all_cves_from_nvd: pagination, empty results, retry delegation
- _get_all_cves_from_github: pagination via Link headers, error handling, deduplication
- _filter_cves_by_epss: threshold logic, empty input, API errors
- _enrich_with_cisa_kev: enrichment flags, missing CVEs, API errors
- _submit_in_batches: formatting, batching, POST failures
- obtain_cves (main entry): end-to-end orchestration, merging, dedup, error paths
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# TOOL_SPEC
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Validate the Strands tool spec is well-formed."""

    def test_tool_spec_name(self):
        from manus_agent.tools.obtain_cves import TOOL_SPEC

        assert TOOL_SPEC["name"] == "obtain_cves"

    def test_tool_spec_has_description(self):
        from manus_agent.tools.obtain_cves import TOOL_SPEC

        assert "description" in TOOL_SPEC
        assert len(TOOL_SPEC["description"]) > 10

    def test_tool_spec_input_schema_requires_dates(self):
        from manus_agent.tools.obtain_cves import TOOL_SPEC

        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "start_date" in schema["properties"]
        assert "end_date" in schema["properties"]
        assert "start_date" in schema["required"]
        assert "end_date" in schema["required"]

    def test_tool_spec_date_properties_are_strings(self):
        from manus_agent.tools.obtain_cves import TOOL_SPEC

        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert props["start_date"]["type"] == "string"
        assert props["end_date"]["type"] == "string"


# ---------------------------------------------------------------------------
# _get_all_cves_from_nvd
# ---------------------------------------------------------------------------


class TestGetAllCvesFromNvd:
    """Tests for NVD CVE fetching with pagination."""

    @patch("manus_agent.tools.obtain_cves._nvd_get_with_retry")
    def test_single_page_returns_all_cves(self, mock_retry):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_nvd

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cve": {"id": "CVE-2025-0001"}},
                {"cve": {"id": "CVE-2025-0002"}},
            ],
            "totalResults": 2,
        }
        mock_retry.return_value = mock_resp

        result = _get_all_cves_from_nvd("2025-01-01T00:00:00.000Z", "2025-01-07T00:00:00.000Z")

        assert len(result) == 2
        assert result[0]["cve"]["id"] == "CVE-2025-0001"
        assert mock_retry.call_count == 1

    @patch("manus_agent.tools.obtain_cves._nvd_get_with_retry")
    def test_multi_page_pagination(self, mock_retry):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_nvd

        page1 = MagicMock()
        page1.json.return_value = {
            "vulnerabilities": [{"cve": {"id": f"CVE-2025-{i:04d}"}} for i in range(100)],
            "totalResults": 150,
        }
        page2 = MagicMock()
        page2.json.return_value = {
            "vulnerabilities": [{"cve": {"id": f"CVE-2025-{i:04d}"}} for i in range(100, 150)],
            "totalResults": 150,
        }
        mock_retry.side_effect = [page1, page2]

        result = _get_all_cves_from_nvd("2025-01-01T00:00:00.000Z", "2025-01-14T00:00:00.000Z")

        assert len(result) == 150
        assert mock_retry.call_count == 2

    @patch("manus_agent.tools.obtain_cves._nvd_get_with_retry")
    def test_empty_results(self, mock_retry):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_nvd

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [], "totalResults": 0}
        mock_retry.return_value = mock_resp

        result = _get_all_cves_from_nvd("2025-06-01T00:00:00.000Z", "2025-06-02T00:00:00.000Z")

        assert result == []
        assert mock_retry.call_count == 1

    @patch("manus_agent.tools.obtain_cves._nvd_get_with_retry")
    def test_url_contains_severity_filters(self, mock_retry):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_nvd

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [], "totalResults": 0}
        mock_retry.return_value = mock_resp

        _get_all_cves_from_nvd("2025-01-01T00:00:00.000Z", "2025-01-07T00:00:00.000Z")

        url_called = mock_retry.call_args[0][0]
        assert "cvssV3Severity=HIGH" in url_called
        assert "cvssV3Severity=CRITICAL" in url_called
        assert "cvssV4Severity=HIGH" in url_called
        assert "cvssV4Severity=CRITICAL" in url_called

    @patch("manus_agent.tools.obtain_cves._nvd_get_with_retry")
    def test_url_contains_date_range(self, mock_retry):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_nvd

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [], "totalResults": 0}
        mock_retry.return_value = mock_resp

        _get_all_cves_from_nvd("2025-03-15T00:00:00.000Z", "2025-03-22T00:00:00.000Z")

        url_called = mock_retry.call_args[0][0]
        assert "pubStartDate=2025-03-15T00:00:00.000Z" in url_called
        assert "pubEndDate=2025-03-22T00:00:00.000Z" in url_called

    @patch("manus_agent.tools.obtain_cves._nvd_get_with_retry")
    def test_three_pages_pagination(self, mock_retry):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_nvd

        pages = []
        for page_num in range(3):
            mock_resp = MagicMock()
            start = page_num * 100
            end = min(start + 100, 250)
            mock_resp.json.return_value = {
                "vulnerabilities": [{"cve": {"id": f"CVE-2025-{i:04d}"}} for i in range(start, end)],
                "totalResults": 250,
            }
            pages.append(mock_resp)
        mock_retry.side_effect = pages

        result = _get_all_cves_from_nvd("2025-01-01T00:00:00.000Z", "2025-02-01T00:00:00.000Z")

        assert len(result) == 250
        assert mock_retry.call_count == 3


# ---------------------------------------------------------------------------
# _get_all_cves_from_github
# ---------------------------------------------------------------------------


class TestGetAllCvesFromGithub:
    """Tests for GitHub Advisory fetching with pagination."""

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_single_page_no_pagination(self, mock_get):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_github

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"cve_id": "CVE-2025-1001", "summary": "Test vuln", "published_at": "2025-01-05"},
            {"cve_id": "CVE-2025-1002", "summary": "Another vuln", "published_at": "2025-01-06"},
        ]
        mock_resp.links = {}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _get_all_cves_from_github("2025-01-01T00:00:00.000Z", "2025-01-07T00:00:00.000Z")

        assert len(result) == 2
        assert result[0]["cve"]["id"] == "CVE-2025-1001"
        assert result[0]["cve"]["descriptions"][0]["value"] == "Test vuln"

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_pagination_with_link_header(self, mock_get):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_github

        page1_resp = MagicMock()
        page1_resp.json.return_value = [
            {"cve_id": "CVE-2025-1001", "summary": "Vuln 1", "published_at": "2025-01-01"},
        ]
        page1_resp.links = {"next": {"url": "https://api.github.com/advisories?page=2"}}
        page1_resp.raise_for_status = MagicMock()

        page2_resp = MagicMock()
        page2_resp.json.return_value = [
            {"cve_id": "CVE-2025-1002", "summary": "Vuln 2", "published_at": "2025-01-02"},
        ]
        page2_resp.links = {}
        page2_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [page1_resp, page2_resp]

        result = _get_all_cves_from_github("2025-01-01T00:00:00.000Z", "2025-01-07T00:00:00.000Z")

        assert len(result) == 2
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_skips_advisories_without_cve_id(self, mock_get):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_github

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"cve_id": "CVE-2025-1001", "summary": "Has CVE"},
            {"cve_id": None, "summary": "No CVE"},
            {"summary": "Missing cve_id field entirely"},
        ]
        mock_resp.links = {}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _get_all_cves_from_github("2025-01-01T00:00:00.000Z", "2025-01-07T00:00:00.000Z")

        assert len(result) == 1
        assert result[0]["cve"]["id"] == "CVE-2025-1001"

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_request_exception_returns_empty(self, mock_get):
        import requests

        from manus_agent.tools.obtain_cves import _get_all_cves_from_github

        mock_get.side_effect = requests.exceptions.ConnectionError("Network unreachable")

        result = _get_all_cves_from_github("2025-01-01T00:00:00.000Z", "2025-01-07T00:00:00.000Z")

        assert result == []

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_http_error_returns_empty(self, mock_get):
        import requests

        from manus_agent.tools.obtain_cves import _get_all_cves_from_github

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Forbidden")
        mock_get.return_value = mock_resp

        result = _get_all_cves_from_github("2025-01-01T00:00:00.000Z", "2025-01-07T00:00:00.000Z")

        assert result == []

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_date_extraction_from_iso_format(self, mock_get):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_github

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.links = {}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        _get_all_cves_from_github("2025-03-15T10:30:00.000Z", "2025-03-22T10:30:00.000Z")

        url_called = mock_get.call_args[0][0]
        assert "published=2025-03-15..2025-03-22" in url_called

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_sets_accept_header(self, mock_get):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_github

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.links = {}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        _get_all_cves_from_github("2025-01-01T00:00:00.000Z", "2025-01-07T00:00:00.000Z")

        headers = mock_get.call_args[1]["headers"]
        assert headers["Accept"] == "application/vnd.github+json"

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_cvss_severities_included(self, mock_get):
        from manus_agent.tools.obtain_cves import _get_all_cves_from_github

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "cve_id": "CVE-2025-5001",
                "summary": "With CVSS",
                "published_at": "2025-01-01",
                "cvss_severities": {"cvss_v3": {"score": 9.8}},
            },
        ]
        mock_resp.links = {}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _get_all_cves_from_github("2025-01-01T00:00:00.000Z", "2025-01-07T00:00:00.000Z")

        assert result[0]["cve"]["cvss_score"] == {"cvss_v3": {"score": 9.8}}


# ---------------------------------------------------------------------------
# _filter_cves_by_epss
# ---------------------------------------------------------------------------


class TestFilterCvesByEpss:
    """Tests for EPSS-based CVE filtering."""

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_filters_by_epss_threshold(self, mock_get):
        from manus_agent.tools.obtain_cves import _filter_cves_by_epss

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"cve": "CVE-2025-0001", "epss": "0.10", "percentile": "0.8"},
                {"cve": "CVE-2025-0002", "epss": "0.01", "percentile": "0.2"},
                {"cve": "CVE-2025-0003", "epss": "0.03", "percentile": "0.6"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        cves = [
            {"cve": {"id": "CVE-2025-0001"}},
            {"cve": {"id": "CVE-2025-0002"}},
            {"cve": {"id": "CVE-2025-0003"}},
        ]

        result = _filter_cves_by_epss(cves)

        # CVE-0001: epss > 0.05 → included
        # CVE-0002: epss < 0.05 and percentile < 0.5 → excluded
        # CVE-0003: epss < 0.05 but percentile > 0.5 → included
        assert len(result) == 2
        ids = [c["cve"]["id"] for c in result]
        assert "CVE-2025-0001" in ids
        assert "CVE-2025-0003" in ids
        assert "CVE-2025-0002" not in ids

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_empty_input_returns_empty(self, mock_get):
        from manus_agent.tools.obtain_cves import _filter_cves_by_epss

        result = _filter_cves_by_epss([])

        assert result == []
        mock_get.assert_not_called()

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_enriches_cves_with_epss_data(self, mock_get):
        from manus_agent.tools.obtain_cves import _filter_cves_by_epss

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"cve": "CVE-2025-0001", "epss": "0.50", "percentile": "0.95"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        cves = [{"cve": {"id": "CVE-2025-0001"}}]
        result = _filter_cves_by_epss(cves)

        assert len(result) == 1
        assert result[0]["epss_data"]["epss"] == "0.50"
        assert result[0]["epss_data"]["percentile"] == "0.95"

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_cves_not_in_epss_data_excluded(self, mock_get):
        from manus_agent.tools.obtain_cves import _filter_cves_by_epss

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}  # No EPSS data returned
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        cves = [{"cve": {"id": "CVE-2025-9999"}}]
        result = _filter_cves_by_epss(cves)

        assert result == []

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_epss_url_contains_cve_ids(self, mock_get):
        from manus_agent.tools.obtain_cves import _filter_cves_by_epss

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        cves = [
            {"cve": {"id": "CVE-2025-0001"}},
            {"cve": {"id": "CVE-2025-0002"}},
        ]
        _filter_cves_by_epss(cves)

        url_called = mock_get.call_args[0][0]
        assert "CVE-2025-0001" in url_called
        assert "CVE-2025-0002" in url_called
        assert "api.first.org" in url_called

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_exact_threshold_epss_above_0_05(self, mock_get):
        from manus_agent.tools.obtain_cves import _filter_cves_by_epss

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"cve": "CVE-2025-0001", "epss": "0.05", "percentile": "0.3"},
                {"cve": "CVE-2025-0002", "epss": "0.06", "percentile": "0.3"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        cves = [
            {"cve": {"id": "CVE-2025-0001"}},
            {"cve": {"id": "CVE-2025-0002"}},
        ]
        result = _filter_cves_by_epss(cves)

        # 0.05 is NOT > 0.05, and percentile 0.3 < 0.5, so excluded
        # 0.06 IS > 0.05, so included
        ids = [c["cve"]["id"] for c in result]
        assert "CVE-2025-0001" not in ids
        assert "CVE-2025-0002" in ids

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_exact_threshold_percentile_above_0_5(self, mock_get):
        from manus_agent.tools.obtain_cves import _filter_cves_by_epss

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"cve": "CVE-2025-0001", "epss": "0.01", "percentile": "0.5"},
                {"cve": "CVE-2025-0002", "epss": "0.01", "percentile": "0.51"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        cves = [
            {"cve": {"id": "CVE-2025-0001"}},
            {"cve": {"id": "CVE-2025-0002"}},
        ]
        result = _filter_cves_by_epss(cves)

        # 0.5 is NOT > 0.5, so excluded
        # 0.51 IS > 0.5, so included
        ids = [c["cve"]["id"] for c in result]
        assert "CVE-2025-0001" not in ids
        assert "CVE-2025-0002" in ids


# ---------------------------------------------------------------------------
# _enrich_with_cisa_kev
# ---------------------------------------------------------------------------


class TestEnrichWithCisaKev:
    """Tests for CISA KEV enrichment."""

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_marks_kev_cves_as_true(self, mock_get):
        from manus_agent.tools.obtain_cves import _enrich_with_cisa_kev

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cveID": "CVE-2025-0001"},
                {"cveID": "CVE-2025-0003"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        cves = [
            {"cve": {"id": "CVE-2025-0001"}},
            {"cve": {"id": "CVE-2025-0002"}},
            {"cve": {"id": "CVE-2025-0003"}},
        ]

        result = _enrich_with_cisa_kev(cves)

        assert result[0]["cisa_kev"] is True
        assert result[1]["cisa_kev"] is False
        assert result[2]["cisa_kev"] is True

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_empty_kev_catalog(self, mock_get):
        from manus_agent.tools.obtain_cves import _enrich_with_cisa_kev

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        cves = [{"cve": {"id": "CVE-2025-0001"}}]
        result = _enrich_with_cisa_kev(cves)

        assert result[0]["cisa_kev"] is False

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_fetches_from_cisa_url(self, mock_get):
        from manus_agent.tools.obtain_cves import _enrich_with_cisa_kev

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        _enrich_with_cisa_kev([{"cve": {"id": "CVE-2025-0001"}}])

        url_called = mock_get.call_args[0][0]
        assert "cisa.gov" in url_called
        assert "known_exploited_vulnerabilities" in url_called

    @patch("manus_agent.tools.obtain_cves.requests.get")
    def test_returns_same_list_mutated(self, mock_get):
        from manus_agent.tools.obtain_cves import _enrich_with_cisa_kev

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        original = [{"cve": {"id": "CVE-2025-0001"}}]
        result = _enrich_with_cisa_kev(original)

        assert result is original


# ---------------------------------------------------------------------------
# _submit_in_batches
# ---------------------------------------------------------------------------


class TestSubmitInBatches:
    """Tests for batch submission formatting."""

    @patch("manus_agent.tools.obtain_cves.requests.post")
    def test_formats_cve_fields_correctly(self, mock_post):
        from manus_agent.tools.obtain_cves import _submit_in_batches

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cves = [
            {
                "cve": {
                    "id": "CVE-2025-0001",
                    "descriptions": [{"lang": "en", "value": "Test description"}],
                    "published": "2025-01-01T00:00:00Z",
                    "metrics": {"cvssMetricV31": [{"cvssData": {"baseSeverity": "CRITICAL", "baseScore": 9.8}}]},
                    "weaknesses": [{"description": [{"value": "CWE-79"}]}],
                    "configurations": [
                        {
                            "nodes": [
                                {
                                    "cpeMatch": [
                                        {"vulnerable": True, "criteria": "cpe:2.3:a:vendor:product:1.0:*:*:*:*:*:*:*"}
                                    ]
                                }
                            ]
                        }
                    ],
                },
                "epss_data": {"epss": "0.9", "percentile": "0.99"},
                "cisa_kev": True,
            }
        ]

        _submit_in_batches(cves)

        assert mock_post.call_count == 1
        posted_json = mock_post.call_args[1]["json"]
        assert posted_json["cve_id"] == "CVE-2025-0001"
        assert posted_json["epss_score"] == "0.9"
        assert posted_json["epss_percentile"] == "0.99"
        assert posted_json["cisa_kev"] is True
        assert posted_json["description"] == "Test description"

    @patch("manus_agent.tools.obtain_cves.requests.post")
    def test_handles_missing_metrics(self, mock_post):
        from manus_agent.tools.obtain_cves import _submit_in_batches

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cves = [
            {
                "cve": {
                    "id": "CVE-2025-0002",
                    "descriptions": [{"lang": "en", "value": "Minimal CVE"}],
                    "published": "2025-01-01",
                },
                "epss_data": {},
                "cisa_kev": False,
            }
        ]

        _submit_in_batches(cves)

        posted_json = mock_post.call_args[1]["json"]
        assert posted_json["cve_id"] == "CVE-2025-0002"
        assert posted_json["cisa_kev"] is False

    @patch("manus_agent.tools.obtain_cves.requests.post")
    def test_handles_missing_descriptions(self, mock_post):
        from manus_agent.tools.obtain_cves import _submit_in_batches

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        cves = [
            {
                "cve": {"id": "CVE-2025-0003"},
                "epss_data": {},
                "cisa_kev": False,
            }
        ]

        _submit_in_batches(cves)

        posted_json = mock_post.call_args[1]["json"]
        assert posted_json["description"] == "No description available."

    @patch("manus_agent.tools.obtain_cves.requests.post")
    def test_batches_large_input(self, mock_post):
        from manus_agent.tools.obtain_cves import _submit_in_batches

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        # 250 CVEs → should be 3 batches (100 + 100 + 50) with individual POSTs per CVE
        cves = [
            {
                "cve": {"id": f"CVE-2025-{i:04d}", "descriptions": [{"lang": "en", "value": f"Vuln {i}"}]},
                "epss_data": {},
                "cisa_kev": False,
            }
            for i in range(250)
        ]

        _submit_in_batches(cves)

        # Each CVE gets a POST
        assert mock_post.call_count == 250


# ---------------------------------------------------------------------------
# obtain_cves — main orchestration function
# ---------------------------------------------------------------------------


class TestObtainCvesMain:
    """Tests for the main obtain_cves Strands tool function."""

    @patch("manus_agent.tools.obtain_cves._filter_cves_by_epss")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_github")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_nvd")
    def test_merges_nvd_and_github_results(self, mock_nvd, mock_github, mock_filter):
        from manus_agent.tools.obtain_cves import obtain_cves

        mock_nvd.return_value = [
            {"cve": {"id": "CVE-2025-0001"}},
            {"cve": {"id": "CVE-2025-0002"}},
        ]
        mock_github.return_value = [
            {"cve": {"id": "CVE-2025-0003"}},
        ]
        mock_filter.return_value = []

        tool_use = {
            "toolUseId": "test-id-1",
            "input": {
                "start_date": "2025-01-01T00:00:00.000Z",
                "end_date": "2025-01-07T00:00:00.000Z",
            },
        }

        result = obtain_cves(tool_use)

        assert result["status"] == "success"
        # Should mention 3 total CVEs
        text = result["content"][0]["text"]
        assert "3" in text

    @patch("manus_agent.tools.obtain_cves._filter_cves_by_epss")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_github")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_nvd")
    def test_deduplicates_cves_by_id(self, mock_nvd, mock_github, mock_filter):
        from manus_agent.tools.obtain_cves import obtain_cves

        # Same CVE in both sources
        mock_nvd.return_value = [{"cve": {"id": "CVE-2025-0001", "source": "nvd"}}]
        mock_github.return_value = [{"cve": {"id": "CVE-2025-0001", "source": "github"}}]
        mock_filter.return_value = []

        tool_use = {
            "toolUseId": "test-id-2",
            "input": {
                "start_date": "2025-01-01T00:00:00.000Z",
                "end_date": "2025-01-07T00:00:00.000Z",
            },
        }

        result = obtain_cves(tool_use)

        # Only 1 unique CVE
        text = result["content"][0]["text"]
        assert "Total 1 CVEs found" in text

    @patch("manus_agent.tools.obtain_cves._filter_cves_by_epss")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_github")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_nvd")
    def test_nvd_takes_priority_in_dedup(self, mock_nvd, mock_github, mock_filter):
        from manus_agent.tools.obtain_cves import obtain_cves

        mock_nvd.return_value = [{"cve": {"id": "CVE-2025-0001", "source": "nvd"}}]
        mock_github.return_value = [{"cve": {"id": "CVE-2025-0001", "source": "github"}}]
        mock_filter.return_value = [{"cve": {"id": "CVE-2025-0001", "source": "nvd"}}]

        tool_use = {
            "toolUseId": "test-id-3",
            "input": {
                "start_date": "2025-01-01T00:00:00.000Z",
                "end_date": "2025-01-07T00:00:00.000Z",
            },
        }

        obtain_cves(tool_use)

        # The filter should receive the NVD version (added first to dict)
        filter_arg = mock_filter.call_args[0][0]
        assert filter_arg[0]["cve"]["source"] == "nvd"

    @patch("manus_agent.tools.obtain_cves._filter_cves_by_epss")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_github")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_nvd")
    def test_no_cves_found_returns_success_message(self, mock_nvd, mock_github, mock_filter):
        from manus_agent.tools.obtain_cves import obtain_cves

        mock_nvd.return_value = []
        mock_github.return_value = []

        tool_use = {
            "toolUseId": "test-id-4",
            "input": {
                "start_date": "2025-06-01T00:00:00.000Z",
                "end_date": "2025-06-02T00:00:00.000Z",
            },
        }

        result = obtain_cves(tool_use)

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-id-4"
        assert "No new high/critical CVEs found" in result["content"][0]["text"]
        mock_filter.assert_not_called()

    @patch("manus_agent.tools.obtain_cves._filter_cves_by_epss")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_github")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_nvd")
    def test_exception_returns_error_status(self, mock_nvd, mock_github, mock_filter):
        from manus_agent.tools.obtain_cves import obtain_cves

        mock_nvd.side_effect = RuntimeError("Connection pool exhausted")

        tool_use = {
            "toolUseId": "test-id-5",
            "input": {
                "start_date": "2025-01-01T00:00:00.000Z",
                "end_date": "2025-01-07T00:00:00.000Z",
            },
        }

        result = obtain_cves(tool_use)

        assert result["status"] == "error"
        assert "Connection pool exhausted" in result["content"][0]["text"]

    @patch("manus_agent.tools.obtain_cves._filter_cves_by_epss")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_github")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_nvd")
    def test_result_includes_json_payload(self, mock_nvd, mock_github, mock_filter):
        from manus_agent.tools.obtain_cves import obtain_cves

        mock_nvd.return_value = [{"cve": {"id": "CVE-2025-0001"}}]
        mock_github.return_value = []
        mock_filter.return_value = [{"cve": {"id": "CVE-2025-0001"}, "epss_data": {"epss": "0.5"}}]

        tool_use = {
            "toolUseId": "test-id-6",
            "input": {
                "start_date": "2025-01-01T00:00:00.000Z",
                "end_date": "2025-01-07T00:00:00.000Z",
            },
        }

        result = obtain_cves(tool_use)

        assert result["status"] == "success"
        json_content = result["content"][1]["json"]
        assert json_content["total_found"] == 1
        assert json_content["total_with_high_epss"] == 1
        assert len(json_content["cves_with_high_epss"]) == 1

    @patch("manus_agent.tools.obtain_cves._filter_cves_by_epss")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_github")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_nvd")
    def test_filters_in_chunks_of_100(self, mock_nvd, mock_github, mock_filter):
        from manus_agent.tools.obtain_cves import obtain_cves

        # 250 CVEs → should call filter 3 times (100, 100, 50)
        mock_nvd.return_value = [{"cve": {"id": f"CVE-2025-{i:04d}"}} for i in range(250)]
        mock_github.return_value = []
        mock_filter.return_value = []

        tool_use = {
            "toolUseId": "test-id-7",
            "input": {
                "start_date": "2025-01-01T00:00:00.000Z",
                "end_date": "2025-02-01T00:00:00.000Z",
            },
        }

        obtain_cves(tool_use)

        assert mock_filter.call_count == 3
        # First chunk is 100
        assert len(mock_filter.call_args_list[0][0][0]) == 100
        # Second chunk is 100
        assert len(mock_filter.call_args_list[1][0][0]) == 100
        # Third chunk is 50
        assert len(mock_filter.call_args_list[2][0][0]) == 50

    @patch("manus_agent.tools.obtain_cves._filter_cves_by_epss")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_github")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_nvd")
    def test_result_text_includes_date_range(self, mock_nvd, mock_github, mock_filter):
        from manus_agent.tools.obtain_cves import obtain_cves

        mock_nvd.return_value = [{"cve": {"id": "CVE-2025-0001"}}]
        mock_github.return_value = []
        mock_filter.return_value = []

        tool_use = {
            "toolUseId": "test-id-8",
            "input": {
                "start_date": "2025-03-15T00:00:00.000Z",
                "end_date": "2025-03-22T00:00:00.000Z",
            },
        }

        result = obtain_cves(tool_use)

        text = result["content"][0]["text"]
        assert "2025-03-15" in text
        assert "2025-03-22" in text

    @patch("manus_agent.tools.obtain_cves._filter_cves_by_epss")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_github")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_nvd")
    def test_tool_use_id_propagated(self, mock_nvd, mock_github, mock_filter):
        from manus_agent.tools.obtain_cves import obtain_cves

        mock_nvd.return_value = []
        mock_github.return_value = []

        tool_use = {
            "toolUseId": "custom-uuid-12345",
            "input": {
                "start_date": "2025-01-01T00:00:00.000Z",
                "end_date": "2025-01-07T00:00:00.000Z",
            },
        }

        result = obtain_cves(tool_use)

        assert result["toolUseId"] == "custom-uuid-12345"

    @patch("manus_agent.tools.obtain_cves._filter_cves_by_epss")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_github")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_nvd")
    def test_github_only_cves_included(self, mock_nvd, mock_github, mock_filter):
        from manus_agent.tools.obtain_cves import obtain_cves

        mock_nvd.return_value = []
        mock_github.return_value = [
            {"cve": {"id": "CVE-2025-9001"}},
            {"cve": {"id": "CVE-2025-9002"}},
        ]
        mock_filter.return_value = [{"cve": {"id": "CVE-2025-9001"}}]

        tool_use = {
            "toolUseId": "test-id-9",
            "input": {
                "start_date": "2025-01-01T00:00:00.000Z",
                "end_date": "2025-01-07T00:00:00.000Z",
            },
        }

        result = obtain_cves(tool_use)

        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "2 CVEs found" in text

    @patch("manus_agent.tools.obtain_cves._filter_cves_by_epss")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_github")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_nvd")
    def test_filter_exception_propagates_as_error(self, mock_nvd, mock_github, mock_filter):
        from manus_agent.tools.obtain_cves import obtain_cves

        mock_nvd.return_value = [{"cve": {"id": "CVE-2025-0001"}}]
        mock_github.return_value = []
        mock_filter.side_effect = Exception("EPSS API timeout")

        tool_use = {
            "toolUseId": "test-id-10",
            "input": {
                "start_date": "2025-01-01T00:00:00.000Z",
                "end_date": "2025-01-07T00:00:00.000Z",
            },
        }

        result = obtain_cves(tool_use)

        assert result["status"] == "error"
        assert "EPSS API timeout" in result["content"][0]["text"]

    @patch("manus_agent.tools.obtain_cves._filter_cves_by_epss")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_github")
    @patch("manus_agent.tools.obtain_cves._get_all_cves_from_nvd")
    def test_high_epss_rate_calculation(self, mock_nvd, mock_github, mock_filter):
        from manus_agent.tools.obtain_cves import obtain_cves

        mock_nvd.return_value = [{"cve": {"id": f"CVE-2025-{i:04d}"}} for i in range(10)]
        mock_github.return_value = []
        mock_filter.return_value = [{"cve": {"id": f"CVE-2025-{i:04d}"}} for i in range(3)]

        tool_use = {
            "toolUseId": "test-id-11",
            "input": {
                "start_date": "2025-01-01T00:00:00.000Z",
                "end_date": "2025-01-07T00:00:00.000Z",
            },
        }

        result = obtain_cves(tool_use)

        text = result["content"][0]["text"]
        # Should report "3/10" ratio
        assert "3" in text
        assert "10" in text
        json_content = result["content"][1]["json"]
        assert json_content["total_found"] == 10
        assert json_content["total_with_high_epss"] == 3


# ---------------------------------------------------------------------------
# Module-level imports and exports
# ---------------------------------------------------------------------------


class TestModuleStructure:
    """Tests for module-level structure."""

    def test_module_imports_cleanly(self):
        import manus_agent.tools.obtain_cves  # noqa: F401

    def test_obtain_cves_is_callable(self):
        from manus_agent.tools.obtain_cves import obtain_cves

        assert callable(obtain_cves)

    def test_tool_spec_is_dict(self):
        from manus_agent.tools.obtain_cves import TOOL_SPEC

        assert isinstance(TOOL_SPEC, dict)

    def test_helper_functions_importable(self):
        from manus_agent.tools.obtain_cves import (  # noqa: F401
            _enrich_with_cisa_kev,
            _filter_cves_by_epss,
            _get_all_cves_from_github,
            _get_all_cves_from_nvd,
            _submit_in_batches,
        )
