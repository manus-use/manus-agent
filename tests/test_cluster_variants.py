#!/usr/bin/env python3
"""Comprehensive test suite for cluster_variants tool and CLI subcommand."""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from manus_agent.tools.cluster_variants import (
    TOOL_SPEC,
    _build_clusters,
    _extract_cpe_vendors_products,
    _extract_cwes,
    _extract_reference_domains,
    _format_text,
    _nvd_get,
    _search_by_cpe,
    _search_by_cwe,
    _search_by_reference_domain,
    _summarize_cve,
    cluster_variants,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tool_use(cve_id):
    return {"toolUseId": "test-id-001", "input": {"cve_id": cve_id}}


def _nvd_cve_response(cve_id="CVE-2021-44228", cwes=None, vendor="apache", product="log4j"):
    """Build a realistic NVD API response for testing."""
    if cwes is None:
        cwes = ["CWE-502"]
    cve_data = {
        "id": cve_id,
        "descriptions": [{"lang": "en", "value": f"Test description for {cve_id}"}],
        "weaknesses": [{"description": [{"lang": "en", "value": cwe}]} for cwe in cwes],
        "configurations": [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "criteria": f"cpe:2.3:a:{vendor}:{product}:2.14.0:*:*:*:*:*:*:*",
                                "vulnerable": True,
                            }
                        ]
                    }
                ]
            }
        ],
        "references": [
            {"url": "https://logging.apache.org/log4j/2.x/security.html"},
            {"url": "https://www.lunasec.io/docs/blog/log4j-zero-day/"},
            {"url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228"},
        ],
        "metrics": {
            "cvssMetricV31": [{"cvssData": {"baseScore": 10.0, "vectorString": "AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}}]
        },
        "published": "2021-12-10T10:15:00.000",
    }
    return {"vulnerabilities": [{"cve": cve_data}]}


def _nvd_search_response(cve_ids):
    """Build NVD search response with multiple CVEs."""
    vulns = []
    for cve_id in cve_ids:
        vulns.append(
            {
                "cve": {
                    "id": cve_id,
                    "descriptions": [{"lang": "en", "value": f"Description of {cve_id}"}],
                    "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 7.5}}]},
                    "published": "2022-01-15T10:00:00.000",
                    "references": [
                        {"url": "https://logging.apache.org/advisory.html"},
                    ],
                }
            }
        )
    return {"vulnerabilities": vulns}


# ---------------------------------------------------------------------------
# TOOL_SPEC contract tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_has_required_keys(self):
        assert "name" in TOOL_SPEC
        assert "description" in TOOL_SPEC
        assert "inputSchema" in TOOL_SPEC

    def test_name_matches_module(self):
        assert TOOL_SPEC["name"] == "cluster_variants"

    def test_input_schema_requires_cve_id(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_description_is_nonempty(self):
        assert len(TOOL_SPEC["description"]) > 20


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_cve_id(self):
        tool = {"toolUseId": "t1", "input": {}}
        result = cluster_variants(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_empty_string(self):
        result = cluster_variants(_make_tool_use(""))
        assert result["status"] == "error"

    def test_non_string_input(self):
        tool = {"toolUseId": "t1", "input": {"cve_id": 12345}}
        result = cluster_variants(tool)
        assert result["status"] == "error"

    def test_invalid_format_no_prefix(self):
        result = cluster_variants(_make_tool_use("2021-44228"))
        assert result["status"] == "error"

    def test_invalid_format_short_number(self):
        result = cluster_variants(_make_tool_use("CVE-2021-12"))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# HTTP retry tests
# ---------------------------------------------------------------------------


class TestHttpRetry:
    @patch("manus_agent.tools.cluster_variants.time.sleep")
    @patch("manus_agent.tools.cluster_variants.requests.get")
    def test_retries_on_429(self, mock_get, mock_sleep):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp_429)

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.raise_for_status.return_value = None

        mock_get.side_effect = [resp_429, resp_ok]
        result = _nvd_get("https://example.com")
        assert result == resp_ok
        assert mock_sleep.called

    @patch("manus_agent.tools.cluster_variants.time.sleep")
    @patch("manus_agent.tools.cluster_variants.requests.get")
    def test_retries_on_503(self, mock_get, mock_sleep):
        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_503.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp_503)

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.raise_for_status.return_value = None

        mock_get.side_effect = [resp_503, resp_503, resp_ok]
        result = _nvd_get("https://example.com")
        assert result == resp_ok

    @patch("manus_agent.tools.cluster_variants.time.sleep")
    @patch("manus_agent.tools.cluster_variants.requests.get")
    def test_raises_after_max_retries(self, mock_get, mock_sleep):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp_429)
        mock_get.return_value = resp_429

        with pytest.raises(requests.exceptions.HTTPError):
            _nvd_get("https://example.com")

    @patch("manus_agent.tools.cluster_variants.requests.get")
    def test_no_retry_on_404(self, mock_get):
        resp_404 = MagicMock()
        resp_404.status_code = 404
        resp_404.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp_404)
        mock_get.return_value = resp_404

        with pytest.raises(requests.exceptions.HTTPError):
            _nvd_get("https://example.com")
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.cluster_variants.time.sleep")
    @patch("manus_agent.tools.cluster_variants.requests.get")
    def test_retries_on_connection_error(self, mock_get, mock_sleep):
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.raise_for_status.return_value = None

        mock_get.side_effect = [
            requests.exceptions.ConnectionError("network error"),
            resp_ok,
        ]
        result = _nvd_get("https://example.com")
        assert result == resp_ok


# ---------------------------------------------------------------------------
# CPE extraction tests
# ---------------------------------------------------------------------------


class TestExtractCpeVendorsProducts:
    def test_basic_extraction(self):
        cve_data = {
            "configurations": [
                {
                    "nodes": [
                        {
                            "cpeMatch": [
                                {"criteria": "cpe:2.3:a:apache:log4j:2.14.0:*:*:*:*:*:*:*"},
                            ]
                        }
                    ]
                }
            ]
        }
        result = _extract_cpe_vendors_products(cve_data)
        assert result == [("apache", "log4j")]

    def test_multiple_products(self):
        cve_data = {
            "configurations": [
                {
                    "nodes": [
                        {
                            "cpeMatch": [
                                {"criteria": "cpe:2.3:a:apache:log4j:2.14.0:*:*:*:*:*:*:*"},
                                {"criteria": "cpe:2.3:a:apache:tomcat:9.0.50:*:*:*:*:*:*:*"},
                            ]
                        }
                    ]
                }
            ]
        }
        result = _extract_cpe_vendors_products(cve_data)
        assert ("apache", "log4j") in result
        assert ("apache", "tomcat") in result

    def test_deduplicates(self):
        cve_data = {
            "configurations": [
                {
                    "nodes": [
                        {
                            "cpeMatch": [
                                {"criteria": "cpe:2.3:a:apache:log4j:2.14.0:*:*:*:*:*:*:*"},
                                {"criteria": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"},
                            ]
                        }
                    ]
                }
            ]
        }
        result = _extract_cpe_vendors_products(cve_data)
        assert result == [("apache", "log4j")]

    def test_skips_wildcard_vendor(self):
        cve_data = {
            "configurations": [{"nodes": [{"cpeMatch": [{"criteria": "cpe:2.3:a:*:log4j:2.14.0:*:*:*:*:*:*:*"}]}]}]
        }
        result = _extract_cpe_vendors_products(cve_data)
        assert result == []

    def test_empty_configurations(self):
        result = _extract_cpe_vendors_products({"configurations": []})
        assert result == []

    def test_missing_configurations(self):
        result = _extract_cpe_vendors_products({})
        assert result == []

    def test_short_cpe_string_ignored(self):
        cve_data = {"configurations": [{"nodes": [{"cpeMatch": [{"criteria": "cpe:2.3:a:vendor"}]}]}]}
        result = _extract_cpe_vendors_products(cve_data)
        assert result == []


# ---------------------------------------------------------------------------
# CWE extraction tests
# ---------------------------------------------------------------------------


class TestExtractCwes:
    def test_basic_extraction(self):
        cve_data = {"weaknesses": [{"description": [{"lang": "en", "value": "CWE-502"}]}]}
        assert _extract_cwes(cve_data) == ["CWE-502"]

    def test_multiple_cwes(self):
        cve_data = {
            "weaknesses": [
                {"description": [{"lang": "en", "value": "CWE-502"}]},
                {"description": [{"lang": "en", "value": "CWE-917"}]},
            ]
        }
        result = _extract_cwes(cve_data)
        assert "CWE-502" in result
        assert "CWE-917" in result

    def test_deduplicates(self):
        cve_data = {
            "weaknesses": [
                {"description": [{"lang": "en", "value": "CWE-79"}]},
                {"description": [{"lang": "en", "value": "CWE-79"}]},
            ]
        }
        assert _extract_cwes(cve_data) == ["CWE-79"]

    def test_skips_noinfo(self):
        cve_data = {"weaknesses": [{"description": [{"lang": "en", "value": "CWE-noinfo"}]}]}
        assert _extract_cwes(cve_data) == []

    def test_case_insensitive(self):
        cve_data = {"weaknesses": [{"description": [{"lang": "en", "value": "cwe-79"}]}]}
        assert _extract_cwes(cve_data) == ["CWE-79"]

    def test_empty_weaknesses(self):
        assert _extract_cwes({"weaknesses": []}) == []

    def test_missing_weaknesses(self):
        assert _extract_cwes({}) == []


# ---------------------------------------------------------------------------
# Reference domain extraction tests
# ---------------------------------------------------------------------------


class TestExtractReferenceDomains:
    def test_basic_extraction(self):
        cve_data = {
            "references": [
                {"url": "https://logging.apache.org/log4j/security.html"},
            ]
        }
        result = _extract_reference_domains(cve_data)
        assert "logging.apache.org" in result

    def test_strips_www(self):
        cve_data = {
            "references": [
                {"url": "https://www.example.com/advisory"},
            ]
        }
        result = _extract_reference_domains(cve_data)
        assert "example.com" in result

    def test_filters_ignored_domains(self):
        cve_data = {
            "references": [
                {"url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228"},
                {"url": "https://github.com/advisories/GHSA-xxx"},
                {"url": "https://logging.apache.org/security.html"},
            ]
        }
        result = _extract_reference_domains(cve_data)
        assert "nvd.nist.gov" not in result
        assert "github.com" not in result
        assert "logging.apache.org" in result

    def test_deduplicates(self):
        cve_data = {
            "references": [
                {"url": "https://example.com/a"},
                {"url": "https://example.com/b"},
            ]
        }
        result = _extract_reference_domains(cve_data)
        assert result.count("example.com") == 1

    def test_empty_references(self):
        assert _extract_reference_domains({"references": []}) == []

    def test_missing_references(self):
        assert _extract_reference_domains({}) == []

    def test_invalid_url_skipped(self):
        cve_data = {
            "references": [
                {"url": "not-a-valid-url"},
                {"url": "https://valid.example.com/page"},
            ]
        }
        result = _extract_reference_domains(cve_data)
        assert "valid.example.com" in result


# ---------------------------------------------------------------------------
# CVE summarize tests
# ---------------------------------------------------------------------------


class TestSummarizeCve:
    def test_basic_summary(self):
        cve = {
            "id": "CVE-2022-1234",
            "descriptions": [{"lang": "en", "value": "A test vulnerability"}],
            "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 9.8}}]},
            "published": "2022-05-01T10:00:00.000",
        }
        result = _summarize_cve(cve)
        assert result["cve_id"] == "CVE-2022-1234"
        assert result["description"] == "A test vulnerability"
        assert result["cvss_score"] == 9.8
        assert result["published"] == "2022-05-01"

    def test_truncates_long_description(self):
        cve = {
            "id": "CVE-2022-1234",
            "descriptions": [{"lang": "en", "value": "A" * 300}],
            "metrics": {},
            "published": "2022-05-01T10:00:00.000",
        }
        result = _summarize_cve(cve)
        assert len(result["description"]) == 200
        assert result["description"].endswith("...")

    def test_missing_cvss(self):
        cve = {
            "id": "CVE-2022-1234",
            "descriptions": [{"lang": "en", "value": "Test"}],
            "metrics": {},
            "published": "",
        }
        result = _summarize_cve(cve)
        assert result["cvss_score"] is None

    def test_falls_back_to_cvss_v30(self):
        cve = {
            "id": "CVE-2022-1234",
            "descriptions": [{"lang": "en", "value": "Test"}],
            "metrics": {"cvssMetricV30": [{"cvssData": {"baseScore": 7.0}}]},
            "published": "",
        }
        result = _summarize_cve(cve)
        assert result["cvss_score"] == 7.0

    def test_falls_back_to_non_english_description(self):
        cve = {
            "id": "CVE-2022-1234",
            "descriptions": [{"lang": "es", "value": "Una vulnerabilidad"}],
            "metrics": {},
            "published": "",
        }
        result = _summarize_cve(cve)
        assert result["description"] == "Una vulnerabilidad"


# ---------------------------------------------------------------------------
# Search by CPE tests
# ---------------------------------------------------------------------------


class TestSearchByCpe:
    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_returns_matching_cves(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _nvd_search_response(["CVE-2022-0001", "CVE-2022-0002"])
        mock_get.return_value = mock_resp

        results = _search_by_cpe("apache", "log4j", "CVE-2021-44228")
        assert len(results) == 2
        assert results[0]["cve_id"] == "CVE-2022-0001"

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_excludes_seed_cve(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _nvd_search_response(["CVE-2021-44228", "CVE-2022-0001"])
        mock_get.return_value = mock_resp

        results = _search_by_cpe("apache", "log4j", "CVE-2021-44228")
        assert len(results) == 1
        assert results[0]["cve_id"] == "CVE-2022-0001"

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_handles_empty_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        results = _search_by_cpe("unknown", "pkg", "CVE-2021-44228")
        assert results == []

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_handles_exception(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("fail")
        results = _search_by_cpe("apache", "log4j", "CVE-2021-44228")
        assert results == []


# ---------------------------------------------------------------------------
# Search by CWE tests
# ---------------------------------------------------------------------------


class TestSearchByCwe:
    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_returns_matching_cves(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _nvd_search_response(["CVE-2022-1111", "CVE-2022-2222"])
        mock_get.return_value = mock_resp

        results = _search_by_cwe("CWE-502", "CVE-2021-44228")
        assert len(results) == 2

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_excludes_seed_cve(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _nvd_search_response(["CVE-2021-44228", "CVE-2022-1111"])
        mock_get.return_value = mock_resp

        results = _search_by_cwe("CWE-502", "CVE-2021-44228")
        assert len(results) == 1

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_handles_exception(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        results = _search_by_cwe("CWE-79", "CVE-2021-44228")
        assert results == []


# ---------------------------------------------------------------------------
# Search by reference domain tests
# ---------------------------------------------------------------------------


class TestSearchByReferenceDomain:
    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_returns_matching_cves_with_domain(self, mock_get):
        mock_resp = MagicMock()
        resp_data = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2022-3333",
                        "descriptions": [{"lang": "en", "value": "Test"}],
                        "metrics": {},
                        "published": "2022-03-01T00:00:00.000",
                        "references": [
                            {"url": "https://logging.apache.org/advisory.html"},
                        ],
                    }
                }
            ]
        }
        mock_resp.json.return_value = resp_data
        mock_get.return_value = mock_resp

        results = _search_by_reference_domain("logging.apache.org", "CVE-2021-44228")
        assert len(results) == 1
        assert results[0]["cve_id"] == "CVE-2022-3333"

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_filters_out_non_matching_domain(self, mock_get):
        mock_resp = MagicMock()
        resp_data = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2022-4444",
                        "descriptions": [{"lang": "en", "value": "Test"}],
                        "metrics": {},
                        "published": "2022-04-01T00:00:00.000",
                        "references": [
                            {"url": "https://other-domain.com/page"},
                        ],
                    }
                }
            ]
        }
        mock_resp.json.return_value = resp_data
        mock_get.return_value = mock_resp

        results = _search_by_reference_domain("logging.apache.org", "CVE-2021-44228")
        assert results == []

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_handles_exception(self, mock_get):
        mock_get.side_effect = Exception("fail")
        results = _search_by_reference_domain("example.com", "CVE-2021-44228")
        assert results == []

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_excludes_seed_cve(self, mock_get):
        mock_resp = MagicMock()
        resp_data = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2021-44228",
                        "descriptions": [{"lang": "en", "value": "Seed"}],
                        "metrics": {},
                        "published": "2021-12-10T00:00:00.000",
                        "references": [
                            {"url": "https://logging.apache.org/log4j"},
                        ],
                    }
                }
            ]
        }
        mock_resp.json.return_value = resp_data
        mock_get.return_value = mock_resp

        results = _search_by_reference_domain("logging.apache.org", "CVE-2021-44228")
        assert results == []


# ---------------------------------------------------------------------------
# Build clusters integration tests
# ---------------------------------------------------------------------------


class TestBuildClusters:
    @patch("manus_agent.tools.cluster_variants._search_by_reference_domain")
    @patch("manus_agent.tools.cluster_variants._search_by_cwe")
    @patch("manus_agent.tools.cluster_variants._search_by_cpe")
    def test_returns_all_dimensions(self, mock_cpe, mock_cwe, mock_ref):
        mock_cpe.return_value = [
            {"cve_id": "CVE-2022-0001", "description": "A", "cvss_score": 7.0, "published": "2022-01-01"}
        ]
        mock_cwe.return_value = [
            {"cve_id": "CVE-2022-0002", "description": "B", "cvss_score": 8.0, "published": "2022-02-01"}
        ]
        mock_ref.return_value = [
            {"cve_id": "CVE-2022-0003", "description": "C", "cvss_score": 6.0, "published": "2022-03-01"}
        ]

        cve_data = _nvd_cve_response()["vulnerabilities"][0]["cve"]
        clusters = _build_clusters("CVE-2021-44228", cve_data)

        assert clusters["seed_cve"] == "CVE-2021-44228"
        assert "component" in clusters["dimensions"]
        assert "weakness" in clusters["dimensions"]
        assert "researcher" in clusters["dimensions"]
        assert clusters["total_unique_variants"] == 3

    @patch("manus_agent.tools.cluster_variants._search_by_reference_domain")
    @patch("manus_agent.tools.cluster_variants._search_by_cwe")
    @patch("manus_agent.tools.cluster_variants._search_by_cpe")
    def test_deduplicates_across_dimensions(self, mock_cpe, mock_cwe, mock_ref):
        same_cve = {"cve_id": "CVE-2022-0001", "description": "Dup", "cvss_score": 7.0, "published": "2022-01-01"}
        mock_cpe.return_value = [same_cve]
        mock_cwe.return_value = [same_cve]
        mock_ref.return_value = [same_cve]

        cve_data = _nvd_cve_response()["vulnerabilities"][0]["cve"]
        clusters = _build_clusters("CVE-2021-44228", cve_data)

        # Should only appear once across all dimensions
        assert clusters["total_unique_variants"] == 1

    @patch("manus_agent.tools.cluster_variants._search_by_reference_domain")
    @patch("manus_agent.tools.cluster_variants._search_by_cwe")
    @patch("manus_agent.tools.cluster_variants._search_by_cpe")
    def test_empty_clusters(self, mock_cpe, mock_cwe, mock_ref):
        mock_cpe.return_value = []
        mock_cwe.return_value = []
        mock_ref.return_value = []

        cve_data = _nvd_cve_response()["vulnerabilities"][0]["cve"]
        clusters = _build_clusters("CVE-2021-44228", cve_data)
        assert clusters["total_unique_variants"] == 0

    @patch("manus_agent.tools.cluster_variants._search_by_reference_domain")
    @patch("manus_agent.tools.cluster_variants._search_by_cwe")
    @patch("manus_agent.tools.cluster_variants._search_by_cpe")
    def test_no_cpe_no_cwe(self, mock_cpe, mock_cwe, mock_ref):
        """CVE with no CPE/CWE data still returns valid structure."""
        mock_cpe.return_value = []
        mock_cwe.return_value = []
        mock_ref.return_value = []

        cve_data = {"id": "CVE-2021-44228", "references": [], "weaknesses": [], "configurations": []}
        clusters = _build_clusters("CVE-2021-44228", cve_data)
        assert clusters["dimensions"]["component"]["search_keys"] == []
        assert clusters["dimensions"]["weakness"]["search_keys"] == []


# ---------------------------------------------------------------------------
# Text formatting tests
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_includes_seed_cve(self):
        clusters = {
            "seed_cve": "CVE-2021-44228",
            "dimensions": {
                "component": {"search_keys": ["apache:log4j"], "variants": []},
                "weakness": {"search_keys": ["CWE-502"], "variants": []},
                "researcher": {"search_keys": ["lunasec.io"], "variants": []},
            },
            "total_unique_variants": 0,
        }
        text = _format_text(clusters)
        assert "CVE-2021-44228" in text

    def test_includes_dimension_labels(self):
        clusters = {
            "seed_cve": "CVE-2021-44228",
            "dimensions": {
                "component": {"search_keys": [], "variants": []},
                "weakness": {"search_keys": [], "variants": []},
                "researcher": {"search_keys": [], "variants": []},
            },
            "total_unique_variants": 0,
        }
        text = _format_text(clusters)
        assert "Same Component/Vendor" in text
        assert "Same CWE Weakness Class" in text
        assert "Same Researcher/Disclosure Domain" in text

    def test_includes_variant_details(self):
        clusters = {
            "seed_cve": "CVE-2021-44228",
            "dimensions": {
                "component": {
                    "search_keys": ["apache:log4j"],
                    "variants": [
                        {
                            "cve_id": "CVE-2022-0001",
                            "description": "Test vuln",
                            "cvss_score": 9.0,
                            "published": "2022-01-01",
                        }
                    ],
                },
                "weakness": {"search_keys": [], "variants": []},
                "researcher": {"search_keys": [], "variants": []},
            },
            "total_unique_variants": 1,
        }
        text = _format_text(clusters)
        assert "CVE-2022-0001" in text
        assert "CVSS 9.0" in text
        assert "Test vuln" in text

    def test_no_variants_message(self):
        clusters = {
            "seed_cve": "CVE-2021-44228",
            "dimensions": {
                "component": {"search_keys": [], "variants": []},
                "weakness": {"search_keys": [], "variants": []},
                "researcher": {"search_keys": [], "variants": []},
            },
            "total_unique_variants": 0,
        }
        text = _format_text(clusters)
        assert "no variants found" in text

    def test_total_count_in_output(self):
        clusters = {
            "seed_cve": "CVE-2021-44228",
            "dimensions": {
                "component": {
                    "search_keys": [],
                    "variants": [{"cve_id": "CVE-A", "description": "", "cvss_score": None, "published": ""}],
                },
                "weakness": {"search_keys": [], "variants": []},
                "researcher": {"search_keys": [], "variants": []},
            },
            "total_unique_variants": 1,
        }
        text = _format_text(clusters)
        assert "Total unique variants: 1" in text


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------


class TestClusterVariantsHandler:
    @patch("manus_agent.tools.cluster_variants._nvd_get")
    @patch("manus_agent.tools.cluster_variants._search_by_reference_domain")
    @patch("manus_agent.tools.cluster_variants._search_by_cwe")
    @patch("manus_agent.tools.cluster_variants._search_by_cpe")
    def test_success(self, mock_cpe, mock_cwe, mock_ref, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _nvd_cve_response()
        mock_get.return_value = mock_resp
        mock_cpe.return_value = []
        mock_cwe.return_value = []
        mock_ref.return_value = []

        result = cluster_variants(_make_tool_use("CVE-2021-44228"))
        assert result["status"] == "success"
        assert len(result["content"]) == 2  # text + json
        assert "CVE-2021-44228" in result["content"][0]["text"]

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_nvd_network_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("fail")
        result = cluster_variants(_make_tool_use("CVE-2021-44228"))
        assert result["status"] == "error"
        assert "Failed to fetch" in result["content"][0]["text"]

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_nvd_empty_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        result = cluster_variants(_make_tool_use("CVE-2021-44228"))
        assert result["status"] == "error"
        assert "No vulnerability data" in result["content"][0]["text"]

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_nvd_json_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.side_effect = json.JSONDecodeError("err", "", 0)
        mock_get.return_value = mock_resp

        result = cluster_variants(_make_tool_use("CVE-2021-44228"))
        assert result["status"] == "error"
        assert "Failed to parse" in result["content"][0]["text"]

    def test_case_insensitive_cve_id(self):
        with patch("manus_agent.tools.cluster_variants._nvd_get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _nvd_cve_response()
            mock_get.return_value = mock_resp

            with patch("manus_agent.tools.cluster_variants._search_by_cpe", return_value=[]):
                with patch("manus_agent.tools.cluster_variants._search_by_cwe", return_value=[]):
                    with patch("manus_agent.tools.cluster_variants._search_by_reference_domain", return_value=[]):
                        result = cluster_variants(_make_tool_use("cve-2021-44228"))
                        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# CLI subcommand tests
# ---------------------------------------------------------------------------


class TestCliClusterVariants:
    @patch("manus_agent.tools.cluster_variants._nvd_get")
    @patch("manus_agent.tools.cluster_variants._search_by_reference_domain")
    @patch("manus_agent.tools.cluster_variants._search_by_cwe")
    @patch("manus_agent.tools.cluster_variants._search_by_cpe")
    def test_text_output(self, mock_cpe, mock_cwe, mock_ref, mock_get, capsys):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _nvd_cve_response()
        mock_get.return_value = mock_resp
        mock_cpe.return_value = []
        mock_cwe.return_value = []
        mock_ref.return_value = []

        from manus_agent.cli import _run_cluster_variants

        rc = _run_cluster_variants(["CVE-2021-44228"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "CVE-2021-44228" in captured.out

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    @patch("manus_agent.tools.cluster_variants._search_by_reference_domain")
    @patch("manus_agent.tools.cluster_variants._search_by_cwe")
    @patch("manus_agent.tools.cluster_variants._search_by_cpe")
    def test_json_output(self, mock_cpe, mock_cwe, mock_ref, mock_get, capsys):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _nvd_cve_response()
        mock_get.return_value = mock_resp
        mock_cpe.return_value = []
        mock_cwe.return_value = []
        mock_ref.return_value = []

        from manus_agent.cli import _run_cluster_variants

        rc = _run_cluster_variants(["CVE-2021-44228", "--output", "json"])
        assert rc == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["seed_cve"] == "CVE-2021-44228"

    def test_invalid_cve_id(self, capsys):
        from manus_agent.cli import _run_cluster_variants

        rc = _run_cluster_variants(["not-a-cve"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Invalid CVE ID" in captured.err

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_nvd_error(self, mock_get, capsys):
        mock_get.side_effect = requests.exceptions.ConnectionError("network")

        from manus_agent.cli import _run_cluster_variants

        rc = _run_cluster_variants(["CVE-2021-44228"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    def test_no_vulnerabilities(self, mock_get, capsys):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        from manus_agent.cli import _run_cluster_variants

        rc = _run_cluster_variants(["CVE-2021-44228"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "No vulnerability data" in captured.err


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @patch("manus_agent.tools.cluster_variants._nvd_get")
    @patch("manus_agent.tools.cluster_variants._search_by_reference_domain")
    @patch("manus_agent.tools.cluster_variants._search_by_cwe")
    @patch("manus_agent.tools.cluster_variants._search_by_cpe")
    def test_cve_with_no_metadata(self, mock_cpe, mock_cwe, mock_ref, mock_get):
        """CVE with minimal NVD data (no CPE, no CWE, no refs)."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2099-0001",
                        "descriptions": [{"lang": "en", "value": "Empty CVE"}],
                        "weaknesses": [],
                        "configurations": [],
                        "references": [],
                        "metrics": {},
                        "published": "2099-01-01T00:00:00.000",
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp
        mock_cpe.return_value = []
        mock_cwe.return_value = []
        mock_ref.return_value = []

        result = cluster_variants(_make_tool_use("CVE-2099-0001"))
        assert result["status"] == "success"
        json_data = result["content"][1]["json"]
        assert json_data["total_unique_variants"] == 0

    def test_whitespace_around_cve_id(self):
        with patch("manus_agent.tools.cluster_variants._nvd_get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _nvd_cve_response()
            mock_get.return_value = mock_resp

            with patch("manus_agent.tools.cluster_variants._search_by_cpe", return_value=[]):
                with patch("manus_agent.tools.cluster_variants._search_by_cwe", return_value=[]):
                    with patch("manus_agent.tools.cluster_variants._search_by_reference_domain", return_value=[]):
                        result = cluster_variants(_make_tool_use("  CVE-2021-44228  "))
                        assert result["status"] == "success"

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    @patch("manus_agent.tools.cluster_variants._search_by_reference_domain")
    @patch("manus_agent.tools.cluster_variants._search_by_cwe")
    @patch("manus_agent.tools.cluster_variants._search_by_cpe")
    def test_many_variants_across_dimensions(self, mock_cpe, mock_cwe, mock_ref, mock_get):
        """Test with many variants to verify counting."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _nvd_cve_response()
        mock_get.return_value = mock_resp

        cpe_results = [
            {"cve_id": f"CVE-2022-{i:04d}", "description": f"D{i}", "cvss_score": 5.0, "published": "2022-01-01"}
            for i in range(5)
        ]
        cwe_results = [
            {"cve_id": f"CVE-2022-{i:04d}", "description": f"D{i}", "cvss_score": 6.0, "published": "2022-02-01"}
            for i in range(5, 10)
        ]
        ref_results = [
            {"cve_id": f"CVE-2022-{i:04d}", "description": f"D{i}", "cvss_score": 7.0, "published": "2022-03-01"}
            for i in range(10, 15)
        ]

        mock_cpe.return_value = cpe_results
        mock_cwe.return_value = cwe_results
        mock_ref.return_value = ref_results

        result = cluster_variants(_make_tool_use("CVE-2021-44228"))
        assert result["status"] == "success"
        json_data = result["content"][1]["json"]
        assert json_data["total_unique_variants"] == 15

    @patch("manus_agent.tools.cluster_variants._nvd_get")
    @patch("manus_agent.tools.cluster_variants._search_by_reference_domain")
    @patch("manus_agent.tools.cluster_variants._search_by_cwe")
    @patch("manus_agent.tools.cluster_variants._search_by_cpe")
    def test_json_serializable_output(self, mock_cpe, mock_cwe, mock_ref, mock_get):
        """Ensure the JSON content is fully serializable."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _nvd_cve_response()
        mock_get.return_value = mock_resp
        mock_cpe.return_value = [
            {"cve_id": "CVE-2022-0001", "description": "Test", "cvss_score": 8.0, "published": "2022-01-01"}
        ]
        mock_cwe.return_value = []
        mock_ref.return_value = []

        result = cluster_variants(_make_tool_use("CVE-2021-44228"))
        json_data = result["content"][1]["json"]
        # Should not raise
        serialized = json.dumps(json_data)
        assert "CVE-2022-0001" in serialized

    @patch("manus_agent.tools.cluster_variants.os.environ.get")
    @patch("manus_agent.tools.cluster_variants.requests.get")
    def test_api_key_passed_in_header(self, mock_get, mock_env):
        """Verify NVD_API_KEY is sent as header."""
        mock_env.side_effect = lambda key, default="": "test-api-key" if key == "NVD_API_KEY" else default

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        _nvd_get("https://example.com")
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["apiKey"] == "test-api-key"
