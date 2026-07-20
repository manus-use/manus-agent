"""Comprehensive tests for cluster_variants tool and CLI subcommand."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from manus_agent.tools.cluster_variants import (
    _extract_cpe_info,
    _extract_cvss_score,
    _extract_cwes,
    _extract_description,
    _extract_source_domains,
    _extract_sources,
    _fetch_cves_by_cpe,
    _fetch_cves_by_cwe,
    _fetch_cves_by_keyword,
    _fetch_cves_by_source,
    _render_text,
    _summarize_cve,
    cluster_variants,
    cluster_variants_tool,
)

# ---------------------------------------------------------------------------
# Fixtures / sample data
# ---------------------------------------------------------------------------

SAMPLE_VULN = {
    "cve": {
        "id": "CVE-2021-44228",
        "descriptions": [{"lang": "en", "value": "Apache Log4j2 JNDI features used in configuration allow RCE."}],
        "configurations": [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "criteria": "cpe:2.3:a:apache:log4j:2.0:*:*:*:*:*:*:*",
                                "vulnerable": True,
                            },
                            {
                                "criteria": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
                                "vulnerable": True,
                            },
                        ]
                    }
                ]
            }
        ],
        "weaknesses": [
            {
                "description": [
                    {"lang": "en", "value": "CWE-917"},
                    {"lang": "en", "value": "CWE-502"},
                ]
            }
        ],
        "references": [
            {"url": "https://logging.apache.org/log4j/2.x/security.html", "source": "security@apache.org"},
            {"url": "https://github.com/advisories/GHSA-jfh8-c2jp-5v3q", "source": "security@apache.org"},
        ],
        "metrics": {
            "cvssMetricV31": [
                {"cvssData": {"baseScore": 10.0, "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}}
            ]
        },
        "published": "2021-12-10T10:15:00.000",
    }
}

SAMPLE_RELATED_VULN = {
    "cve": {
        "id": "CVE-2021-45046",
        "descriptions": [{"lang": "en", "value": "Apache Log4j2 Thread Context DoS."}],
        "configurations": [],
        "weaknesses": [{"description": [{"lang": "en", "value": "CWE-917"}]}],
        "references": [{"url": "https://logging.apache.org/", "source": "security@apache.org"}],
        "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 9.0}}]},
        "published": "2021-12-14T00:00:00.000",
    }
}

SAMPLE_RELATED_VULN_2 = {
    "cve": {
        "id": "CVE-2021-45105",
        "descriptions": [{"lang": "en", "value": "Apache Log4j2 does not protect from uncontrolled recursion."}],
        "configurations": [],
        "weaknesses": [{"description": [{"lang": "en", "value": "CWE-674"}]}],
        "references": [],
        "metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 7.5}}]},
        "published": "2021-12-18T00:00:00.000",
    }
}


# ---------------------------------------------------------------------------
# Unit tests: extraction helpers
# ---------------------------------------------------------------------------


class TestExtractCpeInfo:
    def test_extracts_vendor_product(self):
        pairs = _extract_cpe_info(SAMPLE_VULN)
        assert len(pairs) == 1  # deduplicated (same vendor/product)
        assert pairs[0] == {"vendor": "apache", "product": "log4j"}

    def test_skips_wildcard_entries(self):
        vuln = {
            "cve": {"configurations": [{"nodes": [{"cpeMatch": [{"criteria": "cpe:2.3:a:*:*:1.0:*:*:*:*:*:*:*"}]}]}]}
        }
        assert _extract_cpe_info(vuln) == []

    def test_empty_configurations(self):
        vuln = {"cve": {"configurations": []}}
        assert _extract_cpe_info(vuln) == []

    def test_no_configurations_key(self):
        vuln = {"cve": {}}
        assert _extract_cpe_info(vuln) == []

    def test_multiple_products(self):
        vuln = {
            "cve": {
                "configurations": [
                    {
                        "nodes": [
                            {
                                "cpeMatch": [
                                    {"criteria": "cpe:2.3:a:vendor1:product1:1.0:*:*:*:*:*:*:*"},
                                    {"criteria": "cpe:2.3:a:vendor2:product2:2.0:*:*:*:*:*:*:*"},
                                ]
                            }
                        ]
                    }
                ]
            }
        }
        pairs = _extract_cpe_info(vuln)
        assert len(pairs) == 2
        assert {"vendor": "vendor1", "product": "product1"} in pairs
        assert {"vendor": "vendor2", "product": "product2"} in pairs


class TestExtractCwes:
    def test_extracts_cwes(self):
        cwes = _extract_cwes(SAMPLE_VULN)
        assert "CWE-917" in cwes
        assert "CWE-502" in cwes

    def test_skips_noinfo(self):
        vuln = {"cve": {"weaknesses": [{"description": [{"lang": "en", "value": "CWE-noinfo"}]}]}}
        assert _extract_cwes(vuln) == []

    def test_empty_weaknesses(self):
        vuln = {"cve": {"weaknesses": []}}
        assert _extract_cwes(vuln) == []

    def test_no_weaknesses_key(self):
        vuln = {"cve": {}}
        assert _extract_cwes(vuln) == []


class TestExtractSources:
    def test_extracts_source_orgs(self):
        sources = _extract_sources(SAMPLE_VULN)
        assert "security@apache.org" in sources

    def test_deduplicates(self):
        sources = _extract_sources(SAMPLE_VULN)
        # Same source appears twice in refs but should be deduplicated
        assert sources.count("security@apache.org") == 1

    def test_empty_references(self):
        vuln = {"cve": {"references": []}}
        assert _extract_sources(vuln) == []


class TestExtractSourceDomains:
    def test_extracts_domains(self):
        domains = _extract_source_domains(SAMPLE_VULN)
        assert "logging.apache.org" in domains
        assert "github.com" in domains

    def test_skips_nvd_domains(self):
        vuln = {
            "cve": {
                "references": [
                    {"url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228"},
                    {"url": "https://cve.mitre.org/something"},
                ]
            }
        }
        domains = _extract_source_domains(vuln)
        assert "nvd.nist.gov" not in domains
        assert "cve.mitre.org" not in domains


class TestExtractCvssScore:
    def test_cvss31(self):
        assert _extract_cvss_score(SAMPLE_VULN) == 10.0

    def test_cvss30_fallback(self):
        vuln = {"cve": {"metrics": {"cvssMetricV30": [{"cvssData": {"baseScore": 8.5}}]}}}
        assert _extract_cvss_score(vuln) == 8.5

    def test_cvss2_fallback(self):
        vuln = {"cve": {"metrics": {"cvssMetricV2": [{"cvssData": {"baseScore": 7.0}}]}}}
        assert _extract_cvss_score(vuln) == 7.0

    def test_no_metrics(self):
        vuln = {"cve": {"metrics": {}}}
        assert _extract_cvss_score(vuln) is None

    def test_no_metrics_key(self):
        vuln = {"cve": {}}
        assert _extract_cvss_score(vuln) is None


class TestExtractDescription:
    def test_english_description(self):
        desc = _extract_description(SAMPLE_VULN)
        assert "Log4j2" in desc

    def test_fallback_to_first(self):
        vuln = {"cve": {"descriptions": [{"lang": "fr", "value": "Description en français"}]}}
        assert _extract_description(vuln) == "Description en français"

    def test_empty_descriptions(self):
        vuln = {"cve": {"descriptions": []}}
        assert _extract_description(vuln) == ""


class TestSummarizeCve:
    def test_summarizes_fields(self):
        summary = _summarize_cve(SAMPLE_VULN)
        assert summary["cve_id"] == "CVE-2021-44228"
        assert summary["cvss_score"] == 10.0
        assert "CWE-917" in summary["cwes"]
        assert summary["published"] == "2021-12-10"

    def test_truncates_long_description(self):
        vuln = {
            "cve": {
                "id": "CVE-TEST-0001",
                "descriptions": [{"lang": "en", "value": "A" * 300}],
                "weaknesses": [],
                "metrics": {},
                "published": "2024-01-01",
            }
        }
        summary = _summarize_cve(vuln)
        assert len(summary["description"]) <= 203  # 200 + "..."


# ---------------------------------------------------------------------------
# Unit tests: fetch helpers (mocked HTTP)
# ---------------------------------------------------------------------------


class TestFetchCvesByCpe:
    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_returns_related_cves(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [SAMPLE_RELATED_VULN]}
        mock_get.return_value = mock_resp

        results = _fetch_cves_by_cpe("apache", "log4j", "CVE-2021-44228")
        assert len(results) == 1
        assert results[0]["cve"]["id"] == "CVE-2021-45046"

    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_excludes_input_cve(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [SAMPLE_VULN]}
        mock_get.return_value = mock_resp

        results = _fetch_cves_by_cpe("apache", "log4j", "CVE-2021-44228")
        assert len(results) == 0

    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_handles_request_exception(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("timeout")
        results = _fetch_cves_by_cpe("apache", "log4j", "CVE-2021-44228")
        assert results == []


class TestFetchCvesByKeyword:
    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_returns_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [SAMPLE_RELATED_VULN]}
        mock_get.return_value = mock_resp

        results = _fetch_cves_by_keyword("log4j", "CVE-2021-44228")
        assert len(results) == 1

    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_handles_json_error(self, mock_get):
        mock_get.side_effect = ValueError("bad json")
        results = _fetch_cves_by_keyword("log4j", "CVE-2021-44228")
        assert results == []


class TestFetchCvesByCwe:
    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_returns_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [SAMPLE_RELATED_VULN, SAMPLE_RELATED_VULN_2]}
        mock_get.return_value = mock_resp

        results = _fetch_cves_by_cwe("CWE-917", "CVE-2021-44228")
        assert len(results) == 2

    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_excludes_input(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [SAMPLE_VULN]}
        mock_get.return_value = mock_resp

        results = _fetch_cves_by_cwe("CWE-917", "CVE-2021-44228")
        assert len(results) == 0


class TestFetchCvesBySource:
    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_returns_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [SAMPLE_RELATED_VULN]}
        mock_get.return_value = mock_resp

        results = _fetch_cves_by_source("security@apache.org", "CVE-2021-44228")
        assert len(results) == 1

    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_handles_exception(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("refused")
        results = _fetch_cves_by_source("security@apache.org", "CVE-2021-44228")
        assert results == []


# ---------------------------------------------------------------------------
# Integration tests: cluster_variants main function
# ---------------------------------------------------------------------------


class TestClusterVariants:
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_source")
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_cwe")
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_cpe")
    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_full_clustering(self, mock_nvd, mock_cpe, mock_cwe, mock_src):
        # Seed CVE fetch
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [SAMPLE_VULN]}
        mock_nvd.return_value = mock_resp

        # Cluster fetches
        mock_cpe.return_value = [SAMPLE_RELATED_VULN]
        mock_cwe.return_value = [SAMPLE_RELATED_VULN_2]
        mock_src.return_value = [SAMPLE_RELATED_VULN]

        result = cluster_variants("CVE-2021-44228")

        assert "error" not in result
        assert result["input_cve"]["cve_id"] == "CVE-2021-44228"
        assert result["input_cve"]["cvss_score"] == 10.0
        assert "CWE-917" in result["input_cve"]["cwes"]
        assert "apache/log4j" in result["input_cve"]["cpe_vendors"]

        # Clusters populated
        assert len(result["clusters"]["component"]["cves"]) == 1
        assert len(result["clusters"]["cwe"]["cves"]) == 1
        assert len(result["clusters"]["source"]["cves"]) == 1

        # Summary
        assert result["summary"]["total_unique"] >= 1

    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_cve_not_found(self, mock_nvd):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_nvd.return_value = mock_resp

        result = cluster_variants("CVE-9999-99999")
        assert "error" in result
        assert "No vulnerability data" in result["error"]

    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_network_error(self, mock_nvd):
        import requests

        mock_nvd.side_effect = requests.exceptions.Timeout("timeout")

        result = cluster_variants("CVE-2021-44228")
        assert "error" in result
        assert "Failed to fetch" in result["error"]

    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_source")
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_cwe")
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_cpe")
    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_handles_cve_with_no_cpe(self, mock_nvd, mock_cpe, mock_cwe, mock_src):
        vuln_no_cpe = {
            "cve": {
                "id": "CVE-2024-0001",
                "descriptions": [{"lang": "en", "value": "Test vuln no CPE."}],
                "configurations": [],
                "weaknesses": [{"description": [{"lang": "en", "value": "CWE-79"}]}],
                "references": [{"url": "https://example.com", "source": "test@example.com"}],
                "metrics": {},
                "published": "2024-01-01",
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [vuln_no_cpe]}
        mock_nvd.return_value = mock_resp

        mock_cpe.return_value = []
        mock_cwe.return_value = []
        mock_src.return_value = []

        result = cluster_variants("CVE-2024-0001")
        assert "error" not in result
        assert result["clusters"]["component"]["cves"] == []

    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_source")
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_cwe")
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_cpe")
    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_lowercase_cve_id_normalized(self, mock_nvd, mock_cpe, mock_cwe, mock_src):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [SAMPLE_VULN]}
        mock_nvd.return_value = mock_resp
        mock_cpe.return_value = []
        mock_cwe.return_value = []
        mock_src.return_value = []

        result = cluster_variants("cve-2021-44228")
        assert result["input_cve"]["cve_id"] == "CVE-2021-44228"


# ---------------------------------------------------------------------------
# Tests: text rendering
# ---------------------------------------------------------------------------


class TestRenderText:
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_source")
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_cwe")
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_cpe")
    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_renders_full_output(self, mock_nvd, mock_cpe, mock_cwe, mock_src):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [SAMPLE_VULN]}
        mock_nvd.return_value = mock_resp
        mock_cpe.return_value = [SAMPLE_RELATED_VULN]
        mock_cwe.return_value = [SAMPLE_RELATED_VULN_2]
        mock_src.return_value = []

        result = cluster_variants("CVE-2021-44228")
        text = _render_text(result)

        assert "CVE-2021-44228" in text
        assert "Cluster 1" in text
        assert "Cluster 2" in text
        assert "Cluster 3" in text
        assert "CVE-2021-45046" in text
        assert "CVE-2021-45105" in text

    def test_renders_error(self):
        text = _render_text({"error": "Something went wrong"})
        assert "❌" in text
        assert "Something went wrong" in text


# ---------------------------------------------------------------------------
# Tests: Strands tool entry point
# ---------------------------------------------------------------------------


class TestClusterVariantsTool:
    @patch("manus_agent.tools.cluster_variants.cluster_variants")
    def test_success(self, mock_cluster):
        mock_cluster.return_value = {
            "input_cve": {"cve_id": "CVE-2021-44228"},
            "clusters": {"component": {"cves": []}, "cwe": {"cves": []}, "source": {"cves": []}},
            "summary": {"total_unique": 0},
        }

        tool_use = {"toolUseId": "test-123", "input": {"cve_id": "CVE-2021-44228"}}
        result = cluster_variants_tool(tool_use)

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"

    @patch("manus_agent.tools.cluster_variants.cluster_variants")
    def test_error_result(self, mock_cluster):
        mock_cluster.return_value = {"error": "CVE not found"}

        tool_use = {"toolUseId": "test-456", "input": {"cve_id": "CVE-9999-99999"}}
        result = cluster_variants_tool(tool_use)

        assert result["status"] == "error"
        assert "CVE not found" in result["content"][0]["text"]

    def test_invalid_cve_format(self):
        tool_use = {"toolUseId": "test-789", "input": {"cve_id": "not-a-cve"}}
        result = cluster_variants_tool(tool_use)

        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_empty_cve_id(self):
        tool_use = {"toolUseId": "test-000", "input": {"cve_id": ""}}
        result = cluster_variants_tool(tool_use)

        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Tests: CLI subcommand
# ---------------------------------------------------------------------------


class TestCliClusterVariants:
    @patch("manus_agent.tools.cluster_variants.cluster_variants")
    def test_text_output(self, mock_cluster, capsys):
        mock_cluster.return_value = {
            "input_cve": {
                "cve_id": "CVE-2021-44228",
                "description": "Log4Shell",
                "cvss_score": 10.0,
                "cwes": ["CWE-917"],
                "cpe_vendors": ["apache/log4j"],
                "sources": ["security@apache.org"],
            },
            "clusters": {
                "component": {"dimension": "Same Component/Vendor", "criteria": ["apache/log4j"], "cves": []},
                "cwe": {"dimension": "Same CWE Weakness Class", "criteria": ["CWE-917"], "cves": []},
                "source": {
                    "dimension": "Same Researcher/Disclosure Source",
                    "criteria": ["security@apache.org"],
                    "cves": [],
                },
            },
            "summary": {"component_count": 0, "cwe_count": 0, "source_count": 0, "total_unique": 0},
        }

        from manus_agent.cli import _run_cluster_variants

        exit_code = _run_cluster_variants(["CVE-2021-44228"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "CVE-2021-44228" in captured.out

    @patch("manus_agent.tools.cluster_variants.cluster_variants")
    def test_json_output(self, mock_cluster, capsys):
        mock_cluster.return_value = {
            "input_cve": {"cve_id": "CVE-2021-44228"},
            "clusters": {
                "component": {"dimension": "Same Component/Vendor", "criteria": [], "cves": []},
                "cwe": {"dimension": "Same CWE Weakness Class", "criteria": [], "cves": []},
                "source": {"dimension": "Same Researcher/Disclosure Source", "criteria": [], "cves": []},
            },
            "summary": {"component_count": 0, "cwe_count": 0, "source_count": 0, "total_unique": 0},
        }

        from manus_agent.cli import _run_cluster_variants

        exit_code = _run_cluster_variants(["CVE-2021-44228", "--output", "json"])
        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["input_cve"]["cve_id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.cluster_variants.cluster_variants")
    def test_error_exits_nonzero(self, mock_cluster, capsys):
        mock_cluster.return_value = {"error": "NVD unavailable"}

        from manus_agent.cli import _run_cluster_variants

        exit_code = _run_cluster_variants(["CVE-2021-44228"])
        assert exit_code == 1

    def test_invalid_cve_exits_nonzero(self, capsys):
        from manus_agent.cli import _run_cluster_variants

        exit_code = _run_cluster_variants(["not-a-cve"])
        assert exit_code == 1

    def test_help_flag(self):
        from manus_agent.cli import _build_cluster_variants_parser

        parser = _build_cluster_variants_parser()
        assert parser.prog == "manus-agent cluster-variants"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_source")
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_cwe")
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_cpe")
    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_deduplicates_across_cpe_queries(self, mock_nvd, mock_cpe, mock_cwe, mock_src):
        """When same CVE appears for multiple CPE vendor/product pairs, it's deduplicated."""
        vuln_multi_cpe = {
            "cve": {
                "id": "CVE-2024-MULTI",
                "descriptions": [{"lang": "en", "value": "Multi CPE vuln"}],
                "configurations": [
                    {
                        "nodes": [
                            {
                                "cpeMatch": [
                                    {"criteria": "cpe:2.3:a:vendorA:productA:1.0:*:*:*:*:*:*:*"},
                                    {"criteria": "cpe:2.3:a:vendorB:productB:2.0:*:*:*:*:*:*:*"},
                                ]
                            }
                        ]
                    }
                ],
                "weaknesses": [],
                "references": [],
                "metrics": {},
                "published": "2024-01-01",
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [vuln_multi_cpe]}
        mock_nvd.return_value = mock_resp

        # Same CVE returned for both CPE queries
        mock_cpe.return_value = [SAMPLE_RELATED_VULN]
        mock_cwe.return_value = []
        mock_src.return_value = []

        result = cluster_variants("CVE-2024-MULTI")
        # CPE fetch called twice (2 pairs), but the same result deduplicated
        assert mock_cpe.call_count == 2
        # Only one unique entry in component cluster
        assert len(result["clusters"]["component"]["cves"]) == 1

    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_source")
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_cwe")
    @patch("manus_agent.tools.cluster_variants._fetch_cves_by_cpe")
    @patch("manus_agent.tools.cluster_variants._nvd_get_with_retry")
    def test_whitespace_cve_id_handled(self, mock_nvd, mock_cpe, mock_cwe, mock_src):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [SAMPLE_VULN]}
        mock_nvd.return_value = mock_resp
        mock_cpe.return_value = []
        mock_cwe.return_value = []
        mock_src.return_value = []

        result = cluster_variants("  CVE-2021-44228  ")
        assert result["input_cve"]["cve_id"] == "CVE-2021-44228"
