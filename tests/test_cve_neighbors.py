"""Comprehensive test suite for get_cve_neighbors tool and CLI subcommand.

Tests cover:
- Input validation (invalid CVE formats)
- NVD fetch failures (network errors, empty responses)
- CPE product extraction from NVD configurations
- Neighbor search with NVD keyword search
- EPSS enrichment (success and graceful degradation)
- Result sorting (EPSS descending, then CVSS)
- max_results limiting
- CLI argument parsing and dispatch
- JSON and text output formats
- Edge cases (no CPE data, no neighbors found, target CVE excluded)

All HTTP calls are fully mocked — no real network access.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

from manus_agent.tools.get_cve_neighbors import (
    TOOL_SPEC,
    _build_nvd_headers,
    _extract_cpe_products,
    _extract_cvss,
    _extract_description,
    _fetch_epss_scores,
    get_cve_neighbors,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mock_response(payload, status_code: int = 200, raise_for_status_exc=None):
    """Return a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    if raise_for_status_exc is not None:
        resp.raise_for_status.side_effect = raise_for_status_exc
    else:
        resp.raise_for_status.return_value = None
    return resp


def _tool_use(cve_id: str = "CVE-2021-44228", max_results: int = 10) -> dict:
    return {
        "toolUseId": "test-123",
        "input": {"cve_id": cve_id, "max_results": max_results},
    }


def _nvd_cve_record(
    cve_id: str = "CVE-2021-44228",
    vendor: str = "apache",
    product: str = "log4j",
    cvss_score: float = 10.0,
    severity: str = "CRITICAL",
    description: str = "Remote code execution via JNDI lookups in Apache Log4j2.",
    published: str = "2021-12-10T10:15:00.000",
):
    """Build a minimal NVD CVE record with CPE data."""
    return {
        "cve": {
            "id": cve_id,
            "published": published,
            "descriptions": [{"lang": "en", "value": description}],
            "metrics": {
                "cvssMetricV31": [
                    {
                        "cvssData": {
                            "baseScore": cvss_score,
                            "baseSeverity": severity,
                            "version": "3.1",
                            "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                        }
                    }
                ]
            },
            "configurations": [
                {
                    "nodes": [
                        {
                            "cpeMatch": [
                                {
                                    "vulnerable": True,
                                    "criteria": f"cpe:2.3:a:{vendor}:{product}:*:*:*:*:*:*:*:*",
                                }
                            ]
                        }
                    ]
                }
            ],
        }
    }


# ---------------------------------------------------------------------------
# TOOL_SPEC tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_tool_spec_name(self):
        assert TOOL_SPEC["name"] == "get_cve_neighbors"

    def test_tool_spec_has_required_fields(self):
        assert "description" in TOOL_SPEC
        assert "inputSchema" in TOOL_SPEC

    def test_cve_id_is_required(self):
        required = TOOL_SPEC["inputSchema"]["json"]["required"]
        assert "cve_id" in required


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_cve_id(self):
        result = get_cve_neighbors(_tool_use(cve_id=""))
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_invalid_format_no_prefix(self):
        result = get_cve_neighbors(_tool_use(cve_id="2021-44228"))
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_invalid_format_random_string(self):
        result = get_cve_neighbors(_tool_use(cve_id="not-a-cve"))
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_whitespace_only(self):
        result = get_cve_neighbors(_tool_use(cve_id="   "))
        assert result["status"] == "error"

    def test_lowercase_accepted(self):
        """Lowercase CVE IDs should be normalised to uppercase."""
        with patch("manus_agent.tools.get_cve_neighbors._nvd_get") as mock_get:
            mock_get.return_value = _mock_response({"vulnerabilities": []})
            result = get_cve_neighbors(_tool_use(cve_id="cve-2021-44228"))
            # Should at least attempt the NVD fetch (valid format after upper())
            assert result["status"] == "error"
            assert "No NVD record" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# NVD fetch failure tests
# ---------------------------------------------------------------------------


class TestNvdFetchFailures:
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_network_error_on_target_fetch(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        result = get_cve_neighbors(_tool_use())
        assert result["status"] == "error"
        assert "Failed to fetch NVD data" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_timeout_on_target_fetch(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Read timed out")
        result = get_cve_neighbors(_tool_use())
        assert result["status"] == "error"
        assert "Failed to fetch NVD data" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_empty_vulnerabilities_list(self, mock_get):
        mock_get.return_value = _mock_response({"vulnerabilities": []})
        result = get_cve_neighbors(_tool_use())
        assert result["status"] == "error"
        assert "No NVD record found" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_http_error_on_target_fetch(self, mock_get):
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")
        result = get_cve_neighbors(_tool_use())
        assert result["status"] == "error"
        assert "Failed to fetch NVD data" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# CPE product extraction tests
# ---------------------------------------------------------------------------


class TestExtractCpeProducts:
    def test_single_product(self):
        record = {
            "configurations": [
                {
                    "nodes": [
                        {
                            "cpeMatch": [
                                {
                                    "vulnerable": True,
                                    "criteria": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        products = _extract_cpe_products(record)
        assert len(products) == 1
        assert products[0] == {"vendor": "apache", "product": "log4j"}

    def test_multiple_products(self):
        record = {
            "configurations": [
                {
                    "nodes": [
                        {
                            "cpeMatch": [
                                {
                                    "vulnerable": True,
                                    "criteria": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
                                },
                                {
                                    "vulnerable": True,
                                    "criteria": "cpe:2.3:a:apache:log4j_api:2.14.1:*:*:*:*:*:*:*",
                                },
                            ]
                        }
                    ]
                }
            ]
        }
        products = _extract_cpe_products(record)
        assert len(products) == 2

    def test_deduplication(self):
        record = {
            "configurations": [
                {
                    "nodes": [
                        {
                            "cpeMatch": [
                                {
                                    "vulnerable": True,
                                    "criteria": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
                                },
                                {
                                    "vulnerable": True,
                                    "criteria": "cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*",
                                },
                            ]
                        }
                    ]
                }
            ]
        }
        products = _extract_cpe_products(record)
        assert len(products) == 1  # Same vendor:product, different version

    def test_empty_configurations(self):
        record = {"configurations": []}
        products = _extract_cpe_products(record)
        assert products == []

    def test_no_configurations_key(self):
        record = {}
        products = _extract_cpe_products(record)
        assert products == []

    def test_short_cpe_string(self):
        """CPE strings with fewer than 5 parts should be skipped."""
        record = {
            "configurations": [
                {"nodes": [{"cpeMatch": [{"vulnerable": True, "criteria": "cpe:2.3:a"}]}]}
            ]
        }
        products = _extract_cpe_products(record)
        assert products == []


# ---------------------------------------------------------------------------
# CVSS extraction tests
# ---------------------------------------------------------------------------


class TestExtractCvss:
    def test_cvss_v31(self):
        cve_item = {
            "metrics": {
                "cvssMetricV31": [
                    {"cvssData": {"baseScore": 9.8, "baseSeverity": "CRITICAL", "version": "3.1"}}
                ]
            }
        }
        result = _extract_cvss(cve_item)
        assert result["score"] == 9.8
        assert result["severity"] == "CRITICAL"

    def test_cvss_v30_fallback(self):
        cve_item = {
            "metrics": {
                "cvssMetricV30": [
                    {"cvssData": {"baseScore": 7.5, "baseSeverity": "HIGH", "version": "3.0"}}
                ]
            }
        }
        result = _extract_cvss(cve_item)
        assert result["score"] == 7.5
        assert result["severity"] == "HIGH"

    def test_cvss_v2_fallback(self):
        cve_item = {
            "metrics": {
                "cvssMetricV2": [{"cvssData": {"baseScore": 5.0, "baseSeverity": "MEDIUM"}}]
            }
        }
        result = _extract_cvss(cve_item)
        assert result["score"] == 5.0
        assert result["version"] == "2.0"

    def test_no_metrics(self):
        cve_item = {"metrics": {}}
        result = _extract_cvss(cve_item)
        assert result["score"] is None
        assert result["severity"] == "UNKNOWN"

    def test_empty_cve_item(self):
        result = _extract_cvss({})
        assert result["score"] is None


# ---------------------------------------------------------------------------
# Description extraction tests
# ---------------------------------------------------------------------------


class TestExtractDescription:
    def test_english_description(self):
        cve_item = {
            "descriptions": [{"lang": "en", "value": "A critical vulnerability."}]
        }
        assert _extract_description(cve_item) == "A critical vulnerability."

    def test_truncation(self):
        long_desc = "A" * 300
        cve_item = {"descriptions": [{"lang": "en", "value": long_desc}]}
        result = _extract_description(cve_item)
        assert len(result) == 200
        assert result.endswith("...")

    def test_no_english(self):
        cve_item = {"descriptions": [{"lang": "es", "value": "Una vulnerabilidad."}]}
        assert _extract_description(cve_item) == "No description available."

    def test_empty_descriptions(self):
        cve_item = {"descriptions": []}
        assert _extract_description(cve_item) == "No description available."


# ---------------------------------------------------------------------------
# EPSS fetch tests
# ---------------------------------------------------------------------------


class TestFetchEpssScores:
    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    def test_successful_batch_fetch(self, mock_get):
        mock_get.return_value = _mock_response(
            {
                "data": [
                    {"cve": "CVE-2021-44228", "epss": "0.97565", "percentile": "0.99987"},
                    {"cve": "CVE-2021-45046", "epss": "0.12345", "percentile": "0.85000"},
                ]
            }
        )
        result = _fetch_epss_scores(["CVE-2021-44228", "CVE-2021-45046"])
        assert result["CVE-2021-44228"]["epss"] == pytest.approx(0.97565)
        assert result["CVE-2021-45046"]["percentile"] == pytest.approx(0.85)

    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    def test_network_error_returns_empty(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("fail")
        result = _fetch_epss_scores(["CVE-2021-44228"])
        assert result == {}

    def test_empty_list_returns_empty(self):
        result = _fetch_epss_scores([])
        assert result == {}

    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    def test_timeout_returns_empty(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("timeout")
        result = _fetch_epss_scores(["CVE-2021-44228"])
        assert result == {}


# ---------------------------------------------------------------------------
# Main tool function — success paths
# ---------------------------------------------------------------------------


class TestGetCveNeighborsSuccess:
    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_finds_neighbors(self, mock_nvd_get, mock_requests_get):
        """Happy path: finds neighbors and enriches with EPSS."""
        # First call: target CVE
        target_record = _nvd_cve_record("CVE-2021-44228")
        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [target_record]}),
            _mock_response(
                {
                    "vulnerabilities": [
                        target_record,  # should be excluded from results
                        _nvd_cve_record(
                            "CVE-2021-45046",
                            cvss_score=9.0,
                            severity="CRITICAL",
                            description="DoS and RCE in Log4j2.",
                        ),
                        _nvd_cve_record(
                            "CVE-2021-45105",
                            cvss_score=7.5,
                            severity="HIGH",
                            description="DoS in Log4j2.",
                        ),
                    ]
                }
            ),
        ]

        # EPSS response
        mock_requests_get.return_value = _mock_response(
            {
                "data": [
                    {"cve": "CVE-2021-45046", "epss": "0.5", "percentile": "0.9"},
                    {"cve": "CVE-2021-45105", "epss": "0.3", "percentile": "0.7"},
                ]
            }
        )

        result = get_cve_neighbors(_tool_use())
        assert result["status"] == "success"

        # Check JSON payload
        json_block = next(c for c in result["content"] if "json" in c)["json"]
        assert json_block["cve_id"] == "CVE-2021-44228"
        assert json_block["product"] == "apache:log4j"
        assert json_block["neighbor_count"] == 2
        # Sorted by EPSS descending
        assert json_block["neighbors"][0]["cve_id"] == "CVE-2021-45046"
        assert json_block["neighbors"][1]["cve_id"] == "CVE-2021-45105"

    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_excludes_target_cve(self, mock_nvd_get, mock_requests_get):
        """The target CVE should never appear in the neighbor list."""
        target = _nvd_cve_record("CVE-2021-44228")
        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [target]}),
            _mock_response({"vulnerabilities": [target]}),  # only itself returned
        ]
        mock_requests_get.return_value = _mock_response({"data": []})

        result = get_cve_neighbors(_tool_use())
        assert result["status"] == "success"
        # Text output should say no neighbors found
        text = result["content"][0]["text"]
        assert "No neighboring CVEs found" in text

    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_max_results_limits_output(self, mock_nvd_get, mock_requests_get):
        """max_results should cap the number of returned neighbors."""
        target = _nvd_cve_record("CVE-2021-44228")
        neighbors = [
            _nvd_cve_record(f"CVE-2021-{45000 + i}", cvss_score=5.0 + i * 0.1)
            for i in range(20)
        ]
        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [target]}),
            _mock_response({"vulnerabilities": [target] + neighbors}),
        ]
        mock_requests_get.return_value = _mock_response({"data": []})

        result = get_cve_neighbors(_tool_use(max_results=5))
        assert result["status"] == "success"
        json_block = next(c for c in result["content"] if "json" in c)["json"]
        assert json_block["neighbor_count"] == 5
        assert len(json_block["neighbors"]) == 5

    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_epss_failure_graceful(self, mock_nvd_get, mock_requests_get):
        """If EPSS fetch fails, neighbors should still be returned with 0.0 scores."""
        target = _nvd_cve_record("CVE-2021-44228")
        neighbor = _nvd_cve_record("CVE-2021-45046", cvss_score=9.0)
        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [target]}),
            _mock_response({"vulnerabilities": [target, neighbor]}),
        ]
        mock_requests_get.side_effect = requests.exceptions.Timeout("timeout")

        result = get_cve_neighbors(_tool_use())
        assert result["status"] == "success"
        json_block = next(c for c in result["content"] if "json" in c)["json"]
        assert json_block["neighbors"][0]["epss"] == 0.0

    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_no_cpe_data(self, mock_nvd_get):
        """CVE with no CPE configurations should return a descriptive message."""
        record = {
            "cve": {
                "id": "CVE-2024-99999",
                "published": "2024-01-01T00:00:00.000",
                "descriptions": [{"lang": "en", "value": "No CPE."}],
                "metrics": {},
                "configurations": [],
            }
        }
        mock_nvd_get.return_value = _mock_response({"vulnerabilities": [record]})

        result = get_cve_neighbors(_tool_use(cve_id="CVE-2024-99999"))
        assert result["status"] == "success"
        assert "No CPE product data" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_neighbor_search_failure(self, mock_nvd_get, mock_requests_get):
        """If the neighbor search fails, should return an error."""
        target = _nvd_cve_record("CVE-2021-44228")
        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [target]}),
            requests.exceptions.ConnectionError("fail"),
        ]

        result = get_cve_neighbors(_tool_use())
        assert result["status"] == "error"
        assert "Failed to search NVD for neighbors" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_sorting_by_epss_then_cvss(self, mock_nvd_get, mock_requests_get):
        """Neighbors should be sorted by EPSS desc, then CVSS desc."""
        target = _nvd_cve_record("CVE-2021-44228")
        n1 = _nvd_cve_record("CVE-2021-00001", cvss_score=9.0, severity="CRITICAL")
        n2 = _nvd_cve_record("CVE-2021-00002", cvss_score=5.0, severity="MEDIUM")
        n3 = _nvd_cve_record("CVE-2021-00003", cvss_score=10.0, severity="CRITICAL")

        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [target]}),
            _mock_response({"vulnerabilities": [target, n1, n2, n3]}),
        ]
        mock_requests_get.return_value = _mock_response(
            {
                "data": [
                    {"cve": "CVE-2021-00001", "epss": "0.1", "percentile": "0.5"},
                    {"cve": "CVE-2021-00002", "epss": "0.9", "percentile": "0.99"},
                    {"cve": "CVE-2021-00003", "epss": "0.1", "percentile": "0.5"},
                ]
            }
        )

        result = get_cve_neighbors(_tool_use())
        json_block = next(c for c in result["content"] if "json" in c)["json"]
        # CVE-2021-00002 has highest EPSS (0.9)
        assert json_block["neighbors"][0]["cve_id"] == "CVE-2021-00002"
        # CVE-2021-00003 and CVE-2021-00001 have same EPSS but 00003 has higher CVSS
        assert json_block["neighbors"][1]["cve_id"] == "CVE-2021-00003"
        assert json_block["neighbors"][2]["cve_id"] == "CVE-2021-00001"


# ---------------------------------------------------------------------------
# NVD headers tests
# ---------------------------------------------------------------------------


class TestBuildNvdHeaders:
    def test_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            headers = _build_nvd_headers()
            assert "apiKey" not in headers

    def test_with_api_key(self):
        with patch.dict("os.environ", {"NVD_API_KEY": "test-key-123"}):
            headers = _build_nvd_headers()
            assert headers["apiKey"] == "test-key-123"

    def test_whitespace_api_key_ignored(self):
        with patch.dict("os.environ", {"NVD_API_KEY": "   "}):
            headers = _build_nvd_headers()
            assert "apiKey" not in headers


# ---------------------------------------------------------------------------
# CLI subcommand tests
# ---------------------------------------------------------------------------


class TestCliCveNeighbors:
    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_text_output(self, mock_nvd_get, mock_requests_get, capsys):
        from manus_agent.cli import _run_cve_neighbors

        target = _nvd_cve_record("CVE-2021-44228")
        neighbor = _nvd_cve_record("CVE-2021-45046", cvss_score=9.0, severity="CRITICAL")
        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [target]}),
            _mock_response({"vulnerabilities": [target, neighbor]}),
        ]
        mock_requests_get.return_value = _mock_response(
            {"data": [{"cve": "CVE-2021-45046", "epss": "0.5", "percentile": "0.9"}]}
        )

        exit_code = _run_cve_neighbors(["CVE-2021-44228"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "CVE-2021-45046" in captured.out

    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_json_output(self, mock_nvd_get, mock_requests_get, capsys):
        from manus_agent.cli import _run_cve_neighbors

        target = _nvd_cve_record("CVE-2021-44228")
        neighbor = _nvd_cve_record("CVE-2021-45046", cvss_score=9.0, severity="CRITICAL")
        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [target]}),
            _mock_response({"vulnerabilities": [target, neighbor]}),
        ]
        mock_requests_get.return_value = _mock_response(
            {"data": [{"cve": "CVE-2021-45046", "epss": "0.5", "percentile": "0.9"}]}
        )

        with patch("manus_agent.tools.get_cve_neighbors.log_tool_output_size"):
            exit_code = _run_cve_neighbors(["CVE-2021-44228", "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["cve_id"] == "CVE-2021-44228"
        assert "neighbors" in data

    def test_invalid_cve_id_exits_with_error(self, capsys):
        from manus_agent.cli import _run_cve_neighbors

        exit_code = _run_cve_neighbors(["not-a-cve"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Invalid CVE ID" in captured.err

    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_max_results_flag(self, mock_nvd_get, capsys):
        from manus_agent.cli import _run_cve_neighbors

        target = _nvd_cve_record("CVE-2021-44228")
        mock_nvd_get.return_value = _mock_response({"vulnerabilities": [target]})

        # With --max-results but no CPE → success with message
        with patch(
            "manus_agent.tools.get_cve_neighbors._extract_cpe_products",
            return_value=[],
        ):
            exit_code = _run_cve_neighbors(["CVE-2021-44228", "--max-results", "5"])
            assert exit_code == 0

    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_network_error_exits_with_error(self, mock_nvd_get, capsys):
        from manus_agent.cli import _run_cve_neighbors

        mock_nvd_get.side_effect = requests.exceptions.ConnectionError("fail")
        exit_code = _run_cve_neighbors(["CVE-2021-44228"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_multiple_products_uses_first(self, mock_nvd_get, mock_requests_get):
        """When a CVE has multiple CPE products, the first one is used for search."""
        record = {
            "cve": {
                "id": "CVE-2021-44228",
                "published": "2021-12-10T10:15:00.000",
                "descriptions": [{"lang": "en", "value": "RCE"}],
                "metrics": {},
                "configurations": [
                    {
                        "nodes": [
                            {
                                "cpeMatch": [
                                    {
                                        "vulnerable": True,
                                        "criteria": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
                                    },
                                    {
                                        "vulnerable": True,
                                        "criteria": "cpe:2.3:a:siemens:some_product:1.0:*:*:*:*:*:*:*",
                                    },
                                ]
                            }
                        ]
                    }
                ],
            }
        }

        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [record]}),
            _mock_response({"vulnerabilities": []}),
        ]
        mock_requests_get.return_value = _mock_response({"data": []})

        result = get_cve_neighbors(_tool_use())
        # Verify the search URL used the first product
        call_url = mock_nvd_get.call_args_list[1][0][0]
        assert "apache" in call_url
        assert "log4j" in call_url

    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_max_results_capped_at_50(self, mock_nvd_get, mock_requests_get):
        """max_results above 50 should be capped."""
        target = _nvd_cve_record("CVE-2021-44228")
        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [target]}),
            _mock_response({"vulnerabilities": [target]}),
        ]
        mock_requests_get.return_value = _mock_response({"data": []})

        # Pass max_results=100 — internally capped to 50
        result = get_cve_neighbors(_tool_use(max_results=100))
        assert result["status"] == "success"

    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_neighbor_without_description(self, mock_nvd_get, mock_requests_get):
        """Neighbors without English description should use fallback text."""
        target = _nvd_cve_record("CVE-2021-44228")
        neighbor_record = {
            "cve": {
                "id": "CVE-2021-45046",
                "published": "2021-12-14T00:00:00.000",
                "descriptions": [],  # No description
                "metrics": {
                    "cvssMetricV31": [
                        {"cvssData": {"baseScore": 9.0, "baseSeverity": "CRITICAL", "version": "3.1"}}
                    ]
                },
                "configurations": [
                    {
                        "nodes": [
                            {
                                "cpeMatch": [
                                    {
                                        "vulnerable": True,
                                        "criteria": "cpe:2.3:a:apache:log4j:2.15.0:*:*:*:*:*:*:*",
                                    }
                                ]
                            }
                        ]
                    }
                ],
            }
        }

        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [target]}),
            _mock_response({"vulnerabilities": [target, neighbor_record]}),
        ]
        mock_requests_get.return_value = _mock_response({"data": []})

        result = get_cve_neighbors(_tool_use())
        json_block = next(c for c in result["content"] if "json" in c)["json"]
        assert json_block["neighbors"][0]["description"] == "No description available."

    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_text_output_format(self, mock_nvd_get, mock_requests_get):
        """Verify the text output contains expected formatting."""
        target = _nvd_cve_record("CVE-2021-44228")
        neighbor = _nvd_cve_record("CVE-2021-45046", cvss_score=9.0, severity="CRITICAL")
        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [target]}),
            _mock_response({"vulnerabilities": [target, neighbor]}),
        ]
        mock_requests_get.return_value = _mock_response(
            {"data": [{"cve": "CVE-2021-45046", "epss": "0.5", "percentile": "0.9"}]}
        )

        result = get_cve_neighbors(_tool_use())
        text = result["content"][0]["text"]
        assert "neighboring CVE" in text
        assert "apache:log4j" in text
        assert "CVE-2021-45046" in text
        assert "EPSS=" in text
        assert "CVSS=" in text

    @patch("manus_agent.tools.get_cve_neighbors.requests.get")
    @patch("manus_agent.tools.get_cve_neighbors._nvd_get")
    def test_affected_products_in_json(self, mock_nvd_get, mock_requests_get):
        """JSON output should include all affected products from the target CVE."""
        target = _nvd_cve_record("CVE-2021-44228")
        mock_nvd_get.side_effect = [
            _mock_response({"vulnerabilities": [target]}),
            _mock_response({"vulnerabilities": [target]}),
        ]
        mock_requests_get.return_value = _mock_response({"data": []})

        result = get_cve_neighbors(_tool_use())
        # Even with no neighbors, the text mentions the product
        text = result["content"][0]["text"]
        assert "apache:log4j" in text
