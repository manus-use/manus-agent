"""Comprehensive test suite for correlate_cves tool and CLI subcommand.

All HTTP calls are 100% mocked — no real network requests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.correlate_cves import (
    TOOL_SPEC,
    _bulk_epss,
    _enrich_results,
    _extract_cpe_products,
    _extract_cwes,
    _fetch_kev_set,
    _fetch_seed_cve,
    _nvd_get,
    _nvd_headers,
    _search_by_cpe,
    _search_by_cwe,
    _summarize_cve,
    correlate_cves,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SEED_CVE_RECORD = {
    "id": "CVE-2024-3094",
    "descriptions": [{"lang": "en", "value": "Backdoor in xz/liblzma compression library."}],
    "metrics": {
        "cvssMetricV31": [
            {
                "cvssData": {
                    "baseScore": 10.0,
                    "baseSeverity": "CRITICAL",
                }
            }
        ]
    },
    "weaknesses": [
        {"description": [{"lang": "en", "value": "CWE-506"}]},
    ],
    "configurations": [
        {
            "nodes": [
                {
                    "cpeMatch": [
                        {
                            "vulnerable": True,
                            "criteria": "cpe:2.3:a:tukaani:xz:5.6.0:*:*:*:*:*:*:*",
                        },
                        {
                            "vulnerable": True,
                            "criteria": "cpe:2.3:a:tukaani:xz:5.6.1:*:*:*:*:*:*:*",
                        },
                    ]
                }
            ]
        }
    ],
    "published": "2024-03-29T17:15:00.000",
}

_CORRELATED_CVE_RECORD = {
    "id": "CVE-2024-9999",
    "descriptions": [{"lang": "en", "value": "Another vulnerability in xz."}],
    "metrics": {
        "cvssMetricV31": [
            {
                "cvssData": {
                    "baseScore": 7.5,
                    "baseSeverity": "HIGH",
                }
            }
        ]
    },
    "weaknesses": [
        {"description": [{"lang": "en", "value": "CWE-506"}]},
    ],
    "configurations": [
        {
            "nodes": [
                {
                    "cpeMatch": [
                        {
                            "vulnerable": True,
                            "criteria": "cpe:2.3:a:tukaani:xz:5.4.0:*:*:*:*:*:*:*",
                        }
                    ]
                }
            ]
        }
    ],
    "published": "2024-04-01T10:00:00.000",
}

_CWE_CORRELATED_CVE = {
    "id": "CVE-2023-1111",
    "descriptions": [{"lang": "en", "value": "Different software with same CWE."}],
    "metrics": {
        "cvssMetricV31": [
            {
                "cvssData": {
                    "baseScore": 8.0,
                    "baseSeverity": "HIGH",
                }
            }
        ]
    },
    "weaknesses": [
        {"description": [{"lang": "en", "value": "CWE-506"}]},
    ],
    "configurations": [],
    "published": "2023-06-15T12:00:00.000",
}


def _make_tool_input(cve_id: str = "CVE-2024-3094", max_results: int = 20) -> dict:
    return {
        "toolUseId": "test-correlate",
        "input": {"cve_id": cve_id, "max_results": max_results},
    }


def _mock_nvd_response(vulnerabilities: list[dict], total: int | None = None) -> MagicMock:
    """Create a mock NVD API response."""
    resp = MagicMock()
    resp.status_code = 200
    data = {
        "vulnerabilities": [{"cve": v} for v in vulnerabilities],
        "totalResults": total if total is not None else len(vulnerabilities),
    }
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _mock_epss_response(data: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": data}
    resp.raise_for_status = MagicMock()
    return resp


def _mock_kev_response(cve_ids: list[str]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"vulnerabilities": [{"cveID": cid} for cid in cve_ids]}
    resp.raise_for_status = MagicMock()
    return resp


# ===========================================================================
# TOOL_SPEC validation
# ===========================================================================


class TestToolSpec:
    def test_tool_spec_has_required_keys(self):
        assert "name" in TOOL_SPEC
        assert "description" in TOOL_SPEC
        assert "inputSchema" in TOOL_SPEC

    def test_tool_spec_name(self):
        assert TOOL_SPEC["name"] == "correlate_cves"

    def test_tool_spec_input_schema_requires_cve_id(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_tool_spec_has_max_results_param(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "max_results" in schema["properties"]
        assert schema["properties"]["max_results"]["default"] == 20


# ===========================================================================
# CVE-ID validation
# ===========================================================================


class TestCveIdValidation:
    def test_invalid_cve_id_empty(self):
        result = correlate_cves(_make_tool_input(""))
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_invalid_cve_id_no_prefix(self):
        result = correlate_cves(_make_tool_input("2024-3094"))
        assert result["status"] == "error"

    def test_invalid_cve_id_short_digits(self):
        result = correlate_cves(_make_tool_input("CVE-2024-123"))
        assert result["status"] == "error"

    def test_invalid_cve_id_non_string(self):
        tool_input = {"toolUseId": "test", "input": {"cve_id": 12345}}
        result = correlate_cves(tool_input)
        assert result["status"] == "error"

    def test_invalid_cve_id_garbage(self):
        result = correlate_cves(_make_tool_input("not-a-cve"))
        assert result["status"] == "error"


# ===========================================================================
# CPE extraction
# ===========================================================================


class TestExtractCpeProducts:
    def test_extracts_vendor_product(self):
        products = _extract_cpe_products(_SEED_CVE_RECORD)
        assert len(products) == 1
        assert products[0]["vendor"] == "tukaani"
        assert products[0]["product"] == "xz"

    def test_deduplicates_same_product(self):
        """Two CPE entries with same vendor:product should produce one result."""
        products = _extract_cpe_products(_SEED_CVE_RECORD)
        assert len(products) == 1

    def test_skips_wildcard_vendor(self):
        record = {
            "configurations": [
                {"nodes": [{"cpeMatch": [{"vulnerable": True, "criteria": "cpe:2.3:a:*:product:1.0:*:*:*:*:*:*:*"}]}]}
            ]
        }
        products = _extract_cpe_products(record)
        assert len(products) == 0

    def test_skips_wildcard_product(self):
        record = {
            "configurations": [
                {"nodes": [{"cpeMatch": [{"vulnerable": True, "criteria": "cpe:2.3:a:vendor:*:1.0:*:*:*:*:*:*:*"}]}]}
            ]
        }
        products = _extract_cpe_products(record)
        assert len(products) == 0

    def test_skips_non_vulnerable(self):
        record = {
            "configurations": [
                {
                    "nodes": [
                        {"cpeMatch": [{"vulnerable": False, "criteria": "cpe:2.3:a:vendor:product:1.0:*:*:*:*:*:*:*"}]}
                    ]
                }
            ]
        }
        products = _extract_cpe_products(record)
        assert len(products) == 0

    def test_empty_configurations(self):
        products = _extract_cpe_products({"configurations": []})
        assert len(products) == 0

    def test_missing_configurations_key(self):
        products = _extract_cpe_products({})
        assert len(products) == 0

    def test_multiple_products(self):
        record = {
            "configurations": [
                {
                    "nodes": [
                        {
                            "cpeMatch": [
                                {"vulnerable": True, "criteria": "cpe:2.3:a:apache:httpd:2.4.49:*:*:*:*:*:*:*"},
                                {"vulnerable": True, "criteria": "cpe:2.3:a:apache:tomcat:9.0.0:*:*:*:*:*:*:*"},
                            ]
                        }
                    ]
                }
            ]
        }
        products = _extract_cpe_products(record)
        assert len(products) == 2
        vendors = {p["vendor"] for p in products}
        products_set = {p["product"] for p in products}
        assert "apache" in vendors
        assert "httpd" in products_set
        assert "tomcat" in products_set

    def test_short_cpe_string(self):
        """CPE with fewer than 5 parts should be skipped."""
        record = {"configurations": [{"nodes": [{"cpeMatch": [{"vulnerable": True, "criteria": "cpe:2.3:a:short"}]}]}]}
        products = _extract_cpe_products(record)
        assert len(products) == 0


# ===========================================================================
# CWE extraction
# ===========================================================================


class TestExtractCwes:
    def test_extracts_cwe(self):
        cwes = _extract_cwes(_SEED_CVE_RECORD)
        assert cwes == ["CWE-506"]

    def test_skips_noinfo(self):
        record = {"weaknesses": [{"description": [{"value": "CWE-noinfo"}]}]}
        cwes = _extract_cwes(record)
        assert cwes == []

    def test_deduplicates(self):
        record = {
            "weaknesses": [
                {"description": [{"value": "CWE-79"}]},
                {"description": [{"value": "CWE-79"}]},
            ]
        }
        cwes = _extract_cwes(record)
        assert cwes == ["CWE-79"]

    def test_multiple_cwes(self):
        record = {
            "weaknesses": [
                {"description": [{"value": "CWE-79"}]},
                {"description": [{"value": "CWE-89"}]},
            ]
        }
        cwes = _extract_cwes(record)
        assert cwes == ["CWE-79", "CWE-89"]

    def test_empty_weaknesses(self):
        cwes = _extract_cwes({"weaknesses": []})
        assert cwes == []

    def test_missing_weaknesses_key(self):
        cwes = _extract_cwes({})
        assert cwes == []

    def test_non_cwe_values_skipped(self):
        record = {"weaknesses": [{"description": [{"value": "NVD-CWE-Other"}]}]}
        cwes = _extract_cwes(record)
        assert cwes == []


# ===========================================================================
# _summarize_cve
# ===========================================================================


class TestSummarizeCve:
    def test_basic_summary(self):
        s = _summarize_cve(_SEED_CVE_RECORD)
        assert s["cve_id"] == "CVE-2024-3094"
        assert s["cvss_score"] == 10.0
        assert s["cvss_severity"] == "CRITICAL"
        assert "xz" in s["description"].lower()
        assert s["cwes"] == ["CWE-506"]
        assert s["published"] == "2024-03-29"

    def test_truncates_long_description(self):
        record = {
            "id": "CVE-2024-0001",
            "descriptions": [{"lang": "en", "value": "A" * 300}],
            "metrics": {},
            "weaknesses": [],
            "published": "",
        }
        s = _summarize_cve(record)
        assert len(s["description"]) <= 201  # 200 + "…"
        assert s["description"].endswith("…")

    def test_no_english_description(self):
        record = {
            "id": "CVE-2024-0002",
            "descriptions": [{"lang": "fr", "value": "French only"}],
            "metrics": {},
            "weaknesses": [],
            "published": "",
        }
        s = _summarize_cve(record)
        assert s["description"] == ""

    def test_fallback_cvss_v30(self):
        record = {
            "id": "CVE-2024-0003",
            "descriptions": [],
            "metrics": {"cvssMetricV30": [{"cvssData": {"baseScore": 5.0, "baseSeverity": "MEDIUM"}}]},
            "weaknesses": [],
            "published": "",
        }
        s = _summarize_cve(record)
        assert s["cvss_score"] == 5.0
        assert s["cvss_severity"] == "MEDIUM"

    def test_fallback_cvss_v2(self):
        record = {
            "id": "CVE-2024-0004",
            "descriptions": [],
            "metrics": {"cvssMetricV2": [{"cvssData": {"baseScore": 4.0, "baseSeverity": "MEDIUM"}}]},
            "weaknesses": [],
            "published": "",
        }
        s = _summarize_cve(record)
        assert s["cvss_score"] == 4.0

    def test_no_cvss_metrics(self):
        record = {
            "id": "CVE-2024-0005",
            "descriptions": [],
            "metrics": {},
            "weaknesses": [],
            "published": "",
        }
        s = _summarize_cve(record)
        assert s["cvss_score"] is None
        assert s["cvss_severity"] == "UNKNOWN"

    def test_published_date_truncated(self):
        record = {
            "id": "CVE-2024-0006",
            "descriptions": [],
            "metrics": {},
            "weaknesses": [],
            "published": "2024-01-15T12:00:00.000",
        }
        s = _summarize_cve(record)
        assert s["published"] == "2024-01-15"


# ===========================================================================
# NVD headers
# ===========================================================================


class TestNvdHeaders:
    def test_headers_without_api_key(self, monkeypatch):
        monkeypatch.delenv("NVD_API_KEY", raising=False)
        headers = _nvd_headers()
        assert "User-Agent" in headers
        assert "apiKey" not in headers

    def test_headers_with_api_key(self, monkeypatch):
        monkeypatch.setenv("NVD_API_KEY", "test-key-123")
        headers = _nvd_headers()
        assert headers["apiKey"] == "test-key-123"

    def test_headers_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("NVD_API_KEY", "  key  ")
        headers = _nvd_headers()
        assert headers["apiKey"] == "key"

    def test_headers_empty_key_ignored(self, monkeypatch):
        monkeypatch.setenv("NVD_API_KEY", "   ")
        headers = _nvd_headers()
        assert "apiKey" not in headers


# ===========================================================================
# _nvd_get retry logic
# ===========================================================================


class TestNvdGet:
    @patch("manus_agent.tools.correlate_cves.requests.get")
    @patch("manus_agent.tools.correlate_cves.time.sleep")
    def test_success_on_first_try(self, mock_sleep, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _nvd_get("https://example.com")
        assert result == mock_resp
        mock_sleep.assert_not_called()

    @patch("manus_agent.tools.correlate_cves.requests.get")
    @patch("manus_agent.tools.correlate_cves.time.sleep")
    def test_retry_on_403(self, mock_sleep, mock_get):
        forbidden = MagicMock()
        forbidden.status_code = 403

        success = MagicMock()
        success.status_code = 200
        success.raise_for_status = MagicMock()

        mock_get.side_effect = [forbidden, success]
        result = _nvd_get("https://example.com")
        assert result == success

    @patch("manus_agent.tools.correlate_cves.requests.get")
    @patch("manus_agent.tools.correlate_cves.time.sleep")
    def test_retry_on_request_exception(self, mock_sleep, mock_get):
        import requests as req

        mock_get.side_effect = [
            req.exceptions.ConnectionError("conn fail"),
            MagicMock(status_code=200, raise_for_status=MagicMock()),
        ]
        result = _nvd_get("https://example.com")
        assert result.status_code == 200

    @patch("manus_agent.tools.correlate_cves.requests.get")
    @patch("manus_agent.tools.correlate_cves.time.sleep")
    def test_raises_after_all_retries_exhausted(self, mock_sleep, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("permanent")
        with pytest.raises(req.exceptions.ConnectionError):
            _nvd_get("https://example.com")
        assert mock_get.call_count == 3


# ===========================================================================
# _fetch_seed_cve
# ===========================================================================


class TestFetchSeedCve:
    @patch("manus_agent.tools.correlate_cves._nvd_get")
    def test_returns_cve_record(self, mock_get):
        mock_get.return_value = _mock_nvd_response([_SEED_CVE_RECORD])
        result = _fetch_seed_cve("CVE-2024-3094")
        assert result["id"] == "CVE-2024-3094"

    @patch("manus_agent.tools.correlate_cves._nvd_get")
    def test_returns_none_when_not_found(self, mock_get):
        mock_get.return_value = _mock_nvd_response([])
        result = _fetch_seed_cve("CVE-2099-0001")
        assert result is None


# ===========================================================================
# _search_by_cpe
# ===========================================================================


class TestSearchByCpe:
    @patch("manus_agent.tools.correlate_cves._nvd_get")
    def test_finds_correlated_cves(self, mock_get):
        mock_get.return_value = _mock_nvd_response([_CORRELATED_CVE_RECORD])
        results = _search_by_cpe("tukaani", "xz", "CVE-2024-3094", 20)
        assert len(results) == 1
        assert results[0]["cve_id"] == "CVE-2024-9999"

    @patch("manus_agent.tools.correlate_cves._nvd_get")
    def test_excludes_seed_cve(self, mock_get):
        mock_get.return_value = _mock_nvd_response([_SEED_CVE_RECORD])
        results = _search_by_cpe("tukaani", "xz", "CVE-2024-3094", 20)
        assert len(results) == 0

    @patch("manus_agent.tools.correlate_cves._nvd_get")
    def test_excludes_non_matching_cpe(self, mock_get):
        """CVEs returned by keyword search but without matching CPE are filtered."""
        non_matching = {
            **_CWE_CORRELATED_CVE,
            "configurations": [
                {
                    "nodes": [
                        {"cpeMatch": [{"vulnerable": True, "criteria": "cpe:2.3:a:other:product:1.0:*:*:*:*:*:*:*"}]}
                    ]
                }
            ],
        }
        mock_get.return_value = _mock_nvd_response([non_matching])
        results = _search_by_cpe("tukaani", "xz", "CVE-2024-3094", 20)
        assert len(results) == 0

    @patch("manus_agent.tools.correlate_cves._nvd_get")
    def test_respects_max_results(self, mock_get):
        cves = []
        for i in range(10):
            cve = {
                **_CORRELATED_CVE_RECORD,
                "id": f"CVE-2024-{9000 + i}",
            }
            cves.append(cve)
        mock_get.return_value = _mock_nvd_response(cves)
        results = _search_by_cpe("tukaani", "xz", "CVE-2024-3094", 5)
        assert len(results) <= 5

    @patch("manus_agent.tools.correlate_cves._nvd_get")
    def test_handles_api_error(self, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("fail")
        results = _search_by_cpe("tukaani", "xz", "CVE-2024-3094", 20)
        assert results == []


# ===========================================================================
# _search_by_cwe
# ===========================================================================


class TestSearchByCwe:
    @patch("manus_agent.tools.correlate_cves._nvd_get")
    def test_finds_cves_with_same_cwe(self, mock_get):
        mock_get.return_value = _mock_nvd_response([_CWE_CORRELATED_CVE])
        results = _search_by_cwe("CWE-506", "CVE-2024-3094", 20)
        assert len(results) == 1
        assert results[0]["cve_id"] == "CVE-2023-1111"

    @patch("manus_agent.tools.correlate_cves._nvd_get")
    def test_excludes_seed_cve(self, mock_get):
        mock_get.return_value = _mock_nvd_response([_SEED_CVE_RECORD])
        results = _search_by_cwe("CWE-506", "CVE-2024-3094", 20)
        assert len(results) == 0

    @patch("manus_agent.tools.correlate_cves._nvd_get")
    def test_handles_api_error(self, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("fail")
        results = _search_by_cwe("CWE-506", "CVE-2024-3094", 20)
        assert results == []


# ===========================================================================
# _bulk_epss
# ===========================================================================


class TestBulkEpss:
    @patch("manus_agent.tools.correlate_cves.requests.get")
    def test_fetches_epss_scores(self, mock_get):
        mock_get.return_value = _mock_epss_response(
            [
                {"cve": "CVE-2024-9999", "epss": "0.543", "percentile": "0.95"},
            ]
        )
        result = _bulk_epss(["CVE-2024-9999"])
        assert "CVE-2024-9999" in result
        assert abs(result["CVE-2024-9999"]["epss"] - 0.543) < 0.001

    @patch("manus_agent.tools.correlate_cves.requests.get")
    def test_handles_empty_list(self, mock_get):
        result = _bulk_epss([])
        assert result == {}
        mock_get.assert_not_called()

    @patch("manus_agent.tools.correlate_cves.requests.get")
    def test_handles_api_error(self, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("fail")
        result = _bulk_epss(["CVE-2024-9999"])
        assert result == {}

    @patch("manus_agent.tools.correlate_cves.requests.get")
    def test_chunks_large_batches(self, mock_get):
        """More than 100 CVEs should result in multiple API calls."""
        cve_ids = [f"CVE-2024-{i:04d}" for i in range(150)]
        mock_get.return_value = _mock_epss_response([])
        _bulk_epss(cve_ids)
        assert mock_get.call_count == 2  # 100 + 50


# ===========================================================================
# _fetch_kev_set
# ===========================================================================


class TestFetchKevSet:
    @patch("manus_agent.tools.correlate_cves.requests.get")
    def test_returns_set_of_cve_ids(self, mock_get):
        mock_get.return_value = _mock_kev_response(["CVE-2024-3094", "CVE-2021-44228"])
        result = _fetch_kev_set()
        assert "CVE-2024-3094" in result
        assert "CVE-2021-44228" in result

    @patch("manus_agent.tools.correlate_cves.requests.get")
    def test_handles_api_error(self, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("fail")
        result = _fetch_kev_set()
        assert result == set()


# ===========================================================================
# _enrich_results
# ===========================================================================


class TestEnrichResults:
    def test_adds_epss_and_kev(self):
        items = [{"cve_id": "CVE-2024-9999", "cvss_score": 7.5}]
        epss_map = {"CVE-2024-9999": {"epss": 0.543, "percentile": 0.95}}
        kev_set = {"CVE-2024-9999"}

        enriched = _enrich_results(items, epss_map, kev_set)
        assert enriched[0]["epss_score"] == 0.543
        assert enriched[0]["in_kev"] is True

    def test_missing_epss_data(self):
        items = [{"cve_id": "CVE-2024-0001", "cvss_score": 5.0}]
        enriched = _enrich_results(items, {}, set())
        assert enriched[0]["epss_score"] is None
        assert enriched[0]["in_kev"] is False

    def test_sorts_kev_first(self):
        items = [
            {"cve_id": "CVE-A", "cvss_score": 5.0},
            {"cve_id": "CVE-B", "cvss_score": 9.0},
        ]
        epss_map = {
            "CVE-A": {"epss": 0.9, "percentile": 0.99},
            "CVE-B": {"epss": 0.1, "percentile": 0.50},
        }
        kev_set = {"CVE-B"}
        enriched = _enrich_results(items, epss_map, kev_set)
        assert enriched[0]["cve_id"] == "CVE-B"  # KEV comes first

    def test_sorts_by_epss_within_non_kev(self):
        items = [
            {"cve_id": "CVE-A", "cvss_score": 5.0},
            {"cve_id": "CVE-B", "cvss_score": 9.0},
        ]
        epss_map = {
            "CVE-A": {"epss": 0.9, "percentile": 0.99},
            "CVE-B": {"epss": 0.1, "percentile": 0.50},
        }
        enriched = _enrich_results(items, epss_map, set())
        assert enriched[0]["cve_id"] == "CVE-A"  # Higher EPSS first


# ===========================================================================
# Full tool function — correlate_cves
# ===========================================================================


class TestCorrelateCves:
    @patch("manus_agent.tools.correlate_cves._fetch_kev_set")
    @patch("manus_agent.tools.correlate_cves._bulk_epss")
    @patch("manus_agent.tools.correlate_cves._search_by_cwe")
    @patch("manus_agent.tools.correlate_cves._search_by_cpe")
    @patch("manus_agent.tools.correlate_cves._fetch_seed_cve")
    @patch("manus_agent.tools.correlate_cves.time.sleep")
    def test_full_correlation_flow(
        self, mock_sleep, mock_fetch_seed, mock_search_cpe, mock_search_cwe, mock_epss, mock_kev
    ):
        mock_fetch_seed.return_value = _SEED_CVE_RECORD
        mock_search_cpe.return_value = [_summarize_cve(_CORRELATED_CVE_RECORD)]
        mock_search_cwe.return_value = [_summarize_cve(_CWE_CORRELATED_CVE)]
        mock_epss.return_value = {
            "CVE-2024-9999": {"epss": 0.5, "percentile": 0.9},
            "CVE-2023-1111": {"epss": 0.2, "percentile": 0.7},
        }
        mock_kev.return_value = {"CVE-2024-9999"}

        result = correlate_cves(_make_tool_input())
        assert result["status"] == "success"

        # Check text content
        text = result["content"][0]["text"]
        assert "CVE-2024-3094" in text
        assert "tukaani:xz" in text

        # Check JSON content
        json_data = result["content"][1]["json"]
        assert json_data["seed_cve"] == "CVE-2024-3094"
        assert json_data["total_unique_correlated"] == 2
        assert json_data["kev_correlated_count"] == 1
        assert len(json_data["correlations"]["by_component"]) == 1
        assert len(json_data["correlations"]["by_weakness"]) == 1

    @patch("manus_agent.tools.correlate_cves._fetch_seed_cve")
    def test_seed_cve_not_found(self, mock_fetch_seed):
        mock_fetch_seed.return_value = None
        result = correlate_cves(_make_tool_input())
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    @patch("manus_agent.tools.correlate_cves._fetch_seed_cve")
    def test_seed_cve_fetch_error(self, mock_fetch_seed):
        import requests as req

        mock_fetch_seed.side_effect = req.exceptions.ConnectionError("NVD down")
        result = correlate_cves(_make_tool_input())
        assert result["status"] == "error"
        assert "Failed to fetch" in result["content"][0]["text"]

    @patch("manus_agent.tools.correlate_cves._fetch_seed_cve")
    def test_no_correlation_dimensions(self, mock_fetch_seed):
        """CVE with no CPEs and no CWEs returns empty correlation."""
        mock_fetch_seed.return_value = {
            "id": "CVE-2024-3094",
            "configurations": [],
            "weaknesses": [],
        }
        result = correlate_cves(_make_tool_input())
        assert result["status"] == "success"
        json_data = result["content"][1]["json"]
        assert json_data["total_unique_correlated"] == 0

    @patch("manus_agent.tools.correlate_cves._fetch_kev_set")
    @patch("manus_agent.tools.correlate_cves._bulk_epss")
    @patch("manus_agent.tools.correlate_cves._search_by_cwe")
    @patch("manus_agent.tools.correlate_cves._search_by_cpe")
    @patch("manus_agent.tools.correlate_cves._fetch_seed_cve")
    @patch("manus_agent.tools.correlate_cves.time.sleep")
    def test_deduplication_across_dimensions(
        self, mock_sleep, mock_fetch_seed, mock_search_cpe, mock_search_cwe, mock_epss, mock_kev
    ):
        """Same CVE found by both CPE and CWE should appear only once in total count."""
        shared_cve = _summarize_cve(_CORRELATED_CVE_RECORD)
        mock_fetch_seed.return_value = _SEED_CVE_RECORD
        mock_search_cpe.return_value = [shared_cve]
        mock_search_cwe.return_value = [shared_cve]  # Same CVE
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        result = correlate_cves(_make_tool_input())
        json_data = result["content"][1]["json"]
        # CPE search finds it, CWE search sees it's already in seen_ids
        assert json_data["total_unique_correlated"] == 1
        assert len(json_data["correlations"]["by_component"]) == 1
        assert len(json_data["correlations"]["by_weakness"]) == 0

    @patch("manus_agent.tools.correlate_cves._fetch_kev_set")
    @patch("manus_agent.tools.correlate_cves._bulk_epss")
    @patch("manus_agent.tools.correlate_cves._search_by_cwe")
    @patch("manus_agent.tools.correlate_cves._search_by_cpe")
    @patch("manus_agent.tools.correlate_cves._fetch_seed_cve")
    @patch("manus_agent.tools.correlate_cves.time.sleep")
    def test_max_results_respected(
        self, mock_sleep, mock_fetch_seed, mock_search_cpe, mock_search_cwe, mock_epss, mock_kev
    ):
        mock_fetch_seed.return_value = _SEED_CVE_RECORD
        many_cves = []
        for i in range(30):
            cve = {**_summarize_cve(_CORRELATED_CVE_RECORD), "cve_id": f"CVE-2024-{8000 + i}"}
            many_cves.append(cve)
        mock_search_cpe.return_value = many_cves
        mock_search_cwe.return_value = []
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        result = correlate_cves(_make_tool_input(max_results=5))
        json_data = result["content"][1]["json"]
        assert len(json_data["correlations"]["by_component"]) <= 5

    def test_cve_id_case_insensitive(self):
        """Lowercase CVE ID should be normalized to uppercase."""
        tool_input = _make_tool_input("cve-2024-3094")
        with patch("manus_agent.tools.correlate_cves._fetch_seed_cve") as mock_fetch:
            mock_fetch.return_value = None
            correlate_cves(tool_input)
            # Should have called with uppercase
            mock_fetch.assert_called_once_with("CVE-2024-3094")

    def test_cve_id_whitespace_stripped(self):
        tool_input = _make_tool_input("  CVE-2024-3094  ")
        with patch("manus_agent.tools.correlate_cves._fetch_seed_cve") as mock_fetch:
            mock_fetch.return_value = None
            correlate_cves(tool_input)
            mock_fetch.assert_called_once_with("CVE-2024-3094")

    @patch("manus_agent.tools.correlate_cves._fetch_kev_set")
    @patch("manus_agent.tools.correlate_cves._bulk_epss")
    @patch("manus_agent.tools.correlate_cves._search_by_cwe")
    @patch("manus_agent.tools.correlate_cves._search_by_cpe")
    @patch("manus_agent.tools.correlate_cves._fetch_seed_cve")
    @patch("manus_agent.tools.correlate_cves.time.sleep")
    def test_kev_and_high_epss_counts(
        self, mock_sleep, mock_fetch_seed, mock_search_cpe, mock_search_cwe, mock_epss, mock_kev
    ):
        mock_fetch_seed.return_value = _SEED_CVE_RECORD
        cves = [
            {**_summarize_cve(_CORRELATED_CVE_RECORD), "cve_id": "CVE-2024-9001"},
            {**_summarize_cve(_CORRELATED_CVE_RECORD), "cve_id": "CVE-2024-9002"},
            {**_summarize_cve(_CORRELATED_CVE_RECORD), "cve_id": "CVE-2024-9003"},
        ]
        mock_search_cpe.return_value = cves
        mock_search_cwe.return_value = []
        mock_epss.return_value = {
            "CVE-2024-9001": {"epss": 0.5, "percentile": 0.9},
            "CVE-2024-9002": {"epss": 0.05, "percentile": 0.3},
            "CVE-2024-9003": {"epss": 0.2, "percentile": 0.8},
        }
        mock_kev.return_value = {"CVE-2024-9001", "CVE-2024-9003"}

        result = correlate_cves(_make_tool_input())
        json_data = result["content"][1]["json"]
        assert json_data["kev_correlated_count"] == 2
        assert json_data["high_epss_count"] == 2  # 0.5 and 0.2 are >= 0.1

    @patch("manus_agent.tools.correlate_cves._fetch_kev_set")
    @patch("manus_agent.tools.correlate_cves._bulk_epss")
    @patch("manus_agent.tools.correlate_cves._search_by_cwe")
    @patch("manus_agent.tools.correlate_cves._search_by_cpe")
    @patch("manus_agent.tools.correlate_cves._fetch_seed_cve")
    @patch("manus_agent.tools.correlate_cves.time.sleep")
    def test_text_output_contains_key_info(
        self, mock_sleep, mock_fetch_seed, mock_search_cpe, mock_search_cwe, mock_epss, mock_kev
    ):
        mock_fetch_seed.return_value = _SEED_CVE_RECORD
        mock_search_cpe.return_value = [_summarize_cve(_CORRELATED_CVE_RECORD)]
        mock_search_cwe.return_value = []
        mock_epss.return_value = {"CVE-2024-9999": {"epss": 0.5, "percentile": 0.9}}
        mock_kev.return_value = {"CVE-2024-9999"}

        result = correlate_cves(_make_tool_input())
        text = result["content"][0]["text"]
        assert "CVE-2024-3094" in text
        assert "By Component" in text
        assert "CVE-2024-9999" in text
        assert "KEV" in text

    @patch("manus_agent.tools.correlate_cves._fetch_kev_set")
    @patch("manus_agent.tools.correlate_cves._bulk_epss")
    @patch("manus_agent.tools.correlate_cves._search_by_cwe")
    @patch("manus_agent.tools.correlate_cves._search_by_cpe")
    @patch("manus_agent.tools.correlate_cves._fetch_seed_cve")
    @patch("manus_agent.tools.correlate_cves.time.sleep")
    def test_only_cwe_correlations(
        self, mock_sleep, mock_fetch_seed, mock_search_cpe, mock_search_cwe, mock_epss, mock_kev
    ):
        """When seed has no CPEs but has CWEs, only CWE correlations are returned."""
        seed = {
            "id": "CVE-2024-3094",
            "configurations": [],
            "weaknesses": [{"description": [{"value": "CWE-506"}]}],
        }
        mock_fetch_seed.return_value = seed
        mock_search_cpe.return_value = []
        mock_search_cwe.return_value = [_summarize_cve(_CWE_CORRELATED_CVE)]
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        result = correlate_cves(_make_tool_input())
        json_data = result["content"][1]["json"]
        assert len(json_data["correlations"]["by_component"]) == 0
        assert len(json_data["correlations"]["by_weakness"]) == 1


# ===========================================================================
# CLI subcommand — _build_correlate_parser / _run_correlate
# ===========================================================================


class TestCorrelateParser:
    def test_parser_accepts_cve_id(self):
        from manus_agent.cli import _build_correlate_parser

        parser = _build_correlate_parser()
        args = parser.parse_args(["CVE-2024-3094"])
        assert args.cve_id == "CVE-2024-3094"

    def test_parser_default_max_results(self):
        from manus_agent.cli import _build_correlate_parser

        parser = _build_correlate_parser()
        args = parser.parse_args(["CVE-2024-3094"])
        assert args.max_results == 20

    def test_parser_custom_max_results(self):
        from manus_agent.cli import _build_correlate_parser

        parser = _build_correlate_parser()
        args = parser.parse_args(["CVE-2024-3094", "--max-results", "10"])
        assert args.max_results == 10

    def test_parser_default_output(self):
        from manus_agent.cli import _build_correlate_parser

        parser = _build_correlate_parser()
        args = parser.parse_args(["CVE-2024-3094"])
        assert args.output == "text"

    def test_parser_json_output(self):
        from manus_agent.cli import _build_correlate_parser

        parser = _build_correlate_parser()
        args = parser.parse_args(["CVE-2024-3094", "--output", "json"])
        assert args.output == "json"

    def test_parser_rejects_no_args(self):
        from manus_agent.cli import _build_correlate_parser

        parser = _build_correlate_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestRunCorrelate:
    @patch("manus_agent.tools.correlate_cves.correlate_cves")
    @patch("manus_agent.cli.console")
    def test_text_output(self, mock_console, mock_correlate):
        from manus_agent.cli import _run_correlate

        mock_correlate.return_value = {
            "status": "success",
            "content": [
                {"text": "Correlation report for CVE-2024-3094"},
                {"json": {"seed_cve": "CVE-2024-3094", "total_unique_correlated": 5}},
            ],
        }
        exit_code = _run_correlate(["CVE-2024-3094"])
        assert exit_code == 0
        mock_console.print.assert_called()

    @patch("manus_agent.tools.correlate_cves.correlate_cves")
    @patch("manus_agent.cli.console")
    def test_json_output(self, mock_console, mock_correlate):
        from manus_agent.cli import _run_correlate

        mock_correlate.return_value = {
            "status": "success",
            "content": [
                {"text": "report text"},
                {"json": {"seed_cve": "CVE-2024-3094", "total_unique_correlated": 5}},
            ],
        }
        exit_code = _run_correlate(["CVE-2024-3094", "--output", "json"])
        assert exit_code == 0
        mock_console.print_json.assert_called()

    @patch("manus_agent.cli.console")
    def test_invalid_cve_id_rejected(self, mock_console):
        from manus_agent.cli import _run_correlate

        exit_code = _run_correlate(["not-a-cve"])
        assert exit_code == 1

    @patch("manus_agent.tools.correlate_cves.correlate_cves")
    @patch("manus_agent.cli.console")
    def test_error_result(self, mock_console, mock_correlate):
        from manus_agent.cli import _run_correlate

        mock_correlate.return_value = {
            "status": "error",
            "content": [{"text": "NVD API failed"}],
        }
        exit_code = _run_correlate(["CVE-2024-3094"])
        assert exit_code == 1


# ===========================================================================
# CLI dispatch integration
# ===========================================================================


class TestCorrelateDispatch:
    @patch("manus_agent.cli._run_correlate")
    def test_main_dispatches_correlate(self, mock_run):
        """Verify main() routes 'correlate' to _run_correlate."""
        import sys

        mock_run.return_value = 0
        with patch.object(sys, "argv", ["manus-agent", "correlate", "CVE-2024-3094"]):
            with pytest.raises(SystemExit) as exc_info:
                from manus_agent.cli import main

                main()
            assert exc_info.value.code == 0
        mock_run.assert_called_once_with(["CVE-2024-3094"])
