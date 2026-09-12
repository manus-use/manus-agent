"""Comprehensive test suite for generate_vuln_report tool and vuln-report CLI subcommand."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.generate_vuln_report import (
    TOOL_SPEC,
    _compute_statistics,
    _extract_affected_products,
    _extract_cvss,
    _extract_cwe,
    _extract_description,
    _extract_published,
    _extract_references,
    _fetch_epss,
    _fetch_kev,
    _fetch_nvd,
    _fetch_osv,
    _fetch_vulncheck_kev,
    _gather_cve_data,
    _get_with_retry,
    _prefetch_kev_catalog,
    _render_finding,
    _render_markdown,
    _severity_sort_key,
    generate_report,
    handler,
)

# Type alias for test annotations
_ToolUse = dict[str, Any]

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def sample_nvd_record() -> dict[str, Any]:
    """A realistic NVD CVE record."""
    return {
        "id": "CVE-2024-3094",
        "published": "2024-03-29T07:15:00.000",
        "descriptions": [
            {"lang": "en", "value": "Malicious code was discovered in xz-utils versions 5.6.0 and 5.6.1."},
            {"lang": "es", "value": "Se descubrió código malicioso en xz-utils."},
        ],
        "metrics": {
            "cvssMetricV31": [
                {
                    "cvssData": {
                        "version": "3.1",
                        "baseScore": 10.0,
                        "baseSeverity": "CRITICAL",
                        "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                    }
                }
            ]
        },
        "weaknesses": [
            {
                "description": [
                    {"lang": "en", "value": "CWE-506"},
                ]
            }
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
                            {
                                "vulnerable": False,
                                "criteria": "cpe:2.3:a:tukaani:xz:5.6.2:*:*:*:*:*:*:*",
                            },
                        ]
                    }
                ]
            }
        ],
        "references": [
            {"url": "https://nvd.nist.gov/vuln/detail/CVE-2024-3094", "source": "nvd"},
            {"url": "https://github.com/tukaani-project/xz/security", "source": "github"},
        ],
    }


@pytest.fixture()
def sample_kev_catalog() -> list[dict[str, Any]]:
    """A CISA KEV catalog snippet."""
    return [
        {
            "cveID": "CVE-2024-3094",
            "vendorProject": "Tukaani",
            "product": "xz-utils",
            "dateAdded": "2024-03-30",
            "dueDate": "2024-04-15",
            "requiredAction": "Apply mitigations per vendor instructions.",
        },
        {
            "cveID": "CVE-2021-44228",
            "vendorProject": "Apache",
            "product": "Log4j",
            "dateAdded": "2021-12-10",
            "dueDate": "2021-12-24",
            "requiredAction": "Apply updates per vendor instructions.",
        },
    ]


@pytest.fixture()
def sample_epss_response() -> dict[str, Any]:
    """An EPSS API response."""
    return {
        "status": "OK",
        "data": [{"cve": "CVE-2024-3094", "epss": "0.97565", "percentile": "0.99978"}],
    }


@pytest.fixture()
def sample_osv_response() -> dict[str, Any]:
    """An OSV.dev API response."""
    return {
        "vulns": [
            {
                "id": "GHSA-xxxx-xxxx-xxxx",
                "affected": [
                    {
                        "package": {"ecosystem": "PyPI", "name": "xz"},
                        "ranges": [
                            {
                                "type": "ECOSYSTEM",
                                "events": [{"introduced": "5.6.0"}, {"fixed": "5.6.2"}],
                            }
                        ],
                    }
                ],
            }
        ]
    }


@pytest.fixture()
def sample_finding() -> dict[str, Any]:
    """A complete finding dict for rendering tests."""
    return {
        "cve_id": "CVE-2024-3094",
        "description": "Malicious code was discovered in xz-utils versions 5.6.0 and 5.6.1.",
        "published": "2024-03-29T07:15:00.000",
        "cvss": {
            "version": "3.1",
            "score": 10.0,
            "severity": "CRITICAL",
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
        },
        "cwes": ["CWE-506"],
        "affected_products": ["tukaani:xz"],
        "references": [
            {"url": "https://nvd.nist.gov/vuln/detail/CVE-2024-3094", "source": "nvd"},
        ],
        "epss_score": 0.97565,
        "epss_percentile": 0.99978,
        "in_cisa_kev": True,
        "kev_details": {
            "in_kev": True,
            "date_added": "2024-03-30",
            "due_date": "2024-04-15",
            "required_action": "Apply mitigations per vendor instructions.",
        },
        "vulncheck_kev": {
            "available": True,
            "in_kev": True,
            "ransomware_use": True,
            "date_added": "2024-03-30",
        },
        "osv_packages": [
            {"ecosystem": "PyPI", "name": "xz", "fixed_versions": ["5.6.2"]},
        ],
        "data_errors": [],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. TOOL_SPEC contract tests
# ═══════════════════════════════════════════════════════════════════════════


class TestToolSpec:
    """TOOL_SPEC schema contract."""

    def test_name(self):
        assert TOOL_SPEC["name"] == "generate_vuln_report"

    def test_has_description(self):
        assert len(TOOL_SPEC["description"]) > 20

    def test_input_schema_has_cve_ids(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert "cve_ids" in props
        assert props["cve_ids"]["type"] == "array"

    def test_cve_ids_is_required(self):
        assert "cve_ids" in TOOL_SPEC["inputSchema"]["json"]["required"]

    def test_optional_fields(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert "title" in props
        assert "output_format" in props

    def test_output_format_enum(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert set(props["output_format"]["enum"]) == {"markdown", "json"}


# ═══════════════════════════════════════════════════════════════════════════
# 2. HTTP retry helper tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGetWithRetry:
    """_get_with_retry retry/back-off."""

    @patch("manus_agent.tools.generate_vuln_report.requests.get")
    @patch("manus_agent.tools.generate_vuln_report._RETRY_BASE_DELAY", 0)
    def test_success_first_try(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        resp = _get_with_retry("https://example.com")
        assert resp.status_code == 200
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.generate_vuln_report.requests.get")
    @patch("manus_agent.tools.generate_vuln_report._RETRY_BASE_DELAY", 0)
    @patch("manus_agent.tools.generate_vuln_report._MAX_RETRIES", 3)
    def test_retries_on_429(self, mock_get):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_200 = MagicMock()
        resp_200.status_code = 200
        mock_get.side_effect = [resp_429, resp_200]
        resp = _get_with_retry("https://example.com")
        assert resp.status_code == 200
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.generate_vuln_report.requests.get")
    @patch("manus_agent.tools.generate_vuln_report._RETRY_BASE_DELAY", 0)
    @patch("manus_agent.tools.generate_vuln_report._MAX_RETRIES", 2)
    def test_exhausts_retries(self, mock_get):
        resp_503 = MagicMock()
        resp_503.status_code = 503
        mock_get.return_value = resp_503
        with pytest.raises(Exception):  # noqa: B017
            _get_with_retry("https://example.com")
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.generate_vuln_report.requests.get")
    @patch("manus_agent.tools.generate_vuln_report._RETRY_BASE_DELAY", 0)
    @patch("manus_agent.tools.generate_vuln_report._MAX_RETRIES", 3)
    def test_retries_on_connection_error(self, mock_get):
        import requests as _req

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        mock_get.side_effect = [_req.ConnectionError("fail"), resp_ok]
        resp = _get_with_retry("https://example.com")
        assert resp.status_code == 200

    @patch("manus_agent.tools.generate_vuln_report.requests.get")
    @patch("manus_agent.tools.generate_vuln_report._RETRY_BASE_DELAY", 0)
    def test_non_retryable_status_returned_immediately(self, mock_get):
        resp_404 = MagicMock()
        resp_404.status_code = 404
        mock_get.return_value = resp_404
        resp = _get_with_retry("https://example.com")
        assert resp.status_code == 404
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.generate_vuln_report.requests.get")
    @patch("manus_agent.tools.generate_vuln_report._RETRY_BASE_DELAY", 0)
    def test_passes_params_and_headers(self, mock_get):
        resp_ok = MagicMock()
        resp_ok.status_code = 200
        mock_get.return_value = resp_ok
        _get_with_retry("https://example.com", params={"k": "v"}, headers={"H": "val"})
        mock_get.assert_called_once_with(
            "https://example.com",
            params={"k": "v"},
            headers={"H": "val"},
            timeout=20,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. Per-source fetch tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchNvd:
    """_fetch_nvd NVD API."""

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_success(self, mock_get, sample_nvd_record):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [{"cve": sample_nvd_record}]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        result = _fetch_nvd("CVE-2024-3094")
        assert result["id"] == "CVE-2024-3094"

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_no_vulns(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        result = _fetch_nvd("CVE-9999-99999")
        assert "error" in result

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_request_exception(self, mock_get):
        import requests as _req

        mock_get.side_effect = _req.ConnectionError("fail")
        result = _fetch_nvd("CVE-2024-3094")
        assert "error" in result

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_nvd_api_key_header(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [{"cve": {"id": "CVE-2024-3094"}}]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        with patch.dict("os.environ", {"NVD_API_KEY": "test-key"}):
            _fetch_nvd("CVE-2024-3094")
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["headers"]["apiKey"] == "test-key"

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_no_api_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [{"cve": {"id": "CVE-2024-3094"}}]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        with patch.dict("os.environ", {}, clear=True):
            _fetch_nvd("CVE-2024-3094")
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["headers"] is None


class TestFetchEpss:
    """_fetch_epss EPSS API."""

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_success(self, mock_get, sample_epss_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_epss_response
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        result = _fetch_epss("CVE-2024-3094")
        assert result["epss"] == "0.97565"

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_no_data(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        result = _fetch_epss("CVE-9999-99999")
        assert "error" in result

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_exception(self, mock_get):
        import requests as _req

        mock_get.side_effect = _req.Timeout("timeout")
        result = _fetch_epss("CVE-2024-3094")
        assert "error" in result


class TestFetchKev:
    """_fetch_kev CISA KEV catalog."""

    def test_found_with_catalog(self, sample_kev_catalog):
        result = _fetch_kev("CVE-2024-3094", catalog=sample_kev_catalog)
        assert result["in_kev"] is True
        assert result["date_added"] == "2024-03-30"

    def test_not_found_with_catalog(self, sample_kev_catalog):
        result = _fetch_kev("CVE-9999-99999", catalog=sample_kev_catalog)
        assert result["in_kev"] is False

    def test_case_insensitive(self, sample_kev_catalog):
        result = _fetch_kev("cve-2024-3094", catalog=sample_kev_catalog)
        assert result["in_kev"] is True

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_fetches_catalog_when_none(self, mock_get, sample_kev_catalog):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": sample_kev_catalog}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        result = _fetch_kev("CVE-2024-3094")
        assert result["in_kev"] is True

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_catalog_fetch_error(self, mock_get):
        import requests as _req

        mock_get.side_effect = _req.ConnectionError("fail")
        result = _fetch_kev("CVE-2024-3094")
        assert result["in_kev"] is False
        assert "error" in result


class TestFetchVulncheckKev:
    """_fetch_vulncheck_kev VulnCheck KEV."""

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_success_in_kev(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {
                    "cveID": "CVE-2024-3094",
                    "knownRansomwareCampaignUse": "Known",
                    "dateAdded": "2024-03-30",
                    "dueDate": "2024-04-15",
                }
            ]
        }
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        with patch.dict("os.environ", {"VULNCHECK_API_KEY": "test-key"}):
            result = _fetch_vulncheck_kev("CVE-2024-3094")
        assert result["available"] is True
        assert result["in_kev"] is True
        assert result["ransomware_use"] is True

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_not_in_kev(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        with patch.dict("os.environ", {"VULNCHECK_API_KEY": "test-key"}):
            result = _fetch_vulncheck_kev("CVE-9999-99999")
        assert result["available"] is True
        assert result["in_kev"] is False

    def test_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            result = _fetch_vulncheck_kev("CVE-2024-3094")
        assert result["available"] is False

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_request_error(self, mock_get):
        import requests as _req

        mock_get.side_effect = _req.ConnectionError("fail")
        with patch.dict("os.environ", {"VULNCHECK_API_KEY": "test-key"}):
            result = _fetch_vulncheck_kev("CVE-2024-3094")
        assert result["available"] is False

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_ransomware_unknown(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"cveID": "CVE-2024-3094", "knownRansomwareCampaignUse": "Unknown"}]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        with patch.dict("os.environ", {"VULNCHECK_API_KEY": "test-key"}):
            result = _fetch_vulncheck_kev("CVE-2024-3094")
        assert result["ransomware_use"] is False


class TestFetchOsv:
    """_fetch_osv OSV.dev API."""

    @patch("manus_agent.tools.generate_vuln_report.requests.post")
    def test_success(self, mock_post, sample_osv_response):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_osv_response
        mock_post.return_value = mock_resp
        result = _fetch_osv("CVE-2024-3094")
        assert len(result["affected_packages"]) == 1
        assert result["affected_packages"][0]["name"] == "xz"

    @patch("manus_agent.tools.generate_vuln_report.requests.post")
    def test_no_vulns(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"vulns": []}
        mock_post.return_value = mock_resp
        result = _fetch_osv("CVE-9999-99999")
        assert result["affected_packages"] == []

    @patch("manus_agent.tools.generate_vuln_report.requests.post")
    def test_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp
        result = _fetch_osv("CVE-2024-3094")
        assert "error" in result

    @patch("manus_agent.tools.generate_vuln_report.requests.post")
    def test_exception(self, mock_post):
        import requests as _req

        mock_post.side_effect = _req.Timeout("timeout")
        result = _fetch_osv("CVE-2024-3094")
        assert "error" in result

    @patch("manus_agent.tools.generate_vuln_report.requests.post")
    def test_multiple_packages(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "vulns": [
                {
                    "affected": [
                        {
                            "package": {"ecosystem": "PyPI", "name": "pkg1"},
                            "ranges": [{"events": [{"fixed": "1.0.1"}]}],
                        },
                        {
                            "package": {"ecosystem": "npm", "name": "pkg2"},
                            "ranges": [{"events": [{"fixed": "2.0.0"}]}],
                        },
                    ]
                }
            ]
        }
        mock_post.return_value = mock_resp
        result = _fetch_osv("CVE-2024-3094")
        assert len(result["affected_packages"]) == 2

    @patch("manus_agent.tools.generate_vuln_report.requests.post")
    def test_no_fixed_version(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "vulns": [
                {
                    "affected": [
                        {
                            "package": {"ecosystem": "PyPI", "name": "pkg1"},
                            "ranges": [{"events": [{"introduced": "0"}]}],
                        }
                    ]
                }
            ]
        }
        mock_post.return_value = mock_resp
        result = _fetch_osv("CVE-2024-3094")
        assert result["affected_packages"][0]["fixed_versions"] == []


# ═══════════════════════════════════════════════════════════════════════════
# 4. Data extraction tests
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractCvss:
    """_extract_cvss from NVD records."""

    def test_cvss31(self, sample_nvd_record):
        result = _extract_cvss(sample_nvd_record)
        assert result["version"] == "3.1"
        assert result["score"] == 10.0
        assert result["severity"] == "CRITICAL"
        assert "CVSS:3.1" in result["vector"]

    def test_cvss30_fallback(self):
        nvd = {
            "metrics": {
                "cvssMetricV30": [
                    {
                        "cvssData": {
                            "version": "3.0",
                            "baseScore": 7.5,
                            "baseSeverity": "HIGH",
                            "vectorString": "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
                        }
                    }
                ]
            }
        }
        result = _extract_cvss(nvd)
        assert result["version"] == "3.0"
        assert result["score"] == 7.5

    def test_cvss2_fallback(self):
        nvd = {
            "metrics": {
                "cvssMetricV2": [
                    {
                        "baseSeverity": "MEDIUM",
                        "cvssData": {
                            "version": "2.0",
                            "baseScore": 5.0,
                            "vectorString": "AV:N/AC:L/Au:N/C:P/I:N/A:N",
                        },
                    }
                ]
            }
        }
        result = _extract_cvss(nvd)
        assert result["version"] == "2.0"
        assert result["score"] == 5.0

    def test_no_metrics(self):
        result = _extract_cvss({"metrics": {}})
        assert result["version"] == "N/A"
        assert result["score"] == 0.0

    def test_empty_record(self):
        result = _extract_cvss({})
        assert result["severity"] == "UNKNOWN"


class TestExtractCwe:
    """_extract_cwe from NVD records."""

    def test_extracts_cwes(self, sample_nvd_record):
        cwes = _extract_cwe(sample_nvd_record)
        assert cwes == ["CWE-506"]

    def test_multiple_cwes(self):
        nvd = {
            "weaknesses": [
                {"description": [{"value": "CWE-79"}, {"value": "CWE-89"}]},
                {"description": [{"value": "CWE-200"}]},
            ]
        }
        assert _extract_cwe(nvd) == ["CWE-79", "CWE-89", "CWE-200"]

    def test_no_weaknesses(self):
        assert _extract_cwe({}) == []

    def test_non_cwe_values_excluded(self):
        nvd = {"weaknesses": [{"description": [{"value": "NVD-CWE-noinfo"}]}]}
        assert _extract_cwe(nvd) == []


class TestExtractDescription:
    """_extract_description from NVD records."""

    def test_english(self, sample_nvd_record):
        desc = _extract_description(sample_nvd_record)
        assert "xz-utils" in desc

    def test_no_english(self):
        nvd = {"descriptions": [{"lang": "es", "value": "Descripción en español"}]}
        assert _extract_description(nvd) == "Descripción en español"

    def test_empty(self):
        assert _extract_description({}) == ""
        assert _extract_description({"descriptions": []}) == ""


class TestExtractReferences:
    """_extract_references from NVD records."""

    def test_extracts_refs(self, sample_nvd_record):
        refs = _extract_references(sample_nvd_record)
        assert len(refs) == 2
        assert refs[0]["url"].startswith("https://")

    def test_caps_at_10(self):
        nvd = {"references": [{"url": f"https://example.com/{i}", "source": "test"} for i in range(20)]}
        refs = _extract_references(nvd)
        assert len(refs) == 10

    def test_empty(self):
        assert _extract_references({}) == []


class TestExtractPublished:
    """_extract_published from NVD records."""

    def test_published(self, sample_nvd_record):
        assert "2024-03-29" in _extract_published(sample_nvd_record)

    def test_empty(self):
        assert _extract_published({}) == ""


class TestExtractAffectedProducts:
    """_extract_affected_products from NVD configurations."""

    def test_extracts_vulnerable(self, sample_nvd_record):
        products = _extract_affected_products(sample_nvd_record)
        assert "tukaani:xz" in products

    def test_excludes_non_vulnerable(self, sample_nvd_record):
        products = _extract_affected_products(sample_nvd_record)
        # Deduped, so only one tukaani:xz
        assert products.count("tukaani:xz") == 1

    def test_caps_at_10(self):
        nodes = [
            {"cpeMatch": [{"vulnerable": True, "criteria": f"cpe:2.3:a:vendor{i}:product{i}:1.0:*:*:*:*:*:*:*"}]}
            for i in range(20)
        ]
        nvd = {"configurations": [{"nodes": nodes}]}
        products = _extract_affected_products(nvd)
        assert len(products) <= 10

    def test_empty(self):
        assert _extract_affected_products({}) == []


# ═══════════════════════════════════════════════════════════════════════════
# 5. Severity sorting tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSeveritySort:
    """_severity_sort_key ordering."""

    def test_kev_first(self):
        kev = {"cvss": {"severity": "MEDIUM"}, "in_cisa_kev": True, "epss_score": 0.1}
        no_kev = {"cvss": {"severity": "CRITICAL"}, "in_cisa_kev": False, "epss_score": 0.9}
        assert _severity_sort_key(kev) < _severity_sort_key(no_kev)

    def test_severity_order(self):
        critical = {"cvss": {"severity": "CRITICAL"}, "epss_score": 0.5}
        high = {"cvss": {"severity": "HIGH"}, "epss_score": 0.5}
        medium = {"cvss": {"severity": "MEDIUM"}, "epss_score": 0.5}
        low = {"cvss": {"severity": "LOW"}, "epss_score": 0.5}
        assert _severity_sort_key(critical) < _severity_sort_key(high)
        assert _severity_sort_key(high) < _severity_sort_key(medium)
        assert _severity_sort_key(medium) < _severity_sort_key(low)

    def test_epss_tiebreaker(self):
        high_epss = {"cvss": {"severity": "HIGH"}, "epss_score": 0.9}
        low_epss = {"cvss": {"severity": "HIGH"}, "epss_score": 0.1}
        assert _severity_sort_key(high_epss) < _severity_sort_key(low_epss)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Statistics tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeStatistics:
    """_compute_statistics aggregate calculations."""

    def test_empty(self):
        stats = _compute_statistics([])
        assert stats["total_cves"] == 0
        assert stats["kev_count"] == 0
        assert stats["mean_epss"] == 0.0

    def test_single_finding(self, sample_finding):
        stats = _compute_statistics([sample_finding])
        assert stats["total_cves"] == 1
        assert stats["kev_count"] == 1
        assert stats["max_cvss"] == 10.0
        assert stats["severity_distribution"]["CRITICAL"] == 1

    def test_multiple_findings(self):
        findings = [
            {"cvss": {"severity": "CRITICAL", "score": 10.0}, "in_cisa_kev": True, "epss_score": 0.9},
            {"cvss": {"severity": "HIGH", "score": 7.5}, "in_cisa_kev": False, "epss_score": 0.1},
            {"cvss": {"severity": "HIGH", "score": 8.0}, "in_cisa_kev": True, "epss_score": 0.5},
        ]
        stats = _compute_statistics(findings)
        assert stats["total_cves"] == 3
        assert stats["kev_count"] == 2
        assert stats["max_cvss"] == 10.0
        assert stats["severity_distribution"]["CRITICAL"] == 1
        assert stats["severity_distribution"]["HIGH"] == 2
        assert abs(stats["mean_epss"] - 0.5) < 0.001

    def test_mean_epss_rounding(self):
        findings = [
            {"cvss": {"severity": "LOW", "score": 2.0}, "in_cisa_kev": False, "epss_score": 0.33333},
            {"cvss": {"severity": "LOW", "score": 2.0}, "in_cisa_kev": False, "epss_score": 0.33333},
            {"cvss": {"severity": "LOW", "score": 2.0}, "in_cisa_kev": False, "epss_score": 0.33334},
        ]
        stats = _compute_statistics(findings)
        assert stats["mean_epss"] == round(1.0 / 3.0, 5)


# ═══════════════════════════════════════════════════════════════════════════
# 7. KEV catalog pre-fetch tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPrefetchKevCatalog:
    """_prefetch_kev_catalog."""

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_success(self, mock_get, sample_kev_catalog):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": sample_kev_catalog}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        catalog = _prefetch_kev_catalog()
        assert len(catalog) == 2

    @patch("manus_agent.tools.generate_vuln_report._get_with_retry")
    def test_failure_returns_none(self, mock_get):
        import requests as _req

        mock_get.side_effect = _req.ConnectionError("fail")
        assert _prefetch_kev_catalog() is None


# ═══════════════════════════════════════════════════════════════════════════
# 8. _gather_cve_data integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGatherCveData:
    """_gather_cve_data end-to-end per-CVE data collection."""

    @patch("manus_agent.tools.generate_vuln_report._fetch_osv")
    @patch("manus_agent.tools.generate_vuln_report._fetch_vulncheck_kev")
    @patch("manus_agent.tools.generate_vuln_report._fetch_kev")
    @patch("manus_agent.tools.generate_vuln_report._fetch_epss")
    @patch("manus_agent.tools.generate_vuln_report._fetch_nvd")
    def test_all_sources_succeed(self, mock_nvd, mock_epss, mock_kev, mock_vc, mock_osv, sample_nvd_record):
        mock_nvd.return_value = sample_nvd_record
        mock_epss.return_value = {"epss": "0.97565", "percentile": "0.99978"}
        mock_kev.return_value = {"in_kev": True, "date_added": "2024-03-30"}
        mock_vc.return_value = {"available": True, "in_kev": True, "ransomware_use": True}
        mock_osv.return_value = {
            "affected_packages": [{"ecosystem": "PyPI", "name": "xz", "fixed_versions": ["5.6.2"]}]
        }

        result = _gather_cve_data("CVE-2024-3094")
        assert result["cve_id"] == "CVE-2024-3094"
        assert result["cvss"]["score"] == 10.0
        assert result["in_cisa_kev"] is True
        assert result["epss_score"] == 0.97565
        assert len(result["osv_packages"]) == 1
        assert result["data_errors"] == []

    @patch("manus_agent.tools.generate_vuln_report._fetch_osv")
    @patch("manus_agent.tools.generate_vuln_report._fetch_vulncheck_kev")
    @patch("manus_agent.tools.generate_vuln_report._fetch_kev")
    @patch("manus_agent.tools.generate_vuln_report._fetch_epss")
    @patch("manus_agent.tools.generate_vuln_report._fetch_nvd")
    def test_nvd_error_graceful(self, mock_nvd, mock_epss, mock_kev, mock_vc, mock_osv):
        mock_nvd.return_value = {"error": "NVD fetch failed"}
        mock_epss.return_value = {"epss": "0.5", "percentile": "0.5"}
        mock_kev.return_value = {"in_kev": False}
        mock_vc.return_value = {"available": False}
        mock_osv.return_value = {"affected_packages": []}

        result = _gather_cve_data("CVE-2024-3094")
        assert result["description"] == ""
        assert result["cvss"]["severity"] == "UNKNOWN"
        assert "nvd" in result["data_errors"]

    @patch("manus_agent.tools.generate_vuln_report._fetch_osv")
    @patch("manus_agent.tools.generate_vuln_report._fetch_vulncheck_kev")
    @patch("manus_agent.tools.generate_vuln_report._fetch_kev")
    @patch("manus_agent.tools.generate_vuln_report._fetch_epss")
    @patch("manus_agent.tools.generate_vuln_report._fetch_nvd")
    def test_all_sources_fail(self, mock_nvd, mock_epss, mock_kev, mock_vc, mock_osv):
        mock_nvd.return_value = {"error": "fail"}
        mock_epss.return_value = {"error": "fail"}
        mock_kev.return_value = {"in_kev": False, "error": "fail"}
        mock_vc.return_value = {"available": False}
        mock_osv.return_value = {"error": "fail"}

        result = _gather_cve_data("CVE-2024-3094")
        assert len(result["data_errors"]) >= 3

    @patch("manus_agent.tools.generate_vuln_report._fetch_osv")
    @patch("manus_agent.tools.generate_vuln_report._fetch_vulncheck_kev")
    @patch("manus_agent.tools.generate_vuln_report._fetch_kev")
    @patch("manus_agent.tools.generate_vuln_report._fetch_epss")
    @patch("manus_agent.tools.generate_vuln_report._fetch_nvd")
    def test_case_normalisation(self, mock_nvd, mock_epss, mock_kev, mock_vc, mock_osv):
        mock_nvd.return_value = {"error": "fail"}
        mock_epss.return_value = {"error": "fail"}
        mock_kev.return_value = {"in_kev": False}
        mock_vc.return_value = {"available": False}
        mock_osv.return_value = {"affected_packages": []}

        result = _gather_cve_data("cve-2024-3094")
        assert result["cve_id"] == "CVE-2024-3094"


# ═══════════════════════════════════════════════════════════════════════════
# 9. Markdown rendering tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderFinding:
    """_render_finding Markdown formatting."""

    def test_basic_structure(self, sample_finding):
        lines = _render_finding(sample_finding)
        text = "\n".join(lines)
        assert "### 🔴 CVE-2024-3094" in text
        assert "CVSS:" in text
        assert "EPSS:" in text

    def test_kev_banner(self, sample_finding):
        lines = _render_finding(sample_finding)
        text = "\n".join(lines)
        assert "ACTIVELY EXPLOITED" in text

    def test_ransomware_banner(self, sample_finding):
        lines = _render_finding(sample_finding)
        text = "\n".join(lines)
        assert "RANSOMWARE ASSOCIATED" in text

    def test_osv_packages(self, sample_finding):
        lines = _render_finding(sample_finding)
        text = "\n".join(lines)
        assert "[PyPI]" in text
        assert "`xz`" in text

    def test_data_errors_shown(self):
        finding = {
            "cve_id": "CVE-2024-3094",
            "description": "",
            "published": "",
            "cvss": {"score": 0.0, "severity": "UNKNOWN", "version": "N/A", "vector": ""},
            "cwes": [],
            "affected_products": [],
            "references": [],
            "epss_score": 0.0,
            "epss_percentile": 0.0,
            "in_cisa_kev": False,
            "kev_details": None,
            "vulncheck_kev": None,
            "osv_packages": [],
            "data_errors": ["nvd", "epss"],
        }
        lines = _render_finding(finding)
        text = "\n".join(lines)
        assert "Data unavailable from: nvd, epss" in text

    def test_no_kev_no_banner(self):
        finding = {
            "cve_id": "CVE-2024-0001",
            "description": "Test",
            "published": "2024-01-01",
            "cvss": {"score": 5.0, "severity": "MEDIUM", "version": "3.1", "vector": ""},
            "cwes": [],
            "affected_products": [],
            "references": [],
            "epss_score": 0.05,
            "epss_percentile": 0.5,
            "in_cisa_kev": False,
            "kev_details": None,
            "vulncheck_kev": None,
            "osv_packages": [],
            "data_errors": [],
        }
        lines = _render_finding(finding)
        text = "\n".join(lines)
        assert "ACTIVELY EXPLOITED" not in text

    def test_vulncheck_no_ransomware(self):
        finding = {
            "cve_id": "CVE-2024-0002",
            "description": "Test",
            "published": "2024-01-01",
            "cvss": {"score": 9.0, "severity": "CRITICAL", "version": "3.1", "vector": ""},
            "cwes": [],
            "affected_products": [],
            "references": [],
            "epss_score": 0.8,
            "epss_percentile": 0.99,
            "in_cisa_kev": False,
            "kev_details": None,
            "vulncheck_kev": {"available": True, "in_kev": True, "ransomware_use": False},
            "osv_packages": [],
            "data_errors": [],
        }
        lines = _render_finding(finding)
        text = "\n".join(lines)
        assert "confirmed exploitation" in text
        assert "RANSOMWARE" not in text

    def test_high_epss_label(self):
        finding = {
            "cve_id": "CVE-2024-0003",
            "description": "",
            "published": "",
            "cvss": {"score": 9.0, "severity": "CRITICAL", "version": "3.1", "vector": ""},
            "cwes": [],
            "affected_products": [],
            "references": [],
            "epss_score": 0.5,
            "epss_percentile": 0.99,
            "in_cisa_kev": False,
            "kev_details": None,
            "vulncheck_kev": None,
            "osv_packages": [],
            "data_errors": [],
        }
        lines = _render_finding(finding)
        text = "\n".join(lines)
        assert "high exploitation probability" in text

    def test_low_epss_label(self):
        finding = {
            "cve_id": "CVE-2024-0004",
            "description": "",
            "published": "",
            "cvss": {"score": 3.0, "severity": "LOW", "version": "3.1", "vector": ""},
            "cwes": [],
            "affected_products": [],
            "references": [],
            "epss_score": 0.001,
            "epss_percentile": 0.1,
            "in_cisa_kev": False,
            "kev_details": None,
            "vulncheck_kev": None,
            "osv_packages": [],
            "data_errors": [],
        }
        lines = _render_finding(finding)
        text = "\n".join(lines)
        assert "low exploitation probability" in text


class TestRenderMarkdown:
    """_render_markdown full report."""

    def test_full_report_structure(self, sample_finding):
        text = _render_markdown(
            "Test Report",
            "2024-12-01 12:00:00 UTC",
            [sample_finding],
            _compute_statistics([sample_finding]),
        )
        assert "# Test Report" in text
        assert "Executive Summary" in text
        assert "Risk-Ranked CVE Overview" in text
        assert "Detailed Findings" in text
        assert "Report Metadata" in text

    def test_summary_table_headers(self, sample_finding):
        text = _render_markdown(
            "Report",
            "2024-12-01 12:00:00 UTC",
            [sample_finding],
            _compute_statistics([sample_finding]),
        )
        assert "| CVE ID |" in text
        assert "| Severity |" in text

    def test_kev_alert(self, sample_finding):
        text = _render_markdown(
            "Report",
            "now",
            [sample_finding],
            _compute_statistics([sample_finding]),
        )
        assert "CISA Known Exploited Vulnerabilities catalog" in text

    def test_no_kev_no_alert(self):
        finding = {
            "cve_id": "CVE-2024-0001",
            "description": "Test",
            "cvss": {"score": 5.0, "severity": "MEDIUM"},
            "epss_score": 0.01,
            "in_cisa_kev": False,
        }
        stats = _compute_statistics([finding])
        text = _render_markdown("Report", "now", [finding], stats)
        assert "CISA Known Exploited Vulnerabilities catalog" not in text

    def test_severity_breakdown(self, sample_finding):
        text = _render_markdown(
            "Report",
            "now",
            [sample_finding],
            _compute_statistics([sample_finding]),
        )
        assert "CRITICAL" in text

    def test_data_sources_attribution(self, sample_finding):
        text = _render_markdown(
            "Report",
            "now",
            [sample_finding],
            _compute_statistics([sample_finding]),
        )
        assert "NVD" in text
        assert "EPSS" in text
        assert "CISA KEV" in text
        assert "OSV.dev" in text


# ═══════════════════════════════════════════════════════════════════════════
# 10. generate_report integration tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerateReport:
    """generate_report top-level function."""

    def test_empty_cve_ids(self):
        result = generate_report([])
        assert result.get("error")
        assert result["data"]["findings"] == []

    @patch("manus_agent.tools.generate_vuln_report._gather_cve_data")
    @patch("manus_agent.tools.generate_vuln_report._prefetch_kev_catalog")
    def test_markdown_output(self, mock_catalog, mock_gather, sample_finding):
        mock_catalog.return_value = []
        mock_gather.return_value = sample_finding
        result = generate_report(["CVE-2024-3094"], output_format="markdown")
        assert "# Vulnerability Intelligence Report" in result["report"]
        assert len(result["data"]["findings"]) == 1

    @patch("manus_agent.tools.generate_vuln_report._gather_cve_data")
    @patch("manus_agent.tools.generate_vuln_report._prefetch_kev_catalog")
    def test_json_output(self, mock_catalog, mock_gather, sample_finding):
        mock_catalog.return_value = []
        mock_gather.return_value = sample_finding
        result = generate_report(["CVE-2024-3094"], output_format="json")
        parsed = json.loads(result["report"])
        assert "findings" in parsed
        assert "statistics" in parsed

    @patch("manus_agent.tools.generate_vuln_report._gather_cve_data")
    @patch("manus_agent.tools.generate_vuln_report._prefetch_kev_catalog")
    def test_custom_title(self, mock_catalog, mock_gather, sample_finding):
        mock_catalog.return_value = []
        mock_gather.return_value = sample_finding
        result = generate_report(["CVE-2024-3094"], title="My Custom Report")
        assert "My Custom Report" in result["report"]

    @patch("manus_agent.tools.generate_vuln_report._gather_cve_data")
    @patch("manus_agent.tools.generate_vuln_report._prefetch_kev_catalog")
    def test_deduplicates_cve_ids(self, mock_catalog, mock_gather, sample_finding):
        mock_catalog.return_value = []
        mock_gather.return_value = sample_finding
        generate_report(["CVE-2024-3094", "cve-2024-3094", " CVE-2024-3094 "])
        assert mock_gather.call_count == 1

    @patch("manus_agent.tools.generate_vuln_report._gather_cve_data")
    @patch("manus_agent.tools.generate_vuln_report._prefetch_kev_catalog")
    def test_multiple_cves(self, mock_catalog, mock_gather):
        mock_catalog.return_value = []
        findings = [
            {
                "cve_id": "CVE-2024-3094",
                "cvss": {"severity": "CRITICAL", "score": 10.0},
                "epss_score": 0.9,
                "in_cisa_kev": True,
                "description": "Critical vuln",
                "data_errors": [],
            },
            {
                "cve_id": "CVE-2021-44228",
                "cvss": {"severity": "HIGH", "score": 7.5},
                "epss_score": 0.5,
                "in_cisa_kev": False,
                "description": "High vuln",
                "data_errors": [],
            },
        ]
        mock_gather.side_effect = findings
        result = generate_report(["CVE-2024-3094", "CVE-2021-44228"])
        assert len(result["data"]["findings"]) == 2

    @patch("manus_agent.tools.generate_vuln_report._gather_cve_data")
    @patch("manus_agent.tools.generate_vuln_report._prefetch_kev_catalog")
    def test_generated_at_present(self, mock_catalog, mock_gather, sample_finding):
        mock_catalog.return_value = []
        mock_gather.return_value = sample_finding
        result = generate_report(["CVE-2024-3094"])
        assert "UTC" in result["generated_at"]

    @patch("manus_agent.tools.generate_vuln_report._gather_cve_data")
    @patch("manus_agent.tools.generate_vuln_report._prefetch_kev_catalog")
    def test_findings_sorted_by_risk(self, mock_catalog, mock_gather):
        mock_catalog.return_value = []
        f_critical = {
            "cve_id": "CVE-A",
            "cvss": {"severity": "CRITICAL", "score": 10.0},
            "epss_score": 0.9,
            "in_cisa_kev": False,
            "description": "",
            "data_errors": [],
        }
        f_low_kev = {
            "cve_id": "CVE-B",
            "cvss": {"severity": "LOW", "score": 2.0},
            "epss_score": 0.01,
            "in_cisa_kev": True,
            "description": "",
            "data_errors": [],
        }
        mock_gather.side_effect = [f_critical, f_low_kev]
        result = generate_report(["CVE-A", "CVE-B"])
        # KEV CVE should be sorted first
        assert result["data"]["findings"][0]["cve_id"] == "CVE-B"


# ═══════════════════════════════════════════════════════════════════════════
# 11. Strands handler tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHandler:
    """Strands tool handler function."""

    @patch("manus_agent.tools.generate_vuln_report.generate_report")
    def test_success(self, mock_report):
        mock_report.return_value = {"report": "# Report", "data": {}}
        tool_use: _ToolUse = {
            "toolUseId": "test-123",
            "name": "generate_vuln_report",
            "input": {"cve_ids": ["CVE-2024-3094"]},
        }
        result = handler(tool_use)
        assert result["status"] == "success"
        assert result["content"][0]["text"] == "# Report"

    def test_empty_cve_ids(self):
        tool_use: _ToolUse = {
            "toolUseId": "test-123",
            "name": "generate_vuln_report",
            "input": {"cve_ids": []},
        }
        result = handler(tool_use)
        assert result["status"] == "error"

    def test_invalid_cve_id(self):
        tool_use: _ToolUse = {
            "toolUseId": "test-123",
            "name": "generate_vuln_report",
            "input": {"cve_ids": ["not-a-cve"]},
        }
        result = handler(tool_use)
        assert result["status"] == "error"
        assert "Invalid" in result["content"][0]["text"]

    def test_mixed_valid_invalid(self):
        tool_use: _ToolUse = {
            "toolUseId": "test-123",
            "name": "generate_vuln_report",
            "input": {"cve_ids": ["CVE-2024-3094", "badid"]},
        }
        result = handler(tool_use)
        assert result["status"] == "error"

    @patch("manus_agent.tools.generate_vuln_report.generate_report")
    def test_custom_title(self, mock_report):
        mock_report.return_value = {"report": "# My Title", "data": {}}
        tool_use: _ToolUse = {
            "toolUseId": "test-123",
            "name": "generate_vuln_report",
            "input": {"cve_ids": ["CVE-2024-3094"], "title": "My Title"},
        }
        result = handler(tool_use)
        assert result["status"] == "success"
        mock_report.assert_called_once_with(["CVE-2024-3094"], title="My Title", output_format="markdown")

    @patch("manus_agent.tools.generate_vuln_report.generate_report")
    def test_json_format(self, mock_report):
        mock_report.return_value = {"report": "{}", "data": {}}
        tool_use: _ToolUse = {
            "toolUseId": "test-123",
            "name": "generate_vuln_report",
            "input": {"cve_ids": ["CVE-2024-3094"], "output_format": "json"},
        }
        handler(tool_use)
        mock_report.assert_called_once_with(
            ["CVE-2024-3094"], title="Vulnerability Intelligence Report", output_format="json"
        )

    def test_preserves_tool_use_id(self):
        tool_use: _ToolUse = {
            "toolUseId": "abc-789",
            "name": "generate_vuln_report",
            "input": {"cve_ids": []},
        }
        result = handler(tool_use)
        assert result["toolUseId"] == "abc-789"


# ═══════════════════════════════════════════════════════════════════════════
# 12. CLI subcommand tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCliVulnReport:
    """manus-agent vuln-report CLI subcommand."""

    @patch("manus_agent.tools.generate_vuln_report.generate_report")
    def test_basic_invocation(self, mock_report, capsys):
        mock_report.return_value = {"report": "# Test Report", "data": {}}
        from manus_agent.cli import _run_vuln_report

        rc = _run_vuln_report(["CVE-2024-3094"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "# Test Report" in captured.out

    @patch("manus_agent.tools.generate_vuln_report.generate_report")
    def test_multiple_cves(self, mock_report, capsys):
        mock_report.return_value = {"report": "# Report", "data": {}}
        from manus_agent.cli import _run_vuln_report

        _run_vuln_report(["CVE-2024-3094", "CVE-2021-44228"])
        mock_report.assert_called_once()
        call_args = mock_report.call_args
        assert len(call_args[0][0]) == 2

    @patch("manus_agent.tools.generate_vuln_report.generate_report")
    def test_json_output(self, mock_report, capsys):
        mock_report.return_value = {"report": '{"findings": []}', "data": {}}
        from manus_agent.cli import _run_vuln_report

        rc = _run_vuln_report(["CVE-2024-3094", "--output", "json"])
        assert rc == 0
        mock_report.assert_called_once_with(
            ["CVE-2024-3094"], title="Vulnerability Intelligence Report", output_format="json"
        )

    @patch("manus_agent.tools.generate_vuln_report.generate_report")
    def test_custom_title(self, mock_report, capsys):
        mock_report.return_value = {"report": "# Custom", "data": {}}
        from manus_agent.cli import _run_vuln_report

        _run_vuln_report(["CVE-2024-3094", "--title", "My Security Brief"])
        mock_report.assert_called_once_with(["CVE-2024-3094"], title="My Security Brief", output_format="markdown")

    @patch("manus_agent.tools.generate_vuln_report.generate_report")
    def test_out_file(self, mock_report, tmp_path, capsys):
        mock_report.return_value = {"report": "# File Report", "data": {}}
        from manus_agent.cli import _run_vuln_report

        out = str(tmp_path / "report.md")
        rc = _run_vuln_report(["CVE-2024-3094", "--out-file", out])
        assert rc == 0
        content = (tmp_path / "report.md").read_text()
        assert "# File Report" in content
        captured = capsys.readouterr()
        assert "written to" in captured.out

    def test_invalid_cve_exits(self):
        from manus_agent.cli import _run_vuln_report

        with pytest.raises(SystemExit):
            _run_vuln_report(["not-a-cve"])

    def test_no_args_exits(self):
        from manus_agent.cli import _run_vuln_report

        with pytest.raises(SystemExit):
            _run_vuln_report([])

    def test_subcommand_in_set(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "vuln-report" in _SUBCOMMANDS

    @patch("manus_agent.tools.generate_vuln_report.generate_report")
    def test_text_default_format(self, mock_report, capsys):
        mock_report.return_value = {"report": "# Report", "data": {}}
        from manus_agent.cli import _run_vuln_report

        _run_vuln_report(["CVE-2024-3094"])
        mock_report.assert_called_once_with(
            ["CVE-2024-3094"], title="Vulnerability Intelligence Report", output_format="markdown"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 13. Edge case tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Boundary and edge cases."""

    def test_description_truncation_in_table(self, sample_finding):
        long_desc = "A" * 200
        sample_finding["description"] = long_desc
        stats = _compute_statistics([sample_finding])
        text = _render_markdown("Report", "now", [sample_finding], stats)
        # Table row should truncate to ~80 chars + "…"
        assert "…" in text

    def test_empty_findings_report(self):
        text = _render_markdown("Empty Report", "now", [], _compute_statistics([]))
        assert "# Empty Report" in text
        assert "CVEs analysed:** 0" in text

    @patch("manus_agent.tools.generate_vuln_report._gather_cve_data")
    @patch("manus_agent.tools.generate_vuln_report._prefetch_kev_catalog")
    def test_whitespace_cve_ids(self, mock_catalog, mock_gather, sample_finding):
        mock_catalog.return_value = []
        mock_gather.return_value = sample_finding
        result = generate_report(["  CVE-2024-3094  "])
        assert len(result["data"]["findings"]) == 1

    @patch("manus_agent.tools.generate_vuln_report._gather_cve_data")
    @patch("manus_agent.tools.generate_vuln_report._prefetch_kev_catalog")
    def test_empty_string_in_cve_ids(self, mock_catalog, mock_gather):
        mock_catalog.return_value = []
        result = generate_report(["", " ", "  "])
        assert len(result["data"]["findings"]) == 0

    def test_finding_with_many_products(self):
        finding = {
            "cve_id": "CVE-2024-0001",
            "description": "Test",
            "published": "2024-01-01",
            "cvss": {"score": 5.0, "severity": "MEDIUM", "version": "3.1", "vector": ""},
            "cwes": [],
            "affected_products": [f"vendor{i}:product{i}" for i in range(20)],
            "references": [],
            "epss_score": 0.01,
            "epss_percentile": 0.1,
            "in_cisa_kev": False,
            "kev_details": None,
            "vulncheck_kev": None,
            "osv_packages": [],
            "data_errors": [],
        }
        lines = _render_finding(finding)
        text = "\n".join(lines)
        assert "… and" in text

    def test_finding_with_many_osv_packages(self):
        finding = {
            "cve_id": "CVE-2024-0001",
            "description": "Test",
            "published": "2024-01-01",
            "cvss": {"score": 5.0, "severity": "MEDIUM", "version": "3.1", "vector": ""},
            "cwes": [],
            "affected_products": [],
            "references": [],
            "epss_score": 0.01,
            "epss_percentile": 0.1,
            "in_cisa_kev": False,
            "kev_details": None,
            "vulncheck_kev": None,
            "osv_packages": [{"ecosystem": "PyPI", "name": f"pkg{i}", "fixed_versions": []} for i in range(10)],
            "data_errors": [],
        }
        lines = _render_finding(finding)
        text = "\n".join(lines)
        assert "… and" in text

    def test_finding_with_many_references(self):
        finding = {
            "cve_id": "CVE-2024-0001",
            "description": "Test",
            "published": "2024-01-01",
            "cvss": {"score": 5.0, "severity": "MEDIUM", "version": "3.1", "vector": ""},
            "cwes": [],
            "affected_products": [],
            "references": [{"url": f"https://example.com/{i}", "source": "test"} for i in range(10)],
            "epss_score": 0.01,
            "epss_percentile": 0.1,
            "in_cisa_kev": False,
            "kev_details": None,
            "vulncheck_kev": None,
            "osv_packages": [],
            "data_errors": [],
        }
        lines = _render_finding(finding)
        text = "\n".join(lines)
        assert "… and" in text

    def test_moderate_epss_label(self):
        finding = {
            "cve_id": "CVE-2024-0001",
            "description": "",
            "published": "",
            "cvss": {"score": 5.0, "severity": "MEDIUM", "version": "3.1", "vector": ""},
            "cwes": [],
            "affected_products": [],
            "references": [],
            "epss_score": 0.05,
            "epss_percentile": 0.7,
            "in_cisa_kev": False,
            "kev_details": None,
            "vulncheck_kev": None,
            "osv_packages": [],
            "data_errors": [],
        }
        lines = _render_finding(finding)
        text = "\n".join(lines)
        assert "moderate exploitation probability" in text
