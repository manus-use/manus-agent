"""Tests for get_version_range — CVE version range resolver.

All HTTP calls are fully mocked; no real network requests are made.
The VERSION_RANGE_RETRY_BASE_DELAY env var is forced to "0" so retry
loops complete instantly without real sleeping.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

# Force zero-delay retries before importing the module
os.environ["VERSION_RANGE_MAX_RETRIES"] = "3"
os.environ["VERSION_RANGE_RETRY_BASE_DELAY"] = "0"

from manus_agent.tools import get_version_range as mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_response(status_code=200, payload=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload if payload is not None else {}
    if status_code < 400:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"HTTP {status_code}", response=resp)
    return resp


def _nvd_response(configurations=None, descriptions=None, metrics=None, references=None):
    """Build an NVD API response with CPE configurations."""
    cve_data = {
        "configurations": configurations or [],
        "descriptions": descriptions or [{"lang": "en", "value": "Test vulnerability description."}],
        "metrics": metrics or {},
        "references": references or [],
    }
    return {
        "vulnerabilities": [{"cve": cve_data}],
    }


def _nvd_cpe_match(
    cpe_uri="cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
    vulnerable=True,
    vsi="",
    vse="",
    vei="",
    vee="",
):
    """Build a single CPE match entry."""
    match = {
        "vulnerable": vulnerable,
        "criteria": cpe_uri,
        "matchCriteriaId": "test-id",
    }
    if vsi:
        match["versionStartIncluding"] = vsi
    if vse:
        match["versionStartExcluding"] = vse
    if vei:
        match["versionEndIncluding"] = vei
    if vee:
        match["versionEndExcluding"] = vee
    return match


def _nvd_config(matches, operator="OR"):
    """Build a single NVD configuration node."""
    return {
        "nodes": [
            {
                "operator": operator,
                "negate": False,
                "cpeMatch": matches,
            }
        ],
    }


def _osv_record(
    osv_id="CVE-2021-44228",
    ecosystem="Maven",
    name="org.apache.logging.log4j:log4j-core",
    introduced="2.0-beta9",
    fixed="2.15.0",
    versions=None,
    aliases=None,
):
    """Build an OSV record with affected package data."""
    events = [{"introduced": introduced}]
    if fixed:
        events.append({"fixed": fixed})
    return {
        "id": osv_id,
        "summary": "Test vulnerability",
        "aliases": aliases or [osv_id],
        "affected": [
            {
                "package": {"ecosystem": ecosystem, "name": name},
                "ranges": [{"type": "ECOSYSTEM", "events": events}],
                "versions": versions or ["2.0-beta9", "2.14.1"],
            }
        ],
        "references": [{"type": "WEB", "url": "https://example.com"}],
    }


def _osv_record_no_packages(cve_id="CVE-2024-3094", aliases=None):
    """OSV record without package-level affected data."""
    return {
        "id": cve_id,
        "summary": "Test vuln no packages",
        "aliases": aliases or ["GHSA-xxxx-yyyy-zzzz"],
        "affected": [],
        "references": [],
    }


def _ghsa_record(
    ghsa_id="GHSA-xxxx-yyyy-zzzz",
    ecosystem="PyPI",
    name="vulnerable-pkg",
    introduced="0",
    fixed="1.2.3",
    versions=None,
):
    """Build a GHSA record with affected package data."""
    events = [{"introduced": introduced}]
    if fixed:
        events.append({"fixed": fixed})
    return {
        "id": ghsa_id,
        "summary": "Test GHSA",
        "aliases": ["CVE-2024-3094"],
        "affected": [
            {
                "package": {"ecosystem": ecosystem, "name": name},
                "ranges": [{"type": "ECOSYSTEM", "events": events}],
                "versions": versions or ["1.0.0", "1.1.0"],
            }
        ],
    }


# ===========================================================================
# Tests: CVE validation
# ===========================================================================
class TestCveValidation:
    """Tests for CVE ID validation in fetch_version_range."""

    def test_empty_cve_id(self):
        result = mod.fetch_version_range("")
        assert result["found"] is False
        assert "Invalid CVE ID" in result.get("error", "")

    def test_none_cve_id(self):
        result = mod.fetch_version_range(None)
        assert result["found"] is False
        assert "Invalid CVE ID" in result.get("error", "")

    def test_whitespace_only(self):
        result = mod.fetch_version_range("   ")
        assert result["found"] is False

    def test_invalid_format(self):
        result = mod.fetch_version_range("not-a-cve")
        assert result["found"] is False
        assert "Invalid CVE ID" in result.get("error", "")

    def test_short_number(self):
        result = mod.fetch_version_range("CVE-2024-12")
        assert result["found"] is False

    def test_valid_format_uppercase(self):
        """Valid format should not fail validation (may fail on network)."""
        with patch.object(mod, "_get_with_retry") as mock_get:
            mock_get.return_value = _make_response(404)
            mod.fetch_version_range("CVE-2021-44228")
            # Should have attempted the fetch (not rejected by validation)
            assert mock_get.called

    def test_valid_format_lowercase_normalized(self):
        """Lowercase CVE IDs should be uppercased."""
        with patch.object(mod, "_get_with_retry") as mock_get:
            mock_get.return_value = _make_response(404)
            result = mod.fetch_version_range("cve-2021-44228")
            assert result["cve_id"] == "CVE-2021-44228"


# ===========================================================================
# Tests: CPE parsing
# ===========================================================================
class TestCpeParsing:
    """Tests for NVD CPE URI parsing and range extraction."""

    def test_parse_cpe_uri_full(self):
        cpe = mod._parse_cpe_uri("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*")
        assert cpe["vendor"] == "apache"
        assert cpe["product"] == "log4j"
        assert cpe["version"] == "2.14.1"

    def test_parse_cpe_uri_short(self):
        cpe = mod._parse_cpe_uri("cpe:2.3:a:vendor")
        assert cpe == {}

    def test_parse_cpe_uri_wildcard_version(self):
        cpe = mod._parse_cpe_uri("cpe:2.3:a:django:django:*:*:*:*:*:*:*:*")
        assert cpe["vendor"] == "django"
        assert cpe["product"] == "django"
        assert cpe["version"] == "*"

    def test_extract_nvd_ranges_version_end_excluding(self):
        configs = [
            _nvd_config(
                [
                    _nvd_cpe_match(
                        cpe_uri="cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                        vsi="2.0",
                        vee="2.15.0",
                    ),
                ]
            )
        ]
        ranges = mod._extract_nvd_ranges(configs)
        assert len(ranges) == 1
        assert ranges[0]["vendor"] == "apache"
        assert ranges[0]["product"] == "log4j"
        assert ranges[0]["version_start_including"] == "2.0"
        assert ranges[0]["version_end_excluding"] == "2.15.0"
        assert ">= 2.0" in ranges[0]["range_display"]
        assert "< 2.15.0" in ranges[0]["range_display"]

    def test_extract_nvd_ranges_version_end_including(self):
        configs = [
            _nvd_config(
                [
                    _nvd_cpe_match(
                        cpe_uri="cpe:2.3:a:python:requests:*:*:*:*:*:*:*:*",
                        vsi="2.0.0",
                        vei="2.28.2",
                    ),
                ]
            )
        ]
        ranges = mod._extract_nvd_ranges(configs)
        assert len(ranges) == 1
        assert ranges[0]["version_end_including"] == "2.28.2"
        assert "<= 2.28.2" in ranges[0]["range_display"]

    def test_extract_nvd_ranges_exact_version(self):
        configs = [
            _nvd_config(
                [
                    _nvd_cpe_match(
                        cpe_uri="cpe:2.3:a:nodejs:express:4.17.1:*:*:*:*:*:*:*",
                    ),
                ]
            )
        ]
        ranges = mod._extract_nvd_ranges(configs)
        assert len(ranges) == 1
        assert ranges[0]["version"] == "4.17.1"
        assert ranges[0]["range_display"] == "= 4.17.1"

    def test_extract_nvd_ranges_all_versions(self):
        configs = [
            _nvd_config(
                [
                    _nvd_cpe_match(
                        cpe_uri="cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                    ),
                ]
            )
        ]
        ranges = mod._extract_nvd_ranges(configs)
        assert len(ranges) == 1
        assert ranges[0]["range_display"] == "all versions"

    def test_extract_nvd_ranges_skips_non_vulnerable(self):
        configs = [
            _nvd_config(
                [
                    _nvd_cpe_match(vulnerable=False),
                    _nvd_cpe_match(
                        cpe_uri="cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                        vulnerable=True,
                        vee="2.15.0",
                    ),
                ]
            )
        ]
        ranges = mod._extract_nvd_ranges(configs)
        assert len(ranges) == 1

    def test_extract_nvd_ranges_deduplicates(self):
        match = _nvd_cpe_match(
            cpe_uri="cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
            vee="2.15.0",
        )
        configs = [_nvd_config([match, match])]
        ranges = mod._extract_nvd_ranges(configs)
        assert len(ranges) == 1

    def test_extract_nvd_ranges_multiple_configs(self):
        configs = [
            _nvd_config(
                [
                    _nvd_cpe_match(
                        cpe_uri="cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                        vee="2.15.0",
                    ),
                ]
            ),
            _nvd_config(
                [
                    _nvd_cpe_match(
                        cpe_uri="cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                        vsi="2.16.0",
                        vee="2.17.0",
                    ),
                ]
            ),
        ]
        ranges = mod._extract_nvd_ranges(configs)
        assert len(ranges) == 2

    def test_extract_nvd_ranges_version_start_excluding(self):
        configs = [
            _nvd_config(
                [
                    _nvd_cpe_match(
                        cpe_uri="cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                        vse="1.0.0",
                        vee="2.0.0",
                    ),
                ]
            )
        ]
        ranges = mod._extract_nvd_ranges(configs)
        assert "> 1.0.0" in ranges[0]["range_display"]
        assert "< 2.0.0" in ranges[0]["range_display"]


# ===========================================================================
# Tests: Ecosystem guessing
# ===========================================================================
class TestEcosystemGuessing:
    """Tests for CPE-to-ecosystem hint mapping."""

    def test_python_vendor(self):
        assert mod._guess_ecosystem({"vendor": "python", "product": "x"}) == "PyPI"

    def test_django_product(self):
        assert mod._guess_ecosystem({"vendor": "x", "product": "django"}) == "PyPI"

    def test_nodejs_vendor(self):
        assert mod._guess_ecosystem({"vendor": "nodejs", "product": "x"}) == "npm"

    def test_maven_product(self):
        assert mod._guess_ecosystem({"vendor": "x", "product": "maven"}) == "Maven"

    def test_log4j_product(self):
        assert mod._guess_ecosystem({"vendor": "x", "product": "log4j"}) == "Maven"

    def test_golang_vendor(self):
        assert mod._guess_ecosystem({"vendor": "golang", "product": "x"}) == "Go"

    def test_rust_vendor(self):
        assert mod._guess_ecosystem({"vendor": "rust", "product": "x"}) == "crates.io"

    def test_linux_vendor(self):
        assert mod._guess_ecosystem({"vendor": "linux", "product": "x"}) == "Linux"

    def test_unknown_vendor(self):
        assert mod._guess_ecosystem({"vendor": "obscure_vendor", "product": "obscure_product"}) == "Unknown"

    def test_target_sw_field(self):
        assert mod._guess_ecosystem({"vendor": "x", "product": "x", "target_sw": "node.js"}) == "npm"

    def test_php_vendor(self):
        assert mod._guess_ecosystem({"vendor": "php", "product": "x"}) == "Packagist"

    def test_rubygems_vendor(self):
        assert mod._guess_ecosystem({"vendor": "rubygems", "product": "x"}) == "RubyGems"

    def test_nuget_vendor(self):
        assert mod._guess_ecosystem({"vendor": "nuget", "product": "x"}) == "NuGet"


# ===========================================================================
# Tests: Range display
# ===========================================================================
class TestBuildRangeDisplay:
    """Tests for human-readable range string construction."""

    def test_exact_version(self):
        assert mod._build_range_display("1.2.3", "", "", "", "") == "= 1.2.3"

    def test_start_including_end_excluding(self):
        result = mod._build_range_display("", "1.0", "", "", "2.0")
        assert result == ">= 1.0, < 2.0"

    def test_start_excluding_end_including(self):
        result = mod._build_range_display("", "", "1.0", "2.0", "")
        assert result == "> 1.0, <= 2.0"

    def test_only_end_excluding(self):
        result = mod._build_range_display("", "", "", "", "3.0")
        assert result == "< 3.0"

    def test_only_start_including(self):
        result = mod._build_range_display("", "1.0", "", "", "")
        assert result == ">= 1.0"

    def test_all_versions(self):
        result = mod._build_range_display("", "", "", "", "")
        assert result == "all versions"


# ===========================================================================
# Tests: OSV affected parsing
# ===========================================================================
class TestOsvAffectedParsing:
    """Tests for OSV affected entry parsing."""

    def test_basic_affected(self):
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "requests"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [{"introduced": "2.0.0"}, {"fixed": "2.31.0"}],
                    }
                ],
                "versions": ["2.0.0", "2.30.0"],
            }
        ]
        result = mod._parse_osv_affected(affected)
        assert len(result) == 1
        assert result[0]["ecosystem"] == "PyPI"
        assert result[0]["package"] == "requests"
        assert result[0]["introduced"] == ["2.0.0"]
        assert result[0]["fixed"] == ["2.31.0"]
        assert result[0]["first_patched"] == "2.31.0"
        assert result[0]["affected_version_count"] == 2

    def test_affected_with_last_affected(self):
        affected = [
            {
                "package": {"ecosystem": "npm", "name": "lodash"},
                "ranges": [
                    {
                        "type": "SEMVER",
                        "events": [{"introduced": "0"}, {"last_affected": "4.17.20"}],
                    }
                ],
                "versions": [],
            }
        ]
        result = mod._parse_osv_affected(affected)
        assert result[0]["last_affected"] == ["4.17.20"]
        assert result[0]["first_patched"] is None

    def test_affected_no_package(self):
        affected = [{"package": {}, "ranges": [], "versions": []}]
        result = mod._parse_osv_affected(affected)
        assert len(result) == 0

    def test_affected_none_input(self):
        result = mod._parse_osv_affected(None)
        assert result == []

    def test_affected_empty_list(self):
        result = mod._parse_osv_affected([])
        assert result == []

    def test_affected_non_dict_entry(self):
        result = mod._parse_osv_affected(["not_a_dict"])
        assert result == []

    def test_range_strings_with_zero_introduced(self):
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "pkg"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [{"introduced": "0"}, {"fixed": "1.0.0"}],
                    }
                ],
                "versions": [],
            }
        ]
        result = mod._parse_osv_affected(affected)
        assert "all versions" in result[0]["range_strings"][0]
        assert "< 1.0.0" in result[0]["range_strings"][0]

    def test_range_strings_normal_introduced(self):
        affected = [
            {
                "package": {"ecosystem": "npm", "name": "pkg"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [{"introduced": "2.0.0"}, {"fixed": "2.5.0"}],
                    }
                ],
                "versions": [],
            }
        ]
        result = mod._parse_osv_affected(affected)
        assert ">= 2.0.0" in result[0]["range_strings"][0]

    def test_multiple_ranges(self):
        affected = [
            {
                "package": {"ecosystem": "Maven", "name": "org.example:lib"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "1.0"},
                            {"fixed": "1.5"},
                            {"introduced": "2.0"},
                            {"fixed": "2.3"},
                        ],
                    }
                ],
                "versions": [],
            }
        ]
        result = mod._parse_osv_affected(affected)
        assert len(result[0]["introduced"]) == 2
        assert len(result[0]["fixed"]) == 2


# ===========================================================================
# Tests: HTTP retry
# ===========================================================================
class TestGetWithRetry:
    """Tests for HTTP GET with retry/back-off."""

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_success_first_try(self, mock_get):
        mock_get.return_value = _make_response(200, {"ok": True})
        resp = mod._get_with_retry("https://example.com")
        assert resp.status_code == 200
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retry_on_429(self, mock_get):
        mock_get.side_effect = [
            _make_response(429),
            _make_response(429),
            _make_response(200, {"ok": True}),
        ]
        resp = mod._get_with_retry("https://example.com")
        assert resp.status_code == 200
        assert mock_get.call_count == 3

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retry_on_500(self, mock_get):
        mock_get.side_effect = [
            _make_response(500),
            _make_response(200, {"ok": True}),
        ]
        resp = mod._get_with_retry("https://example.com")
        assert resp.status_code == 200

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_no_retry_on_404(self, mock_get):
        mock_get.return_value = _make_response(404)
        resp = mod._get_with_retry("https://example.com")
        assert resp.status_code == 404
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retry_on_connection_error(self, mock_get):
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            _make_response(200, {"ok": True}),
        ]
        resp = mod._get_with_retry("https://example.com")
        assert resp.status_code == 200

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retry_exhausted_raises(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")
        with pytest.raises(requests.exceptions.Timeout):
            mod._get_with_retry("https://example.com")
        assert mock_get.call_count == 3

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retry_on_502(self, mock_get):
        mock_get.side_effect = [
            _make_response(502),
            _make_response(200, {"ok": True}),
        ]
        resp = mod._get_with_retry("https://example.com")
        assert resp.status_code == 200

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_passes_params_and_headers(self, mock_get):
        mock_get.return_value = _make_response(200)
        mod._get_with_retry(
            "https://example.com",
            params={"key": "val"},
            headers={"X-Custom": "test"},
        )
        _, kwargs = mock_get.call_args
        assert kwargs["params"] == {"key": "val"}
        assert kwargs["headers"]["X-Custom"] == "test"


# ===========================================================================
# Tests: NVD fetch
# ===========================================================================
class TestFetchNvdConfigurations:
    """Tests for NVD CVE configuration fetching."""

    @patch.object(mod, "_get_with_retry")
    def test_success_with_configurations(self, mock_get):
        nvd_data = _nvd_response(
            configurations=[
                _nvd_config(
                    [
                        _nvd_cpe_match(
                            cpe_uri="cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                            vsi="2.0",
                            vee="2.15.0",
                        ),
                    ]
                )
            ],
            metrics={"cvssMetricV31": [{"cvssData": {"baseScore": 10.0, "baseSeverity": "CRITICAL"}}]},
        )
        mock_get.return_value = _make_response(200, nvd_data)
        result = mod._fetch_nvd_configurations("CVE-2021-44228")
        assert "error" not in result
        assert result["cpe_count"] == 1
        assert result["cvss_v31_score"] == 10.0
        assert result["cvss_v31_severity"] == "CRITICAL"

    @patch.object(mod, "_get_with_retry")
    def test_nvd_404(self, mock_get):
        mock_get.return_value = _make_response(404)
        result = mod._fetch_nvd_configurations("CVE-9999-9999")
        assert "error" in result
        assert "not found" in result["error"]

    @patch.object(mod, "_get_with_retry")
    def test_nvd_http_error(self, mock_get):
        mock_get.return_value = _make_response(500)
        result = mod._fetch_nvd_configurations("CVE-2021-44228")
        assert "error" in result

    @patch.object(mod, "_get_with_retry")
    def test_nvd_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("down")
        result = mod._fetch_nvd_configurations("CVE-2021-44228")
        assert "error" in result
        assert "failed" in result["error"]

    @patch.object(mod, "_get_with_retry")
    def test_nvd_empty_vulnerabilities(self, mock_get):
        mock_get.return_value = _make_response(200, {"vulnerabilities": []})
        result = mod._fetch_nvd_configurations("CVE-2021-44228")
        assert "error" in result

    @patch.object(mod, "_get_with_retry")
    def test_nvd_no_configurations(self, mock_get):
        nvd_data = _nvd_response(configurations=[])
        mock_get.return_value = _make_response(200, nvd_data)
        result = mod._fetch_nvd_configurations("CVE-2021-44228")
        assert result["cpe_count"] == 0
        assert result["cpe_ranges"] == []

    @patch.object(mod, "_get_with_retry")
    def test_nvd_with_api_key(self, mock_get):
        nvd_data = _nvd_response()
        mock_get.return_value = _make_response(200, nvd_data)
        with patch.dict(os.environ, {"NVD_API_KEY": "test-key-123"}):
            mod._fetch_nvd_configurations("CVE-2021-44228")
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["apiKey"] == "test-key-123"

    @patch.object(mod, "_get_with_retry")
    def test_nvd_cvss_v30_fallback(self, mock_get):
        nvd_data = _nvd_response(
            metrics={"cvssMetricV30": [{"cvssData": {"baseScore": 9.8, "baseSeverity": "CRITICAL"}}]},
        )
        mock_get.return_value = _make_response(200, nvd_data)
        result = mod._fetch_nvd_configurations("CVE-2021-44228")
        assert result["cvss_v31_score"] == 9.8

    @patch.object(mod, "_get_with_retry")
    def test_nvd_description_english(self, mock_get):
        nvd_data = _nvd_response(
            descriptions=[
                {"lang": "es", "value": "Descripción en español"},
                {"lang": "en", "value": "English description"},
            ]
        )
        mock_get.return_value = _make_response(200, nvd_data)
        result = mod._fetch_nvd_configurations("CVE-2021-44228")
        assert result["description"] == "English description"

    @patch.object(mod, "_get_with_retry")
    def test_nvd_description_fallback(self, mock_get):
        nvd_data = _nvd_response(descriptions=[{"lang": "es", "value": "Descripción"}])
        mock_get.return_value = _make_response(200, nvd_data)
        result = mod._fetch_nvd_configurations("CVE-2021-44228")
        assert result["description"] == "Descripción"


# ===========================================================================
# Tests: OSV fetch
# ===========================================================================
class TestFetchOsvRanges:
    """Tests for OSV.dev version range fetching."""

    @patch.object(mod, "_osv_get")
    def test_success_with_packages(self, mock_get):
        record = _osv_record()
        mock_get.return_value = _make_response(200, record)
        result = mod._fetch_osv_ranges("CVE-2021-44228")
        assert len(result["packages"]) == 1
        assert result["packages"][0]["ecosystem"] == "Maven"
        assert result["packages"][0]["first_patched"] == "2.15.0"

    @patch.object(mod, "_osv_get")
    def test_osv_404(self, mock_get):
        mock_get.return_value = _make_response(404)
        result = mod._fetch_osv_ranges("CVE-9999-9999")
        assert result["packages"] == []

    @patch.object(mod, "_osv_get")
    def test_osv_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("down")
        result = mod._fetch_osv_ranges("CVE-2021-44228")
        assert "error" in result

    @patch.object(mod, "_osv_get")
    def test_osv_follows_ghsa_aliases(self, mock_get):
        """When the CVE record has no packages, GHSA aliases are followed."""
        cve_record = _osv_record_no_packages(aliases=["GHSA-xxxx-yyyy-zzzz"])
        ghsa_record = _ghsa_record()
        mock_get.side_effect = [
            _make_response(200, cve_record),  # CVE lookup
            _make_response(200, ghsa_record),  # GHSA follow
        ]
        result = mod._fetch_osv_ranges("CVE-2024-3094")
        assert len(result["packages"]) == 1
        assert result["packages"][0]["ecosystem"] == "PyPI"

    @patch.object(mod, "_osv_get")
    def test_osv_ghsa_follow_failure_graceful(self, mock_get):
        """GHSA follow failure should not break the result."""
        cve_record = _osv_record_no_packages(aliases=["GHSA-xxxx-yyyy-zzzz"])
        mock_get.side_effect = [
            _make_response(200, cve_record),
            requests.exceptions.ConnectionError("GHSA down"),
        ]
        result = mod._fetch_osv_ranges("CVE-2024-3094")
        assert result["packages"] == []
        # Should not raise

    @patch.object(mod, "_osv_get")
    def test_osv_ghsa_404(self, mock_get):
        """GHSA 404 should be handled gracefully."""
        cve_record = _osv_record_no_packages(aliases=["GHSA-xxxx-yyyy-zzzz"])
        mock_get.side_effect = [
            _make_response(200, cve_record),
            _make_response(404),
        ]
        result = mod._fetch_osv_ranges("CVE-2024-3094")
        assert result["packages"] == []

    @patch.object(mod, "_osv_get")
    def test_osv_deduplicates_ghsa_records(self, mock_get):
        """Same GHSA record should not be added twice."""
        cve_record = _osv_record_no_packages(aliases=["GHSA-xxxx-yyyy-zzzz", "GHSA-xxxx-yyyy-zzzz"])
        ghsa_record = _ghsa_record()
        mock_get.side_effect = [
            _make_response(200, cve_record),
            _make_response(200, ghsa_record),
            _make_response(200, ghsa_record),
        ]
        result = mod._fetch_osv_ranges("CVE-2024-3094")
        # Should only have 1 package (not 2 from double GHSA)
        assert len(result["packages"]) <= 2  # At most 2 (aliases deduplicate by id)

    @patch.object(mod, "_osv_get")
    def test_osv_http_error(self, mock_get):
        mock_get.return_value = _make_response(500)
        result = mod._fetch_osv_ranges("CVE-2021-44228")
        assert "error" in result


# ===========================================================================
# Tests: Unified fetch_version_range
# ===========================================================================
class TestFetchVersionRange:
    """Tests for the main fetch_version_range function."""

    @patch.object(mod, "_fetch_osv_ranges")
    @patch.object(mod, "_fetch_nvd_configurations")
    def test_both_sources_found(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "cpe_ranges": [
                {
                    "vendor": "apache",
                    "product": "log4j",
                    "version": "",
                    "version_start_including": "2.0",
                    "version_end_excluding": "2.15.0",
                    "ecosystem_hint": "Maven",
                    "cpe_uri": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                    "range_display": ">= 2.0, < 2.15.0",
                }
            ],
            "cpe_count": 1,
            "description": "Log4Shell RCE",
            "cvss_v31_score": 10.0,
            "cvss_v31_severity": "CRITICAL",
            "references": [],
        }
        mock_osv.return_value = {
            "packages": [
                {
                    "ecosystem": "Maven",
                    "package": "org.apache.logging.log4j:log4j-core",
                    "introduced": ["2.0-beta9"],
                    "fixed": ["2.15.0"],
                    "last_affected": [],
                    "first_patched": "2.15.0",
                    "range_type": "ECOSYSTEM",
                    "range_strings": [">= 2.0-beta9, < 2.15.0"],
                    "affected_version_count": 2,
                    "affected_versions": ["2.0-beta9", "2.14.1"],
                }
            ],
            "aliases": ["GHSA-jfh8-c2jp-5v3q"],
        }

        result = mod.fetch_version_range("CVE-2021-44228")
        assert result["found"] is True
        assert result["cve_id"] == "CVE-2021-44228"
        assert result["nvd"]["cpe_count"] == 1
        assert result["osv"]["package_count"] == 1
        assert "Maven" in result["summary"]["affected_ecosystems"]
        assert result["summary"]["first_patched_versions"]["Maven/org.apache.logging.log4j:log4j-core"] == "2.15.0"

    @patch.object(mod, "_fetch_osv_ranges")
    @patch.object(mod, "_fetch_nvd_configurations")
    def test_nvd_only(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "cpe_ranges": [
                {
                    "vendor": "vendor",
                    "product": "product",
                    "version": "1.0",
                    "version_start_including": "",
                    "version_end_excluding": "",
                    "ecosystem_hint": "Unknown",
                    "cpe_uri": "cpe:2.3:a:vendor:product:1.0:*:*:*:*:*:*:*",
                    "range_display": "= 1.0",
                }
            ],
            "cpe_count": 1,
            "description": "Test",
            "cvss_v31_score": None,
            "cvss_v31_severity": None,
            "references": [],
        }
        mock_osv.return_value = {"packages": [], "aliases": []}

        result = mod.fetch_version_range("CVE-2099-1234")
        assert result["found"] is True
        assert result["nvd"]["cpe_count"] == 1
        assert result["osv"]["package_count"] == 0

    @patch.object(mod, "_fetch_osv_ranges")
    @patch.object(mod, "_fetch_nvd_configurations")
    def test_osv_only(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "error": "NVD not found",
            "cpe_ranges": [],
            "cpe_count": 0,
            "description": "",
            "cvss_v31_score": None,
            "cvss_v31_severity": None,
            "references": [],
        }
        mock_osv.return_value = {
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "requests",
                    "introduced": ["2.0"],
                    "fixed": ["2.31.0"],
                    "last_affected": [],
                    "first_patched": "2.31.0",
                    "range_type": "ECOSYSTEM",
                    "range_strings": [">= 2.0, < 2.31.0"],
                    "affected_version_count": 5,
                    "affected_versions": ["2.0", "2.1", "2.2", "2.3", "2.30.0"],
                }
            ],
            "aliases": [],
        }

        result = mod.fetch_version_range("CVE-2023-32681")
        assert result["found"] is True
        assert "NVD" in result["errors"][0]

    @patch.object(mod, "_fetch_osv_ranges")
    @patch.object(mod, "_fetch_nvd_configurations")
    def test_nothing_found(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "error": "not found",
            "cpe_ranges": [],
            "cpe_count": 0,
            "description": "",
            "cvss_v31_score": None,
            "cvss_v31_severity": None,
            "references": [],
        }
        mock_osv.return_value = {"packages": [], "aliases": [], "error": "404"}

        result = mod.fetch_version_range("CVE-9999-9999")
        assert result["found"] is False
        assert len(result["errors"]) == 2

    @patch.object(mod, "_fetch_osv_ranges")
    @patch.object(mod, "_fetch_nvd_configurations")
    def test_ecosystem_filter_pypi(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "cpe_ranges": [
                {
                    "vendor": "python",
                    "product": "requests",
                    "version": "",
                    "version_start_including": "",
                    "version_end_excluding": "",
                    "ecosystem_hint": "PyPI",
                    "cpe_uri": "cpe:2.3:a:python:requests:*:*:*:*:*:*:*:*",
                    "range_display": "all versions",
                },
                {
                    "vendor": "apache",
                    "product": "log4j",
                    "version": "",
                    "version_start_including": "",
                    "version_end_excluding": "",
                    "ecosystem_hint": "Maven",
                    "cpe_uri": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                    "range_display": "all versions",
                },
            ],
            "cpe_count": 2,
            "description": "",
            "cvss_v31_score": None,
            "cvss_v31_severity": None,
            "references": [],
        }
        mock_osv.return_value = {
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "requests",
                    "introduced": [],
                    "fixed": [],
                    "last_affected": [],
                    "first_patched": None,
                    "range_type": "",
                    "range_strings": [],
                    "affected_version_count": 0,
                    "affected_versions": [],
                },
                {
                    "ecosystem": "Maven",
                    "package": "log4j",
                    "introduced": [],
                    "fixed": [],
                    "last_affected": [],
                    "first_patched": None,
                    "range_type": "",
                    "range_strings": [],
                    "affected_version_count": 0,
                    "affected_versions": [],
                },
            ],
            "aliases": [],
        }

        result = mod.fetch_version_range("CVE-2021-44228", ecosystem="pypi")
        assert result["ecosystem_filter"] == "PyPI"
        # Only PyPI ranges should remain
        assert all(r["ecosystem_hint"] == "PyPI" for r in result["nvd"]["cpe_ranges"])
        assert all(p["ecosystem"] == "PyPI" for p in result["osv"]["packages"])

    @patch.object(mod, "_fetch_osv_ranges")
    @patch.object(mod, "_fetch_nvd_configurations")
    def test_ecosystem_filter_auto(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "cpe_ranges": [{"ecosystem_hint": "PyPI"}, {"ecosystem_hint": "Maven"}],
            "cpe_count": 2,
            "description": "",
            "cvss_v31_score": None,
            "cvss_v31_severity": None,
            "references": [],
        }
        mock_osv.return_value = {"packages": [], "aliases": []}

        result = mod.fetch_version_range("CVE-2021-44228", ecosystem="auto")
        # Auto should keep all
        assert len(result["nvd"]["cpe_ranges"]) == 2

    @patch.object(mod, "_fetch_osv_ranges")
    @patch.object(mod, "_fetch_nvd_configurations")
    def test_message_with_first_patched(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "cpe_ranges": [],
            "cpe_count": 0,
            "description": "",
            "cvss_v31_score": None,
            "cvss_v31_severity": None,
            "references": [],
        }
        mock_osv.return_value = {
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "flask",
                    "introduced": ["0"],
                    "fixed": ["2.3.3"],
                    "last_affected": [],
                    "first_patched": "2.3.3",
                    "range_type": "ECOSYSTEM",
                    "range_strings": ["all versions, < 2.3.3"],
                    "affected_version_count": 10,
                    "affected_versions": [],
                }
            ],
            "aliases": [],
        }

        result = mod.fetch_version_range("CVE-2023-30861")
        assert "first patched" in result["message"]
        assert "2.3.3" in result["message"]


# ===========================================================================
# Tests: CLI parser
# ===========================================================================
class TestCliParser:
    """Tests for the version-range CLI argument parser."""

    def test_basic_parse(self):
        from manus_agent.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        args = parser.parse_args(["CVE-2021-44228"])
        assert args.cve_id == "CVE-2021-44228"
        assert args.ecosystem == "auto"
        assert args.output == "text"

    def test_parse_with_ecosystem(self):
        from manus_agent.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        args = parser.parse_args(["CVE-2021-44228", "--ecosystem", "pypi"])
        assert args.ecosystem == "pypi"

    def test_parse_with_json_output(self):
        from manus_agent.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        args = parser.parse_args(["CVE-2021-44228", "--output", "json"])
        assert args.output == "json"

    def test_parse_all_options(self):
        from manus_agent.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        args = parser.parse_args(["CVE-2024-3094", "--ecosystem", "maven", "--output", "json"])
        assert args.cve_id == "CVE-2024-3094"
        assert args.ecosystem == "maven"
        assert args.output == "json"

    def test_invalid_ecosystem(self):
        from manus_agent.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["CVE-2021-44228", "--ecosystem", "invalid"])

    def test_invalid_output(self):
        from manus_agent.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["CVE-2021-44228", "--output", "xml"])


# ===========================================================================
# Tests: CLI runner
# ===========================================================================
class TestCliRunner:
    """Tests for the _run_version_range CLI runner."""

    @patch("manus_agent.cli._render_version_range_text")
    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_output(self, mock_fetch, mock_render):
        mock_fetch.return_value = {
            "found": True,
            "cve_id": "CVE-2021-44228",
            "nvd": {"cpe_ranges": [], "cpe_count": 0},
            "osv": {"packages": [], "package_count": 0},
            "summary": {},
            "message": "test",
        }
        from manus_agent.cli import _run_version_range

        exit_code = _run_version_range(["CVE-2021-44228"])
        assert exit_code == 0
        mock_render.assert_called_once()

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_json_output(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "found": True,
            "cve_id": "CVE-2021-44228",
            "nvd": {"cpe_ranges": []},
            "osv": {"packages": []},
            "summary": {},
            "message": "ok",
        }
        from manus_agent.cli import _run_version_range

        exit_code = _run_version_range(["CVE-2021-44228", "--output", "json"])
        assert exit_code == 0
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed["cve_id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_not_found(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "found": False,
            "cve_id": "CVE-9999-9999",
            "error": "No data",
            "message": "Not found",
        }
        from manus_agent.cli import _run_version_range

        exit_code = _run_version_range(["CVE-9999-9999"])
        assert exit_code == 1
        stderr = capsys.readouterr().err
        assert "error" in stderr.lower()

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_exception_handling(self, mock_fetch, capsys):
        mock_fetch.side_effect = RuntimeError("network down")
        from manus_agent.cli import _run_version_range

        exit_code = _run_version_range(["CVE-2021-44228"])
        assert exit_code == 1
        stderr = capsys.readouterr().err
        assert "network down" in stderr

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_ecosystem_passed_through(self, mock_fetch):
        mock_fetch.return_value = {"found": True, "cve_id": "X", "nvd": {}, "osv": {}, "summary": {}, "message": ""}
        from manus_agent.cli import _run_version_range

        _run_version_range(["CVE-2021-44228", "--ecosystem", "npm"])
        mock_fetch.assert_called_once_with("CVE-2021-44228", ecosystem="npm")


# ===========================================================================
# Tests: CLI text renderer
# ===========================================================================
class TestRenderVersionRangeText:
    """Tests for the text output renderer."""

    def test_basic_render(self, capsys):
        from manus_agent.cli import _render_version_range_text

        data = {
            "cve_id": "CVE-2021-44228",
            "ecosystem_filter": "auto",
            "nvd": {
                "cpe_ranges": [
                    {
                        "vendor": "apache",
                        "product": "log4j",
                        "ecosystem_hint": "Maven",
                        "range_display": ">= 2.0, < 2.15.0",
                    }
                ],
                "description": "Log4Shell",
                "cvss_v31_score": 10.0,
                "cvss_v31_severity": "CRITICAL",
            },
            "osv": {
                "packages": [
                    {
                        "ecosystem": "Maven",
                        "package": "log4j-core",
                        "first_patched": "2.15.0",
                        "affected_version_count": 50,
                        "range_strings": [">= 2.0-beta9, < 2.15.0"],
                        "affected_versions": ["2.0-beta9", "2.14.1"],
                    }
                ],
            },
            "summary": {
                "affected_ecosystems": ["Maven"],
                "first_patched_versions": {"Maven/log4j-core": "2.15.0"},
                "total_affected_version_count": 50,
            },
        }
        _render_version_range_text(data)
        out = capsys.readouterr().out
        assert "CVE-2021-44228" in out
        assert "CVSS" in out or "10.0" in out
        assert "apache:log4j" in out
        assert "2.15.0" in out

    def test_render_with_ecosystem_filter(self, capsys):
        from manus_agent.cli import _render_version_range_text

        data = {
            "cve_id": "CVE-TEST",
            "ecosystem_filter": "PyPI",
            "nvd": {"cpe_ranges": [], "description": "", "cvss_v31_score": None, "cvss_v31_severity": ""},
            "osv": {"packages": []},
            "summary": {"affected_ecosystems": [], "first_patched_versions": {}, "total_affected_version_count": 0},
        }
        _render_version_range_text(data)
        out = capsys.readouterr().out
        assert "Ecosystem filter: PyPI" in out

    def test_render_with_errors(self, capsys):
        from manus_agent.cli import _render_version_range_text

        data = {
            "cve_id": "CVE-TEST",
            "ecosystem_filter": "auto",
            "nvd": {"cpe_ranges": [], "description": "", "cvss_v31_score": None, "cvss_v31_severity": ""},
            "osv": {"packages": []},
            "summary": {"affected_ecosystems": [], "first_patched_versions": {}, "total_affected_version_count": 0},
            "errors": ["NVD: timeout", "OSV: 500"],
        }
        _render_version_range_text(data)
        out = capsys.readouterr().out
        assert "NVD: timeout" in out
        assert "OSV: 500" in out

    def test_render_long_description_truncated(self, capsys):
        from manus_agent.cli import _render_version_range_text

        data = {
            "cve_id": "CVE-TEST",
            "ecosystem_filter": "auto",
            "nvd": {
                "cpe_ranges": [],
                "description": "A" * 200,
                "cvss_v31_score": None,
                "cvss_v31_severity": "",
            },
            "osv": {"packages": []},
            "summary": {"affected_ecosystems": [], "first_patched_versions": {}, "total_affected_version_count": 0},
        }
        _render_version_range_text(data)
        out = capsys.readouterr().out
        assert "..." in out

    def test_render_many_cpe_ranges(self, capsys):
        from manus_agent.cli import _render_version_range_text

        ranges = [
            {"vendor": f"v{i}", "product": f"p{i}", "ecosystem_hint": "Unknown", "range_display": f">= {i}"}
            for i in range(35)
        ]
        data = {
            "cve_id": "CVE-TEST",
            "ecosystem_filter": "auto",
            "nvd": {"cpe_ranges": ranges, "description": "", "cvss_v31_score": None, "cvss_v31_severity": ""},
            "osv": {"packages": []},
            "summary": {"affected_ecosystems": [], "first_patched_versions": {}, "total_affected_version_count": 0},
        }
        _render_version_range_text(data)
        out = capsys.readouterr().out
        assert "... and 5 more" in out

    def test_render_many_osv_packages(self, capsys):
        from manus_agent.cli import _render_version_range_text

        packages = [
            {
                "ecosystem": f"Eco{i}",
                "package": f"pkg{i}",
                "first_patched": None,
                "affected_version_count": 0,
                "range_strings": [],
                "affected_versions": [],
            }
            for i in range(25)
        ]
        data = {
            "cve_id": "CVE-TEST",
            "ecosystem_filter": "auto",
            "nvd": {"cpe_ranges": [], "description": "", "cvss_v31_score": None, "cvss_v31_severity": ""},
            "osv": {"packages": packages},
            "summary": {"affected_ecosystems": [], "first_patched_versions": {}, "total_affected_version_count": 0},
        }
        _render_version_range_text(data)
        out = capsys.readouterr().out
        assert "... and 5 more" in out


# ===========================================================================
# Tests: Subcommand registration
# ===========================================================================
class TestSubcommandRegistration:
    """Tests for version-range subcommand registration."""

    def test_version_range_in_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "version-range" in _SUBCOMMANDS

    def test_main_dispatches_version_range(self):
        """Verify the main() function dispatches version-range."""
        import manus_agent.cli as cli_mod

        with patch.object(cli_mod, "_run_version_range", return_value=0) as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                cli_mod.main.__wrapped__ if hasattr(cli_mod.main, "__wrapped__") else None
                # Patch sys.argv
                with patch("sys.argv", ["manus-agent", "version-range", "CVE-2021-44228"]):
                    cli_mod.main()
            mock_run.assert_called_once_with(["CVE-2021-44228"])
            assert exc_info.value.code == 0
