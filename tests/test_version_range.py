#!/usr/bin/env python3
"""Comprehensive test suite for get_version_range tool and version-range CLI subcommand.

100% mocked — no real HTTP calls.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

# ---------------------------------------------------------------------------
# Ensure the package is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manus_agent.tools.get_version_range import (  # noqa: E402
    _CPE_ECOSYSTEM_MAP,
    _ECOSYSTEM_ALIASES,
    TOOL_SPEC,
    _extract_nvd_ranges,
    _fetch_nvd_ranges,
    _fetch_osv_ranges,
    _filter_by_ecosystem,
    _format_nvd_constraint,
    _normalise_ecosystem,
    _parse_cpe_uri,
    _parse_osv_affected,
    fetch_version_range,
    get_version_range,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """Disable retry delays in all tests."""
    monkeypatch.setenv("VERSION_RANGE_MAX_RETRIES", "1")
    monkeypatch.setenv("VERSION_RANGE_RETRY_BASE_DELAY", "0")


@pytest.fixture()
def mock_tool_use():
    """Factory for Strands ToolUse dicts."""

    def _make(cve_id: str = "CVE-2021-44228", ecosystem: str = ""):
        inp = {"cve_id": cve_id}
        if ecosystem:
            inp["ecosystem"] = ecosystem
        return {"toolUseId": "test-123", "input": inp}

    return _make


def _nvd_response(
    configurations=None,
    descriptions=None,
    metrics=None,
    published="2021-12-10T00:00:00.000",
    modified="2021-12-14T00:00:00.000",
):
    """Build a minimal NVD API response."""
    cve = {
        "configurations": configurations or [],
        "descriptions": descriptions or [{"lang": "en", "value": "Test vuln description"}],
        "metrics": metrics or {},
        "published": published,
        "lastModified": modified,
    }
    return {"vulnerabilities": [{"cve": cve}]}


def _osv_response(affected=None, aliases=None, severity=None, osv_id="CVE-2021-44228"):
    """Build a minimal OSV API response."""
    return {
        "id": osv_id,
        "affected": affected or [],
        "aliases": aliases or [],
        "severity": severity or [],
        "summary": "Test vulnerability",
    }


def _make_nvd_config(vendor, product, **kwargs):
    """Build an NVD CPE configuration node."""
    cpe_match = {
        "vulnerable": True,
        "criteria": f"cpe:2.3:a:{vendor}:{product}:*:*:*:*:*:*:*:*",
    }
    cpe_match.update(kwargs)
    return {
        "nodes": [{"operator": "OR", "cpeMatch": [cpe_match]}],
    }


def _make_osv_affected(ecosystem, package, introduced="0", fixed=None, last_affected=None, versions=None):
    """Build an OSV affected entry."""
    events = [{"introduced": introduced}]
    if fixed:
        events.append({"fixed": fixed})
    if last_affected:
        events.append({"last_affected": last_affected})
    entry = {
        "package": {"ecosystem": ecosystem, "name": package},
        "ranges": [{"type": "ECOSYSTEM", "events": events}],
    }
    if versions:
        entry["versions"] = versions
    return entry


# ===========================================================================
# TOOL_SPEC
# ===========================================================================
class TestToolSpec:
    def test_tool_spec_name(self):
        assert TOOL_SPEC["name"] == "get_version_range"

    def test_tool_spec_has_description(self):
        assert "version range" in TOOL_SPEC["description"].lower()

    def test_tool_spec_required_fields(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["required"]

    def test_tool_spec_has_ecosystem_property(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert "ecosystem" in props

    def test_tool_spec_cve_id_type(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert props["cve_id"]["type"] == "string"


# ===========================================================================
# CPE parsing
# ===========================================================================
class TestParseCpeUri:
    def test_full_cpe(self):
        cpe = "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"
        result = _parse_cpe_uri(cpe)
        assert result["vendor"] == "apache"
        assert result["product"] == "log4j"
        assert result["version"] == "2.14.1"

    def test_wildcard_version(self):
        cpe = "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*"
        result = _parse_cpe_uri(cpe)
        assert result["version"] == ""

    def test_wildcard_vendor(self):
        cpe = "cpe:2.3:a:*:log4j:1.0:*:*:*:*:*:*:*"
        result = _parse_cpe_uri(cpe)
        assert result["vendor"] == ""
        assert result["product"] == "log4j"

    def test_short_cpe(self):
        result = _parse_cpe_uri("cpe:2.3:a")
        assert result == {}

    def test_minimal_cpe(self):
        result = _parse_cpe_uri("cpe:2.3:a:vendor:product")
        assert result["vendor"] == "vendor"
        assert result["product"] == "product"

    def test_empty_string(self):
        result = _parse_cpe_uri("")
        assert result == {}


# ===========================================================================
# NVD range extraction
# ===========================================================================
class TestExtractNvdRanges:
    def test_single_range_with_start_end(self):
        config = _make_nvd_config(
            "apache",
            "log4j",
            versionStartIncluding="2.0",
            versionEndExcluding="2.17.0",
        )
        ranges = _extract_nvd_ranges({"configurations": [config]})
        assert len(ranges) == 1
        assert ranges[0]["vendor"] == "apache"
        assert ranges[0]["product"] == "log4j"
        assert ranges[0]["versionStartIncluding"] == "2.0"
        assert ranges[0]["versionEndExcluding"] == "2.17.0"

    def test_ecosystem_inferred_from_product(self):
        config = _make_nvd_config("python", "django")
        ranges = _extract_nvd_ranges({"configurations": [config]})
        assert ranges[0]["inferred_ecosystem"] == "PyPI"

    def test_ecosystem_inferred_from_vendor(self):
        config = _make_nvd_config("apache", "some_project")
        ranges = _extract_nvd_ranges({"configurations": [config]})
        assert ranges[0]["inferred_ecosystem"] == "Maven"

    def test_no_ecosystem_inferred(self):
        config = _make_nvd_config("unknown_vendor", "unknown_product")
        ranges = _extract_nvd_ranges({"configurations": [config]})
        assert ranges[0]["inferred_ecosystem"] == ""

    def test_non_vulnerable_skipped(self):
        config = {
            "nodes": [
                {
                    "operator": "OR",
                    "cpeMatch": [
                        {
                            "vulnerable": False,
                            "criteria": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                        }
                    ],
                }
            ],
        }
        ranges = _extract_nvd_ranges({"configurations": [config]})
        assert len(ranges) == 0

    def test_multiple_nodes(self):
        config = {
            "nodes": [
                {
                    "operator": "OR",
                    "cpeMatch": [
                        {
                            "vulnerable": True,
                            "criteria": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
                        },
                    ],
                },
                {
                    "operator": "OR",
                    "cpeMatch": [
                        {
                            "vulnerable": True,
                            "criteria": "cpe:2.3:a:apache:log4j:2.14.0:*:*:*:*:*:*:*",
                        },
                    ],
                },
            ],
        }
        ranges = _extract_nvd_ranges({"configurations": [config]})
        assert len(ranges) == 2

    def test_empty_configurations(self):
        assert _extract_nvd_ranges({"configurations": []}) == []
        assert _extract_nvd_ranges({}) == []

    def test_invalid_cpe_match_entry(self):
        config = {
            "nodes": [{"operator": "OR", "cpeMatch": ["not-a-dict"]}],
        }
        ranges = _extract_nvd_ranges({"configurations": [config]})
        assert len(ranges) == 0

    def test_version_end_including(self):
        config = _make_nvd_config(
            "vendor",
            "product",
            versionEndIncluding="3.0.0",
        )
        ranges = _extract_nvd_ranges({"configurations": [config]})
        assert ranges[0].get("versionEndIncluding") == "3.0.0"

    def test_version_start_excluding(self):
        config = _make_nvd_config(
            "vendor",
            "product",
            versionStartExcluding="1.0.0",
        )
        ranges = _extract_nvd_ranges({"configurations": [config]})
        assert ranges[0].get("versionStartExcluding") == "1.0.0"

    def test_operator_preserved(self):
        config = {
            "nodes": [
                {
                    "operator": "AND",
                    "cpeMatch": [
                        {
                            "vulnerable": True,
                            "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                        }
                    ],
                }
            ],
        }
        ranges = _extract_nvd_ranges({"configurations": [config]})
        assert ranges[0]["operator"] == "AND"


# ===========================================================================
# Format NVD constraint
# ===========================================================================
class TestFormatNvdConstraint:
    def test_start_end_range(self):
        r = {"versionStartIncluding": "2.0", "versionEndExcluding": "2.17.0"}
        assert _format_nvd_constraint(r) == ">= 2.0, < 2.17.0"

    def test_only_start_including(self):
        r = {"versionStartIncluding": "1.0"}
        assert _format_nvd_constraint(r) == ">= 1.0"

    def test_only_end_including(self):
        r = {"versionEndIncluding": "3.0"}
        assert _format_nvd_constraint(r) == "<= 3.0"

    def test_start_excluding(self):
        r = {"versionStartExcluding": "1.0"}
        assert _format_nvd_constraint(r) == "> 1.0"

    def test_exact_version(self):
        r = {"exact_version": "2.14.1"}
        assert _format_nvd_constraint(r) == "== 2.14.1"

    def test_all_versions(self):
        assert _format_nvd_constraint({}) == "(all versions)"

    def test_all_four_constraints(self):
        r = {
            "versionStartIncluding": "1.0",
            "versionStartExcluding": "0.9",
            "versionEndIncluding": "3.0",
            "versionEndExcluding": "3.1",
        }
        result = _format_nvd_constraint(r)
        assert ">= 1.0" in result
        assert "> 0.9" in result
        assert "<= 3.0" in result
        assert "< 3.1" in result


# ===========================================================================
# OSV affected parsing
# ===========================================================================
class TestParseOsvAffected:
    def test_basic_affected(self):
        affected = [_make_osv_affected("PyPI", "django", introduced="0", fixed="3.2.12")]
        result = _parse_osv_affected(affected)
        assert len(result) == 1
        assert result[0]["ecosystem"] == "PyPI"
        assert result[0]["package"] == "django"
        assert result[0]["introduced"] == ["0"]
        assert result[0]["fixed"] == ["3.2.12"]
        assert result[0]["first_patched_version"] == "3.2.12"

    def test_no_fix(self):
        affected = [_make_osv_affected("npm", "lodash", introduced="0", last_affected="4.17.20")]
        result = _parse_osv_affected(affected)
        assert result[0]["last_affected"] == ["4.17.20"]
        assert result[0]["first_patched_version"] is None

    def test_with_versions_list(self):
        affected = [_make_osv_affected("PyPI", "pkg", versions=["1.0", "1.1", "1.2"])]
        result = _parse_osv_affected(affected)
        assert result[0]["affected_version_count"] == 3
        assert result[0]["affected_versions_sample"] == ["1.0", "1.1", "1.2"]

    def test_range_strings_with_fix(self):
        affected = [_make_osv_affected("PyPI", "pkg", introduced="1.0", fixed="2.0")]
        result = _parse_osv_affected(affected)
        assert ">= 1.0, < 2.0" in result[0]["range_strings"]

    def test_range_strings_with_last_affected(self):
        affected = [_make_osv_affected("PyPI", "pkg", introduced="1.0", last_affected="1.9")]
        result = _parse_osv_affected(affected)
        assert ">= 1.0, <= 1.9" in result[0]["range_strings"]

    def test_range_strings_no_upper_bound(self):
        affected = [_make_osv_affected("PyPI", "pkg", introduced="1.0")]
        result = _parse_osv_affected(affected)
        assert ">= 1.0" in result[0]["range_strings"]

    def test_empty_affected(self):
        assert _parse_osv_affected([]) == []
        assert _parse_osv_affected(None) == []

    def test_non_dict_entries_skipped(self):
        assert _parse_osv_affected(["not-a-dict"]) == []

    def test_no_package_skipped(self):
        result = _parse_osv_affected([{"package": {}, "ranges": []}])
        assert len(result) == 0

    def test_non_dict_range_events_skipped(self):
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "pkg"},
                "ranges": [{"type": "ECOSYSTEM", "events": ["not-a-dict"]}],
            }
        ]
        result = _parse_osv_affected(affected)
        assert len(result) == 1
        assert result[0]["introduced"] == []

    def test_non_dict_ranges_skipped(self):
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "pkg"},
                "ranges": ["not-a-dict"],
            }
        ]
        result = _parse_osv_affected(affected)
        assert len(result) == 1

    def test_versions_sample_capped(self):
        versions = [str(i) for i in range(30)]
        affected = [_make_osv_affected("PyPI", "pkg", versions=versions)]
        result = _parse_osv_affected(affected)
        assert result[0]["affected_version_count"] == 30
        assert len(result[0]["affected_versions_sample"]) == 15

    def test_range_type_preserved(self):
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "pkg"},
                "ranges": [{"type": "SEMVER", "events": [{"introduced": "0"}]}],
            }
        ]
        result = _parse_osv_affected(affected)
        assert result[0]["range_type"] == "SEMVER"


# ===========================================================================
# Ecosystem normalisation
# ===========================================================================
class TestNormaliseEcosystem:
    def test_pypi_alias(self):
        assert _normalise_ecosystem("pypi") == "PyPI"
        assert _normalise_ecosystem("python") == "PyPI"

    def test_npm_alias(self):
        assert _normalise_ecosystem("npm") == "npm"
        assert _normalise_ecosystem("node") == "npm"

    def test_maven_alias(self):
        assert _normalise_ecosystem("maven") == "Maven"
        assert _normalise_ecosystem("java") == "Maven"

    def test_go_alias(self):
        assert _normalise_ecosystem("go") == "Go"
        assert _normalise_ecosystem("golang") == "Go"

    def test_auto_returns_empty(self):
        assert _normalise_ecosystem("auto") == ""

    def test_empty_returns_empty(self):
        assert _normalise_ecosystem("") == ""

    def test_case_insensitive(self):
        assert _normalise_ecosystem("PyPI") == "PyPI"
        assert _normalise_ecosystem("NPM") == "npm"

    def test_unknown_passthrough(self):
        assert _normalise_ecosystem("CustomEco") == "CustomEco"

    def test_whitespace_stripped(self):
        assert _normalise_ecosystem("  pypi  ") == "PyPI"

    def test_crates_io(self):
        assert _normalise_ecosystem("crates.io") == "crates.io"
        assert _normalise_ecosystem("rust") == "crates.io"

    def test_rubygems(self):
        assert _normalise_ecosystem("rubygems") == "RubyGems"
        assert _normalise_ecosystem("ruby") == "RubyGems"


# ===========================================================================
# Ecosystem filter
# ===========================================================================
class TestFilterByEcosystem:
    def test_no_filter(self):
        pkgs = [{"ecosystem": "PyPI"}, {"ecosystem": "npm"}]
        nvd = [{"inferred_ecosystem": "PyPI"}]
        fp, fn = _filter_by_ecosystem(pkgs, nvd, "")
        assert len(fp) == 2
        assert len(fn) == 1

    def test_filter_pypi(self):
        pkgs = [{"ecosystem": "PyPI"}, {"ecosystem": "npm"}]
        nvd = [{"inferred_ecosystem": "PyPI"}, {"inferred_ecosystem": "npm"}]
        fp, fn = _filter_by_ecosystem(pkgs, nvd, "PyPI")
        assert len(fp) == 1
        assert fp[0]["ecosystem"] == "PyPI"
        assert len(fn) == 1
        assert fn[0]["inferred_ecosystem"] == "PyPI"

    def test_filter_case_insensitive(self):
        pkgs = [{"ecosystem": "PyPI"}]
        nvd = [{"inferred_ecosystem": "PyPI"}]
        fp, fn = _filter_by_ecosystem(pkgs, nvd, "pypi")
        assert len(fp) == 1
        assert len(fn) == 1

    def test_filter_no_matches(self):
        pkgs = [{"ecosystem": "PyPI"}]
        nvd = [{"inferred_ecosystem": "PyPI"}]
        fp, fn = _filter_by_ecosystem(pkgs, nvd, "npm")
        assert len(fp) == 0
        assert len(fn) == 0


# ===========================================================================
# fetch_osv_ranges
# ===========================================================================
class TestFetchOsvRanges:
    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_success(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _osv_response(
            affected=[_make_osv_affected("PyPI", "django", fixed="3.2.12")],
            aliases=["CVE-2021-44228", "GHSA-xxxx-xxxx-xxxx"],
            severity=[{"type": "CVSS_V3", "score": "9.8"}],
        )
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = _fetch_osv_ranges("CVE-2021-44228")
        assert result["found"] is True
        assert len(result["packages"]) == 1
        assert result["packages"][0]["ecosystem"] == "PyPI"
        assert result["aliases"] == ["CVE-2021-44228", "GHSA-xxxx-xxxx-xxxx"]

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_404(self, mock_get):
        resp = MagicMock()
        resp.status_code = 404
        mock_get.return_value = resp

        result = _fetch_osv_ranges("CVE-9999-9999")
        assert result["found"] is False
        assert result["packages"] == []

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_network_error(self, mock_get):
        import requests as _req

        mock_get.side_effect = _req.exceptions.ConnectionError("fail")

        result = _fetch_osv_ranges("CVE-2021-44228")
        assert result["found"] is False
        assert "error" in result

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_follows_ghsa_aliases(self, mock_get):
        # Primary record: no affected data, has GHSA alias
        primary = MagicMock()
        primary.status_code = 200
        primary.json.return_value = _osv_response(
            affected=[],
            aliases=["CVE-2021-44228", "GHSA-1234-1234-1234"],
        )
        primary.raise_for_status = MagicMock()

        # GHSA follow-up: has affected data
        ghsa_resp = MagicMock()
        ghsa_resp.status_code = 200
        ghsa_resp.json.return_value = _osv_response(
            osv_id="GHSA-1234-1234-1234",
            affected=[_make_osv_affected("npm", "log4js", fixed="6.4.0")],
        )
        ghsa_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [primary, ghsa_resp]

        result = _fetch_osv_ranges("CVE-2021-44228")
        assert result["found"] is True
        assert len(result["packages"]) == 1
        assert result["packages"][0]["package"] == "log4js"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_ghsa_follow_deduplicates(self, mock_get):
        # Primary record: no affected, has GHSA alias
        primary = MagicMock()
        primary.status_code = 200
        primary.json.return_value = _osv_response(
            affected=[],
            aliases=["CVE-2021-44228", "GHSA-1234-1234-1234"],
            osv_id="CVE-2021-44228",
        )
        primary.raise_for_status = MagicMock()

        # GHSA returns same OSV id (should be skipped as duplicate)
        ghsa_resp = MagicMock()
        ghsa_resp.status_code = 200
        ghsa_resp.json.return_value = _osv_response(
            osv_id="CVE-2021-44228",  # same ID as primary
            affected=[_make_osv_affected("npm", "log4js")],
        )
        ghsa_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [primary, ghsa_resp]

        result = _fetch_osv_ranges("CVE-2021-44228")
        assert result["found"] is True
        # Should still have 0 packages since the duplicate ID was skipped
        assert len(result["packages"]) == 0

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_ghsa_follow_failure_ignored(self, mock_get):
        primary = MagicMock()
        primary.status_code = 200
        primary.json.return_value = _osv_response(
            affected=[],
            aliases=["CVE-2021-44228", "GHSA-fail-fail-fail"],
        )
        primary.raise_for_status = MagicMock()

        mock_get.side_effect = [primary, Exception("GHSA fetch failed")]

        result = _fetch_osv_ranges("CVE-2021-44228")
        assert result["found"] is True
        assert len(result["packages"]) == 0

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_http_error_response(self, mock_get):
        import requests as _req

        resp = MagicMock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = _req.exceptions.HTTPError("500")
        mock_get.return_value = resp

        result = _fetch_osv_ranges("CVE-2021-44228")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_json_decode_error(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = resp

        result = _fetch_osv_ranges("CVE-2021-44228")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_severity_merged_from_alias(self, mock_get):
        primary = MagicMock()
        primary.status_code = 200
        primary.json.return_value = _osv_response(
            affected=[],
            aliases=["GHSA-abcd-abcd-abcd"],
            severity=[],
        )
        primary.raise_for_status = MagicMock()

        ghsa_resp = MagicMock()
        ghsa_resp.status_code = 200
        ghsa_resp.json.return_value = _osv_response(
            osv_id="GHSA-abcd-abcd-abcd",
            affected=[_make_osv_affected("PyPI", "pkg", fixed="2.0")],
            severity=[{"type": "CVSS_V3", "score": "7.5"}],
        )
        ghsa_resp.raise_for_status = MagicMock()

        mock_get.side_effect = [primary, ghsa_resp]

        result = _fetch_osv_ranges("CVE-2021-44228")
        assert len(result["severity"]) == 1
        assert result["severity"][0]["score"] == "7.5"


# ===========================================================================
# fetch_nvd_ranges
# ===========================================================================
class TestFetchNvdRanges:
    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_success(self, mock_get):
        config = _make_nvd_config(
            "apache",
            "log4j",
            versionStartIncluding="2.0",
            versionEndExcluding="2.17.0",
        )
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _nvd_response(
            configurations=[config],
            metrics={
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
        )
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = _fetch_nvd_ranges("CVE-2021-44228")
        assert result["found"] is True
        assert len(result["ranges"]) == 1
        assert result["description"] == "Test vuln description"
        assert result["cvss"]["baseScore"] == 10.0

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_no_vulnerabilities(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"vulnerabilities": []}
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = _fetch_nvd_ranges("CVE-9999-9999")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_network_error(self, mock_get):
        import requests as _req

        mock_get.side_effect = _req.exceptions.ConnectionError("fail")

        result = _fetch_nvd_ranges("CVE-2021-44228")
        assert result["found"] is False
        assert "error" in result

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_http_error(self, mock_get):
        import requests as _req

        resp = MagicMock()
        resp.status_code = 500
        resp.raise_for_status.side_effect = _req.exceptions.HTTPError("500")
        mock_get.return_value = resp

        result = _fetch_nvd_ranges("CVE-2021-44228")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_json_decode_error(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = resp

        result = _fetch_nvd_ranges("CVE-2021-44228")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_cvss_v30_fallback(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _nvd_response(
            metrics={
                "cvssMetricV30": [
                    {
                        "cvssData": {
                            "version": "3.0",
                            "baseScore": 9.0,
                            "baseSeverity": "CRITICAL",
                            "vectorString": "test",
                        }
                    }
                ]
            },
        )
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = _fetch_nvd_ranges("CVE-2021-44228")
        assert result["cvss"]["version"] == "3.0"
        assert result["cvss"]["baseScore"] == 9.0

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_no_english_description(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _nvd_response(
            descriptions=[{"lang": "es", "value": "Descripción en español"}],
        )
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        result = _fetch_nvd_ranges("CVE-2021-44228")
        assert result["description"] == "Descripción en español"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_nvd_api_key_header(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _nvd_response()
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp

        with patch.dict(os.environ, {"NVD_API_KEY": "test-key"}):
            from manus_agent.tools.get_version_range import _build_nvd_headers

            headers = _build_nvd_headers()
            assert headers["apiKey"] == "test-key"

    def test_nvd_headers_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            from manus_agent.tools.get_version_range import _build_nvd_headers

            headers = _build_nvd_headers()
            assert "apiKey" not in headers


# ===========================================================================
# fetch_version_range (integration of both sources)
# ===========================================================================
class TestFetchVersionRange:
    @patch("manus_agent.tools.get_version_range._fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_ranges")
    def test_both_sources_found(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": True,
            "ranges": [
                {
                    "vendor": "apache",
                    "product": "log4j",
                    "exact_version": "",
                    "operator": "OR",
                    "versionStartIncluding": "2.0",
                    "versionEndExcluding": "2.17.0",
                    "inferred_ecosystem": "Maven",
                }
            ],
            "description": "Remote code execution",
            "cvss": {"baseScore": 10.0, "baseSeverity": "CRITICAL"},
            "published": "2021-12-10",
            "modified": "2021-12-14",
        }
        mock_osv.return_value = {
            "found": True,
            "packages": [
                {
                    "ecosystem": "Maven",
                    "package": "org.apache.logging.log4j:log4j-core",
                    "introduced": ["2.0"],
                    "fixed": ["2.17.0"],
                    "last_affected": [],
                    "range_strings": [">= 2.0, < 2.17.0"],
                    "first_patched_version": "2.17.0",
                    "affected_version_count": 50,
                    "affected_versions_sample": ["2.14.0", "2.14.1"],
                    "range_type": "ECOSYSTEM",
                }
            ],
            "aliases": ["GHSA-jfh8-c2jp-5v3q"],
            "severity": [{"type": "CVSS_V3", "score": "10.0"}],
        }

        result = fetch_version_range("CVE-2021-44228")
        assert result["found"] is True
        assert result["cve_id"] == "CVE-2021-44228"
        assert len(result["osv_packages"]) == 1
        assert len(result["nvd_ranges"]) == 1
        assert len(result["first_patched_versions"]) == 1
        assert result["first_patched_versions"][0]["fixed_version"] == "2.17.0"
        assert "Maven" in result["ecosystems"]
        assert result["summary"]["has_fix"] is True

    @patch("manus_agent.tools.get_version_range._fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_ranges")
    def test_only_nvd_found(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": True,
            "ranges": [
                {
                    "vendor": "vendor",
                    "product": "product",
                    "exact_version": "1.0",
                    "operator": "OR",
                    "inferred_ecosystem": "",
                }
            ],
            "description": "Test",
            "cvss": {},
            "published": "",
            "modified": "",
        }
        mock_osv.return_value = {
            "found": False,
            "packages": [],
            "aliases": [],
            "severity": [],
        }

        result = fetch_version_range("CVE-2024-1234")
        assert result["found"] is True
        assert "NVD" in result["message"]
        assert "no osv" in result["message"].lower() or "no osv package" in result["message"].lower()

    @patch("manus_agent.tools.get_version_range._fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_ranges")
    def test_only_osv_found(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": False,
            "ranges": [],
            "description": "",
            "cvss": {},
            "published": "",
            "modified": "",
        }
        mock_osv.return_value = {
            "found": True,
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "django",
                    "introduced": ["0"],
                    "fixed": ["3.2.12"],
                    "last_affected": [],
                    "range_strings": [">= 0, < 3.2.12"],
                    "first_patched_version": "3.2.12",
                    "affected_version_count": 10,
                    "affected_versions_sample": [],
                    "range_type": "ECOSYSTEM",
                }
            ],
            "aliases": [],
            "severity": [],
        }

        result = fetch_version_range("CVE-2024-5678")
        assert result["found"] is True
        assert len(result["osv_packages"]) == 1

    @patch("manus_agent.tools.get_version_range._fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_ranges")
    def test_neither_found(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": False,
            "ranges": [],
            "description": "",
            "cvss": {},
            "published": "",
            "modified": "",
        }
        mock_osv.return_value = {
            "found": False,
            "packages": [],
            "aliases": [],
            "severity": [],
        }

        result = fetch_version_range("CVE-9999-9999")
        assert result["found"] is False
        assert "No data found" in result.get("error", "")

    @patch("manus_agent.tools.get_version_range._fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_ranges")
    def test_neither_found_with_errors(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": False,
            "ranges": [],
            "description": "",
            "cvss": {},
            "published": "",
            "modified": "",
            "error": "NVD timeout",
        }
        mock_osv.return_value = {
            "found": False,
            "packages": [],
            "aliases": [],
            "severity": [],
            "error": "OSV timeout",
        }

        result = fetch_version_range("CVE-2024-1111")
        assert result["found"] is False
        assert "NVD" in result["error"]
        assert "OSV" in result["error"]

    def test_invalid_cve_id(self):
        result = fetch_version_range("")
        assert result["found"] is False
        assert "Invalid" in result["error"]

    def test_invalid_cve_format(self):
        result = fetch_version_range("not-a-cve")
        assert result["found"] is False
        assert "Invalid" in result["error"]

    def test_cve_normalised_to_upper(self):
        with (
            patch("manus_agent.tools.get_version_range._fetch_nvd_ranges") as mock_nvd,
            patch("manus_agent.tools.get_version_range._fetch_osv_ranges") as mock_osv,
        ):
            mock_nvd.return_value = {
                "found": False,
                "ranges": [],
                "description": "",
                "cvss": {},
                "published": "",
                "modified": "",
            }
            mock_osv.return_value = {
                "found": False,
                "packages": [],
                "aliases": [],
                "severity": [],
            }
            result = fetch_version_range("cve-2021-44228")
            assert result["cve_id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.get_version_range._fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_ranges")
    def test_ecosystem_filter_applied(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": True,
            "ranges": [
                {"vendor": "a", "product": "b", "exact_version": "", "operator": "OR", "inferred_ecosystem": "PyPI"},
                {"vendor": "c", "product": "d", "exact_version": "", "operator": "OR", "inferred_ecosystem": "npm"},
            ],
            "description": "",
            "cvss": {},
            "published": "",
            "modified": "",
        }
        mock_osv.return_value = {
            "found": True,
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "pkg1",
                    "introduced": ["0"],
                    "fixed": ["1.0"],
                    "last_affected": [],
                    "range_strings": [],
                    "first_patched_version": "1.0",
                    "affected_version_count": 0,
                    "affected_versions_sample": [],
                    "range_type": "",
                },
                {
                    "ecosystem": "npm",
                    "package": "pkg2",
                    "introduced": ["0"],
                    "fixed": [],
                    "last_affected": [],
                    "range_strings": [],
                    "first_patched_version": None,
                    "affected_version_count": 0,
                    "affected_versions_sample": [],
                    "range_type": "",
                },
            ],
            "aliases": [],
            "severity": [],
        }

        result = fetch_version_range("CVE-2024-1234", ecosystem="pypi")
        assert result["found"] is True
        assert len(result["osv_packages"]) == 1
        assert result["osv_packages"][0]["package"] == "pkg1"
        assert len(result["nvd_ranges"]) == 1
        assert result["ecosystem_filter"] == "PyPI"

    @patch("manus_agent.tools.get_version_range._fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_ranges")
    def test_first_patched_deduplicated(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": True,
            "ranges": [],
            "description": "",
            "cvss": {},
            "published": "",
            "modified": "",
        }
        mock_osv.return_value = {
            "found": True,
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "pkg",
                    "introduced": ["0"],
                    "fixed": ["2.0"],
                    "last_affected": [],
                    "range_strings": [],
                    "first_patched_version": "2.0",
                    "affected_version_count": 0,
                    "affected_versions_sample": [],
                    "range_type": "",
                },
                {
                    "ecosystem": "PyPI",
                    "package": "pkg",
                    "introduced": ["0"],
                    "fixed": ["2.0"],
                    "last_affected": [],
                    "range_strings": [],
                    "first_patched_version": "2.0",
                    "affected_version_count": 0,
                    "affected_versions_sample": [],
                    "range_type": "",
                },
            ],
            "aliases": [],
            "severity": [],
        }

        result = fetch_version_range("CVE-2024-1234")
        assert len(result["first_patched_versions"]) == 1

    @patch("manus_agent.tools.get_version_range._fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_ranges")
    def test_message_includes_first_patched(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": True,
            "ranges": [],
            "description": "",
            "cvss": {},
            "published": "",
            "modified": "",
        }
        mock_osv.return_value = {
            "found": True,
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "django",
                    "introduced": ["0"],
                    "fixed": ["3.2.12"],
                    "last_affected": [],
                    "range_strings": [],
                    "first_patched_version": "3.2.12",
                    "affected_version_count": 0,
                    "affected_versions_sample": [],
                    "range_type": "",
                }
            ],
            "aliases": [],
            "severity": [],
        }

        result = fetch_version_range("CVE-2024-1234")
        assert "django@3.2.12" in result["message"]

    @patch("manus_agent.tools.get_version_range._fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_ranges")
    def test_found_but_no_structured_ranges(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": True,
            "ranges": [],
            "description": "desc",
            "cvss": {},
            "published": "",
            "modified": "",
        }
        mock_osv.return_value = {
            "found": True,
            "packages": [],
            "aliases": [],
            "severity": [],
        }

        result = fetch_version_range("CVE-2024-1234")
        assert result["found"] is True
        assert "no structured version ranges" in result["message"].lower()

    @patch("manus_agent.tools.get_version_range._fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_ranges")
    def test_many_first_patched_truncated_in_message(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": True,
            "ranges": [],
            "description": "",
            "cvss": {},
            "published": "",
            "modified": "",
        }
        pkgs = []
        for i in range(8):
            pkgs.append(
                {
                    "ecosystem": "PyPI",
                    "package": f"pkg{i}",
                    "introduced": ["0"],
                    "fixed": [f"{i}.0"],
                    "last_affected": [],
                    "range_strings": [],
                    "first_patched_version": f"{i}.0",
                    "affected_version_count": 0,
                    "affected_versions_sample": [],
                    "range_type": "",
                }
            )
        mock_osv.return_value = {
            "found": True,
            "packages": pkgs,
            "aliases": [],
            "severity": [],
        }

        result = fetch_version_range("CVE-2024-1234")
        assert "+3 more" in result["message"]


# ===========================================================================
# Strands tool entry point
# ===========================================================================
class TestGetVersionRangeTool:
    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_success(self, mock_fetch, mock_tool_use):
        mock_fetch.return_value = {
            "found": True,
            "cve_id": "CVE-2021-44228",
            "message": "test",
            "osv_packages": [],
            "nvd_ranges": [],
        }
        result = get_version_range(mock_tool_use())
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_error(self, mock_fetch, mock_tool_use):
        mock_fetch.return_value = {
            "found": False,
            "message": "No data found.",
        }
        result = get_version_range(mock_tool_use())
        assert result["status"] == "error"
        assert "No data found" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_ecosystem_passed(self, mock_fetch, mock_tool_use):
        mock_fetch.return_value = {"found": False, "message": "Not found"}
        get_version_range(mock_tool_use(ecosystem="pypi"))
        mock_fetch.assert_called_once_with("CVE-2021-44228", "pypi")


# ===========================================================================
# CLI: _build_version_range_parser
# ===========================================================================
class TestBuildVersionRangeParser:
    def test_parser_accepts_cve_id(self):
        from manus_agent.cli import _build_version_range_parser

        p = _build_version_range_parser()
        args = p.parse_args(["CVE-2021-44228"])
        assert args.cve_id == "CVE-2021-44228"

    def test_parser_default_ecosystem(self):
        from manus_agent.cli import _build_version_range_parser

        p = _build_version_range_parser()
        args = p.parse_args(["CVE-2021-44228"])
        assert args.ecosystem == "auto"

    def test_parser_ecosystem_override(self):
        from manus_agent.cli import _build_version_range_parser

        p = _build_version_range_parser()
        args = p.parse_args(["CVE-2021-44228", "--ecosystem", "pypi"])
        assert args.ecosystem == "pypi"

    def test_parser_output_json(self):
        from manus_agent.cli import _build_version_range_parser

        p = _build_version_range_parser()
        args = p.parse_args(["CVE-2021-44228", "--output", "json"])
        assert args.output == "json"

    def test_parser_default_output(self):
        from manus_agent.cli import _build_version_range_parser

        p = _build_version_range_parser()
        args = p.parse_args(["CVE-2021-44228"])
        assert args.output == "text"


# ===========================================================================
# CLI: _run_version_range
# ===========================================================================
class TestRunVersionRange:
    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_json_output(self, mock_fetch, capsys):
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {
            "found": True,
            "cve_id": "CVE-2021-44228",
            "message": "test",
            "osv_packages": [],
            "nvd_ranges": [],
            "summary": {"has_fix": False},
        }
        rc = _run_version_range(["CVE-2021-44228", "--output", "json"])
        assert rc == 0
        output = json.loads(capsys.readouterr().out)
        assert output["cve_id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_output_with_osv(self, mock_fetch, capsys):
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {
            "found": True,
            "cve_id": "CVE-2021-44228",
            "description": "Remote code execution in Log4j",
            "cvss": {"baseScore": 10.0, "baseSeverity": "CRITICAL", "vectorString": "CVSS:3.1/test"},
            "published": "2021-12-10",
            "modified": "2021-12-14",
            "aliases": ["GHSA-jfh8-c2jp-5v3q"],
            "ecosystem_filter": "",
            "ecosystems": ["Maven"],
            "message": "CVE-2021-44228: 1 package",
            "osv_packages": [
                {
                    "ecosystem": "Maven",
                    "package": "org.apache.logging.log4j:log4j-core",
                    "range_strings": [">= 2.0, < 2.17.0"],
                    "first_patched_version": "2.17.0",
                    "last_affected": [],
                    "affected_versions_sample": ["2.14.0", "2.14.1"],
                    "affected_version_count": 50,
                }
            ],
            "nvd_ranges": [
                {
                    "vendor": "apache",
                    "product": "log4j",
                    "constraint": ">= 2.0, < 2.17.0",
                    "inferred_ecosystem": "Maven",
                }
            ],
            "first_patched_versions": [
                {
                    "ecosystem": "Maven",
                    "package": "org.apache.logging.log4j:log4j-core",
                    "fixed_version": "2.17.0",
                }
            ],
            "summary": {"has_fix": True, "fix_count": 1},
        }
        rc = _run_version_range(["CVE-2021-44228"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2021-44228" in out
        assert "log4j-core" in out
        assert "2.17.0" in out
        assert "CRITICAL" in out
        assert "Maven" in out

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_not_found(self, mock_fetch, capsys):
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {
            "found": False,
            "message": "No version range data found for CVE-9999-9999.",
        }
        rc = _run_version_range(["CVE-9999-9999"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "No version range data found" in out

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_ecosystem_auto_passes_empty(self, mock_fetch):
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {"found": False, "message": "Not found"}
        _run_version_range(["CVE-2021-44228"])
        mock_fetch.assert_called_once_with("CVE-2021-44228", "")

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_ecosystem_pypi_passed(self, mock_fetch):
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {"found": False, "message": "Not found"}
        _run_version_range(["CVE-2021-44228", "--ecosystem", "pypi"])
        mock_fetch.assert_called_once_with("CVE-2021-44228", "pypi")

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_with_last_affected_no_fix(self, mock_fetch, capsys):
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {
            "found": True,
            "cve_id": "CVE-2024-1234",
            "description": "",
            "cvss": {},
            "published": "",
            "modified": "",
            "aliases": [],
            "ecosystem_filter": "",
            "ecosystems": ["npm"],
            "message": "test",
            "osv_packages": [
                {
                    "ecosystem": "npm",
                    "package": "lodash",
                    "range_strings": [">= 0, <= 4.17.20"],
                    "first_patched_version": None,
                    "last_affected": ["4.17.20"],
                    "affected_versions_sample": [],
                    "affected_version_count": 0,
                }
            ],
            "nvd_ranges": [],
            "first_patched_versions": [],
            "summary": {"has_fix": False, "fix_count": 0},
        }
        rc = _run_version_range(["CVE-2024-1234"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Last affected" in out
        assert "No fix version" in out

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_long_description_truncated(self, mock_fetch, capsys):
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {
            "found": True,
            "cve_id": "CVE-2024-1234",
            "description": "A" * 300,
            "cvss": {},
            "published": "",
            "modified": "",
            "aliases": [],
            "ecosystem_filter": "",
            "ecosystems": [],
            "message": "test",
            "osv_packages": [],
            "nvd_ranges": [],
            "first_patched_versions": [],
            "summary": {"has_fix": False, "fix_count": 0},
        }
        _run_version_range(["CVE-2024-1234"])
        out = capsys.readouterr().out
        assert "..." in out

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_ecosystem_filter_shown(self, mock_fetch, capsys):
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {
            "found": True,
            "cve_id": "CVE-2024-1234",
            "description": "",
            "cvss": {},
            "published": "",
            "modified": "",
            "aliases": [],
            "ecosystem_filter": "PyPI",
            "ecosystems": ["PyPI"],
            "message": "test",
            "osv_packages": [],
            "nvd_ranges": [],
            "first_patched_versions": [],
            "summary": {"has_fix": False, "fix_count": 0},
        }
        _run_version_range(["CVE-2024-1234", "--ecosystem", "pypi"])
        out = capsys.readouterr().out
        assert "Ecosystem filter: PyPI" in out


# ===========================================================================
# CLI: subcommand in _SUBCOMMANDS
# ===========================================================================
class TestSubcommandRegistration:
    def test_version_range_in_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "version-range" in _SUBCOMMANDS


# ===========================================================================
# HTTP retry helper
# ===========================================================================
class TestGetWithRetry:
    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_success_first_attempt(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        mock_get.return_value = resp

        from manus_agent.tools.get_version_range import _get_with_retry

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retries_on_429(self, mock_get):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_200 = MagicMock()
        resp_200.status_code = 200
        mock_get.side_effect = [resp_429, resp_200]

        from manus_agent.tools.get_version_range import _get_with_retry

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retries_exhausted(self, mock_get):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        mock_get.return_value = resp_429

        from manus_agent.tools.get_version_range import _get_with_retry

        with pytest.raises((requests.exceptions.HTTPError, RuntimeError)):
            _get_with_retry("https://example.com")

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retries_on_connection_error(self, mock_get):
        import requests as _req

        resp_200 = MagicMock()
        resp_200.status_code = 200
        mock_get.side_effect = [_req.exceptions.ConnectionError("fail"), resp_200]

        from manus_agent.tools.get_version_range import _get_with_retry

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_non_retryable_status_returned(self, mock_get):
        resp = MagicMock()
        resp.status_code = 403
        mock_get.return_value = resp

        from manus_agent.tools.get_version_range import _get_with_retry

        result = _get_with_retry("https://example.com")
        assert result.status_code == 403


# ===========================================================================
# CPE ecosystem map coverage
# ===========================================================================
class TestCpeEcosystemMap:
    def test_python_ecosystems(self):
        for key in ["python", "django", "flask", "numpy", "requests"]:
            assert _CPE_ECOSYSTEM_MAP[key] == "PyPI"

    def test_npm_ecosystems(self):
        for key in ["node.js", "nodejs", "express", "lodash"]:
            assert _CPE_ECOSYSTEM_MAP[key] == "npm"

    def test_maven_ecosystems(self):
        for key in ["apache", "spring", "log4j", "tomcat"]:
            assert _CPE_ECOSYSTEM_MAP[key] == "Maven"

    def test_go_ecosystems(self):
        assert _CPE_ECOSYSTEM_MAP["golang"] == "Go"
        assert _CPE_ECOSYSTEM_MAP["go"] == "Go"

    def test_rust_ecosystems(self):
        assert _CPE_ECOSYSTEM_MAP["rust"] == "crates.io"

    def test_ruby_ecosystems(self):
        assert _CPE_ECOSYSTEM_MAP["ruby"] == "RubyGems"
        assert _CPE_ECOSYSTEM_MAP["rails"] == "RubyGems"

    def test_php_ecosystems(self):
        assert _CPE_ECOSYSTEM_MAP["php"] == "Packagist"
        assert _CPE_ECOSYSTEM_MAP["laravel"] == "Packagist"


# ===========================================================================
# Ecosystem alias coverage
# ===========================================================================
class TestEcosystemAliases:
    def test_all_aliases_resolve(self):
        for _key, val in _ECOSYSTEM_ALIASES.items():
            assert isinstance(val, str)

    def test_auto_resolves_empty(self):
        assert _ECOSYSTEM_ALIASES["auto"] == ""

    def test_nuget(self):
        assert _ECOSYSTEM_ALIASES["nuget"] == "NuGet"

    def test_hex(self):
        assert _ECOSYSTEM_ALIASES["hex"] == "Hex"

    def test_pub(self):
        assert _ECOSYSTEM_ALIASES["pub"] == "Pub"
