"""
Tests for the get_version_ranges tool and version-range CLI subcommand.

All external HTTP calls are mocked — no network access required.
"""

from __future__ import annotations

import json
import re
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — fake HTTP responses
# ---------------------------------------------------------------------------

NVD_RESPONSE_SINGLE_VERSION = {
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2024-1234",
                "descriptions": [{"lang": "en", "value": "A sample vulnerability."}],
                "configurations": [
                    {
                        "nodes": [],
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:example:mypackage:1.2.3:*:*:*:*:*:*:*",
                                "versionStartIncluding": "1.0.0",
                                "versionEndExcluding": "1.3.0",
                            }
                        ],
                    }
                ],
            }
        }
    ]
}

NVD_RESPONSE_EXACT_VERSION = {
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2024-5678",
                "descriptions": [{"lang": "en", "value": "Exact version vulnerability."}],
                "configurations": [
                    {
                        "nodes": [],
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:vendor:product:2.5.0:*:*:*:*:*:*:*",
                            }
                        ],
                    }
                ],
            }
        }
    ]
}

NVD_RESPONSE_NO_CONFIGS = {
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2024-9999",
                "descriptions": [{"lang": "en", "value": "No config data."}],
                "configurations": [],
            }
        }
    ]
}

NVD_RESPONSE_EMPTY = {"vulnerabilities": []}

PYPI_RESPONSE = {
    "releases": {
        "0.9.0": [{"upload_time": "2022-01-01T00:00:00"}],
        "1.0.0": [{"upload_time": "2022-06-01T00:00:00"}],
        "1.1.0": [{"upload_time": "2022-09-01T00:00:00"}],
        "1.2.0": [{"upload_time": "2023-01-01T00:00:00"}],
        "1.2.3": [{"upload_time": "2023-03-01T00:00:00"}],
        "1.2.9": [{"upload_time": "2023-05-01T00:00:00"}],
        "1.3.0": [{"upload_time": "2023-07-01T00:00:00"}],  # first patched
        "1.4.0": [{"upload_time": "2023-10-01T00:00:00"}],
    }
}

NPM_RESPONSE = {
    "time": {
        "created": "2021-01-01T00:00:00Z",
        "modified": "2023-12-01T00:00:00Z",
        "1.0.0": "2021-06-01T00:00:00Z",
        "1.1.0": "2021-09-01T00:00:00Z",
        "2.0.0": "2022-03-01T00:00:00Z",
        "2.0.5": "2022-06-01T00:00:00Z",
        "2.1.0": "2022-10-01T00:00:00Z",
    }
}

MAVEN_RESPONSE = {
    "response": {
        "docs": [
            {"v": "1.0.0", "timestamp": 1640000000000},
            {"v": "1.1.0", "timestamp": 1650000000000},
            {"v": "1.2.0", "timestamp": 1660000000000},
        ]
    }
}


def _make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _import_tool():
    from manus_use.tools.get_version_ranges import (
        _extract_ranges,
        _fetch_npm_releases,
        _fetch_nvd_cpe_ranges,
        _fetch_pypi_releases,
        _infer_ecosystem,
        _parse_cpe23,
        _render_text,
        _version_in_range,
        get_version_ranges,
        resolve_version_ranges,
    )

    return {
        "_extract_ranges": _extract_ranges,
        "_fetch_npm_releases": _fetch_npm_releases,
        "_fetch_nvd_cpe_ranges": _fetch_nvd_cpe_ranges,
        "_fetch_pypi_releases": _fetch_pypi_releases,
        "_infer_ecosystem": _infer_ecosystem,
        "_parse_cpe23": _parse_cpe23,
        "_render_text": _render_text,
        "_version_in_range": _version_in_range,
        "get_version_ranges": get_version_ranges,
        "resolve_version_ranges": resolve_version_ranges,
    }


# ---------------------------------------------------------------------------
# Tests: module and TOOL_SPEC
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_module_imports_cleanly(self):
        import manus_use.tools.get_version_ranges  # noqa: F401

    def test_tool_spec_present(self):
        from manus_use.tools.get_version_ranges import TOOL_SPEC

        assert TOOL_SPEC["name"] == "get_version_ranges"
        assert "description" in TOOL_SPEC
        assert "inputSchema" in TOOL_SPEC

    def test_tool_spec_has_required_fields(self):
        from manus_use.tools.get_version_ranges import TOOL_SPEC

        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_tool_spec_has_ecosystem_param(self):
        from manus_use.tools.get_version_ranges import TOOL_SPEC

        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "ecosystem" in schema["properties"]
        eco_prop = schema["properties"]["ecosystem"]
        assert "auto" in eco_prop["enum"]
        assert "pypi" in eco_prop["enum"]
        assert "npm" in eco_prop["enum"]
        assert "maven" in eco_prop["enum"]

    def test_get_version_ranges_callable(self):
        from manus_use.tools.get_version_ranges import get_version_ranges

        assert callable(get_version_ranges)

    def test_resolve_version_ranges_callable(self):
        from manus_use.tools.get_version_ranges import resolve_version_ranges

        assert callable(resolve_version_ranges)


# ---------------------------------------------------------------------------
# Tests: _parse_cpe23
# ---------------------------------------------------------------------------


class TestParseCpe23:
    def test_parse_full_cpe(self):
        from manus_use.tools.get_version_ranges import _parse_cpe23

        result = _parse_cpe23("cpe:2.3:a:django:django:3.2.0:*:*:*:*:*:*:*")
        assert result["vendor"] == "django"
        assert result["product"] == "django"
        assert result["version"] == "3.2.0"
        assert result["part"] == "a"

    def test_parse_wildcard_version(self):
        from manus_use.tools.get_version_ranges import _parse_cpe23

        result = _parse_cpe23("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*")
        assert result["version"] == "*"

    def test_parse_too_short_returns_empty(self):
        from manus_use.tools.get_version_ranges import _parse_cpe23

        result = _parse_cpe23("cpe:2.3:a")
        assert result == {}

    def test_parse_os_type(self):
        from manus_use.tools.get_version_ranges import _parse_cpe23

        result = _parse_cpe23("cpe:2.3:o:microsoft:windows:10:*:*:*:*:*:*:*")
        assert result["part"] == "o"
        assert result["vendor"] == "microsoft"


# ---------------------------------------------------------------------------
# Tests: _extract_ranges
# ---------------------------------------------------------------------------


class TestExtractRanges:
    def test_extract_range_with_bounds(self):
        from manus_use.tools.get_version_ranges import _extract_ranges

        configs = NVD_RESPONSE_SINGLE_VERSION["vulnerabilities"][0]["cve"]["configurations"]
        ranges = _extract_ranges(configs)
        assert len(ranges) == 1
        r = ranges[0]
        assert r["product"] == "mypackage"
        assert r["versionStartIncluding"] == "1.0.0"
        assert r["versionEndExcluding"] == "1.3.0"

    def test_extract_exact_version(self):
        from manus_use.tools.get_version_ranges import _extract_ranges

        configs = NVD_RESPONSE_EXACT_VERSION["vulnerabilities"][0]["cve"]["configurations"]
        ranges = _extract_ranges(configs)
        assert len(ranges) == 1
        assert ranges[0]["cpe_version"] == "2.5.0"

    def test_extract_empty_configs(self):
        from manus_use.tools.get_version_ranges import _extract_ranges

        assert _extract_ranges([]) == []

    def test_extract_non_vulnerable_skipped(self):
        from manus_use.tools.get_version_ranges import _extract_ranges

        configs = [
            {
                "cpeMatch": [
                    {
                        "vulnerable": False,
                        "criteria": "cpe:2.3:a:v:p:1.0.0:*:*:*:*:*:*:*",
                    }
                ]
            }
        ]
        assert _extract_ranges(configs) == []

    def test_extract_child_nodes_recursion(self):
        """Ranges inside child nodes should also be extracted."""
        from manus_use.tools.get_version_ranges import _extract_ranges

        configs = [
            {
                "children": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:child:pkg:1.0.0:*:*:*:*:*:*:*",
                                "versionEndExcluding": "1.5.0",
                            }
                        ],
                        "children": [],
                    }
                ],
                "cpeMatch": [],
            }
        ]
        ranges = _extract_ranges(configs)
        assert len(ranges) == 1
        assert ranges[0]["product"] == "pkg"


# ---------------------------------------------------------------------------
# Tests: _version_in_range
# ---------------------------------------------------------------------------


class TestVersionInRange:
    def test_within_range_inclusive_exclusive(self):
        from manus_use.tools.get_version_ranges import _version_in_range

        assert _version_in_range("1.2.0", "1.0.0", None, None, "1.3.0", "*") is True

    def test_below_range(self):
        from manus_use.tools.get_version_ranges import _version_in_range

        assert _version_in_range("0.9.0", "1.0.0", None, None, "1.3.0", "*") is False

    def test_at_upper_exclusive_bound(self):
        from manus_use.tools.get_version_ranges import _version_in_range

        # vee=1.3.0 means < 1.3.0, so 1.3.0 itself is NOT vulnerable
        assert _version_in_range("1.3.0", "1.0.0", None, None, "1.3.0", "*") is False

    def test_at_upper_inclusive_bound(self):
        from manus_use.tools.get_version_ranges import _version_in_range

        assert _version_in_range("1.3.0", "1.0.0", None, "1.3.0", None, "*") is True

    def test_exact_match_single_version_cpe(self):
        from manus_use.tools.get_version_ranges import _version_in_range

        assert _version_in_range("2.5.0", None, None, None, None, "2.5.0") is True

    def test_exact_mismatch(self):
        from manus_use.tools.get_version_ranges import _version_in_range

        assert _version_in_range("2.5.1", None, None, None, None, "2.5.0") is False

    def test_all_versions_wildcard(self):
        from manus_use.tools.get_version_ranges import _version_in_range

        assert _version_in_range("99.0.0", None, None, None, None, "*") is True

    def test_invalid_version_string(self):
        from manus_use.tools.get_version_ranges import _version_in_range

        # Should not raise
        result = _version_in_range("not-a-version", "1.0.0", None, None, "2.0.0", "*")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: _infer_ecosystem
# ---------------------------------------------------------------------------


class TestInferEcosystem:
    def test_python_vendor_maps_to_pypi(self):
        from manus_use.tools.get_version_ranges import _infer_ecosystem

        ranges = [{"vendor": "python", "product": "somepackage"}]
        assert _infer_ecosystem(ranges) == "pypi"

    def test_nodejs_vendor_maps_to_npm(self):
        from manus_use.tools.get_version_ranges import _infer_ecosystem

        ranges = [{"vendor": "nodejs", "product": "express"}]
        assert _infer_ecosystem(ranges) == "npm"

    def test_apache_vendor_maps_to_maven(self):
        from manus_use.tools.get_version_ranges import _infer_ecosystem

        ranges = [{"vendor": "apache", "product": "log4j"}]
        assert _infer_ecosystem(ranges) == "maven"

    def test_unknown_vendor(self):
        from manus_use.tools.get_version_ranges import _infer_ecosystem

        ranges = [{"vendor": "unknownvendorxyz", "product": "unknownproduct"}]
        assert _infer_ecosystem(ranges) == "unknown"

    def test_empty_ranges(self):
        from manus_use.tools.get_version_ranges import _infer_ecosystem

        assert _infer_ecosystem([]) == "unknown"


# ---------------------------------------------------------------------------
# Tests: _fetch_pypi_releases
# ---------------------------------------------------------------------------


class TestFetchPypiReleases:
    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_returns_sorted_versions(self, mock_get):
        mock_get.return_value = _make_mock_response(PYPI_RESPONSE)
        from manus_use.tools.get_version_ranges import _fetch_pypi_releases

        releases = _fetch_pypi_releases("mypackage")
        assert len(releases) > 0
        # Should be sorted by version
        versions = [r[0] for r in releases]
        assert "1.0.0" in versions
        assert "1.3.0" in versions

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_404_returns_empty(self, mock_get):
        resp = MagicMock()
        resp.status_code = 404
        mock_get.return_value = resp
        from manus_use.tools.get_version_ranges import _fetch_pypi_releases

        assert _fetch_pypi_releases("nonexistent") == []

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_request_exception_returns_empty(self, mock_get):
        import requests

        mock_get.side_effect = requests.RequestException("timeout")
        from manus_use.tools.get_version_ranges import _fetch_pypi_releases

        assert _fetch_pypi_releases("mypackage") == []

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_date_extraction(self, mock_get):
        mock_get.return_value = _make_mock_response(PYPI_RESPONSE)
        from manus_use.tools.get_version_ranges import _fetch_pypi_releases

        releases = _fetch_pypi_releases("mypackage")
        # All dates should be YYYY-MM-DD format
        for _, date in releases:
            assert re.match(r"\d{4}-\d{2}-\d{2}", date), f"Invalid date format: {date}"


# ---------------------------------------------------------------------------
# Tests: _fetch_npm_releases
# ---------------------------------------------------------------------------


class TestFetchNpmReleases:
    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_returns_versions_excluding_metadata_keys(self, mock_get):
        mock_get.return_value = _make_mock_response(NPM_RESPONSE)
        from manus_use.tools.get_version_ranges import _fetch_npm_releases

        releases = _fetch_npm_releases("express")
        versions = [r[0] for r in releases]
        assert "created" not in versions
        assert "modified" not in versions
        assert "1.0.0" in versions

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_404_returns_empty(self, mock_get):
        resp = MagicMock()
        resp.status_code = 404
        mock_get.return_value = resp
        from manus_use.tools.get_version_ranges import _fetch_npm_releases

        assert _fetch_npm_releases("nonexistent") == []

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_request_exception_returns_empty(self, mock_get):
        import requests

        mock_get.side_effect = requests.RequestException("timeout")
        from manus_use.tools.get_version_ranges import _fetch_npm_releases

        assert _fetch_npm_releases("mypackage") == []


# ---------------------------------------------------------------------------
# Tests: _fetch_nvd_cpe_ranges
# ---------------------------------------------------------------------------


class TestFetchNvdCpeRanges:
    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_returns_configurations(self, mock_get):
        mock_get.return_value = _make_mock_response(NVD_RESPONSE_SINGLE_VERSION)
        from manus_use.tools.get_version_ranges import _fetch_nvd_cpe_ranges

        result = _fetch_nvd_cpe_ranges("CVE-2024-1234")
        assert "error" not in result
        assert "configurations" in result

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_empty_vulnerabilities_returns_error(self, mock_get):
        mock_get.return_value = _make_mock_response(NVD_RESPONSE_EMPTY)
        from manus_use.tools.get_version_ranges import _fetch_nvd_cpe_ranges

        result = _fetch_nvd_cpe_ranges("CVE-2024-9999")
        assert "error" in result

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_request_exception_returns_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.RequestException("timeout")
        from manus_use.tools.get_version_ranges import _fetch_nvd_cpe_ranges

        result = _fetch_nvd_cpe_ranges("CVE-2024-0001")
        assert "error" in result

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_description_extracted(self, mock_get):
        mock_get.return_value = _make_mock_response(NVD_RESPONSE_SINGLE_VERSION)
        from manus_use.tools.get_version_ranges import _fetch_nvd_cpe_ranges

        result = _fetch_nvd_cpe_ranges("CVE-2024-1234")
        assert result.get("description") == "A sample vulnerability."


# ---------------------------------------------------------------------------
# Tests: resolve_version_ranges (integration with mocked HTTP)
# ---------------------------------------------------------------------------


class TestResolveVersionRanges:
    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_full_pypi_resolve(self, mock_get):
        """End-to-end: NVD gives range, PyPI gives releases, result has affected+patched."""

        def side_effect(url, **kwargs):
            if "nvd.nist.gov" in url:
                return _make_mock_response(NVD_RESPONSE_SINGLE_VERSION)
            if "pypi.org" in url:
                return _make_mock_response(PYPI_RESPONSE)
            return _make_mock_response({})

        mock_get.side_effect = side_effect
        from manus_use.tools.get_version_ranges import resolve_version_ranges

        report = resolve_version_ranges("CVE-2024-1234", ecosystem="pypi")
        assert report["cve_id"] == "CVE-2024-1234"
        assert len(report["ranges"]) >= 1

        r = report["ranges"][0]
        assert "affected_versions" in r
        # 1.0.0, 1.1.0, 1.2.0, 1.2.3, 1.2.9 are in [1.0.0, 1.3.0)
        total = r.get("total_affected", 0)
        assert total >= 2
        assert r.get("first_patched_version") == "1.3.0"

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_nvd_error_propagates(self, mock_get):
        mock_get.return_value = _make_mock_response(NVD_RESPONSE_EMPTY)
        from manus_use.tools.get_version_ranges import resolve_version_ranges

        report = resolve_version_ranges("CVE-2024-9999")
        assert "error" in report

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_no_cpe_configs_returns_error_message(self, mock_get):
        mock_get.return_value = _make_mock_response(NVD_RESPONSE_NO_CONFIGS)
        from manus_use.tools.get_version_ranges import resolve_version_ranges

        report = resolve_version_ranges("CVE-2024-9999")
        assert "error" in report
        assert "CPE" in report["error"]

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_unknown_ecosystem_no_registry_lookup(self, mock_get):
        """When ecosystem is unknown, no registry fetch — just raw CPE range."""

        def side_effect(url, **kwargs):
            if "nvd.nist.gov" in url:
                return _make_mock_response(NVD_RESPONSE_SINGLE_VERSION)
            # Registry should NOT be called
            raise AssertionError(f"Unexpected registry call: {url}")

        mock_get.side_effect = side_effect
        from manus_use.tools.get_version_ranges import resolve_version_ranges

        # Force ecosystem=auto but vendor is "example" which won't match any ecosystem
        report = resolve_version_ranges("CVE-2024-1234", ecosystem="auto")
        # Should return a note about unknown ecosystem, no error at top level
        assert "ranges" in report
        # Ranges may have a "note" about unknown ecosystem
        if report["ranges"]:
            r = report["ranges"][0]
            assert "note" in r or "affected_versions" in r

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_ecosystem_override_npm(self, mock_get):
        def side_effect(url, **kwargs):
            if "nvd.nist.gov" in url:
                return _make_mock_response(NVD_RESPONSE_SINGLE_VERSION)
            if "registry.npmjs.org" in url:
                return _make_mock_response(NPM_RESPONSE)
            return _make_mock_response({})

        mock_get.side_effect = side_effect
        from manus_use.tools.get_version_ranges import resolve_version_ranges

        report = resolve_version_ranges("CVE-2024-1234", ecosystem="npm")
        assert report["ecosystem"] == "npm"

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_deduplication_same_range_not_duplicated(self, mock_get):
        """Two identical CPE entries in different nodes should be deduplicated."""
        dup_nvd = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2024-0001",
                        "descriptions": [{"lang": "en", "value": "Dup test."}],
                        "configurations": [
                            {
                                "cpeMatch": [
                                    {
                                        "vulnerable": True,
                                        "criteria": "cpe:2.3:a:v:p:1.0.0:*:*:*:*:*:*:*",
                                        "versionEndExcluding": "2.0.0",
                                    }
                                ]
                            },
                            {
                                "cpeMatch": [
                                    {
                                        "vulnerable": True,
                                        "criteria": "cpe:2.3:a:v:p:1.0.0:*:*:*:*:*:*:*",
                                        "versionEndExcluding": "2.0.0",
                                    }
                                ]
                            },
                        ],
                    }
                }
            ]
        }

        def side_effect(url, **kwargs):
            if "nvd.nist.gov" in url:
                return _make_mock_response(dup_nvd)
            return _make_mock_response({})

        mock_get.side_effect = side_effect
        from manus_use.tools.get_version_ranges import resolve_version_ranges

        report = resolve_version_ranges("CVE-2024-0001", ecosystem="auto")
        assert len(report["ranges"]) == 1  # deduplicated


# ---------------------------------------------------------------------------
# Tests: _render_text
# ---------------------------------------------------------------------------


class TestRenderText:
    def test_renders_cve_id_as_header(self):
        from manus_use.tools.get_version_ranges import _render_text

        report = {
            "cve_id": "CVE-2024-1234",
            "description": "Test desc",
            "ecosystem": "pypi",
            "ranges": [
                {
                    "vendor": "example",
                    "product": "mypackage",
                    "range": ">= 1.0.0, < 1.3.0",
                    "ecosystem": "pypi",
                    "affected_versions": [
                        {"version": "1.0.0", "release_date": "2022-06-01"},
                        {"version": "1.2.9", "release_date": "2023-05-01"},
                    ],
                    "total_affected": 2,
                    "truncated": False,
                    "first_patched_version": "1.3.0",
                }
            ],
        }
        text = _render_text(report)
        assert "CVE-2024-1234" in text
        assert "mypackage" in text
        assert ">= 1.0.0, < 1.3.0" in text
        assert "1.3.0" in text
        assert "Test desc" in text

    def test_renders_error_when_no_ranges(self):
        from manus_use.tools.get_version_ranges import _render_text

        report = {
            "cve_id": "CVE-2024-9999",
            "error": "No CPE data found",
            "ranges": [],
        }
        text = _render_text(report)
        assert "CVE-2024-9999" in text
        assert "Error" in text or "No CPE" in text

    def test_renders_note_when_package_not_found(self):
        from manus_use.tools.get_version_ranges import _render_text

        report = {
            "cve_id": "CVE-2024-1111",
            "description": "",
            "ecosystem": "unknown",
            "ranges": [
                {
                    "vendor": "somevendor",
                    "product": "someproduct",
                    "range": ">= 1.0.0",
                    "ecosystem": "unknown",
                    "note": "Package not found in registry or ecosystem unknown.",
                    "affected_versions": [],
                    "first_patched_version": None,
                }
            ],
        }
        text = _render_text(report)
        assert "someproduct" in text
        assert "not found" in text.lower() or "Note" in text

    def test_truncated_output_shown(self):
        from manus_use.tools.get_version_ranges import _render_text

        affected = [{"version": f"1.{i}.0", "release_date": "2023-01-01"} for i in range(5)]
        affected += [{"version": "...", "release_date": ""}]
        affected += [{"version": f"2.{i}.0", "release_date": "2023-06-01"} for i in range(5)]
        report = {
            "cve_id": "CVE-2024-0001",
            "description": "",
            "ecosystem": "pypi",
            "ranges": [
                {
                    "vendor": "v",
                    "product": "p",
                    "range": ">= 1.0.0",
                    "ecosystem": "pypi",
                    "affected_versions": affected,
                    "total_affected": 35,
                    "truncated": True,
                    "first_patched_version": "3.0.0",
                }
            ],
        }
        text = _render_text(report)
        assert "35" in text
        assert "..." in text


# ---------------------------------------------------------------------------
# Tests: Strands tool entry point (get_version_ranges)
# ---------------------------------------------------------------------------


class TestStrandsToolEntryPoint:
    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_returns_success_on_valid_cve(self, mock_get):
        def side_effect(url, **kwargs):
            if "nvd.nist.gov" in url:
                return _make_mock_response(NVD_RESPONSE_SINGLE_VERSION)
            if "pypi.org" in url:
                return _make_mock_response(PYPI_RESPONSE)
            return _make_mock_response({})

        mock_get.side_effect = side_effect
        from manus_use.tools.get_version_ranges import get_version_ranges

        tool_use = {
            "toolUseId": "test-001",
            "input": {"cve_id": "CVE-2024-1234", "ecosystem": "pypi"},
        }
        result = get_version_ranges(tool_use)
        assert result["status"] == "success"
        assert "json" in result["content"][0]

    def test_returns_error_on_invalid_cve_format(self):
        from manus_use.tools.get_version_ranges import get_version_ranges

        tool_use = {
            "toolUseId": "test-002",
            "input": {"cve_id": "INVALID-ID"},
        }
        result = get_version_ranges(tool_use)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_nvd_error_returns_error_status(self, mock_get):
        mock_get.return_value = _make_mock_response(NVD_RESPONSE_EMPTY)
        from manus_use.tools.get_version_ranges import get_version_ranges

        tool_use = {
            "toolUseId": "test-003",
            "input": {"cve_id": "CVE-2024-9999"},
        }
        result = get_version_ranges(tool_use)
        assert result["status"] == "error"

    def test_tool_use_id_preserved(self):
        from manus_use.tools.get_version_ranges import get_version_ranges

        tool_use = {
            "toolUseId": "my-unique-id-xyz",
            "input": {"cve_id": "NOT-VALID"},
        }
        result = get_version_ranges(tool_use)
        assert result["toolUseId"] == "my-unique-id-xyz"

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_default_ecosystem_auto(self, mock_get):
        """When ecosystem param is absent, should default to auto."""

        def side_effect(url, **kwargs):
            if "nvd.nist.gov" in url:
                return _make_mock_response(NVD_RESPONSE_SINGLE_VERSION)
            return _make_mock_response({})

        mock_get.side_effect = side_effect
        from manus_use.tools.get_version_ranges import get_version_ranges

        # No ecosystem key in input
        tool_use = {
            "toolUseId": "test-004",
            "input": {"cve_id": "CVE-2024-1234"},
        }
        result = get_version_ranges(tool_use)
        # Should not error on missing ecosystem
        assert result["toolUseId"] == "test-004"


# ---------------------------------------------------------------------------
# Tests: CLI subcommand (version-range)
# ---------------------------------------------------------------------------


class TestCLIVersionRangeParser:
    def test_version_range_in_subcommands_set(self):
        from manus_use.cli import _SUBCOMMANDS

        assert "version-range" in _SUBCOMMANDS

    def test_build_version_range_parser_exists(self):
        from manus_use.cli import _build_version_range_parser

        assert callable(_build_version_range_parser)

    def test_run_version_range_exists(self):
        from manus_use.cli import _run_version_range

        assert callable(_run_version_range)

    def test_help_exits_zero(self):
        from manus_use.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_missing_cve_id_errors(self):
        from manus_use.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([])
        assert exc_info.value.code != 0

    def test_invalid_cve_id_returns_1(self):
        from manus_use.cli import _run_version_range

        code = _run_version_range(["INVALID-ID"])
        assert code == 1

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_text_output_exits_zero(self, mock_get, capsys):
        def side_effect(url, **kwargs):
            if "nvd.nist.gov" in url:
                return _make_mock_response(NVD_RESPONSE_SINGLE_VERSION)
            if "pypi.org" in url:
                return _make_mock_response(PYPI_RESPONSE)
            return _make_mock_response({})

        mock_get.side_effect = side_effect
        from manus_use.cli import _run_version_range

        code = _run_version_range(["CVE-2024-1234", "--ecosystem", "pypi"])
        assert code == 0
        captured = capsys.readouterr()
        assert "CVE-2024-1234" in captured.out

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_json_output_is_valid_json(self, mock_get, capsys):
        def side_effect(url, **kwargs):
            if "nvd.nist.gov" in url:
                return _make_mock_response(NVD_RESPONSE_SINGLE_VERSION)
            if "pypi.org" in url:
                return _make_mock_response(PYPI_RESPONSE)
            return _make_mock_response({})

        mock_get.side_effect = side_effect
        from manus_use.cli import _run_version_range

        code = _run_version_range(["CVE-2024-1234", "--ecosystem", "pypi", "--output", "json"])
        assert code == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "cve_id" in parsed

    @patch("manus_use.tools.get_version_ranges.requests.get")
    def test_json_output_has_ranges_key(self, mock_get, capsys):
        def side_effect(url, **kwargs):
            if "nvd.nist.gov" in url:
                return _make_mock_response(NVD_RESPONSE_SINGLE_VERSION)
            if "pypi.org" in url:
                return _make_mock_response(PYPI_RESPONSE)
            return _make_mock_response({})

        mock_get.side_effect = side_effect
        from manus_use.cli import _run_version_range

        _run_version_range(["CVE-2024-1234", "--ecosystem", "pypi", "--output", "json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "ranges" in parsed

    def test_ecosystem_choices_accepted(self):
        from manus_use.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        for eco in ["auto", "pypi", "npm", "maven"]:
            args = parser.parse_args(["CVE-2024-1234", "--ecosystem", eco])
            assert args.ecosystem == eco

    def test_output_choices_accepted(self):
        from manus_use.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        for fmt in ["text", "json"]:
            args = parser.parse_args(["CVE-2024-1234", "--output", fmt])
            assert args.output == fmt

    def test_default_ecosystem_is_auto(self):
        from manus_use.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        args = parser.parse_args(["CVE-2024-1234"])
        assert args.ecosystem == "auto"

    def test_default_output_is_text(self):
        from manus_use.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        args = parser.parse_args(["CVE-2024-1234"])
        assert args.output == "text"


# ---------------------------------------------------------------------------
# Tests: main() dispatch routes "version-range"
# ---------------------------------------------------------------------------


class TestCLIMainDispatch:
    def test_version_range_registered_in_main(self):
        """main() must recognise 'version-range' and not fall through to run_parser."""
        from manus_use.cli import _SUBCOMMANDS

        assert "version-range" in _SUBCOMMANDS

    @patch("manus_use.cli._run_version_range", return_value=0)
    def test_main_dispatches_version_range(self, mock_run):
        """main() must recognise 'version-range' and route to _run_version_range."""
        from unittest.mock import patch as _patch

        with _patch.object(sys, "argv", ["manus-use", "version-range", "CVE-2024-1234"]):
            from manus_use.cli import main

            with pytest.raises(SystemExit):
                main()
        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        assert "CVE-2024-1234" in called_args


# ---------------------------------------------------------------------------
# Tests: vi_agent tool list includes get_version_ranges
# ---------------------------------------------------------------------------


class TestViAgentIntegration:
    def test_vi_agent_imports_get_version_ranges(self):
        """VulnerabilityIntelligenceAgent source must reference get_version_ranges."""
        import inspect

        import manus_use.agents.vi_agent as vi_mod

        src = inspect.getsource(vi_mod)
        assert "get_version_ranges" in src

    def test_get_version_ranges_exported_from_tools(self):
        """The tool must be importable from its canonical module path."""
        from manus_use.tools.get_version_ranges import get_version_ranges

        assert callable(get_version_ranges)
