"""Comprehensive test suite for get_version_range module.

Tests cover:
- TOOL_SPEC contract validation
- Input validation and edge cases
- NVD CPE configuration parsing
- OSV.dev version range parsing
- Ecosystem filtering
- CPE-to-ecosystem inference
- Merged NVD + OSV output
- CLI subcommand dispatch
- Retry/back-off behaviour
- Error handling and graceful degradation
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.get_version_range import (
    _CPE_TO_ECOSYSTEM,
    TOOL_SPEC,
    _extract_cpe_ranges,
    _fetch_osv_record,
    _http_get_with_retry,
    _infer_ecosystem_from_cpe,
    _parse_cpe_uri,
    _parse_osv_affected,
    fetch_nvd_cpe_ranges,
    fetch_osv_version_ranges,
    fetch_version_range,
    get_version_range,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_nvd_response_log4j():
    """NVD CVE-2021-44228 (Log4Shell) CPE configurations."""
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "id": "CVE-2021-44228",
                    "configurations": [
                        {
                            "nodes": [
                                {
                                    "cpeMatch": [
                                        {
                                            "vulnerable": True,
                                            "criteria": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                                            "versionStartIncluding": "2.0",
                                            "versionEndExcluding": "2.15.0",
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                }
            }
        ]
    }


@pytest.fixture
def mock_osv_response_log4j():
    """OSV response for CVE-2021-44228."""
    return {
        "id": "CVE-2021-44228",
        "aliases": ["GHSA-jfh8-c2jp-5v3q"],
        "affected": [
            {
                "package": {"ecosystem": "Maven", "name": "org.apache.logging.log4j:log4j-core"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "2.0"},
                            {"fixed": "2.15.0"},
                        ],
                    }
                ],
                "versions": ["2.0", "2.1", "2.2", "2.3", "2.14.1"],
            }
        ],
    }


@pytest.fixture
def mock_osv_response_no_affected():
    """OSV response with no package-level affected data."""
    return {
        "id": "CVE-2024-1234",
        "aliases": ["GHSA-abcd-efgh-ijkl"],
        "affected": [],
    }


@pytest.fixture
def mock_ghsa_response():
    """GHSA alias response with package data."""
    return {
        "id": "GHSA-abcd-efgh-ijkl",
        "aliases": ["CVE-2024-1234"],
        "affected": [
            {
                "package": {"ecosystem": "PyPI", "name": "vulnerable-pkg"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "1.0.0"},
                            {"fixed": "1.2.3"},
                        ],
                    }
                ],
                "versions": ["1.0.0", "1.1.0", "1.2.0"],
            }
        ],
    }


# ===========================================================================
# TOOL_SPEC contract tests
# ===========================================================================


class TestToolSpec:
    """TOOL_SPEC contract compliance."""

    def test_has_required_keys(self):
        assert "name" in TOOL_SPEC
        assert "description" in TOOL_SPEC
        assert "inputSchema" in TOOL_SPEC

    def test_name_is_get_version_range(self):
        assert TOOL_SPEC["name"] == "get_version_range"

    def test_description_is_non_empty_string(self):
        assert isinstance(TOOL_SPEC["description"], str)
        assert len(TOOL_SPEC["description"]) > 50

    def test_input_schema_requires_cve_id(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_input_schema_has_ecosystem_property(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "ecosystem" in schema["properties"]


# ===========================================================================
# Input validation tests
# ===========================================================================


class TestInputValidation:
    """Input validation and edge cases."""

    def test_empty_cve_id_returns_error(self):
        result = fetch_version_range("")
        assert "error" in result
        assert result["nvd_ranges"] == []
        assert result["osv_packages"] == []

    def test_none_cve_id_returns_error(self):
        result = fetch_version_range(None)
        assert "error" in result

    def test_whitespace_cve_id_returns_error(self):
        result = fetch_version_range("   ")
        assert "error" in result

    def test_invalid_format_returns_error(self):
        result = fetch_version_range("not-a-cve")
        assert "error" in result
        assert "Invalid CVE ID" in result.get("summary", "")

    def test_cve_id_normalised_to_uppercase(self):
        with patch(
            "manus_agent.tools.get_version_range.fetch_nvd_cpe_ranges",
            return_value={"success": True, "cpe_ranges": [], "error": None},
        ):
            with patch(
                "manus_agent.tools.get_version_range.fetch_osv_version_ranges",
                return_value={"success": True, "packages": [], "error": None},
            ):
                result = fetch_version_range("cve-2021-44228")
                assert result["cve_id"] == "CVE-2021-44228"

    def test_cve_id_with_leading_trailing_whitespace(self):
        with patch(
            "manus_agent.tools.get_version_range.fetch_nvd_cpe_ranges",
            return_value={"success": True, "cpe_ranges": [], "error": None},
        ):
            with patch(
                "manus_agent.tools.get_version_range.fetch_osv_version_ranges",
                return_value={"success": True, "packages": [], "error": None},
            ):
                result = fetch_version_range("  CVE-2021-44228  ")
                assert result["cve_id"] == "CVE-2021-44228"


# ===========================================================================
# CPE URI parsing tests
# ===========================================================================


class TestCpeUriParsing:
    """CPE 2.3 URI parsing."""

    def test_full_cpe23_uri(self):
        cpe = "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"
        parsed = _parse_cpe_uri(cpe)
        assert parsed["part"] == "a"
        assert parsed["vendor"] == "apache"
        assert parsed["product"] == "log4j"
        assert parsed["version"] == "2.14.1"

    def test_wildcard_version(self):
        cpe = "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*"
        parsed = _parse_cpe_uri(cpe)
        assert parsed["version"] == "*"

    def test_short_cpe_uri_returns_raw(self):
        cpe = "cpe:2.3:a"
        parsed = _parse_cpe_uri(cpe)
        assert "raw" in parsed

    def test_os_type_cpe(self):
        cpe = "cpe:2.3:o:linux:linux_kernel:*:*:*:*:*:*:*:*"
        parsed = _parse_cpe_uri(cpe)
        assert parsed["part"] == "o"
        assert parsed["vendor"] == "linux"
        assert parsed["product"] == "linux_kernel"


# ===========================================================================
# NVD CPE range extraction tests
# ===========================================================================


class TestExtractCpeRanges:
    """NVD CPE configuration range extraction."""

    def test_version_start_end_including(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                                "versionStartIncluding": "1.0",
                                "versionEndIncluding": "1.5",
                            }
                        ]
                    }
                ]
            }
        ]
        ranges = _extract_cpe_ranges(configs)
        assert len(ranges) == 1
        assert ranges[0]["version_start"] == "1.0"
        assert ranges[0]["start_type"] == "including"
        assert ranges[0]["version_end"] == "1.5"
        assert ranges[0]["end_type"] == "including"

    def test_version_start_end_excluding(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                                "versionStartExcluding": "0.9",
                                "versionEndExcluding": "2.0",
                            }
                        ]
                    }
                ]
            }
        ]
        ranges = _extract_cpe_ranges(configs)
        assert len(ranges) == 1
        assert ranges[0]["version_start"] == "0.9"
        assert ranges[0]["start_type"] == "excluding"
        assert ranges[0]["version_end"] == "2.0"
        assert ranges[0]["end_type"] == "excluding"

    def test_specific_version_match(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:vendor:product:3.1.0:*:*:*:*:*:*:*",
                            }
                        ]
                    }
                ]
            }
        ]
        ranges = _extract_cpe_ranges(configs)
        assert len(ranges) == 1
        assert ranges[0]["exact_version"] == "3.1.0"

    def test_non_vulnerable_cpe_skipped(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": False,
                                "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                                "versionStartIncluding": "1.0",
                                "versionEndExcluding": "2.0",
                            }
                        ]
                    }
                ]
            }
        ]
        ranges = _extract_cpe_ranges(configs)
        assert len(ranges) == 0

    def test_multiple_nodes(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:vendor:product_a:*:*:*:*:*:*:*:*",
                                "versionEndExcluding": "1.0",
                            }
                        ]
                    },
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:vendor:product_b:*:*:*:*:*:*:*:*",
                                "versionEndExcluding": "2.0",
                            }
                        ]
                    },
                ]
            }
        ]
        ranges = _extract_cpe_ranges(configs)
        assert len(ranges) == 2

    def test_nested_child_nodes(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [],
                        "nodes": [
                            {
                                "cpeMatch": [
                                    {
                                        "vulnerable": True,
                                        "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                                        "versionEndExcluding": "3.0",
                                    }
                                ]
                            }
                        ],
                    }
                ]
            }
        ]
        ranges = _extract_cpe_ranges(configs)
        assert len(ranges) == 1
        assert ranges[0]["version_end"] == "3.0"

    def test_empty_configurations(self):
        ranges = _extract_cpe_ranges([])
        assert ranges == []

    def test_no_version_constraints(self):
        """CPE with wildcard version and no start/end should produce range entry."""
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                            }
                        ]
                    }
                ]
            }
        ]
        ranges = _extract_cpe_ranges(configs)
        assert len(ranges) == 1
        assert "version_start" not in ranges[0]
        assert "version_end" not in ranges[0]
        assert "exact_version" not in ranges[0]


# ===========================================================================
# NVD fetch tests
# ===========================================================================


class TestFetchNvdCpeRanges:
    """NVD CPE range fetching with mocked HTTP."""

    @patch("manus_agent.tools.get_version_range._http_get_with_retry")
    def test_successful_fetch(self, mock_get, mock_nvd_response_log4j):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_nvd_response_log4j
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_nvd_cpe_ranges("CVE-2021-44228")
        assert result["success"] is True
        assert len(result["cpe_ranges"]) == 1
        assert result["cpe_ranges"][0]["version_start"] == "2.0"
        assert result["cpe_ranges"][0]["version_end"] == "2.15.0"

    @patch("manus_agent.tools.get_version_range._http_get_with_retry")
    def test_404_returns_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = fetch_nvd_cpe_ranges("CVE-9999-99999")
        assert result["success"] is False
        assert "No NVD record" in result["error"]

    @patch("manus_agent.tools.get_version_range._http_get_with_retry")
    def test_network_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("timeout")

        result = fetch_nvd_cpe_ranges("CVE-2021-44228")
        assert result["success"] is False
        assert "failed" in result["error"].lower()

    @patch("manus_agent.tools.get_version_range._http_get_with_retry")
    def test_empty_vulnerabilities(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_nvd_cpe_ranges("CVE-2021-44228")
        assert result["success"] is False
        assert "No vulnerability data" in result["error"]

    @patch("manus_agent.tools.get_version_range._http_get_with_retry")
    def test_http_error_response(self, mock_get):
        import requests

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_resp

        result = fetch_nvd_cpe_ranges("CVE-2021-44228")
        assert result["success"] is False


# ===========================================================================
# OSV affected parsing tests
# ===========================================================================


class TestParseOsvAffected:
    """OSV affected entry parsing."""

    def test_basic_affected_entry(self):
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "flask"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [{"introduced": "0"}, {"fixed": "2.3.2"}],
                    }
                ],
                "versions": ["2.3.0", "2.3.1"],
            }
        ]
        packages = _parse_osv_affected(affected)
        assert len(packages) == 1
        assert packages[0]["ecosystem"] == "PyPI"
        assert packages[0]["package"] == "flask"
        assert packages[0]["first_patched"] == "2.3.2"
        assert packages[0]["affected_version_count"] == 2

    def test_multiple_ranges(self):
        affected = [
            {
                "package": {"ecosystem": "npm", "name": "lodash"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [{"introduced": "0"}, {"fixed": "4.17.21"}],
                    },
                    {
                        "type": "SEMVER",
                        "events": [{"introduced": "0"}, {"fixed": "4.17.21"}],
                    },
                ],
                "versions": [],
            }
        ]
        packages = _parse_osv_affected(affected)
        assert len(packages) == 1
        assert len(packages[0]["ranges"]) == 2

    def test_last_affected_event(self):
        affected = [
            {
                "package": {"ecosystem": "Go", "name": "golang.org/x/net"},
                "ranges": [
                    {
                        "type": "SEMVER",
                        "events": [{"introduced": "0"}, {"last_affected": "0.17.0"}],
                    }
                ],
                "versions": [],
            }
        ]
        packages = _parse_osv_affected(affected)
        assert packages[0]["ranges"][0]["last_affected"] == ["0.17.0"]

    def test_no_package_data_skipped(self):
        affected = [
            {"package": {}, "ranges": [], "versions": []},
            {"ranges": [], "versions": []},
        ]
        packages = _parse_osv_affected(affected)
        assert len(packages) == 0

    def test_non_dict_entries_skipped(self):
        affected = [None, "invalid", 42]
        packages = _parse_osv_affected(affected)
        assert len(packages) == 0

    def test_versions_capped_at_20(self):
        versions = [f"1.0.{i}" for i in range(50)]
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "pkg"},
                "ranges": [],
                "versions": versions,
            }
        ]
        packages = _parse_osv_affected(affected)
        assert packages[0]["affected_version_count"] == 50
        assert len(packages[0]["affected_versions"]) == 20

    def test_multiple_fixed_versions(self):
        affected = [
            {
                "package": {"ecosystem": "Maven", "name": "org.example:lib"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "1.0"},
                            {"fixed": "1.5.1"},
                            {"introduced": "2.0"},
                            {"fixed": "2.1.0"},
                        ],
                    }
                ],
                "versions": [],
            }
        ]
        packages = _parse_osv_affected(affected)
        assert packages[0]["first_patched"] == "1.5.1"
        assert packages[0]["all_patched_versions"] == ["1.5.1", "2.1.0"]


# ===========================================================================
# OSV fetch tests
# ===========================================================================


class TestFetchOsvVersionRanges:
    """OSV version range fetching with mocked HTTP."""

    @patch("manus_agent.tools.get_version_range._fetch_osv_record")
    def test_successful_fetch(self, mock_fetch, mock_osv_response_log4j):
        mock_fetch.return_value = mock_osv_response_log4j

        result = fetch_osv_version_ranges("CVE-2021-44228")
        assert result["success"] is True
        assert len(result["packages"]) == 1
        assert result["packages"][0]["ecosystem"] == "Maven"
        assert result["packages"][0]["first_patched"] == "2.15.0"

    @patch("manus_agent.tools.get_version_range._fetch_osv_record")
    def test_not_found_returns_error(self, mock_fetch):
        mock_fetch.return_value = None

        result = fetch_osv_version_ranges("CVE-9999-99999")
        assert result["success"] is False
        assert "No OSV record" in result["error"]

    @patch("manus_agent.tools.get_version_range._fetch_osv_record")
    def test_follows_ghsa_aliases_when_no_affected(self, mock_fetch, mock_osv_response_no_affected, mock_ghsa_response):
        """When CVE record has no packages, follow GHSA aliases."""
        mock_fetch.side_effect = [mock_osv_response_no_affected, mock_ghsa_response]

        result = fetch_osv_version_ranges("CVE-2024-1234")
        assert result["success"] is True
        assert len(result["packages"]) == 1
        assert result["packages"][0]["ecosystem"] == "PyPI"
        assert result["packages"][0]["first_patched"] == "1.2.3"

    @patch("manus_agent.tools.get_version_range._fetch_osv_record")
    def test_does_not_follow_aliases_when_packages_present(self, mock_fetch, mock_osv_response_log4j):
        """When primary record has packages, don't chase aliases."""
        mock_fetch.return_value = mock_osv_response_log4j

        fetch_osv_version_ranges("CVE-2021-44228")
        # Only called once (no alias following)
        assert mock_fetch.call_count == 1


# ===========================================================================
# Ecosystem inference tests
# ===========================================================================


class TestEcosystemInference:
    """CPE vendor/product to ecosystem mapping."""

    def test_python_vendor(self):
        assert _infer_ecosystem_from_cpe("python", "requests") == "PyPI"

    def test_npm_product(self):
        assert _infer_ecosystem_from_cpe("some_vendor", "npm") == "npm"

    def test_apache_vendor(self):
        assert _infer_ecosystem_from_cpe("apache", "log4j") == "Maven"

    def test_linux_kernel(self):
        assert _infer_ecosystem_from_cpe("linux", "linux_kernel") == "Linux"

    def test_go_vendor(self):
        assert _infer_ecosystem_from_cpe("golang", "some_module") == "Go"

    def test_rust_vendor(self):
        assert _infer_ecosystem_from_cpe("rust", "some_crate") == "crates.io"

    def test_unknown_returns_none(self):
        assert _infer_ecosystem_from_cpe("unknown_vendor", "unknown_product") is None

    def test_case_insensitive(self):
        assert _infer_ecosystem_from_cpe("PYTHON", "REQUESTS") == "PyPI"

    def test_php_packagist(self):
        assert _infer_ecosystem_from_cpe("php", "laravel") == "Packagist"


# ===========================================================================
# Core fetch_version_range integration tests
# ===========================================================================


class TestFetchVersionRange:
    """Integration tests for the main fetch_version_range function."""

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_cpe_ranges")
    def test_combined_nvd_and_osv(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "success": True,
            "cpe_ranges": [
                {
                    "vendor": "apache",
                    "product": "log4j",
                    "cpe": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                    "version_start": "2.0",
                    "start_type": "including",
                    "version_end": "2.15.0",
                    "end_type": "excluding",
                }
            ],
            "error": None,
        }
        mock_osv.return_value = {
            "success": True,
            "packages": [
                {
                    "ecosystem": "Maven",
                    "package": "org.apache.logging.log4j:log4j-core",
                    "ranges": [
                        {
                            "type": "ECOSYSTEM",
                            "introduced": ["2.0"],
                            "fixed": ["2.15.0"],
                            "last_affected": [],
                            "limit": [],
                        }
                    ],
                    "first_patched": "2.15.0",
                    "all_patched_versions": ["2.15.0"],
                    "affected_version_count": 5,
                    "affected_versions": ["2.0", "2.1", "2.14.1"],
                }
            ],
            "error": None,
        }

        result = fetch_version_range("CVE-2021-44228")
        assert result["cve_id"] == "CVE-2021-44228"
        assert len(result["nvd_ranges"]) == 1
        assert len(result["osv_packages"]) == 1
        assert result["first_patched"] == "2.15.0"
        assert "Maven" in result["ecosystems"]
        assert "CVE-2021-44228" in result["summary"]

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_cpe_ranges")
    def test_ecosystem_filter_applied(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "success": True,
            "cpe_ranges": [
                {"vendor": "apache", "product": "log4j", "cpe": "..."},
                {"vendor": "python", "product": "django", "cpe": "..."},
            ],
            "error": None,
        }
        mock_osv.return_value = {
            "success": True,
            "packages": [
                {
                    "ecosystem": "Maven",
                    "package": "log4j",
                    "ranges": [],
                    "first_patched": "2.15.0",
                    "all_patched_versions": ["2.15.0"],
                    "affected_version_count": 0,
                    "affected_versions": [],
                },
                {
                    "ecosystem": "PyPI",
                    "package": "django",
                    "ranges": [],
                    "first_patched": "4.0.1",
                    "all_patched_versions": ["4.0.1"],
                    "affected_version_count": 0,
                    "affected_versions": [],
                },
            ],
            "error": None,
        }

        result = fetch_version_range("CVE-2021-44228", ecosystem_filter="pypi")
        # Only PyPI packages remain
        assert all(p["ecosystem"] == "PyPI" for p in result["osv_packages"])
        assert result["first_patched"] == "4.0.1"

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_cpe_ranges")
    def test_no_data_from_either_source(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {"success": True, "cpe_ranges": [], "error": None}
        mock_osv.return_value = {"success": True, "packages": [], "error": None}

        result = fetch_version_range("CVE-2021-44228")
        assert result["first_patched"] is None
        assert "No affected version ranges" in result["summary"]

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_cpe_ranges")
    def test_both_sources_fail(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {"success": False, "cpe_ranges": [], "error": "NVD down"}
        mock_osv.return_value = {"success": False, "packages": [], "error": "OSV down"}

        result = fetch_version_range("CVE-2021-44228")
        assert "NVD" in result["summary"]
        assert "OSV" in result["summary"]

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_cpe_ranges")
    def test_nvd_enriches_ecosystem(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "success": True,
            "cpe_ranges": [
                {"vendor": "python", "product": "django", "cpe": "..."},
            ],
            "error": None,
        }
        mock_osv.return_value = {"success": True, "packages": [], "error": None}

        result = fetch_version_range("CVE-2021-44228")
        assert result["nvd_ranges"][0]["inferred_ecosystem"] == "PyPI"

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_cpe_ranges")
    def test_auto_ecosystem_no_filter(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "success": True,
            "cpe_ranges": [{"vendor": "x", "product": "y", "cpe": "..."}],
            "error": None,
        }
        mock_osv.return_value = {
            "success": True,
            "packages": [
                {
                    "ecosystem": "npm",
                    "package": "p",
                    "ranges": [],
                    "first_patched": None,
                    "all_patched_versions": [],
                    "affected_version_count": 0,
                    "affected_versions": [],
                }
            ],
            "error": None,
        }

        result = fetch_version_range("CVE-2021-44228", ecosystem_filter="auto")
        # auto means no filtering
        assert len(result["osv_packages"]) == 1
        assert len(result["nvd_ranges"]) == 1


# ===========================================================================
# Strands tool entry point tests
# ===========================================================================


class TestGetVersionRangeTool:
    """Strands tool handler tests."""

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_success_response(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "nvd_ranges": [{"vendor": "apache"}],
            "osv_packages": [{"ecosystem": "Maven"}],
            "ecosystems": ["Maven"],
            "first_patched": "2.15.0",
            "summary": "test",
            "nvd_error": None,
            "osv_error": None,
        }

        tool_use = {"toolUseId": "test-123", "input": {"cve_id": "CVE-2021-44228"}}
        result = get_version_range(tool_use)
        assert result["toolUseId"] == "test-123"
        assert result["status"] == "success"

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_error_when_no_data(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-9999-99999",
            "nvd_ranges": [],
            "osv_packages": [],
            "ecosystems": [],
            "first_patched": None,
            "summary": "No data",
            "nvd_error": None,
            "osv_error": None,
        }

        tool_use = {"toolUseId": "test-456", "input": {"cve_id": "CVE-9999-99999"}}
        result = get_version_range(tool_use)
        assert result["status"] == "error"

    def test_missing_cve_id(self):
        tool_use = {"toolUseId": "test-789", "input": {}}
        result = get_version_range(tool_use)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_empty_string_cve_id(self):
        tool_use = {"toolUseId": "test-000", "input": {"cve_id": ""}}
        result = get_version_range(tool_use)
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_ecosystem_param_forwarded(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "nvd_ranges": [],
            "osv_packages": [{"ecosystem": "PyPI"}],
            "ecosystems": ["PyPI"],
            "first_patched": "1.0.1",
            "summary": "test",
            "nvd_error": None,
            "osv_error": None,
        }

        tool_use = {"toolUseId": "t1", "input": {"cve_id": "CVE-2021-44228", "ecosystem": "pypi"}}
        get_version_range(tool_use)
        mock_fetch.assert_called_once_with("CVE-2021-44228", ecosystem_filter="pypi")


# ===========================================================================
# CLI subcommand tests
# ===========================================================================


class TestCliVersionRange:
    """CLI version-range subcommand tests."""

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_output(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "nvd_ranges": [
                {
                    "vendor": "apache",
                    "product": "log4j",
                    "cpe": "...",
                    "version_start": "2.0",
                    "start_type": "including",
                    "version_end": "2.15.0",
                    "end_type": "excluding",
                }
            ],
            "osv_packages": [
                {
                    "ecosystem": "Maven",
                    "package": "org.apache.logging.log4j:log4j-core",
                    "ranges": [{"introduced": ["2.0"], "fixed": ["2.15.0"], "last_affected": []}],
                    "first_patched": "2.15.0",
                    "all_patched_versions": ["2.15.0"],
                    "affected_version_count": 3,
                    "affected_versions": ["2.0", "2.1", "2.14.1"],
                }
            ],
            "ecosystems": ["Maven"],
            "first_patched": "2.15.0",
            "summary": "CVE-2021-44228: 1 NVD CPE range(s); 1 OSV package(s) [Maven]; first patched: 2.15.0.",
            "nvd_error": None,
            "osv_error": None,
        }

        from manus_agent.cli import _run_version_range

        exit_code = _run_version_range(["CVE-2021-44228"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "CVE-2021-44228" in captured.out
        assert "NVD CPE Ranges" in captured.out
        assert "OSV.dev Packages" in captured.out
        assert "2.15.0" in captured.out

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_json_output(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "nvd_ranges": [],
            "osv_packages": [],
            "ecosystems": [],
            "first_patched": None,
            "summary": "No data.",
            "nvd_error": None,
            "osv_error": None,
        }

        from manus_agent.cli import _run_version_range

        exit_code = _run_version_range(["CVE-2021-44228", "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["cve_id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_ecosystem_flag(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "nvd_ranges": [],
            "osv_packages": [],
            "ecosystems": [],
            "first_patched": None,
            "summary": "No data.",
            "nvd_error": None,
            "osv_error": None,
        }

        from manus_agent.cli import _run_version_range

        _run_version_range(["CVE-2021-44228", "--ecosystem", "pypi"])
        mock_fetch.assert_called_once_with("CVE-2021-44228", ecosystem_filter="pypi")

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_auto_ecosystem_passes_none(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "nvd_ranges": [],
            "osv_packages": [],
            "ecosystems": [],
            "first_patched": None,
            "summary": "No data.",
            "nvd_error": None,
            "osv_error": None,
        }

        from manus_agent.cli import _run_version_range

        _run_version_range(["CVE-2021-44228", "--ecosystem", "auto"])
        mock_fetch.assert_called_once_with("CVE-2021-44228", ecosystem_filter=None)

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_output_no_data(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "cve_id": "CVE-9999-99999",
            "nvd_ranges": [],
            "osv_packages": [],
            "ecosystems": [],
            "first_patched": None,
            "summary": "No data.",
            "nvd_error": "NVD request failed: timeout",
            "osv_error": "No OSV record for CVE-9999-99999",
        }

        from manus_agent.cli import _run_version_range

        exit_code = _run_version_range(["CVE-9999-99999"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "NVD" in captured.out
        assert "OSV" in captured.out

    def test_version_range_in_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "version-range" in _SUBCOMMANDS

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_output_exact_version(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "nvd_ranges": [
                {
                    "vendor": "vendor",
                    "product": "product",
                    "cpe": "...",
                    "exact_version": "3.1.0",
                }
            ],
            "osv_packages": [],
            "ecosystems": [],
            "first_patched": None,
            "summary": "CVE-2021-44228: 1 NVD CPE range(s).",
            "nvd_error": None,
            "osv_error": None,
        }

        from manus_agent.cli import _run_version_range

        exit_code = _run_version_range(["CVE-2021-44228"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "3.1.0" in captured.out

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_output_multiple_patched_versions(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "nvd_ranges": [],
            "osv_packages": [
                {
                    "ecosystem": "Maven",
                    "package": "lib",
                    "ranges": [{"introduced": ["1.0"], "fixed": ["1.5.1", "2.1.0"], "last_affected": []}],
                    "first_patched": "1.5.1",
                    "all_patched_versions": ["1.5.1", "2.1.0"],
                    "affected_version_count": 0,
                    "affected_versions": [],
                }
            ],
            "ecosystems": ["Maven"],
            "first_patched": "1.5.1",
            "summary": "CVE-2021-44228: 1 OSV package(s) [Maven]; first patched: 1.5.1.",
            "nvd_error": None,
            "osv_error": None,
        }

        from manus_agent.cli import _run_version_range

        exit_code = _run_version_range(["CVE-2021-44228"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "1.5.1" in captured.out
        assert "2.1.0" in captured.out


# ===========================================================================
# Retry/back-off tests
# ===========================================================================


class TestRetryBackoff:
    """HTTP retry/back-off behaviour."""

    @patch("manus_agent.tools.get_version_range.time.sleep")
    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retries_on_429(self, mock_get, mock_sleep):
        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429
        mock_resp_success = MagicMock()
        mock_resp_success.status_code = 200
        mock_get.side_effect = [mock_resp_429, mock_resp_success]

        with patch("manus_agent.tools.get_version_range._MAX_RETRIES", 3):
            with patch("manus_agent.tools.get_version_range._RETRY_BASE_DELAY", 0.0):
                resp = _http_get_with_retry("http://example.com")
                assert resp.status_code == 200

    @patch("manus_agent.tools.get_version_range.time.sleep")
    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retries_on_connection_error(self, mock_get, mock_sleep):
        import requests as req

        mock_resp_success = MagicMock()
        mock_resp_success.status_code = 200
        mock_get.side_effect = [req.exceptions.ConnectionError("conn"), mock_resp_success]

        with patch("manus_agent.tools.get_version_range._MAX_RETRIES", 3):
            with patch("manus_agent.tools.get_version_range._RETRY_BASE_DELAY", 0.0):
                resp = _http_get_with_retry("http://example.com")
                assert resp.status_code == 200

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_raises_after_max_retries(self, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.Timeout("timeout")

        with patch("manus_agent.tools.get_version_range._MAX_RETRIES", 2):
            with patch("manus_agent.tools.get_version_range._RETRY_BASE_DELAY", 0.0):
                with pytest.raises(req.exceptions.Timeout):
                    _http_get_with_retry("http://example.com")

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_non_retryable_status_returned_immediately(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_get.return_value = mock_resp

        resp = _http_get_with_retry("http://example.com")
        assert resp.status_code == 403
        assert mock_get.call_count == 1


# ===========================================================================
# OSV record fetch tests
# ===========================================================================


class TestFetchOsvRecord:
    """_fetch_osv_record helper tests."""

    @patch("manus_agent.tools.get_version_range._http_get_with_retry")
    def test_returns_json_on_200(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "CVE-2021-44228"}
        mock_get.return_value = mock_resp

        result = _fetch_osv_record("CVE-2021-44228")
        assert result == {"id": "CVE-2021-44228"}

    @patch("manus_agent.tools.get_version_range._http_get_with_retry")
    def test_returns_none_on_404(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = _fetch_osv_record("CVE-9999-99999")
        assert result is None

    @patch("manus_agent.tools.get_version_range._http_get_with_retry")
    def test_returns_none_on_exception(self, mock_get):
        mock_get.side_effect = Exception("network error")

        result = _fetch_osv_record("CVE-2021-44228")
        assert result is None


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_cpe_ranges")
    def test_nvd_only_data(self, mock_nvd, mock_osv):
        """Only NVD has data; OSV has nothing."""
        mock_nvd.return_value = {
            "success": True,
            "cpe_ranges": [{"vendor": "vendor", "product": "product", "cpe": "..."}],
            "error": None,
        }
        mock_osv.return_value = {"success": True, "packages": [], "error": None}

        result = fetch_version_range("CVE-2021-44228")
        assert len(result["nvd_ranges"]) == 1
        assert result["first_patched"] is None
        assert "NVD CPE range" in result["summary"]

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_cpe_ranges")
    def test_osv_only_data(self, mock_nvd, mock_osv):
        """Only OSV has data; NVD has nothing."""
        mock_nvd.return_value = {"success": True, "cpe_ranges": [], "error": None}
        mock_osv.return_value = {
            "success": True,
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "pkg",
                    "ranges": [],
                    "first_patched": "1.0.0",
                    "all_patched_versions": ["1.0.0"],
                    "affected_version_count": 0,
                    "affected_versions": [],
                }
            ],
            "error": None,
        }

        result = fetch_version_range("CVE-2021-44228")
        assert result["first_patched"] == "1.0.0"
        assert "OSV package" in result["summary"]

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_cpe_ranges")
    def test_multiple_ecosystems(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {"success": True, "cpe_ranges": [], "error": None}
        mock_osv.return_value = {
            "success": True,
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "a",
                    "ranges": [],
                    "first_patched": "1.0",
                    "all_patched_versions": ["1.0"],
                    "affected_version_count": 0,
                    "affected_versions": [],
                },
                {
                    "ecosystem": "npm",
                    "package": "b",
                    "ranges": [],
                    "first_patched": "2.0",
                    "all_patched_versions": ["2.0"],
                    "affected_version_count": 0,
                    "affected_versions": [],
                },
                {
                    "ecosystem": "Maven",
                    "package": "c",
                    "ranges": [],
                    "first_patched": "3.0",
                    "all_patched_versions": ["3.0"],
                    "affected_version_count": 0,
                    "affected_versions": [],
                },
            ],
            "error": None,
        }

        result = fetch_version_range("CVE-2021-44228")
        assert sorted(result["ecosystems"]) == ["Maven", "PyPI", "npm"]

    def test_cpe_to_ecosystem_dict_populated(self):
        """Ensure the mapping dict has key ecosystem targets."""
        assert "python" in _CPE_TO_ECOSYSTEM
        assert "npm" in _CPE_TO_ECOSYSTEM
        assert "maven" in _CPE_TO_ECOSYSTEM
        assert "go" in _CPE_TO_ECOSYSTEM
        assert "rust" in _CPE_TO_ECOSYSTEM

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_cpe_ranges")
    def test_ecosystem_filter_case_insensitive(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {"success": True, "cpe_ranges": [], "error": None}
        mock_osv.return_value = {
            "success": True,
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "a",
                    "ranges": [],
                    "first_patched": "1.0",
                    "all_patched_versions": ["1.0"],
                    "affected_version_count": 0,
                    "affected_versions": [],
                },
                {
                    "ecosystem": "npm",
                    "package": "b",
                    "ranges": [],
                    "first_patched": "2.0",
                    "all_patched_versions": ["2.0"],
                    "affected_version_count": 0,
                    "affected_versions": [],
                },
            ],
            "error": None,
        }

        result = fetch_version_range("CVE-2021-44228", ecosystem_filter="PyPI")
        # Filter should work case-insensitively
        assert all(p["ecosystem"] == "PyPI" for p in result["osv_packages"])
