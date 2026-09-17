#!/usr/bin/env python3
"""Comprehensive test suite for get_version_range tool and version-range CLI subcommand.

All HTTP calls are mocked — no real network traffic.
"""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure retry/backoff is instant in tests
# ---------------------------------------------------------------------------
os.environ.setdefault("VERSION_RANGE_MAX_RETRIES", "1")
os.environ.setdefault("VERSION_RANGE_RETRY_BASE_DELAY", "0")

from manus_agent.tools.get_version_range import (
    TOOL_SPEC,
    VALID_ECOSYSTEMS,
    _cpe_to_ecosystem,
    _get_with_retry,
    _normalise_ecosystem,
    _parse_nvd_configurations,
    _parse_osv_affected,
    _pick_first_patched,
    fetch_nvd_version_ranges,
    fetch_osv_version_ranges,
    fetch_version_range,
    get_version_range,
)

# ===================================================================
# Fixtures
# ===================================================================


def _make_tool_use(cve_id: str, ecosystem: str | None = None) -> dict:
    inp: dict[str, Any] = {"cve_id": cve_id}
    if ecosystem is not None:
        inp["ecosystem"] = ecosystem
    return {"toolUseId": "test-123", "input": inp}


def _mock_response(status_code: int = 200, json_data: Any = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests

        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"HTTP {status_code}", response=resp)
    return resp


def _nvd_cve_response(
    cve_id: str = "CVE-2021-44228",
    configurations: list[dict] | None = None,
    description: str = "Apache Log4j2 RCE vulnerability",
) -> dict:
    """Build a minimal NVD API response."""
    if configurations is None:
        configurations = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "criteria": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                                "vulnerable": True,
                                "versionStartIncluding": "2.0.0",
                                "versionEndExcluding": "2.15.0",
                            }
                        ]
                    }
                ]
            }
        ]
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "id": cve_id,
                    "descriptions": [{"lang": "en", "value": description}],
                    "configurations": configurations,
                }
            }
        ]
    }


def _osv_record(
    osv_id: str = "CVE-2021-44228",
    aliases: list[str] | None = None,
    affected: list[dict] | None = None,
) -> dict:
    """Build a minimal OSV.dev record."""
    if aliases is None:
        aliases = ["GHSA-jfh8-c2jp-5v3q"]
    if affected is None:
        affected = [
            {
                "package": {"ecosystem": "Maven", "name": "org.apache.logging.log4j:log4j-core"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "2.0.0"},
                            {"fixed": "2.15.0"},
                        ],
                    }
                ],
                "versions": ["2.0", "2.1", "2.2", "2.3", "2.14.1"],
            }
        ]
    return {"id": osv_id, "aliases": aliases, "affected": affected}


# ===================================================================
# TOOL_SPEC tests
# ===================================================================


class TestToolSpec:
    def test_tool_spec_name(self):
        assert TOOL_SPEC["name"] == "get_version_range"

    def test_tool_spec_has_description(self):
        assert "version" in TOOL_SPEC["description"].lower()

    def test_tool_spec_requires_cve_id(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_tool_spec_optional_ecosystem(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "ecosystem" in schema["properties"]
        assert "ecosystem" not in schema["required"]


# ===================================================================
# _normalise_ecosystem tests
# ===================================================================


class TestNormaliseEcosystem:
    def test_empty(self):
        assert _normalise_ecosystem("") == ""
        assert _normalise_ecosystem(None) == ""

    def test_pypi_variants(self):
        assert _normalise_ecosystem("pypi") == "PyPI"
        assert _normalise_ecosystem("pip") == "PyPI"
        assert _normalise_ecosystem("python") == "PyPI"
        assert _normalise_ecosystem("PyPI") == "PyPI"

    def test_npm(self):
        assert _normalise_ecosystem("npm") == "npm"
        assert _normalise_ecosystem("node") == "npm"

    def test_maven(self):
        assert _normalise_ecosystem("maven") == "Maven"
        assert _normalise_ecosystem("java") == "Maven"

    def test_go(self):
        assert _normalise_ecosystem("go") == "Go"
        assert _normalise_ecosystem("golang") == "Go"

    def test_crates(self):
        assert _normalise_ecosystem("crates.io") == "crates.io"
        assert _normalise_ecosystem("rust") == "crates.io"

    def test_unknown_passthrough(self):
        assert _normalise_ecosystem("UnknownEco") == "UnknownEco"

    def test_whitespace(self):
        assert _normalise_ecosystem("  pypi  ") == "PyPI"


# ===================================================================
# _cpe_to_ecosystem tests
# ===================================================================


class TestCpeToEcosystem:
    def test_python_vendor(self):
        assert _cpe_to_ecosystem("cpe:2.3:a:python:requests:*:*:*:*:*:*:*:*") == "PyPI"

    def test_djangoproject(self):
        assert _cpe_to_ecosystem("cpe:2.3:a:djangoproject:django:*:*:*:*:*:*:*:*") == "PyPI"

    def test_apache_vendor(self):
        assert _cpe_to_ecosystem("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*") == "Maven"

    def test_npmjs_vendor(self):
        assert _cpe_to_ecosystem("cpe:2.3:a:npmjs:express:*:*:*:*:*:*:*:*") == "npm"

    def test_linux_vendor(self):
        assert _cpe_to_ecosystem("cpe:2.3:a:linux:kernel:*:*:*:*:*:*:*:*") == "Linux"

    def test_unknown_vendor(self):
        assert _cpe_to_ecosystem("cpe:2.3:a:foobar:baz:*:*:*:*:*:*:*:*") == ""

    def test_short_cpe(self):
        assert _cpe_to_ecosystem("cpe:2.3:a") == ""

    def test_golang(self):
        assert _cpe_to_ecosystem("cpe:2.3:a:golang:go:*:*:*:*:*:*:*:*") == "Go"


# ===================================================================
# _parse_nvd_configurations tests
# ===================================================================


class TestParseNvdConfigurations:
    def test_empty_configurations(self):
        assert _parse_nvd_configurations([]) == []
        assert _parse_nvd_configurations(None) == []

    def test_basic_range(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "criteria": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                                "vulnerable": True,
                                "versionStartIncluding": "2.0.0",
                                "versionEndExcluding": "2.15.0",
                            }
                        ]
                    }
                ]
            }
        ]
        result = _parse_nvd_configurations(configs)
        assert len(result) == 1
        r = result[0]
        assert r["vendor"] == "apache"
        assert r["product"] == "log4j"
        assert r["versionStartIncluding"] == "2.0.0"
        assert r["versionEndExcluding"] == "2.15.0"
        assert r["vulnerable"] is True
        assert ">= 2.0.0" in r["range_summary"]
        assert "< 2.15.0" in r["range_summary"]

    def test_exact_version(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "criteria": "cpe:2.3:a:vendor:product:1.2.3:*:*:*:*:*:*:*",
                                "vulnerable": True,
                            }
                        ]
                    }
                ]
            }
        ]
        result = _parse_nvd_configurations(configs)
        assert len(result) == 1
        assert result[0]["version_exact"] == "1.2.3"
        assert result[0]["range_summary"] == "= 1.2.3"

    def test_wildcard_version(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                                "vulnerable": True,
                            }
                        ]
                    }
                ]
            }
        ]
        result = _parse_nvd_configurations(configs)
        assert result[0]["range_summary"] == "all versions"
        assert result[0]["version_exact"] is None

    def test_end_including(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                                "vulnerable": True,
                                "versionEndIncluding": "3.0.0",
                            }
                        ]
                    }
                ]
            }
        ]
        result = _parse_nvd_configurations(configs)
        assert "<= 3.0.0" in result[0]["range_summary"]

    def test_start_excluding(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                                "vulnerable": True,
                                "versionStartExcluding": "1.0.0",
                            }
                        ]
                    }
                ]
            }
        ]
        result = _parse_nvd_configurations(configs)
        assert "> 1.0.0" in result[0]["range_summary"]

    def test_multiple_nodes(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "criteria": "cpe:2.3:a:vendor:prod_a:*:*:*:*:*:*:*:*",
                                "vulnerable": True,
                                "versionEndExcluding": "1.0",
                            },
                            {
                                "criteria": "cpe:2.3:a:vendor:prod_b:*:*:*:*:*:*:*:*",
                                "vulnerable": True,
                                "versionEndExcluding": "2.0",
                            },
                        ]
                    }
                ]
            }
        ]
        result = _parse_nvd_configurations(configs)
        assert len(result) == 2

    def test_non_vulnerable_match(self):
        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                                "vulnerable": False,
                            }
                        ]
                    }
                ]
            }
        ]
        result = _parse_nvd_configurations(configs)
        assert result[0]["vulnerable"] is False

    def test_non_dict_cpe_match_skipped(self):
        configs = [{"nodes": [{"cpeMatch": ["not-a-dict", None]}]}]
        result = _parse_nvd_configurations(configs)
        assert result == []


# ===================================================================
# _parse_osv_affected tests
# ===================================================================


class TestParseOsvAffected:
    def test_empty(self):
        assert _parse_osv_affected([]) == []
        assert _parse_osv_affected(None) == []

    def test_basic_package(self):
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "django"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "3.0"},
                            {"fixed": "3.2.4"},
                        ],
                    }
                ],
                "versions": ["3.0", "3.0.1", "3.1", "3.2", "3.2.3"],
            }
        ]
        result = _parse_osv_affected(affected)
        assert len(result) == 1
        r = result[0]
        assert r["ecosystem"] == "PyPI"
        assert r["package"] == "django"
        assert r["introduced"] == ["3.0"]
        assert r["fixed"] == ["3.2.4"]
        assert r["first_patched_version"] == "3.2.4"
        assert r["affected_version_count"] == 5
        assert ">= 3.0" in r["range_summary"]
        assert "< 3.2.4" in r["range_summary"]

    def test_introduced_from_zero(self):
        affected = [
            {
                "package": {"ecosystem": "npm", "name": "lodash"},
                "ranges": [
                    {
                        "type": "SEMVER",
                        "events": [
                            {"introduced": "0"},
                            {"fixed": "4.17.21"},
                        ],
                    }
                ],
            }
        ]
        result = _parse_osv_affected(affected)
        assert result[0]["range_summary"] == "< 4.17.21"

    def test_last_affected(self):
        affected = [
            {
                "package": {"ecosystem": "Go", "name": "github.com/foo/bar"},
                "ranges": [
                    {
                        "type": "SEMVER",
                        "events": [
                            {"introduced": "1.0.0"},
                            {"last_affected": "1.5.0"},
                        ],
                    }
                ],
            }
        ]
        result = _parse_osv_affected(affected)
        assert result[0]["last_affected"] == ["1.5.0"]
        assert "<= 1.5.0" in result[0]["range_summary"]

    def test_no_fixed_version(self):
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "vulnerable-pkg"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [{"introduced": "1.0"}],
                    }
                ],
            }
        ]
        result = _parse_osv_affected(affected)
        assert result[0]["first_patched_version"] is None
        assert result[0]["range_summary"] == ">= 1.0"

    def test_multiple_ranges(self):
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "flask"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "0.1"},
                            {"fixed": "1.0"},
                        ],
                    },
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "2.0"},
                            {"fixed": "2.1"},
                        ],
                    },
                ],
            }
        ]
        result = _parse_osv_affected(affected)
        assert len(result[0]["introduced"]) == 2
        assert len(result[0]["fixed"]) == 2

    def test_versions_capped_at_50(self):
        affected = [
            {
                "package": {"ecosystem": "npm", "name": "big-pkg"},
                "ranges": [],
                "versions": [f"{i}.0.0" for i in range(100)],
            }
        ]
        result = _parse_osv_affected(affected)
        assert result[0]["affected_version_count"] == 100
        assert len(result[0]["affected_versions"]) == 50

    def test_non_dict_entries_skipped(self):
        assert _parse_osv_affected(["not-a-dict", None, 42]) == []

    def test_no_package_info_skipped(self):
        affected = [{"package": {"ecosystem": None, "name": None}}]
        result = _parse_osv_affected(affected)
        assert result == []

    def test_non_dict_range_events_skipped(self):
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "pkg"},
                "ranges": [{"type": "ECOSYSTEM", "events": ["bad", None]}],
            }
        ]
        result = _parse_osv_affected(affected)
        assert len(result) == 1
        assert result[0]["introduced"] == []

    def test_non_dict_ranges_skipped(self):
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "pkg"},
                "ranges": ["bad", None, 42],
            }
        ]
        result = _parse_osv_affected(affected)
        assert len(result) == 1
        assert result[0]["introduced"] == []


# ===================================================================
# _pick_first_patched tests
# ===================================================================


class TestPickFirstPatched:
    def test_empty(self):
        assert _pick_first_patched([]) is None

    def test_single(self):
        assert _pick_first_patched(["2.15.0"]) == "2.15.0"

    def test_semver_sort(self):
        assert _pick_first_patched(["2.15.0", "2.3.1", "2.16.0"]) == "2.3.1"

    def test_non_semver_fallback(self):
        # If sorting fails, returns first element
        result = _pick_first_patched(["abc", "def"])
        assert result in ("abc", "def")

    def test_complex_versions(self):
        result = _pick_first_patched(["1.2.3-beta", "1.2.3", "1.2.4"])
        assert result is not None


# ===================================================================
# _get_with_retry tests
# ===================================================================


class TestGetWithRetry:
    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_response(200, {"ok": True})
        resp = _get_with_retry("https://example.com")
        assert resp.status_code == 200

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_connection_error_raises(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("fail")
        with pytest.raises(requests.exceptions.ConnectionError):
            _get_with_retry("https://example.com")

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_timeout_raises(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("timeout")
        with pytest.raises(requests.exceptions.Timeout):
            _get_with_retry("https://example.com")

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retryable_status_returns_on_last_attempt(self, mock_get):
        mock_get.return_value = _mock_response(429)
        resp = _get_with_retry("https://example.com")
        assert resp.status_code == 429


# ===================================================================
# fetch_nvd_version_ranges tests
# ===================================================================


class TestFetchNvdVersionRanges:
    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_response(200, _nvd_cve_response())
        result = fetch_nvd_version_ranges("CVE-2021-44228")
        assert result["found"] is True
        assert len(result["ranges"]) == 1
        assert result["ranges"][0]["product"] == "log4j"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_404(self, mock_get):
        mock_get.return_value = _mock_response(404)
        result = fetch_nvd_version_ranges("CVE-9999-99999")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_network_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("fail")
        result = fetch_nvd_version_ranges("CVE-2021-44228")
        assert result["found"] is False
        assert "error" in result

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_empty_vulnerabilities(self, mock_get):
        mock_get.return_value = _mock_response(200, {"vulnerabilities": []})
        result = fetch_nvd_version_ranges("CVE-2021-44228")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_http_error(self, mock_get):
        import requests

        resp = _mock_response(500)
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        mock_get.return_value = resp
        result = fetch_nvd_version_ranges("CVE-2021-44228")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_json_decode_error(self, mock_get):
        resp = _mock_response(200)
        resp.json.side_effect = ValueError("bad json")
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp
        result = fetch_nvd_version_ranges("CVE-2021-44228")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_description_extraction(self, mock_get):
        nvd = _nvd_cve_response(description="Test description here")
        mock_get.return_value = _mock_response(200, nvd)
        result = fetch_nvd_version_ranges("CVE-2021-44228")
        assert "Test description" in result["description"]

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_no_configurations(self, mock_get):
        nvd = {"vulnerabilities": [{"cve": {"id": "CVE-2021-44228", "descriptions": [], "configurations": []}}]}
        mock_get.return_value = _mock_response(200, nvd)
        result = fetch_nvd_version_ranges("CVE-2021-44228")
        assert result["found"] is True
        assert result["ranges"] == []

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    @patch.dict(os.environ, {"NVD_API_KEY": "test-key"})
    def test_api_key_header(self, mock_get):
        mock_get.return_value = _mock_response(200, _nvd_cve_response())
        fetch_nvd_version_ranges("CVE-2021-44228")
        # Verify we called _get_with_retry (which handles headers)
        mock_get.assert_called_once()


# ===================================================================
# fetch_osv_version_ranges tests
# ===================================================================


class TestFetchOsvVersionRanges:
    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_response(200, _osv_record())
        result = fetch_osv_version_ranges("CVE-2021-44228")
        assert result["found"] is True
        assert len(result["packages"]) == 1
        assert result["packages"][0]["package"] == "org.apache.logging.log4j:log4j-core"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_404(self, mock_get):
        mock_get.return_value = _mock_response(404)
        result = fetch_osv_version_ranges("CVE-9999-99999")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_network_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("fail")
        result = fetch_osv_version_ranges("CVE-2021-44228")
        assert result["found"] is False
        assert "error" in result

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_http_error(self, mock_get):
        import requests

        resp = _mock_response(500)
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        mock_get.return_value = resp
        result = fetch_osv_version_ranges("CVE-2021-44228")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_json_decode_error(self, mock_get):
        resp = _mock_response(200)
        resp.json.side_effect = ValueError("bad json")
        resp.raise_for_status = MagicMock()
        mock_get.return_value = resp
        result = fetch_osv_version_ranges("CVE-2021-44228")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_ghsa_alias_follow(self, mock_get):
        """When primary has no affected data, follow GHSA aliases."""
        primary = _osv_record(affected=[])  # No affected data
        ghsa_record = _osv_record(
            osv_id="GHSA-jfh8-c2jp-5v3q",
            affected=[
                {
                    "package": {"ecosystem": "Maven", "name": "log4j-core"},
                    "ranges": [
                        {
                            "type": "ECOSYSTEM",
                            "events": [{"introduced": "2.0"}, {"fixed": "2.15.0"}],
                        }
                    ],
                }
            ],
        )
        # First call returns primary (no affected), second returns GHSA record
        mock_get.side_effect = [
            _mock_response(200, primary),
            _mock_response(200, ghsa_record),
        ]
        result = fetch_osv_version_ranges("CVE-2021-44228")
        assert result["found"] is True
        assert len(result["packages"]) == 1

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_ghsa_alias_404(self, mock_get):
        """GHSA alias 404 is handled gracefully."""
        primary = _osv_record(affected=[])
        mock_get.side_effect = [
            _mock_response(200, primary),
            _mock_response(404),
        ]
        result = fetch_osv_version_ranges("CVE-2021-44228")
        assert result["found"] is True  # Primary exists
        assert result["packages"] == []

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_ghsa_alias_network_error(self, mock_get):
        """GHSA alias network error is handled gracefully."""
        import requests

        primary = _osv_record(affected=[])
        mock_get.side_effect = [
            _mock_response(200, primary),
            requests.exceptions.ConnectionError("fail"),
        ]
        result = fetch_osv_version_ranges("CVE-2021-44228")
        assert result["found"] is True
        assert result["packages"] == []

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_deduplication(self, mock_get):
        """Same OSV id is not processed twice."""
        primary = _osv_record(osv_id="CVE-2021-44228", affected=[])
        # Alias returns same id
        alias_record = _osv_record(osv_id="CVE-2021-44228", affected=[])
        mock_get.side_effect = [
            _mock_response(200, primary),
            _mock_response(200, alias_record),
        ]
        result = fetch_osv_version_ranges("CVE-2021-44228")
        assert result["packages"] == []

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_aliases_collected(self, mock_get):
        record = _osv_record(aliases=["GHSA-jfh8-c2jp-5v3q", "CVE-2021-44228"])
        mock_get.return_value = _mock_response(200, record)
        result = fetch_osv_version_ranges("CVE-2021-44228")
        assert "GHSA-jfh8-c2jp-5v3q" in result["aliases"]


# ===================================================================
# fetch_version_range (unified) tests
# ===================================================================


class TestFetchVersionRange:
    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_version_ranges")
    def test_combined_data(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": True,
            "cve_id": "CVE-2021-44228",
            "description": "Log4j RCE",
            "ranges": [
                {
                    "source": "NVD",
                    "cpe23Uri": "cpe:2.3:a:apache:log4j:*",
                    "vendor": "apache",
                    "product": "log4j",
                    "ecosystem": "Maven",
                    "vulnerable": True,
                    "versionStartIncluding": "2.0.0",
                    "versionEndExcluding": "2.15.0",
                    "range_summary": ">= 2.0.0 && < 2.15.0",
                }
            ],
            "configuration_count": 1,
        }
        mock_osv.return_value = {
            "found": True,
            "cve_id": "CVE-2021-44228",
            "aliases": ["GHSA-jfh8-c2jp-5v3q"],
            "packages": [
                {
                    "source": "OSV",
                    "ecosystem": "Maven",
                    "package": "org.apache.logging.log4j:log4j-core",
                    "introduced": ["2.0.0"],
                    "fixed": ["2.15.0"],
                    "last_affected": [],
                    "range_summary": ">= 2.0.0, < 2.15.0",
                    "first_patched_version": "2.15.0",
                    "affected_versions": ["2.0", "2.14.1"],
                    "affected_version_count": 2,
                }
            ],
        }
        result = fetch_version_range("CVE-2021-44228")
        assert result["cve_id"] == "CVE-2021-44228"
        assert result["description"] == "Log4j RCE"
        assert len(result["packages"]) >= 1
        assert result["first_patched_version"] == "2.15.0"
        assert result["summary"]["total_packages"] >= 1
        assert "Maven" in result["summary"]["ecosystems"]

    def test_invalid_cve_id(self):
        result = fetch_version_range("")
        assert "error" in result
        assert result["packages"] == []

    def test_invalid_cve_format(self):
        result = fetch_version_range("not-a-cve")
        assert "error" in result

    def test_none_cve_id(self):
        result = fetch_version_range(None)
        assert "error" in result

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_version_ranges")
    def test_ecosystem_filter(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {"found": True, "description": "", "ranges": [], "configuration_count": 0}
        mock_osv.return_value = {
            "found": True,
            "aliases": [],
            "packages": [
                {
                    "source": "OSV",
                    "ecosystem": "PyPI",
                    "package": "django",
                    "range_summary": ">= 3.0, < 3.2.4",
                    "first_patched_version": "3.2.4",
                    "introduced": ["3.0"],
                    "fixed": ["3.2.4"],
                    "last_affected": [],
                    "affected_versions": [],
                    "affected_version_count": 0,
                },
                {
                    "source": "OSV",
                    "ecosystem": "npm",
                    "package": "django-npm",
                    "range_summary": ">= 1.0, < 2.0",
                    "first_patched_version": "2.0",
                    "introduced": ["1.0"],
                    "fixed": ["2.0"],
                    "last_affected": [],
                    "affected_versions": [],
                    "affected_version_count": 0,
                },
            ],
        }
        result = fetch_version_range("CVE-2021-44228", ecosystem="pypi")
        assert len(result["packages"]) == 1
        assert result["packages"][0]["ecosystem"] == "PyPI"

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_version_ranges")
    def test_osv_preferred_over_nvd(self, mock_nvd, mock_osv):
        """OSV data is preferred; NVD fills gaps."""
        mock_nvd.return_value = {
            "found": True,
            "description": "",
            "ranges": [
                {
                    "source": "NVD",
                    "vendor": "apache",
                    "product": "log4j-core",
                    "ecosystem": "Maven",
                    "vulnerable": True,
                    "range_summary": ">= 2.0, < 2.15",
                    "cpe23Uri": "cpe:...",
                    "versionEndExcluding": "2.15.0",
                }
            ],
            "configuration_count": 1,
        }
        mock_osv.return_value = {
            "found": True,
            "aliases": [],
            "packages": [
                {
                    "source": "OSV",
                    "ecosystem": "Maven",
                    "package": "log4j-core",
                    "range_summary": ">= 2.0, < 2.15.0",
                    "first_patched_version": "2.15.0",
                    "introduced": ["2.0"],
                    "fixed": ["2.15.0"],
                    "last_affected": [],
                    "affected_versions": ["2.0", "2.14.1"],
                    "affected_version_count": 2,
                }
            ],
        }
        result = fetch_version_range("CVE-2021-44228")
        # OSV package should be present
        osv_pkgs = [p for p in result["packages"] if p["source"] == "OSV"]
        assert len(osv_pkgs) == 1
        # NVD shouldn't duplicate the same ecosystem+package
        nvd_pkgs = [p for p in result["packages"] if p["source"] == "NVD"]
        assert len(nvd_pkgs) == 0  # deduped because OSV already covers maven/log4j-core

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_version_ranges")
    def test_nvd_only_data(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": True,
            "description": "A vuln",
            "ranges": [
                {
                    "source": "NVD",
                    "vendor": "vendor",
                    "product": "product",
                    "ecosystem": "",
                    "vulnerable": True,
                    "range_summary": "< 2.0",
                    "cpe23Uri": "cpe:2.3:a:vendor:product:*",
                    "versionEndExcluding": "2.0",
                }
            ],
            "configuration_count": 1,
        }
        mock_osv.return_value = {"found": False, "aliases": [], "packages": []}
        result = fetch_version_range("CVE-2021-44228")
        assert len(result["packages"]) == 1
        assert result["packages"][0]["source"] == "NVD"
        assert result["packages"][0]["first_patched_version"] == "2.0"

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_version_ranges")
    def test_no_data_found(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {"found": False, "error": "Not found", "ranges": []}
        mock_osv.return_value = {"found": False, "error": "Not found", "packages": []}
        result = fetch_version_range("CVE-9999-99999")
        assert result["packages"] == []
        assert result["summary"]["total_packages"] == 0

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_version_ranges")
    def test_non_vulnerable_nvd_ranges_excluded(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {
            "found": True,
            "description": "",
            "ranges": [
                {
                    "source": "NVD",
                    "vendor": "v",
                    "product": "p",
                    "ecosystem": "",
                    "vulnerable": False,
                    "range_summary": "all",
                    "cpe23Uri": "cpe:...",
                    "versionEndExcluding": None,
                }
            ],
            "configuration_count": 1,
        }
        mock_osv.return_value = {"found": False, "aliases": [], "packages": []}
        result = fetch_version_range("CVE-2021-44228")
        assert result["packages"] == []

    @patch("manus_agent.tools.get_version_range.fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_version_ranges")
    def test_cve_id_uppercased(self, mock_nvd, mock_osv):
        mock_nvd.return_value = {"found": False, "ranges": [], "description": ""}
        mock_osv.return_value = {"found": False, "aliases": [], "packages": []}
        result = fetch_version_range("cve-2021-44228")
        assert result["cve_id"] == "CVE-2021-44228"


# ===================================================================
# get_version_range (Strands tool) tests
# ===================================================================


class TestGetVersionRangeTool:
    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_success(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "packages": [{"ecosystem": "Maven", "package": "log4j"}],
            "summary": {"total_packages": 1},
        }
        tool = _make_tool_use("CVE-2021-44228")
        result = get_version_range(tool)
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_no_packages(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-9999-99999",
            "packages": [],
            "summary": {"total_packages": 0},
        }
        tool = _make_tool_use("CVE-9999-99999")
        result = get_version_range(tool)
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_with_error(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "bad",
            "error": "Invalid",
            "packages": [],
        }
        tool = _make_tool_use("bad")
        result = get_version_range(tool)
        assert result["status"] == "error"

    def test_empty_cve_id(self):
        tool = {"toolUseId": "t1", "input": {"cve_id": ""}}
        result = get_version_range(tool)
        assert result["status"] == "error"

    def test_none_cve_id(self):
        tool = {"toolUseId": "t1", "input": {"cve_id": None}}
        result = get_version_range(tool)
        assert result["status"] == "error"

    def test_missing_cve_id(self):
        tool = {"toolUseId": "t1", "input": {}}
        result = get_version_range(tool)
        assert result["status"] == "error"

    def test_none_input(self):
        tool = {"toolUseId": "t1", "input": None}
        result = get_version_range(tool)
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_ecosystem_passed(self, mock_fetch):
        mock_fetch.return_value = {"cve_id": "CVE-2021-44228", "packages": [{"a": 1}], "summary": {}}
        tool = _make_tool_use("CVE-2021-44228", ecosystem="pypi")
        get_version_range(tool)
        mock_fetch.assert_called_once_with("CVE-2021-44228", ecosystem="pypi")

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_default_ecosystem(self, mock_fetch):
        mock_fetch.return_value = {"cve_id": "CVE-2021-44228", "packages": [{"a": 1}], "summary": {}}
        tool = _make_tool_use("CVE-2021-44228")
        get_version_range(tool)
        mock_fetch.assert_called_once_with("CVE-2021-44228", ecosystem="auto")


# ===================================================================
# CLI _run_version_range tests
# ===================================================================


class TestCliVersionRange:
    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_output(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "description": "Apache Log4j2 RCE vulnerability",
            "first_patched_version": "2.15.0",
            "packages": [
                {
                    "ecosystem": "Maven",
                    "package": "org.apache.logging.log4j:log4j-core",
                    "source": "OSV",
                    "vulnerable_range": ">= 2.0.0, < 2.15.0",
                    "introduced": ["2.0.0"],
                    "fixed": ["2.15.0"],
                    "last_affected": [],
                    "first_patched_version": "2.15.0",
                    "affected_versions": ["2.0", "2.14.1"],
                    "affected_version_count": 2,
                }
            ],
            "osv": {"aliases": ["GHSA-jfh8-c2jp-5v3q"]},
            "summary": {
                "total_packages": 1,
                "ecosystems": ["Maven"],
                "has_nvd_cpe_data": True,
                "has_osv_data": True,
            },
        }
        from manus_agent.cli import _run_version_range

        captured = StringIO()
        sys.stdout = captured
        try:
            rc = _run_version_range(["CVE-2021-44228"])
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert rc == 0
        assert "CVE-2021-44228" in output
        assert "Maven" in output
        assert "log4j" in output
        assert "2.15.0" in output
        assert ">= 2.0.0" in output

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_json_output(self, mock_fetch):
        payload = {
            "cve_id": "CVE-2021-44228",
            "packages": [{"ecosystem": "Maven", "package": "log4j"}],
            "summary": {"total_packages": 1, "ecosystems": ["Maven"]},
        }
        mock_fetch.return_value = payload
        from manus_agent.cli import _run_version_range

        captured = StringIO()
        sys.stdout = captured
        try:
            rc = _run_version_range(["CVE-2021-44228", "--output", "json"])
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert rc == 0
        parsed = json.loads(output)
        assert parsed["cve_id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_error_returns_1(self, mock_fetch):
        mock_fetch.return_value = {"error": "Invalid CVE", "cve_id": "bad"}
        from manus_agent.cli import _run_version_range

        captured = StringIO()
        sys.stderr = captured
        try:
            rc = _run_version_range(["bad"])
        finally:
            sys.stderr = sys.__stderr__
        assert rc == 1

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_no_packages_text(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-9999-99999",
            "description": "",
            "first_patched_version": None,
            "packages": [],
            "osv": {"aliases": []},
            "summary": {
                "total_packages": 0,
                "ecosystems": [],
                "has_nvd_cpe_data": False,
                "has_osv_data": False,
            },
        }
        from manus_agent.cli import _run_version_range

        captured = StringIO()
        sys.stdout = captured
        try:
            rc = _run_version_range(["CVE-9999-99999"])
        finally:
            sys.stdout = sys.__stdout__
        assert rc == 0
        assert "No affected package records found" in captured.getvalue()

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_ecosystem_filter_passed(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "packages": [{"ecosystem": "PyPI", "package": "django"}],
            "description": "",
            "first_patched_version": None,
            "osv": {"aliases": []},
            "summary": {
                "total_packages": 1,
                "ecosystems": ["PyPI"],
                "has_nvd_cpe_data": False,
                "has_osv_data": True,
            },
        }
        from manus_agent.cli import _run_version_range

        captured = StringIO()
        sys.stdout = captured
        try:
            _run_version_range(["CVE-2021-44228", "--ecosystem", "pypi"])
        finally:
            sys.stdout = sys.__stdout__
        mock_fetch.assert_called_once_with("CVE-2021-44228", ecosystem="pypi")

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_cpe_in_output(self, mock_fetch):
        """NVD packages show CPE URI in text output."""
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "description": "",
            "first_patched_version": None,
            "packages": [
                {
                    "ecosystem": "CPE",
                    "package": "apache/log4j",
                    "source": "NVD",
                    "vulnerable_range": ">= 2.0, < 2.15",
                    "cpe23Uri": "cpe:2.3:a:apache:log4j:*",
                    "first_patched_version": "2.15.0",
                    "affected_versions": [],
                    "affected_version_count": 0,
                }
            ],
            "osv": {"aliases": []},
            "summary": {
                "total_packages": 1,
                "ecosystems": ["CPE"],
                "has_nvd_cpe_data": True,
                "has_osv_data": False,
            },
        }
        from manus_agent.cli import _run_version_range

        captured = StringIO()
        sys.stdout = captured
        try:
            _run_version_range(["CVE-2021-44228"])
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert "cpe:2.3" in output

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_long_description_truncated(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "description": "A" * 300,
            "first_patched_version": None,
            "packages": [],
            "osv": {"aliases": []},
            "summary": {
                "total_packages": 0,
                "ecosystems": [],
                "has_nvd_cpe_data": False,
                "has_osv_data": False,
            },
        }
        from manus_agent.cli import _run_version_range

        captured = StringIO()
        sys.stdout = captured
        try:
            _run_version_range(["CVE-2021-44228"])
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert "..." in output

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_data_sources_shown(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "description": "",
            "first_patched_version": None,
            "packages": [],
            "osv": {"aliases": []},
            "summary": {
                "total_packages": 0,
                "ecosystems": [],
                "has_nvd_cpe_data": True,
                "has_osv_data": True,
            },
        }
        from manus_agent.cli import _run_version_range

        captured = StringIO()
        sys.stdout = captured
        try:
            _run_version_range(["CVE-2021-44228"])
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert "NVD" in output
        assert "OSV.dev" in output

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_versions_shown_with_overflow(self, mock_fetch):
        """When version count > 10, show truncated list."""
        versions = [f"{i}.0.0" for i in range(20)]
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "description": "",
            "first_patched_version": None,
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "big-pkg",
                    "source": "OSV",
                    "vulnerable_range": "all",
                    "first_patched_version": None,
                    "affected_versions": versions[:10],
                    "affected_version_count": 20,
                }
            ],
            "osv": {"aliases": []},
            "summary": {
                "total_packages": 1,
                "ecosystems": ["PyPI"],
                "has_nvd_cpe_data": False,
                "has_osv_data": True,
            },
        }
        from manus_agent.cli import _run_version_range

        captured = StringIO()
        sys.stdout = captured
        try:
            _run_version_range(["CVE-2021-44228"])
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert "+10 more" in output

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_aliases_shown(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "description": "",
            "first_patched_version": None,
            "packages": [{"ecosystem": "Maven", "package": "log4j", "source": "OSV", "vulnerable_range": "all"}],
            "osv": {"aliases": ["GHSA-jfh8-c2jp-5v3q", "CVE-2021-44228"]},
            "summary": {
                "total_packages": 1,
                "ecosystems": ["Maven"],
                "has_nvd_cpe_data": False,
                "has_osv_data": True,
            },
        }
        from manus_agent.cli import _run_version_range

        captured = StringIO()
        sys.stdout = captured
        try:
            _run_version_range(["CVE-2021-44228"])
        finally:
            sys.stdout = sys.__stdout__
        output = captured.getvalue()
        assert "GHSA-jfh8-c2jp-5v3q" in output


# ===================================================================
# CLI dispatch tests
# ===================================================================


class TestCliDispatch:
    def test_version_range_in_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "version-range" in _SUBCOMMANDS

    def test_parser_help(self):
        from manus_agent.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        assert parser.prog == "manus-agent version-range"


# ===================================================================
# VI agent wiring tests
# ===================================================================


class TestViAgentWiring:
    def test_import_get_version_range(self):
        """get_version_range can be imported from the tools package."""
        from manus_agent.tools.get_version_range import get_version_range

        assert callable(get_version_range)

    def test_tool_spec_accessible(self):
        from manus_agent.tools.get_version_range import TOOL_SPEC

        assert TOOL_SPEC["name"] == "get_version_range"


# ===================================================================
# VALID_ECOSYSTEMS constant tests
# ===================================================================


class TestValidEcosystems:
    def test_auto_included(self):
        assert "auto" in VALID_ECOSYSTEMS

    def test_common_ecosystems(self):
        for eco in ("pypi", "npm", "maven", "go"):
            assert eco in VALID_ECOSYSTEMS


# ===================================================================
# Integration-style tests (all mocked)
# ===================================================================


class TestIntegration:
    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_end_to_end_tool(self, mock_get):
        """End-to-end Strands tool call with mocked HTTP."""
        nvd_resp = _mock_response(200, _nvd_cve_response())
        osv_resp = _mock_response(200, _osv_record())
        mock_get.side_effect = [nvd_resp, osv_resp]

        tool = _make_tool_use("CVE-2021-44228")
        result = get_version_range(tool)
        assert result["status"] == "success"
        payload = json.loads(result["content"][0]["text"])
        assert payload["cve_id"] == "CVE-2021-44228"
        assert len(payload["packages"]) >= 1

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_end_to_end_cli_text(self, mock_get):
        """End-to-end CLI text output with mocked HTTP."""
        nvd_resp = _mock_response(200, _nvd_cve_response())
        osv_resp = _mock_response(200, _osv_record())
        mock_get.side_effect = [nvd_resp, osv_resp]

        from manus_agent.cli import _run_version_range

        captured = StringIO()
        sys.stdout = captured
        try:
            rc = _run_version_range(["CVE-2021-44228"])
        finally:
            sys.stdout = sys.__stdout__
        assert rc == 0
        output = captured.getvalue()
        assert "CVE-2021-44228" in output
        assert "2.15.0" in output

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_end_to_end_cli_json(self, mock_get):
        """End-to-end CLI JSON output with mocked HTTP."""
        nvd_resp = _mock_response(200, _nvd_cve_response())
        osv_resp = _mock_response(200, _osv_record())
        mock_get.side_effect = [nvd_resp, osv_resp]

        from manus_agent.cli import _run_version_range

        captured = StringIO()
        sys.stdout = captured
        try:
            rc = _run_version_range(["CVE-2021-44228", "--output", "json"])
        finally:
            sys.stdout = sys.__stdout__
        assert rc == 0
        parsed = json.loads(captured.getvalue())
        assert parsed["cve_id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_nvd_failure_osv_success(self, mock_get):
        """If NVD fails, OSV data is still returned."""
        import requests

        nvd_error = requests.exceptions.ConnectionError("NVD down")
        osv_resp = _mock_response(200, _osv_record())
        mock_get.side_effect = [nvd_error, osv_resp]

        result = fetch_version_range("CVE-2021-44228")
        assert len(result["packages"]) >= 1
        assert result["nvd"]["found"] is False
        assert result["summary"]["has_osv_data"] is True

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_osv_failure_nvd_success(self, mock_get):
        """If OSV fails, NVD data is still returned."""
        import requests

        nvd_resp = _mock_response(200, _nvd_cve_response())
        osv_error = requests.exceptions.ConnectionError("OSV down")
        mock_get.side_effect = [nvd_resp, osv_error]

        result = fetch_version_range("CVE-2021-44228")
        assert len(result["packages"]) >= 1
        assert result["summary"]["has_nvd_cpe_data"] is True

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_both_sources_fail(self, mock_get):
        """If both NVD and OSV fail, empty packages returned."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("all down")

        result = fetch_version_range("CVE-2021-44228")
        assert result["packages"] == []
        assert result["summary"]["total_packages"] == 0
