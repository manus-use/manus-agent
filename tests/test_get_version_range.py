#!/usr/bin/env python3
"""Comprehensive test suite for get_version_range tool and version-range CLI."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from manus_agent.tools.get_version_range import (
    TOOL_SPEC,
    _build_version_constraint,
    _fetch_nvd_version_ranges,
    _fetch_osv_version_ranges,
    _guess_ecosystem_from_cpe,
    _merge_results,
    _normalise_ecosystem,
    _parse_cpe_uri,
    _summarise_osv_range,
    get_version_range,
)

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """Disable retry delays in all tests."""
    monkeypatch.setenv("VERSION_RANGE_MAX_RETRIES", "1")
    monkeypatch.setenv("VERSION_RANGE_RETRY_DELAY", "0")


@pytest.fixture()
def tool_use():
    """Factory for ToolUse dicts."""

    def _make(cve_id: str = "CVE-2021-44228", ecosystem: str | None = None):
        inp = {"cve_id": cve_id}
        if ecosystem is not None:
            inp["ecosystem"] = ecosystem
        return {"toolUseId": "test-id-1", "input": inp}

    return _make


# Sample NVD response with CPE configurations
NVD_LOG4J_RESPONSE = {
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2021-44228",
                "configurations": [
                    {
                        "operator": "OR",
                        "nodes": [
                            {
                                "operator": "OR",
                                "cpeMatch": [
                                    {
                                        "vulnerable": True,
                                        "criteria": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                                        "versionStartIncluding": "2.0",
                                        "versionEndExcluding": "2.15.0",
                                    },
                                    {
                                        "vulnerable": False,
                                        "criteria": "cpe:2.3:a:apache:log4j:1.2:*:*:*:*:*:*:*",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        }
    ]
}

# Sample NVD response with exact version (no range)
NVD_EXACT_VERSION_RESPONSE = {
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2024-1234",
                "configurations": [
                    {
                        "operator": "OR",
                        "nodes": [
                            {
                                "operator": "OR",
                                "cpeMatch": [
                                    {
                                        "vulnerable": True,
                                        "criteria": "cpe:2.3:a:vendor:product:1.0.0:*:*:*:*:*:*:*",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        }
    ]
}

# Sample NVD response with versionEndIncluding
NVD_END_INCLUDING_RESPONSE = {
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2024-5678",
                "configurations": [
                    {
                        "operator": "AND",
                        "nodes": [
                            {
                                "operator": "OR",
                                "cpeMatch": [
                                    {
                                        "vulnerable": True,
                                        "criteria": "cpe:2.3:a:django:django:*:*:*:*:*:*:*:*",
                                        "versionStartIncluding": "3.0",
                                        "versionEndIncluding": "3.2.5",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        }
    ]
}

# Sample NVD with versionStartExcluding
NVD_START_EXCLUDING_RESPONSE = {
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2024-9999",
                "configurations": [
                    {
                        "operator": "OR",
                        "nodes": [
                            {
                                "operator": "OR",
                                "cpeMatch": [
                                    {
                                        "vulnerable": True,
                                        "criteria": "cpe:2.3:a:nodejs:express:*:*:*:*:*:*:*:*",
                                        "versionStartExcluding": "4.0.0",
                                        "versionEndExcluding": "4.18.2",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        }
    ]
}

# Sample OSV record for Log4j
OSV_LOG4J_CVE_RECORD = {
    "id": "CVE-2021-44228",
    "aliases": ["GHSA-jfh8-c2jp-5v3q"],
    "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}],
    "affected": [],
}

OSV_LOG4J_GHSA_RECORD = {
    "id": "GHSA-jfh8-c2jp-5v3q",
    "aliases": ["CVE-2021-44228"],
    "affected": [
        {
            "package": {"name": "org.apache.logging.log4j:log4j-core", "ecosystem": "Maven"},
            "ranges": [
                {
                    "type": "ECOSYSTEM",
                    "events": [
                        {"introduced": "2.0-beta9"},
                        {"fixed": "2.15.0"},
                    ],
                }
            ],
            "versions": ["2.0", "2.0.1", "2.0.2", "2.1", "2.2", "2.3", "2.14.1"],
            "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}],
        }
    ],
}

# OSV record with last_affected instead of fixed
OSV_LAST_AFFECTED_RECORD = {
    "id": "CVE-2024-0001",
    "aliases": [],
    "affected": [
        {
            "package": {"name": "some-package", "ecosystem": "PyPI"},
            "ranges": [
                {
                    "type": "ECOSYSTEM",
                    "events": [
                        {"introduced": "1.0.0"},
                        {"last_affected": "1.9.9"},
                    ],
                }
            ],
            "versions": ["1.0.0", "1.5.0", "1.9.9"],
        }
    ],
}

# OSV record with multiple ranges
OSV_MULTI_RANGE_RECORD = {
    "id": "CVE-2024-0002",
    "aliases": [],
    "affected": [
        {
            "package": {"name": "multi-pkg", "ecosystem": "npm"},
            "ranges": [
                {
                    "type": "SEMVER",
                    "events": [
                        {"introduced": "1.0.0"},
                        {"fixed": "1.5.0"},
                    ],
                },
                {
                    "type": "SEMVER",
                    "events": [
                        {"introduced": "2.0.0"},
                        {"fixed": "2.3.0"},
                    ],
                },
            ],
            "versions": ["1.0.0", "1.4.9", "2.0.0", "2.2.9"],
        }
    ],
}

# NVD response with wildcard version
NVD_WILDCARD_RESPONSE = {
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2024-0003",
                "configurations": [
                    {
                        "operator": "OR",
                        "nodes": [
                            {
                                "operator": "OR",
                                "cpeMatch": [
                                    {
                                        "vulnerable": True,
                                        "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        }
    ]
}


def _mock_response(json_data, status_code=200):
    """Create a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


def _mock_error_response(status_code=404):
    """Create a mock error response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = {}
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"HTTP {status_code}", response=resp)
    return resp


# ═══════════════════════════════════════════════════════════════════════════
# TOOL_SPEC contract
# ═══════════════════════════════════════════════════════════════════════════


class TestToolSpec:
    """TOOL_SPEC contract tests."""

    def test_tool_spec_has_required_keys(self):
        assert "name" in TOOL_SPEC
        assert "description" in TOOL_SPEC
        assert "inputSchema" in TOOL_SPEC

    def test_tool_spec_name(self):
        assert TOOL_SPEC["name"] == "get_version_range"

    def test_tool_spec_description_nonempty(self):
        assert len(TOOL_SPEC["description"]) > 50

    def test_tool_spec_input_schema_requires_cve_id(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_tool_spec_ecosystem_property(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "ecosystem" in schema["properties"]
        # ecosystem should not be required
        assert "ecosystem" not in schema.get("required", [])


# ═══════════════════════════════════════════════════════════════════════════
# CPE parsing helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestParseCpeUri:
    """Tests for _parse_cpe_uri."""

    def test_full_cpe23(self):
        result = _parse_cpe_uri("cpe:2.3:a:apache:log4j:2.0:*:*:*:*:*:*:*")
        assert result["vendor"] == "apache"
        assert result["product"] == "log4j"
        assert result["version"] == "2.0"

    def test_wildcard_version(self):
        result = _parse_cpe_uri("cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*")
        assert result["version"] == "*"

    def test_short_cpe(self):
        result = _parse_cpe_uri("cpe:2.3:a:v")
        assert result["vendor"] == "v"
        assert result["product"] == ""
        assert result["version"] == "*"

    def test_empty_string(self):
        result = _parse_cpe_uri("")
        assert result["vendor"] == ""
        assert result["product"] == ""

    def test_typical_npm_cpe(self):
        result = _parse_cpe_uri("cpe:2.3:a:nodejs:express:4.0.0:*:*:*:*:*:*:*")
        assert result["vendor"] == "nodejs"
        assert result["product"] == "express"
        assert result["version"] == "4.0.0"


# ═══════════════════════════════════════════════════════════════════════════
# Ecosystem guessing
# ═══════════════════════════════════════════════════════════════════════════


class TestGuessEcosystem:
    """Tests for _guess_ecosystem_from_cpe."""

    def test_python_vendor(self):
        assert _guess_ecosystem_from_cpe("python", "requests") == "PyPI"

    def test_django_product(self):
        assert _guess_ecosystem_from_cpe("djangoproject", "django") == "PyPI"

    def test_nodejs_vendor(self):
        assert _guess_ecosystem_from_cpe("nodejs", "express") == "npm"

    def test_npm_vendor(self):
        assert _guess_ecosystem_from_cpe("npm", "lodash") == "npm"

    def test_apache_vendor(self):
        assert _guess_ecosystem_from_cpe("apache", "tomcat") == "Maven"

    def test_spring_product(self):
        assert _guess_ecosystem_from_cpe("vmware", "spring_framework") == "Maven"

    def test_golang_vendor(self):
        assert _guess_ecosystem_from_cpe("golang", "mylib") == "Go"

    def test_rust_vendor(self):
        assert _guess_ecosystem_from_cpe("rust", "serde") == "crates.io"

    def test_ruby_vendor(self):
        assert _guess_ecosystem_from_cpe("rubygems", "rails") == "RubyGems"

    def test_nuget_vendor(self):
        assert _guess_ecosystem_from_cpe("nuget", "newtonsoft") == "NuGet"

    def test_php_vendor(self):
        assert _guess_ecosystem_from_cpe("php", "laravel") == "Packagist"

    def test_unknown_vendor(self):
        assert _guess_ecosystem_from_cpe("acme_corp", "widget") == "unknown"

    def test_log4j_is_maven(self):
        assert _guess_ecosystem_from_cpe("apache", "log4j") == "Maven"


# ═══════════════════════════════════════════════════════════════════════════
# Version constraint building
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildVersionConstraint:
    """Tests for _build_version_constraint."""

    def test_start_including_end_excluding(self):
        match = {
            "versionStartIncluding": "2.0",
            "versionEndExcluding": "2.15.0",
            "criteria": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
        }
        assert _build_version_constraint(match) == ">=2.0, <2.15.0"

    def test_start_including_end_including(self):
        match = {
            "versionStartIncluding": "3.0",
            "versionEndIncluding": "3.2.5",
            "criteria": "cpe:2.3:a:django:django:*:*:*:*:*:*:*:*",
        }
        assert _build_version_constraint(match) == ">=3.0, <=3.2.5"

    def test_only_end_excluding(self):
        match = {
            "versionEndExcluding": "1.5.0",
            "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
        }
        assert _build_version_constraint(match) == "<1.5.0"

    def test_only_start_including(self):
        match = {
            "versionStartIncluding": "2.0",
            "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
        }
        assert _build_version_constraint(match) == ">=2.0"

    def test_start_excluding_end_excluding(self):
        match = {
            "versionStartExcluding": "4.0.0",
            "versionEndExcluding": "4.18.2",
            "criteria": "cpe:2.3:a:nodejs:express:*:*:*:*:*:*:*:*",
        }
        assert _build_version_constraint(match) == ">4.0.0, <4.18.2"

    def test_exact_version_fallback(self):
        match = {"criteria": "cpe:2.3:a:vendor:product:1.0.0:*:*:*:*:*:*:*"}
        assert _build_version_constraint(match) == "1.0.0"

    def test_wildcard_fallback(self):
        match = {"criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*"}
        assert _build_version_constraint(match) == "all versions"

    def test_only_end_including(self):
        match = {
            "versionEndIncluding": "2.0.0",
            "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
        }
        assert _build_version_constraint(match) == "<=2.0.0"


# ═══════════════════════════════════════════════════════════════════════════
# OSV range summarisation
# ═══════════════════════════════════════════════════════════════════════════


class TestSummariseOsvRange:
    """Tests for _summarise_osv_range."""

    def test_introduced_and_fixed(self):
        rng = {
            "type": "ECOSYSTEM",
            "events": [{"introduced": "1.0.0"}, {"fixed": "1.5.0"}],
        }
        result = _summarise_osv_range(rng)
        assert result["range"] == ">=1.0.0, <1.5.0"
        assert result["introduced"] == "1.0.0"
        assert result["fixed"] == "1.5.0"

    def test_introduced_zero(self):
        rng = {
            "type": "SEMVER",
            "events": [{"introduced": "0"}, {"fixed": "2.0.0"}],
        }
        result = _summarise_osv_range(rng)
        assert result["range"] == ">=0, <2.0.0"

    def test_last_affected_no_fix(self):
        rng = {
            "type": "ECOSYSTEM",
            "events": [{"introduced": "1.0.0"}, {"last_affected": "1.9.9"}],
        }
        result = _summarise_osv_range(rng)
        assert result["range"] == ">=1.0.0, <=1.9.9"
        assert result["last_affected"] == "1.9.9"
        assert result["fixed"] == ""

    def test_only_introduced(self):
        rng = {"type": "ECOSYSTEM", "events": [{"introduced": "3.0.0"}]}
        result = _summarise_osv_range(rng)
        assert result["range"] == ">=3.0.0"

    def test_empty_events(self):
        rng = {"type": "ECOSYSTEM", "events": []}
        result = _summarise_osv_range(rng)
        assert result["range"] == "all versions"

    def test_no_events_key(self):
        rng = {"type": "ECOSYSTEM"}
        result = _summarise_osv_range(rng)
        assert result["range"] == "all versions"


# ═══════════════════════════════════════════════════════════════════════════
# Ecosystem normalisation
# ═══════════════════════════════════════════════════════════════════════════


class TestNormaliseEcosystem:
    """Tests for _normalise_ecosystem."""

    def test_pypi(self):
        assert _normalise_ecosystem("pypi") == "PyPI"

    def test_npm(self):
        assert _normalise_ecosystem("npm") == "npm"

    def test_maven(self):
        assert _normalise_ecosystem("maven") == "Maven"

    def test_go(self):
        assert _normalise_ecosystem("go") == "Go"

    def test_passthrough_unknown(self):
        assert _normalise_ecosystem("unknown_eco") == "unknown_eco"

    def test_case_insensitive(self):
        assert _normalise_ecosystem("PYPI") == "PyPI"
        assert _normalise_ecosystem("NPM") == "npm"


# ═══════════════════════════════════════════════════════════════════════════
# NVD fetching
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchNvdVersionRanges:
    """Tests for _fetch_nvd_version_ranges."""

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_log4j_range(self, mock_get):
        mock_get.return_value = _mock_response(NVD_LOG4J_RESPONSE)
        results = _fetch_nvd_version_ranges("CVE-2021-44228")
        assert len(results) == 1
        r = results[0]
        assert r["source"] == "nvd"
        assert r["product"] == "log4j"
        assert r["vulnerable_range"] == ">=2.0, <2.15.0"
        assert r["first_patched"] == "2.15.0"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_non_vulnerable_cpe_skipped(self, mock_get):
        mock_get.return_value = _mock_response(NVD_LOG4J_RESPONSE)
        results = _fetch_nvd_version_ranges("CVE-2021-44228")
        # Only vulnerable=True entries should be included
        assert all(r["product"] != "log4j" or r["vulnerable_range"] != "" for r in results)

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_exact_version(self, mock_get):
        mock_get.return_value = _mock_response(NVD_EXACT_VERSION_RESPONSE)
        results = _fetch_nvd_version_ranges("CVE-2024-1234")
        assert len(results) == 1
        assert results[0]["vulnerable_range"] == "1.0.0"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_end_including(self, mock_get):
        mock_get.return_value = _mock_response(NVD_END_INCLUDING_RESPONSE)
        results = _fetch_nvd_version_ranges("CVE-2024-5678")
        assert len(results) == 1
        assert results[0]["vulnerable_range"] == ">=3.0, <=3.2.5"
        assert results[0]["operator"] == "AND/OR"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_start_excluding(self, mock_get):
        mock_get.return_value = _mock_response(NVD_START_EXCLUDING_RESPONSE)
        results = _fetch_nvd_version_ranges("CVE-2024-9999")
        assert len(results) == 1
        assert results[0]["vulnerable_range"] == ">4.0.0, <4.18.2"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_wildcard_version(self, mock_get):
        mock_get.return_value = _mock_response(NVD_WILDCARD_RESPONSE)
        results = _fetch_nvd_version_ranges("CVE-2024-0003")
        assert len(results) == 1
        assert results[0]["vulnerable_range"] == "all versions"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_empty_vulnerabilities(self, mock_get):
        mock_get.return_value = _mock_response({"vulnerabilities": []})
        results = _fetch_nvd_version_ranges("CVE-2024-0000")
        assert results == []

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_no_configurations(self, mock_get):
        resp = {"vulnerabilities": [{"cve": {"id": "CVE-2024-0000"}}]}
        mock_get.return_value = _mock_response(resp)
        results = _fetch_nvd_version_ranges("CVE-2024-0000")
        assert results == []

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_network_error_returns_empty(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("timeout")
        results = _fetch_nvd_version_ranges("CVE-2021-44228")
        assert results == []

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_nvd_api_key_passed(self, mock_get):
        mock_get.return_value = _mock_response({"vulnerabilities": []})
        with patch.dict(os.environ, {"NVD_API_KEY": "test-key-123"}):
            _fetch_nvd_version_ranges("CVE-2024-0000")
        # Verify _get_with_retry was called with headers
        call_kwargs = mock_get.call_args
        assert "headers" in call_kwargs.kwargs

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_ecosystem_hint_from_cpe(self, mock_get):
        mock_get.return_value = _mock_response(NVD_LOG4J_RESPONSE)
        results = _fetch_nvd_version_ranges("CVE-2021-44228")
        # apache/log4j should be hinted as Maven
        assert results[0]["ecosystem"] == "Maven"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_django_ecosystem_hint(self, mock_get):
        mock_get.return_value = _mock_response(NVD_END_INCLUDING_RESPONSE)
        results = _fetch_nvd_version_ranges("CVE-2024-5678")
        assert results[0]["ecosystem"] == "PyPI"


# ═══════════════════════════════════════════════════════════════════════════
# OSV fetching
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchOsvVersionRanges:
    """Tests for _fetch_osv_version_ranges."""

    @patch("manus_agent.tools.get_version_range._fetch_osv_single")
    def test_log4j_with_ghsa_follow(self, mock_fetch):
        def side_effect(osv_id):
            if osv_id == "CVE-2021-44228":
                return OSV_LOG4J_CVE_RECORD
            if osv_id == "GHSA-jfh8-c2jp-5v3q":
                return OSV_LOG4J_GHSA_RECORD
            return None

        mock_fetch.side_effect = side_effect
        results = _fetch_osv_version_ranges("CVE-2021-44228")
        assert len(results) == 1
        r = results[0]
        assert r["ecosystem"] == "Maven"
        assert r["package"] == "org.apache.logging.log4j:log4j-core"
        assert "2.0-beta9" in r["vulnerable_range"]
        assert r["first_patched"] == "2.15.0"
        assert len(r["affected_versions"]) > 0

    @patch("manus_agent.tools.get_version_range._fetch_osv_single")
    def test_last_affected_version(self, mock_fetch):
        mock_fetch.return_value = OSV_LAST_AFFECTED_RECORD
        results = _fetch_osv_version_ranges("CVE-2024-0001")
        assert len(results) == 1
        assert results[0]["last_affected"] == "1.9.9"
        assert results[0]["first_patched"] == ""

    @patch("manus_agent.tools.get_version_range._fetch_osv_single")
    def test_multi_range(self, mock_fetch):
        mock_fetch.return_value = OSV_MULTI_RANGE_RECORD
        results = _fetch_osv_version_ranges("CVE-2024-0002")
        assert len(results) == 1
        assert "||" in results[0]["vulnerable_range"]
        assert results[0]["first_patched"] == "1.5.0"

    @patch("manus_agent.tools.get_version_range._fetch_osv_single")
    def test_cve_not_found(self, mock_fetch):
        mock_fetch.return_value = None
        results = _fetch_osv_version_ranges("CVE-9999-0000")
        assert results == []

    @patch("manus_agent.tools.get_version_range._fetch_osv_single")
    def test_deduplication(self, mock_fetch):
        """Same ecosystem:package from CVE and GHSA should be deduplicated."""
        cve_record = {
            "id": "CVE-2024-DUPE",
            "aliases": ["GHSA-xxxx-xxxx-xxxx"],
            "affected": [
                {
                    "package": {"name": "my-pkg", "ecosystem": "npm"},
                    "ranges": [{"type": "SEMVER", "events": [{"introduced": "0"}, {"fixed": "1.0.0"}]}],
                    "versions": ["0.1.0"],
                }
            ],
        }
        ghsa_record = {
            "id": "GHSA-xxxx-xxxx-xxxx",
            "aliases": ["CVE-2024-DUPE"],
            "affected": [
                {
                    "package": {"name": "my-pkg", "ecosystem": "npm"},
                    "ranges": [{"type": "SEMVER", "events": [{"introduced": "0"}, {"fixed": "1.0.0"}]}],
                    "versions": ["0.1.0", "0.2.0"],
                }
            ],
        }

        def side_effect(osv_id):
            if osv_id == "CVE-2024-DUPE":
                return cve_record
            if osv_id == "GHSA-xxxx-xxxx-xxxx":
                return ghsa_record
            return None

        mock_fetch.side_effect = side_effect
        results = _fetch_osv_version_ranges("CVE-2024-DUPE")
        # Should be deduplicated to 1 entry
        assert len(results) == 1

    @patch("manus_agent.tools.get_version_range._fetch_osv_single")
    def test_max_alias_follows_cap(self, mock_fetch):
        """Should cap alias follows at _MAX_ALIAS_FOLLOWS."""
        aliases = [f"GHSA-xxxx-xxxx-{i:04d}" for i in range(20)]
        cve_record = {
            "id": "CVE-2024-MANY",
            "aliases": aliases,
            "affected": [],
        }
        mock_fetch.return_value = cve_record  # all aliases return same empty record
        _fetch_osv_version_ranges("CVE-2024-MANY")
        # Should have called: 1 (CVE) + _MAX_ALIAS_FOLLOWS (GHSA) times
        from manus_agent.tools.get_version_range import _MAX_ALIAS_FOLLOWS

        assert mock_fetch.call_count <= 1 + _MAX_ALIAS_FOLLOWS + 1

    @patch("manus_agent.tools.get_version_range._fetch_osv_single")
    def test_no_package_name_skipped(self, mock_fetch):
        record = {
            "id": "CVE-2024-NONAME",
            "aliases": [],
            "affected": [
                {
                    "package": {"name": "", "ecosystem": "npm"},
                    "ranges": [],
                    "versions": [],
                }
            ],
        }
        mock_fetch.return_value = record
        results = _fetch_osv_version_ranges("CVE-2024-NONAME")
        assert results == []

    @patch("manus_agent.tools.get_version_range._fetch_osv_single")
    def test_severity_from_affected(self, mock_fetch):
        mock_fetch.return_value = OSV_LOG4J_GHSA_RECORD
        results = _fetch_osv_version_ranges("GHSA-jfh8-c2jp-5v3q")
        assert len(results) == 1
        assert len(results[0]["severity"]) > 0

    @patch("manus_agent.tools.get_version_range._fetch_osv_single")
    def test_versions_capped_at_100(self, mock_fetch):
        record = {
            "id": "CVE-2024-LOTS",
            "aliases": [],
            "affected": [
                {
                    "package": {"name": "big-pkg", "ecosystem": "PyPI"},
                    "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}]}],
                    "versions": [f"1.0.{i}" for i in range(200)],
                }
            ],
        }
        mock_fetch.return_value = record
        results = _fetch_osv_version_ranges("CVE-2024-LOTS")
        assert len(results[0]["affected_versions"]) == 100


# ═══════════════════════════════════════════════════════════════════════════
# Merge results
# ═══════════════════════════════════════════════════════════════════════════


class TestMergeResults:
    """Tests for _merge_results."""

    def test_osv_preferred_over_nvd(self):
        nvd = [
            {
                "source": "nvd",
                "vendor": "apache",
                "product": "log4j",
                "ecosystem": "Maven",
                "vulnerable_range": ">=2.0, <2.15.0",
                "first_patched": "2.15.0",
            }
        ]
        osv = [
            {
                "source": "osv",
                "ecosystem": "Maven",
                "package": "org.apache.logging.log4j:log4j-core",
                "vulnerable_range": ">=2.0-beta9, <2.15.0",
                "first_patched": "2.15.0",
                "last_affected": "",
                "affected_versions": ["2.0", "2.14.1"],
                "range_details": [],
                "severity": [],
            }
        ]
        merged = _merge_results(nvd, osv, "auto")
        # OSV entry should be included; NVD log4j should be covered
        osv_entries = [m for m in merged if m["source"] == "osv"]
        assert len(osv_entries) >= 1
        assert osv_entries[0]["package"] == "org.apache.logging.log4j:log4j-core"

    def test_nvd_added_when_not_in_osv(self):
        nvd = [
            {
                "source": "nvd",
                "vendor": "acme",
                "product": "unique-thing",
                "ecosystem": "unknown",
                "vulnerable_range": ">=1.0, <2.0",
                "first_patched": "2.0",
            }
        ]
        osv = []
        merged = _merge_results(nvd, osv, "auto")
        assert len(merged) == 1
        assert merged[0]["source"] == "nvd"

    def test_ecosystem_filter_osv(self):
        osv = [
            {
                "source": "osv",
                "ecosystem": "Maven",
                "package": "log4j-core",
                "vulnerable_range": ">=2.0, <2.15.0",
                "first_patched": "2.15.0",
                "last_affected": "",
                "affected_versions": [],
                "range_details": [],
                "severity": [],
            },
            {
                "source": "osv",
                "ecosystem": "PyPI",
                "package": "some-pypi-pkg",
                "vulnerable_range": ">=1.0, <2.0",
                "first_patched": "2.0",
                "last_affected": "",
                "affected_versions": [],
                "range_details": [],
                "severity": [],
            },
        ]
        merged = _merge_results([], osv, "maven")
        assert len(merged) == 1
        assert merged[0]["ecosystem"] == "Maven"

    def test_ecosystem_filter_auto_returns_all(self):
        osv = [
            {
                "source": "osv",
                "ecosystem": "Maven",
                "package": "a",
                "vulnerable_range": "",
                "first_patched": "",
                "last_affected": "",
                "affected_versions": [],
                "range_details": [],
                "severity": [],
            },
            {
                "source": "osv",
                "ecosystem": "PyPI",
                "package": "b",
                "vulnerable_range": "",
                "first_patched": "",
                "last_affected": "",
                "affected_versions": [],
                "range_details": [],
                "severity": [],
            },
        ]
        merged = _merge_results([], osv, "auto")
        assert len(merged) == 2

    def test_empty_both_sources(self):
        merged = _merge_results([], [], "auto")
        assert merged == []

    def test_nvd_dedup_with_osv_coverage(self):
        """NVD entry for 'log4j' should be suppressed when OSV covers 'log4j-core'."""
        nvd = [
            {
                "source": "nvd",
                "vendor": "apache",
                "product": "log4j",
                "ecosystem": "Maven",
                "vulnerable_range": ">=2.0, <2.15.0",
                "first_patched": "2.15.0",
            }
        ]
        osv = [
            {
                "source": "osv",
                "ecosystem": "Maven",
                "package": "org.apache.logging.log4j:log4j-core",
                "vulnerable_range": ">=2.0-beta9, <2.15.0",
                "first_patched": "2.15.0",
                "last_affected": "",
                "affected_versions": [],
                "range_details": [],
                "severity": [],
            }
        ]
        merged = _merge_results(nvd, osv, "auto")
        nvd_entries = [m for m in merged if m["source"] == "nvd"]
        # NVD log4j should be covered by OSV log4j-core (product name in key)
        assert len(nvd_entries) == 0

    def test_nvd_unknown_ecosystem_passes_auto_filter(self):
        nvd = [
            {
                "source": "nvd",
                "vendor": "vendor",
                "product": "thing",
                "ecosystem": "unknown",
                "vulnerable_range": "all versions",
                "first_patched": "",
            }
        ]
        merged = _merge_results(nvd, [], "auto")
        assert len(merged) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Strands handler (end-to-end)
# ═══════════════════════════════════════════════════════════════════════════


class TestGetVersionRangeHandler:
    """End-to-end tests for the Strands tool handler."""

    def test_invalid_cve_format(self, tool_use):
        result = get_version_range(tool_use(cve_id="not-a-cve"))
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_empty_cve_id(self, tool_use):
        result = get_version_range(tool_use(cve_id=""))
        assert result["status"] == "error"

    def test_numeric_cve_id(self, tool_use):
        result = get_version_range({"toolUseId": "t1", "input": {"cve_id": 12345}})
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_success_with_results(self, mock_nvd, mock_osv, tool_use):
        mock_nvd.return_value = []
        mock_osv.return_value = [
            {
                "source": "osv",
                "ecosystem": "Maven",
                "package": "log4j-core",
                "vulnerable_range": ">=2.0-beta9, <2.15.0",
                "first_patched": "2.15.0",
                "last_affected": "",
                "affected_versions": ["2.0", "2.14.1"],
                "range_details": [],
                "severity": ["CVSS:3.1/AV:N/AC:L"],
            }
        ]
        result = get_version_range(tool_use())
        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert data["cve_id"] == "CVE-2021-44228"
        assert data["packages_affected"] == 1
        assert data["packages"][0]["package"] == "log4j-core"
        assert data["packages"][0]["first_patched"] == "2.15.0"

    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_no_results_found(self, mock_nvd, mock_osv, tool_use):
        mock_nvd.return_value = []
        mock_osv.return_value = []
        result = get_version_range(tool_use())
        assert result["status"] == "success"
        assert "No affected version range data found" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_ecosystem_filter_passed(self, mock_nvd, mock_osv, tool_use):
        mock_nvd.return_value = []
        mock_osv.return_value = [
            {
                "source": "osv",
                "ecosystem": "Maven",
                "package": "x",
                "vulnerable_range": ">=1, <2",
                "first_patched": "2",
                "last_affected": "",
                "affected_versions": [],
                "range_details": [],
                "severity": [],
            }
        ]
        result = get_version_range(tool_use(ecosystem="maven"))
        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert data["ecosystem_filter"] == "maven"

    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_cve_id_normalised_to_uppercase(self, mock_nvd, mock_osv, tool_use):
        mock_nvd.return_value = []
        mock_osv.return_value = []
        result = get_version_range(tool_use(cve_id="cve-2021-44228"))
        assert result["status"] == "success"
        assert "CVE-2021-44228" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_output_includes_affected_versions(self, mock_nvd, mock_osv, tool_use):
        mock_nvd.return_value = []
        mock_osv.return_value = [
            {
                "source": "osv",
                "ecosystem": "npm",
                "package": "pkg",
                "vulnerable_range": ">=0, <1.0",
                "first_patched": "1.0",
                "last_affected": "0.9.9",
                "affected_versions": [f"0.{i}" for i in range(60)],
                "range_details": [],
                "severity": [],
            }
        ]
        result = get_version_range(tool_use())
        data = result["content"][0]["json"]
        pkg = data["packages"][0]
        # Versions should be capped at 50 in output
        assert pkg["affected_versions_count"] == 60
        assert len(pkg["affected_versions"]) == 50

    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_output_includes_cpe(self, mock_nvd, mock_osv, tool_use):
        mock_nvd.return_value = [
            {
                "source": "nvd",
                "vendor": "v",
                "product": "p",
                "ecosystem": "unknown",
                "vulnerable_range": "all versions",
                "first_patched": "",
                "cpe": "cpe:2.3:a:v:p:*:*:*:*:*:*:*:*",
            }
        ]
        mock_osv.return_value = []
        result = get_version_range(tool_use())
        data = result["content"][0]["json"]
        assert "cpe" in data["packages"][0]

    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_tool_use_id_propagated(self, mock_nvd, mock_osv):
        mock_nvd.return_value = []
        mock_osv.return_value = []
        tu = {"toolUseId": "custom-id-42", "input": {"cve_id": "CVE-2024-0000"}}
        result = get_version_range(tu)
        assert result["toolUseId"] == "custom-id-42"


# ═══════════════════════════════════════════════════════════════════════════
# HTTP retry logic
# ═══════════════════════════════════════════════════════════════════════════


class TestGetWithRetry:
    """Tests for _get_with_retry."""

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_success_first_try(self, mock_get):
        mock_get.return_value = _mock_response({"ok": True})
        from manus_agent.tools.get_version_range import _get_with_retry

        resp = _get_with_retry("https://example.com")
        assert resp.json() == {"ok": True}

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retry_on_429(self, mock_get):
        err_resp = MagicMock(spec=requests.Response)
        err_resp.status_code = 429
        ok_resp = _mock_response({"ok": True})
        mock_get.side_effect = [err_resp, ok_resp]

        from manus_agent.tools.get_version_range import _get_with_retry

        resp = _get_with_retry("https://example.com")
        assert resp.json() == {"ok": True}
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_non_retryable_error_fails_immediately(self, mock_get):
        mock_get.return_value = _mock_error_response(404)

        from manus_agent.tools.get_version_range import _get_with_retry

        with pytest.raises(requests.exceptions.HTTPError):
            _get_with_retry("https://example.com")
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_connection_error_retries(self, mock_get):
        ok_resp = _mock_response({"ok": True})
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("timeout"),
            ok_resp,
        ]

        from manus_agent.tools.get_version_range import _get_with_retry

        resp = _get_with_retry("https://example.com")
        assert resp.json() == {"ok": True}

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_all_retries_exhausted(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("timeout")

        from manus_agent.tools.get_version_range import _get_with_retry

        with pytest.raises(requests.exceptions.ConnectionError):
            _get_with_retry("https://example.com")


# ═══════════════════════════════════════════════════════════════════════════
# OSV single fetch
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchOsvSingle:
    """Tests for _fetch_osv_single."""

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_success(self, mock_get):
        mock_get.return_value = _mock_response({"id": "CVE-2024-0001"})
        from manus_agent.tools.get_version_range import _fetch_osv_single

        result = _fetch_osv_single("CVE-2024-0001")
        assert result["id"] == "CVE-2024-0001"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_failure_returns_none(self, mock_get):
        mock_get.side_effect = requests.exceptions.HTTPError("404")
        from manus_agent.tools.get_version_range import _fetch_osv_single

        result = _fetch_osv_single("CVE-9999-0000")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# CLI subcommand
# ═══════════════════════════════════════════════════════════════════════════


class TestVersionRangeCli:
    """Tests for the version-range CLI subcommand."""

    @patch("manus_agent.tools.get_version_range._merge_results")
    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_text_output(self, mock_nvd, mock_osv, mock_merge, capsys):
        mock_nvd.return_value = []
        mock_osv.return_value = []
        mock_merge.return_value = [
            {
                "source": "osv",
                "ecosystem": "Maven",
                "package": "log4j-core",
                "vulnerable_range": ">=2.0-beta9, <2.15.0",
                "first_patched": "2.15.0",
                "last_affected": "",
                "affected_versions": ["2.0", "2.14.1"],
                "severity": ["CVSS:3.1/AV:N"],
                "cpe": "",
            }
        ]
        from manus_agent.cli import _run_version_range

        rc = _run_version_range(["CVE-2021-44228"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2021-44228" in out
        assert "log4j-core" in out
        assert "2.15.0" in out

    @patch("manus_agent.tools.get_version_range._merge_results")
    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_json_output(self, mock_nvd, mock_osv, mock_merge, capsys):
        mock_nvd.return_value = []
        mock_osv.return_value = []
        mock_merge.return_value = [
            {
                "source": "osv",
                "ecosystem": "Maven",
                "package": "log4j-core",
                "vulnerable_range": ">=2.0-beta9, <2.15.0",
                "first_patched": "2.15.0",
                "last_affected": "",
                "affected_versions": ["2.0"],
                "severity": [],
                "cpe": "",
            }
        ]
        from manus_agent.cli import _run_version_range

        rc = _run_version_range(["CVE-2021-44228", "--output", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["cve_id"] == "CVE-2021-44228"
        assert data["packages_affected"] == 1
        assert data["packages"][0]["first_patched"] == "2.15.0"

    @patch("manus_agent.tools.get_version_range._merge_results")
    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_ecosystem_filter(self, mock_nvd, mock_osv, mock_merge, capsys):
        mock_nvd.return_value = []
        mock_osv.return_value = []
        mock_merge.return_value = [
            {
                "source": "osv",
                "ecosystem": "PyPI",
                "package": "django",
                "vulnerable_range": ">=3.0, <3.2.6",
                "first_patched": "3.2.6",
                "last_affected": "",
                "affected_versions": [],
                "severity": [],
                "cpe": "",
            }
        ]
        from manus_agent.cli import _run_version_range

        rc = _run_version_range(["CVE-2024-5678", "--ecosystem", "pypi"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "pypi" in out.lower()

    @patch("manus_agent.tools.get_version_range._merge_results")
    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_no_results_exits_1(self, mock_nvd, mock_osv, mock_merge, capsys):
        mock_nvd.return_value = []
        mock_osv.return_value = []
        mock_merge.return_value = []
        from manus_agent.cli import _run_version_range

        rc = _run_version_range(["CVE-2024-0000"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "No affected version range data found" in err

    def test_invalid_cve_exits_1(self, capsys):
        from manus_agent.cli import _run_version_range

        rc = _run_version_range(["not-a-cve"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid CVE ID" in err

    @patch("manus_agent.tools.get_version_range._merge_results")
    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_text_output_with_cpe(self, mock_nvd, mock_osv, mock_merge, capsys):
        mock_nvd.return_value = []
        mock_osv.return_value = []
        mock_merge.return_value = [
            {
                "source": "nvd",
                "ecosystem": "unknown",
                "package": "vendor/product",
                "vulnerable_range": "all versions",
                "first_patched": "",
                "last_affected": "",
                "affected_versions": [],
                "severity": [],
                "cpe": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
            }
        ]
        from manus_agent.cli import _run_version_range

        rc = _run_version_range(["CVE-2024-0003"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CPE" in out

    @patch("manus_agent.tools.get_version_range._merge_results")
    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_text_output_with_last_affected(self, mock_nvd, mock_osv, mock_merge, capsys):
        mock_nvd.return_value = []
        mock_osv.return_value = []
        mock_merge.return_value = [
            {
                "source": "osv",
                "ecosystem": "PyPI",
                "package": "old-pkg",
                "vulnerable_range": ">=1.0, <=1.9.9",
                "first_patched": "",
                "last_affected": "1.9.9",
                "affected_versions": [],
                "severity": [],
                "cpe": "",
            }
        ]
        from manus_agent.cli import _run_version_range

        rc = _run_version_range(["CVE-2024-0001"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Last affected" in out
        assert "1.9.9" in out

    @patch("manus_agent.tools.get_version_range._merge_results")
    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_text_output_many_versions_truncated(self, mock_nvd, mock_osv, mock_merge, capsys):
        mock_nvd.return_value = []
        mock_osv.return_value = []
        versions = [f"1.0.{i}" for i in range(30)]
        mock_merge.return_value = [
            {
                "source": "osv",
                "ecosystem": "npm",
                "package": "big-pkg",
                "vulnerable_range": ">=1.0.0, <2.0.0",
                "first_patched": "2.0.0",
                "last_affected": "",
                "affected_versions": versions,
                "severity": [],
                "cpe": "",
            }
        ]
        from manus_agent.cli import _run_version_range

        rc = _run_version_range(["CVE-2024-0004"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "and 10 more" in out

    @patch("manus_agent.tools.get_version_range._merge_results")
    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_json_output_with_severity(self, mock_nvd, mock_osv, mock_merge, capsys):
        mock_nvd.return_value = []
        mock_osv.return_value = []
        mock_merge.return_value = [
            {
                "source": "osv",
                "ecosystem": "Maven",
                "package": "pkg",
                "vulnerable_range": ">=1, <2",
                "first_patched": "2",
                "last_affected": "1.9",
                "affected_versions": ["1.0", "1.5"],
                "severity": ["CVSS:3.1/AV:N"],
                "cpe": "",
            }
        ]
        from manus_agent.cli import _run_version_range

        rc = _run_version_range(["CVE-2024-0005", "--output", "json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["packages"][0]["severity"] == ["CVSS:3.1/AV:N"]
        assert data["packages"][0]["last_affected"] == "1.9"

    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_nvd_fetch_error_handled(self, mock_nvd, capsys):
        mock_nvd.side_effect = RuntimeError("NVD down")
        from manus_agent.cli import _run_version_range

        with patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges") as mock_osv:
            mock_osv.return_value = []
            rc = _run_version_range(["CVE-2024-0006"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "NVD API request failed" in err

    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_osv_fetch_error_handled(self, mock_nvd, mock_osv, capsys):
        mock_nvd.return_value = []
        mock_osv.side_effect = RuntimeError("OSV down")
        from manus_agent.cli import _run_version_range

        rc = _run_version_range(["CVE-2024-0007"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "OSV.dev API request failed" in err


# ═══════════════════════════════════════════════════════════════════════════
# CLI parser
# ═══════════════════════════════════════════════════════════════════════════


class TestVersionRangeParser:
    """Tests for _build_version_range_parser."""

    def test_parser_prog(self):
        from manus_agent.cli import _build_version_range_parser

        p = _build_version_range_parser()
        assert p.prog == "manus-agent version-range"

    def test_parser_cve_id_required(self):
        from manus_agent.cli import _build_version_range_parser

        p = _build_version_range_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_parser_defaults(self):
        from manus_agent.cli import _build_version_range_parser

        p = _build_version_range_parser()
        args = p.parse_args(["CVE-2021-44228"])
        assert args.cve_id == "CVE-2021-44228"
        assert args.ecosystem == "auto"
        assert args.output == "text"

    def test_parser_ecosystem_choices(self):
        from manus_agent.cli import _build_version_range_parser

        p = _build_version_range_parser()
        for eco in ["auto", "pypi", "npm", "maven", "go", "crates.io", "rubygems", "nuget", "packagist"]:
            args = p.parse_args(["CVE-2021-44228", "--ecosystem", eco])
            assert args.ecosystem == eco

    def test_parser_invalid_ecosystem(self):
        from manus_agent.cli import _build_version_range_parser

        p = _build_version_range_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["CVE-2021-44228", "--ecosystem", "invalid"])

    def test_parser_output_choices(self):
        from manus_agent.cli import _build_version_range_parser

        p = _build_version_range_parser()
        for fmt in ["text", "json"]:
            args = p.parse_args(["CVE-2021-44228", "--output", fmt])
            assert args.output == fmt


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand registration
# ═══════════════════════════════════════════════════════════════════════════


class TestSubcommandRegistration:
    """Tests that version-range is registered in _SUBCOMMANDS."""

    def test_in_subcommands_set(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "version-range" in _SUBCOMMANDS

    def test_main_dispatches_version_range(self):
        """Verify main() routes 'version-range' to _run_version_range."""

        with patch("manus_agent.cli._run_version_range", return_value=0) as mock_run:
            with patch("sys.argv", ["manus-agent", "version-range", "CVE-2021-44228"]):
                from manus_agent.cli import main

                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0
            mock_run.assert_called_once_with(["CVE-2021-44228"])


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_cve_id_whitespace_trimmed(self, tool_use):
        with patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges") as mn:
            with patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges") as mo:
                mn.return_value = []
                mo.return_value = []
                result = get_version_range(tool_use(cve_id="  CVE-2021-44228  "))
                assert result["status"] == "success"
                # Should have normalised to uppercase
                assert "CVE-2021-44228" in str(result["content"])

    def test_ecosystem_default_is_auto(self, tool_use):
        with patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges") as mn:
            with patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges") as mo:
                mn.return_value = []
                mo.return_value = []
                result = get_version_range(tool_use())
                assert result["status"] == "success"

    @patch("manus_agent.tools.get_version_range._fetch_osv_version_ranges")
    @patch("manus_agent.tools.get_version_range._fetch_nvd_version_ranges")
    def test_multiple_packages(self, mock_nvd, mock_osv, tool_use):
        mock_nvd.return_value = []
        mock_osv.return_value = [
            {
                "source": "osv",
                "ecosystem": "Maven",
                "package": "log4j-core",
                "vulnerable_range": ">=2.0, <2.15.0",
                "first_patched": "2.15.0",
                "last_affected": "",
                "affected_versions": [],
                "range_details": [],
                "severity": [],
            },
            {
                "source": "osv",
                "ecosystem": "Maven",
                "package": "log4j-api",
                "vulnerable_range": ">=2.0, <2.15.0",
                "first_patched": "2.15.0",
                "last_affected": "",
                "affected_versions": [],
                "range_details": [],
                "severity": [],
            },
        ]
        result = get_version_range(tool_use())
        data = result["content"][0]["json"]
        assert data["packages_affected"] == 2
