#!/usr/bin/env python3
"""Comprehensive test suite for get_version_range tool and version-range CLI.

All HTTP calls are mocked — no real network traffic.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the package is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable retry delays in all tests."""
    monkeypatch.setenv("VR_MAX_RETRIES", "1")
    monkeypatch.setenv("VR_RETRY_BASE_DELAY", "0")


@pytest.fixture()
def _no_nvd_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no NVD API key is set."""
    monkeypatch.delenv("NVD_API_KEY", raising=False)


@pytest.fixture()
def _with_nvd_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set a fake NVD API key."""
    monkeypatch.setenv("NVD_API_KEY", "fake-nvd-key-12345")


# ---------------------------------------------------------------------------
# Sample NVD response factory
# ---------------------------------------------------------------------------
def _make_nvd_response(
    cve_id: str = "CVE-2021-44228",
    configurations: list[dict[str, Any]] | None = None,
    descriptions: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if configurations is None:
        configurations = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                                "versionStartIncluding": "2.0.0",
                                "versionEndExcluding": "2.15.0",
                            },
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                                "versionStartIncluding": "2.0-beta9",
                                "versionEndExcluding": "2.3.1",
                            },
                        ]
                    }
                ]
            }
        ]
    if descriptions is None:
        descriptions = [
            {
                "lang": "en",
                "value": "Apache Log4j2 2.0-beta9 through 2.15.0 JNDI features do not protect against attacker controlled LDAP.",
            }
        ]
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "id": cve_id,
                    "configurations": configurations,
                    "descriptions": descriptions,
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# Sample OSV response factory
# ---------------------------------------------------------------------------
def _make_osv_response(
    osv_id: str = "CVE-2021-44228",
    affected: list[dict[str, Any]] | None = None,
    aliases: list[str] | None = None,
) -> dict[str, Any]:
    if affected is None:
        affected = [
            {
                "package": {"ecosystem": "Maven", "name": "org.apache.logging.log4j:log4j-core"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "2.0-beta9"},
                            {"fixed": "2.3.1"},
                        ],
                    },
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "2.4.0"},
                            {"fixed": "2.12.3"},
                        ],
                    },
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "2.13.0"},
                            {"fixed": "2.15.0"},
                        ],
                    },
                ],
                "versions": ["2.0", "2.0-beta9", "2.1", "2.2", "2.14.1"],
            },
        ]
    if aliases is None:
        aliases = ["CVE-2021-44228", "GHSA-jfh8-c2jp-5v3q"]
    return {"id": osv_id, "affected": affected, "aliases": aliases}


def _mock_response(status_code: int = 200, json_data: Any = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = __import__("requests").exceptions.HTTPError(
            f"HTTP {status_code}", response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ===================================================================
# Tests: _parse_cpe_uri
# ===================================================================
class TestParseCpeUri:
    def test_full_cpe23(self) -> None:
        from manus_agent.tools.get_version_range import _parse_cpe_uri

        result = _parse_cpe_uri("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*")
        assert result["vendor"] == "apache"
        assert result["product"] == "log4j"
        assert result["version"] == "2.14.1"
        assert result["part"] == "a"

    def test_wildcard_version(self) -> None:
        from manus_agent.tools.get_version_range import _parse_cpe_uri

        result = _parse_cpe_uri("cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*")
        assert result["version"] == "*"

    def test_short_cpe(self) -> None:
        from manus_agent.tools.get_version_range import _parse_cpe_uri

        result = _parse_cpe_uri("cpe:2.3:a")
        assert result["vendor"] == "*"
        assert result["product"] == "*"
        assert result["version"] == "*"


# ===================================================================
# Tests: _build_range_expression
# ===================================================================
class TestBuildRangeExpression:
    def test_exact_version(self) -> None:
        from manus_agent.tools.get_version_range import _build_range_expression

        assert _build_range_expression({"exact_version": "1.2.3"}) == "== 1.2.3"

    def test_start_including_end_excluding(self) -> None:
        from manus_agent.tools.get_version_range import _build_range_expression

        result = _build_range_expression(
            {"exact_version": None, "version_start_including": "2.0", "version_end_excluding": "2.15.0"}
        )
        assert result == ">= 2.0, < 2.15.0"

    def test_start_excluding_end_including(self) -> None:
        from manus_agent.tools.get_version_range import _build_range_expression

        result = _build_range_expression(
            {"exact_version": None, "version_start_excluding": "1.0", "version_end_including": "3.0"}
        )
        assert result == "> 1.0, <= 3.0"

    def test_only_end_excluding(self) -> None:
        from manus_agent.tools.get_version_range import _build_range_expression

        result = _build_range_expression({"exact_version": None, "version_end_excluding": "5.0"})
        assert result == "< 5.0"

    def test_no_bounds(self) -> None:
        from manus_agent.tools.get_version_range import _build_range_expression

        result = _build_range_expression({"exact_version": None})
        assert result == "all versions"


# ===================================================================
# Tests: _camel_to_snake
# ===================================================================
class TestCamelToSnake:
    def test_simple(self) -> None:
        from manus_agent.tools.get_version_range import _camel_to_snake

        assert _camel_to_snake("versionStartIncluding") == "version_start_including"

    def test_already_snake(self) -> None:
        from manus_agent.tools.get_version_range import _camel_to_snake

        assert _camel_to_snake("hello_world") == "hello_world"

    def test_single_word(self) -> None:
        from manus_agent.tools.get_version_range import _camel_to_snake

        assert _camel_to_snake("version") == "version"


# ===================================================================
# Tests: _guess_ecosystem_from_cpe
# ===================================================================
class TestGuessEcosystem:
    def test_python_vendor(self) -> None:
        from manus_agent.tools.get_version_range import _guess_ecosystem_from_cpe

        assert _guess_ecosystem_from_cpe("python", "requests") == "PyPI"

    def test_nodejs_product(self) -> None:
        from manus_agent.tools.get_version_range import _guess_ecosystem_from_cpe

        assert _guess_ecosystem_from_cpe("unknown_vendor", "nodejs") == "npm"

    def test_apache_vendor(self) -> None:
        from manus_agent.tools.get_version_range import _guess_ecosystem_from_cpe

        assert _guess_ecosystem_from_cpe("apache", "unknown_product") == "Maven"

    def test_unknown(self) -> None:
        from manus_agent.tools.get_version_range import _guess_ecosystem_from_cpe

        assert _guess_ecosystem_from_cpe("acme", "widget") is None

    def test_go_vendor(self) -> None:
        from manus_agent.tools.get_version_range import _guess_ecosystem_from_cpe

        assert _guess_ecosystem_from_cpe("golang", "something") == "Go"

    def test_rust_vendor(self) -> None:
        from manus_agent.tools.get_version_range import _guess_ecosystem_from_cpe

        assert _guess_ecosystem_from_cpe("rust", "tokio") == "crates.io"


# ===================================================================
# Tests: _pick_first_patched
# ===================================================================
class TestPickFirstPatched:
    def test_numeric_versions(self) -> None:
        from manus_agent.tools.get_version_range import _pick_first_patched

        assert _pick_first_patched(["2.15.0", "2.3.1", "2.12.3"]) == "2.3.1"

    def test_single_version(self) -> None:
        from manus_agent.tools.get_version_range import _pick_first_patched

        assert _pick_first_patched(["1.0.0"]) == "1.0.0"

    def test_empty_list(self) -> None:
        from manus_agent.tools.get_version_range import _pick_first_patched

        assert _pick_first_patched([]) == ""

    def test_mixed_formats(self) -> None:
        from manus_agent.tools.get_version_range import _pick_first_patched

        # Non-numeric segments pushed to end → numeric-prefixed wins
        result = _pick_first_patched(["3.0.0", "2.0.0-rc1", "1.5.0"])
        assert result == "1.5.0"


# ===================================================================
# Tests: _extract_cpe_ranges
# ===================================================================
class TestExtractCpeRanges:
    def test_basic_extraction(self) -> None:
        from manus_agent.tools.get_version_range import _extract_cpe_ranges

        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                                "versionStartIncluding": "1.0",
                                "versionEndExcluding": "2.0",
                            }
                        ]
                    }
                ]
            }
        ]
        result = _extract_cpe_ranges(configs)
        assert len(result) == 1
        assert result[0]["vendor"] == "vendor"
        assert result[0]["product"] == "product"
        assert result[0]["version_start_including"] == "1.0"
        assert result[0]["version_end_excluding"] == "2.0"
        assert ">= 1.0" in result[0]["range_expression"]

    def test_non_vulnerable_skipped(self) -> None:
        from manus_agent.tools.get_version_range import _extract_cpe_ranges

        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": False,
                                "criteria": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                                "versionStartIncluding": "1.0",
                            }
                        ]
                    }
                ]
            }
        ]
        result = _extract_cpe_ranges(configs)
        assert len(result) == 0

    def test_exact_version_in_cpe(self) -> None:
        from manus_agent.tools.get_version_range import _extract_cpe_ranges

        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:vendor:product:3.1.4:*:*:*:*:*:*:*",
                            }
                        ]
                    }
                ]
            }
        ]
        result = _extract_cpe_ranges(configs)
        assert len(result) == 1
        assert result[0]["exact_version"] == "3.1.4"
        assert result[0]["range_expression"] == "== 3.1.4"

    def test_empty_configurations(self) -> None:
        from manus_agent.tools.get_version_range import _extract_cpe_ranges

        assert _extract_cpe_ranges([]) == []
        assert _extract_cpe_ranges(None) == []

    def test_multiple_nodes(self) -> None:
        from manus_agent.tools.get_version_range import _extract_cpe_ranges

        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:v1:p1:*:*:*:*:*:*:*:*",
                                "versionEndExcluding": "1.0",
                            }
                        ]
                    },
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:v2:p2:*:*:*:*:*:*:*:*",
                                "versionEndExcluding": "2.0",
                            }
                        ]
                    },
                ]
            }
        ]
        result = _extract_cpe_ranges(configs)
        assert len(result) == 2


# ===================================================================
# Tests: _parse_osv_ranges
# ===================================================================
class TestParseOsvRanges:
    def test_basic_parsing(self) -> None:
        from manus_agent.tools.get_version_range import _parse_osv_ranges

        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "requests"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "0"},
                            {"fixed": "2.28.1"},
                        ],
                    }
                ],
                "versions": ["2.27.0", "2.27.1", "2.28.0"],
            }
        ]
        result = _parse_osv_ranges(affected)
        assert len(result) == 1
        assert result[0]["ecosystem"] == "PyPI"
        assert result[0]["package"] == "requests"
        assert result[0]["first_patched"] == "2.28.1"
        assert result[0]["affected_version_count"] == 3

    def test_multiple_fixed_picks_lowest(self) -> None:
        from manus_agent.tools.get_version_range import _parse_osv_ranges

        affected = [
            {
                "package": {"ecosystem": "Maven", "name": "log4j"},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "2.0"},
                            {"fixed": "2.15.0"},
                        ],
                    },
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "2.0-beta9"},
                            {"fixed": "2.3.1"},
                        ],
                    },
                ],
                "versions": [],
            }
        ]
        result = _parse_osv_ranges(affected)
        assert result[0]["first_patched"] == "2.3.1"

    def test_last_affected(self) -> None:
        from manus_agent.tools.get_version_range import _parse_osv_ranges

        affected = [
            {
                "package": {"ecosystem": "npm", "name": "express"},
                "ranges": [
                    {
                        "type": "SEMVER",
                        "events": [
                            {"introduced": "0"},
                            {"last_affected": "4.17.20"},
                        ],
                    }
                ],
                "versions": [],
            }
        ]
        result = _parse_osv_ranges(affected)
        assert result[0]["ranges"][0]["last_affected"] == ["4.17.20"]
        assert result[0]["first_patched"] is None

    def test_empty_affected(self) -> None:
        from manus_agent.tools.get_version_range import _parse_osv_ranges

        assert _parse_osv_ranges([]) == []
        assert _parse_osv_ranges(None) == []

    def test_skip_non_dict_entries(self) -> None:
        from manus_agent.tools.get_version_range import _parse_osv_ranges

        result = _parse_osv_ranges(["not_a_dict", None, 42])
        assert result == []

    def test_skip_entries_without_package(self) -> None:
        from manus_agent.tools.get_version_range import _parse_osv_ranges

        affected = [{"package": {}, "ranges": [], "versions": []}]
        result = _parse_osv_ranges(affected)
        assert result == []

    def test_versions_sample_capped(self) -> None:
        from manus_agent.tools.get_version_range import _parse_osv_ranges

        versions = [f"1.{i}.0" for i in range(50)]
        affected = [
            {
                "package": {"ecosystem": "PyPI", "name": "big"},
                "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "99.0"}]}],
                "versions": versions,
            }
        ]
        result = _parse_osv_ranges(affected)
        assert result[0]["affected_version_count"] == 50
        assert len(result[0]["affected_versions_sample"]) == 20


# ===================================================================
# Tests: fetch_nvd_ranges
# ===================================================================
class TestFetchNvdRanges:
    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_success(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_nvd_ranges

        mock_get.return_value = _mock_response(200, _make_nvd_response())
        result = fetch_nvd_ranges("CVE-2021-44228")
        assert result["found"] is True
        assert len(result["cpe_ranges"]) == 2
        assert result["descriptions"][0].startswith("Apache Log4j2")

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_not_found(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_nvd_ranges

        mock_get.return_value = _mock_response(404)
        result = fetch_nvd_ranges("CVE-9999-99999")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_no_vulnerabilities(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_nvd_ranges

        mock_get.return_value = _mock_response(200, {"vulnerabilities": []})
        result = fetch_nvd_ranges("CVE-2099-00001")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_network_error(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_nvd_ranges

        mock_get.side_effect = __import__("requests").exceptions.ConnectionError("fail")
        result = fetch_nvd_ranges("CVE-2021-44228")
        assert result["found"] is False
        assert "error" in result

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_invalid_json(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_nvd_ranges

        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = resp
        result = fetch_nvd_ranges("CVE-2021-44228")
        assert result["found"] is False
        assert "invalid JSON" in result.get("error", "")

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_http_error(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_nvd_ranges

        mock_get.return_value = _mock_response(500)
        result = fetch_nvd_ranges("CVE-2021-44228")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_no_configurations(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_nvd_ranges

        mock_get.return_value = _mock_response(200, _make_nvd_response(configurations=[]))
        result = fetch_nvd_ranges("CVE-2021-44228")
        assert result["found"] is True
        assert result["cpe_ranges"] == []

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_ecosystem_hint_annotated(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_nvd_ranges

        mock_get.return_value = _mock_response(200, _make_nvd_response())
        result = fetch_nvd_ranges("CVE-2021-44228")
        # apache:log4j should hint Maven
        hints = [r.get("ecosystem_hint") for r in result["cpe_ranges"]]
        assert "Maven" in hints


# ===================================================================
# Tests: fetch_osv_ranges
# ===================================================================
class TestFetchOsvRanges:
    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_success(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_osv_ranges

        mock_get.return_value = _mock_response(200, _make_osv_response())
        result = fetch_osv_ranges("CVE-2021-44228")
        assert result["found"] is True
        assert len(result["packages"]) == 1
        assert result["packages"][0]["ecosystem"] == "Maven"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_not_found(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_osv_ranges

        mock_get.return_value = _mock_response(404)
        result = fetch_osv_ranges("CVE-9999-99999")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_network_error(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_osv_ranges

        mock_get.side_effect = __import__("requests").exceptions.Timeout("timeout")
        result = fetch_osv_ranges("CVE-2021-44228")
        assert result["found"] is False
        assert "error" in result

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_ecosystem_filter(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_osv_ranges

        mock_get.return_value = _mock_response(200, _make_osv_response())
        # Filter to PyPI — should get empty since the sample is Maven
        result = fetch_osv_ranges("CVE-2021-44228", ecosystem_filter="pypi")
        assert result["found"] is False
        assert result["packages"] == []

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_ecosystem_filter_match(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_osv_ranges

        mock_get.return_value = _mock_response(200, _make_osv_response())
        result = fetch_osv_ranges("CVE-2021-44228", ecosystem_filter="maven")
        assert result["found"] is True
        assert len(result["packages"]) == 1

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_ghsa_alias_follow(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_osv_ranges

        # Primary has no affected data, but has a GHSA alias
        primary = {"id": "CVE-2024-1234", "affected": [], "aliases": ["CVE-2024-1234", "GHSA-abcd-efgh-1234"]}
        ghsa_record = _make_osv_response(
            osv_id="GHSA-abcd-efgh-1234",
            affected=[
                {
                    "package": {"ecosystem": "npm", "name": "vulnerable-pkg"},
                    "ranges": [{"type": "SEMVER", "events": [{"introduced": "0"}, {"fixed": "1.2.3"}]}],
                    "versions": ["1.0.0", "1.1.0"],
                }
            ],
        )
        mock_get.side_effect = [
            _mock_response(200, primary),
            _mock_response(200, ghsa_record),
        ]
        result = fetch_osv_ranges("CVE-2024-1234")
        assert result["found"] is True
        assert result["packages"][0]["package"] == "vulnerable-pkg"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_ghsa_alias_failure_graceful(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_osv_ranges

        primary = {"id": "CVE-2024-9999", "affected": [], "aliases": ["GHSA-xxxx-yyyy-zzzz"]}
        mock_get.side_effect = [
            _mock_response(200, primary),
            Exception("network failure"),
        ]
        # Should not raise — graceful degradation
        result = fetch_osv_ranges("CVE-2024-9999")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_http_error_response(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_osv_ranges

        mock_get.return_value = _mock_response(500)
        result = fetch_osv_ranges("CVE-2021-44228")
        assert result["found"] is False

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_empty_cve_id(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_osv_ranges

        fetch_osv_ranges("")
        mock_get.assert_not_called()

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_auto_filter_returns_all(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_osv_ranges

        mock_get.return_value = _mock_response(200, _make_osv_response())
        result = fetch_osv_ranges("CVE-2021-44228", ecosystem_filter="auto")
        assert len(result["packages"]) == 1


# ===================================================================
# Tests: fetch_version_range (combined)
# ===================================================================
class TestFetchVersionRange:
    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_combined_success(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        mock_nvd.return_value = {
            "found": True,
            "cpe_ranges": [
                {
                    "vendor": "apache",
                    "product": "log4j",
                    "exact_version": None,
                    "version_start_including": "2.0",
                    "version_end_excluding": "2.15.0",
                    "range_expression": ">= 2.0, < 2.15.0",
                    "ecosystem_hint": "Maven",
                }
            ],
            "descriptions": ["Log4j vulnerability"],
        }
        mock_osv.return_value = {
            "found": True,
            "packages": [
                {
                    "ecosystem": "Maven",
                    "package": "log4j-core",
                    "ranges": [{"type": "ECOSYSTEM", "introduced": ["2.0"], "fixed": ["2.15.0"], "last_affected": []}],
                    "first_patched": "2.3.1",
                    "affected_version_count": 5,
                    "affected_versions_sample": [],
                }
            ],
            "aliases": ["CVE-2021-44228"],
        }

        result = fetch_version_range("CVE-2021-44228")
        assert result["found"] is True
        assert result["cve_id"] == "CVE-2021-44228"
        assert len(result["nvd_cpe_ranges"]) == 1
        assert len(result["osv_packages"]) == 1
        assert "Maven" in result["ecosystems"]
        assert result["first_patched"]["Maven:log4j-core"] == "2.3.1"
        assert result["description"] == "Log4j vulnerability"

    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_invalid_cve_id(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        result = fetch_version_range("not-a-cve")
        assert result["found"] is False
        assert "error" in result
        mock_nvd.assert_not_called()
        mock_osv.assert_not_called()

    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_empty_cve_id(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        result = fetch_version_range("")
        assert result["found"] is False
        mock_nvd.assert_not_called()

    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_nvd_only(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        mock_nvd.return_value = {
            "found": True,
            "cpe_ranges": [
                {
                    "vendor": "vendor",
                    "product": "product",
                    "exact_version": None,
                    "range_expression": "all versions",
                    "ecosystem_hint": None,
                }
            ],
            "descriptions": ["Some vuln"],
        }
        mock_osv.return_value = {"found": False, "packages": [], "aliases": []}

        result = fetch_version_range("CVE-2024-0001")
        assert result["found"] is True
        assert len(result["nvd_cpe_ranges"]) == 1
        assert result["osv_packages"] == []

    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_osv_only(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        mock_nvd.return_value = {"found": False, "cpe_ranges": [], "descriptions": []}
        mock_osv.return_value = {
            "found": True,
            "packages": [
                {
                    "ecosystem": "npm",
                    "package": "lodash",
                    "ranges": [],
                    "first_patched": "4.17.21",
                    "affected_version_count": 0,
                    "affected_versions_sample": [],
                }
            ],
            "aliases": [],
        }

        result = fetch_version_range("CVE-2024-0002")
        assert result["found"] is True
        assert result["first_patched"]["npm:lodash"] == "4.17.21"

    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_both_fail(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        mock_nvd.return_value = {"found": False, "cpe_ranges": [], "descriptions": [], "error": "NVD down"}
        mock_osv.return_value = {"found": False, "packages": [], "aliases": [], "error": "OSV down"}

        result = fetch_version_range("CVE-2024-0003")
        assert result["found"] is False
        assert len(result["errors"]) == 2

    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_ecosystem_filter(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        mock_nvd.return_value = {
            "found": True,
            "cpe_ranges": [
                {
                    "vendor": "apache",
                    "product": "log4j",
                    "exact_version": None,
                    "range_expression": ">= 2.0",
                    "ecosystem_hint": "Maven",
                },
                {
                    "vendor": "python",
                    "product": "requests",
                    "exact_version": None,
                    "range_expression": "< 2.28",
                    "ecosystem_hint": "PyPI",
                },
            ],
            "descriptions": [],
        }
        mock_osv.return_value = {"found": False, "packages": [], "aliases": []}

        result = fetch_version_range("CVE-2024-0004", ecosystem="pypi")
        # CPE ranges should be filtered to only PyPI
        assert len(result["nvd_cpe_ranges"]) == 1
        assert result["nvd_cpe_ranges"][0]["ecosystem_hint"] == "PyPI"

    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_message_with_first_patched(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        mock_nvd.return_value = {"found": True, "cpe_ranges": [], "descriptions": []}
        mock_osv.return_value = {
            "found": True,
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "flask",
                    "ranges": [],
                    "first_patched": "2.3.3",
                    "affected_version_count": 0,
                    "affected_versions_sample": [],
                }
            ],
            "aliases": [],
        }

        result = fetch_version_range("CVE-2024-0005")
        assert "First patched" in result["message"]
        assert "flask" in result["message"]

    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_no_errors_when_both_succeed(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        mock_nvd.return_value = {"found": True, "cpe_ranges": [], "descriptions": []}
        mock_osv.return_value = {"found": True, "packages": [], "aliases": []}

        result = fetch_version_range("CVE-2024-0006")
        assert result["errors"] is None


# ===================================================================
# Tests: Strands tool entry point
# ===================================================================
class TestStrandsTool:
    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_success(self, mock_fetch: MagicMock) -> None:
        from manus_agent.tools.get_version_range import get_version_range

        mock_fetch.return_value = {"cve_id": "CVE-2021-44228", "found": True, "message": "ok"}
        tool_use: dict[str, Any] = {
            "toolUseId": "test-001",
            "input": {"cve_id": "CVE-2021-44228"},
        }
        result = get_version_range(tool_use)
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-001"
        payload = json.loads(result["content"][0]["text"])
        assert payload["found"] is True

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_not_found(self, mock_fetch: MagicMock) -> None:
        from manus_agent.tools.get_version_range import get_version_range

        mock_fetch.return_value = {"cve_id": "CVE-9999-99999", "found": False}
        tool_use: dict[str, Any] = {
            "toolUseId": "test-002",
            "input": {"cve_id": "CVE-9999-99999"},
        }
        result = get_version_range(tool_use)
        assert result["status"] == "error"

    def test_invalid_cve_id(self) -> None:
        from manus_agent.tools.get_version_range import get_version_range

        tool_use: dict[str, Any] = {
            "toolUseId": "test-003",
            "input": {"cve_id": ""},
        }
        result = get_version_range(tool_use)
        assert result["status"] == "error"
        assert "Invalid" in result["content"][0]["text"]

    def test_missing_cve_id(self) -> None:
        from manus_agent.tools.get_version_range import get_version_range

        tool_use: dict[str, Any] = {
            "toolUseId": "test-004",
            "input": {},
        }
        result = get_version_range(tool_use)
        assert result["status"] == "error"

    def test_none_input(self) -> None:
        from manus_agent.tools.get_version_range import get_version_range

        tool_use: dict[str, Any] = {
            "toolUseId": "test-005",
            "input": None,
        }
        result = get_version_range(tool_use)
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_ecosystem_passed(self, mock_fetch: MagicMock) -> None:
        from manus_agent.tools.get_version_range import get_version_range

        mock_fetch.return_value = {"cve_id": "CVE-2021-44228", "found": True}
        tool_use: dict[str, Any] = {
            "toolUseId": "test-006",
            "input": {"cve_id": "CVE-2021-44228", "ecosystem": "pypi"},
        }
        get_version_range(tool_use)
        mock_fetch.assert_called_once_with("CVE-2021-44228", ecosystem="pypi")


# ===================================================================
# Tests: TOOL_SPEC
# ===================================================================
class TestToolSpec:
    def test_spec_structure(self) -> None:
        from manus_agent.tools.get_version_range import TOOL_SPEC

        assert TOOL_SPEC["name"] == "get_version_range"
        assert "inputSchema" in TOOL_SPEC
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "ecosystem" in schema["properties"]
        assert schema["required"] == ["cve_id"]


# ===================================================================
# Tests: _get_with_retry
# ===================================================================
class TestGetWithRetry:
    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_success_first_try(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import _get_with_retry

        mock_get.return_value = _mock_response(200, {"ok": True})
        resp = _get_with_retry("https://example.com")
        assert resp.status_code == 200

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retry_on_429(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import _get_with_retry

        mock_get.side_effect = [
            _mock_response(429),
            _mock_response(200, {"ok": True}),
        ]
        resp = _get_with_retry("https://example.com")
        assert resp.status_code == 200
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_retry_on_connection_error(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import _get_with_retry

        mock_get.side_effect = [
            __import__("requests").exceptions.ConnectionError("fail"),
            _mock_response(200, {"ok": True}),
        ]
        resp = _get_with_retry("https://example.com")
        assert resp.status_code == 200

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_exhausted_retries_raises(self, mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        from manus_agent.tools.get_version_range import _get_with_retry

        mock_get.side_effect = __import__("requests").exceptions.Timeout("timeout")
        with pytest.raises(__import__("requests").exceptions.Timeout):
            _get_with_retry("https://example.com")

    @patch("manus_agent.tools.get_version_range.requests.get")
    def test_non_retryable_error_not_retried(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import _get_with_retry

        mock_get.return_value = _mock_response(403)
        # 403 is not retryable, so it should return immediately
        resp = _get_with_retry("https://example.com")
        assert resp.status_code == 403
        assert mock_get.call_count == 1


# ===================================================================
# Tests: NVD API key header
# ===================================================================
class TestNvdHeaders:
    @pytest.mark.usefixtures("_no_nvd_key")
    def test_no_key(self) -> None:
        from manus_agent.tools.get_version_range import _build_nvd_headers

        assert _build_nvd_headers() == {}

    @pytest.mark.usefixtures("_with_nvd_key")
    def test_with_key(self) -> None:
        from manus_agent.tools.get_version_range import _build_nvd_headers

        headers = _build_nvd_headers()
        assert headers["apiKey"] == "fake-nvd-key-12345"


# ===================================================================
# Tests: CLI version-range subcommand
# ===================================================================
class TestCliVersionRange:
    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_json_output(self, mock_fetch: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "found": True,
            "ecosystems": ["Maven"],
            "nvd_cpe_ranges": [],
            "osv_packages": [],
            "first_patched": {},
            "message": "ok",
            "errors": None,
        }
        rc = _run_version_range(["CVE-2021-44228", "--output", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["cve_id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_output_found(self, mock_fetch: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "found": True,
            "description": "Log4j vuln",
            "ecosystems": ["Maven"],
            "nvd_cpe_ranges": [
                {
                    "vendor": "apache",
                    "product": "log4j",
                    "range_expression": ">= 2.0, < 2.15.0",
                    "ecosystem_hint": "Maven",
                }
            ],
            "osv_packages": [
                {
                    "ecosystem": "Maven",
                    "package": "log4j-core",
                    "ranges": [{"type": "ECOSYSTEM", "introduced": ["2.0"], "fixed": ["2.15.0"], "last_affected": []}],
                    "first_patched": "2.3.1",
                    "affected_version_count": 5,
                    "affected_versions_sample": ["2.0", "2.1", "2.2", "2.3", "2.14.1"],
                }
            ],
            "first_patched": {"Maven:log4j-core": "2.3.1"},
            "message": "ok",
            "errors": None,
        }
        rc = _run_version_range(["CVE-2021-44228"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2021-44228" in out
        assert "Maven" in out
        assert "log4j" in out
        assert "2.3.1" in out

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_text_output_not_found(self, mock_fetch: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {
            "cve_id": "CVE-9999-99999",
            "found": False,
            "description": None,
            "ecosystems": [],
            "nvd_cpe_ranges": [],
            "osv_packages": [],
            "first_patched": {},
            "message": "No data",
            "errors": None,
        }
        rc = _run_version_range(["CVE-9999-99999"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No version range data found" in out

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_ecosystem_flag(self, mock_fetch: MagicMock) -> None:
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "found": True,
            "description": None,
            "ecosystems": [],
            "nvd_cpe_ranges": [],
            "osv_packages": [],
            "first_patched": {},
            "message": "ok",
            "errors": None,
        }
        rc = _run_version_range(["CVE-2021-44228", "--ecosystem", "pypi"])
        assert rc == 0
        mock_fetch.assert_called_once_with("CVE-2021-44228", ecosystem="pypi")

    def test_invalid_cve_format(self) -> None:
        from manus_agent.cli import _run_version_range

        with pytest.raises(SystemExit):
            _run_version_range(["not-a-cve"])

    def test_help_flag(self) -> None:
        from manus_agent.cli import _run_version_range

        with pytest.raises(SystemExit) as exc_info:
            _run_version_range(["--help"])
        assert exc_info.value.code == 0

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_errors_printed_to_stderr(self, mock_fetch: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        from manus_agent.cli import _run_version_range

        mock_fetch.return_value = {
            "cve_id": "CVE-2024-0001",
            "found": True,
            "description": None,
            "ecosystems": [],
            "nvd_cpe_ranges": [],
            "osv_packages": [],
            "first_patched": {},
            "message": "partial",
            "errors": ["NVD down"],
        }
        rc = _run_version_range(["CVE-2024-0001"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "NVD down" in err

    @patch("manus_agent.tools.get_version_range.fetch_version_range")
    def test_long_description_truncated(self, mock_fetch: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        from manus_agent.cli import _run_version_range

        long_desc = "A" * 200
        mock_fetch.return_value = {
            "cve_id": "CVE-2024-0002",
            "found": True,
            "description": long_desc,
            "ecosystems": [],
            "nvd_cpe_ranges": [],
            "osv_packages": [],
            "first_patched": {},
            "message": "ok",
            "errors": None,
        }
        rc = _run_version_range(["CVE-2024-0002"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "..." in out


# ===================================================================
# Tests: CLI parser
# ===================================================================
class TestCliParser:
    def test_parser_defaults(self) -> None:
        from manus_agent.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        args = parser.parse_args(["CVE-2021-44228"])
        assert args.cve_id == "CVE-2021-44228"
        assert args.ecosystem == "auto"
        assert args.output == "text"

    def test_parser_all_flags(self) -> None:
        from manus_agent.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        args = parser.parse_args(["CVE-2021-44228", "--ecosystem", "npm", "--output", "json"])
        assert args.ecosystem == "npm"
        assert args.output == "json"

    def test_parser_invalid_ecosystem(self) -> None:
        from manus_agent.cli import _build_version_range_parser

        parser = _build_version_range_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["CVE-2021-44228", "--ecosystem", "invalid"])


# ===================================================================
# Tests: CLI dispatch integration (version-range in _SUBCOMMANDS)
# ===================================================================
class TestCliDispatch:
    def test_version_range_in_subcommands(self) -> None:
        from manus_agent.cli import _SUBCOMMANDS

        assert "version-range" in _SUBCOMMANDS

    @patch("manus_agent.cli._run_version_range")
    def test_main_dispatches_version_range(self, mock_run: MagicMock) -> None:
        from manus_agent.cli import main

        mock_run.return_value = 0
        with pytest.raises(SystemExit) as exc_info:
            with patch("sys.argv", ["manus-agent", "version-range", "CVE-2021-44228"]):
                main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(["CVE-2021-44228"])


# ===================================================================
# Tests: Edge cases and robustness
# ===================================================================
class TestEdgeCases:
    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_whitespace_cve_id_stripped(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        mock_nvd.return_value = {"found": False, "cpe_ranges": [], "descriptions": []}
        mock_osv.return_value = {"found": False, "packages": [], "aliases": []}
        result = fetch_version_range("  CVE-2021-44228  ")
        assert result["cve_id"] == "CVE-2021-44228"

    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_lowercase_cve_uppercased(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        mock_nvd.return_value = {"found": False, "cpe_ranges": [], "descriptions": []}
        mock_osv.return_value = {"found": False, "packages": [], "aliases": []}
        result = fetch_version_range("cve-2021-44228")
        assert result["cve_id"] == "CVE-2021-44228"

    def test_ecosystem_aliases(self) -> None:
        from manus_agent.tools.get_version_range import _ECOSYSTEM_ALIASES

        assert _ECOSYSTEM_ALIASES["pypi"] == "PyPI"
        assert _ECOSYSTEM_ALIASES["npm"] == "npm"
        assert _ECOSYSTEM_ALIASES["maven"] == "Maven"
        assert _ECOSYSTEM_ALIASES["go"] == "Go"

    @patch("manus_agent.tools.get_version_range._get_with_retry")
    def test_fetch_nvd_ranges_ecosystem_hint(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_nvd_ranges

        configs = [
            {
                "nodes": [
                    {
                        "cpeMatch": [
                            {
                                "vulnerable": True,
                                "criteria": "cpe:2.3:a:python:django:*:*:*:*:*:*:*:*",
                                "versionEndExcluding": "4.0",
                            }
                        ]
                    }
                ]
            }
        ]
        mock_get.return_value = _mock_response(
            200,
            _make_nvd_response(cve_id="CVE-2024-0010", configurations=configs),
        )
        result = fetch_nvd_ranges("CVE-2024-0010")
        assert result["cpe_ranges"][0]["ecosystem_hint"] == "PyPI"

    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_multiple_first_patched_per_package(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        mock_nvd.return_value = {"found": False, "cpe_ranges": [], "descriptions": []}
        mock_osv.return_value = {
            "found": True,
            "packages": [
                {
                    "ecosystem": "PyPI",
                    "package": "flask",
                    "ranges": [],
                    "first_patched": "2.3.3",
                    "affected_version_count": 0,
                    "affected_versions_sample": [],
                },
                {
                    "ecosystem": "npm",
                    "package": "express",
                    "ranges": [],
                    "first_patched": "4.18.3",
                    "affected_version_count": 0,
                    "affected_versions_sample": [],
                },
            ],
            "aliases": [],
        }

        result = fetch_version_range("CVE-2024-0011")
        assert result["first_patched"]["PyPI:flask"] == "2.3.3"
        assert result["first_patched"]["npm:express"] == "4.18.3"

    @patch("manus_agent.tools.get_version_range.fetch_osv_ranges")
    @patch("manus_agent.tools.get_version_range.fetch_nvd_ranges")
    def test_none_description(self, mock_nvd: MagicMock, mock_osv: MagicMock) -> None:
        from manus_agent.tools.get_version_range import fetch_version_range

        mock_nvd.return_value = {"found": True, "cpe_ranges": [], "descriptions": []}
        mock_osv.return_value = {"found": False, "packages": [], "aliases": []}
        result = fetch_version_range("CVE-2024-0012")
        assert result["description"] is None
