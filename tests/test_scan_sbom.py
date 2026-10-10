#!/usr/bin/env python3
"""Comprehensive test suite for the scan_sbom tool and sbom-scan CLI subcommand.

All HTTP calls are mocked — no real network access.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.scan_sbom import (
    Component,
    _fetch_epss_scores,
    _fetch_kev_cve_set,
    _parse_cdx_json,
    _parse_cdx_xml,
    _parse_purl,
    _parse_spdx_json,
    _query_osv_batch,
    build_scan_results,
    format_text_report,
    parse_sbom,
    scan_sbom,
)

# Type alias for tool use dicts (avoids importing from strands in tests)
_ToolUse = dict

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "_sbom_fixtures"


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch):
    """Disable retry delays for all tests."""
    monkeypatch.setenv("SBOM_SCAN_MAX_RETRIES", "1")
    monkeypatch.setenv("SBOM_SCAN_RETRY_BASE_DELAY", "0")


@pytest.fixture()
def cdx_json_sbom(tmp_path: Path) -> Path:
    """Create a minimal CycloneDX JSON SBOM fixture."""
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "components": [
            {
                "type": "library",
                "name": "requests",
                "version": "2.28.0",
                "purl": "pkg:pypi/requests@2.28.0",
            },
            {
                "type": "library",
                "name": "express",
                "version": "4.17.1",
                "purl": "pkg:npm/express@4.17.1",
            },
            {
                "type": "library",
                "name": "lodash",
                "version": "4.17.20",
                "purl": "pkg:npm/lodash@4.17.20",
            },
        ],
    }
    p = tmp_path / "bom.json"
    p.write_text(json.dumps(sbom))
    return p


@pytest.fixture()
def cdx_xml_sbom(tmp_path: Path) -> Path:
    """Create a minimal CycloneDX XML SBOM fixture."""
    xml = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <bom xmlns="http://cyclonedx.org/schema/bom/1.5">
          <components>
            <component type="library">
              <name>django</name>
              <version>3.2.0</version>
              <purl>pkg:pypi/django@3.2.0</purl>
            </component>
            <component type="library">
              <name>flask</name>
              <version>2.0.0</version>
              <purl>pkg:pypi/flask@2.0.0</purl>
            </component>
          </components>
        </bom>
    """)
    p = tmp_path / "bom.xml"
    p.write_text(xml)
    return p


@pytest.fixture()
def spdx_json_sbom(tmp_path: Path) -> Path:
    """Create a minimal SPDX JSON SBOM fixture."""
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "packages": [
            {
                "name": "numpy",
                "versionInfo": "1.24.0",
                "externalRefs": [
                    {
                        "referenceType": "purl",
                        "referenceLocator": "pkg:pypi/numpy@1.24.0",
                    }
                ],
            },
            {
                "name": "pandas",
                "versionInfo": "2.0.0",
                "externalRefs": [
                    {
                        "referenceType": "purl",
                        "referenceLocator": "pkg:pypi/pandas@2.0.0",
                    }
                ],
            },
        ],
    }
    p = tmp_path / "sbom.spdx.json"
    p.write_text(json.dumps(sbom))
    return p


@pytest.fixture()
def empty_sbom(tmp_path: Path) -> Path:
    """SBOM with no components."""
    sbom = {"bomFormat": "CycloneDX", "specVersion": "1.5", "components": []}
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(sbom))
    return p


@pytest.fixture()
def no_purl_sbom(tmp_path: Path) -> Path:
    """SBOM with components lacking purl identifiers."""
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "components": [
            {"type": "library", "name": "some-lib", "version": "1.0.0"},
        ],
    }
    p = tmp_path / "no-purl.json"
    p.write_text(json.dumps(sbom))
    return p


def _make_osv_vuln(
    vuln_id: str = "GHSA-xxxx-yyyy-zzzz",
    aliases: list[str] | None = None,
    summary: str = "Test vulnerability",
    ecosystem: str = "PyPI",
    package: str = "requests",
    introduced: str = "0",
    fixed: str = "2.28.1",
    severity_score: float | None = None,
) -> dict:
    """Helper to build an OSV vulnerability dict."""
    vuln: dict = {
        "id": vuln_id,
        "aliases": aliases or [],
        "summary": summary,
        "affected": [
            {
                "package": {"name": package, "ecosystem": ecosystem},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": introduced},
                            {"fixed": fixed},
                        ],
                    }
                ],
            }
        ],
    }
    if severity_score is not None:
        vuln["severity"] = [{"type": "CVSS_V3", "score": severity_score}]
    else:
        vuln["severity"] = []
    return vuln


# ═══════════════════════════════════════════════════════════════════════════
# PURL Parsing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParsePurl:
    """Tests for _parse_purl."""

    def test_pypi_purl(self):
        result = _parse_purl("pkg:pypi/requests@2.28.0")
        assert result == ("PyPI", "requests", "2.28.0")

    def test_npm_purl(self):
        result = _parse_purl("pkg:npm/express@4.17.1")
        assert result == ("npm", "express", "4.17.1")

    def test_npm_scoped_purl(self):
        result = _parse_purl("pkg:npm/%40angular/core@14.0.0")
        assert result == ("npm", "%40angular/core", "14.0.0")

    def test_maven_purl(self):
        result = _parse_purl("pkg:maven/org.apache.logging.log4j/log4j-core@2.17.0")
        assert result == ("Maven", "org.apache.logging.log4j:log4j-core", "2.17.0")

    def test_golang_purl(self):
        result = _parse_purl("pkg:golang/github.com/gin-gonic/gin@1.8.0")
        assert result == ("Go", "github.com/gin-gonic/gin", "1.8.0")

    def test_cargo_purl(self):
        result = _parse_purl("pkg:cargo/serde@1.0.160")
        assert result == ("crates.io", "serde", "1.0.160")

    def test_gem_purl(self):
        result = _parse_purl("pkg:gem/rails@7.0.0")
        assert result == ("RubyGems", "rails", "7.0.0")

    def test_nuget_purl(self):
        result = _parse_purl("pkg:nuget/Newtonsoft.Json@13.0.1")
        assert result == ("NuGet", "Newtonsoft.Json", "13.0.1")

    def test_composer_purl(self):
        result = _parse_purl("pkg:composer/laravel/framework@9.0.0")
        assert result == ("Packagist", "laravel/framework", "9.0.0")

    def test_no_pkg_prefix(self):
        assert _parse_purl("not-a-purl") is None

    def test_no_version(self):
        assert _parse_purl("pkg:pypi/requests") is None

    def test_empty_version(self):
        assert _parse_purl("pkg:pypi/requests@") is None

    def test_unknown_type(self):
        assert _parse_purl("pkg:unknown/lib@1.0") is None

    def test_insufficient_parts(self):
        assert _parse_purl("pkg:pypi") is None

    def test_purl_with_qualifiers(self):
        result = _parse_purl("pkg:pypi/requests@2.28.0?vcs_url=https://github.com")
        assert result == ("PyPI", "requests", "2.28.0")

    def test_purl_with_subpath(self):
        result = _parse_purl("pkg:pypi/requests@2.28.0#sub/path")
        assert result == ("PyPI", "requests", "2.28.0")

    def test_purl_with_qualifiers_and_subpath(self):
        result = _parse_purl("pkg:pypi/requests@2.28.0?foo=bar#baz")
        assert result == ("PyPI", "requests", "2.28.0")

    def test_pub_purl(self):
        result = _parse_purl("pkg:pub/http@0.13.0")
        assert result == ("Pub", "http", "0.13.0")

    def test_hex_purl(self):
        result = _parse_purl("pkg:hex/phoenix@1.7.0")
        assert result == ("Hex", "phoenix", "1.7.0")


# ═══════════════════════════════════════════════════════════════════════════
# CycloneDX JSON Parsing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParseCdxJson:
    """Tests for _parse_cdx_json."""

    def test_basic_parsing(self):
        data = {
            "bomFormat": "CycloneDX",
            "components": [
                {
                    "type": "library",
                    "name": "requests",
                    "version": "2.28.0",
                    "purl": "pkg:pypi/requests@2.28.0",
                }
            ],
        }
        result = _parse_cdx_json(data)
        assert len(result) == 1
        assert result[0] == ("PyPI", "requests", "2.28.0")

    def test_multiple_components(self):
        data = {
            "components": [
                {"purl": "pkg:pypi/requests@2.28.0"},
                {"purl": "pkg:npm/express@4.17.1"},
            ]
        }
        result = _parse_cdx_json(data)
        assert len(result) == 2

    def test_empty_components(self):
        data = {"components": []}
        result = _parse_cdx_json(data)
        assert result == []

    def test_no_components_key(self):
        data = {"bomFormat": "CycloneDX"}
        result = _parse_cdx_json(data)
        assert result == []

    def test_component_without_purl(self):
        data = {"components": [{"type": "library", "name": "some-lib", "version": "1.0.0"}]}
        result = _parse_cdx_json(data)
        assert result == []

    def test_mixed_with_and_without_purl(self):
        data = {
            "components": [
                {"purl": "pkg:pypi/requests@2.28.0"},
                {"type": "library", "name": "no-purl", "version": "1.0"},
                {"purl": "pkg:npm/lodash@4.17.20"},
            ]
        }
        result = _parse_cdx_json(data)
        assert len(result) == 2

    def test_component_missing_name_and_version(self):
        data = {"components": [{"type": "library"}]}
        result = _parse_cdx_json(data)
        assert result == []

    def test_component_with_invalid_purl(self):
        data = {"components": [{"purl": "not-a-purl"}]}
        result = _parse_cdx_json(data)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# CycloneDX XML Parsing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParseCdxXml:
    """Tests for _parse_cdx_xml."""

    def test_basic_xml_parsing(self):
        import xml.etree.ElementTree as ET

        xml = textwrap.dedent("""\
            <bom xmlns="http://cyclonedx.org/schema/bom/1.5">
              <components>
                <component type="library">
                  <name>requests</name>
                  <version>2.28.0</version>
                  <purl>pkg:pypi/requests@2.28.0</purl>
                </component>
              </components>
            </bom>
        """)
        root = ET.fromstring(xml)
        result = _parse_cdx_xml(root)
        assert len(result) == 1
        assert result[0] == ("PyPI", "requests", "2.28.0")

    def test_xml_without_namespace(self):
        import xml.etree.ElementTree as ET

        xml = textwrap.dedent("""\
            <bom>
              <components>
                <component type="library">
                  <name>flask</name>
                  <version>2.0.0</version>
                  <purl>pkg:pypi/flask@2.0.0</purl>
                </component>
              </components>
            </bom>
        """)
        root = ET.fromstring(xml)
        result = _parse_cdx_xml(root)
        assert len(result) == 1
        assert result[0] == ("PyPI", "flask", "2.0.0")

    def test_xml_multiple_components(self):
        import xml.etree.ElementTree as ET

        xml = textwrap.dedent("""\
            <bom xmlns="http://cyclonedx.org/schema/bom/1.5">
              <components>
                <component type="library">
                  <purl>pkg:pypi/flask@2.0.0</purl>
                </component>
                <component type="library">
                  <purl>pkg:npm/react@18.0.0</purl>
                </component>
              </components>
            </bom>
        """)
        root = ET.fromstring(xml)
        result = _parse_cdx_xml(root)
        assert len(result) == 2

    def test_xml_component_without_purl(self):
        import xml.etree.ElementTree as ET

        xml = textwrap.dedent("""\
            <bom>
              <components>
                <component type="library">
                  <name>mylib</name>
                  <version>1.0.0</version>
                </component>
              </components>
            </bom>
        """)
        root = ET.fromstring(xml)
        result = _parse_cdx_xml(root)
        # No purl → skipped
        assert result == []

    def test_xml_empty_purl(self):
        import xml.etree.ElementTree as ET

        xml = textwrap.dedent("""\
            <bom>
              <components>
                <component type="library">
                  <name>mylib</name>
                  <version>1.0.0</version>
                  <purl></purl>
                </component>
              </components>
            </bom>
        """)
        root = ET.fromstring(xml)
        result = _parse_cdx_xml(root)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# SPDX JSON Parsing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParseSpdxJson:
    """Tests for _parse_spdx_json."""

    def test_basic_spdx_parsing(self):
        data = {
            "spdxVersion": "SPDX-2.3",
            "packages": [
                {
                    "name": "numpy",
                    "externalRefs": [
                        {
                            "referenceType": "purl",
                            "referenceLocator": "pkg:pypi/numpy@1.24.0",
                        }
                    ],
                }
            ],
        }
        result = _parse_spdx_json(data)
        assert len(result) == 1
        assert result[0] == ("PyPI", "numpy", "1.24.0")

    def test_spdx_multiple_packages(self):
        data = {
            "packages": [
                {
                    "externalRefs": [
                        {
                            "referenceType": "purl",
                            "referenceLocator": "pkg:pypi/numpy@1.24.0",
                        }
                    ]
                },
                {
                    "externalRefs": [
                        {
                            "referenceType": "purl",
                            "referenceLocator": "pkg:npm/react@18.0.0",
                        }
                    ]
                },
            ]
        }
        result = _parse_spdx_json(data)
        assert len(result) == 2

    def test_spdx_no_packages(self):
        data = {"spdxVersion": "SPDX-2.3"}
        result = _parse_spdx_json(data)
        assert result == []

    def test_spdx_no_external_refs(self):
        data = {"packages": [{"name": "numpy", "versionInfo": "1.24.0"}]}
        result = _parse_spdx_json(data)
        assert result == []

    def test_spdx_non_purl_ref(self):
        data = {"packages": [{"externalRefs": [{"referenceType": "cpe23Type", "referenceLocator": "cpe:2.3:a:*"}]}]}
        result = _parse_spdx_json(data)
        assert result == []

    def test_spdx_mixed_refs(self):
        data = {
            "packages": [
                {
                    "externalRefs": [
                        {"referenceType": "cpe23Type", "referenceLocator": "cpe:2.3:a:*"},
                        {
                            "referenceType": "purl",
                            "referenceLocator": "pkg:pypi/requests@2.28.0",
                        },
                    ]
                }
            ]
        }
        result = _parse_spdx_json(data)
        assert len(result) == 1
        assert result[0] == ("PyPI", "requests", "2.28.0")


# ═══════════════════════════════════════════════════════════════════════════
# parse_sbom (top-level format detection) Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParseSbom:
    """Tests for the top-level parse_sbom function."""

    def test_parse_cdx_json(self, cdx_json_sbom):
        result = parse_sbom(str(cdx_json_sbom))
        assert len(result) == 3

    def test_parse_cdx_xml(self, cdx_xml_sbom):
        result = parse_sbom(str(cdx_xml_sbom))
        assert len(result) == 2

    def test_parse_spdx_json(self, spdx_json_sbom):
        result = parse_sbom(str(spdx_json_sbom))
        assert len(result) == 2

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_sbom("/nonexistent/path/bom.json")

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{{{{not json")
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_sbom(str(p))

    def test_unrecognised_json_format(self, tmp_path):
        p = tmp_path / "unknown.json"
        p.write_text(json.dumps({"random": "data"}))
        with pytest.raises(ValueError, match="Unrecognised SBOM format"):
            parse_sbom(str(p))

    def test_empty_components_sbom(self, empty_sbom):
        result = parse_sbom(str(empty_sbom))
        assert result == []

    def test_no_purl_sbom(self, no_purl_sbom):
        result = parse_sbom(str(no_purl_sbom))
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# OSV Batch Query Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryOsvBatch:
    """Tests for _query_osv_batch."""

    @patch("manus_agent.tools.scan_sbom._http_post")
    def test_basic_batch_query(self, mock_post):
        vuln = _make_osv_vuln()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"vulns": [vuln]},
                {"vulns": []},
            ]
        }
        mock_post.return_value = mock_resp

        components = [
            ("PyPI", "requests", "2.28.0"),
            ("npm", "express", "4.17.1"),
        ]
        result = _query_osv_batch(components)

        assert len(result) == 1
        assert ("PyPI", "requests", "2.28.0") in result
        assert ("npm", "express", "4.17.1") not in result

    @patch("manus_agent.tools.scan_sbom._http_post")
    def test_no_vulns(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": [{"vulns": []}]}
        mock_post.return_value = mock_resp

        result = _query_osv_batch([("PyPI", "requests", "2.28.0")])
        assert result == {}

    @patch("manus_agent.tools.scan_sbom._http_post")
    def test_multiple_vulns_per_component(self, mock_post):
        vuln1 = _make_osv_vuln(vuln_id="GHSA-1111-2222-3333")
        vuln2 = _make_osv_vuln(vuln_id="GHSA-4444-5555-6666")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": [{"vulns": [vuln1, vuln2]}]}
        mock_post.return_value = mock_resp

        result = _query_osv_batch([("PyPI", "requests", "2.28.0")])
        assert len(result[("PyPI", "requests", "2.28.0")]) == 2

    @patch("manus_agent.tools.scan_sbom._http_post")
    def test_http_error_raises(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.raise_for_status.side_effect = Exception("Bad Request")
        mock_post.return_value = mock_resp

        with pytest.raises(Exception, match="Bad Request"):
            _query_osv_batch([("PyPI", "requests", "2.28.0")])

    @patch("manus_agent.tools.scan_sbom._http_post")
    def test_batching_large_input(self, mock_post):
        """Verify that large component lists are batched correctly."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        # Each batch returns empty results
        mock_resp.json.return_value = {"results": [{"vulns": []} for _ in range(1000)]}
        mock_post.return_value = mock_resp

        # 1500 components → should batch into 2 calls (1000 + 500)
        components = [("PyPI", f"pkg-{i}", "1.0.0") for i in range(1500)]
        _query_osv_batch(components)
        assert mock_post.call_count == 2

    @patch("manus_agent.tools.scan_sbom._http_post")
    def test_empty_results_key(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_post.return_value = mock_resp

        result = _query_osv_batch([])
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# EPSS Enrichment Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchEpssScores:
    """Tests for _fetch_epss_scores."""

    @patch("manus_agent.tools.scan_sbom._http_get")
    def test_basic_epss_fetch(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"cve": "CVE-2023-1234", "epss": "0.5432"},
                {"cve": "CVE-2023-5678", "epss": "0.0012"},
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_epss_scores(["CVE-2023-1234", "CVE-2023-5678"])
        assert result["CVE-2023-1234"] == pytest.approx(0.5432)
        assert result["CVE-2023-5678"] == pytest.approx(0.0012)

    @patch("manus_agent.tools.scan_sbom._http_get")
    def test_empty_input(self, mock_get):
        result = _fetch_epss_scores([])
        assert result == {}
        mock_get.assert_not_called()

    @patch("manus_agent.tools.scan_sbom._http_get")
    def test_epss_api_failure_graceful(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = _fetch_epss_scores(["CVE-2023-1234"])
        assert result == {}

    @patch("manus_agent.tools.scan_sbom._http_get")
    def test_epss_batching(self, mock_get):
        """Test that >100 CVEs are batched."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        cves = [f"CVE-2023-{i:04d}" for i in range(150)]
        _fetch_epss_scores(cves)
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.scan_sbom._http_get")
    def test_epss_missing_fields(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [
                {"cve": "CVE-2023-1234"},  # no epss field
                {"epss": "0.5"},  # no cve field
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_epss_scores(["CVE-2023-1234"])
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# CISA KEV Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchKevCveSet:
    """Tests for _fetch_kev_cve_set."""

    @patch("manus_agent.tools.scan_sbom._http_get")
    def test_basic_kev_fetch(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cveID": "CVE-2021-44228"},
                {"cveID": "CVE-2024-3094"},
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_kev_cve_set()
        assert "CVE-2021-44228" in result
        assert "CVE-2024-3094" in result

    @patch("manus_agent.tools.scan_sbom._http_get")
    def test_kev_api_failure_graceful(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = _fetch_kev_cve_set()
        assert result == set()

    @patch("manus_agent.tools.scan_sbom._http_get")
    def test_kev_empty_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        result = _fetch_kev_cve_set()
        assert result == set()

    @patch("manus_agent.tools.scan_sbom._http_get")
    def test_kev_missing_cve_id(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cveID": "CVE-2021-44228"},
                {"other_field": "no-cve"},
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_kev_cve_set()
        assert len(result) == 1
        assert "CVE-2021-44228" in result


# ═══════════════════════════════════════════════════════════════════════════
# Report Building Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildScanResults:
    """Tests for build_scan_results."""

    def test_no_vulnerabilities(self):
        components = [("PyPI", "safe-lib", "1.0.0")]
        report = build_scan_results(components, {}, {}, set())
        assert report["total_components"] == 1
        assert report["vulnerable_components"] == 0
        assert report["total_findings"] == 0

    def test_single_finding(self):
        components = [("PyPI", "requests", "2.28.0")]
        vuln = _make_osv_vuln(
            vuln_id="GHSA-1234",
            aliases=["CVE-2023-1234"],
            severity_score=7.5,
        )
        osv_results = {("PyPI", "requests", "2.28.0"): [vuln]}
        epss = {"CVE-2023-1234": 0.42}

        report = build_scan_results(components, osv_results, epss, set())
        assert report["total_findings"] == 1
        f = report["findings"][0]
        assert f["vuln_id"] == "GHSA-1234"
        assert f["epss_score"] == pytest.approx(0.42)
        assert f["cvss_score"] == pytest.approx(7.5)
        assert f["in_kev"] is False

    def test_kev_finding_ranks_first(self):
        components = [
            ("PyPI", "requests", "2.28.0"),
            ("npm", "lodash", "4.17.20"),
        ]
        vuln_no_kev = _make_osv_vuln(
            vuln_id="GHSA-1111",
            aliases=["CVE-2023-1111"],
            severity_score=9.9,
            ecosystem="PyPI",
            package="requests",
        )
        vuln_kev = _make_osv_vuln(
            vuln_id="GHSA-2222",
            aliases=["CVE-2023-2222"],
            severity_score=5.0,
            ecosystem="npm",
            package="lodash",
        )
        osv_results = {
            ("PyPI", "requests", "2.28.0"): [vuln_no_kev],
            ("npm", "lodash", "4.17.20"): [vuln_kev],
        }
        kev = {"CVE-2023-2222"}

        report = build_scan_results(components, osv_results, {}, kev)
        assert report["findings"][0]["in_kev"] is True
        assert report["findings"][0]["vuln_id"] == "GHSA-2222"

    def test_epss_ranking_after_kev(self):
        components = [
            ("PyPI", "a", "1.0"),
            ("PyPI", "b", "1.0"),
        ]
        vuln_a = _make_osv_vuln(
            vuln_id="GHSA-AAAA",
            aliases=["CVE-2023-AAAA"],
            ecosystem="PyPI",
            package="a",
        )
        vuln_b = _make_osv_vuln(
            vuln_id="GHSA-BBBB",
            aliases=["CVE-2023-BBBB"],
            ecosystem="PyPI",
            package="b",
        )
        osv_results = {
            ("PyPI", "a", "1.0"): [vuln_a],
            ("PyPI", "b", "1.0"): [vuln_b],
        }
        epss = {"CVE-2023-AAAA": 0.1, "CVE-2023-BBBB": 0.9}

        report = build_scan_results(components, osv_results, epss, set())
        # Higher EPSS ranks first
        assert report["findings"][0]["vuln_id"] == "GHSA-BBBB"
        assert report["findings"][1]["vuln_id"] == "GHSA-AAAA"

    def test_deduplication(self):
        components = [("PyPI", "requests", "2.28.0")]
        vuln = _make_osv_vuln(vuln_id="GHSA-1234")
        # Same vuln appears twice (e.g. from batch overlap)
        osv_results = {("PyPI", "requests", "2.28.0"): [vuln, vuln]}

        report = build_scan_results(components, osv_results, {}, set())
        assert report["total_findings"] == 1

    def test_severity_counts(self):
        components = [("PyPI", "a", "1.0")]
        vulns = [
            _make_osv_vuln(vuln_id="V1", severity_score=9.5, package="a"),
            _make_osv_vuln(vuln_id="V2", severity_score=7.5, package="a"),
            _make_osv_vuln(vuln_id="V3", severity_score=5.0, package="a"),
            _make_osv_vuln(vuln_id="V4", severity_score=2.0, package="a"),
            _make_osv_vuln(vuln_id="V5", package="a"),  # no severity
        ]
        osv_results = {("PyPI", "a", "1.0"): vulns}

        report = build_scan_results(components, osv_results, {}, set())
        assert report["critical_count"] == 1
        assert report["high_count"] == 1
        assert report["medium_count"] == 1
        assert report["low_count"] == 1
        assert report["unscored_count"] == 1

    def test_cve_id_as_vuln_id(self):
        components = [("PyPI", "a", "1.0")]
        vuln = _make_osv_vuln(
            vuln_id="CVE-2023-9999",
            aliases=[],
            package="a",
        )
        osv_results = {("PyPI", "a", "1.0"): [vuln]}

        report = build_scan_results(components, osv_results, {}, set())
        f = report["findings"][0]
        assert "CVE-2023-9999" in f["aliases"]

    def test_affected_range_extraction(self):
        components = [("PyPI", "requests", "2.28.0")]
        vuln = _make_osv_vuln(
            vuln_id="GHSA-1234",
            package="requests",
            ecosystem="PyPI",
            introduced="2.0.0",
            fixed="2.28.1",
        )
        osv_results = {("PyPI", "requests", "2.28.0"): [vuln]}

        report = build_scan_results(components, osv_results, {}, set())
        f = report["findings"][0]
        assert ">=2.0.0" in f["affected_range"]
        assert "<2.28.1" in f["affected_range"]

    def test_best_epss_from_multiple_aliases(self):
        components = [("PyPI", "a", "1.0")]
        vuln = _make_osv_vuln(
            vuln_id="GHSA-1234",
            aliases=["CVE-2023-0001", "CVE-2023-0002"],
            package="a",
        )
        osv_results = {("PyPI", "a", "1.0"): [vuln]}
        epss = {"CVE-2023-0001": 0.2, "CVE-2023-0002": 0.8}

        report = build_scan_results(components, osv_results, epss, set())
        # Should pick the highest EPSS across aliases
        assert report["findings"][0]["epss_score"] == pytest.approx(0.8)

    def test_kev_count(self):
        components = [("PyPI", "a", "1.0"), ("PyPI", "b", "1.0")]
        v1 = _make_osv_vuln(vuln_id="V1", aliases=["CVE-2023-0001"], package="a")
        v2 = _make_osv_vuln(vuln_id="V2", aliases=["CVE-2023-0002"], package="b")
        osv_results = {
            ("PyPI", "a", "1.0"): [v1],
            ("PyPI", "b", "1.0"): [v2],
        }
        kev = {"CVE-2023-0001", "CVE-2023-0002"}

        report = build_scan_results(components, osv_results, {}, kev)
        assert report["kev_count"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# Text Report Formatting Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatTextReport:
    """Tests for format_text_report."""

    def test_no_findings_report(self):
        report = {
            "total_components": 10,
            "vulnerable_components": 0,
            "total_findings": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "unscored_count": 0,
            "kev_count": 0,
            "findings": [],
        }
        text = format_text_report(report)
        assert "No known vulnerabilities" in text
        assert "Components scanned" in text

    def test_findings_present(self):
        report = {
            "total_components": 5,
            "vulnerable_components": 1,
            "total_findings": 1,
            "critical_count": 1,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "unscored_count": 0,
            "kev_count": 1,
            "findings": [
                {
                    "vuln_id": "GHSA-1234",
                    "aliases": ["CVE-2023-1234"],
                    "ecosystem": "PyPI",
                    "package": "requests",
                    "installed_version": "2.28.0",
                    "affected_range": ">=0, <2.28.1",
                    "cvss_score": 9.5,
                    "epss_score": 0.95,
                    "in_kev": True,
                    "summary": "Critical vuln",
                }
            ],
        }
        text = format_text_report(report)
        assert "GHSA-1234" in text
        assert "KEV" in text
        assert "9.5" in text
        assert "0.9500" in text
        assert "requests" in text

    def test_null_scores_display(self):
        report = {
            "total_components": 1,
            "vulnerable_components": 1,
            "total_findings": 1,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "unscored_count": 1,
            "kev_count": 0,
            "findings": [
                {
                    "vuln_id": "GHSA-5678",
                    "aliases": [],
                    "ecosystem": "npm",
                    "package": "lodash",
                    "installed_version": "4.17.20",
                    "affected_range": "",
                    "cvss_score": None,
                    "epss_score": None,
                    "in_kev": False,
                    "summary": "",
                }
            ],
        }
        text = format_text_report(report)
        assert "N/A" in text
        assert "GHSA-5678" in text

    def test_summary_header(self):
        report = {
            "total_components": 42,
            "vulnerable_components": 3,
            "total_findings": 5,
            "critical_count": 1,
            "high_count": 2,
            "medium_count": 1,
            "low_count": 1,
            "unscored_count": 0,
            "kev_count": 0,
            "findings": [],
        }
        text = format_text_report(report)
        assert "42" in text
        assert "SBOM Vulnerability Scan" in text


# ═══════════════════════════════════════════════════════════════════════════
# Strands Tool Entry Point Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestScanSbomTool:
    """Tests for the scan_sbom Strands tool entry point."""

    def test_missing_path(self):
        tool_use = {"toolUseId": "t1", "name": "scan_sbom", "input": {}}
        result = scan_sbom(tool_use)
        assert result["status"] == "error"
        assert "required" in result["content"][0]["text"].lower()

    def test_file_not_found(self):
        tool_use = {
            "toolUseId": "t1",
            "name": "scan_sbom",
            "input": {"sbom_path": "/nonexistent/bom.json"},
        }
        result = scan_sbom(tool_use)
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"].lower()

    def test_empty_sbom_no_components(self, empty_sbom):
        tool_use = {
            "toolUseId": "t1",
            "name": "scan_sbom",
            "input": {"sbom_path": str(empty_sbom)},
        }
        result = scan_sbom(tool_use)
        assert result["status"] == "success"
        assert "no components" in result["content"][0]["text"].lower()

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_successful_scan(self, mock_osv, mock_epss, mock_kev, cdx_json_sbom):
        vuln = _make_osv_vuln(
            vuln_id="GHSA-test",
            aliases=["CVE-2023-9999"],
        )
        mock_osv.return_value = {("PyPI", "requests", "2.28.0"): [vuln]}
        mock_epss.return_value = {"CVE-2023-9999": 0.75}
        mock_kev.return_value = set()

        tool_use = {
            "toolUseId": "t1",
            "name": "scan_sbom",
            "input": {"sbom_path": str(cdx_json_sbom)},
        }
        result = scan_sbom(tool_use)
        assert result["status"] == "success"
        assert "GHSA-test" in result["content"][0]["text"]

    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_osv_failure(self, mock_osv, cdx_json_sbom):
        mock_osv.side_effect = Exception("OSV down")

        tool_use = {
            "toolUseId": "t1",
            "name": "scan_sbom",
            "input": {"sbom_path": str(cdx_json_sbom)},
        }
        result = scan_sbom(tool_use)
        assert result["status"] == "error"
        assert "OSV" in result["content"][0]["text"]

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_no_vulns_found(self, mock_osv, mock_epss, mock_kev, cdx_json_sbom):
        mock_osv.return_value = {}
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        tool_use = {
            "toolUseId": "t1",
            "name": "scan_sbom",
            "input": {"sbom_path": str(cdx_json_sbom)},
        }
        result = scan_sbom(tool_use)
        assert result["status"] == "success"
        assert "No known vulnerabilities" in result["content"][0]["text"]

    def test_invalid_sbom_format(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text('{"random": "stuff"}')
        tool_use = {
            "toolUseId": "t1",
            "name": "scan_sbom",
            "input": {"sbom_path": str(p)},
        }
        result = scan_sbom(tool_use)
        assert result["status"] == "error"


# ═══════════════════════════════════════════════════════════════════════════
# CLI Subcommand Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSbomScanCli:
    """Tests for _run_sbom_scan CLI subcommand."""

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_text_output(self, mock_osv, mock_epss, mock_kev, cdx_json_sbom, capsys):
        from manus_agent.cli import _run_sbom_scan

        vuln = _make_osv_vuln(
            vuln_id="GHSA-cli-test",
            aliases=["CVE-2023-0001"],
        )
        mock_osv.return_value = {("PyPI", "requests", "2.28.0"): [vuln]}
        mock_epss.return_value = {"CVE-2023-0001": 0.5}
        mock_kev.return_value = set()

        code = _run_sbom_scan([str(cdx_json_sbom)])
        assert code == 0
        captured = capsys.readouterr()
        assert "GHSA-cli-test" in captured.out

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_json_output(self, mock_osv, mock_epss, mock_kev, cdx_json_sbom, capsys):
        from manus_agent.cli import _run_sbom_scan

        mock_osv.return_value = {}
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        code = _run_sbom_scan([str(cdx_json_sbom), "--output", "json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "total_components" in data
        assert "findings" in data

    def test_file_not_found(self, capsys):
        from manus_agent.cli import _run_sbom_scan

        code = _run_sbom_scan(["/nonexistent/bom.json"])
        assert code == 1

    def test_no_components(self, empty_sbom, capsys):
        from manus_agent.cli import _run_sbom_scan

        code = _run_sbom_scan([str(empty_sbom)])
        assert code == 1
        captured = capsys.readouterr()
        assert "no components" in captured.err.lower()

    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_osv_error(self, mock_osv, cdx_json_sbom, capsys):
        from manus_agent.cli import _run_sbom_scan

        mock_osv.side_effect = Exception("OSV down")

        code = _run_sbom_scan([str(cdx_json_sbom)])
        assert code == 1

    def test_help_flag(self):
        from manus_agent.cli import _run_sbom_scan

        with pytest.raises(SystemExit) as exc:
            _run_sbom_scan(["--help"])
        assert exc.value.code == 0


# ═══════════════════════════════════════════════════════════════════════════
# HTTP Helper Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHttpHelpers:
    """Tests for _http_get and _http_post retry logic."""

    @patch("manus_agent.tools.scan_sbom.requests.get")
    def test_http_get_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        from manus_agent.tools.scan_sbom import _http_get

        result = _http_get("https://example.com")
        assert result.status_code == 200

    @patch("manus_agent.tools.scan_sbom.requests.get")
    def test_http_get_network_error(self, mock_get):
        mock_get.side_effect = ConnectionError("fail")

        from manus_agent.tools.scan_sbom import _http_get

        with pytest.raises(ConnectionError):
            _http_get("https://example.com")

    @patch("manus_agent.tools.scan_sbom.requests.post")
    def test_http_post_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        from manus_agent.tools.scan_sbom import _http_post

        result = _http_post("https://example.com", {"key": "value"})
        assert result.status_code == 200

    @patch("manus_agent.tools.scan_sbom.requests.post")
    def test_http_post_network_error(self, mock_post):
        mock_post.side_effect = ConnectionError("fail")

        from manus_agent.tools.scan_sbom import _http_post

        with pytest.raises(ConnectionError):
            _http_post("https://example.com", {})


# ═══════════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_purl_with_at_in_namespace(self):
        """npm scoped packages: @scope/name."""
        result = _parse_purl("pkg:npm/@types/node@18.0.0")
        assert result is not None
        assert result[0] == "npm"
        assert result[2] == "18.0.0"

    def test_component_equality(self):
        """Components are plain tuples — equality works naturally."""
        c1: Component = ("PyPI", "requests", "2.28.0")
        c2: Component = ("PyPI", "requests", "2.28.0")
        assert c1 == c2

    def test_vuln_with_no_affected(self):
        """Vuln without affected ranges still produces a finding."""
        components = [("PyPI", "a", "1.0")]
        vuln = {"id": "GHSA-edge", "aliases": [], "summary": "x", "severity": []}
        osv_results = {("PyPI", "a", "1.0"): [vuln]}
        report = build_scan_results(components, osv_results, {}, set())
        assert report["total_findings"] == 1
        assert report["findings"][0]["affected_range"] == ""

    def test_vuln_with_introduced_only(self):
        """Vuln with introduced but no fixed version."""
        components = [("PyPI", "a", "1.0")]
        vuln = {
            "id": "GHSA-nf",
            "aliases": [],
            "summary": "no fix",
            "severity": [],
            "affected": [
                {
                    "package": {"name": "a", "ecosystem": "PyPI"},
                    "ranges": [
                        {
                            "type": "ECOSYSTEM",
                            "events": [{"introduced": "0"}],
                        }
                    ],
                }
            ],
        }
        osv_results = {("PyPI", "a", "1.0"): [vuln]}
        report = build_scan_results(components, osv_results, {}, set())
        f = report["findings"][0]
        assert "(unfixed)" in f["affected_range"]

    def test_large_component_list_parsing(self, tmp_path):
        """Parsing a large SBOM works correctly."""
        sbom = {
            "bomFormat": "CycloneDX",
            "components": [{"purl": f"pkg:pypi/lib-{i}@1.{i}.0"} for i in range(500)],
        }
        p = tmp_path / "large.json"
        p.write_text(json.dumps(sbom))
        result = parse_sbom(str(p))
        assert len(result) == 500

    def test_cdx_json_detected_by_components_key(self, tmp_path):
        """CycloneDX without bomFormat but with components key."""
        sbom = {
            "components": [
                {"purl": "pkg:pypi/flask@2.0.0"},
            ]
        }
        p = tmp_path / "no-bomformat.json"
        p.write_text(json.dumps(sbom))
        result = parse_sbom(str(p))
        assert len(result) == 1

    def test_spdx_detected_by_packages_key(self, tmp_path):
        """SPDX without spdxVersion but with packages key."""
        sbom = {
            "packages": [
                {
                    "externalRefs": [
                        {
                            "referenceType": "purl",
                            "referenceLocator": "pkg:pypi/numpy@1.24.0",
                        }
                    ]
                }
            ]
        }
        p = tmp_path / "no-spdxversion.json"
        p.write_text(json.dumps(sbom))
        result = parse_sbom(str(p))
        assert len(result) == 1

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_tool_with_kev_findings(self, mock_osv, mock_epss, mock_kev, cdx_json_sbom):
        vuln = _make_osv_vuln(
            vuln_id="GHSA-kev",
            aliases=["CVE-2021-44228"],
        )
        mock_osv.return_value = {("PyPI", "requests", "2.28.0"): [vuln]}
        mock_epss.return_value = {"CVE-2021-44228": 0.975}
        mock_kev.return_value = {"CVE-2021-44228"}

        tool_use = {
            "toolUseId": "t1",
            "name": "scan_sbom",
            "input": {"sbom_path": str(cdx_json_sbom)},
        }
        result = scan_sbom(tool_use)
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "KEV" in text

    def test_xml_sbom_via_parse_sbom(self, cdx_xml_sbom):
        result = parse_sbom(str(cdx_xml_sbom))
        assert len(result) == 2
        ecosystems = {c[0] for c in result}
        assert "PyPI" in ecosystems

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_cli_json_with_findings(self, mock_osv, mock_epss, mock_kev, cdx_json_sbom, capsys):
        from manus_agent.cli import _run_sbom_scan

        vuln = _make_osv_vuln(
            vuln_id="GHSA-json-cli",
            aliases=["CVE-2023-0099"],
            severity_score=8.1,
        )
        mock_osv.return_value = {("PyPI", "requests", "2.28.0"): [vuln]}
        mock_epss.return_value = {"CVE-2023-0099": 0.33}
        mock_kev.return_value = set()

        code = _run_sbom_scan([str(cdx_json_sbom), "--output", "json"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["total_findings"] == 1
        assert data["findings"][0]["cvss_score"] == pytest.approx(8.1)

    def test_multiple_spdx_purls_per_package(self, tmp_path):
        """SPDX package with multiple externalRefs — only purl used."""
        sbom = {
            "spdxVersion": "SPDX-2.3",
            "packages": [
                {
                    "externalRefs": [
                        {"referenceType": "cpe23Type", "referenceLocator": "cpe:2.3:a:*"},
                        {"referenceType": "purl", "referenceLocator": "pkg:pypi/flask@2.0.0"},
                    ]
                }
            ],
        }
        p = tmp_path / "multi-ref.spdx.json"
        p.write_text(json.dumps(sbom))
        result = parse_sbom(str(p))
        assert len(result) == 1
        assert result[0] == ("PyPI", "flask", "2.0.0")
