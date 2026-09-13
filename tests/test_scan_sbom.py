"""Comprehensive test suite for scan_sbom module.

All HTTP calls are fully mocked — no real network requests.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Disable retries and delays for tests
os.environ.setdefault("SBOM_SCAN_MAX_RETRIES", "1")
os.environ.setdefault("SBOM_SCAN_RETRY_BASE_DELAY", "0")

from manus_agent.tools.scan_sbom import (
    Component,
    Finding,
    _build_osv_query,
    _extract_fixed_versions,
    _extract_severity,
    _http_request,
    _parse_cyclonedx,
    _parse_purl,
    _parse_spdx,
    detect_sbom_format,
    fetch_epss_scores,
    fetch_kev_set,
    parse_sbom,
    query_osv_batch,
    scan_sbom,
    scan_sbom_tool,
)

# ===========================================================================
# Fixtures
# ===========================================================================


def _make_cyclonedx_sbom(components: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build a minimal CycloneDX 1.5 SBOM."""
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "components": components or [],
    }


def _make_spdx_sbom(packages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build a minimal SPDX 2.3 SBOM."""
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "test-sbom",
        "packages": packages or [],
    }


def _make_osv_vuln(
    vuln_id: str = "GHSA-test-1234",
    aliases: list[str] | None = None,
    summary: str = "Test vulnerability",
    fixed: str | None = None,
    severity_score: str | None = None,
) -> dict[str, Any]:
    """Build a minimal OSV vulnerability record."""
    vuln: dict[str, Any] = {
        "id": vuln_id,
        "aliases": aliases or [],
        "summary": summary,
    }
    if fixed:
        vuln["affected"] = [
            {
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [
                            {"introduced": "0"},
                            {"fixed": fixed},
                        ],
                    }
                ]
            }
        ]
    if severity_score:
        vuln["severity"] = [{"type": "CVSS_V3", "score": severity_score}]
    return vuln


def _mock_response(
    status_code: int = 200,
    json_data: Any = None,
    raise_for_status: bool = False,
) -> MagicMock:
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if raise_for_status:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


# ===========================================================================
# Component class tests
# ===========================================================================


class TestComponent:
    def test_key(self):
        c = Component(ecosystem="PyPI", name="requests", version="2.28.0")
        assert c.key() == "PyPI:requests:2.28.0"

    def test_key_empty_ecosystem(self):
        c = Component(ecosystem="", name="mylib", version="1.0")
        assert c.key() == ":mylib:1.0"

    def test_to_dict(self):
        c = Component(ecosystem="npm", name="lodash", version="4.17.21", purl="pkg:npm/lodash@4.17.21")
        d = c.to_dict()
        assert d == {
            "ecosystem": "npm",
            "name": "lodash",
            "version": "4.17.21",
            "purl": "pkg:npm/lodash@4.17.21",
        }

    def test_to_dict_no_purl(self):
        c = Component(ecosystem="PyPI", name="flask", version="2.0.0")
        d = c.to_dict()
        assert d["purl"] == ""

    def test_slots(self):
        c = Component(ecosystem="Go", name="mymod", version="1.0.0")
        with pytest.raises(AttributeError):
            c.extra_field = "nope"


# ===========================================================================
# Finding class tests
# ===========================================================================


class TestFinding:
    def test_cve_id_from_aliases(self):
        c = Component("PyPI", "pkg", "1.0")
        f = Finding(c, vuln_id="GHSA-xxxx", aliases=["CVE-2024-1234", "GHSA-xxxx"])
        assert f.cve_id() == "CVE-2024-1234"

    def test_cve_id_from_vuln_id(self):
        c = Component("PyPI", "pkg", "1.0")
        f = Finding(c, vuln_id="CVE-2024-5678")
        assert f.cve_id() == "CVE-2024-5678"

    def test_cve_id_no_cve(self):
        c = Component("PyPI", "pkg", "1.0")
        f = Finding(c, vuln_id="GHSA-xxxx", aliases=["GHSA-yyyy"])
        assert f.cve_id() == ""

    def test_sort_key_kev_first(self):
        c = Component("PyPI", "pkg", "1.0")
        f_kev = Finding(c, vuln_id="CVE-1", in_kev=True, epss=0.1)
        f_high_epss = Finding(c, vuln_id="CVE-2", in_kev=False, epss=0.99)
        assert f_kev.sort_key() < f_high_epss.sort_key()

    def test_sort_key_epss_descending(self):
        c = Component("PyPI", "pkg", "1.0")
        f_high = Finding(c, vuln_id="CVE-1", epss=0.9)
        f_low = Finding(c, vuln_id="CVE-2", epss=0.1)
        assert f_high.sort_key() < f_low.sort_key()

    def test_sort_key_none_epss(self):
        c = Component("PyPI", "pkg", "1.0")
        f_no_epss = Finding(c, vuln_id="CVE-1", epss=None)
        f_with_epss = Finding(c, vuln_id="CVE-2", epss=0.5)
        # None EPSS should sort after any EPSS value
        assert f_with_epss.sort_key() < f_no_epss.sort_key()

    def test_to_dict(self):
        c = Component("PyPI", "pkg", "1.0")
        f = Finding(
            c,
            vuln_id="CVE-2024-1234",
            aliases=["GHSA-xxxx"],
            summary="A bug",
            severity=[{"type": "CVSS_V3", "score": "7.5"}],
            epss=0.42,
            in_kev=True,
            fixed_versions=["1.1"],
        )
        d = f.to_dict()
        assert d["vuln_id"] == "CVE-2024-1234"
        assert d["epss"] == 0.42
        assert d["in_kev"] is True
        assert d["fixed_versions"] == ["1.1"]
        assert d["component"]["name"] == "pkg"

    def test_to_dict_no_epss(self):
        c = Component("PyPI", "pkg", "1.0")
        f = Finding(c, vuln_id="CVE-1")
        d = f.to_dict()
        assert "epss" not in d


# ===========================================================================
# Purl parsing tests
# ===========================================================================


class TestParsePurl:
    def test_npm_simple(self):
        eco, name, ver = _parse_purl("pkg:npm/lodash@4.17.21")
        assert eco == "npm"
        assert name == "lodash"
        assert ver == "4.17.21"

    def test_npm_scoped(self):
        eco, name, ver = _parse_purl("pkg:npm/%40angular/core@14.0.0")
        assert eco == "npm"
        assert name == "@angular/core"
        assert ver == "14.0.0"

    def test_pypi(self):
        eco, name, ver = _parse_purl("pkg:pypi/requests@2.28.0")
        assert eco == "PyPI"
        assert name == "requests"
        assert ver == "2.28.0"

    def test_maven(self):
        eco, name, ver = _parse_purl("pkg:maven/org.apache.logging.log4j/log4j-core@2.17.0")
        assert eco == "Maven"
        assert name == "org.apache.logging.log4j:log4j-core"
        assert ver == "2.17.0"

    def test_golang(self):
        eco, name, ver = _parse_purl("pkg:golang/github.com/gin-gonic/gin@1.9.0")
        assert eco == "Go"
        assert name == "gin"
        assert ver == "1.9.0"

    def test_cargo(self):
        eco, name, ver = _parse_purl("pkg:cargo/serde@1.0.160")
        assert eco == "crates.io"
        assert name == "serde"
        assert ver == "1.0.160"

    def test_gem(self):
        eco, name, ver = _parse_purl("pkg:gem/rails@7.0.0")
        assert eco == "RubyGems"
        assert name == "rails"
        assert ver == "7.0.0"

    def test_nuget(self):
        eco, name, ver = _parse_purl("pkg:nuget/Newtonsoft.Json@13.0.1")
        assert eco == "NuGet"
        assert name == "Newtonsoft.Json"
        assert ver == "13.0.1"

    def test_composer(self):
        eco, name, ver = _parse_purl("pkg:composer/laravel/framework@9.0.0")
        assert eco == "Packagist"
        assert name == "framework"
        assert ver == "9.0.0"

    def test_with_qualifiers(self):
        eco, name, ver = _parse_purl("pkg:npm/lodash@4.17.21?repository_url=https://npm.org")
        assert eco == "npm"
        assert name == "lodash"
        assert ver == "4.17.21"

    def test_with_subpath(self):
        eco, name, ver = _parse_purl("pkg:npm/lodash@4.17.21#subpath/file.js")
        assert eco == "npm"
        assert name == "lodash"
        assert ver == "4.17.21"

    def test_no_version(self):
        eco, name, ver = _parse_purl("pkg:npm/lodash")
        assert eco == "npm"
        assert name == "lodash"
        assert ver == ""

    def test_empty_string(self):
        assert _parse_purl("") == ("", "", "")

    def test_none(self):
        assert _parse_purl(None) == ("", "", "")

    def test_invalid_no_slash(self):
        assert _parse_purl("pkg:noslash") == ("", "", "")

    def test_unknown_type(self):
        eco, name, ver = _parse_purl("pkg:unknown/mylib@1.0")
        assert eco == ""
        assert name == "mylib"
        assert ver == "1.0"

    def test_percent_encoded_scope(self):
        eco, name, ver = _parse_purl("pkg:npm/%40scope/package@1.0.0")
        assert name == "@scope/package"


# ===========================================================================
# SBOM format detection tests
# ===========================================================================


class TestDetectSbomFormat:
    def test_cyclonedx_bom_format(self):
        assert detect_sbom_format({"bomFormat": "CycloneDX"}) == "cyclonedx"

    def test_spdx_version(self):
        assert detect_sbom_format({"spdxVersion": "SPDX-2.3"}) == "spdx"

    def test_cyclonedx_heuristic_components(self):
        assert detect_sbom_format({"components": []}) == "cyclonedx"

    def test_spdx_heuristic_packages(self):
        assert detect_sbom_format({"packages": []}) == "spdx"

    def test_unknown(self):
        assert detect_sbom_format({}) == "unknown"

    def test_unknown_non_list(self):
        assert detect_sbom_format({"components": "not a list"}) == "unknown"


# ===========================================================================
# CycloneDX parsing tests
# ===========================================================================


class TestParseCyclonedx:
    def test_basic_component_with_purl(self):
        components = _parse_cyclonedx(
            _make_cyclonedx_sbom(
                [
                    {
                        "type": "library",
                        "name": "requests",
                        "version": "2.28.0",
                        "purl": "pkg:pypi/requests@2.28.0",
                    }
                ]
            )
        )
        assert len(components) == 1
        assert components[0].ecosystem == "PyPI"
        assert components[0].name == "requests"
        assert components[0].version == "2.28.0"

    def test_component_without_purl(self):
        components = _parse_cyclonedx(
            _make_cyclonedx_sbom(
                [
                    {
                        "type": "library",
                        "name": "mylib",
                        "version": "1.0.0",
                    }
                ]
            )
        )
        assert len(components) == 1
        assert components[0].name == "mylib"
        assert components[0].ecosystem == ""

    def test_deduplication(self):
        comp = {
            "type": "library",
            "name": "requests",
            "version": "2.28.0",
            "purl": "pkg:pypi/requests@2.28.0",
        }
        components = _parse_cyclonedx(_make_cyclonedx_sbom([comp, comp]))
        assert len(components) == 1

    def test_missing_name_skipped(self):
        components = _parse_cyclonedx(_make_cyclonedx_sbom([{"type": "library", "version": "1.0"}]))
        assert len(components) == 0

    def test_missing_version_skipped(self):
        components = _parse_cyclonedx(_make_cyclonedx_sbom([{"type": "library", "name": "mylib"}]))
        assert len(components) == 0

    def test_non_dict_component_skipped(self):
        components = _parse_cyclonedx(_make_cyclonedx_sbom(["not a dict"]))
        assert len(components) == 0

    def test_empty_components(self):
        components = _parse_cyclonedx(_make_cyclonedx_sbom([]))
        assert len(components) == 0

    def test_none_components(self):
        sbom = _make_cyclonedx_sbom()
        sbom["components"] = None
        components = _parse_cyclonedx(sbom)
        assert len(components) == 0

    def test_bom_ref_preserved(self):
        components = _parse_cyclonedx(
            _make_cyclonedx_sbom(
                [
                    {
                        "bom-ref": "ref-001",
                        "name": "flask",
                        "version": "2.0.0",
                        "purl": "pkg:pypi/flask@2.0.0",
                    }
                ]
            )
        )
        assert components[0].bom_ref == "ref-001"

    def test_purl_name_fallback(self):
        """When name is empty but purl has a name, use purl name."""
        components = _parse_cyclonedx(
            _make_cyclonedx_sbom(
                [
                    {
                        "type": "library",
                        "name": "",
                        "version": "1.0.0",
                        "purl": "pkg:pypi/flask@2.0.0",
                    }
                ]
            )
        )
        assert len(components) == 1
        assert components[0].name == "flask"

    def test_multiple_ecosystems(self):
        components = _parse_cyclonedx(
            _make_cyclonedx_sbom(
                [
                    {"name": "flask", "version": "2.0", "purl": "pkg:pypi/flask@2.0"},
                    {"name": "lodash", "version": "4.17", "purl": "pkg:npm/lodash@4.17"},
                    {"name": "serde", "version": "1.0", "purl": "pkg:cargo/serde@1.0"},
                ]
            )
        )
        assert len(components) == 3
        ecosystems = {c.ecosystem for c in components}
        assert ecosystems == {"PyPI", "npm", "crates.io"}


# ===========================================================================
# SPDX parsing tests
# ===========================================================================


class TestParseSpdx:
    def test_basic_package_with_purl(self):
        packages = _parse_spdx(
            _make_spdx_sbom(
                [
                    {
                        "SPDXID": "SPDXRef-Package",
                        "name": "requests",
                        "versionInfo": "2.28.0",
                        "externalRefs": [
                            {
                                "referenceType": "purl",
                                "referenceLocator": "pkg:pypi/requests@2.28.0",
                            }
                        ],
                    }
                ]
            )
        )
        assert len(packages) == 1
        assert packages[0].ecosystem == "PyPI"
        assert packages[0].name == "requests"
        assert packages[0].version == "2.28.0"

    def test_package_without_purl(self):
        packages = _parse_spdx(
            _make_spdx_sbom(
                [
                    {
                        "SPDXID": "SPDXRef-Package",
                        "name": "mylib",
                        "versionInfo": "1.0.0",
                    }
                ]
            )
        )
        assert len(packages) == 1
        assert packages[0].name == "mylib"
        assert packages[0].ecosystem == ""

    def test_document_package_skipped(self):
        packages = _parse_spdx(
            _make_spdx_sbom(
                [
                    {
                        "SPDXID": "SPDXRef-DOCUMENT",
                        "name": "root-doc",
                        "versionInfo": "1.0",
                    }
                ]
            )
        )
        assert len(packages) == 0

    def test_missing_name_skipped(self):
        packages = _parse_spdx(_make_spdx_sbom([{"SPDXID": "SPDXRef-1", "versionInfo": "1.0"}]))
        assert len(packages) == 0

    def test_missing_version_skipped(self):
        packages = _parse_spdx(_make_spdx_sbom([{"SPDXID": "SPDXRef-1", "name": "mylib"}]))
        assert len(packages) == 0

    def test_deduplication(self):
        pkg = {
            "SPDXID": "SPDXRef-1",
            "name": "requests",
            "versionInfo": "2.28.0",
            "externalRefs": [
                {
                    "referenceType": "purl",
                    "referenceLocator": "pkg:pypi/requests@2.28.0",
                }
            ],
        }
        packages = _parse_spdx(_make_spdx_sbom([pkg, pkg]))
        assert len(packages) == 1

    def test_non_dict_package_skipped(self):
        packages = _parse_spdx(_make_spdx_sbom(["not a dict"]))
        assert len(packages) == 0

    def test_none_packages(self):
        sbom = _make_spdx_sbom()
        sbom["packages"] = None
        packages = _parse_spdx(sbom)
        assert len(packages) == 0

    def test_non_dict_external_ref_skipped(self):
        packages = _parse_spdx(
            _make_spdx_sbom(
                [
                    {
                        "SPDXID": "SPDXRef-1",
                        "name": "mylib",
                        "versionInfo": "1.0",
                        "externalRefs": ["not a dict"],
                    }
                ]
            )
        )
        assert len(packages) == 1
        assert packages[0].purl == ""

    def test_purl_locator_detection(self):
        """referenceType doesn't say purl but locator starts with pkg:."""
        packages = _parse_spdx(
            _make_spdx_sbom(
                [
                    {
                        "SPDXID": "SPDXRef-1",
                        "name": "oldname",
                        "versionInfo": "1.0",
                        "externalRefs": [
                            {
                                "referenceType": "something-else",
                                "referenceLocator": "pkg:npm/lodash@4.17.21",
                            }
                        ],
                    }
                ]
            )
        )
        assert len(packages) == 1
        assert packages[0].ecosystem == "npm"
        assert packages[0].name == "lodash"


# ===========================================================================
# parse_sbom tests
# ===========================================================================


class TestParseSbom:
    def test_cyclonedx(self):
        fmt, comps = parse_sbom(_make_cyclonedx_sbom([{"name": "a", "version": "1", "purl": "pkg:pypi/a@1"}]))
        assert fmt == "cyclonedx"
        assert len(comps) == 1

    def test_spdx(self):
        fmt, comps = parse_sbom(_make_spdx_sbom([{"SPDXID": "SPDXRef-1", "name": "a", "versionInfo": "1"}]))
        assert fmt == "spdx"
        assert len(comps) == 1

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unrecognized SBOM format"):
            parse_sbom({"random": "data"})


# ===========================================================================
# OSV query building tests
# ===========================================================================


class TestBuildOsvQuery:
    def test_with_purl(self):
        c = Component("PyPI", "requests", "2.28.0", purl="pkg:pypi/requests@2.28.0")
        q = _build_osv_query(c)
        assert q == {"package": {"purl": "pkg:pypi/requests@2.28.0"}}

    def test_with_ecosystem(self):
        c = Component("PyPI", "requests", "2.28.0")
        q = _build_osv_query(c)
        assert q == {
            "package": {"name": "requests", "ecosystem": "PyPI"},
            "version": "2.28.0",
        }

    def test_without_ecosystem(self):
        c = Component("", "mylib", "1.0.0")
        q = _build_osv_query(c)
        assert q == {
            "package": {"name": "mylib"},
            "version": "1.0.0",
        }


# ===========================================================================
# OSV batch query tests
# ===========================================================================


class TestQueryOsvBatch:
    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_basic_query(self, mock_req):
        vuln = _make_osv_vuln("GHSA-1234", aliases=["CVE-2024-0001"])
        mock_req.return_value = _mock_response(json_data={"results": [{"vulns": [vuln]}]})
        comps = [Component("PyPI", "requests", "2.28.0", purl="pkg:pypi/requests@2.28.0")]
        results = query_osv_batch(comps)
        assert len(results) == 1
        assert len(results[comps[0].key()]) == 1
        assert results[comps[0].key()][0]["id"] == "GHSA-1234"

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_no_vulns(self, mock_req):
        mock_req.return_value = _mock_response(json_data={"results": [{"vulns": []}]})
        comps = [Component("PyPI", "safe-lib", "1.0.0")]
        results = query_osv_batch(comps)
        assert len(results[comps[0].key()]) == 0

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_api_failure_graceful(self, mock_req):
        mock_req.side_effect = Exception("Network error")
        comps = [Component("PyPI", "requests", "2.28.0")]
        results = query_osv_batch(comps)
        assert comps[0].key() in results
        assert len(results[comps[0].key()]) == 0

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_multiple_components(self, mock_req):
        vuln1 = _make_osv_vuln("GHSA-1111")
        vuln2 = _make_osv_vuln("GHSA-2222")
        mock_req.return_value = _mock_response(
            json_data={
                "results": [
                    {"vulns": [vuln1]},
                    {"vulns": [vuln2]},
                ]
            }
        )
        comps = [
            Component("PyPI", "requests", "2.28.0"),
            Component("npm", "lodash", "4.17.20"),
        ]
        results = query_osv_batch(comps)
        assert len(results) == 2

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_empty_results_array(self, mock_req):
        mock_req.return_value = _mock_response(json_data={"results": []})
        comps = [Component("PyPI", "requests", "2.28.0")]
        results = query_osv_batch(comps)
        assert comps[0].key() in results
        assert len(results[comps[0].key()]) == 0

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_non_dict_vuln_skipped(self, mock_req):
        mock_req.return_value = _mock_response(json_data={"results": [{"vulns": ["not-a-dict"]}]})
        comps = [Component("PyPI", "requests", "2.28.0")]
        results = query_osv_batch(comps)
        assert len(results[comps[0].key()]) == 0

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_null_vulns_list(self, mock_req):
        mock_req.return_value = _mock_response(json_data={"results": [{"vulns": None}]})
        comps = [Component("PyPI", "requests", "2.28.0")]
        results = query_osv_batch(comps)
        assert len(results[comps[0].key()]) == 0


# ===========================================================================
# EPSS enrichment tests
# ===========================================================================


class TestFetchEpssScores:
    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_basic_fetch(self, mock_req):
        mock_req.return_value = _mock_response(
            json_data={
                "data": [
                    {"cve": "CVE-2024-0001", "epss": "0.85"},
                    {"cve": "CVE-2024-0002", "epss": "0.12"},
                ]
            }
        )
        scores = fetch_epss_scores(["CVE-2024-0001", "CVE-2024-0002"])
        assert scores["CVE-2024-0001"] == pytest.approx(0.85)
        assert scores["CVE-2024-0002"] == pytest.approx(0.12)

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_empty_input(self, mock_req):
        scores = fetch_epss_scores([])
        assert scores == {}
        mock_req.assert_not_called()

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_api_failure_graceful(self, mock_req):
        mock_req.side_effect = Exception("EPSS down")
        scores = fetch_epss_scores(["CVE-2024-0001"])
        assert scores == {}

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_deduplication(self, mock_req):
        mock_req.return_value = _mock_response(json_data={"data": [{"cve": "CVE-2024-0001", "epss": "0.5"}]})
        scores = fetch_epss_scores(["CVE-2024-0001", "CVE-2024-0001", "CVE-2024-0001"])
        assert len(scores) == 1

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_invalid_epss_value_skipped(self, mock_req):
        mock_req.return_value = _mock_response(json_data={"data": [{"cve": "CVE-2024-0001", "epss": "not-a-number"}]})
        scores = fetch_epss_scores(["CVE-2024-0001"])
        assert "CVE-2024-0001" not in scores

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_missing_data_key(self, mock_req):
        mock_req.return_value = _mock_response(json_data={})
        scores = fetch_epss_scores(["CVE-2024-0001"])
        assert scores == {}

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_null_data_key(self, mock_req):
        mock_req.return_value = _mock_response(json_data={"data": None})
        scores = fetch_epss_scores(["CVE-2024-0001"])
        assert scores == {}


# ===========================================================================
# CISA KEV enrichment tests
# ===========================================================================


class TestFetchKevSet:
    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_basic_fetch(self, mock_req):
        mock_req.return_value = _mock_response(
            json_data={
                "vulnerabilities": [
                    {"cveID": "CVE-2024-0001"},
                    {"cveID": "CVE-2024-0002"},
                ]
            }
        )
        kev = fetch_kev_set()
        assert kev == {"CVE-2024-0001", "CVE-2024-0002"}

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_api_failure_returns_empty(self, mock_req):
        mock_req.side_effect = Exception("KEV down")
        kev = fetch_kev_set()
        assert kev == set()

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_empty_catalog(self, mock_req):
        mock_req.return_value = _mock_response(json_data={"vulnerabilities": []})
        kev = fetch_kev_set()
        assert kev == set()

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_null_vulnerabilities(self, mock_req):
        mock_req.return_value = _mock_response(json_data={"vulnerabilities": None})
        kev = fetch_kev_set()
        assert kev == set()

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_non_dict_entry_skipped(self, mock_req):
        mock_req.return_value = _mock_response(
            json_data={"vulnerabilities": ["not-a-dict", {"cveID": "CVE-2024-0001"}]}
        )
        kev = fetch_kev_set()
        assert kev == {"CVE-2024-0001"}


# ===========================================================================
# Extraction helper tests
# ===========================================================================


class TestExtractFixedVersions:
    def test_basic_fixed(self):
        vuln = _make_osv_vuln(fixed="1.2.3")
        assert _extract_fixed_versions(vuln) == ["1.2.3"]

    def test_no_affected(self):
        assert _extract_fixed_versions({}) == []

    def test_null_affected(self):
        assert _extract_fixed_versions({"affected": None}) == []

    def test_multiple_fixed(self):
        vuln = {
            "affected": [
                {
                    "ranges": [
                        {"events": [{"introduced": "0"}, {"fixed": "1.0.0"}]},
                        {"events": [{"introduced": "2.0.0"}, {"fixed": "2.1.0"}]},
                    ]
                }
            ]
        }
        fixed = _extract_fixed_versions(vuln)
        assert "1.0.0" in fixed
        assert "2.1.0" in fixed

    def test_dedup_fixed(self):
        vuln = {
            "affected": [
                {
                    "ranges": [
                        {"events": [{"fixed": "1.0.0"}]},
                        {"events": [{"fixed": "1.0.0"}]},
                    ]
                }
            ]
        }
        fixed = _extract_fixed_versions(vuln)
        assert fixed == ["1.0.0"]

    def test_non_dict_affected_skipped(self):
        assert _extract_fixed_versions({"affected": ["not-a-dict"]}) == []

    def test_non_dict_range_skipped(self):
        assert _extract_fixed_versions({"affected": [{"ranges": ["not-a-dict"]}]}) == []

    def test_non_dict_event_skipped(self):
        assert _extract_fixed_versions({"affected": [{"ranges": [{"events": ["not"]}]}]}) == []


class TestExtractSeverity:
    def test_basic_severity(self):
        vuln = {"severity": [{"type": "CVSS_V3", "score": "7.5"}]}
        sev = _extract_severity(vuln)
        assert len(sev) == 1
        assert sev[0]["type"] == "CVSS_V3"
        assert sev[0]["score"] == "7.5"

    def test_no_severity(self):
        assert _extract_severity({}) == []

    def test_null_severity(self):
        assert _extract_severity({"severity": None}) == []

    def test_database_specific_fallback(self):
        vuln = {"database_specific": {"severity": "HIGH"}}
        sev = _extract_severity(vuln)
        assert len(sev) == 1
        assert sev[0]["score"] == "HIGH"

    def test_database_specific_not_used_when_severity_exists(self):
        vuln = {
            "severity": [{"type": "CVSS_V3", "score": "9.8"}],
            "database_specific": {"severity": "CRITICAL"},
        }
        sev = _extract_severity(vuln)
        assert len(sev) == 1
        assert sev[0]["score"] == "9.8"

    def test_non_dict_severity_skipped(self):
        assert _extract_severity({"severity": ["not-a-dict"]}) == []

    def test_null_database_specific(self):
        assert _extract_severity({"database_specific": None}) == []


# ===========================================================================
# HTTP helper tests
# ===========================================================================


class TestHttpRequest:
    @patch("manus_agent.tools.scan_sbom.requests.request")
    def test_basic_get(self, mock_req):
        mock_req.return_value = _mock_response(json_data={"ok": True})
        resp = _http_request("GET", "https://example.com/api")
        assert resp.json() == {"ok": True}

    @patch("manus_agent.tools.scan_sbom.requests.request")
    def test_connection_error_raises(self, mock_req):
        import requests as req_lib

        mock_req.side_effect = req_lib.exceptions.ConnectionError("refused")
        with pytest.raises(req_lib.exceptions.ConnectionError):
            _http_request("GET", "https://example.com/api")

    @patch("manus_agent.tools.scan_sbom.requests.request")
    def test_timeout_error_raises(self, mock_req):
        import requests as req_lib

        mock_req.side_effect = req_lib.exceptions.Timeout("timed out")
        with pytest.raises(req_lib.exceptions.Timeout):
            _http_request("GET", "https://example.com/api")


# ===========================================================================
# Core scan_sbom tests
# ===========================================================================


class TestScanSbom:
    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_basic_scan_with_findings(self, mock_osv, mock_epss, mock_kev):
        vuln = _make_osv_vuln(
            "GHSA-test-0001",
            aliases=["CVE-2024-0001"],
            summary="A critical bug",
            fixed="2.29.0",
            severity_score="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        )
        mock_osv.return_value = {"PyPI:requests:2.28.0": [vuln]}
        mock_epss.return_value = {"CVE-2024-0001": 0.85}
        mock_kev.return_value = {"CVE-2024-0001"}

        sbom = _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
        result = scan_sbom(sbom)

        assert result["format"] == "cyclonedx"
        assert result["component_count"] == 1
        assert result["vulnerable_component_count"] == 1
        assert result["total_finding_count"] == 1
        assert result["kev_count"] == 1
        assert result["critical_count"] == 1
        assert len(result["findings"]) == 1
        assert result["findings"][0]["in_kev"] is True
        assert result["findings"][0]["epss"] == pytest.approx(0.85)
        assert result["findings"][0]["fixed_versions"] == ["2.29.0"]

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_no_vulnerabilities(self, mock_osv, mock_epss, mock_kev):
        mock_osv.return_value = {"PyPI:requests:2.28.0": []}
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        sbom = _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
        result = scan_sbom(sbom)

        assert result["total_finding_count"] == 0
        assert result["vulnerable_component_count"] == 0
        assert "No known vulnerabilities" in result["message"]

    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_skip_epss_and_kev(self, mock_osv):
        vuln = _make_osv_vuln("GHSA-0001", aliases=["CVE-2024-0001"])
        mock_osv.return_value = {"PyPI:requests:2.28.0": [vuln]}

        sbom = _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
        result = scan_sbom(sbom, skip_epss=True, skip_kev=True)

        assert result["total_finding_count"] == 1
        assert result["kev_count"] == 0

    def test_invalid_sbom_format(self):
        result = scan_sbom({"random": "data"})
        assert result["format"] == "unknown"
        assert result["component_count"] == 0
        assert "parsing failed" in result["message"].lower()

    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_empty_components(self, mock_osv):
        sbom = _make_cyclonedx_sbom([])
        result = scan_sbom(sbom)
        assert result["component_count"] == 0
        assert "No components" in result["message"]
        mock_osv.assert_not_called()

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_osv_failure_graceful(self, mock_osv, mock_epss, mock_kev):
        mock_osv.side_effect = Exception("OSV down")
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        sbom = _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
        result = scan_sbom(sbom)
        assert result["total_finding_count"] == 0
        assert any("OSV" in e for e in result["errors"])

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_epss_failure_graceful(self, mock_osv, mock_epss, mock_kev):
        vuln = _make_osv_vuln("GHSA-0001", aliases=["CVE-2024-0001"])
        mock_osv.return_value = {"PyPI:requests:2.28.0": [vuln]}
        mock_epss.side_effect = Exception("EPSS down")
        mock_kev.return_value = set()

        sbom = _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
        result = scan_sbom(sbom)
        assert result["total_finding_count"] == 1
        assert any("EPSS" in e for e in result["errors"])

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_kev_failure_graceful(self, mock_osv, mock_epss, mock_kev):
        vuln = _make_osv_vuln("GHSA-0001", aliases=["CVE-2024-0001"])
        mock_osv.return_value = {"PyPI:requests:2.28.0": [vuln]}
        mock_epss.return_value = {"CVE-2024-0001": 0.5}
        mock_kev.side_effect = Exception("KEV down")

        sbom = _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
        result = scan_sbom(sbom)
        assert result["total_finding_count"] == 1
        assert any("KEV" in e for e in result["errors"])

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_finding_deduplication(self, mock_osv, mock_epss, mock_kev):
        vuln = _make_osv_vuln("GHSA-0001")
        # Same vuln appears twice for same component
        mock_osv.return_value = {"PyPI:requests:2.28.0": [vuln, vuln]}
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        sbom = _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
        result = scan_sbom(sbom)
        assert result["total_finding_count"] == 1

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_critical_count_high_epss(self, mock_osv, mock_epss, mock_kev):
        vuln = _make_osv_vuln("GHSA-0001", aliases=["CVE-2024-0001"])
        mock_osv.return_value = {"PyPI:requests:2.28.0": [vuln]}
        mock_epss.return_value = {"CVE-2024-0001": 0.75}  # >= 0.7 threshold
        mock_kev.return_value = set()

        sbom = _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
        result = scan_sbom(sbom)
        assert result["critical_count"] == 1
        assert result["kev_count"] == 0

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_findings_sorted_kev_first(self, mock_osv, mock_epss, mock_kev):
        vuln1 = _make_osv_vuln("GHSA-0001", aliases=["CVE-2024-0001"])
        vuln2 = _make_osv_vuln("GHSA-0002", aliases=["CVE-2024-0002"])
        mock_osv.return_value = {
            "PyPI:pkg-a:1.0": [vuln1],
            "PyPI:pkg-b:2.0": [vuln2],
        }
        mock_epss.return_value = {"CVE-2024-0001": 0.99, "CVE-2024-0002": 0.01}
        mock_kev.return_value = {"CVE-2024-0002"}  # lower EPSS but in KEV

        sbom = _make_cyclonedx_sbom(
            [
                {"name": "pkg-a", "version": "1.0", "purl": "pkg:pypi/pkg-a@1.0"},
                {"name": "pkg-b", "version": "2.0", "purl": "pkg:pypi/pkg-b@2.0"},
            ]
        )
        result = scan_sbom(sbom)
        findings = result["findings"]
        assert len(findings) == 2
        # KEV finding should be first despite lower EPSS
        assert findings[0]["in_kev"] is True
        assert findings[0]["vuln_id"] == "GHSA-0002"

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_kev_match_via_alias(self, mock_osv, mock_epss, mock_kev):
        """KEV matching should work even when the vuln_id is GHSA and alias is CVE."""
        vuln = _make_osv_vuln("GHSA-0001", aliases=["CVE-2024-0001"])
        mock_osv.return_value = {"PyPI:requests:2.28.0": [vuln]}
        mock_epss.return_value = {}
        mock_kev.return_value = {"CVE-2024-0001"}

        sbom = _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
        result = scan_sbom(sbom)
        assert result["findings"][0]["in_kev"] is True

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_long_summary_truncated(self, mock_osv, mock_epss, mock_kev):
        long_summary = "A" * 500
        vuln = _make_osv_vuln("GHSA-0001", summary=long_summary)
        mock_osv.return_value = {"PyPI:requests:2.28.0": [vuln]}
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        sbom = _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
        result = scan_sbom(sbom)
        assert len(result["findings"][0]["summary"]) <= 300

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_vuln_without_id_skipped(self, mock_osv, mock_epss, mock_kev):
        mock_osv.return_value = {"PyPI:requests:2.28.0": [{"id": "", "aliases": []}]}
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        sbom = _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
        result = scan_sbom(sbom)
        assert result["total_finding_count"] == 0

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_spdx_scan(self, mock_osv, mock_epss, mock_kev):
        vuln = _make_osv_vuln("GHSA-0001", aliases=["CVE-2024-0001"])
        mock_osv.return_value = {"npm:lodash:4.17.20": [vuln]}
        mock_epss.return_value = {"CVE-2024-0001": 0.3}
        mock_kev.return_value = set()

        sbom = _make_spdx_sbom(
            [
                {
                    "SPDXID": "SPDXRef-1",
                    "name": "lodash",
                    "versionInfo": "4.17.20",
                    "externalRefs": [{"referenceType": "purl", "referenceLocator": "pkg:npm/lodash@4.17.20"}],
                }
            ]
        )
        result = scan_sbom(sbom)
        assert result["format"] == "spdx"
        assert result["total_finding_count"] == 1

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_multiple_vulns_one_component(self, mock_osv, mock_epss, mock_kev):
        vuln1 = _make_osv_vuln("GHSA-0001", aliases=["CVE-2024-0001"])
        vuln2 = _make_osv_vuln("GHSA-0002", aliases=["CVE-2024-0002"])
        mock_osv.return_value = {"PyPI:requests:2.28.0": [vuln1, vuln2]}
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        sbom = _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
        result = scan_sbom(sbom)
        assert result["total_finding_count"] == 2
        assert result["vulnerable_component_count"] == 1


# ===========================================================================
# Strands tool entry point tests
# ===========================================================================


class TestScanSbomTool:
    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_with_sbom_content(self, mock_scan):
        mock_scan.return_value = {
            "format": "cyclonedx",
            "component_count": 1,
            "total_finding_count": 0,
            "errors": [],
        }
        sbom_json = json.dumps(_make_cyclonedx_sbom([{"name": "a", "version": "1"}]))
        tool_use = {
            "toolUseId": "test-001",
            "input": {"sbom_content": sbom_json},
        }
        result = scan_sbom_tool(tool_use)
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-001"

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    @patch("builtins.open", new_callable=mock_open, read_data='{"bomFormat": "CycloneDX", "components": []}')
    def test_with_sbom_path(self, mock_file, mock_scan):
        mock_scan.return_value = {
            "format": "cyclonedx",
            "component_count": 0,
            "total_finding_count": 0,
            "errors": [],
        }
        tool_use = {
            "toolUseId": "test-002",
            "input": {"sbom_path": "/tmp/bom.json"},
        }
        result = scan_sbom_tool(tool_use)
        assert result["status"] == "success"

    def test_file_not_found(self):
        tool_use = {
            "toolUseId": "test-003",
            "input": {"sbom_path": "/nonexistent/bom.json"},
        }
        result = scan_sbom_tool(tool_use)
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    def test_invalid_json_content(self):
        tool_use = {
            "toolUseId": "test-004",
            "input": {"sbom_content": "not json"},
        }
        result = scan_sbom_tool(tool_use)
        assert result["status"] == "error"
        assert "Invalid JSON" in result["content"][0]["text"]

    def test_no_input(self):
        tool_use = {
            "toolUseId": "test-005",
            "input": {},
        }
        result = scan_sbom_tool(tool_use)
        assert result["status"] == "error"
        assert "required" in result["content"][0]["text"].lower()

    def test_empty_input(self):
        tool_use = {
            "toolUseId": "test-006",
            "input": None,
        }
        result = scan_sbom_tool(tool_use)
        assert result["status"] == "error"

    @patch("builtins.open", side_effect=FileNotFoundError("No such file"))
    def test_sbom_path_file_not_found(self, mock_file):
        tool_use = {
            "toolUseId": "test-007",
            "input": {"sbom_path": "/missing/bom.json"},
        }
        result = scan_sbom_tool(tool_use)
        assert result["status"] == "error"

    @patch("builtins.open", new_callable=mock_open, read_data="{{invalid json")
    def test_sbom_path_invalid_json(self, mock_file):
        tool_use = {
            "toolUseId": "test-008",
            "input": {"sbom_path": "/tmp/bad.json"},
        }
        result = scan_sbom_tool(tool_use)
        assert result["status"] == "error"
        assert "Invalid JSON" in result["content"][0]["text"]

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_skip_flags_passed(self, mock_scan):
        mock_scan.return_value = {
            "format": "cyclonedx",
            "component_count": 0,
            "total_finding_count": 0,
            "errors": [],
        }
        sbom_json = json.dumps(_make_cyclonedx_sbom([]))
        tool_use = {
            "toolUseId": "test-009",
            "input": {
                "sbom_content": sbom_json,
                "skip_epss": True,
                "skip_kev": True,
            },
        }
        scan_sbom_tool(tool_use)
        mock_scan.assert_called_once()
        call_kwargs = mock_scan.call_args
        assert call_kwargs[1]["skip_epss"] is True
        assert call_kwargs[1]["skip_kev"] is True


# ===========================================================================
# CLI subcommand tests
# ===========================================================================


class TestCliSbomScan:
    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_text_output(self, mock_scan, tmp_path, capsys):
        mock_scan.return_value = {
            "format": "cyclonedx",
            "component_count": 5,
            "vulnerable_component_count": 1,
            "total_finding_count": 2,
            "kev_count": 1,
            "critical_count": 1,
            "findings": [
                {
                    "vuln_id": "CVE-2024-0001",
                    "aliases": ["CVE-2024-0001"],
                    "summary": "A bug",
                    "severity": [{"type": "CVSS_V3", "score": "9.8"}],
                    "epss": 0.9,
                    "in_kev": True,
                    "fixed_versions": ["2.29.0"],
                    "component": {"ecosystem": "PyPI", "name": "requests", "version": "2.28.0", "purl": ""},
                }
            ],
            "components_scanned": [],
            "errors": [],
            "message": "test",
        }

        sbom_file = tmp_path / "bom.json"
        sbom_file.write_text(
            json.dumps(
                _make_cyclonedx_sbom([{"name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"}])
            )
        )

        from manus_agent.cli import _run_sbom_scan

        exit_code = _run_sbom_scan([str(sbom_file)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "SBOM Vulnerability Scan" in captured.out
        assert "CVE-2024-0001" in captured.out
        assert "[KEV]" in captured.out
        assert "EPSS: 0.9" in captured.out
        assert "Fixed in:" in captured.out

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_json_output(self, mock_scan, tmp_path, capsys):
        mock_scan.return_value = {
            "format": "cyclonedx",
            "component_count": 1,
            "vulnerable_component_count": 0,
            "total_finding_count": 0,
            "kev_count": 0,
            "critical_count": 0,
            "findings": [],
            "components_scanned": [],
            "errors": [],
            "message": "No vulns",
        }

        sbom_file = tmp_path / "bom.json"
        sbom_file.write_text(json.dumps(_make_cyclonedx_sbom([])))

        from manus_agent.cli import _run_sbom_scan

        exit_code = _run_sbom_scan([str(sbom_file), "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["format"] == "cyclonedx"

    def test_file_not_found(self, capsys):
        from manus_agent.cli import _run_sbom_scan

        exit_code = _run_sbom_scan(["/nonexistent/bom.json"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()

    def test_invalid_json(self, tmp_path, capsys):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all")

        from manus_agent.cli import _run_sbom_scan

        exit_code = _run_sbom_scan([str(bad_file)])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Invalid JSON" in captured.err

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_skip_flags(self, mock_scan, tmp_path):
        mock_scan.return_value = {
            "format": "cyclonedx",
            "component_count": 0,
            "vulnerable_component_count": 0,
            "total_finding_count": 0,
            "kev_count": 0,
            "critical_count": 0,
            "findings": [],
            "components_scanned": [],
            "errors": [],
            "message": "No components",
        }

        sbom_file = tmp_path / "bom.json"
        sbom_file.write_text(json.dumps(_make_cyclonedx_sbom([])))

        from manus_agent.cli import _run_sbom_scan

        _run_sbom_scan([str(sbom_file), "--skip-epss", "--skip-kev"])
        mock_scan.assert_called_once()
        call_kwargs = mock_scan.call_args
        assert call_kwargs[1]["skip_epss"] is True
        assert call_kwargs[1]["skip_kev"] is True

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_no_findings_message(self, mock_scan, tmp_path, capsys):
        mock_scan.return_value = {
            "format": "cyclonedx",
            "component_count": 3,
            "vulnerable_component_count": 0,
            "total_finding_count": 0,
            "kev_count": 0,
            "critical_count": 0,
            "findings": [],
            "components_scanned": [],
            "errors": [],
            "message": "No vulns",
        }

        sbom_file = tmp_path / "bom.json"
        sbom_file.write_text(json.dumps(_make_cyclonedx_sbom([])))

        from manus_agent.cli import _run_sbom_scan

        exit_code = _run_sbom_scan([str(sbom_file)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No known vulnerabilities" in captured.out

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_errors_in_stderr(self, mock_scan, tmp_path, capsys):
        mock_scan.return_value = {
            "format": "cyclonedx",
            "component_count": 1,
            "vulnerable_component_count": 0,
            "total_finding_count": 0,
            "kev_count": 0,
            "critical_count": 0,
            "findings": [],
            "components_scanned": [],
            "errors": ["EPSS enrichment failed: timeout"],
            "message": "test",
        }

        sbom_file = tmp_path / "bom.json"
        sbom_file.write_text(json.dumps(_make_cyclonedx_sbom([])))

        from manus_agent.cli import _run_sbom_scan

        _run_sbom_scan([str(sbom_file)])
        captured = capsys.readouterr()
        assert "EPSS enrichment failed" in captured.err


# ===========================================================================
# CLI dispatch test
# ===========================================================================


class TestCliDispatch:
    @patch("manus_agent.cli._run_sbom_scan")
    def test_sbom_scan_dispatch(self, mock_run):
        mock_run.return_value = 0
        from manus_agent.cli import main

        with patch("sys.argv", ["manus-agent", "sbom-scan", "bom.json"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        mock_run.assert_called_once_with(["bom.json"])


# ===========================================================================
# End-to-end integration tests (all mocked)
# ===========================================================================


class TestEndToEnd:
    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_full_cyclonedx_pipeline(self, mock_req):
        """E2E test: CycloneDX SBOM → OSV batch → EPSS → KEV → ranked findings."""
        vuln = _make_osv_vuln(
            "GHSA-abcd-1234",
            aliases=["CVE-2024-9999"],
            summary="XSS in template engine",
            fixed="3.0.1",
            severity_score="CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N",
        )

        def side_effect(method, url, **kwargs):
            if "osv.dev" in url:
                return _mock_response(json_data={"results": [{"vulns": [vuln]}]})
            elif "first.org" in url:
                return _mock_response(json_data={"data": [{"cve": "CVE-2024-9999", "epss": "0.42"}]})
            elif "cisa.gov" in url:
                return _mock_response(json_data={"vulnerabilities": [{"cveID": "CVE-2024-9999"}]})
            return _mock_response()

        mock_req.side_effect = side_effect

        sbom = _make_cyclonedx_sbom(
            [
                {
                    "type": "library",
                    "name": "jinja2",
                    "version": "2.11.0",
                    "purl": "pkg:pypi/jinja2@2.11.0",
                }
            ]
        )
        result = scan_sbom(sbom)

        assert result["format"] == "cyclonedx"
        assert result["component_count"] == 1
        assert result["total_finding_count"] == 1
        assert result["kev_count"] == 1
        assert result["critical_count"] == 1

        finding = result["findings"][0]
        assert finding["vuln_id"] == "GHSA-abcd-1234"
        assert finding["in_kev"] is True
        assert finding["epss"] == pytest.approx(0.42)
        assert finding["fixed_versions"] == ["3.0.1"]

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_full_spdx_pipeline(self, mock_req):
        """E2E test: SPDX SBOM → OSV batch → clean scan."""

        def side_effect(method, url, **kwargs):
            if "osv.dev" in url:
                return _mock_response(json_data={"results": [{"vulns": []}]})
            elif "first.org" in url:
                return _mock_response(json_data={"data": []})
            elif "cisa.gov" in url:
                return _mock_response(json_data={"vulnerabilities": []})
            return _mock_response()

        mock_req.side_effect = side_effect

        sbom = _make_spdx_sbom(
            [
                {
                    "SPDXID": "SPDXRef-Package",
                    "name": "safe-lib",
                    "versionInfo": "5.0.0",
                    "externalRefs": [{"referenceType": "purl", "referenceLocator": "pkg:pypi/safe-lib@5.0.0"}],
                }
            ]
        )
        result = scan_sbom(sbom)
        assert result["format"] == "spdx"
        assert result["total_finding_count"] == 0
        assert "No known vulnerabilities" in result["message"]

    @patch("manus_agent.tools.scan_sbom._http_request")
    def test_large_sbom_multiple_components(self, mock_req):
        """Test with many components — verifies batching and aggregation."""
        components = []
        osv_results = []
        for i in range(10):
            components.append(
                {
                    "name": f"pkg-{i}",
                    "version": "1.0.0",
                    "purl": f"pkg:pypi/pkg-{i}@1.0.0",
                }
            )
            if i < 3:
                osv_results.append({"vulns": [_make_osv_vuln(f"GHSA-{i:04d}", aliases=[f"CVE-2024-{i:04d}"])]})
            else:
                osv_results.append({"vulns": []})

        def side_effect(method, url, **kwargs):
            if "osv.dev" in url:
                return _mock_response(json_data={"results": osv_results})
            elif "first.org" in url:
                return _mock_response(json_data={"data": []})
            elif "cisa.gov" in url:
                return _mock_response(json_data={"vulnerabilities": []})
            return _mock_response()

        mock_req.side_effect = side_effect

        sbom = _make_cyclonedx_sbom(components)
        result = scan_sbom(sbom)
        assert result["component_count"] == 10
        assert result["total_finding_count"] == 3
        assert result["vulnerable_component_count"] == 3
