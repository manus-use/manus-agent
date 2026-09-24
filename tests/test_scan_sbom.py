"""Comprehensive test suite for scan_sbom tool and sbom-scan CLI subcommand.

All HTTP calls are mocked — no real network traffic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the package is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ===================================================================
# Fixtures: sample SBOM data
# ===================================================================


def _cyclonedx_sbom(components: list[dict] | None = None) -> dict:
    """Return a minimal CycloneDX 1.5 JSON SBOM."""
    if components is None:
        components = [
            {
                "type": "library",
                "name": "requests",
                "version": "2.28.0",
                "purl": "pkg:pypi/requests@2.28.0",
            },
            {
                "type": "library",
                "name": "lodash",
                "version": "4.17.20",
                "purl": "pkg:npm/lodash@4.17.20",
            },
        ]
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "components": components,
    }


def _spdx_sbom(packages: list[dict] | None = None) -> dict:
    """Return a minimal SPDX 2.3 JSON SBOM."""
    if packages is None:
        packages = [
            {
                "SPDXID": "SPDXRef-Package-requests",
                "name": "requests",
                "versionInfo": "2.28.0",
                "externalRefs": [
                    {
                        "referenceType": "purl",
                        "referenceLocator": "pkg:pypi/requests@2.28.0",
                    }
                ],
            },
            {
                "SPDXID": "SPDXRef-Package-express",
                "name": "express",
                "versionInfo": "4.18.2",
                "externalRefs": [
                    {
                        "referenceType": "purl",
                        "referenceLocator": "pkg:npm/express@4.18.2",
                    }
                ],
            },
        ]
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "test-sbom",
        "packages": packages,
    }


def _osv_vuln(
    osv_id: str = "GHSA-abc-123",
    cve: str = "CVE-2024-1234",
    summary: str = "Test vuln",
    cvss_score: float | None = None,
) -> dict:
    """Return a minimal OSV vulnerability record."""
    v: dict[str, Any] = {
        "id": osv_id,
        "aliases": [cve] if cve else [],
        "summary": summary,
    }
    if cvss_score is not None:
        v["severity"] = [{"type": "CVSS_V3", "score": str(cvss_score)}]
    return v


# ===================================================================
# 1. SBOM format detection
# ===================================================================
class TestDetectSbomFormat:
    def test_cyclonedx_bomformat(self):
        from manus_agent.tools.scan_sbom import detect_sbom_format

        assert detect_sbom_format({"bomFormat": "CycloneDX"}) == "cyclonedx"

    def test_cyclonedx_components_only(self):
        from manus_agent.tools.scan_sbom import detect_sbom_format

        assert detect_sbom_format({"components": []}) == "cyclonedx"

    def test_spdx_version(self):
        from manus_agent.tools.scan_sbom import detect_sbom_format

        assert detect_sbom_format({"spdxVersion": "SPDX-2.3"}) == "spdx"

    def test_spdx_packages_only(self):
        from manus_agent.tools.scan_sbom import detect_sbom_format

        assert detect_sbom_format({"packages": []}) == "spdx"

    def test_unknown_format(self):
        from manus_agent.tools.scan_sbom import detect_sbom_format

        assert detect_sbom_format({"random": "data"}) == "unknown"

    def test_empty_dict(self):
        from manus_agent.tools.scan_sbom import detect_sbom_format

        assert detect_sbom_format({}) == "unknown"


# ===================================================================
# 2. PURL parsing
# ===================================================================
class TestParsePurl:
    def test_pypi_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:pypi/requests@2.28.0")
        assert result["ecosystem"] == "PyPI"
        assert result["name"] == "requests"
        assert result["version"] == "2.28.0"

    def test_npm_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:npm/lodash@4.17.20")
        assert result["ecosystem"] == "npm"
        assert result["name"] == "lodash"
        assert result["version"] == "4.17.20"

    def test_npm_scoped_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:npm/%40angular/core@16.0.0")
        assert result["ecosystem"] == "npm"
        assert result["name"] == "@angular/core"
        assert result["version"] == "16.0.0"

    def test_maven_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:maven/org.apache.logging.log4j/log4j-core@2.14.1")
        assert result["ecosystem"] == "Maven"
        assert result["name"] == "org.apache.logging.log4j:log4j-core"
        assert result["version"] == "2.14.1"

    def test_golang_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:golang/github.com/gin-gonic/gin@1.9.1")
        assert result["ecosystem"] == "Go"
        assert result["version"] == "1.9.1"

    def test_cargo_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:cargo/serde@1.0.193")
        assert result["ecosystem"] == "crates.io"
        assert result["name"] == "serde"
        assert result["version"] == "1.0.193"

    def test_nuget_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:nuget/Newtonsoft.Json@13.0.1")
        assert result["ecosystem"] == "NuGet"
        assert result["name"] == "Newtonsoft.Json"

    def test_purl_without_version(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:pypi/flask")
        assert result["name"] == "flask"
        assert result["version"] == ""

    def test_purl_with_qualifiers(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:pypi/requests@2.28.0?repository_url=https://pypi.org")
        assert result["name"] == "requests"
        assert result["version"] == "2.28.0"

    def test_purl_with_subpath(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:npm/lodash@4.17.20#dist/lodash.min.js")
        assert result["name"] == "lodash"
        assert result["version"] == "4.17.20"

    def test_empty_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("")
        assert result["name"] == ""

    def test_invalid_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("not-a-purl")
        assert result["name"] == ""

    def test_unknown_ecosystem_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:deb/debian/curl@7.68.0")
        assert result["ecosystem"] == "deb"
        assert result["name"] == "curl"


# ===================================================================
# 3. CycloneDX parsing
# ===================================================================
class TestParseCyclonedx:
    def test_basic_components(self):
        from manus_agent.tools.scan_sbom import _parse_cyclonedx

        data = _cyclonedx_sbom()
        components = _parse_cyclonedx(data)
        assert len(components) == 2
        assert components[0]["name"] == "requests"
        assert components[0]["ecosystem"] == "PyPI"
        assert components[0]["version"] == "2.28.0"
        assert components[1]["name"] == "lodash"
        assert components[1]["ecosystem"] == "npm"

    def test_component_without_purl(self):
        from manus_agent.tools.scan_sbom import _parse_cyclonedx

        data = _cyclonedx_sbom(
            [
                {"type": "library", "name": "some-lib", "version": "1.0.0"},
            ]
        )
        components = _parse_cyclonedx(data)
        assert len(components) == 1
        assert components[0]["name"] == "some-lib"
        assert components[0]["ecosystem"] == ""

    def test_skips_non_library_types(self):
        from manus_agent.tools.scan_sbom import _parse_cyclonedx

        data = _cyclonedx_sbom(
            [
                {"type": "application", "name": "my-app", "version": "1.0"},
                {"type": "library", "name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"},
            ]
        )
        components = _parse_cyclonedx(data)
        assert len(components) == 1
        assert components[0]["name"] == "requests"

    def test_framework_type_included(self):
        from manus_agent.tools.scan_sbom import _parse_cyclonedx

        data = _cyclonedx_sbom(
            [
                {"type": "framework", "name": "django", "version": "4.2", "purl": "pkg:pypi/django@4.2"},
            ]
        )
        components = _parse_cyclonedx(data)
        assert len(components) == 1
        assert components[0]["name"] == "django"

    def test_empty_components(self):
        from manus_agent.tools.scan_sbom import _parse_cyclonedx

        data = _cyclonedx_sbom([])
        assert _parse_cyclonedx(data) == []

    def test_non_dict_component_skipped(self):
        from manus_agent.tools.scan_sbom import _parse_cyclonedx

        data = _cyclonedx_sbom(["not-a-dict", {"type": "library", "name": "x", "version": "1"}])
        components = _parse_cyclonedx(data)
        assert len(components) == 1

    def test_purl_version_overrides_component_version(self):
        from manus_agent.tools.scan_sbom import _parse_cyclonedx

        data = _cyclonedx_sbom(
            [
                {"type": "library", "name": "foo", "version": "1.0.0", "purl": "pkg:pypi/foo@2.0.0"},
            ]
        )
        components = _parse_cyclonedx(data)
        assert components[0]["version"] == "2.0.0"

    def test_purl_without_version_falls_back(self):
        from manus_agent.tools.scan_sbom import _parse_cyclonedx

        data = _cyclonedx_sbom(
            [
                {"type": "library", "name": "foo", "version": "1.0.0", "purl": "pkg:pypi/foo"},
            ]
        )
        components = _parse_cyclonedx(data)
        assert components[0]["version"] == "1.0.0"


# ===================================================================
# 4. SPDX parsing
# ===================================================================
class TestParseSpdx:
    def test_basic_packages(self):
        from manus_agent.tools.scan_sbom import _parse_spdx

        data = _spdx_sbom()
        components = _parse_spdx(data)
        assert len(components) == 2
        assert components[0]["name"] == "requests"
        assert components[0]["ecosystem"] == "PyPI"
        assert components[1]["name"] == "express"

    def test_skips_document_package(self):
        from manus_agent.tools.scan_sbom import _parse_spdx

        data = _spdx_sbom(
            [
                {"SPDXID": "SPDXRef-DOCUMENT", "name": "root-doc", "versionInfo": "1.0"},
                {"SPDXID": "SPDXRef-Pkg", "name": "lib", "versionInfo": "2.0"},
            ]
        )
        components = _parse_spdx(data)
        assert len(components) == 1
        assert components[0]["name"] == "lib"

    def test_package_without_purl(self):
        from manus_agent.tools.scan_sbom import _parse_spdx

        data = _spdx_sbom(
            [
                {"SPDXID": "SPDXRef-Pkg", "name": "unknown-pkg", "versionInfo": "1.0"},
            ]
        )
        components = _parse_spdx(data)
        assert len(components) == 1
        assert components[0]["name"] == "unknown-pkg"
        assert components[0]["ecosystem"] == ""

    def test_empty_packages(self):
        from manus_agent.tools.scan_sbom import _parse_spdx

        data = _spdx_sbom([])
        assert _parse_spdx(data) == []

    def test_non_dict_package_skipped(self):
        from manus_agent.tools.scan_sbom import _parse_spdx

        data = _spdx_sbom(["not-a-dict"])
        assert _parse_spdx(data) == []

    def test_purl_version_overrides_versioninfo(self):
        from manus_agent.tools.scan_sbom import _parse_spdx

        data = _spdx_sbom(
            [
                {
                    "SPDXID": "SPDXRef-Pkg",
                    "name": "foo",
                    "versionInfo": "1.0.0",
                    "externalRefs": [{"referenceType": "purl", "referenceLocator": "pkg:pypi/foo@2.0.0"}],
                }
            ]
        )
        components = _parse_spdx(data)
        assert components[0]["version"] == "2.0.0"


# ===================================================================
# 5. parse_sbom dispatcher
# ===================================================================
class TestParseSbom:
    def test_cyclonedx(self):
        from manus_agent.tools.scan_sbom import parse_sbom

        fmt, comps = parse_sbom(_cyclonedx_sbom())
        assert fmt == "cyclonedx"
        assert len(comps) == 2

    def test_spdx(self):
        from manus_agent.tools.scan_sbom import parse_sbom

        fmt, comps = parse_sbom(_spdx_sbom())
        assert fmt == "spdx"
        assert len(comps) == 2

    def test_unknown(self):
        from manus_agent.tools.scan_sbom import parse_sbom

        fmt, comps = parse_sbom({"random": True})
        assert fmt == "unknown"
        assert comps == []


# ===================================================================
# 6. OSV batch query builder
# ===================================================================
class TestBuildOsvQueries:
    def test_basic_query(self):
        from manus_agent.tools.scan_sbom import _build_osv_queries

        comps = [{"name": "requests", "ecosystem": "PyPI", "version": "2.28.0"}]
        queries = _build_osv_queries(comps)
        assert len(queries) == 1
        assert queries[0]["package"]["name"] == "requests"
        assert queries[0]["package"]["ecosystem"] == "PyPI"
        assert queries[0]["version"] == "2.28.0"

    def test_query_without_ecosystem(self):
        from manus_agent.tools.scan_sbom import _build_osv_queries

        comps = [{"name": "foo", "ecosystem": "", "version": "1.0"}]
        queries = _build_osv_queries(comps)
        assert "ecosystem" not in queries[0]["package"]

    def test_query_without_version(self):
        from manus_agent.tools.scan_sbom import _build_osv_queries

        comps = [{"name": "foo", "ecosystem": "PyPI", "version": ""}]
        queries = _build_osv_queries(comps)
        assert "version" not in queries[0]


# ===================================================================
# 7. OSV batch query execution
# ===================================================================
class TestQueryOsvBatch:
    @patch("manus_agent.tools.scan_sbom._post_with_retry")
    def test_successful_batch(self, mock_post):
        from manus_agent.tools.scan_sbom import _query_osv_batch

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"vulns": [{"id": "GHSA-123", "aliases": ["CVE-2024-1234"]}]},
                {"vulns": []},
            ]
        }
        mock_post.return_value = mock_resp

        queries = [{"package": {"name": "a"}}, {"package": {"name": "b"}}]
        results = _query_osv_batch(queries)
        assert len(results) == 2
        assert len(results[0]) == 1
        assert results[1] == []

    @patch("manus_agent.tools.scan_sbom._post_with_retry")
    def test_api_error_returns_empty(self, mock_post):
        from manus_agent.tools.scan_sbom import _query_osv_batch

        mock_resp = Mock()
        mock_resp.status_code = 500
        mock_post.return_value = mock_resp

        results = _query_osv_batch([{"package": {"name": "a"}}])
        assert results == [[]]

    @patch("manus_agent.tools.scan_sbom._post_with_retry")
    def test_network_error_returns_empty(self, mock_post):
        import requests as req

        from manus_agent.tools.scan_sbom import _query_osv_batch

        mock_post.side_effect = req.ConnectionError("network fail")

        results = _query_osv_batch([{"package": {"name": "a"}}])
        assert results == [[]]

    @patch("manus_agent.tools.scan_sbom._post_with_retry")
    def test_padded_when_fewer_results(self, mock_post):
        from manus_agent.tools.scan_sbom import _query_osv_batch

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": [{"vulns": []}]}
        mock_post.return_value = mock_resp

        results = _query_osv_batch(
            [
                {"package": {"name": "a"}},
                {"package": {"name": "b"}},
            ]
        )
        assert len(results) == 2


# ===================================================================
# 8. EPSS enrichment
# ===================================================================
class TestFetchEpssScores:
    @patch("manus_agent.tools.scan_sbom._get_with_retry")
    def test_basic_scores(self, mock_get):
        from manus_agent.tools.scan_sbom import _fetch_epss_scores

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"cve": "CVE-2024-1234", "epss": "0.85"},
                {"cve": "CVE-2024-5678", "epss": "0.12"},
            ]
        }
        mock_get.return_value = mock_resp

        scores = _fetch_epss_scores(["CVE-2024-1234", "CVE-2024-5678"])
        assert scores["CVE-2024-1234"] == pytest.approx(0.85)
        assert scores["CVE-2024-5678"] == pytest.approx(0.12)

    @patch("manus_agent.tools.scan_sbom._get_with_retry")
    def test_empty_input(self, mock_get):
        from manus_agent.tools.scan_sbom import _fetch_epss_scores

        assert _fetch_epss_scores([]) == {}
        mock_get.assert_not_called()

    @patch("manus_agent.tools.scan_sbom._get_with_retry")
    def test_api_error_returns_empty(self, mock_get):
        from manus_agent.tools.scan_sbom import _fetch_epss_scores

        mock_resp = Mock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        assert _fetch_epss_scores(["CVE-2024-1234"]) == {}

    @patch("manus_agent.tools.scan_sbom._get_with_retry")
    def test_network_error_returns_empty(self, mock_get):
        import requests as req

        from manus_agent.tools.scan_sbom import _fetch_epss_scores

        mock_get.side_effect = req.ConnectionError("fail")

        assert _fetch_epss_scores(["CVE-2024-1234"]) == {}

    @patch("manus_agent.tools.scan_sbom._get_with_retry")
    def test_invalid_epss_value_skipped(self, mock_get):
        from manus_agent.tools.scan_sbom import _fetch_epss_scores

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"cve": "CVE-2024-1234", "epss": "not-a-number"},
                {"cve": "CVE-2024-5678", "epss": "0.5"},
            ]
        }
        mock_get.return_value = mock_resp

        scores = _fetch_epss_scores(["CVE-2024-1234", "CVE-2024-5678"])
        assert "CVE-2024-1234" not in scores
        assert scores["CVE-2024-5678"] == pytest.approx(0.5)


# ===================================================================
# 9. CISA KEV enrichment
# ===================================================================
class TestFetchKevSet:
    @patch("manus_agent.tools.scan_sbom._get_with_retry")
    def test_basic_kev(self, mock_get):
        from manus_agent.tools.scan_sbom import _fetch_kev_set

        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cveID": "CVE-2024-1234"},
                {"cveID": "CVE-2021-44228"},
            ]
        }
        mock_get.return_value = mock_resp

        kev = _fetch_kev_set()
        assert "CVE-2024-1234" in kev
        assert "CVE-2021-44228" in kev

    @patch("manus_agent.tools.scan_sbom._get_with_retry")
    def test_api_error_returns_empty(self, mock_get):
        from manus_agent.tools.scan_sbom import _fetch_kev_set

        mock_resp = Mock()
        mock_resp.status_code = 500
        mock_get.return_value = mock_resp

        assert _fetch_kev_set() == set()

    @patch("manus_agent.tools.scan_sbom._get_with_retry")
    def test_network_error_returns_empty(self, mock_get):
        import requests as req

        from manus_agent.tools.scan_sbom import _fetch_kev_set

        mock_get.side_effect = req.ConnectionError("fail")

        assert _fetch_kev_set() == set()


# ===================================================================
# 10. CVE extraction
# ===================================================================
class TestExtractCveIds:
    def test_cve_in_id(self):
        from manus_agent.tools.scan_sbom import _extract_cve_ids

        vulns = [{"id": "CVE-2024-1234", "aliases": []}]
        assert _extract_cve_ids(vulns) == ["CVE-2024-1234"]

    def test_cve_in_aliases(self):
        from manus_agent.tools.scan_sbom import _extract_cve_ids

        vulns = [{"id": "GHSA-xxx", "aliases": ["CVE-2024-5678"]}]
        assert _extract_cve_ids(vulns) == ["CVE-2024-5678"]

    def test_deduplication(self):
        from manus_agent.tools.scan_sbom import _extract_cve_ids

        vulns = [{"id": "CVE-2024-1234", "aliases": ["CVE-2024-1234"]}]
        assert _extract_cve_ids(vulns) == ["CVE-2024-1234"]

    def test_no_cves(self):
        from manus_agent.tools.scan_sbom import _extract_cve_ids

        vulns = [{"id": "GHSA-xxx", "aliases": ["GHSA-yyy"]}]
        assert _extract_cve_ids(vulns) == []


# ===================================================================
# 11. Severity label
# ===================================================================
class TestSeverityLabel:
    def test_critical(self):
        from manus_agent.tools.scan_sbom import _severity_label

        assert _severity_label(9.8) == "CRITICAL"

    def test_high(self):
        from manus_agent.tools.scan_sbom import _severity_label

        assert _severity_label(7.5) == "HIGH"

    def test_medium(self):
        from manus_agent.tools.scan_sbom import _severity_label

        assert _severity_label(5.0) == "MEDIUM"

    def test_low(self):
        from manus_agent.tools.scan_sbom import _severity_label

        assert _severity_label(2.0) == "LOW"

    def test_none(self):
        from manus_agent.tools.scan_sbom import _severity_label

        assert _severity_label(0.0) == "NONE"

    def test_unknown(self):
        from manus_agent.tools.scan_sbom import _severity_label

        assert _severity_label(None) == "UNKNOWN"


# ===================================================================
# 12. CVSS extraction from OSV record
# ===================================================================
class TestExtractCvss:
    def test_basic_score(self):
        from manus_agent.tools.scan_sbom import _extract_cvss

        vuln = {"severity": [{"type": "CVSS_V3", "score": "9.8"}]}
        assert _extract_cvss(vuln) == pytest.approx(9.8)

    def test_no_severity(self):
        from manus_agent.tools.scan_sbom import _extract_cvss

        assert _extract_cvss({}) is None

    def test_invalid_score(self):
        from manus_agent.tools.scan_sbom import _extract_cvss

        vuln = {"severity": [{"score": "not-a-number"}]}
        assert _extract_cvss(vuln) is None

    def test_picks_highest(self):
        from manus_agent.tools.scan_sbom import _extract_cvss

        vuln = {"severity": [{"score": "5.0"}, {"score": "9.0"}]}
        assert _extract_cvss(vuln) == pytest.approx(9.0)

    def test_non_list_severity(self):
        from manus_agent.tools.scan_sbom import _extract_cvss

        assert _extract_cvss({"severity": "not-a-list"}) is None


# ===================================================================
# 13. Full scan_sbom_file integration (mocked HTTP)
# ===================================================================
class TestScanSbomFile:
    def _write_sbom(self, tmp_path: Path, data: dict) -> str:
        p = tmp_path / "bom.json"
        p.write_text(json.dumps(data))
        return str(p)

    def test_file_not_found(self, tmp_path):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        result = scan_sbom_file(str(tmp_path / "nonexistent.json"))
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_invalid_json(self, tmp_path):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        p = tmp_path / "bad.json"
        p.write_text("{not valid json")
        result = scan_sbom_file(str(p))
        assert "error" in result

    def test_unknown_format(self, tmp_path):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        p = tmp_path / "random.json"
        p.write_text(json.dumps({"random": True}))
        result = scan_sbom_file(str(p))
        assert "error" in result
        assert "format" in result["error"].lower()

    def test_empty_components(self, tmp_path):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        path = self._write_sbom(tmp_path, _cyclonedx_sbom([]))
        result = scan_sbom_file(path)
        assert result["component_count"] == 0
        assert result["total_vulnerability_count"] == 0
        assert "no scannable components" in result["message"].lower()

    @patch("manus_agent.tools.scan_sbom._fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_no_vulns_found(self, mock_osv, mock_epss, mock_kev, tmp_path):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [[], []]
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        path = self._write_sbom(tmp_path, _cyclonedx_sbom())
        result = scan_sbom_file(path)
        assert result["total_vulnerability_count"] == 0
        assert "no known vulnerabilities" in result["message"].lower()

    @patch("manus_agent.tools.scan_sbom._fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_vulns_found_with_enrichment(self, mock_osv, mock_epss, mock_kev, tmp_path):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [
            [_osv_vuln("GHSA-aaa", "CVE-2024-1111", "Vuln A", 9.8)],
            [_osv_vuln("GHSA-bbb", "CVE-2024-2222", "Vuln B", 5.0)],
        ]
        mock_epss.return_value = {"CVE-2024-1111": 0.95, "CVE-2024-2222": 0.10}
        mock_kev.return_value = {"CVE-2024-1111"}

        path = self._write_sbom(tmp_path, _cyclonedx_sbom())
        result = scan_sbom_file(path)
        assert result["total_vulnerability_count"] == 2
        assert result["critical_count"] == 1
        assert result["medium_count"] == 1
        assert result["kev_count"] == 1
        assert result["vulnerable_component_count"] == 2

        # First finding should be KEV + highest EPSS
        first = result["findings"][0]
        assert first["vulnerability"]["in_kev"] is True
        assert first["vulnerability"]["epss_score"] == pytest.approx(0.95)

    @patch("manus_agent.tools.scan_sbom._fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_spdx_scan(self, mock_osv, mock_epss, mock_kev, tmp_path):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [
            [_osv_vuln("GHSA-ccc", "CVE-2024-3333", "Vuln C", 7.5)],
            [],
        ]
        mock_epss.return_value = {"CVE-2024-3333": 0.40}
        mock_kev.return_value = set()

        path = self._write_sbom(tmp_path, _spdx_sbom())
        result = scan_sbom_file(path)
        assert result["sbom_format"] == "spdx"
        assert result["total_vulnerability_count"] == 1
        assert result["high_count"] == 1
        assert result["kev_count"] == 0

    @patch("manus_agent.tools.scan_sbom._fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_sorting_kev_first_then_epss(self, mock_osv, mock_epss, mock_kev, tmp_path):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        # Component 1: not in KEV but high EPSS
        # Component 2: in KEV but lower EPSS
        mock_osv.return_value = [
            [_osv_vuln("GHSA-1", "CVE-2024-0001", "No KEV high EPSS", 9.0)],
            [_osv_vuln("GHSA-2", "CVE-2024-0002", "KEV low EPSS", 5.0)],
        ]
        mock_epss.return_value = {"CVE-2024-0001": 0.99, "CVE-2024-0002": 0.05}
        mock_kev.return_value = {"CVE-2024-0002"}

        path = self._write_sbom(tmp_path, _cyclonedx_sbom())
        result = scan_sbom_file(path)

        # KEV should come first regardless of EPSS
        assert result["findings"][0]["vulnerability"]["in_kev"] is True
        assert result["findings"][1]["vulnerability"]["in_kev"] is False

    @patch("manus_agent.tools.scan_sbom._fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_enrichment_failures_graceful(self, mock_osv, mock_epss, mock_kev, tmp_path):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [
            [_osv_vuln("GHSA-x", "CVE-2024-9999", "Test", 8.0)],
            [],
        ]
        # Simulate enrichment failures
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        path = self._write_sbom(tmp_path, _cyclonedx_sbom())
        result = scan_sbom_file(path)
        assert result["total_vulnerability_count"] == 1
        finding = result["findings"][0]
        assert finding["vulnerability"]["epss_score"] is None
        assert finding["vulnerability"]["in_kev"] is False

    @patch("manus_agent.tools.scan_sbom._fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_multiple_vulns_per_component(self, mock_osv, mock_epss, mock_kev, tmp_path):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [
            [
                _osv_vuln("GHSA-a1", "CVE-2024-0001", "Vuln 1", 9.8),
                _osv_vuln("GHSA-a2", "CVE-2024-0002", "Vuln 2", 5.0),
            ],
            [],
        ]
        mock_epss.return_value = {"CVE-2024-0001": 0.80, "CVE-2024-0002": 0.10}
        mock_kev.return_value = set()

        path = self._write_sbom(tmp_path, _cyclonedx_sbom())
        result = scan_sbom_file(path)
        assert result["total_vulnerability_count"] == 2
        assert result["vulnerable_component_count"] == 1


# ===================================================================
# 14. Strands tool entry point
# ===================================================================
class TestScanSbomToolEntryPoint:
    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_success(self, mock_scan):
        from manus_agent.tools.scan_sbom import scan_sbom

        mock_scan.return_value = {
            "sbom_format": "cyclonedx",
            "component_count": 5,
            "total_vulnerability_count": 1,
            "findings": [],
            "message": "ok",
        }

        tool_use = {"toolUseId": "test-1", "input": {"sbom_path": "/tmp/bom.json"}}
        result = scan_sbom(tool_use)
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-1"

    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_error(self, mock_scan):
        from manus_agent.tools.scan_sbom import scan_sbom

        mock_scan.return_value = {
            "error": "File not found",
            "findings": [],
            "message": "not found",
        }

        tool_use = {"toolUseId": "test-2", "input": {"sbom_path": "/tmp/missing.json"}}
        result = scan_sbom(tool_use)
        assert result["status"] == "error"

    def test_missing_path(self):
        from manus_agent.tools.scan_sbom import scan_sbom

        tool_use = {"toolUseId": "test-3", "input": {}}
        result = scan_sbom(tool_use)
        assert result["status"] == "error"
        assert "invalid" in result["content"][0]["text"].lower()

    def test_empty_path(self):
        from manus_agent.tools.scan_sbom import scan_sbom

        tool_use = {"toolUseId": "test-4", "input": {"sbom_path": "  "}}
        result = scan_sbom(tool_use)
        assert result["status"] == "error"


# ===================================================================
# 15. TOOL_SPEC validation
# ===================================================================
class TestToolSpec:
    def test_spec_name(self):
        from manus_agent.tools.scan_sbom import TOOL_SPEC

        assert TOOL_SPEC["name"] == "scan_sbom"

    def test_spec_has_description(self):
        from manus_agent.tools.scan_sbom import TOOL_SPEC

        assert len(TOOL_SPEC["description"]) > 20

    def test_spec_has_input_schema(self):
        from manus_agent.tools.scan_sbom import TOOL_SPEC

        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "sbom_path" in schema["properties"]
        assert "sbom_path" in schema["required"]


# ===================================================================
# 16. HTTP retry helpers
# ===================================================================
class TestGetWithRetry:
    @patch("manus_agent.tools.scan_sbom.requests.get")
    @patch("manus_agent.tools.scan_sbom.time.sleep")
    def test_retries_on_429(self, mock_sleep, mock_get):
        from manus_agent.tools.scan_sbom import _get_with_retry

        resp_429 = Mock()
        resp_429.status_code = 429
        resp_200 = Mock()
        resp_200.status_code = 200
        mock_get.side_effect = [resp_429, resp_200]

        result = _get_with_retry("http://example.com", max_retries=2, base_delay=0.0)
        assert result.status_code == 200

    @patch("manus_agent.tools.scan_sbom.requests.get")
    @patch("manus_agent.tools.scan_sbom.time.sleep")
    def test_retries_on_connection_error(self, mock_sleep, mock_get):
        import requests as req

        from manus_agent.tools.scan_sbom import _get_with_retry

        resp_200 = Mock()
        resp_200.status_code = 200
        mock_get.side_effect = [req.ConnectionError("fail"), resp_200]

        result = _get_with_retry("http://example.com", max_retries=2, base_delay=0.0)
        assert result.status_code == 200

    @patch("manus_agent.tools.scan_sbom.requests.get")
    @patch("manus_agent.tools.scan_sbom.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep, mock_get):
        import requests as req

        from manus_agent.tools.scan_sbom import _get_with_retry

        mock_get.side_effect = req.ConnectionError("fail")

        with pytest.raises(req.ConnectionError):
            _get_with_retry("http://example.com", max_retries=2, base_delay=0.0)

    @patch("manus_agent.tools.scan_sbom.requests.get")
    def test_non_retryable_4xx_returned_immediately(self, mock_get):
        from manus_agent.tools.scan_sbom import _get_with_retry

        resp_403 = Mock()
        resp_403.status_code = 403
        mock_get.return_value = resp_403

        result = _get_with_retry("http://example.com", max_retries=3, base_delay=0.0)
        assert result.status_code == 403
        assert mock_get.call_count == 1


class TestPostWithRetry:
    @patch("manus_agent.tools.scan_sbom.requests.post")
    @patch("manus_agent.tools.scan_sbom.time.sleep")
    def test_retries_on_500(self, mock_sleep, mock_post):
        from manus_agent.tools.scan_sbom import _post_with_retry

        resp_500 = Mock()
        resp_500.status_code = 500
        resp_200 = Mock()
        resp_200.status_code = 200
        mock_post.side_effect = [resp_500, resp_200]

        result = _post_with_retry("http://example.com", max_retries=2, base_delay=0.0)
        assert result.status_code == 200

    @patch("manus_agent.tools.scan_sbom.requests.post")
    @patch("manus_agent.tools.scan_sbom.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep, mock_post):
        import requests as req

        from manus_agent.tools.scan_sbom import _post_with_retry

        mock_post.side_effect = req.Timeout("timeout")

        with pytest.raises(req.Timeout):
            _post_with_retry("http://example.com", max_retries=2, base_delay=0.0)


# ===================================================================
# 17. CLI parser
# ===================================================================
class TestSbomScanParser:
    def test_parser_basic(self):
        from manus_agent.cli import _build_sbom_scan_parser

        parser = _build_sbom_scan_parser()
        args = parser.parse_args(["bom.json"])
        assert args.sbom_file == "bom.json"
        assert args.output == "text"

    def test_parser_json_output(self):
        from manus_agent.cli import _build_sbom_scan_parser

        parser = _build_sbom_scan_parser()
        args = parser.parse_args(["bom.json", "--output", "json"])
        assert args.output == "json"


# ===================================================================
# 18. CLI runner — text output
# ===================================================================
class TestRunSbomScanText:
    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_text_output_no_vulns(self, mock_scan, capsys):
        from manus_agent.cli import _run_sbom_scan

        mock_scan.return_value = {
            "sbom_format": "cyclonedx",
            "component_count": 10,
            "vulnerable_component_count": 0,
            "total_vulnerability_count": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "unknown_count": 0,
            "kev_count": 0,
            "findings": [],
            "message": "No vulns",
        }

        exit_code = _run_sbom_scan(["bom.json"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "CYCLONEDX" in out
        assert "10" in out

    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_text_output_with_vulns(self, mock_scan, capsys):
        from manus_agent.cli import _run_sbom_scan

        mock_scan.return_value = {
            "sbom_format": "cyclonedx",
            "component_count": 5,
            "vulnerable_component_count": 1,
            "total_vulnerability_count": 1,
            "critical_count": 1,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "unknown_count": 0,
            "kev_count": 1,
            "findings": [
                {
                    "component": {"name": "requests", "version": "2.28.0", "ecosystem": "PyPI"},
                    "vulnerability": {
                        "id": "GHSA-aaa",
                        "cve_ids": ["CVE-2024-1234"],
                        "summary": "Bad thing",
                        "cvss_score": 9.8,
                        "severity": "CRITICAL",
                        "epss_score": 0.95,
                        "in_kev": True,
                    },
                }
            ],
            "message": "Found 1 vuln",
        }

        exit_code = _run_sbom_scan(["bom.json"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "requests@2.28.0" in out
        assert "CRITICAL" in out
        assert "CVE-2024-1234" in out
        assert "KEV" in out

    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_text_output_error(self, mock_scan, capsys):
        from manus_agent.cli import _run_sbom_scan

        mock_scan.return_value = {
            "error": "File not found: missing.json",
            "findings": [],
            "message": "not found",
        }

        exit_code = _run_sbom_scan(["missing.json"])
        assert exit_code == 1
        err = capsys.readouterr().err
        assert "Error" in err


# ===================================================================
# 19. CLI runner — JSON output
# ===================================================================
class TestRunSbomScanJson:
    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_json_output(self, mock_scan, capsys):
        from manus_agent.cli import _run_sbom_scan

        expected = {
            "sbom_format": "cyclonedx",
            "component_count": 2,
            "vulnerable_component_count": 1,
            "total_vulnerability_count": 1,
            "critical_count": 0,
            "high_count": 1,
            "medium_count": 0,
            "low_count": 0,
            "unknown_count": 0,
            "kev_count": 0,
            "findings": [
                {
                    "component": {"name": "lodash", "version": "4.17.20", "ecosystem": "npm"},
                    "vulnerability": {
                        "id": "GHSA-bbb",
                        "cve_ids": ["CVE-2024-5678"],
                        "summary": "Prototype pollution",
                        "cvss_score": 7.5,
                        "severity": "HIGH",
                        "epss_score": 0.30,
                        "in_kev": False,
                    },
                }
            ],
            "message": "Found 1 vuln",
        }
        mock_scan.return_value = expected

        exit_code = _run_sbom_scan(["bom.json", "--output", "json"])
        assert exit_code == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["sbom_format"] == "cyclonedx"
        assert parsed["total_vulnerability_count"] == 1


# ===================================================================
# 20. CLI dispatch integration
# ===================================================================
class TestSbomScanDispatch:
    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_dispatch_sbom_scan(self, mock_scan):
        from manus_agent.cli import main

        mock_scan.return_value = {
            "sbom_format": "cyclonedx",
            "component_count": 0,
            "vulnerable_component_count": 0,
            "total_vulnerability_count": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "unknown_count": 0,
            "kev_count": 0,
            "findings": [],
            "message": "No scannable components",
        }

        with patch("sys.argv", ["manus-agent", "sbom-scan", "bom.json"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


# ===================================================================
# 21. Edge cases
# ===================================================================
class TestEdgeCases:
    def test_osv_vuln_without_aliases_key(self):
        from manus_agent.tools.scan_sbom import _extract_cve_ids

        vulns = [{"id": "CVE-2024-1234"}]
        assert _extract_cve_ids(vulns) == ["CVE-2024-1234"]

    def test_severity_label_boundary_9_0(self):
        from manus_agent.tools.scan_sbom import _severity_label

        assert _severity_label(9.0) == "CRITICAL"

    def test_severity_label_boundary_7_0(self):
        from manus_agent.tools.scan_sbom import _severity_label

        assert _severity_label(7.0) == "HIGH"

    def test_severity_label_boundary_4_0(self):
        from manus_agent.tools.scan_sbom import _severity_label

        assert _severity_label(4.0) == "MEDIUM"

    @patch("manus_agent.tools.scan_sbom._fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom._query_osv_batch")
    def test_vuln_without_cvss_gets_unknown_severity(self, mock_osv, mock_epss, mock_kev, tmp_path):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [
            [{"id": "GHSA-no-cvss", "aliases": ["CVE-2024-0000"], "summary": "Test"}],
            [],
        ]
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        path = tmp_path / "bom.json"
        path.write_text(json.dumps(_cyclonedx_sbom()))

        result = scan_sbom_file(str(path))
        assert result["unknown_count"] == 1
        assert result["findings"][0]["vulnerability"]["severity"] == "UNKNOWN"

    def test_npm_scoped_package_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:npm/@types/node@20.0.0")
        assert result["name"] == "@types/node"
        assert result["version"] == "20.0.0"
        assert result["ecosystem"] == "npm"

    def test_gem_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:gem/rails@7.0.0")
        assert result["ecosystem"] == "RubyGems"
        assert result["name"] == "rails"

    def test_composer_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:composer/laravel/framework@10.0.0")
        assert result["ecosystem"] == "Packagist"
        assert result["name"] == "framework"

    def test_hex_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:hex/phoenix@1.7.0")
        assert result["ecosystem"] == "Hex"
        assert result["name"] == "phoenix"
