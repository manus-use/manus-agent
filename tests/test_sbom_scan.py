"""Comprehensive test suite for scan_sbom (SBOM vulnerability scanner).

100 % mocked — no real HTTP calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Disable retry delays in tests
# ---------------------------------------------------------------------------
os.environ.setdefault("SBOM_SCAN_MAX_RETRIES", "1")
os.environ.setdefault("SBOM_SCAN_RETRY_BASE_DELAY", "0")

from manus_agent.tools.scan_sbom import (
    _PURL_TYPE_TO_ECOSYSTEM,
    _extract_cve_ids,
    _extract_severity,
    _parse_cyclonedx,
    _parse_purl,
    _parse_spdx,
    _score_to_label,
    assemble_findings,
    compute_stats,
    fetch_epss_batch,
    fetch_kev_set,
    handler,
    parse_sbom,
    query_osv_batch,
    scan_sbom,
)

# ===================================================================
# Fixtures — reusable SBOM documents
# ===================================================================


def _cdx_sbom(components: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build a minimal CycloneDX SBOM."""
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "components": components or [],
    }


def _spdx_sbom(packages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build a minimal SPDX 2.3 SBOM."""
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "test",
        "packages": packages or [],
    }


def _osv_vuln(
    vuln_id: str = "GHSA-xxxx-yyyy-zzzz",
    aliases: list[str] | None = None,
    summary: str = "Test vulnerability",
    severity: list[dict[str, str]] | None = None,
    database_specific: dict[str, Any] | None = None,
    affected: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal OSV vulnerability record."""
    v: dict[str, Any] = {"id": vuln_id, "summary": summary}
    if aliases is not None:
        v["aliases"] = aliases
    if severity is not None:
        v["severity"] = severity
    if database_specific is not None:
        v["database_specific"] = database_specific
    if affected is not None:
        v["affected"] = affected
    return v


@pytest.fixture()
def tmp_cdx_sbom(tmp_path: Path) -> Path:
    """Write a CycloneDX SBOM to a temp file and return its path."""
    doc = _cdx_sbom(
        [
            {
                "type": "library",
                "name": "requests",
                "version": "2.28.0",
                "purl": "pkg:pypi/requests@2.28.0",
            },
            {
                "type": "library",
                "name": "flask",
                "version": "2.3.2",
                "purl": "pkg:pypi/flask@2.3.2",
            },
        ]
    )
    p = tmp_path / "bom.json"
    p.write_text(json.dumps(doc))
    return p


@pytest.fixture()
def tmp_spdx_sbom(tmp_path: Path) -> Path:
    """Write an SPDX SBOM to a temp file and return its path."""
    doc = _spdx_sbom(
        [
            {
                "SPDXID": "SPDXRef-Package-requests",
                "name": "requests",
                "versionInfo": "2.28.0",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": "pkg:pypi/requests@2.28.0",
                    }
                ],
            },
        ]
    )
    p = tmp_path / "sbom.spdx.json"
    p.write_text(json.dumps(doc))
    return p


# ===================================================================
# purl parsing tests
# ===================================================================


class TestParsePurl:
    """Tests for _parse_purl."""

    def test_pypi_simple(self):
        eco, name, ver = _parse_purl("pkg:pypi/requests@2.28.0")
        assert eco == "PyPI"
        assert name == "requests"
        assert ver == "2.28.0"

    def test_npm_scoped(self):
        eco, name, ver = _parse_purl("pkg:npm/%40angular/core@16.0.0")
        assert eco == "npm"
        assert name == "@angular/core"
        assert ver == "16.0.0"

    def test_maven_with_namespace(self):
        eco, name, ver = _parse_purl("pkg:maven/org.apache.logging.log4j/log4j-core@2.17.0")
        assert eco == "Maven"
        assert name == "org.apache.logging.log4j:log4j-core"
        assert ver == "2.17.0"

    def test_golang(self):
        eco, name, ver = _parse_purl("pkg:golang/github.com/gin-gonic/gin@1.9.0")
        assert eco == "Go"
        assert name == "github.com/gin-gonic/gin"
        assert ver == "1.9.0"

    def test_cargo(self):
        eco, name, ver = _parse_purl("pkg:cargo/serde@1.0.193")
        assert eco == "crates.io"
        assert name == "serde"
        assert ver == "1.0.193"

    def test_nuget(self):
        eco, name, ver = _parse_purl("pkg:nuget/Newtonsoft.Json@13.0.3")
        assert eco == "NuGet"
        assert name == "Newtonsoft.Json"
        assert ver == "13.0.3"

    def test_gem(self):
        eco, name, ver = _parse_purl("pkg:gem/rails@7.1.0")
        assert eco == "RubyGems"
        assert name == "rails"
        assert ver == "7.1.0"

    def test_composer(self):
        eco, name, ver = _parse_purl("pkg:composer/laravel/framework@10.0.0")
        assert eco == "Packagist"
        assert name == "framework"
        assert ver == "10.0.0"

    def test_with_qualifiers_stripped(self):
        eco, name, ver = _parse_purl("pkg:pypi/django@4.2.0?os=linux#subpath")
        assert eco == "PyPI"
        assert name == "django"
        assert ver == "4.2.0"

    def test_no_version_returns_empty(self):
        eco, name, ver = _parse_purl("pkg:pypi/requests")
        assert eco == ""
        assert name == ""
        assert ver == ""

    def test_invalid_prefix(self):
        eco, name, ver = _parse_purl("notapurl://foo@1.0")
        assert eco == ""

    def test_no_slash(self):
        eco, name, ver = _parse_purl("pkg:pypi")
        assert eco == ""

    def test_unknown_type(self):
        eco, name, ver = _parse_purl("pkg:unknown/foo@1.0")
        assert eco == ""
        assert name == "foo"
        assert ver == "1.0"

    def test_deb(self):
        eco, name, ver = _parse_purl("pkg:deb/debian/openssl@3.0.11-1")
        assert eco == "Debian"
        assert name == "openssl"
        assert ver == "3.0.11-1"

    def test_apk(self):
        eco, name, ver = _parse_purl("pkg:apk/alpine/musl@1.2.4-r0")
        assert eco == "Alpine"
        assert name == "musl"
        assert ver == "1.2.4-r0"

    def test_percent_encoded_namespace(self):
        eco, name, ver = _parse_purl("pkg:npm/%40types%2Fnode@20.0.0")
        assert eco == "npm"
        assert name == "@types/node"
        assert ver == "20.0.0"


# ===================================================================
# CycloneDX parsing tests
# ===================================================================


class TestParseCyclonedx:
    """Tests for _parse_cyclonedx."""

    def test_basic_components(self):
        doc = _cdx_sbom(
            [
                {"name": "foo", "version": "1.0", "purl": "pkg:pypi/foo@1.0"},
                {"name": "bar", "version": "2.0", "purl": "pkg:npm/bar@2.0"},
            ]
        )
        comps = _parse_cyclonedx(doc)
        assert len(comps) == 2
        assert comps[0]["ecosystem"] == "PyPI"
        assert comps[0]["name"] == "foo"
        assert comps[0]["version"] == "1.0"
        assert comps[1]["ecosystem"] == "npm"

    def test_component_without_purl(self):
        doc = _cdx_sbom(
            [
                {"name": "mystery", "version": "1.0", "type": "library"},
            ]
        )
        comps = _parse_cyclonedx(doc)
        assert len(comps) == 1
        assert comps[0]["name"] == "mystery"
        # Ecosystem falls back to type hint.
        assert comps[0]["ecosystem"] == "PyPI"

    def test_component_without_version_skipped(self):
        doc = _cdx_sbom(
            [
                {"name": "noversionlib", "purl": "pkg:pypi/noversionlib"},
            ]
        )
        comps = _parse_cyclonedx(doc)
        assert len(comps) == 0

    def test_empty_components(self):
        doc = _cdx_sbom([])
        comps = _parse_cyclonedx(doc)
        assert comps == []

    def test_purl_overrides_name(self):
        doc = _cdx_sbom(
            [
                {"name": "display-name", "version": "1.0", "purl": "pkg:pypi/real-name@1.0"},
            ]
        )
        comps = _parse_cyclonedx(doc)
        assert comps[0]["name"] == "real-name"

    def test_purl_overrides_version(self):
        doc = _cdx_sbom(
            [
                {"name": "foo", "version": "0.0.0", "purl": "pkg:pypi/foo@1.2.3"},
            ]
        )
        comps = _parse_cyclonedx(doc)
        assert comps[0]["version"] == "1.2.3"


# ===================================================================
# SPDX parsing tests
# ===================================================================


class TestParseSpdx:
    """Tests for _parse_spdx."""

    def test_basic_packages(self):
        doc = _spdx_sbom(
            [
                {
                    "SPDXID": "SPDXRef-pkg1",
                    "name": "flask",
                    "versionInfo": "2.3.2",
                    "externalRefs": [
                        {
                            "referenceCategory": "PACKAGE-MANAGER",
                            "referenceType": "purl",
                            "referenceLocator": "pkg:pypi/flask@2.3.2",
                        }
                    ],
                },
            ]
        )
        comps = _parse_spdx(doc)
        assert len(comps) == 1
        assert comps[0]["ecosystem"] == "PyPI"
        assert comps[0]["name"] == "flask"
        assert comps[0]["version"] == "2.3.2"

    def test_package_without_purl(self):
        doc = _spdx_sbom(
            [
                {"SPDXID": "SPDXRef-pkg1", "name": "nopurl", "versionInfo": "3.0"},
            ]
        )
        comps = _parse_spdx(doc)
        assert len(comps) == 1
        assert comps[0]["name"] == "nopurl"
        assert comps[0]["version"] == "3.0"
        assert comps[0]["ecosystem"] == ""

    def test_package_without_version_skipped(self):
        doc = _spdx_sbom(
            [
                {"SPDXID": "SPDXRef-pkg1", "name": "noversion"},
            ]
        )
        comps = _parse_spdx(doc)
        assert comps == []

    def test_empty_packages(self):
        doc = _spdx_sbom([])
        comps = _parse_spdx(doc)
        assert comps == []

    def test_purl_ref_ignored_if_not_purl_type(self):
        doc = _spdx_sbom(
            [
                {
                    "SPDXID": "SPDXRef-pkg1",
                    "name": "foo",
                    "versionInfo": "1.0",
                    "externalRefs": [
                        {
                            "referenceCategory": "SECURITY",
                            "referenceType": "cpe23Type",
                            "referenceLocator": "cpe:2.3:a:foo:foo:1.0",
                        }
                    ],
                },
            ]
        )
        comps = _parse_spdx(doc)
        assert comps[0]["ecosystem"] == ""
        assert comps[0]["purl"] == ""


# ===================================================================
# parse_sbom tests
# ===================================================================


class TestParseSbom:
    """Tests for parse_sbom (file-level)."""

    def test_cyclonedx_detection(self, tmp_cdx_sbom: Path):
        fmt, comps = parse_sbom(str(tmp_cdx_sbom))
        assert fmt == "CycloneDX"
        assert len(comps) == 2

    def test_spdx_detection(self, tmp_spdx_sbom: Path):
        fmt, comps = parse_sbom(str(tmp_spdx_sbom))
        assert fmt == "SPDX"
        assert len(comps) == 1

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not found"):
            parse_sbom(str(tmp_path / "nonexistent.json"))

    def test_invalid_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("not json {{{")
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_sbom(str(p))

    def test_unknown_format(self, tmp_path: Path):
        p = tmp_path / "unknown.json"
        p.write_text(json.dumps({"random": "data"}))
        with pytest.raises(ValueError, match="Unrecognised"):
            parse_sbom(str(p))

    def test_non_dict_root(self, tmp_path: Path):
        p = tmp_path / "array.json"
        p.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="JSON object"):
            parse_sbom(str(p))


# ===================================================================
# OSV batch query tests
# ===================================================================


class TestQueryOsvBatch:
    """Tests for query_osv_batch."""

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_basic_query(self, mock_req):
        vulns = [_osv_vuln("GHSA-1111", aliases=["CVE-2024-1111"])]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"vulns": vulns},
                {"vulns": []},
            ]
        }
        mock_req.return_value = mock_resp

        comps = [
            {"name": "requests", "version": "2.28.0", "ecosystem": "PyPI"},
            {"name": "flask", "version": "2.3.2", "ecosystem": "PyPI"},
        ]
        result = query_osv_batch(comps)
        assert 0 in result
        assert len(result[0]) == 1
        assert 1 not in result  # no vulns for flask

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_non_200_skipped(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_req.return_value = mock_resp
        result = query_osv_batch([{"name": "a", "version": "1", "ecosystem": "PyPI"}])
        assert result == {}

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_request_exception(self, mock_req):
        import requests as req_lib

        mock_req.side_effect = req_lib.RequestException("timeout")
        result = query_osv_batch([{"name": "a", "version": "1", "ecosystem": "PyPI"}])
        assert result == {}

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_json_decode_error(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("bad json")
        mock_req.return_value = mock_resp
        result = query_osv_batch([{"name": "a", "version": "1", "ecosystem": "PyPI"}])
        assert result == {}

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_empty_components(self, mock_req):
        result = query_osv_batch([])
        assert result == {}
        mock_req.assert_not_called()

    @patch("manus_agent.tools.scan_sbom._OSV_BATCH_CHUNK", 2)
    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_chunking(self, mock_req):
        """Verify large component lists are chunked correctly."""
        vuln_a = _osv_vuln("GHSA-AAAA")
        vuln_c = _osv_vuln("GHSA-CCCC")
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"results": [{"vulns": [vuln_a]}, {"vulns": []}]}
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"results": [{"vulns": [vuln_c]}]}
        mock_req.side_effect = [resp1, resp2]

        comps = [
            {"name": "a", "version": "1", "ecosystem": "PyPI"},
            {"name": "b", "version": "1", "ecosystem": "PyPI"},
            {"name": "c", "version": "1", "ecosystem": "PyPI"},
        ]
        result = query_osv_batch(comps)
        assert 0 in result  # first chunk, index 0
        assert 1 not in result
        assert 2 in result  # second chunk, global index 2


# ===================================================================
# EPSS batch lookup tests
# ===================================================================


class TestFetchEpssBatch:
    """Tests for fetch_epss_batch."""

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_basic_batch(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"cve": "CVE-2024-1111", "epss": "0.9876"},
                {"cve": "CVE-2024-2222", "epss": "0.1234"},
            ]
        }
        mock_req.return_value = mock_resp
        scores = fetch_epss_batch(["CVE-2024-1111", "CVE-2024-2222"])
        assert scores["CVE-2024-1111"] == pytest.approx(0.9876)
        assert scores["CVE-2024-2222"] == pytest.approx(0.1234)

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_non_200(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_req.return_value = mock_resp
        scores = fetch_epss_batch(["CVE-2024-1111"])
        assert scores == {}

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_request_exception(self, mock_req):
        import requests as req_lib

        mock_req.side_effect = req_lib.RequestException("timeout")
        scores = fetch_epss_batch(["CVE-2024-1111"])
        assert scores == {}

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_invalid_epss_value(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"cve": "CVE-2024-1111", "epss": "not-a-number"}]}
        mock_req.return_value = mock_resp
        scores = fetch_epss_batch(["CVE-2024-1111"])
        assert "CVE-2024-1111" not in scores

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_empty_list(self, mock_req):
        scores = fetch_epss_batch([])
        assert scores == {}
        mock_req.assert_not_called()

    @patch("manus_agent.tools.scan_sbom._EPSS_BATCH_SIZE", 2)
    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_batch_chunking(self, mock_req):
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {
            "data": [
                {"cve": "CVE-2024-0001", "epss": "0.5"},
                {"cve": "CVE-2024-0002", "epss": "0.3"},
            ]
        }
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"data": [{"cve": "CVE-2024-0003", "epss": "0.1"}]}
        mock_req.side_effect = [resp1, resp2]
        scores = fetch_epss_batch(["CVE-2024-0001", "CVE-2024-0002", "CVE-2024-0003"])
        assert len(scores) == 3
        assert mock_req.call_count == 2

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_missing_cve_field(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"epss": "0.5"}]  # no "cve" key
        }
        mock_req.return_value = mock_resp
        scores = fetch_epss_batch(["CVE-2024-1111"])
        assert scores == {}

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_none_epss_value(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"cve": "CVE-2024-1111", "epss": None}]}
        mock_req.return_value = mock_resp
        scores = fetch_epss_batch(["CVE-2024-1111"])
        assert "CVE-2024-1111" not in scores


# ===================================================================
# CISA KEV lookup tests
# ===================================================================


class TestFetchKevSet:
    """Tests for fetch_kev_set."""

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_basic_fetch(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cveID": "CVE-2024-1111"},
                {"cveID": "CVE-2024-2222"},
            ]
        }
        mock_req.return_value = mock_resp
        kev = fetch_kev_set()
        assert "CVE-2024-1111" in kev
        assert "CVE-2024-2222" in kev

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_non_200(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_req.return_value = mock_resp
        assert fetch_kev_set() == set()

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_request_exception(self, mock_req):
        import requests as req_lib

        mock_req.side_effect = req_lib.RequestException("nope")
        assert fetch_kev_set() == set()

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_case_normalisation(self, mock_req):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"vulnerabilities": [{"cveID": "cve-2024-1111"}]}
        mock_req.return_value = mock_resp
        kev = fetch_kev_set()
        assert "CVE-2024-1111" in kev


# ===================================================================
# Severity extraction tests
# ===================================================================


class TestExtractSeverity:
    """Tests for _extract_severity and _score_to_label."""

    def test_score_to_label_critical(self):
        assert _score_to_label(9.8) == "CRITICAL"

    def test_score_to_label_high(self):
        assert _score_to_label(7.5) == "HIGH"

    def test_score_to_label_medium(self):
        assert _score_to_label(5.0) == "MEDIUM"

    def test_score_to_label_low(self):
        assert _score_to_label(2.0) == "LOW"

    def test_score_to_label_zero(self):
        assert _score_to_label(0.0) == "UNKNOWN"

    def test_unknown_when_no_severity(self):
        vuln = _osv_vuln()
        label, score = _extract_severity(vuln)
        assert label == "UNKNOWN"
        assert score == 0.0

    def test_database_specific_severity_fallback(self):
        vuln = _osv_vuln(database_specific={"severity": "HIGH"})
        label, score = _extract_severity(vuln)
        assert label == "HIGH"
        assert score == 7.5

    def test_affected_severity_upgrade(self):
        vuln = _osv_vuln(
            database_specific={"severity": "MEDIUM"},
            affected=[{"database_specific": {"severity": "CRITICAL"}}],
        )
        label, score = _extract_severity(vuln)
        assert label == "CRITICAL"
        assert score == 9.5

    def test_affected_severity_no_downgrade(self):
        vuln = _osv_vuln(
            database_specific={"severity": "CRITICAL"},
            affected=[{"database_specific": {"severity": "LOW"}}],
        )
        label, score = _extract_severity(vuln)
        assert label == "CRITICAL"

    def test_non_string_severity_ignored(self):
        vuln = _osv_vuln(database_specific={"severity": 42})
        label, _ = _extract_severity(vuln)
        assert label == "UNKNOWN"


# ===================================================================
# CVE ID extraction tests
# ===================================================================


class TestExtractCveIds:
    """Tests for _extract_cve_ids."""

    def test_from_aliases(self):
        vuln = _osv_vuln(aliases=["CVE-2024-1111", "GHSA-xxxx"])
        cves = _extract_cve_ids(vuln)
        assert cves == ["CVE-2024-1111"]

    def test_from_vuln_id(self):
        vuln = _osv_vuln(vuln_id="CVE-2024-5555")
        cves = _extract_cve_ids(vuln)
        assert "CVE-2024-5555" in cves

    def test_deduplication(self):
        vuln = _osv_vuln(vuln_id="CVE-2024-1111", aliases=["CVE-2024-1111"])
        cves = _extract_cve_ids(vuln)
        assert cves.count("CVE-2024-1111") == 1

    def test_no_cves(self):
        vuln = _osv_vuln(vuln_id="GHSA-xxxx", aliases=["GHSA-yyyy"])
        cves = _extract_cve_ids(vuln)
        assert cves == []

    def test_case_normalisation(self):
        vuln = _osv_vuln(aliases=["cve-2024-1111"])
        cves = _extract_cve_ids(vuln)
        assert cves == ["CVE-2024-1111"]

    def test_multiple_cves(self):
        vuln = _osv_vuln(aliases=["CVE-2024-1111", "CVE-2024-2222"])
        cves = _extract_cve_ids(vuln)
        assert len(cves) == 2


# ===================================================================
# Finding assembly tests
# ===================================================================


class TestAssembleFindings:
    """Tests for assemble_findings."""

    def test_basic_assembly(self):
        comps = [{"name": "requests", "version": "2.28.0", "ecosystem": "PyPI"}]
        osv_results = {0: [_osv_vuln("GHSA-1111", aliases=["CVE-2024-1111"])]}
        epss = {"CVE-2024-1111": 0.5}
        kev = {"CVE-2024-9999"}
        findings = assemble_findings(comps, osv_results, epss, kev)
        assert len(findings) == 1
        assert findings[0]["vuln_id"] == "GHSA-1111"
        assert findings[0]["epss"] == 0.5
        assert findings[0]["in_kev"] is False

    def test_kev_finding(self):
        comps = [{"name": "foo", "version": "1.0", "ecosystem": "npm"}]
        osv_results = {0: [_osv_vuln("GHSA-2222", aliases=["CVE-2024-2222"])]}
        epss = {"CVE-2024-2222": 0.3}
        kev = {"CVE-2024-2222"}
        findings = assemble_findings(comps, osv_results, epss, kev)
        assert findings[0]["in_kev"] is True

    def test_ranking_kev_first(self):
        comps = [
            {"name": "a", "version": "1.0", "ecosystem": "PyPI"},
            {"name": "b", "version": "1.0", "ecosystem": "npm"},
        ]
        osv_results = {
            0: [_osv_vuln("V-HIGH-EPSS", aliases=["CVE-2024-1111"])],
            1: [_osv_vuln("V-IN-KEV", aliases=["CVE-2024-2222"])],
        }
        epss = {"CVE-2024-1111": 0.99, "CVE-2024-2222": 0.01}
        kev = {"CVE-2024-2222"}
        findings = assemble_findings(comps, osv_results, epss, kev)
        assert findings[0]["vuln_id"] == "V-IN-KEV"  # KEV trumps EPSS

    def test_ranking_epss_within_non_kev(self):
        comps = [
            {"name": "a", "version": "1.0", "ecosystem": "PyPI"},
            {"name": "b", "version": "1.0", "ecosystem": "PyPI"},
        ]
        osv_results = {
            0: [_osv_vuln("V-LOW", aliases=["CVE-2024-1111"])],
            1: [_osv_vuln("V-HIGH", aliases=["CVE-2024-2222"])],
        }
        epss = {"CVE-2024-1111": 0.1, "CVE-2024-2222": 0.9}
        findings = assemble_findings(comps, osv_results, epss, set())
        assert findings[0]["vuln_id"] == "V-HIGH"

    def test_deduplication(self):
        comps = [{"name": "foo", "version": "1.0", "ecosystem": "PyPI"}]
        same_vuln = _osv_vuln("GHSA-DUP")
        osv_results = {0: [same_vuln, same_vuln]}
        findings = assemble_findings(comps, osv_results, {}, set())
        assert len(findings) == 1

    def test_empty_results(self):
        findings = assemble_findings([], {}, {}, set())
        assert findings == []

    def test_summary_truncation(self):
        long_summary = "A" * 300
        comps = [{"name": "x", "version": "1.0", "ecosystem": "PyPI"}]
        osv_results = {0: [_osv_vuln("V-1", summary=long_summary)]}
        findings = assemble_findings(comps, osv_results, {}, set())
        assert len(findings[0]["summary"]) <= 200


# ===================================================================
# Statistics tests
# ===================================================================


class TestComputeStats:
    """Tests for compute_stats."""

    def test_basic_stats(self):
        findings = [
            {"severity": "CRITICAL", "in_kev": True, "component": {"name": "a", "version": "1"}},
            {"severity": "HIGH", "in_kev": False, "component": {"name": "b", "version": "1"}},
            {"severity": "MEDIUM", "in_kev": False, "component": {"name": "a", "version": "1"}},
        ]
        stats = compute_stats(findings)
        assert stats["total_vulnerabilities"] == 3
        assert stats["kev_count"] == 1
        assert stats["critical_count"] == 1
        assert stats["high_count"] == 1
        assert stats["medium_count"] == 1
        assert stats["affected_components"] == 2

    def test_empty_findings(self):
        stats = compute_stats([])
        assert stats["total_vulnerabilities"] == 0
        assert stats["kev_count"] == 0

    def test_all_same_severity(self):
        findings = [
            {"severity": "LOW", "in_kev": False, "component": {"name": "x", "version": "1"}},
            {"severity": "LOW", "in_kev": False, "component": {"name": "y", "version": "2"}},
        ]
        stats = compute_stats(findings)
        assert stats["low_count"] == 2
        assert stats["high_count"] == 0


# ===================================================================
# Full scan_sbom integration tests (all HTTP mocked)
# ===================================================================


class TestScanSbom:
    """Tests for scan_sbom (end-to-end with mocked HTTP)."""

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_batch")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_full_scan_with_findings(self, mock_osv, mock_epss, mock_kev, tmp_cdx_sbom):
        mock_osv.return_value = {
            0: [_osv_vuln("GHSA-1111", aliases=["CVE-2024-1111"], database_specific={"severity": "HIGH"})],
        }
        mock_epss.return_value = {"CVE-2024-1111": 0.75}
        mock_kev.return_value = {"CVE-2024-1111"}

        result = scan_sbom(str(tmp_cdx_sbom))
        assert result["sbom_format"] == "CycloneDX"
        assert result["total_components"] == 2
        assert len(result["findings"]) == 1
        assert result["findings"][0]["in_kev"] is True
        assert result["findings"][0]["epss"] == 0.75
        assert "KEV" in result["message"]

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_batch")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_no_vulnerabilities(self, mock_osv, mock_epss, mock_kev, tmp_cdx_sbom):
        mock_osv.return_value = {}
        result = scan_sbom(str(tmp_cdx_sbom))
        assert result["total_components"] == 2
        assert len(result["findings"]) == 0
        assert result["stats"]["total_vulnerabilities"] == 0
        # EPSS/KEV should not have been called (no CVEs to enrich).
        mock_epss.assert_not_called()
        mock_kev.assert_not_called()

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_batch")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_spdx_scan(self, mock_osv, mock_epss, mock_kev, tmp_spdx_sbom):
        mock_osv.return_value = {
            0: [_osv_vuln("GHSA-3333", aliases=["CVE-2024-3333"], database_specific={"severity": "CRITICAL"})],
        }
        mock_epss.return_value = {"CVE-2024-3333": 0.95}
        mock_kev.return_value = set()
        result = scan_sbom(str(tmp_spdx_sbom))
        assert result["sbom_format"] == "SPDX"
        assert result["stats"]["critical_count"] == 1

    def test_empty_sbom(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text(json.dumps(_cdx_sbom([])))
        result = scan_sbom(str(p))
        assert result["total_components"] == 0
        assert "no components" in result["message"].lower()

    def test_invalid_file(self, tmp_path):
        with pytest.raises(ValueError):
            scan_sbom(str(tmp_path / "nope.json"))

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_batch")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_message_includes_severity_breakdown(self, mock_osv, mock_epss, mock_kev, tmp_cdx_sbom):
        mock_osv.return_value = {
            0: [_osv_vuln("V-1", aliases=["CVE-2024-0001"], database_specific={"severity": "CRITICAL"})],
            1: [_osv_vuln("V-2", aliases=["CVE-2024-0002"], database_specific={"severity": "MEDIUM"})],
        }
        mock_epss.return_value = {}
        mock_kev.return_value = set()
        result = scan_sbom(str(tmp_cdx_sbom))
        assert "CRITICAL" in result["message"]
        assert "MEDIUM" in result["message"]

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_batch")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_multiple_vulns_per_component(self, mock_osv, mock_epss, mock_kev, tmp_cdx_sbom):
        mock_osv.return_value = {
            0: [
                _osv_vuln("V-A", aliases=["CVE-2024-0001"], database_specific={"severity": "HIGH"}),
                _osv_vuln("V-B", aliases=["CVE-2024-0002"], database_specific={"severity": "LOW"}),
            ],
        }
        mock_epss.return_value = {"CVE-2024-0001": 0.8, "CVE-2024-0002": 0.1}
        mock_kev.return_value = set()
        result = scan_sbom(str(tmp_cdx_sbom))
        assert len(result["findings"]) == 2


# ===================================================================
# Strands handler tests
# ===================================================================


class TestHandler:
    """Tests for the Strands tool handler."""

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_success(self, mock_scan):
        mock_scan.return_value = {
            "sbom_format": "CycloneDX",
            "total_components": 5,
            "findings": [],
            "stats": {"total_vulnerabilities": 0},
            "message": "All clear",
        }
        tool = {
            "toolUseId": "test-123",
            "name": "scan_sbom",
            "input": {"sbom_path": "/tmp/bom.json"},
        }
        result = handler(tool)
        assert result["status"] == "success"
        assert "All clear" in result["content"][0]["text"]

    def test_missing_path(self):
        tool = {
            "toolUseId": "test-456",
            "name": "scan_sbom",
            "input": {},
        }
        result = handler(tool)
        assert result["status"] == "error"

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_value_error(self, mock_scan):
        mock_scan.side_effect = ValueError("bad file")
        tool = {
            "toolUseId": "test-789",
            "name": "scan_sbom",
            "input": {"sbom_path": "/tmp/bad.json"},
        }
        result = handler(tool)
        assert result["status"] == "error"
        assert "bad file" in result["content"][0]["text"]

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_unexpected_exception(self, mock_scan):
        mock_scan.side_effect = RuntimeError("boom")
        tool = {
            "toolUseId": "test-000",
            "name": "scan_sbom",
            "input": {"sbom_path": "/tmp/crash.json"},
        }
        result = handler(tool)
        assert result["status"] == "error"
        assert "boom" in result["content"][0]["text"]


# ===================================================================
# CLI subcommand tests
# ===================================================================


class TestCliSbomScan:
    """Tests for the CLI sbom-scan dispatch."""

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_text_output(self, mock_scan, capsys, tmp_path):
        from manus_agent.cli import _run_sbom_scan

        mock_scan.return_value = {
            "sbom_format": "CycloneDX",
            "total_components": 3,
            "findings": [
                {
                    "vuln_id": "GHSA-1111",
                    "cve_ids": ["CVE-2024-1111"],
                    "component": {"name": "requests", "version": "2.28.0", "ecosystem": "PyPI"},
                    "severity": "HIGH",
                    "severity_score": 7.5,
                    "epss": 0.75,
                    "in_kev": True,
                    "summary": "A serious vulnerability",
                },
            ],
            "stats": {
                "total_vulnerabilities": 1,
                "kev_count": 1,
                "critical_count": 0,
                "high_count": 1,
                "medium_count": 0,
                "low_count": 0,
                "affected_components": 1,
            },
            "message": "Scanned CycloneDX SBOM: 3 components, 1 vulnerabilities found.",
        }
        ret = _run_sbom_scan([str(tmp_path / "bom.json")])
        assert ret == 0
        out = capsys.readouterr().out
        assert "GHSA-1111" in out
        assert "KEV" in out
        assert "EPSS" in out

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_json_output(self, mock_scan, capsys, tmp_path):
        from manus_agent.cli import _run_sbom_scan

        mock_scan.return_value = {
            "sbom_format": "CycloneDX",
            "total_components": 1,
            "findings": [],
            "stats": {
                "total_vulnerabilities": 0,
                "kev_count": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
                "affected_components": 0,
            },
            "message": "No vulns",
        }
        ret = _run_sbom_scan(["--output", "json", str(tmp_path / "bom.json")])
        assert ret == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["total_components"] == 1

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_value_error_returns_1(self, mock_scan, capsys, tmp_path):
        from manus_agent.cli import _run_sbom_scan

        mock_scan.side_effect = ValueError("File not found")
        ret = _run_sbom_scan([str(tmp_path / "missing.json")])
        assert ret == 1
        err = capsys.readouterr().err
        assert "File not found" in err

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_generic_exception_returns_1(self, mock_scan, capsys, tmp_path):
        from manus_agent.cli import _run_sbom_scan

        mock_scan.side_effect = RuntimeError("Unexpected")
        ret = _run_sbom_scan([str(tmp_path / "bom.json")])
        assert ret == 1

    @patch("manus_agent.tools.scan_sbom.scan_sbom")
    def test_no_findings_text(self, mock_scan, capsys, tmp_path):
        from manus_agent.cli import _run_sbom_scan

        mock_scan.return_value = {
            "sbom_format": "CycloneDX",
            "total_components": 5,
            "findings": [],
            "stats": {
                "total_vulnerabilities": 0,
                "kev_count": 0,
                "critical_count": 0,
                "high_count": 0,
                "medium_count": 0,
                "low_count": 0,
                "affected_components": 0,
            },
            "message": "No vulns found",
        }
        ret = _run_sbom_scan([str(tmp_path / "bom.json")])
        assert ret == 0
        out = capsys.readouterr().out
        assert "all clear" in out.lower()

    def test_subcommands_includes_sbom_scan(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "sbom-scan" in _SUBCOMMANDS


# ===================================================================
# HTTP retry tests
# ===================================================================


class TestRequestWithRetry:
    """Tests for _request_with_retry."""

    @patch("manus_agent.tools.scan_sbom.requests.request")
    @patch("manus_agent.tools.scan_sbom.time.sleep")
    def test_success_on_first_try(self, mock_sleep, mock_request):
        from manus_agent.tools.scan_sbom import _request_with_retry

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_request.return_value = mock_resp
        resp = _request_with_retry("GET", "https://example.com")
        assert resp.status_code == 200
        mock_sleep.assert_not_called()

    @patch("manus_agent.tools.scan_sbom._MAX_RETRIES", 2)
    @patch("manus_agent.tools.scan_sbom._RETRY_BASE_DELAY", 0)
    @patch("manus_agent.tools.scan_sbom.requests.request")
    @patch("manus_agent.tools.scan_sbom.time.sleep")
    def test_retries_on_500(self, mock_sleep, mock_request):
        from manus_agent.tools.scan_sbom import _request_with_retry

        resp_500 = MagicMock()
        resp_500.status_code = 500
        resp_200 = MagicMock()
        resp_200.status_code = 200
        mock_request.side_effect = [resp_500, resp_200]
        resp = _request_with_retry("GET", "https://example.com")
        assert resp.status_code == 200
        assert mock_request.call_count == 2

    @patch("manus_agent.tools.scan_sbom._MAX_RETRIES", 2)
    @patch("manus_agent.tools.scan_sbom._RETRY_BASE_DELAY", 0)
    @patch("manus_agent.tools.scan_sbom.requests.request")
    @patch("manus_agent.tools.scan_sbom.time.sleep")
    def test_retries_on_exception(self, mock_sleep, mock_request):
        import requests as req_lib

        from manus_agent.tools.scan_sbom import _request_with_retry

        resp_200 = MagicMock()
        resp_200.status_code = 200
        mock_request.side_effect = [req_lib.ConnectionError("fail"), resp_200]
        resp = _request_with_retry("GET", "https://example.com")
        assert resp.status_code == 200

    @patch("manus_agent.tools.scan_sbom._MAX_RETRIES", 1)
    @patch("manus_agent.tools.scan_sbom.requests.request")
    def test_raises_on_exhausted_retries(self, mock_request):
        import requests as req_lib

        from manus_agent.tools.scan_sbom import _request_with_retry

        mock_request.side_effect = req_lib.ConnectionError("persistent")
        with pytest.raises(req_lib.ConnectionError):
            _request_with_retry("GET", "https://example.com")


# ===================================================================
# Edge case / regression tests
# ===================================================================


class TestEdgeCases:
    """Edge cases and regressions."""

    def test_purl_with_at_in_namespace(self):
        """npm scoped packages have @ in the namespace."""
        eco, name, ver = _parse_purl("pkg:npm/@babel/core@7.23.0")
        assert eco == "npm"
        assert name == "@babel/core"
        assert ver == "7.23.0"

    def test_cyclonedx_component_no_name(self):
        doc = _cdx_sbom([{"version": "1.0"}])
        comps = _parse_cyclonedx(doc)
        assert comps == []

    def test_spdx_package_no_name(self):
        doc = _spdx_sbom([{"versionInfo": "1.0"}])
        comps = _parse_spdx(doc)
        assert comps == []

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_batch")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_vuln_with_no_cves(self, mock_osv, mock_epss, mock_kev, tmp_cdx_sbom):
        """OSV vulns that have no CVE aliases should still appear."""
        mock_osv.return_value = {0: [_osv_vuln("GHSA-ONLY", aliases=["GHSA-yyyy-zzzz-wwww"])]}
        mock_epss.return_value = {}
        mock_kev.return_value = set()
        result = scan_sbom(str(tmp_cdx_sbom))
        assert len(result["findings"]) == 1
        assert result["findings"][0]["cve_ids"] == []
        assert result["findings"][0]["in_kev"] is False
        assert result["findings"][0]["epss"] == 0.0

    def test_build_osv_queries_ecosystem_optional(self):
        from manus_agent.tools.scan_sbom import _build_osv_queries

        comps = [{"name": "foo", "version": "1.0", "ecosystem": ""}]
        queries = _build_osv_queries(comps)
        assert "ecosystem" not in queries[0]["package"]

    def test_build_osv_queries_with_ecosystem(self):
        from manus_agent.tools.scan_sbom import _build_osv_queries

        comps = [{"name": "foo", "version": "1.0", "ecosystem": "PyPI"}]
        queries = _build_osv_queries(comps)
        assert queries[0]["package"]["ecosystem"] == "PyPI"

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_batch")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_large_sbom_stats(self, mock_osv, mock_epss, mock_kev, tmp_path):
        """Test with many components, some vulnerable."""
        components = [
            {"type": "library", "name": f"pkg{i}", "version": "1.0", "purl": f"pkg:pypi/pkg{i}@1.0"} for i in range(50)
        ]
        p = tmp_path / "large.json"
        p.write_text(json.dumps(_cdx_sbom(components)))
        # 10 of 50 have vulns.
        osv_map = {}
        for i in range(0, 10):
            sev = "CRITICAL" if i < 3 else "HIGH" if i < 7 else "MEDIUM"
            osv_map[i] = [
                _osv_vuln(
                    f"GHSA-{i:04d}",
                    aliases=[f"CVE-2024-{i:04d}"],
                    database_specific={"severity": sev},
                )
            ]
        mock_osv.return_value = osv_map
        epss_map = {f"CVE-2024-{i:04d}": 0.9 - i * 0.05 for i in range(10)}
        mock_epss.return_value = epss_map
        mock_kev.return_value = {"CVE-2024-0000", "CVE-2024-0001"}

        result = scan_sbom(str(p))
        assert result["total_components"] == 50
        assert result["stats"]["total_vulnerabilities"] == 10
        assert result["stats"]["kev_count"] == 2
        assert result["stats"]["critical_count"] == 3
        assert result["stats"]["affected_components"] == 10
        # First finding should be a KEV one.
        assert result["findings"][0]["in_kev"] is True

    def test_purl_type_mapping_coverage(self):
        """Ensure all mapped purl types resolve to a non-empty ecosystem."""
        for purl_type, ecosystem in _PURL_TYPE_TO_ECOSYSTEM.items():
            assert ecosystem, f"Empty ecosystem for purl type: {purl_type}"
            eco, _, _ = _parse_purl(f"pkg:{purl_type}/test@1.0")
            assert eco == ecosystem

    @patch("manus_agent.tools.scan_sbom.fetch_kev_set")
    @patch("manus_agent.tools.scan_sbom.fetch_epss_batch")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_scan_sbom_returns_sorted(self, mock_osv, mock_epss, mock_kev, tmp_cdx_sbom):
        """Verify findings are sorted KEV > EPSS > severity."""
        mock_osv.return_value = {
            0: [
                _osv_vuln("V-A", aliases=["CVE-2024-0001"], database_specific={"severity": "HIGH"}),
            ],
            1: [
                _osv_vuln("V-B", aliases=["CVE-2024-0002"], database_specific={"severity": "CRITICAL"}),
            ],
        }
        mock_epss.return_value = {"CVE-2024-0001": 0.2, "CVE-2024-0002": 0.1}
        mock_kev.return_value = {"CVE-2024-0001"}
        result = scan_sbom(str(tmp_cdx_sbom))
        # V-A is in KEV → should come first despite lower EPSS/severity.
        assert result["findings"][0]["vuln_id"] == "V-A"
        assert result["findings"][1]["vuln_id"] == "V-B"
