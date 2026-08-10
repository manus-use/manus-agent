#!/usr/bin/env python3
"""Comprehensive test suite for scan_sbom tool and sbom-scan CLI subcommand."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CYCLONEDX = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.4",
    "components": [
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
        {
            "type": "library",
            "name": "spring-core",
            "version": "5.3.20",
            "purl": "pkg:maven/org.springframework/spring-core@5.3.20",
        },
    ],
}

SAMPLE_SPDX = {
    "spdxVersion": "SPDX-2.3",
    "packages": [
        {
            "name": "axios",
            "versionInfo": "1.6.0",
            "externalRefs": [
                {
                    "referenceType": "purl",
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceLocator": "pkg:npm/axios@1.6.0",
                }
            ],
        },
        {
            "name": "flask",
            "versionInfo": "2.3.0",
            "externalRefs": [
                {
                    "referenceType": "purl",
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceLocator": "pkg:pypi/flask@2.3.0",
                }
            ],
        },
    ],
}

SAMPLE_OSV_BATCH_RESPONSE = {
    "results": [
        {
            "vulns": [
                {
                    "id": "GHSA-j8r2-6x86-q33q",
                    "aliases": ["CVE-2023-32681"],
                    "summary": "Unintended leak of Proxy-Authorization header in requests",
                    "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:H/PR:N/UI:R/S:C/C:H/I:N/A:N"}],
                }
            ]
        },
        {
            "vulns": [
                {
                    "id": "CVE-2021-23337",
                    "aliases": [],
                    "summary": "Lodash Command Injection",
                    "severity": [],
                }
            ]
        },
        {"vulns": []},  # spring-core - no vulns
    ]
}

SAMPLE_EPSS_RESPONSE = {
    "status": "OK",
    "data": [
        {"cve": "CVE-2023-32681", "epss": "0.00234", "percentile": "0.612"},
        {"cve": "CVE-2021-23337", "epss": "0.87654", "percentile": "0.991"},
    ],
}

SAMPLE_KEV_RESPONSE = {
    "vulnerabilities": [
        {"cveID": "CVE-2021-23337"},
        {"cveID": "CVE-2021-44228"},
    ]
}


@pytest.fixture
def cyclonedx_file(tmp_path: Path) -> Path:
    """Create a temp CycloneDX SBOM file."""
    f = tmp_path / "bom.json"
    f.write_text(json.dumps(SAMPLE_CYCLONEDX))
    return f


@pytest.fixture
def spdx_file(tmp_path: Path) -> Path:
    """Create a temp SPDX SBOM file."""
    f = tmp_path / "sbom.spdx.json"
    f.write_text(json.dumps(SAMPLE_SPDX))
    return f


@pytest.fixture
def empty_cyclonedx_file(tmp_path: Path) -> Path:
    """CycloneDX with no components."""
    f = tmp_path / "empty.json"
    f.write_text(json.dumps({"bomFormat": "CycloneDX", "components": []}))
    return f


@pytest.fixture
def invalid_json_file(tmp_path: Path) -> Path:
    """Invalid JSON file."""
    f = tmp_path / "bad.json"
    f.write_text("not valid json {{{")
    return f


@pytest.fixture
def unknown_format_file(tmp_path: Path) -> Path:
    """Valid JSON but unrecognised SBOM format."""
    f = tmp_path / "unknown.json"
    f.write_text(json.dumps({"foo": "bar", "items": []}))
    return f


def _mock_tool_use(file_path: str) -> dict[str, Any]:
    """Build a mock ToolUse dict."""
    return {"toolUseId": "test-123", "input": {"file_path": file_path}}


# ---------------------------------------------------------------------------
# TOOL_SPEC contract tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Verify TOOL_SPEC follows project conventions."""

    def test_tool_spec_exists(self):
        from manus_agent.tools.scan_sbom import TOOL_SPEC

        assert TOOL_SPEC is not None

    def test_tool_spec_has_name(self):
        from manus_agent.tools.scan_sbom import TOOL_SPEC

        assert TOOL_SPEC["name"] == "scan_sbom"

    def test_tool_spec_has_description(self):
        from manus_agent.tools.scan_sbom import TOOL_SPEC

        assert len(TOOL_SPEC["description"]) > 20

    def test_tool_spec_has_input_schema(self):
        from manus_agent.tools.scan_sbom import TOOL_SPEC

        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "file_path" in schema["properties"]
        assert "file_path" in schema["required"]


# ---------------------------------------------------------------------------
# PURL parsing tests
# ---------------------------------------------------------------------------


class TestPurlParsing:
    """Test Package URL parsing logic."""

    def test_parse_pypi_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:pypi/requests@2.28.0")
        assert result == ("PyPI", "requests", "2.28.0")

    def test_parse_npm_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:npm/lodash@4.17.20")
        assert result == ("npm", "lodash", "4.17.20")

    def test_parse_maven_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:maven/org.springframework/spring-core@5.3.20")
        assert result == ("Maven", "org.springframework:spring-core", "5.3.20")

    def test_parse_cargo_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:cargo/serde@1.0.188")
        assert result == ("crates.io", "serde", "1.0.188")

    def test_parse_golang_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:golang/github.com/gin-gonic/gin@1.9.1")
        assert result == ("Go", "github.com/gin-gonic/gin", "1.9.1")

    def test_parse_nuget_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:nuget/Newtonsoft.Json@13.0.3")
        assert result == ("NuGet", "Newtonsoft.Json", "13.0.3")

    def test_parse_gem_purl(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:gem/rails@7.0.0")
        assert result == ("RubyGems", "rails", "7.0.0")

    def test_parse_purl_with_qualifiers(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:npm/lodash@4.17.20?type=module")
        assert result == ("npm", "lodash", "4.17.20")

    def test_parse_purl_with_subpath(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:npm/lodash@4.17.20#sub/path")
        assert result == ("npm", "lodash", "4.17.20")

    def test_parse_purl_unknown_type(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:conan/openssl@3.0.0")
        # Unknown purl type maps to the type itself
        assert result == ("conan", "openssl", "3.0.0")

    def test_parse_purl_no_version_returns_none(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:npm/lodash")
        assert result is None

    def test_parse_purl_invalid_no_pkg_prefix(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("npm/lodash@1.0.0")
        assert result is None

    def test_parse_purl_no_slash_returns_none(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:invalid")
        assert result is None

    def test_parse_purl_encoded_chars(self):
        from manus_agent.tools.scan_sbom import _parse_purl

        result = _parse_purl("pkg:npm/%40scope/pkg@1.0.0")
        assert result == ("npm", "@scope/pkg", "1.0.0")


# ---------------------------------------------------------------------------
# SBOM Parsing tests
# ---------------------------------------------------------------------------


class TestSbomParsing:
    """Test CycloneDX and SPDX parsing."""

    def test_parse_cyclonedx_components(self, cyclonedx_file: Path):
        from manus_agent.tools.scan_sbom import parse_sbom

        fmt, components = parse_sbom(str(cyclonedx_file))
        assert fmt == "CycloneDX"
        assert len(components) == 3
        assert components[0]["name"] == "requests"
        assert components[0]["ecosystem"] == "PyPI"
        assert components[0]["version"] == "2.28.0"

    def test_parse_spdx_packages(self, spdx_file: Path):
        from manus_agent.tools.scan_sbom import parse_sbom

        fmt, components = parse_sbom(str(spdx_file))
        assert fmt == "SPDX"
        assert len(components) == 2
        assert components[0]["name"] == "axios"
        assert components[0]["ecosystem"] == "npm"

    def test_parse_empty_cyclonedx(self, empty_cyclonedx_file: Path):
        from manus_agent.tools.scan_sbom import parse_sbom

        fmt, components = parse_sbom(str(empty_cyclonedx_file))
        assert fmt == "CycloneDX"
        assert components == []

    def test_parse_file_not_found(self):
        from manus_agent.tools.scan_sbom import parse_sbom

        with pytest.raises(FileNotFoundError):
            parse_sbom("/nonexistent/file.json")

    def test_parse_invalid_json(self, invalid_json_file: Path):
        from manus_agent.tools.scan_sbom import parse_sbom

        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_sbom(str(invalid_json_file))

    def test_parse_unknown_format(self, unknown_format_file: Path):
        from manus_agent.tools.scan_sbom import parse_sbom

        with pytest.raises(ValueError, match="Unrecognised SBOM format"):
            parse_sbom(str(unknown_format_file))

    def test_cyclonedx_fallback_no_purl(self, tmp_path: Path):
        """Components without purl use name/version fallback."""
        from manus_agent.tools.scan_sbom import parse_sbom

        data = {
            "bomFormat": "CycloneDX",
            "components": [
                {"type": "library", "name": "mylib", "version": "1.0.0"},
            ],
        }
        f = tmp_path / "nopurl.json"
        f.write_text(json.dumps(data))
        fmt, components = parse_sbom(str(f))
        assert len(components) == 1
        assert components[0]["name"] == "mylib"
        assert components[0]["ecosystem"] == ""

    def test_spdx_fallback_no_purl(self, tmp_path: Path):
        """SPDX packages without externalRefs use name/version fallback."""
        from manus_agent.tools.scan_sbom import parse_sbom

        data = {
            "spdxVersion": "SPDX-2.3",
            "packages": [
                {"name": "mylib", "versionInfo": "2.0.0", "externalRefs": []},
            ],
        }
        f = tmp_path / "nopurl-spdx.json"
        f.write_text(json.dumps(data))
        fmt, components = parse_sbom(str(f))
        assert len(components) == 1
        assert components[0]["name"] == "mylib"
        assert components[0]["version"] == "2.0.0"

    def test_spdx_skips_noassertion(self, tmp_path: Path):
        """SPDX packages with NOASSERTION name or version are skipped."""
        from manus_agent.tools.scan_sbom import parse_sbom

        data = {
            "spdxVersion": "SPDX-2.3",
            "packages": [
                {"name": "NOASSERTION", "versionInfo": "1.0.0", "externalRefs": []},
                {"name": "pkg", "versionInfo": "NOASSERTION", "externalRefs": []},
            ],
        }
        f = tmp_path / "noassert.json"
        f.write_text(json.dumps(data))
        fmt, components = parse_sbom(str(f))
        assert components == []

    def test_root_not_dict(self, tmp_path: Path):
        """Root-level JSON array should raise ValueError."""
        from manus_agent.tools.scan_sbom import parse_sbom

        f = tmp_path / "array.json"
        f.write_text(json.dumps([{"name": "x"}]))
        with pytest.raises(ValueError, match="must contain a JSON object"):
            parse_sbom(str(f))


# ---------------------------------------------------------------------------
# OSV Batch Query tests
# ---------------------------------------------------------------------------


class TestOsvBatchQuery:
    """Test OSV.dev batch query logic."""

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_query_empty_components(self, mock_req):
        from manus_agent.tools.scan_sbom import query_osv_batch

        result = query_osv_batch([])
        assert result == []
        mock_req.assert_not_called()

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_query_returns_vulnerable(self, mock_req):
        from manus_agent.tools.scan_sbom import query_osv_batch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_OSV_BATCH_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_req.return_value = mock_resp

        components = [
            {"ecosystem": "PyPI", "name": "requests", "version": "2.28.0", "purl": "pkg:pypi/requests@2.28.0"},
            {"ecosystem": "npm", "name": "lodash", "version": "4.17.20", "purl": "pkg:npm/lodash@4.17.20"},
            {
                "ecosystem": "Maven",
                "name": "spring-core",
                "version": "5.3.20",
                "purl": "pkg:maven/org.springframework/spring-core@5.3.20",
            },
        ]
        result = query_osv_batch(components)
        # spring-core has no vulns, so only 2 results
        assert len(result) == 2
        assert result[0]["name"] == "requests"
        assert result[1]["name"] == "lodash"

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_query_builds_correct_payload(self, mock_req):
        from manus_agent.tools.scan_sbom import query_osv_batch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": [{"vulns": []}]}
        mock_resp.raise_for_status = MagicMock()
        mock_req.return_value = mock_resp

        components = [
            {"ecosystem": "npm", "name": "express", "version": "4.18.0", "purl": "pkg:npm/express@4.18.0"},
        ]
        query_osv_batch(components)

        call_kwargs = mock_req.call_args
        payload = call_kwargs.kwargs.get("json_body") or call_kwargs[1].get("json_body")
        assert payload["queries"][0]["version"] == "4.18.0"
        assert payload["queries"][0]["package"]["name"] == "express"
        assert payload["queries"][0]["package"]["ecosystem"] == "npm"

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_query_no_ecosystem_omits_ecosystem_key(self, mock_req):
        from manus_agent.tools.scan_sbom import query_osv_batch

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": [{"vulns": []}]}
        mock_resp.raise_for_status = MagicMock()
        mock_req.return_value = mock_resp

        components = [
            {"ecosystem": "", "name": "mylib", "version": "1.0.0", "purl": ""},
        ]
        query_osv_batch(components)

        call_kwargs = mock_req.call_args
        payload = call_kwargs.kwargs.get("json_body") or call_kwargs[1].get("json_body")
        assert "ecosystem" not in payload["queries"][0]["package"]


# ---------------------------------------------------------------------------
# EPSS Enrichment tests
# ---------------------------------------------------------------------------


class TestEpssEnrichment:
    """Test EPSS score fetching."""

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_fetch_epss_scores(self, mock_req):
        from manus_agent.tools.scan_sbom import _fetch_epss_scores

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_EPSS_RESPONSE
        mock_req.return_value = mock_resp

        scores = _fetch_epss_scores(["CVE-2023-32681", "CVE-2021-23337"])
        assert scores["CVE-2023-32681"] == pytest.approx(0.00234)
        assert scores["CVE-2021-23337"] == pytest.approx(0.87654)

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_fetch_epss_empty_list(self, mock_req):
        from manus_agent.tools.scan_sbom import _fetch_epss_scores

        scores = _fetch_epss_scores([])
        assert scores == {}
        mock_req.assert_not_called()

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_fetch_epss_graceful_failure(self, mock_req):
        from manus_agent.tools.scan_sbom import _fetch_epss_scores

        mock_req.side_effect = Exception("network error")
        scores = _fetch_epss_scores(["CVE-2023-32681"])
        assert scores == {}

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_fetch_epss_non_200_graceful(self, mock_req):
        from manus_agent.tools.scan_sbom import _fetch_epss_scores

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_req.return_value = mock_resp
        scores = _fetch_epss_scores(["CVE-2023-32681"])
        assert scores == {}


# ---------------------------------------------------------------------------
# KEV Enrichment tests
# ---------------------------------------------------------------------------


class TestKevEnrichment:
    """Test CISA KEV catalog fetching."""

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_fetch_kev_cve_set(self, mock_req):
        from manus_agent.tools.scan_sbom import _fetch_kev_cve_set

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = SAMPLE_KEV_RESPONSE
        mock_req.return_value = mock_resp

        kev_set = _fetch_kev_cve_set()
        assert "CVE-2021-23337" in kev_set
        assert "CVE-2021-44228" in kev_set

    @patch("manus_agent.tools.scan_sbom._request_with_retry")
    def test_fetch_kev_graceful_failure(self, mock_req):
        from manus_agent.tools.scan_sbom import _fetch_kev_cve_set

        mock_req.side_effect = Exception("timeout")
        kev_set = _fetch_kev_cve_set()
        assert kev_set == set()


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Test HTTP retry/back-off."""

    @patch("manus_agent.tools.scan_sbom.time.sleep")
    @patch("manus_agent.tools.scan_sbom.requests.request")
    def test_retry_on_429(self, mock_request, mock_sleep):
        from manus_agent.tools.scan_sbom import _request_with_retry

        # First call returns 429, second returns 200
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_200 = MagicMock()
        resp_200.status_code = 200
        mock_request.side_effect = [resp_429, resp_200]

        result = _request_with_retry("GET", "http://example.com")
        assert result.status_code == 200
        assert mock_request.call_count == 2
        mock_sleep.assert_called_once()

    @patch("manus_agent.tools.scan_sbom.time.sleep")
    @patch("manus_agent.tools.scan_sbom.requests.request")
    def test_retry_on_network_error(self, mock_request, mock_sleep):
        import requests as req

        from manus_agent.tools.scan_sbom import _request_with_retry

        # All retries fail with connection error
        mock_request.side_effect = req.ConnectionError("refused")

        with pytest.raises(req.ConnectionError):
            _request_with_retry("GET", "http://example.com")

    @patch("manus_agent.tools.scan_sbom.time.sleep")
    @patch("manus_agent.tools.scan_sbom.requests.request")
    def test_no_retry_on_400(self, mock_request, mock_sleep):
        from manus_agent.tools.scan_sbom import _request_with_retry

        resp_400 = MagicMock()
        resp_400.status_code = 400
        mock_request.return_value = resp_400

        result = _request_with_retry("GET", "http://example.com")
        assert result.status_code == 400
        assert mock_request.call_count == 1
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Full scan integration tests (mocked HTTP)
# ---------------------------------------------------------------------------


class TestScanSbomFile:
    """Test the scan_sbom_file orchestration function."""

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_scan_cyclonedx_full(self, mock_osv, mock_epss, mock_kev, cyclonedx_file):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [
            {
                "ecosystem": "PyPI",
                "name": "requests",
                "version": "2.28.0",
                "purl": "pkg:pypi/requests@2.28.0",
                "vulns": [
                    {
                        "id": "GHSA-j8r2-6x86-q33q",
                        "aliases": ["CVE-2023-32681"],
                        "summary": "Proxy leak",
                        "severity": [],
                    },
                ],
            }
        ]
        mock_epss.return_value = {"CVE-2023-32681": 0.05}
        mock_kev.return_value = set()

        result = scan_sbom_file(str(cyclonedx_file))
        assert result["format"] == "CycloneDX"
        assert result["total_components"] == 3
        assert result["vulnerable_components"] == 1
        assert result["total_vulnerabilities"] == 1
        assert result["findings"][0]["cve_id"] == "CVE-2023-32681"
        assert result["findings"][0]["epss"] == 0.05
        assert result["findings"][0]["in_kev"] is False

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_scan_kev_finding_ranked_first(self, mock_osv, mock_epss, mock_kev, cyclonedx_file):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [
            {
                "ecosystem": "PyPI",
                "name": "requests",
                "version": "2.28.0",
                "purl": "",
                "vulns": [
                    {"id": "CVE-2023-32681", "aliases": [], "summary": "Proxy leak", "severity": []},
                    {"id": "CVE-2021-23337", "aliases": [], "summary": "Command injection", "severity": []},
                ],
            }
        ]
        mock_epss.return_value = {"CVE-2023-32681": 0.9, "CVE-2021-23337": 0.1}
        mock_kev.return_value = {"CVE-2021-23337"}

        result = scan_sbom_file(str(cyclonedx_file))
        # KEV finding should come first despite lower EPSS
        assert result["findings"][0]["cve_id"] == "CVE-2021-23337"
        assert result["findings"][0]["in_kev"] is True
        assert result["findings"][1]["cve_id"] == "CVE-2023-32681"

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_scan_no_vulnerabilities(self, mock_osv, mock_epss, mock_kev, cyclonedx_file):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = []
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        result = scan_sbom_file(str(cyclonedx_file))
        assert result["vulnerable_components"] == 0
        assert result["total_vulnerabilities"] == 0
        assert result["findings"] == []

    def test_scan_empty_sbom(self, empty_cyclonedx_file):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        result = scan_sbom_file(str(empty_cyclonedx_file))
        assert result["total_components"] == 0
        assert result["findings"] == []

    def test_scan_file_not_found(self):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        with pytest.raises(FileNotFoundError):
            scan_sbom_file("/nonexistent/bom.json")

    def test_scan_invalid_json(self, invalid_json_file):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        with pytest.raises(ValueError, match="Invalid JSON"):
            scan_sbom_file(str(invalid_json_file))

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_scan_critical_count(self, mock_osv, mock_epss, mock_kev, cyclonedx_file):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [
            {
                "ecosystem": "PyPI",
                "name": "requests",
                "version": "2.28.0",
                "purl": "",
                "vulns": [
                    {"id": "CVE-2023-0001", "aliases": [], "summary": "A", "severity": []},
                    {"id": "CVE-2023-0002", "aliases": [], "summary": "B", "severity": []},
                    {"id": "CVE-2023-0003", "aliases": [], "summary": "C", "severity": []},
                ],
            }
        ]
        # Only CVE-2023-0001 has high EPSS, CVE-2023-0002 is in KEV
        mock_epss.return_value = {"CVE-2023-0001": 0.6, "CVE-2023-0002": 0.01, "CVE-2023-0003": 0.01}
        mock_kev.return_value = {"CVE-2023-0002"}

        result = scan_sbom_file(str(cyclonedx_file))
        # Critical: CVE-2023-0001 (EPSS >= 0.5) + CVE-2023-0002 (in KEV)
        assert result["critical_count"] == 2

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_scan_deduplicates_vulns(self, mock_osv, mock_epss, mock_kev, cyclonedx_file):
        from manus_agent.tools.scan_sbom import scan_sbom_file

        # Same vuln reported by two different packages
        mock_osv.return_value = [
            {
                "ecosystem": "PyPI",
                "name": "requests",
                "version": "2.28.0",
                "purl": "",
                "vulns": [{"id": "CVE-2023-0001", "aliases": [], "summary": "X", "severity": []}],
            },
            {
                "ecosystem": "npm",
                "name": "lodash",
                "version": "4.17.20",
                "purl": "",
                "vulns": [{"id": "CVE-2023-0001", "aliases": [], "summary": "X", "severity": []}],
            },
        ]
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        result = scan_sbom_file(str(cyclonedx_file))
        assert result["total_vulnerabilities"] == 1


# ---------------------------------------------------------------------------
# Strands tool entry point tests
# ---------------------------------------------------------------------------


class TestToolEntryPoint:
    """Test the scan_sbom Strands tool function."""

    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_success(self, mock_scan, cyclonedx_file):
        from manus_agent.tools.scan_sbom import scan_sbom

        mock_scan.return_value = {
            "format": "CycloneDX",
            "findings": [],
            "total_components": 3,
            "vulnerable_components": 0,
            "total_vulnerabilities": 0,
            "critical_count": 0,
        }
        tool_use = _mock_tool_use(str(cyclonedx_file))
        result = scan_sbom(tool_use)
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"

    def test_missing_file_path(self):
        from manus_agent.tools.scan_sbom import scan_sbom

        tool_use = {"toolUseId": "test-456", "input": {"file_path": ""}}
        result = scan_sbom(tool_use)
        assert result["status"] == "error"
        assert "required" in result["content"][0]["text"]

    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_file_not_found_error(self, mock_scan):
        from manus_agent.tools.scan_sbom import scan_sbom

        mock_scan.side_effect = FileNotFoundError("SBOM file not found: /x.json")
        tool_use = _mock_tool_use("/x.json")
        result = scan_sbom(tool_use)
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"]

    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_value_error(self, mock_scan):
        from manus_agent.tools.scan_sbom import scan_sbom

        mock_scan.side_effect = ValueError("Invalid JSON")
        tool_use = _mock_tool_use("/bad.json")
        result = scan_sbom(tool_use)
        assert result["status"] == "error"
        assert "Invalid JSON" in result["content"][0]["text"]

    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_network_error(self, mock_scan):
        import requests as req

        from manus_agent.tools.scan_sbom import scan_sbom

        mock_scan.side_effect = req.ConnectionError("timeout")
        tool_use = _mock_tool_use("/bom.json")
        result = scan_sbom(tool_use)
        assert result["status"] == "error"
        assert "Network error" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# CLI subcommand tests
# ---------------------------------------------------------------------------


class TestCliSbomScan:
    """Test manus-agent sbom-scan CLI subcommand."""

    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_cli_json_output(self, mock_scan, cyclonedx_file, capsys):
        from manus_agent.cli import _run_sbom_scan

        mock_scan.return_value = {
            "format": "CycloneDX",
            "total_components": 3,
            "vulnerable_components": 1,
            "total_vulnerabilities": 2,
            "critical_count": 1,
            "findings": [
                {
                    "vuln_id": "CVE-2023-0001",
                    "cve_id": "CVE-2023-0001",
                    "aliases": [],
                    "summary": "Test",
                    "affected_package": "req",
                    "affected_ecosystem": "PyPI",
                    "affected_version": "2.28.0",
                    "epss": 0.5,
                    "in_kev": True,
                    "cvss_vector": None,
                },
            ],
        }
        exit_code = _run_sbom_scan([str(cyclonedx_file), "--output", "json"])
        assert exit_code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["critical_count"] == 1

    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_cli_text_output_no_findings(self, mock_scan, cyclonedx_file):
        from manus_agent.cli import _run_sbom_scan

        mock_scan.return_value = {
            "format": "CycloneDX",
            "total_components": 3,
            "vulnerable_components": 0,
            "total_vulnerabilities": 0,
            "critical_count": 0,
            "findings": [],
        }
        exit_code = _run_sbom_scan([str(cyclonedx_file)])
        assert exit_code == 0

    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_cli_text_output_with_findings(self, mock_scan, cyclonedx_file):
        from manus_agent.cli import _run_sbom_scan

        mock_scan.return_value = {
            "format": "CycloneDX",
            "total_components": 3,
            "vulnerable_components": 1,
            "total_vulnerabilities": 1,
            "critical_count": 0,
            "findings": [
                {
                    "vuln_id": "GHSA-abc",
                    "cve_id": "CVE-2023-0001",
                    "aliases": [],
                    "summary": "Short summary",
                    "affected_package": "pkg",
                    "affected_ecosystem": "npm",
                    "affected_version": "1.0.0",
                    "epss": 0.01,
                    "in_kev": False,
                    "cvss_vector": None,
                },
            ],
        }
        exit_code = _run_sbom_scan([str(cyclonedx_file)])
        assert exit_code == 0

    def test_cli_file_not_found(self, capsys):
        from manus_agent.cli import _run_sbom_scan

        exit_code = _run_sbom_scan(["/nonexistent/bom.json"])
        assert exit_code == 1

    def test_cli_no_args(self):
        from manus_agent.cli import _run_sbom_scan

        with pytest.raises(SystemExit) as exc_info:
            _run_sbom_scan([])
        assert exc_info.value.code == 2  # argparse error

    @patch("manus_agent.tools.scan_sbom.scan_sbom_file")
    def test_cli_invalid_json_file(self, mock_scan, invalid_json_file):
        from manus_agent.cli import _run_sbom_scan

        mock_scan.side_effect = ValueError("Invalid JSON in SBOM file")
        exit_code = _run_sbom_scan([str(invalid_json_file)])
        assert exit_code == 1

    def test_cli_sbom_scan_in_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "sbom-scan" in _SUBCOMMANDS


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_ghsa_alias_resolution(self, mock_osv, mock_epss, mock_kev, cyclonedx_file):
        """GHSA IDs should resolve CVE from aliases for EPSS/KEV lookup."""
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [
            {
                "ecosystem": "PyPI",
                "name": "requests",
                "version": "2.28.0",
                "purl": "",
                "vulns": [
                    {"id": "GHSA-j8r2-6x86-q33q", "aliases": ["CVE-2023-32681"], "summary": "Leak", "severity": []},
                ],
            }
        ]
        mock_epss.return_value = {"CVE-2023-32681": 0.8}
        mock_kev.return_value = set()

        result = scan_sbom_file(str(cyclonedx_file))
        assert result["findings"][0]["cve_id"] == "CVE-2023-32681"
        assert result["findings"][0]["epss"] == 0.8

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_vuln_without_cve_alias(self, mock_osv, mock_epss, mock_kev, cyclonedx_file):
        """Vulns without any CVE alias should have empty cve_id and 0 EPSS."""
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [
            {
                "ecosystem": "npm",
                "name": "lodash",
                "version": "4.17.20",
                "purl": "",
                "vulns": [
                    {"id": "GHSA-xxxx-yyyy-zzzz", "aliases": ["GHSA-another"], "summary": "No CVE", "severity": []},
                ],
            }
        ]
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        result = scan_sbom_file(str(cyclonedx_file))
        assert result["findings"][0]["cve_id"] == ""
        assert result["findings"][0]["epss"] == 0.0
        assert result["findings"][0]["in_kev"] is False

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_cvss_vector_extraction(self, mock_osv, mock_epss, mock_kev, cyclonedx_file):
        """CVSS vectors should be extracted from severity list."""
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [
            {
                "ecosystem": "PyPI",
                "name": "requests",
                "version": "2.28.0",
                "purl": "",
                "vulns": [
                    {
                        "id": "CVE-2023-32681",
                        "aliases": [],
                        "summary": "Test",
                        "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"}],
                    },
                ],
            }
        ]
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        result = scan_sbom_file(str(cyclonedx_file))
        assert result["findings"][0]["cvss_vector"] == "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_summary_truncation(self, mock_osv, mock_epss, mock_kev, cyclonedx_file):
        """Long summaries should be truncated to 200 chars."""
        from manus_agent.tools.scan_sbom import scan_sbom_file

        long_summary = "x" * 300
        mock_osv.return_value = [
            {
                "ecosystem": "PyPI",
                "name": "requests",
                "version": "2.28.0",
                "purl": "",
                "vulns": [
                    {"id": "CVE-2023-0001", "aliases": [], "summary": long_summary, "severity": []},
                ],
            }
        ]
        mock_epss.return_value = {}
        mock_kev.return_value = set()

        result = scan_sbom_file(str(cyclonedx_file))
        assert len(result["findings"][0]["summary"]) == 200

    @patch("manus_agent.tools.scan_sbom._fetch_kev_cve_set")
    @patch("manus_agent.tools.scan_sbom._fetch_epss_scores")
    @patch("manus_agent.tools.scan_sbom.query_osv_batch")
    def test_spdx_scan(self, mock_osv, mock_epss, mock_kev, spdx_file):
        """SPDX format should work end-to-end."""
        from manus_agent.tools.scan_sbom import scan_sbom_file

        mock_osv.return_value = [
            {
                "ecosystem": "npm",
                "name": "axios",
                "version": "1.6.0",
                "purl": "pkg:npm/axios@1.6.0",
                "vulns": [
                    {"id": "CVE-2024-0001", "aliases": [], "summary": "SSRF", "severity": []},
                ],
            }
        ]
        mock_epss.return_value = {"CVE-2024-0001": 0.3}
        mock_kev.return_value = set()

        result = scan_sbom_file(str(spdx_file))
        assert result["format"] == "SPDX"
        assert result["total_components"] == 2
        assert result["total_vulnerabilities"] == 1
