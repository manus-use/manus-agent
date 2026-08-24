"""Comprehensive test suite for resolve_advisory_aliases tool.

100% mocked — no real HTTP calls.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.resolve_advisory_aliases import (
    ADVISORY_PREFIXES,
    ADVISORY_URL_TEMPLATES,
    _build_advisory_url,
    _classify_advisory,
    _extract_advisory_from_url,
    _extract_nvd_advisory_refs,
    _get_with_retry,
    _post_with_retry,
    _query_osv_aliases,
    _query_vulncheck_aliases,
    fetch_advisory_aliases,
    resolve_advisory_aliases,
)

# ---------------------------------------------------------------------------
# Helper to build a ToolUse dict
# ---------------------------------------------------------------------------


def _make_tool(cve_id: str = "CVE-2021-44228", include_urls: bool = True) -> dict:
    return {
        "toolUseId": "test-tool-id-123",
        "input": {"cve_id": cve_id, "include_urls": include_urls},
    }


# ---------------------------------------------------------------------------
# Tests: _classify_advisory
# ---------------------------------------------------------------------------


class TestClassifyAdvisory:
    def test_ghsa_prefix(self):
        result = _classify_advisory("GHSA-jfh8-c2jp-5v3q")
        assert result["prefix"] == "GHSA"
        assert result["database"] == "GitHub Security Advisory"
        assert "github.com/advisories" in result["url"]

    def test_rhsa_prefix(self):
        result = _classify_advisory("RHSA-2024:1234")
        assert result["prefix"] == "RHSA"
        assert result["database"] == "Red Hat Security Advisory"

    def test_dsa_prefix(self):
        result = _classify_advisory("DSA-5432-1")
        assert result["prefix"] == "DSA"
        assert result["database"] == "Debian Security Advisory"

    def test_usn_prefix(self):
        result = _classify_advisory("USN-6543-1")
        assert result["prefix"] == "USN"
        assert result["database"] == "Ubuntu Security Notice"

    def test_pysec_prefix(self):
        result = _classify_advisory("PYSEC-2021-123")
        assert result["prefix"] == "PYSEC"
        assert result["database"] == "Python Security Advisory (OSV)"

    def test_rustsec_prefix(self):
        result = _classify_advisory("RUSTSEC-2023-0071")
        assert result["prefix"] == "RUSTSEC"
        assert result["database"] == "Rust Security Advisory"

    def test_go_prefix(self):
        result = _classify_advisory("GO-2023-1234")
        assert result["prefix"] == "GO"
        assert result["database"] == "Go Vulnerability"

    def test_cve_prefix(self):
        result = _classify_advisory("CVE-2021-44228")
        assert result["prefix"] == "CVE"
        assert result["database"] == "Common Vulnerabilities and Exposures"
        assert "nvd.nist.gov" in result["url"]

    def test_unknown_prefix(self):
        result = _classify_advisory("CUSTOM-2021-001")
        assert result["database"] == "Unknown"
        assert result["prefix"] == "CUSTOM"
        assert "osv.dev" in result["url"]

    def test_alas_prefix(self):
        result = _classify_advisory("ALAS-2024-001")
        assert result["prefix"] == "ALAS"
        assert result["database"] == "Amazon Linux Security Advisory"

    def test_hsec_prefix(self):
        result = _classify_advisory("HSEC-2023-0001")
        assert result["prefix"] == "HSEC"
        assert result["database"] == "Haskell Security Advisory"

    def test_mal_prefix(self):
        result = _classify_advisory("MAL-2024-1234")
        assert result["prefix"] == "MAL"
        assert result["database"] == "Malicious Package Advisory"

    def test_no_dash_in_id(self):
        result = _classify_advisory("UNKNOWN123")
        assert result["database"] == "Unknown"
        assert result["prefix"] == "UNKNOWN123"


# ---------------------------------------------------------------------------
# Tests: _build_advisory_url
# ---------------------------------------------------------------------------


class TestBuildAdvisoryUrl:
    def test_ghsa_url(self):
        url = _build_advisory_url("GHSA-jfh8-c2jp-5v3q", "GHSA")
        assert url == "https://github.com/advisories/GHSA-jfh8-c2jp-5v3q"

    def test_rhsa_url(self):
        url = _build_advisory_url("RHSA-2024:1234", "RHSA")
        assert url == "https://access.redhat.com/errata/RHSA-2024:1234"

    def test_dsa_url_high_number(self):
        url = _build_advisory_url("DSA-5432-1", "DSA")
        assert "2023" in url
        assert "dsa-5432-1" in url

    def test_dsa_url_mid_number(self):
        url = _build_advisory_url("DSA-4100-1", "DSA")
        assert "2019" in url

    def test_dsa_url_low_number(self):
        url = _build_advisory_url("DSA-3100-1", "DSA")
        assert "2016" in url

    def test_dsa_url_very_low_number(self):
        url = _build_advisory_url("DSA-2500-1", "DSA")
        assert "2014" in url

    def test_dsa_no_match(self):
        url = _build_advisory_url("DSA-invalid", "DSA")
        assert "osv.dev" in url

    def test_usn_url(self):
        url = _build_advisory_url("USN-6543-1", "USN")
        assert url == "https://ubuntu.com/security/notices/USN-6543-1"

    def test_pysec_url(self):
        url = _build_advisory_url("PYSEC-2021-123", "PYSEC")
        assert url == "https://osv.dev/vulnerability/PYSEC-2021-123"

    def test_rustsec_url(self):
        url = _build_advisory_url("RUSTSEC-2023-0071", "RUSTSEC")
        assert url == "https://rustsec.org/advisories/RUSTSEC-2023-0071.html"

    def test_go_url(self):
        url = _build_advisory_url("GO-2023-1234", "GO")
        assert url == "https://pkg.go.dev/vuln/GO-2023-1234"

    def test_unknown_prefix_fallback(self):
        url = _build_advisory_url("CUSTOM-001", "CUSTOM")
        assert url == "https://osv.dev/vulnerability/CUSTOM-001"


# ---------------------------------------------------------------------------
# Tests: _extract_advisory_from_url
# ---------------------------------------------------------------------------


class TestExtractAdvisoryFromUrl:
    def test_ghsa_url(self):
        url = "https://github.com/advisories/GHSA-jfh8-c2jp-5v3q"
        assert _extract_advisory_from_url(url) == "GHSA-jfh8-c2jp-5v3q"

    def test_rhsa_url(self):
        url = "https://access.redhat.com/errata/RHSA-2024:1234"
        assert _extract_advisory_from_url(url) == "RHSA-2024:1234"

    def test_rhba_url(self):
        url = "https://access.redhat.com/errata/RHBA-2024:5678"
        assert _extract_advisory_from_url(url) == "RHBA-2024:5678"

    def test_dsa_url(self):
        url = "https://www.debian.org/security/2023/dsa-5432"
        result = _extract_advisory_from_url(url)
        assert result == "DSA-5432"

    def test_usn_url(self):
        url = "https://ubuntu.com/security/notices/USN-6543-1"
        assert _extract_advisory_from_url(url) == "USN-6543-1"

    def test_suse_url(self):
        url = "https://www.suse.com/support/update/announcement/2024/SUSE-SU-2024:1234-1"
        assert _extract_advisory_from_url(url) == "SUSE-SU-2024:1234-1"

    def test_unrecognized_url(self):
        url = "https://example.com/advisory/123"
        assert _extract_advisory_from_url(url) is None

    def test_empty_url(self):
        assert _extract_advisory_from_url("") is None

    def test_github_non_advisory(self):
        url = "https://github.com/user/repo/issues/123"
        assert _extract_advisory_from_url(url) is None


# ---------------------------------------------------------------------------
# Tests: _get_with_retry
# ---------------------------------------------------------------------------


class TestGetWithRetry:
    @patch("manus_agent.tools.resolve_advisory_aliases.requests.get")
    def test_success_first_attempt(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.resolve_advisory_aliases.time.sleep")
    @patch("manus_agent.tools.resolve_advisory_aliases.requests.get")
    def test_retry_on_429(self, mock_get, mock_sleep):
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_get.side_effect = [mock_429, mock_200]

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    @patch("manus_agent.tools.resolve_advisory_aliases.time.sleep")
    @patch("manus_agent.tools.resolve_advisory_aliases.requests.get")
    def test_retry_on_connection_error(self, mock_get, mock_sleep):
        import requests

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_get.side_effect = [requests.ConnectionError("fail"), mock_200]

        result = _get_with_retry("https://example.com")
        assert result.status_code == 200

    @patch("manus_agent.tools.resolve_advisory_aliases.time.sleep")
    @patch("manus_agent.tools.resolve_advisory_aliases.requests.get")
    def test_raises_after_max_retries(self, mock_get, mock_sleep):
        import requests

        mock_get.side_effect = requests.Timeout("timeout")

        with pytest.raises(requests.Timeout):
            _get_with_retry("https://example.com")
        assert mock_get.call_count == 3


# ---------------------------------------------------------------------------
# Tests: _post_with_retry
# ---------------------------------------------------------------------------


class TestPostWithRetry:
    @patch("manus_agent.tools.resolve_advisory_aliases.requests.post")
    def test_success_first_attempt(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        result = _post_with_retry("https://example.com", {"key": "val"})
        assert result.status_code == 200

    @patch("manus_agent.tools.resolve_advisory_aliases.time.sleep")
    @patch("manus_agent.tools.resolve_advisory_aliases.requests.post")
    def test_retry_on_429(self, mock_post, mock_sleep):
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_post.side_effect = [mock_429, mock_200]

        result = _post_with_retry("https://example.com", {})
        assert result.status_code == 200
        assert mock_post.call_count == 2


# ---------------------------------------------------------------------------
# Tests: _query_osv_aliases
# ---------------------------------------------------------------------------


class TestQueryOsvAliases:
    @patch("manus_agent.tools.resolve_advisory_aliases._post_with_retry")
    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    def test_direct_lookup_with_aliases(self, mock_get, mock_post):
        # Direct lookup returns data
        direct_resp = MagicMock()
        direct_resp.status_code = 200
        direct_resp.json.return_value = {
            "id": "CVE-2021-44228",
            "aliases": ["GHSA-jfh8-c2jp-5v3q", "PYSEC-2021-1234"],
            "summary": "Log4Shell",
            "modified": "2023-01-01T00:00:00Z",
            "affected": [{"package": {"ecosystem": "Maven", "name": "org.apache.logging.log4j:log4j-core"}}],
        }
        mock_get.return_value = direct_resp

        # Query returns no additional vulns
        query_resp = MagicMock()
        query_resp.status_code = 200
        query_resp.json.return_value = {"vulns": []}
        mock_post.return_value = query_resp

        result = _query_osv_aliases("CVE-2021-44228")
        assert "GHSA-jfh8-c2jp-5v3q" in result["aliases"]
        assert "PYSEC-2021-1234" in result["aliases"]
        assert len(result["affected_packages"]) == 1
        assert result["affected_packages"][0]["ecosystem"] == "Maven"

    @patch("manus_agent.tools.resolve_advisory_aliases._post_with_retry")
    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    def test_query_returns_additional_vulns(self, mock_get, mock_post):
        # Direct lookup returns 404
        direct_resp = MagicMock()
        direct_resp.status_code = 404
        mock_get.return_value = direct_resp

        # Query returns multiple vulns
        query_resp = MagicMock()
        query_resp.status_code = 200
        query_resp.json.return_value = {
            "vulns": [
                {
                    "id": "GHSA-abcd-efgh-ijkl",
                    "aliases": ["CVE-2021-44228", "PYSEC-2021-999"],
                    "affected": [{"package": {"ecosystem": "PyPI", "name": "some-package"}}],
                },
                {
                    "id": "RUSTSEC-2021-0145",
                    "aliases": ["CVE-2021-44228"],
                    "affected": [],
                },
            ]
        }
        mock_post.return_value = query_resp

        result = _query_osv_aliases("CVE-2021-44228")
        assert "GHSA-abcd-efgh-ijkl" in result["aliases"]
        assert "PYSEC-2021-999" in result["aliases"]
        assert "RUSTSEC-2021-0145" in result["aliases"]
        assert any(p["ecosystem"] == "PyPI" for p in result["affected_packages"])

    @patch("manus_agent.tools.resolve_advisory_aliases._post_with_retry")
    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    def test_both_fail_gracefully(self, mock_get, mock_post):
        import requests

        mock_get.side_effect = requests.ConnectionError("fail")
        mock_post.side_effect = requests.Timeout("timeout")

        result = _query_osv_aliases("CVE-2021-44228")
        assert result["aliases"] == []
        assert result["affected_packages"] == []

    @patch("manus_agent.tools.resolve_advisory_aliases._post_with_retry")
    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    def test_deduplicates_packages(self, mock_get, mock_post):
        direct_resp = MagicMock()
        direct_resp.status_code = 200
        direct_resp.json.return_value = {
            "id": "CVE-2021-44228",
            "aliases": [],
            "affected": [{"package": {"ecosystem": "PyPI", "name": "pkg1"}}],
        }
        mock_get.return_value = direct_resp

        query_resp = MagicMock()
        query_resp.status_code = 200
        query_resp.json.return_value = {
            "vulns": [
                {
                    "id": "PYSEC-2021-123",
                    "aliases": [],
                    "affected": [{"package": {"ecosystem": "PyPI", "name": "pkg1"}}],
                }
            ]
        }
        mock_post.return_value = query_resp

        result = _query_osv_aliases("CVE-2021-44228")
        # Same package should not be duplicated
        pypi_pkgs = [p for p in result["affected_packages"] if p["name"] == "pkg1"]
        assert len(pypi_pkgs) == 1


# ---------------------------------------------------------------------------
# Tests: _extract_nvd_advisory_refs
# ---------------------------------------------------------------------------


class TestExtractNvdAdvisoryRefs:
    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    def test_extracts_ghsa_from_nvd(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "references": [
                            {
                                "url": "https://github.com/advisories/GHSA-jfh8-c2jp-5v3q",
                                "tags": ["Vendor Advisory"],
                            },
                            {
                                "url": "https://example.com/other",
                                "tags": ["Third Party Advisory"],
                            },
                        ]
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        refs = _extract_nvd_advisory_refs("CVE-2021-44228")
        assert len(refs) == 1
        assert refs[0]["id"] == "GHSA-jfh8-c2jp-5v3q"
        assert refs[0]["url"] == "https://github.com/advisories/GHSA-jfh8-c2jp-5v3q"

    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    def test_extracts_multiple_advisories(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "references": [
                            {"url": "https://github.com/advisories/GHSA-abcd-1234-5678", "tags": []},
                            {"url": "https://access.redhat.com/errata/RHSA-2024:1234", "tags": []},
                            {"url": "https://ubuntu.com/security/notices/USN-6000-1", "tags": []},
                        ]
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        refs = _extract_nvd_advisory_refs("CVE-2024-1234")
        assert len(refs) == 3
        ids = [r["id"] for r in refs]
        assert "GHSA-abcd-1234-5678" in ids
        assert "RHSA-2024:1234" in ids
        assert "USN-6000-1" in ids

    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    def test_handles_404(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        refs = _extract_nvd_advisory_refs("CVE-9999-99999")
        assert refs == []

    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    def test_handles_empty_vulns(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        refs = _extract_nvd_advisory_refs("CVE-2021-44228")
        assert refs == []

    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    def test_handles_request_exception(self, mock_get):
        import requests

        mock_get.side_effect = requests.ConnectionError("fail")
        refs = _extract_nvd_advisory_refs("CVE-2021-44228")
        assert refs == []

    @patch("manus_agent.tools.resolve_advisory_aliases.os.environ", {"NVD_API_KEY": "test-key"})
    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    def test_uses_api_key_header(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        _extract_nvd_advisory_refs("CVE-2021-44228")
        call_args = mock_get.call_args
        assert call_args[1]["headers"]["apiKey"] == "test-key"


# ---------------------------------------------------------------------------
# Tests: _query_vulncheck_aliases
# ---------------------------------------------------------------------------


class TestQueryVulncheckAliases:
    @patch("manus_agent.tools.resolve_advisory_aliases.os.environ", {})
    def test_returns_empty_without_api_key(self):
        result = _query_vulncheck_aliases("CVE-2021-44228")
        assert result == []

    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    @patch(
        "manus_agent.tools.resolve_advisory_aliases.os.environ",
        {"VULNCHECK_API_KEY": "test-key"},
    )
    def test_extracts_refs_from_vulncheck(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "references": [
                        "https://github.com/advisories/GHSA-xxxx-yyyy-zzzz",
                        "https://access.redhat.com/errata/RHSA-2022:1234",
                        "https://example.com/no-match",
                    ]
                }
            ]
        }
        mock_get.return_value = mock_resp

        result = _query_vulncheck_aliases("CVE-2021-44228")
        assert "GHSA-xxxx-yyyy-zzzz" in result
        assert "RHSA-2022:1234" in result
        assert len(result) == 2

    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    @patch(
        "manus_agent.tools.resolve_advisory_aliases.os.environ",
        {"VULNCHECK_API_KEY": "test-key"},
    )
    def test_handles_api_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_get.return_value = mock_resp

        result = _query_vulncheck_aliases("CVE-2021-44228")
        assert result == []

    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    @patch(
        "manus_agent.tools.resolve_advisory_aliases.os.environ",
        {"VULNCHECK_API_KEY": "test-key"},
    )
    def test_handles_network_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.Timeout("timeout")
        result = _query_vulncheck_aliases("CVE-2021-44228")
        assert result == []

    @patch("manus_agent.tools.resolve_advisory_aliases._get_with_retry")
    @patch(
        "manus_agent.tools.resolve_advisory_aliases.os.environ",
        {"VULNCHECK_API_KEY": "test-key"},
    )
    def test_deduplicates_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "references": [
                        "https://github.com/advisories/GHSA-xxxx-yyyy-zzzz",
                        "https://github.com/advisories/GHSA-xxxx-yyyy-zzzz",
                    ]
                }
            ]
        }
        mock_get.return_value = mock_resp

        result = _query_vulncheck_aliases("CVE-2021-44228")
        assert result.count("GHSA-xxxx-yyyy-zzzz") == 1


# ---------------------------------------------------------------------------
# Tests: fetch_advisory_aliases (standalone function)
# ---------------------------------------------------------------------------


class TestFetchAdvisoryAliases:
    @patch("manus_agent.tools.resolve_advisory_aliases._query_vulncheck_aliases")
    @patch("manus_agent.tools.resolve_advisory_aliases._extract_nvd_advisory_refs")
    @patch("manus_agent.tools.resolve_advisory_aliases._query_osv_aliases")
    def test_combines_all_sources(self, mock_osv, mock_nvd, mock_vc):
        mock_osv.return_value = {
            "aliases": ["GHSA-jfh8-c2jp-5v3q", "PYSEC-2021-123"],
            "affected_packages": [{"ecosystem": "Maven", "name": "log4j-core"}],
            "osv_entries": [],
        }
        mock_nvd.return_value = [
            {"id": "RHSA-2024:1234", "url": "https://access.redhat.com/errata/RHSA-2024:1234", "tags": []}
        ]
        mock_vc.return_value = ["USN-6543-1"]

        result = fetch_advisory_aliases("CVE-2021-44228")

        assert result["cve_id"] == "CVE-2021-44228"
        assert result["total_aliases"] == 4  # GHSA + PYSEC + RHSA + USN
        assert "osv.dev" in result["sources_queried"]
        assert "nvd" in result["sources_queried"]
        assert "vulncheck" in result["sources_queried"]

        # Verify grouping by database
        by_db = result["aliases_by_database"]
        assert "GitHub Security Advisory" in by_db
        assert "Python Security Advisory (OSV)" in by_db
        assert "Red Hat Security Advisory" in by_db
        assert "Ubuntu Security Notice" in by_db

    @patch("manus_agent.tools.resolve_advisory_aliases._query_vulncheck_aliases")
    @patch("manus_agent.tools.resolve_advisory_aliases._extract_nvd_advisory_refs")
    @patch("manus_agent.tools.resolve_advisory_aliases._query_osv_aliases")
    def test_no_aliases_found(self, mock_osv, mock_nvd, mock_vc):
        mock_osv.return_value = {
            "aliases": [],
            "affected_packages": [],
            "osv_entries": [],
        }
        mock_nvd.return_value = []
        mock_vc.return_value = []

        result = fetch_advisory_aliases("CVE-9999-99999")
        assert result["total_aliases"] == 0
        assert result["databases_found"] == 0
        assert result["aliases_by_database"] == {}

    @patch("manus_agent.tools.resolve_advisory_aliases._query_vulncheck_aliases")
    @patch("manus_agent.tools.resolve_advisory_aliases._extract_nvd_advisory_refs")
    @patch("manus_agent.tools.resolve_advisory_aliases._query_osv_aliases")
    def test_osv_error_continues(self, mock_osv, mock_nvd, mock_vc):
        mock_osv.side_effect = Exception("OSV down")
        mock_nvd.return_value = [
            {"id": "GHSA-test-1234-5678", "url": "https://github.com/advisories/GHSA-test-1234-5678", "tags": []}
        ]
        mock_vc.return_value = []

        result = fetch_advisory_aliases("CVE-2021-44228")
        assert result["total_aliases"] == 1
        assert "errors" in result
        assert any("OSV.dev" in e for e in result["errors"])

    @patch("manus_agent.tools.resolve_advisory_aliases._query_vulncheck_aliases")
    @patch("manus_agent.tools.resolve_advisory_aliases._extract_nvd_advisory_refs")
    @patch("manus_agent.tools.resolve_advisory_aliases._query_osv_aliases")
    def test_include_urls_false(self, mock_osv, mock_nvd, mock_vc):
        mock_osv.return_value = {
            "aliases": ["GHSA-jfh8-c2jp-5v3q"],
            "affected_packages": [],
            "osv_entries": [],
        }
        mock_nvd.return_value = []
        mock_vc.return_value = []

        result = fetch_advisory_aliases("CVE-2021-44228", include_urls=False)
        for entries in result["aliases_by_database"].values():
            for entry in entries:
                assert "url" not in entry

    @patch("manus_agent.tools.resolve_advisory_aliases._query_vulncheck_aliases")
    @patch("manus_agent.tools.resolve_advisory_aliases._extract_nvd_advisory_refs")
    @patch("manus_agent.tools.resolve_advisory_aliases._query_osv_aliases")
    def test_deduplication_across_sources(self, mock_osv, mock_nvd, mock_vc):
        mock_osv.return_value = {
            "aliases": ["GHSA-jfh8-c2jp-5v3q"],
            "affected_packages": [],
            "osv_entries": [],
        }
        # Same GHSA found via NVD
        mock_nvd.return_value = [
            {"id": "GHSA-jfh8-c2jp-5v3q", "url": "https://github.com/advisories/GHSA-jfh8-c2jp-5v3q", "tags": []}
        ]
        mock_vc.return_value = ["GHSA-jfh8-c2jp-5v3q"]

        result = fetch_advisory_aliases("CVE-2021-44228")
        # Should only appear once despite being in all sources
        assert result["total_aliases"] == 1

    @patch("manus_agent.tools.resolve_advisory_aliases._query_vulncheck_aliases")
    @patch("manus_agent.tools.resolve_advisory_aliases._extract_nvd_advisory_refs")
    @patch("manus_agent.tools.resolve_advisory_aliases._query_osv_aliases")
    def test_caps_affected_packages_at_20(self, mock_osv, mock_nvd, mock_vc):
        mock_osv.return_value = {
            "aliases": [],
            "affected_packages": [{"ecosystem": "PyPI", "name": f"pkg-{i}"} for i in range(30)],
            "osv_entries": [],
        }
        mock_nvd.return_value = []
        mock_vc.return_value = []

        result = fetch_advisory_aliases("CVE-2021-44228")
        assert len(result["affected_packages"]) == 20


# ---------------------------------------------------------------------------
# Tests: resolve_advisory_aliases (Strands tool interface)
# ---------------------------------------------------------------------------


class TestResolveAdvisoryAliasesTool:
    @patch("manus_agent.tools.resolve_advisory_aliases.fetch_advisory_aliases")
    def test_success(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "total_aliases": 3,
            "databases_found": 2,
            "sources_queried": ["osv.dev", "nvd"],
            "aliases_by_database": {"GitHub Security Advisory": [{"id": "GHSA-jfh8-c2jp-5v3q", "source": "osv.dev"}]},
            "affected_packages": [],
        }

        tool = _make_tool("CVE-2021-44228")
        result = resolve_advisory_aliases(tool)

        assert result["status"] == "success"
        content = json.loads(result["content"][0]["text"])
        assert content["cve_id"] == "CVE-2021-44228"
        assert content["total_aliases"] == 3

    def test_empty_cve_id(self):
        tool = _make_tool("")
        result = resolve_advisory_aliases(tool)
        assert result["status"] == "error"
        assert "required" in result["content"][0]["text"]

    def test_invalid_cve_format(self):
        tool = _make_tool("INVALID-123")
        result = resolve_advisory_aliases(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID format" in result["content"][0]["text"]

    def test_cve_format_too_short(self):
        tool = _make_tool("CVE-2021-12")
        result = resolve_advisory_aliases(tool)
        assert result["status"] == "error"

    @patch("manus_agent.tools.resolve_advisory_aliases.fetch_advisory_aliases")
    def test_case_normalization(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "total_aliases": 0,
            "databases_found": 0,
            "sources_queried": [],
            "aliases_by_database": {},
            "affected_packages": [],
        }

        tool = _make_tool("cve-2021-44228")
        result = resolve_advisory_aliases(tool)
        assert result["status"] == "success"
        # Verify the CVE was uppercased
        mock_fetch.assert_called_once_with("CVE-2021-44228", include_urls=True)

    @patch("manus_agent.tools.resolve_advisory_aliases.fetch_advisory_aliases")
    def test_whitespace_stripped(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "total_aliases": 0,
            "databases_found": 0,
            "sources_queried": [],
            "aliases_by_database": {},
            "affected_packages": [],
        }

        tool = _make_tool("  CVE-2021-44228  ")
        result = resolve_advisory_aliases(tool)
        assert result["status"] == "success"

    @patch("manus_agent.tools.resolve_advisory_aliases.fetch_advisory_aliases")
    def test_include_urls_false(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "total_aliases": 0,
            "databases_found": 0,
            "sources_queried": [],
            "aliases_by_database": {},
            "affected_packages": [],
        }

        tool = _make_tool("CVE-2021-44228", include_urls=False)
        resolve_advisory_aliases(tool)
        mock_fetch.assert_called_once_with("CVE-2021-44228", include_urls=False)

    @patch("manus_agent.tools.resolve_advisory_aliases.fetch_advisory_aliases")
    def test_tool_use_id_propagated(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "total_aliases": 0,
            "databases_found": 0,
            "sources_queried": [],
            "aliases_by_database": {},
            "affected_packages": [],
        }

        tool = _make_tool("CVE-2021-44228")
        result = resolve_advisory_aliases(tool)
        assert result["toolUseId"] == "test-tool-id-123"


# ---------------------------------------------------------------------------
# Tests: CLI subcommand (_run_advisory_aliases)
# ---------------------------------------------------------------------------


class TestCliAdvisoryAliases:
    @patch("manus_agent.tools.resolve_advisory_aliases.fetch_advisory_aliases")
    def test_json_output(self, mock_fetch, capsys):
        from manus_agent.cli import _run_advisory_aliases

        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "total_aliases": 2,
            "databases_found": 1,
            "sources_queried": ["osv.dev"],
            "aliases_by_database": {
                "GitHub Security Advisory": [
                    {
                        "id": "GHSA-jfh8-c2jp-5v3q",
                        "url": "https://github.com/advisories/GHSA-jfh8-c2jp-5v3q",
                        "source": "osv.dev",
                    }
                ]
            },
            "affected_packages": [{"ecosystem": "Maven", "name": "log4j-core"}],
        }

        exit_code = _run_advisory_aliases(["CVE-2021-44228", "--output", "json"])
        assert exit_code == 0

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["cve_id"] == "CVE-2021-44228"
        assert output["total_aliases"] == 2

    @patch("manus_agent.tools.resolve_advisory_aliases.fetch_advisory_aliases")
    def test_text_output(self, mock_fetch, capsys):
        from manus_agent.cli import _run_advisory_aliases

        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "total_aliases": 2,
            "databases_found": 2,
            "sources_queried": ["osv.dev", "nvd"],
            "aliases_by_database": {
                "GitHub Security Advisory": [
                    {
                        "id": "GHSA-jfh8-c2jp-5v3q",
                        "url": "https://github.com/advisories/GHSA-jfh8-c2jp-5v3q",
                        "source": "osv.dev",
                    }
                ],
                "Red Hat Security Advisory": [
                    {"id": "RHSA-2024:1234", "url": "https://access.redhat.com/errata/RHSA-2024:1234", "source": "nvd"}
                ],
            },
            "affected_packages": [{"ecosystem": "Maven", "name": "log4j-core"}],
        }

        exit_code = _run_advisory_aliases(["CVE-2021-44228"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Advisory Aliases for CVE-2021-44228" in captured.out
        assert "GHSA-jfh8-c2jp-5v3q" in captured.out
        assert "RHSA-2024:1234" in captured.out
        assert "Maven:log4j-core" in captured.out

    @patch("manus_agent.tools.resolve_advisory_aliases.fetch_advisory_aliases")
    def test_text_output_no_aliases(self, mock_fetch, capsys):
        from manus_agent.cli import _run_advisory_aliases

        mock_fetch.return_value = {
            "cve_id": "CVE-9999-99999",
            "total_aliases": 0,
            "databases_found": 0,
            "sources_queried": ["osv.dev", "nvd"],
            "aliases_by_database": {},
            "affected_packages": [],
        }

        exit_code = _run_advisory_aliases(["CVE-9999-99999"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "No advisory aliases found" in captured.out

    @patch("manus_agent.tools.resolve_advisory_aliases.fetch_advisory_aliases")
    def test_no_urls_flag(self, mock_fetch, capsys):
        from manus_agent.cli import _run_advisory_aliases

        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "total_aliases": 1,
            "databases_found": 1,
            "sources_queried": ["osv.dev"],
            "aliases_by_database": {"GitHub Security Advisory": [{"id": "GHSA-jfh8-c2jp-5v3q", "source": "osv.dev"}]},
            "affected_packages": [],
        }

        exit_code = _run_advisory_aliases(["CVE-2021-44228", "--no-urls"])
        assert exit_code == 0
        mock_fetch.assert_called_once_with("CVE-2021-44228", include_urls=False)

    @patch("manus_agent.tools.resolve_advisory_aliases.fetch_advisory_aliases")
    def test_text_output_with_errors(self, mock_fetch, capsys):
        from manus_agent.cli import _run_advisory_aliases

        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "total_aliases": 0,
            "databases_found": 0,
            "sources_queried": ["nvd"],
            "aliases_by_database": {},
            "affected_packages": [],
            "errors": ["OSV.dev: Connection timeout"],
        }

        exit_code = _run_advisory_aliases(["CVE-2021-44228"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Warnings:" in captured.out
        assert "OSV.dev: Connection timeout" in captured.out

    @patch("manus_agent.tools.resolve_advisory_aliases.fetch_advisory_aliases")
    def test_many_affected_packages_truncated(self, mock_fetch, capsys):
        from manus_agent.cli import _run_advisory_aliases

        mock_fetch.return_value = {
            "cve_id": "CVE-2021-44228",
            "total_aliases": 1,
            "databases_found": 1,
            "sources_queried": ["osv.dev"],
            "aliases_by_database": {"Go Vulnerability": [{"id": "GO-2021-0001", "source": "osv.dev"}]},
            "affected_packages": [{"ecosystem": "Go", "name": f"pkg{i}"} for i in range(15)],
        }

        exit_code = _run_advisory_aliases(["CVE-2021-44228"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "... and 5 more" in captured.out


# ---------------------------------------------------------------------------
# Tests: ADVISORY_PREFIXES constant coverage
# ---------------------------------------------------------------------------


class TestAdvisoryPrefixes:
    def test_all_prefixes_have_string_values(self):
        for prefix, name in ADVISORY_PREFIXES.items():
            assert isinstance(prefix, str)
            assert isinstance(name, str)
            assert len(name) > 0

    def test_url_templates_cover_most_prefixes(self):
        # Not all prefixes need URL templates (SNYK, ALAS don't have simple templates)
        assert "GHSA" in ADVISORY_URL_TEMPLATES
        assert "RHSA" in ADVISORY_URL_TEMPLATES
        assert "CVE" in ADVISORY_URL_TEMPLATES
        assert "PYSEC" in ADVISORY_URL_TEMPLATES
        assert "GO" in ADVISORY_URL_TEMPLATES
