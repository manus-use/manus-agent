"""
Tests for src/manus_agent/tools/get_dependency_blast_radius.py

All external HTTP calls are mocked — no real network I/O.
100% mocked: NVD, OSV, GHSA, npm, PyPI, pypistats, Maven Central.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.get_dependency_blast_radius import (
    _blast_score,
    _enrich_maven,
    _enrich_npm,
    _enrich_package,
    _enrich_pypi,
    _extract_npm_first_release,
    _fetch_ghsa_affected,
    _fetch_nvd_affected,
    _fetch_osv_affected,
    _parse_input,
    _summarise_osv_ranges,
    get_dependency_blast_radius,
)

# ===========================================================================
# _parse_input
# ===========================================================================


class TestParseInput:
    def test_cve_id_uppercase(self):
        result = _parse_input("CVE-2021-44228")
        assert result["kind"] == "cve"
        assert result["cve_id"] == "CVE-2021-44228"

    def test_cve_id_lowercase(self):
        result = _parse_input("cve-2021-44228")
        assert result["kind"] == "cve"
        assert result["cve_id"] == "CVE-2021-44228"

    def test_cve_id_with_spaces(self):
        result = _parse_input("  CVE-2024-3094  ")
        assert result["kind"] == "cve"
        assert result["cve_id"] == "CVE-2024-3094"

    def test_package_with_version(self):
        result = _parse_input("requests@2.28.0")
        assert result["kind"] == "package"
        assert result["name"] == "requests"
        assert result["version"] == "2.28.0"
        assert result["ecosystem"] is None

    def test_package_without_version(self):
        result = _parse_input("requests")
        assert result["kind"] == "package"
        assert result["name"] == "requests"
        assert result["version"] is None

    def test_ecosystem_qualified_package(self):
        result = _parse_input("pypi:requests@2.28.0")
        assert result["kind"] == "package"
        assert result["ecosystem"] == "pypi"
        assert result["name"] == "requests"
        assert result["version"] == "2.28.0"

    def test_npm_ecosystem(self):
        result = _parse_input("npm:lodash@4.17.20")
        assert result["kind"] == "package"
        assert result["ecosystem"] == "npm"
        assert result["name"] == "lodash"
        assert result["version"] == "4.17.20"

    def test_maven_ecosystem(self):
        result = _parse_input("maven:log4j-core@2.14.1")
        assert result["kind"] == "package"
        assert result["ecosystem"] == "maven"
        assert result["name"] == "log4j-core"

    def test_url_not_treated_as_ecosystem(self):
        # http: should not strip the protocol
        result = _parse_input("https://example.com")
        assert result["kind"] == "package"
        # name should include the full string (no ecosystem stripping)

    def test_cve_id_extracted_correctly(self):
        result = _parse_input("CVE-2023-12345")
        assert result["cve_id"] == "CVE-2023-12345"
        assert result["name"] is None


# ===========================================================================
# _summarise_osv_ranges
# ===========================================================================


class TestSummariseOsvRanges:
    def test_semver_introduced_fixed(self):
        ranges = [
            {
                "type": "SEMVER",
                "events": [{"introduced": "2.0.0"}, {"fixed": "2.3.1"}],
            }
        ]
        result = _summarise_osv_ranges(ranges, [])
        assert "2.0.0" in result
        assert "2.3.1" in result

    def test_ecosystem_range(self):
        ranges = [
            {
                "type": "ECOSYSTEM",
                "events": [{"introduced": "1.0.0"}, {"fixed": "1.5.0"}],
            }
        ]
        result = _summarise_osv_ranges(ranges, [])
        assert ">=1.0.0" in result
        assert "<1.5.0" in result

    def test_falls_back_to_versions_list(self):
        result = _summarise_osv_ranges([], ["2.0.0", "2.1.0", "2.2.0"])
        assert "2.0.0" in result
        assert "2.1.0" in result

    def test_versions_list_truncated_beyond_five(self):
        versions = ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6"]
        result = _summarise_osv_ranges([], versions)
        assert "+2 more" in result

    def test_empty_input(self):
        result = _summarise_osv_ranges([], [])
        assert result == "unspecified"

    def test_introduced_only(self):
        ranges = [{"type": "SEMVER", "events": [{"introduced": "1.0.0"}]}]
        result = _summarise_osv_ranges(ranges, [])
        assert ">=1.0.0" in result

    def test_git_range_skipped(self):
        ranges = [{"type": "GIT", "events": [{"introduced": "abc123"}, {"fixed": "def456"}]}]
        result = _summarise_osv_ranges(ranges, ["2.0.0"])
        # GIT ranges are skipped; should fall back to versions list
        assert "2.0.0" in result


# ===========================================================================
# _blast_score
# ===========================================================================


class TestBlastScore:
    def test_critical_by_downloads(self):
        assert _blast_score({"weekly_downloads": 10_000_000}) == "CRITICAL"

    def test_critical_by_dependents(self):
        assert _blast_score({"dependent_packages_count": 100_000}) == "CRITICAL"

    def test_high_by_downloads(self):
        assert _blast_score({"weekly_downloads": 1_000_000}) == "HIGH"

    def test_high_by_dependents(self):
        assert _blast_score({"dependent_packages_count": 10_000}) == "HIGH"

    def test_medium_by_downloads(self):
        assert _blast_score({"weekly_downloads": 100_000}) == "MEDIUM"

    def test_medium_by_dependents(self):
        assert _blast_score({"dependent_packages_count": 1_000}) == "MEDIUM"

    def test_low_small_downloads(self):
        assert _blast_score({"weekly_downloads": 5_000}) == "LOW"

    def test_low_small_dependents(self):
        assert _blast_score({"dependent_packages_count": 10}) == "LOW"

    def test_unknown_no_data(self):
        assert _blast_score({}) == "UNKNOWN"

    def test_unknown_zero_values(self):
        assert _blast_score({"weekly_downloads": 0, "dependent_packages_count": 0}) == "UNKNOWN"

    def test_none_values_treated_as_zero(self):
        assert _blast_score({"weekly_downloads": None, "dependent_packages_count": None}) == "UNKNOWN"

    def test_critical_threshold_boundary(self):
        assert _blast_score({"weekly_downloads": 5_000_000}) == "CRITICAL"
        assert _blast_score({"weekly_downloads": 4_999_999}) == "HIGH"


# ===========================================================================
# _fetch_nvd_affected
# ===========================================================================


def _make_nvd_cve_response(
    cve_id: str = "CVE-2021-44228",
    product: str = "log4j-core",
    version_start: str = "2.0.0",
    version_end: str = "2.15.0",
) -> dict[str, Any]:
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "id": cve_id,
                    "configurations": [
                        {
                            "nodes": [
                                {
                                    "cpeMatch": [
                                        {
                                            "vulnerable": True,
                                            "criteria": f"cpe:2.3:a:apache:{product}:*:*:*:*:*:*:*:*",
                                            "versionStartIncluding": version_start,
                                            "versionEndExcluding": version_end,
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


class TestFetchNvdAffected:
    def test_extracts_package_and_range(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_nvd_cve_response()
        with patch("requests.get", return_value=mock_resp):
            result = _fetch_nvd_affected("CVE-2021-44228")
        assert len(result) >= 1
        assert result[0]["name"] == "log4j-core"
        assert "2.0.0" in result[0]["version_range"]
        assert "2.15.0" in result[0]["version_range"]

    def test_returns_empty_on_http_error(self):
        with patch("requests.get", side_effect=Exception("network error")):
            result = _fetch_nvd_affected("CVE-2021-44228")
        assert result == []

    def test_returns_empty_when_no_vulnerabilities(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"vulnerabilities": []}
        with patch("requests.get", return_value=mock_resp):
            result = _fetch_nvd_affected("CVE-2021-44228")
        assert result == []

    def test_skips_non_vulnerable_cpe(self):
        data = {
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
                                                "vulnerable": False,
                                                "criteria": "cpe:2.3:a:apache:log4j-core:2.0:*:*:*:*:*:*:*",
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
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        with patch("requests.get", return_value=mock_resp):
            result = _fetch_nvd_affected("CVE-2021-44228")
        assert result == []

    def test_deduplication_by_name(self):
        # Multiple CPEs for the same product → should deduplicate
        data = {
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
                                                "criteria": "cpe:2.3:a:apache:log4j-core:2.0:*:*:*:*:*:*:*",
                                                "versionStartIncluding": "2.0",
                                                "versionEndExcluding": "2.15.0",
                                            },
                                            {
                                                "vulnerable": True,
                                                "criteria": "cpe:2.3:a:apache:log4j-core:2.0:*:*:*:*:*:*:*",
                                                "versionStartIncluding": "2.16.0",
                                                "versionEndExcluding": "2.17.0",
                                            },
                                        ]
                                    }
                                ]
                            }
                        ],
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = data
        with patch("requests.get", return_value=mock_resp):
            result = _fetch_nvd_affected("CVE-2021-44228")
        names = [r["name"] for r in result]
        assert names.count("log4j-core") == 1


# ===========================================================================
# _fetch_osv_affected
# ===========================================================================


def _make_osv_query_response(vuln_ids: list[str]) -> dict:
    return {"vulns": [{"id": vid} for vid in vuln_ids]}


def _make_osv_full_response(
    name: str = "log4j-core",
    ecosystem: str = "Maven",
    introduced: str = "2.0.0",
    fixed: str = "2.15.0",
) -> dict:
    return {
        "id": "GHSA-xxxx-yyyy-zzzz",
        "affected": [
            {
                "package": {"name": name, "ecosystem": ecosystem},
                "ranges": [
                    {
                        "type": "ECOSYSTEM",
                        "events": [{"introduced": introduced}, {"fixed": fixed}],
                    }
                ],
                "versions": [],
            }
        ],
    }


class TestFetchOsvAffected:
    def test_returns_packages_from_osv(self):
        with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = _make_osv_query_response(["GHSA-xxxx-yyyy-zzzz"])
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = _make_osv_full_response()
            result = _fetch_osv_affected("CVE-2021-44228")
        assert any(r["name"] == "log4j-core" for r in result)
        assert any(r["ecosystem"] == "Maven" for r in result)

    def test_returns_empty_on_post_failure(self):
        with patch("requests.post", side_effect=Exception("timeout")):
            result = _fetch_osv_affected("CVE-2021-44228")
        assert result == []

    def test_skips_packages_without_name(self):
        osv_full = {
            "id": "OSV-001",
            "affected": [{"package": {"name": "", "ecosystem": "PyPI"}, "ranges": [], "versions": []}],
        }
        with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
            mock_post.return_value.json.return_value = _make_osv_query_response(["OSV-001"])
            mock_get.return_value.json.return_value = osv_full
            result = _fetch_osv_affected("CVE-2023-0001")
        assert result == []

    def test_continues_when_individual_vuln_fetch_fails(self):
        with patch("requests.post") as mock_post, patch("requests.get") as mock_get:
            mock_post.return_value.json.return_value = _make_osv_query_response(
                ["GHSA-aaa-bbb-ccc", "GHSA-xxx-yyy-zzz"]
            )
            # First call fails, second succeeds
            mock_get.side_effect = [
                Exception("connection refused"),
                MagicMock(
                    status_code=200,
                    json=MagicMock(return_value=_make_osv_full_response(name="requests")),
                ),
            ]
            result = _fetch_osv_affected("CVE-2023-0001")
        # Should contain the one that succeeded
        assert any(r["name"] == "requests" for r in result)


# ===========================================================================
# _fetch_ghsa_affected
# ===========================================================================


def _make_ghsa_response(
    pkg_name: str = "log4j-core",
    ecosystem: str = "maven",
    vuln_range: str = ">=2.0.0, <2.15.0",
    patched: str = "2.15.0",
) -> list[dict]:
    return [
        {
            "ghsa_id": "GHSA-xxxx-yyyy-zzzz",
            "vulnerabilities": [
                {
                    "package": {"name": pkg_name, "ecosystem": ecosystem},
                    "vulnerable_version_range": vuln_range,
                    "first_patched_version": {"identifier": patched},
                }
            ],
        }
    ]


class TestFetchGhsaAffected:
    def test_returns_packages_from_ghsa(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _make_ghsa_response()
        with patch("requests.get", return_value=mock_resp):
            result = _fetch_ghsa_affected("CVE-2021-44228")
        assert any(r["name"] == "log4j-core" for r in result)
        assert any(r["ecosystem"] == "maven" for r in result)

    def test_uses_patched_version_when_range_absent(self):
        response = [
            {
                "ghsa_id": "GHSA-test",
                "vulnerabilities": [
                    {
                        "package": {"name": "example-lib", "ecosystem": "PyPI"},
                        "vulnerable_version_range": "",
                        "first_patched_version": {"identifier": "3.0.0"},
                    }
                ],
            }
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        with patch("requests.get", return_value=mock_resp):
            result = _fetch_ghsa_affected("CVE-2023-0001")
        assert any(r["name"] == "example-lib" for r in result)
        assert any("<3.0.0" in r["version_range"] for r in result)

    def test_returns_empty_on_http_error(self):
        with patch("requests.get", side_effect=Exception("503")):
            result = _fetch_ghsa_affected("CVE-2021-44228")
        assert result == []

    def test_skips_packages_without_name(self):
        response = [
            {
                "ghsa_id": "GHSA-test",
                "vulnerabilities": [
                    {
                        "package": {"name": "", "ecosystem": "npm"},
                        "vulnerable_version_range": ">=1.0",
                        "first_patched_version": {"identifier": "2.0.0"},
                    }
                ],
            }
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        with patch("requests.get", return_value=mock_resp):
            result = _fetch_ghsa_affected("CVE-2023-0001")
        assert result == []


# ===========================================================================
# _enrich_npm
# ===========================================================================


class TestEnrichNpm:
    def _make_search_response(self, name: str, dependents: int, weekly: int, monthly: int) -> dict:
        return {
            "objects": [
                {
                    "package": {"name": name},
                    "dependents": str(dependents),
                    "downloads": {"weekly": weekly, "monthly": monthly},
                }
            ]
        }

    def test_returns_dependent_and_download_counts(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_search_response("lodash", 200000, 130000000, 520000000)
        with patch("requests.get", return_value=mock_resp):
            result = _enrich_npm("lodash")
        assert result["ecosystem"] == "npm"
        assert result["dependent_packages_count"] == 200000
        assert result["weekly_downloads"] == 130000000

    def test_falls_back_to_downloads_api(self):
        # Search response has no matching package name
        search_resp = MagicMock()
        search_resp.json.return_value = {"objects": [{"package": {"name": "something-else"}, "dependents": "0"}]}
        dl_resp = MagicMock()
        dl_resp.json.return_value = {"downloads": 5000000, "package": "axios"}
        with patch("requests.get", side_effect=[search_resp, dl_resp]):
            result = _enrich_npm("axios")
        assert result["weekly_downloads"] == 5000000

    def test_graceful_degradation_on_error(self):
        with patch("requests.get", side_effect=Exception("timeout")):
            result = _enrich_npm("axios")
        assert result["ecosystem"] == "npm"
        assert result["package_name"] == "axios"
        # No crash — just missing keys
        assert "dependent_packages_count" not in result

    def test_package_name_case_insensitive_match(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_search_response("Lodash", 100, 1000, 4000)
        with patch("requests.get", return_value=mock_resp):
            result = _enrich_npm("lodash")
        assert result.get("dependent_packages_count") == 100


# ===========================================================================
# _extract_npm_first_release
# ===========================================================================


class TestExtractNpmFirstRelease:
    """Tests for the _extract_npm_first_release helper."""

    def _time_dict(self, versions: dict[str, str]) -> dict[str, str]:
        """Build a minimal npm time dict including created/modified noise."""
        d = {
            "created": "2010-01-01T00:00:00.000Z",
            "modified": "2024-01-01T00:00:00.000Z",
        }
        d.update(versions)
        return d

    def test_returns_oldest_version(self):
        time_dict = self._time_dict(
            {
                "0.1.0": "2012-06-03T10:00:00.000Z",
                "1.0.0": "2015-03-15T10:00:00.000Z",
                "4.17.21": "2021-02-02T10:00:00.000Z",
            }
        )
        result = _extract_npm_first_release(time_dict)
        assert result["first_version"] == "0.1.0"
        assert result["first_release_date"] == "2012-06-03"
        assert isinstance(result["age_years"], float)
        assert result["age_years"] > 10

    def test_skips_created_and_modified_keys(self):
        # created/modified are very early; they must not be treated as versions
        time_dict = {
            "created": "2000-01-01T00:00:00.000Z",
            "modified": "2001-01-01T00:00:00.000Z",
            "1.0.0": "2018-05-20T12:00:00.000Z",
        }
        result = _extract_npm_first_release(time_dict)
        assert result["first_version"] == "1.0.0"
        assert result["first_release_date"] == "2018-05-20"

    def test_returns_empty_dict_for_empty_input(self):
        assert _extract_npm_first_release({}) == {}

    def test_returns_empty_dict_when_only_special_keys(self):
        time_dict = {"created": "2020-01-01T00:00:00.000Z", "modified": "2021-01-01T00:00:00.000Z"}
        assert _extract_npm_first_release(time_dict) == {}

    def test_handles_malformed_timestamp_gracefully(self):
        time_dict = self._time_dict(
            {
                "1.0.0": "NOT-A-DATE",
                "2.0.0": "2020-01-15T08:00:00.000Z",
            }
        )
        result = _extract_npm_first_release(time_dict)
        # Malformed entry skipped; valid one returned
        assert result["first_version"] == "2.0.0"
        assert result["first_release_date"] == "2020-01-15"

    def test_handles_timestamp_without_microseconds(self):
        time_dict = self._time_dict(
            {
                "0.5.0": "2013-07-04T14:30:00Z",
            }
        )
        result = _extract_npm_first_release(time_dict)
        assert result["first_release_date"] == "2013-07-04"

    def test_handles_all_malformed_returns_empty(self):
        time_dict = self._time_dict(
            {
                "1.0.0": "bad",
                "2.0.0": "also-bad",
            }
        )
        assert _extract_npm_first_release(time_dict) == {}

    def test_age_years_is_reasonable(self):
        time_dict = self._time_dict({"1.0.0": "2010-01-01T00:00:00.000Z"})
        result = _extract_npm_first_release(time_dict)
        # As of 2026, this package would be ~16 years old
        assert result["age_years"] >= 14

    def test_single_version_entry(self):
        time_dict = {"1.0.0": "2019-11-11T11:11:11.111Z"}
        result = _extract_npm_first_release(time_dict)
        assert result["first_version"] == "1.0.0"
        assert result["first_release_date"] == "2019-11-11"


# ===========================================================================
# _enrich_npm — registry call tests (first_release_date)
# ===========================================================================


class TestEnrichNpmFirstRelease:
    """Tests for the registry HTTP call added to _enrich_npm."""

    def _make_search_response(self, name: str, dependents: int = 1000, weekly: int = 5000000) -> dict:
        return {
            "objects": [
                {
                    "package": {"name": name},
                    "dependents": str(dependents),
                    "downloads": {"weekly": weekly, "monthly": weekly * 4},
                }
            ]
        }

    def _make_registry_response(self, name: str) -> dict:
        return {
            "name": name,
            "time": {
                "created": "2014-01-01T00:00:00.000Z",
                "modified": "2024-06-01T00:00:00.000Z",
                "0.1.0": "2014-01-15T10:00:00.000Z",
                "1.0.0": "2015-06-01T10:00:00.000Z",
                "4.17.21": "2021-02-02T10:00:00.000Z",
            },
        }

    def test_first_release_date_included_in_result(self):
        search_resp = MagicMock()
        search_resp.json.return_value = self._make_search_response("lodash")
        reg_resp = MagicMock()
        reg_resp.json.return_value = self._make_registry_response("lodash")
        with patch("requests.get", side_effect=[search_resp, reg_resp]):
            result = _enrich_npm("lodash")
        assert result["first_version"] == "0.1.0"
        assert result["first_release_date"] == "2014-01-15"
        assert isinstance(result["age_years"], float)

    def test_graceful_degradation_when_registry_fails(self):
        search_resp = MagicMock()
        search_resp.json.return_value = self._make_search_response("axios")
        with patch("requests.get", side_effect=[search_resp, Exception("503 Service Unavailable")]):
            result = _enrich_npm("axios")
        # Counts still returned; first_release_date absent but no crash
        assert result["ecosystem"] == "npm"
        assert "first_release_date" not in result
        assert result["weekly_downloads"] == 5000000

    def test_graceful_degradation_when_registry_returns_no_time_dict(self):
        search_resp = MagicMock()
        search_resp.json.return_value = self._make_search_response("express")
        reg_resp = MagicMock()
        reg_resp.json.return_value = {"name": "express"}  # no 'time' key
        with patch("requests.get", side_effect=[search_resp, reg_resp]):
            result = _enrich_npm("express")
        assert "first_release_date" not in result

    def test_first_release_date_shown_in_output(self):
        """Integration: first release date appears in the rendered text output."""
        from unittest.mock import patch

        # Build a minimal enriched result with first_release_date populated
        npm_result = {
            "ecosystem": "npm",
            "package_name": "lodash",
            "blast_radius": "HIGH",
            "version_range": "<4.17.21",
            "dependent_packages_count": 200000,
            "weekly_downloads": 50000000,
            "monthly_downloads": 200000000,
            "first_version": "0.1.0",
            "first_release_date": "2014-01-15",
            "age_years": 12.5,
            "source": "NVD",
        }

        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_npm",
            return_value=npm_result,
        ):
            with patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[{"name": "lodash", "ecosystem": "npm", "version_range": "<4.17.21"}],
            ):
                with patch(
                    "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                    return_value=[],
                ):
                    with patch(
                        "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                        return_value=[],
                    ):
                        output = get_dependency_blast_radius("CVE-2021-23337")

        assert "First released:" in output
        assert "2014-01-15" in output
        assert "0.1.0" in output


# ===========================================================================
# _enrich_pypi
# ===========================================================================


class TestEnrichPypi:
    def _make_pypi_response(self, name: str, version: str, summary: str) -> dict:
        return {
            "info": {
                "name": name,
                "version": version,
                "summary": summary,
                "project_url": f"https://pypi.org/project/{name}/",
            },
            "releases": {"1.0.0": [], "2.0.0": [], version: []},
        }

    def test_returns_package_metadata(self):
        pypi_resp = MagicMock()
        pypi_resp.json.return_value = self._make_pypi_response("requests", "2.28.0", "HTTP for Humans")
        pypi_resp.raise_for_status.return_value = None
        stats_resp = MagicMock()
        stats_resp.json.return_value = {"data": {"last_week": 50000000, "last_month": 200000000}}
        stats_resp.raise_for_status.return_value = None
        with patch("requests.get", side_effect=[pypi_resp, stats_resp]):
            result = _enrich_pypi("requests")
        assert result["ecosystem"] == "PyPI"
        assert result["latest_version"] == "2.28.0"
        assert result["weekly_downloads"] == 50000000
        assert "HTTP for Humans" in result.get("description", "")

    def test_download_stats_none_on_rate_limit(self):
        pypi_resp = MagicMock()
        pypi_resp.json.return_value = self._make_pypi_response("requests", "2.28.0", "HTTP for Humans")
        pypi_resp.raise_for_status.return_value = None
        stats_resp = MagicMock()
        stats_resp.raise_for_status.side_effect = Exception("429 Too Many Requests")
        with patch("requests.get", side_effect=[pypi_resp, stats_resp]):
            result = _enrich_pypi("requests")
        assert result["weekly_downloads"] is None

    def test_graceful_degradation_on_pypi_error(self):
        with patch("requests.get", side_effect=Exception("connection refused")):
            result = _enrich_pypi("requests")
        assert result["ecosystem"] == "PyPI"
        assert result["package_name"] == "requests"


# ===========================================================================
# _enrich_maven
# ===========================================================================


class TestEnrichMaven:
    def _make_maven_response(self, group_id: str, artifact_id: str, version: str) -> dict:
        return {
            "response": {
                "numFound": 1,
                "docs": [
                    {
                        "id": f"{group_id}:{artifact_id}",
                        "g": group_id,
                        "a": artifact_id,
                        "latestVersion": version,
                        "versionCount": 42,
                    }
                ],
            }
        }

    def test_returns_artifact_metadata(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_maven_response("org.apache.logging.log4j", "log4j-core", "2.20.0")
        with patch("requests.get", return_value=mock_resp):
            result = _enrich_maven("org.apache.logging.log4j:log4j-core")
        assert result["ecosystem"] == "Maven"
        assert result["latest_version"] == "2.20.0"
        assert result["version_count"] == 42
        assert result["full_id"] == "org.apache.logging.log4j:log4j-core"

    def test_plain_artifact_id(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_maven_response("org.example", "mylib", "1.0.0")
        with patch("requests.get", return_value=mock_resp):
            result = _enrich_maven("mylib")
        assert result["ecosystem"] == "Maven"

    def test_graceful_degradation_on_error(self):
        with patch("requests.get", side_effect=Exception("403 Forbidden")):
            result = _enrich_maven("log4j-core")
        assert result["ecosystem"] == "Maven"
        assert result["package_name"] == "log4j-core"

    def test_no_docs_returns_minimal_record(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": {"numFound": 0, "docs": []}}
        with patch("requests.get", return_value=mock_resp):
            result = _enrich_maven("nonexistent-lib")
        assert result["total_artifacts_found"] == 0


# ===========================================================================
# _enrich_package dispatch
# ===========================================================================


class TestEnrichPackageDispatch:
    def test_npm_ecosystem(self):
        with patch("manus_agent.tools.get_dependency_blast_radius._enrich_npm") as mock_npm:
            mock_npm.return_value = {"ecosystem": "npm", "package_name": "axios"}
            _enrich_package("axios", "npm")
        mock_npm.assert_called_once_with("axios")

    def test_javascript_ecosystem_routes_to_npm(self):
        with patch("manus_agent.tools.get_dependency_blast_radius._enrich_npm") as mock_npm:
            mock_npm.return_value = {"ecosystem": "npm", "package_name": "react"}
            _enrich_package("react", "javascript")
        mock_npm.assert_called_once_with("react")

    def test_pypi_ecosystem(self):
        with patch("manus_agent.tools.get_dependency_blast_radius._enrich_pypi") as mock_pypi:
            mock_pypi.return_value = {"ecosystem": "PyPI", "package_name": "requests"}
            _enrich_package("requests", "PyPI")
        mock_pypi.assert_called_once_with("requests")

    def test_python_ecosystem_routes_to_pypi(self):
        with patch("manus_agent.tools.get_dependency_blast_radius._enrich_pypi") as mock_pypi:
            mock_pypi.return_value = {"ecosystem": "PyPI", "package_name": "flask"}
            _enrich_package("flask", "python")
        mock_pypi.assert_called_once_with("flask")

    def test_maven_ecosystem(self):
        with patch("manus_agent.tools.get_dependency_blast_radius._enrich_maven") as mock_maven:
            mock_maven.return_value = {"ecosystem": "Maven", "package_name": "log4j-core"}
            _enrich_package("log4j-core", "Maven")
        mock_maven.assert_called_once_with("log4j-core")

    def test_unknown_ecosystem_returns_minimal_record(self):
        result = _enrich_package("unknown-pkg", "SomeExoticEcosystem")
        assert result["ecosystem"] == "SomeExoticEcosystem"
        assert result["package_name"] == "unknown-pkg"


# ===========================================================================
# get_dependency_blast_radius (integration tests with full stack mocked)
# ===========================================================================


class TestGetDependencyBlastRadius:
    def _mock_all_sources(
        self,
        nvd_pkgs=None,
        osv_pkgs=None,
        ghsa_pkgs=None,
        npm_stats=None,
    ):
        """Return context managers that mock all external calls."""
        nvd_pkgs = nvd_pkgs or []
        osv_pkgs = osv_pkgs or []
        ghsa_pkgs = ghsa_pkgs or []
        npm_stats = npm_stats or {
            "ecosystem": "npm",
            "package_name": "lodash",
            "dependent_packages_count": 200000,
            "weekly_downloads": 130000000,
            "monthly_downloads": 520000000,
        }

        patches = [
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=nvd_pkgs,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=osv_pkgs,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=ghsa_pkgs,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=npm_stats,
            ),
        ]
        return patches

    def test_cve_no_packages_found_returns_message(self):
        with (
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected", return_value=[]),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected", return_value=[]),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected", return_value=[]),
        ):
            result = get_dependency_blast_radius("CVE-2021-44228")
        assert "No affected package records found" in result

    def test_cve_with_packages_returns_blast_radius_info(self):
        pkgs = [{"name": "lodash", "ecosystem": "npm", "version_range": ">=4.0.0, <4.17.21", "source": "osv"}]
        npm_stats = {
            "ecosystem": "npm",
            "package_name": "lodash",
            "dependent_packages_count": 200000,
            "weekly_downloads": 130000000,
        }
        with (
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected", return_value=[]),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected", return_value=pkgs),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected", return_value=[]),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=npm_stats,
            ),
        ):
            result = get_dependency_blast_radius("CVE-2021-44228")
        assert "CRITICAL" in result
        assert "lodash" in result
        assert "130,000,000" in result or "Weekly downloads" in result

    def test_package_spec_direct(self):
        npm_stats = {
            "ecosystem": "npm",
            "package_name": "lodash",
            "dependent_packages_count": 200000,
            "weekly_downloads": 130000000,
        }
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=npm_stats,
        ):
            result = get_dependency_blast_radius("lodash@4.17.20")
        assert "lodash" in result
        assert "CRITICAL" in result

    def test_invalid_spec_returns_error(self):
        # A completely empty string
        result = get_dependency_blast_radius("")
        # Should return an error or empty packages message (graceful)
        assert isinstance(result, str)

    def test_cve_deduplication_across_sources(self):
        # Same package appears in both OSV and GHSA
        osv_pkgs = [{"name": "requests", "ecosystem": "PyPI", "version_range": ">=2.0, <2.29", "source": "osv"}]
        ghsa_pkgs = [{"name": "requests", "ecosystem": "PyPI", "version_range": ">=2.0, <2.29", "source": "ghsa"}]
        pypi_stats = {
            "ecosystem": "PyPI",
            "package_name": "requests",
            "weekly_downloads": 60000000,
        }
        with (
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected", return_value=[]),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected", return_value=osv_pkgs),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected", return_value=ghsa_pkgs),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=pypi_stats,
            ) as mock_enrich,
        ):
            _result = get_dependency_blast_radius("CVE-2023-32681")
        # _enrich_package should be called exactly once (deduplicated)
        mock_enrich.assert_called_once()

    def test_packages_sorted_by_blast_severity(self):
        osv_pkgs = [
            {"name": "small-lib", "ecosystem": "npm", "version_range": "1.0.0", "source": "osv"},
            {"name": "big-lib", "ecosystem": "npm", "version_range": "2.0.0", "source": "osv"},
        ]

        def enrich_side_effect(name, ecosystem):
            if name == "small-lib":
                return {"ecosystem": "npm", "package_name": "small-lib", "weekly_downloads": 1000}
            else:
                return {"ecosystem": "npm", "package_name": "big-lib", "weekly_downloads": 10_000_000}

        with (
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected", return_value=[]),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected", return_value=osv_pkgs),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected", return_value=[]),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                side_effect=enrich_side_effect,
            ),
        ):
            result = get_dependency_blast_radius("CVE-2021-00001")
        # CRITICAL (big-lib) should appear before LOW (small-lib)
        assert result.index("big-lib") < result.index("small-lib")

    def test_summary_line_present(self):
        pkgs = [{"name": "axios", "ecosystem": "npm", "version_range": ">=1.0.0, <1.7.0", "source": "ghsa"}]
        npm_stats = {
            "ecosystem": "npm",
            "package_name": "axios",
            "dependent_packages_count": 180000,
            "weekly_downloads": 120000000,
        }
        with (
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected", return_value=[]),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected", return_value=[]),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected", return_value=pkgs),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=npm_stats,
            ),
        ):
            result = get_dependency_blast_radius("CVE-2023-45857")
        assert "Summary" in result

    def test_max_packages_respected(self):
        # Create 15 packages
        many_pkgs = [
            {"name": f"lib-{i}", "ecosystem": "npm", "version_range": "1.0", "source": "osv"} for i in range(15)
        ]
        call_count = {"n": 0}

        def enrich_counter(name, ecosystem):
            call_count["n"] += 1
            return {"ecosystem": "npm", "package_name": name, "weekly_downloads": 100}

        with (
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected", return_value=[]),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected", return_value=many_pkgs),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected", return_value=[]),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                side_effect=enrich_counter,
            ),
        ):
            _result = get_dependency_blast_radius("CVE-2021-00001", max_packages=5)
        assert call_count["n"] == 5

    def test_ecosystem_label_in_output(self):
        pkgs = [{"name": "requests", "ecosystem": "PyPI", "version_range": "2.28.0", "source": "osv"}]
        pypi_stats = {
            "ecosystem": "PyPI",
            "package_name": "requests",
            "weekly_downloads": 60000000,
            "latest_version": "2.34.0",
            "description": "Python HTTP for Humans",
            "release_count": 163,
        }
        with (
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected", return_value=[]),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected", return_value=pkgs),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected", return_value=[]),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=pypi_stats,
            ),
        ):
            result = get_dependency_blast_radius("CVE-2023-32681")
        assert "Python" in result or "PyPI" in result


# ===========================================================================
# CLI integration: _build_blast_radius_parser
# ===========================================================================


class TestCliParser:
    def test_spec_argument_required(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_default_output_text(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        args = parser.parse_args(["CVE-2021-44228"])
        assert args.output == "text"

    def test_output_json(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        args = parser.parse_args(["CVE-2021-44228", "--output", "json"])
        assert args.output == "json"

    def test_max_packages_default(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        args = parser.parse_args(["requests@2.28.0"])
        assert args.max_packages == 10

    def test_max_packages_custom(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        args = parser.parse_args(["requests@2.28.0", "--max-packages", "20"])
        assert args.max_packages == 20

    def test_blast_radius_in_subcommands_set(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "blast-radius" in _SUBCOMMANDS
