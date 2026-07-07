"""
Tests for _enrich_nuget, _extract_nuget_first_release, and related
NuGet-specific behaviour in get_dependency_blast_radius.py

All external HTTP calls are fully mocked — zero real network I/O.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from manus_agent.tools.get_dependency_blast_radius import (
    _blast_score,
    _enrich_nuget,
    _enrich_package,
    _extract_nuget_first_release,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_response(
    pkg_id: str = "Newtonsoft.Json",
    total_downloads: int = 4_000_000_000,
    version: str = "13.0.3",
    description: str = "Popular JSON serializer for .NET",
    project_url: str = "https://www.newtonsoft.com/json",
) -> dict[str, Any]:
    return {
        "data": [
            {
                "id": pkg_id,
                "totalDownloads": total_downloads,
                "version": version,
                "description": description,
                "projectUrl": project_url,
            }
        ]
    }


def _make_reg_response(dates: list[str], pkg_id: str = "Newtonsoft.Json") -> dict[str, Any]:
    """Build a minimal NuGet registration index with one page."""
    items = [
        {
            "catalogEntry": {
                "id": pkg_id,
                "version": f"1.{i}.0",
                "published": d,
            }
        }
        for i, d in enumerate(dates)
    ]
    return {"items": [{"items": items}]}


# ===========================================================================
# _extract_nuget_first_release
# ===========================================================================


class TestExtractNugetFirstRelease:
    def test_finds_earliest_date_from_multiple_items(self):
        pages = [
            {
                "items": [
                    {"catalogEntry": {"published": "2020-06-15T00:00:00Z"}},
                    {"catalogEntry": {"published": "2015-03-20T00:00:00Z"}},
                    {"catalogEntry": {"published": "2022-01-10T00:00:00+00:00"}},
                ]
            }
        ]
        result = _extract_nuget_first_release(pages)
        assert result["first_release_date"] == "2015-03-20"

    def test_counts_total_versions(self):
        pages = [
            {
                "items": [
                    {"catalogEntry": {"published": "2021-01-01T00:00:00Z"}},
                    {"catalogEntry": {"published": "2022-01-01T00:00:00Z"}},
                ]
            },
            {
                "items": [
                    {"catalogEntry": {"published": "2023-01-01T00:00:00Z"}},
                ]
            },
        ]
        result = _extract_nuget_first_release(pages)
        assert result["total_versions"] == 3

    def test_handles_z_suffix(self):
        pages = [{"items": [{"catalogEntry": {"published": "2018-05-10T12:34:56Z"}}]}]
        result = _extract_nuget_first_release(pages)
        assert result["first_release_date"] == "2018-05-10"

    def test_handles_offset_tz(self):
        pages = [{"items": [{"catalogEntry": {"published": "2019-11-01T08:00:00+05:30"}}]}]
        result = _extract_nuget_first_release(pages)
        assert "first_release_date" in result

    def test_handles_no_subsecond_fraction(self):
        pages = [{"items": [{"catalogEntry": {"published": "2017-07-04T00:00:00"}}]}]
        result = _extract_nuget_first_release(pages)
        assert result["first_release_date"] == "2017-07-04"

    def test_handles_subsecond_fraction(self):
        pages = [{"items": [{"catalogEntry": {"published": "2020-01-01T00:00:00.1234567Z"}}]}]
        result = _extract_nuget_first_release(pages)
        assert result["first_release_date"] == "2020-01-01"

    def test_skips_malformed_dates(self):
        pages = [
            {
                "items": [
                    {"catalogEntry": {"published": "not-a-date"}},
                    {"catalogEntry": {"published": "2021-06-30T00:00:00Z"}},
                ]
            }
        ]
        result = _extract_nuget_first_release(pages)
        assert result["first_release_date"] == "2021-06-30"

    def test_empty_pages_returns_zero_versions(self):
        result = _extract_nuget_first_release([])
        assert result == {"total_versions": 0}
        assert "first_release_date" not in result

    def test_items_with_missing_published_skipped(self):
        pages = [
            {
                "items": [
                    {"catalogEntry": {"published": ""}},
                    {"catalogEntry": {}},
                    {"catalogEntry": {"published": "2022-03-15T00:00:00Z"}},
                ]
            }
        ]
        result = _extract_nuget_first_release(pages)
        assert result["first_release_date"] == "2022-03-15"
        assert result["total_versions"] == 3

    def test_age_years_computed(self):
        pages = [{"items": [{"catalogEntry": {"published": "2015-01-01T00:00:00Z"}}]}]
        result = _extract_nuget_first_release(pages)
        assert "age_years" in result
        assert result["age_years"] >= 9.0  # At least 9 years from 2015 to 2024+

    def test_multiple_pages_aggregated(self):
        pages = [
            {"items": [{"catalogEntry": {"published": "2021-01-01T00:00:00Z"}}]},
            {"items": [{"catalogEntry": {"published": "2018-06-01T00:00:00Z"}}]},
            {"items": [{"catalogEntry": {"published": "2023-12-31T00:00:00Z"}}]},
        ]
        result = _extract_nuget_first_release(pages)
        assert result["first_release_date"] == "2018-06-01"
        assert result["total_versions"] == 3


# ===========================================================================
# _enrich_nuget
# ===========================================================================


class TestEnrichNuget:
    def _make_mock_responses(
        self,
        search_data: dict | None = None,
        reg_data: dict | None = None,
        search_raises: Exception | None = None,
        reg_raises: Exception | None = None,
    ):
        """Return a list of side-effects for requests.get."""
        responses = []
        if search_raises:
            responses.append(search_raises)
        else:
            m = MagicMock()
            m.raise_for_status.return_value = None
            m.json.return_value = search_data or {"data": []}
            responses.append(m)
        if reg_raises:
            responses.append(reg_raises)
        else:
            m = MagicMock()
            m.raise_for_status.return_value = None
            m.json.return_value = reg_data or {"items": []}
            responses.append(m)
        return responses

    def test_returns_ecosystem_and_package_name(self):
        with patch("requests.get", side_effect=self._make_mock_responses()):
            result = _enrich_nuget("Newtonsoft.Json")
        assert result["ecosystem"] == "NuGet"
        assert result["package_name"] == "Newtonsoft.Json"

    def test_extracts_total_downloads(self):
        search = _make_search_response(pkg_id="Newtonsoft.Json", total_downloads=4_500_000_000)
        with patch("requests.get", side_effect=self._make_mock_responses(search_data=search)):
            result = _enrich_nuget("Newtonsoft.Json")
        assert result["total_downloads"] == 4_500_000_000

    def test_extracts_latest_version(self):
        search = _make_search_response(version="13.0.3")
        with patch("requests.get", side_effect=self._make_mock_responses(search_data=search)):
            result = _enrich_nuget("Newtonsoft.Json")
        assert result["latest_version"] == "13.0.3"

    def test_extracts_description_truncated(self):
        long_desc = "A" * 200
        search = _make_search_response(description=long_desc)
        with patch("requests.get", side_effect=self._make_mock_responses(search_data=search)):
            result = _enrich_nuget("Newtonsoft.Json")
        assert len(result.get("description", "")) <= 120

    def test_extracts_home_page(self):
        search = _make_search_response(project_url="https://www.newtonsoft.com/json")
        with patch("requests.get", side_effect=self._make_mock_responses(search_data=search)):
            result = _enrich_nuget("Newtonsoft.Json")
        assert result.get("home_page") == "https://www.newtonsoft.com/json"

    def test_case_insensitive_id_match(self):
        # NuGet package IDs are case-insensitive; API returns canonical casing
        search = _make_search_response(pkg_id="Newtonsoft.Json", total_downloads=100)
        with patch("requests.get", side_effect=self._make_mock_responses(search_data=search)):
            result = _enrich_nuget("newtonsoft.json")
        assert result.get("total_downloads") == 100

    def test_extracts_first_release_date_from_registration(self):
        reg = _make_reg_response(["2012-04-04T00:00:00Z", "2015-01-01T00:00:00Z"])
        with patch("requests.get", side_effect=self._make_mock_responses(reg_data=reg)):
            result = _enrich_nuget("Newtonsoft.Json")
        assert result.get("first_release_date") == "2012-04-04"

    def test_first_release_absent_when_no_versions(self):
        reg = {"items": []}
        with patch("requests.get", side_effect=self._make_mock_responses(reg_data=reg)):
            result = _enrich_nuget("Newtonsoft.Json")
        assert "first_release_date" not in result

    def test_graceful_degradation_search_error(self):
        """Search fails → still returns ecosystem/package_name (no crash)."""
        reg = _make_reg_response(["2020-01-01T00:00:00Z"])
        with patch(
            "requests.get",
            side_effect=self._make_mock_responses(search_raises=Exception("connection timeout"), reg_data=reg),
        ):
            result = _enrich_nuget("Serilog")
        assert result["ecosystem"] == "NuGet"
        assert result["package_name"] == "Serilog"
        assert "total_downloads" not in result
        # Registration data still extracted
        assert result.get("first_release_date") == "2020-01-01"

    def test_graceful_degradation_registration_error(self):
        """Registration fetch fails → search data still returned (no crash)."""
        search = _make_search_response(pkg_id="Serilog", total_downloads=500_000_000)
        with patch(
            "requests.get",
            side_effect=self._make_mock_responses(search_data=search, reg_raises=Exception("404 Not Found")),
        ):
            result = _enrich_nuget("Serilog")
        assert result["total_downloads"] == 500_000_000
        assert "first_release_date" not in result

    def test_graceful_degradation_both_errors(self):
        """Both calls fail → minimal record returned (no crash)."""
        with patch("requests.get", side_effect=Exception("network error")):
            result = _enrich_nuget("SomePackage")
        assert result["ecosystem"] == "NuGet"
        assert result["package_name"] == "SomePackage"

    def test_no_matching_package_in_search_results(self):
        """Search returns results but none match the queried package name."""
        search = {"data": [{"id": "Other.Package", "totalDownloads": 999, "version": "1.0.0"}]}
        with patch("requests.get", side_effect=self._make_mock_responses(search_data=search)):
            result = _enrich_nuget("Newtonsoft.Json")
        # No downloads extracted for a non-matching result
        assert "total_downloads" not in result

    def test_total_versions_from_registration(self):
        reg = _make_reg_response(["2021-01-01T00:00:00Z", "2022-06-01T00:00:00Z", "2023-11-01T00:00:00Z"])
        with patch("requests.get", side_effect=self._make_mock_responses(reg_data=reg)):
            result = _enrich_nuget("FluentAssertions")
        assert result.get("total_versions") == 3


# ===========================================================================
# _enrich_package dispatch — NuGet routing
# ===========================================================================


class TestEnrichPackageNuGetDispatch:
    def _mock_enrich_nuget(self, return_val: dict | None = None):
        return patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_nuget",
            return_value=return_val or {"ecosystem": "NuGet", "package_name": "Serilog"},
        )

    def test_nuget_ecosystem_dispatches(self):
        with self._mock_enrich_nuget() as mock:
            _enrich_package("Serilog", "NuGet")
        mock.assert_called_once_with("Serilog")

    def test_dotnet_alias_dispatches_to_nuget(self):
        with self._mock_enrich_nuget() as mock:
            _enrich_package("Serilog", "dotnet")
        mock.assert_called_once_with("Serilog")

    def test_net_alias_dispatches_to_nuget(self):
        with self._mock_enrich_nuget() as mock:
            _enrich_package("Serilog", ".net")
        mock.assert_called_once_with("Serilog")

    def test_csharp_alias_dispatches_to_nuget(self):
        with self._mock_enrich_nuget() as mock:
            _enrich_package("Serilog", "csharp")
        mock.assert_called_once_with("Serilog")

    def test_hash_csharp_alias_dispatches_to_nuget(self):
        with self._mock_enrich_nuget() as mock:
            _enrich_package("Serilog", "c#")
        mock.assert_called_once_with("Serilog")


# ===========================================================================
# _blast_score — NuGet total_downloads proxy behaviour
# ===========================================================================


class TestBlastScoreNuGetFallback:
    def test_total_downloads_used_when_weekly_absent(self):
        # 520M all-time over 10 years → ~1M/week → HIGH
        stats = {"total_downloads": 520_000_000, "age_years": 10.0}
        assert _blast_score(stats) in ("HIGH", "CRITICAL")

    def test_large_total_downloads_maps_to_critical(self):
        # 5B all-time over 10 years → ~9.6M/week → CRITICAL
        stats = {"total_downloads": 5_000_000_000, "age_years": 10.0}
        assert _blast_score(stats) == "CRITICAL"

    def test_small_total_downloads_maps_to_low(self):
        # 50K all-time over 5 years → ~192/week → LOW
        stats = {"total_downloads": 50_000, "age_years": 5.0}
        assert _blast_score(stats) in ("LOW", "UNKNOWN")

    def test_weekly_downloads_takes_precedence_over_total(self):
        # weekly_downloads is set → should use it directly, not proxy from total
        stats = {"weekly_downloads": 10_000_000, "total_downloads": 1_000}
        assert _blast_score(stats) == "CRITICAL"

    def test_age_zero_does_not_divide_by_zero(self):
        # age_years is 0 → should not raise ZeroDivisionError
        stats = {"total_downloads": 1_000_000, "age_years": 0.0}
        score = _blast_score(stats)
        assert score in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN")
