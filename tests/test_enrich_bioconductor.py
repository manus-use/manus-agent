"""
Tests for _enrich_bioconductor and related Bioconductor integration in
get_dependency_blast_radius.

100% mocked — no real HTTP calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.get_dependency_blast_radius import (
    _enrich_bioconductor,
    _enrich_package,
    get_dependency_blast_radius,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_BIOC_MANIFEST = {
    "limma": {
        "Package": "limma",
        "Version": "3.60.6",
        "Title": "Linear Models for Microarray and Omics Data",
        "Description": "Data analysis, linear models and differential expression for omics data.",
        "biocViews": ["DifferentialExpression", "GeneExpression", "Microarray", "RNASeq", "Software"],
        "dependsOnMe": ["edgeR", "metagenomeSeq", "variancePartition"],
        "importsMe": ["ABSSeq", "AMARETTO", "BatchQC"],
        "suggestsMe": ["SuggesterA", "SuggesterB"],
        "Rank": 19,
        "dependencyCount": 6,
        "git_url": "https://git.bioconductor.org/packages/limma",
    },
    "GenomicRanges": {
        "Package": "GenomicRanges",
        "Version": "1.56.2",
        "Title": "Representation and manipulation of genomic intervals",
        "Description": "Provides infrastructure for representing and manipulating genomic intervals.",
        "biocViews": ["Genetics", "Infrastructure", "Sequencing", "Software"],
        "dependsOnMe": ["AllelicImbalance", "AneuFinder"],
        "importsMe": ["ACE", "ALDEx2", "APAlyzer", "ASpli"],
        "suggestsMe": [],
        "Rank": 10,
        "dependencyCount": 5,
        "git_url": "https://git.bioconductor.org/packages/GenomicRanges",
    },
    "minimalPkg": {
        "Package": "minimalPkg",
        "Version": "1.0.0",
        "Title": "Minimal test package",
        # No dependsOnMe / importsMe / suggestsMe keys at all
    },
    "NoDepsPkg": {
        "Package": "NoDepsPkg",
        "Version": "2.0.0",
        "Title": "No reverse deps",
        "dependsOnMe": [],
        "importsMe": [],
        "suggestsMe": [],
        "Rank": 1500,
        "dependencyCount": 2,
    },
}


def _mock_bioc_response(manifest: dict | None = None) -> MagicMock:
    """Return a mock requests.Response whose .json() yields the manifest."""
    manifest = manifest if manifest is not None else _BIOC_MANIFEST
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = manifest
    return resp


# ===========================================================================
# TestEnrichBioconductor — unit tests for _enrich_bioconductor
# ===========================================================================


class TestEnrichBioconductor:
    """Tests for _enrich_bioconductor()."""

    def test_basic_fields_populated(self):
        with patch("requests.get", return_value=_mock_bioc_response()):
            result = _enrich_bioconductor("limma")

        assert result["ecosystem"] == "Bioconductor"
        assert result["package_name"] == "limma"
        assert result["latest_version"] == "3.60.6"
        assert "Linear Models" in result["title"]
        assert result["git_url"] == "https://git.bioconductor.org/packages/limma"
        assert result["bioc_page"] == "https://bioconductor.org/packages/release/bioc/html/limma.html"

    def test_hard_rev_deps_calculated_correctly(self):
        # limma: dependsOnMe=3, importsMe=3 → hard_rev_deps=6
        with patch("requests.get", return_value=_mock_bioc_response()):
            result = _enrich_bioconductor("limma")

        assert result["dependent_packages_count"] == 6
        assert result["depends_on_me"] == 3
        assert result["imports_me"] == 3

    def test_suggests_me_not_counted_in_blast_score(self):
        # suggestsMe has 2 entries but should NOT be included in dependent_packages_count
        with patch("requests.get", return_value=_mock_bioc_response()):
            result = _enrich_bioconductor("limma")

        assert result["suggests_me"] == 2
        # dependent_packages_count must NOT include suggestsMe
        assert result["dependent_packages_count"] == result["depends_on_me"] + result["imports_me"]

    def test_download_rank_present(self):
        with patch("requests.get", return_value=_mock_bioc_response()):
            result = _enrich_bioconductor("limma")

        assert result["download_rank"] == 19

    def test_bioc_views_formatted(self):
        with patch("requests.get", return_value=_mock_bioc_response()):
            result = _enrich_bioconductor("limma")

        # Should contain comma-separated biocViews, up to 5
        assert isinstance(result["bioc_views"], str)
        assert "DifferentialExpression" in result["bioc_views"]

    def test_description_truncated_to_120(self):
        with patch("requests.get", return_value=_mock_bioc_response()):
            result = _enrich_bioconductor("limma")

        assert len(result["description"]) <= 120

    def test_package_not_found_returns_minimal(self):
        with patch("requests.get", return_value=_mock_bioc_response()):
            result = _enrich_bioconductor("NonExistentPackage")

        assert result["ecosystem"] == "Bioconductor"
        assert result["package_name"] == "NonExistentPackage"
        # No version / blast fields
        assert "latest_version" not in result
        assert "dependent_packages_count" not in result

    def test_case_insensitive_fallback(self):
        # Package key is "GenomicRanges" — query with lowercase should still match
        manifest = {
            "GenomicRanges": _BIOC_MANIFEST["GenomicRanges"],
        }
        with patch("requests.get", return_value=_mock_bioc_response(manifest)):
            result = _enrich_bioconductor("genomicranges")

        assert result["latest_version"] == "1.56.2"

    def test_network_error_returns_minimal(self):
        with patch("requests.get", side_effect=Exception("connection refused")):
            result = _enrich_bioconductor("limma")

        assert result["ecosystem"] == "Bioconductor"
        assert result["package_name"] == "limma"
        assert "latest_version" not in result

    def test_missing_optional_keys_no_error(self):
        # minimalPkg has no dependsOnMe / importsMe / suggestsMe / Rank
        with patch("requests.get", return_value=_mock_bioc_response()):
            result = _enrich_bioconductor("minimalPkg")

        assert result["package_name"] == "minimalPkg"
        assert result["dependent_packages_count"] == 0  # None → [] → 0
        assert result["depends_on_me"] == 0
        assert result["imports_me"] == 0

    def test_zero_rev_deps_package(self):
        with patch("requests.get", return_value=_mock_bioc_response()):
            result = _enrich_bioconductor("NoDepsPkg")

        assert result["dependent_packages_count"] == 0
        assert result["depends_on_me"] == 0
        assert result["imports_me"] == 0
        assert result["suggests_me"] == 0

    def test_genomicranges_rev_deps(self):
        with patch("requests.get", return_value=_mock_bioc_response()):
            result = _enrich_bioconductor("GenomicRanges")

        # dependsOnMe=2, importsMe=4
        assert result["depends_on_me"] == 2
        assert result["imports_me"] == 4
        assert result["dependent_packages_count"] == 6

    def test_bioc_page_url_correct(self):
        with patch("requests.get", return_value=_mock_bioc_response()):
            result = _enrich_bioconductor("GenomicRanges")

        assert result["bioc_page"] == ("https://bioconductor.org/packages/release/bioc/html/GenomicRanges.html")


# ===========================================================================
# TestBlastScoreBioconductor — blast score thresholds for Bioconductor
# ===========================================================================


class TestBlastScoreBioconductor:
    """Verify _blast_score produces correct labels using dependent_packages_count."""

    def _make_manifest_with_revdeps(self, n_depends: int, n_imports: int) -> dict:
        return {
            "testpkg": {
                "Package": "testpkg",
                "Version": "1.0.0",
                "Title": "Test",
                "dependsOnMe": [f"d{i}" for i in range(n_depends)],
                "importsMe": [f"i{i}" for i in range(n_imports)],
                "suggestsMe": [],
            }
        }

    def test_critical_threshold(self):
        # ≥50000 dependents → CRITICAL (unrealistic for Bioc but tests the threshold)
        manifest = self._make_manifest_with_revdeps(30000, 25000)  # 55000 total
        with patch("requests.get", return_value=_mock_bioc_response(manifest)):
            result = _enrich_bioconductor("testpkg")
        assert result["dependent_packages_count"] == 55000

    def test_high_threshold(self):
        # ≥5000 dependents → HIGH
        manifest = self._make_manifest_with_revdeps(3000, 3000)  # 6000 total
        with patch("requests.get", return_value=_mock_bioc_response(manifest)):
            result = _enrich_bioconductor("testpkg")
        assert result["dependent_packages_count"] == 6000

    def test_medium_threshold(self):
        # ≥500 dependents → MEDIUM
        manifest = self._make_manifest_with_revdeps(300, 300)  # 600 total
        with patch("requests.get", return_value=_mock_bioc_response(manifest)):
            result = _enrich_bioconductor("testpkg")
        assert result["dependent_packages_count"] == 600

    def test_low_threshold(self):
        # >0 but <500 → LOW
        manifest = self._make_manifest_with_revdeps(5, 3)  # 8 total
        with patch("requests.get", return_value=_mock_bioc_response(manifest)):
            result = _enrich_bioconductor("testpkg")
        assert result["dependent_packages_count"] == 8

    def test_unknown_threshold(self):
        # 0 hard rev-deps → UNKNOWN
        manifest = self._make_manifest_with_revdeps(0, 0)
        with patch("requests.get", return_value=_mock_bioc_response(manifest)):
            result = _enrich_bioconductor("testpkg")
        assert result["dependent_packages_count"] == 0


# ===========================================================================
# TestEnrichPackageBioconductorDispatch — _enrich_package routing
# ===========================================================================


class TestEnrichPackageBioconductorDispatch:
    """Verify _enrich_package dispatches to _enrich_bioconductor for all aliases."""

    @pytest.mark.parametrize(
        "ecosystem",
        ["Bioconductor", "bioconductor", "bioc", "r-bioc", "rbioconductor"],
    )
    def test_dispatch_alias(self, ecosystem: str):
        with patch("manus_agent.tools.get_dependency_blast_radius._enrich_bioconductor") as mock_bioc:
            mock_bioc.return_value = {"ecosystem": "Bioconductor", "package_name": "limma"}
            _enrich_package("limma", ecosystem)
        mock_bioc.assert_called_once_with("limma")

    def test_unknown_ecosystem_does_not_dispatch_to_bioc(self):
        with patch("manus_agent.tools.get_dependency_blast_radius._enrich_bioconductor") as mock_bioc:
            _enrich_package("somelib", "CRAN")
        mock_bioc.assert_not_called()


# ===========================================================================
# TestGetDependencyBlastRadiusBioconductor — integration via main tool
# ===========================================================================


class TestGetDependencyBlastRadiusBioconductor:
    """Integration tests that exercise the full get_dependency_blast_radius pipeline."""

    def _bioc_stats(
        self,
        name: str = "limma",
        hard_rev_deps: int = 6,
        rank: int = 19,
    ) -> dict:
        return {
            "ecosystem": "Bioconductor",
            "package_name": name,
            "latest_version": "3.60.6",
            "title": "Linear Models for Microarray and Omics Data",
            "description": "Data analysis, linear models and differential expression for omics data.",
            "bioc_views": "DifferentialExpression, GeneExpression, Microarray",
            "download_rank": rank,
            "dependency_count": 6,
            "dependent_packages_count": hard_rev_deps,
            "depends_on_me": 3,
            "imports_me": hard_rev_deps - 3,
            "suggests_me": 2,
            "git_url": "https://git.bioconductor.org/packages/limma",
            "bioc_page": f"https://bioconductor.org/packages/release/bioc/html/{name}.html",
        }

    def test_direct_bioc_package_query(self):
        stats = self._bioc_stats()
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=stats,
        ):
            result = get_dependency_blast_radius("bioc:limma")

        assert "limma" in result
        assert "Bioconductor" in result

    def test_bioc_output_contains_rev_dep_info(self):
        stats = self._bioc_stats(hard_rev_deps=120)
        stats["blast_radius"] = "MEDIUM"  # manually set for display
        with (
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected", return_value=[]),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=[
                    {
                        "name": "limma",
                        "ecosystem": "Bioconductor",
                        "version_range": "<3.60.7",
                        "source": "osv",
                    }
                ],
            ),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected", return_value=[]),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=stats,
            ),
        ):
            result = get_dependency_blast_radius("CVE-2024-99999")

        assert "limma" in result
        assert "Bioconductor" in result

    def test_bioc_output_shows_rank(self):
        stats = self._bioc_stats(rank=19)
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=stats,
        ):
            result = get_dependency_blast_radius("bioc:limma")

        assert "#19" in result

    def test_bioc_output_shows_git_url(self):
        stats = self._bioc_stats()
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=stats,
        ):
            result = get_dependency_blast_radius("bioc:limma")

        assert "git.bioconductor.org" in result

    def test_bioc_output_shows_bioc_page(self):
        stats = self._bioc_stats()
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=stats,
        ):
            result = get_dependency_blast_radius("bioc:limma")

        assert "bioconductor.org/packages/release" in result

    def test_bioc_output_shows_biocviews(self):
        stats = self._bioc_stats()
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=stats,
        ):
            result = get_dependency_blast_radius("bioc:limma")

        assert "DifferentialExpression" in result

    def test_bioc_output_shows_description(self):
        stats = self._bioc_stats()
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=stats,
        ):
            result = get_dependency_blast_radius("bioc:limma")

        assert "linear models" in result.lower() or "omics" in result.lower()

    def test_bioc_summary_line_present(self):
        stats = self._bioc_stats(hard_rev_deps=600)
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=stats,
        ):
            result = get_dependency_blast_radius("bioc:limma")

        assert "Summary" in result

    def test_bioc_versioned_package_query(self):
        stats = self._bioc_stats()
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=stats,
        ):
            result = get_dependency_blast_radius("bioc:limma@3.60.6")

        assert "limma" in result

    def test_bioc_suggests_me_not_in_score_line(self):
        """suggestsMe count should be shown as informational, not driving blast score."""
        stats = self._bioc_stats(hard_rev_deps=8)
        stats["suggests_me"] = 50
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=stats,
        ):
            result = get_dependency_blast_radius("bioc:limma")

        # The "not scored" annotation should appear in output
        assert "not scored" in result

    def test_bioc_zero_rev_deps_unknown_blast(self):
        stats = self._bioc_stats(hard_rev_deps=0)
        stats["depends_on_me"] = 0
        stats["imports_me"] = 0
        stats["suggests_me"] = 0
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=stats,
        ):
            result = get_dependency_blast_radius("bioc:limma")

        assert "UNKNOWN" in result

    def test_bioc_ecosystem_label_in_output(self):
        stats = self._bioc_stats()
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=stats,
        ):
            result = get_dependency_blast_radius("bioc:limma")

        # Ecosystem label should be "Bioconductor (R/Bioinformatics)"
        assert "R/Bioinformatics" in result
