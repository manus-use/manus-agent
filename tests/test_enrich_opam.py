"""
Tests for OPAM (OCaml) enrichment in get_dependency_blast_radius.

All external HTTP calls and subprocess calls are mocked — no real network I/O.
100%% mocked: GitHub Contents API, GitHub opam file fetch, gh CLI subprocess.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.get_dependency_blast_radius import (
    _enrich_opam,
    _enrich_package,
    _parse_opam_file,
    _parse_opam_versions,
    _pick_latest_opam_version,
)

# ---------------------------------------------------------------------------
# Sample fixtures
# ---------------------------------------------------------------------------

_TLS_VERSIONS = [
    {"name": "tls.0.10.4", "type": "dir"},
    {"name": "tls.0.11.0", "type": "dir"},
    {"name": "tls.1.0.4", "type": "dir"},
    {"name": "tls.2.0.1", "type": "dir"},
    {"name": "tls.2.0.4", "type": "dir"},
    {"name": "tls.2.1.0", "type": "dir"},
    {"name": "tls.2.1.1", "type": "dir"},
]

_TLS_OPAM_CONTENT = """opam-version: "2.0"
homepage:     "https://github.com/mirleft/ocaml-tls"
license:      "BSD-2-Clause"
synopsis:     "Transport Layer Security purely in OCaml"
depends: [
  "ocaml" {>= "4.13.0"}
  "dune" {>= "3.0"}
  "mirage-crypto" {>= "1.1.0"}
  "x509" {>= "1.0.0"}
  "fmt" {>= "0.8.7"}
  "logs"
  "ipaddr"
  "digestif" {>= "1.2.0"}
  "alcotest" {with-test}
  "cmdliner" {with-test & >= "1.3.0"}
]
"""


def _make_opam_file_response(content: str) -> dict:
    """Simulate the GitHub Contents API response for a single file."""
    encoded = base64.b64encode(content.encode()).decode()
    return {"content": encoded, "encoding": "base64"}


# ===========================================================================
# TestParseOpamVersions
# ===========================================================================


class TestParseOpamVersions:
    def test_extracts_versions_from_directory_entries(self):
        versions = _parse_opam_versions(_TLS_VERSIONS, "tls")
        assert "2.1.1" in versions
        assert "0.10.4" in versions
        assert len(versions) == 7

    def test_case_insensitive_prefix_match(self):
        entries = [{"name": "TLS.2.1.1", "type": "dir"}]
        versions = _parse_opam_versions(entries, "tls")
        assert "2.1.1" in versions

    def test_skips_unrelated_entries(self):
        entries = [
            {"name": "tls.2.1.1", "type": "dir"},
            {"name": "tls-utils.0.1.0", "type": "dir"},
            {"name": "mirage-tls.1.0.0", "type": "dir"},
        ]
        versions = _parse_opam_versions(entries, "tls")
        assert versions == ["2.1.1"]

    def test_returns_sorted_descending(self):
        versions = _parse_opam_versions(_TLS_VERSIONS, "tls")
        assert versions[0] == "2.1.1"
        assert versions[-1] == "0.10.4"

    def test_empty_directory(self):
        versions = _parse_opam_versions([], "tls")
        assert versions == []

    def test_skips_entries_without_version(self):
        entries = [{"name": "tls.", "type": "dir"}]
        versions = _parse_opam_versions(entries, "tls")
        assert versions == []

    def test_handles_missing_name_field(self):
        entries = [{"type": "dir"}]
        versions = _parse_opam_versions(entries, "tls")
        assert versions == []

# ===========================================================================
# TestPickLatestOpamVersion
# ===========================================================================


class TestPickLatestOpamVersion:
    def test_picks_latest_stable(self):
        versions = ["2.1.1", "2.1.0", "2.0.4", "0.10.4"]
        assert _pick_latest_opam_version(versions) == "2.1.1"

    def test_prefers_stable_over_rc(self):
        versions = ["2.2.0-beta1", "2.1.1", "2.1.0"]
        # 2.2.0-beta1 is not stable (has a dash), so 2.1.1 is preferred
        assert _pick_latest_opam_version(versions) == "2.1.1"

    def test_falls_back_to_first_if_no_stable(self):
        versions = ["2.0.0-alpha", "1.0.0-rc1"]
        assert _pick_latest_opam_version(versions) == "2.0.0-alpha"

    def test_empty_list_returns_empty_string(self):
        assert _pick_latest_opam_version([]) == ""

    def test_single_version(self):
        assert _pick_latest_opam_version(["1.2.3"]) == "1.2.3"

    def test_all_stable_picks_first(self):
        versions = ["3.0.0", "2.9.9", "1.0.0"]
        assert _pick_latest_opam_version(versions) == "3.0.0"

# ===========================================================================
# TestParseOpamFile
# ===========================================================================


class TestParseOpamFile:
    def test_extracts_synopsis(self):
        meta = _parse_opam_file(_TLS_OPAM_CONTENT)
        assert meta["synopsis"] == "Transport Layer Security purely in OCaml"

    def test_extracts_homepage(self):
        meta = _parse_opam_file(_TLS_OPAM_CONTENT)
        assert meta["homepage"] == "https://github.com/mirleft/ocaml-tls"

    def test_extracts_license(self):
        meta = _parse_opam_file(_TLS_OPAM_CONTENT)
        assert meta["license"] == "BSD-2-Clause"

    def test_counts_real_depends_excludes_meta(self):
        meta = _parse_opam_file(_TLS_OPAM_CONTENT)
        # ocaml, dune excluded; mirage-crypto, x509, fmt, logs, ipaddr,
        # digestif, alcotest, cmdliner = 8
        assert meta["depends_count"] == 8

    def test_empty_content_returns_empty_dict(self):
        meta = _parse_opam_file("")
        assert meta == {}

    def test_missing_synopsis_not_in_result(self):
        meta = _parse_opam_file("opam-version: \"2.0\"\n")
        assert "synopsis" not in meta

    def test_depends_with_only_meta_packages(self):
        content = "depends: [\n  \"ocaml\"\n  \"dune\"\n  \"jbuilder\"\n]\n"
        meta = _parse_opam_file(content)
        assert meta.get("depends_count", 0) == 0

    def test_depends_excludes_conditional_marker_strings(self):
        content = "depends: [\n  \"foo\" {>= \"1.0\"}\n  \"{fake}\"\n]\n"
        meta = _parse_opam_file(content)
        # "{fake}" should be excluded (starts with "{")
        assert meta.get("depends_count", 0) == 1

# ===========================================================================
# TestEnrichOpam
# ===========================================================================


class TestEnrichOpam:
    """Tests for _enrich_opam — all HTTP and subprocess calls are mocked."""

    def _make_mock_get(self, versions_resp, opam_file_resp=None):
        """Build a side_effect list for patched _get."""
        side_effects = [versions_resp]
        if opam_file_resp is not None:
            side_effects.append(opam_file_resp)
        return side_effects

    def test_basic_enrichment_returns_dict(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[
                _TLS_VERSIONS,
                _make_opam_file_response(_TLS_OPAM_CONTENT),
            ],
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"total_count": 20}))
            result = _enrich_opam("tls")
        assert result["ecosystem"] == "OPAM"
        assert result["package_name"] == "tls"

    def test_version_count_extracted(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[
                _TLS_VERSIONS,
                _make_opam_file_response(_TLS_OPAM_CONTENT),
            ],
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"total_count": 20}))
            result = _enrich_opam("tls")
        assert result["version_count"] == 7

    def test_latest_version_picked(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[
                _TLS_VERSIONS,
                _make_opam_file_response(_TLS_OPAM_CONTENT),
            ],
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"total_count": 20}))
            result = _enrich_opam("tls")
        assert result["latest_version"] == "2.1.1"

    def test_synopsis_extracted_from_opam_file(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[
                _TLS_VERSIONS,
                _make_opam_file_response(_TLS_OPAM_CONTENT),
            ],
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"total_count": 20}))
            result = _enrich_opam("tls")
        assert result["synopsis"] == "Transport Layer Security purely in OCaml"

    def test_homepage_extracted(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[
                _TLS_VERSIONS,
                _make_opam_file_response(_TLS_OPAM_CONTENT),
            ],
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"total_count": 20}))
            result = _enrich_opam("tls")
        assert result["homepage"] == "https://github.com/mirleft/ocaml-tls"

    def test_license_extracted(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[
                _TLS_VERSIONS,
                _make_opam_file_response(_TLS_OPAM_CONTENT),
            ],
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"total_count": 20}))
            result = _enrich_opam("tls")
        assert result["license"] == "BSD-2-Clause"

    def test_reverse_dep_count_from_github_search(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[
                _TLS_VERSIONS,
                _make_opam_file_response(_TLS_OPAM_CONTENT),
            ],
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"total_count": 20}))
            result = _enrich_opam("tls")
        # 20 total - 7 own versions = 13
        assert result["dependent_packages_count"] == 13

    def test_opam_page_url_present(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[_TLS_VERSIONS, _make_opam_file_response(_TLS_OPAM_CONTENT)],
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"total_count": 5}))
            result = _enrich_opam("tls")
        assert "opam.ocaml.org" in result["opam_page"]
        assert "tls" in result["opam_page"]

    def test_version_list_fetch_failure_returns_minimal(self):
        """If the GitHub API returns an error, we return a minimal dict."""
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=Exception("network error"),
        ):
            result = _enrich_opam("tls")
        assert result["ecosystem"] == "OPAM"
        assert result["package_name"] == "tls"
        assert "version_count" not in result

    def test_not_a_list_response_returns_minimal(self):
        """GitHub Contents API returning a dict (e.g. 404 message) is handled."""
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            return_value={"message": "Not Found"},
        ):
            result = _enrich_opam("unknown-pkg-xyz")
        assert result["ecosystem"] == "OPAM"
        assert "version_count" not in result

    def test_opam_file_fetch_failure_does_not_crash(self):
        """opam file fetch failing is non-fatal; version_count still populated."""
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[_TLS_VERSIONS, Exception("404 Not Found")],
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"total_count": 10}))
            result = _enrich_opam("tls")
        assert result["version_count"] == 7
        assert "synopsis" not in result

    def test_subprocess_failure_does_not_crash(self):
        """gh CLI unavailable is non-fatal; dependent_packages_count not set."""
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[_TLS_VERSIONS, _make_opam_file_response(_TLS_OPAM_CONTENT)],
        ), patch("subprocess.run", side_effect=Exception("gh not found")):
            result = _enrich_opam("tls")
        assert result["version_count"] == 7
        assert "dependent_packages_count" not in result

    def test_subprocess_nonzero_returncode_does_not_set_dep_count(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[_TLS_VERSIONS, _make_opam_file_response(_TLS_OPAM_CONTENT)],
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = _enrich_opam("tls")
        assert "dependent_packages_count" not in result

    def test_github_search_total_count_none_does_not_set_dep_count(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[_TLS_VERSIONS, _make_opam_file_response(_TLS_OPAM_CONTENT)],
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"total_count": None}))
            result = _enrich_opam("tls")
        assert "dependent_packages_count" not in result

    def test_dep_count_cannot_go_below_zero(self):
        """total_count < own version count should clamp to 0."""
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[_TLS_VERSIONS, _make_opam_file_response(_TLS_OPAM_CONTENT)],
        ), patch("subprocess.run") as mock_run:
            # total_count=3 < 7 versions => max(0, 3-7) = 0
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"total_count": 3}))
            result = _enrich_opam("tls")
        assert result["dependent_packages_count"] == 0

    def test_opam_file_without_content_key_handled(self):
        """GitHub Contents response without content field is handled gracefully."""
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            side_effect=[_TLS_VERSIONS, {"encoding": "base64"}],
        ), patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps({"total_count": 10}))
            result = _enrich_opam("tls")
        assert result["version_count"] == 7
        assert "synopsis" not in result

    def test_empty_version_list_returns_early(self):
        """Package directory exists but has no valid version entries."""
        empty_versions = [{"name": "other-pkg.1.0", "type": "dir"}]
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._get",
            return_value=empty_versions,
        ):
            result = _enrich_opam("tls")
        # version_count=0, no further calls
        assert result["version_count"] == 0
        assert "latest_version" not in result

# ===========================================================================
# TestEnrichPackageDispatch
# ===========================================================================


class TestEnrichPackageDispatchOpam:
    """Verify that _enrich_package routes OPAM ecosystem aliases correctly."""

    @pytest.mark.parametrize("ecosystem", ["opam", "OPAM", "ocaml", "caml"])
    def test_opam_aliases_routed(self, ecosystem):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_opam"
        ) as mock:
            mock.return_value = {"ecosystem": "OPAM", "package_name": "tls"}
            result = _enrich_package("tls", ecosystem)
        mock.assert_called_once_with("tls")
        assert result["ecosystem"] == "OPAM"

    def test_non_opam_ecosystem_not_routed(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_opam"
        ) as mock:
            _enrich_package("requests", "pypi")
        mock.assert_not_called()


# ===========================================================================
# TestBlastScoreOpam
# ===========================================================================


class TestBlastScoreOpam:
    """Verify blast-radius scoring with OPAM-style dependent_packages_count."""

    def test_high_rev_dep_count_scores_critical(self):
        from manus_agent.tools.get_dependency_blast_radius import _blast_score

        result = _blast_score({"ecosystem": "OPAM", "dependent_packages_count": 60000})
        assert result == "CRITICAL"

    def test_medium_rev_dep_count_scores_high(self):
        from manus_agent.tools.get_dependency_blast_radius import _blast_score

        result = _blast_score({"ecosystem": "OPAM", "dependent_packages_count": 10000})
        assert result == "HIGH"

    def test_small_rev_dep_count_scores_medium(self):
        from manus_agent.tools.get_dependency_blast_radius import _blast_score

        result = _blast_score({"ecosystem": "OPAM", "dependent_packages_count": 600})
        assert result == "MEDIUM"

    def test_tiny_rev_dep_count_scores_low(self):
        from manus_agent.tools.get_dependency_blast_radius import _blast_score

        result = _blast_score({"ecosystem": "OPAM", "dependent_packages_count": 5})
        assert result == "LOW"

    def test_zero_rev_dep_count_scores_unknown(self):
        from manus_agent.tools.get_dependency_blast_radius import _blast_score

        result = _blast_score({"ecosystem": "OPAM", "dependent_packages_count": 0})
        assert result == "UNKNOWN"
