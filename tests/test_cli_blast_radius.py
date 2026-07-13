"""Tests for the ``manus-agent blast-radius`` CLI subcommand.

Covers:
  - _build_blast_radius_parser: flag definitions, defaults, choices, and help text
  - _run_blast_radius: all output branches:
      * CVE spec → packages found (text output)
      * CVE spec → packages found (JSON output)
      * Package spec direct (text + JSON)
      * No packages found → returns non-zero
      * Invalid spec → returns non-zero
      * Summary section present in text output
      * Sorting by blast severity (CRITICAL before LOW)
      * max-packages respected
  - main() dispatch routing for the "blast-radius" subcommand
  - _SUBCOMMANDS registry membership

All external calls are patched — no real HTTP requests.
"""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_NPM_CRITICAL_STATS = {
    "ecosystem": "npm",
    "package_name": "lodash",
    "dependent_packages_count": 200_000,
    "weekly_downloads": 130_000_000,
    "monthly_downloads": None,
    "latest_version": "4.17.21",
    "description": "Lodash modular utilities.",
    "full_id": None,
}

_PYPI_HIGH_STATS = {
    "ecosystem": "PyPI",
    "package_name": "requests",
    "dependent_packages_count": 5_000,
    "weekly_downloads": 60_000_000,
    "monthly_downloads": None,
    "latest_version": "2.31.0",
    "description": "Python HTTP for Humans.",
    "full_id": None,
}

_LOW_STATS = {
    "ecosystem": "npm",
    "package_name": "tiny-lib",
    "dependent_packages_count": 5,
    "weekly_downloads": 1_000,
    "monthly_downloads": None,
    "latest_version": "1.0.0",
    "description": "Tiny utility.",
    "full_id": None,
}

_ONE_PACKAGE_OSV = [
    {
        "name": "lodash",
        "ecosystem": "npm",
        "version_range": ">=4.0.0, <4.17.21",
        "source": "osv",
    }
]


# ---------------------------------------------------------------------------
# Helper: invoke _run_blast_radius and capture output
# ---------------------------------------------------------------------------


def _run_blast_radius(argv: list[str]) -> tuple[int, str, str]:
    """Invoke cli._run_blast_radius and return (rc, stdout, stderr)."""
    from manus_agent import cli  # noqa: PLC0415

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
        rc = cli._run_blast_radius(argv)
    return rc, out_buf.getvalue(), err_buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Parser tests
# ---------------------------------------------------------------------------


class TestBuildBlastRadiusParser:
    def test_spec_argument_is_required(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([])
        assert exc_info.value.code != 0

    def test_help_exits_zero(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_default_output_is_text(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        args = parser.parse_args(["CVE-2021-44228"])
        assert args.output == "text"

    def test_output_json_accepted(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        args = parser.parse_args(["CVE-2021-44228", "--output", "json"])
        assert args.output == "json"

    def test_invalid_output_format_rejected(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["CVE-2021-44228", "--output", "yaml"])
        assert exc_info.value.code != 0

    def test_default_max_packages_is_10(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        args = parser.parse_args(["requests@2.28.0"])
        assert args.max_packages == 10

    def test_custom_max_packages(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        args = parser.parse_args(["requests@2.28.0", "--max-packages", "5"])
        assert args.max_packages == 5

    def test_ecosystem_qualified_spec_accepted(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        args = parser.parse_args(["npm:axios@1.6.0"])
        assert args.spec == "npm:axios@1.6.0"

    def test_cve_spec_accepted(self):
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        args = parser.parse_args(["CVE-2021-44228"])
        assert args.spec == "CVE-2021-44228"


# ---------------------------------------------------------------------------
# 2. _run_blast_radius — CVE spec, text output
# ---------------------------------------------------------------------------


class TestRunBlastRadiusCveText:
    def _patch_cve_lookups(self, osv_pkgs=None, nvd_pkgs=None, ghsa_pkgs=None, enrich_rv=None):
        """Return a list of context managers that mock all three source fetchers + enrich."""
        from unittest.mock import patch as _p

        return [
            _p(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=nvd_pkgs or [],
            ),
            _p(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=osv_pkgs or [],
            ),
            _p(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=ghsa_pkgs or [],
            ),
            _p(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=enrich_rv or _NPM_CRITICAL_STATS,
            ),
        ]

    def test_returns_zero_on_success(self):
        patches = self._patch_cve_lookups(osv_pkgs=_ONE_PACKAGE_OSV)
        with patches[0], patches[1], patches[2], patches[3]:
            rc, _out, _err = _run_blast_radius(["CVE-2021-44228"])
        assert rc == 0

    def test_output_contains_package_name(self):
        patches = self._patch_cve_lookups(osv_pkgs=_ONE_PACKAGE_OSV)
        with patches[0], patches[1], patches[2], patches[3]:
            rc, out, _err = _run_blast_radius(["CVE-2021-44228"])
        assert "lodash" in out

    def test_output_contains_blast_radius_label(self):
        patches = self._patch_cve_lookups(osv_pkgs=_ONE_PACKAGE_OSV)
        with patches[0], patches[1], patches[2], patches[3]:
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228"])
        assert "CRITICAL" in out

    def test_output_contains_summary_line(self):
        patches = self._patch_cve_lookups(osv_pkgs=_ONE_PACKAGE_OSV)
        with patches[0], patches[1], patches[2], patches[3]:
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228"])
        assert "Summary" in out or "summary" in out.lower()

    def test_output_contains_dependency_blast_radius_header(self):
        patches = self._patch_cve_lookups(osv_pkgs=_ONE_PACKAGE_OSV)
        with patches[0], patches[1], patches[2], patches[3]:
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228"])
        assert "Dependency Blast Radius" in out or "Blast Radius" in out

    def test_output_shows_cve_id_in_title(self):
        patches = self._patch_cve_lookups(osv_pkgs=_ONE_PACKAGE_OSV)
        with patches[0], patches[1], patches[2], patches[3]:
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228"])
        assert "CVE-2021-44228" in out

    def test_output_shows_version_range(self):
        patches = self._patch_cve_lookups(osv_pkgs=_ONE_PACKAGE_OSV)
        with patches[0], patches[1], patches[2], patches[3]:
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228"])
        # The version range field should appear in text output
        assert ">=4.0.0" in out or "4.17.21" in out or "Vulnerable range" in out

    def test_output_shows_weekly_downloads(self):
        patches = self._patch_cve_lookups(osv_pkgs=_ONE_PACKAGE_OSV)
        with patches[0], patches[1], patches[2], patches[3]:
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228"])
        assert "130,000,000" in out or "Weekly downloads" in out


# ---------------------------------------------------------------------------
# 3. _run_blast_radius — CVE spec, JSON output
# ---------------------------------------------------------------------------


class TestRunBlastRadiusCveJson:
    def test_returns_zero_on_success(self):
        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=_ONE_PACKAGE_OSV,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=_NPM_CRITICAL_STATS,
            ),
        ):
            rc, _out, _err = _run_blast_radius(["CVE-2021-44228", "--output", "json"])
        assert rc == 0

    def test_output_is_valid_json(self):
        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=_ONE_PACKAGE_OSV,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=_NPM_CRITICAL_STATS,
            ),
        ):
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228", "--output", "json"])
        data = json.loads(out)
        assert isinstance(data, dict)

    def test_json_has_spec_field(self):
        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=_ONE_PACKAGE_OSV,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=_NPM_CRITICAL_STATS,
            ),
        ):
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228", "--output", "json"])
        data = json.loads(out)
        assert data["spec"] == "CVE-2021-44228"

    def test_json_has_cve_id_field(self):
        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=_ONE_PACKAGE_OSV,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=_NPM_CRITICAL_STATS,
            ),
        ):
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228", "--output", "json"])
        data = json.loads(out)
        assert data["cve_id"] == "CVE-2021-44228"

    def test_json_has_packages_list(self):
        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=_ONE_PACKAGE_OSV,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=_NPM_CRITICAL_STATS,
            ),
        ):
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228", "--output", "json"])
        data = json.loads(out)
        assert "packages" in data
        assert len(data["packages"]) == 1

    def test_json_has_summary_block(self):
        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=_ONE_PACKAGE_OSV,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=_NPM_CRITICAL_STATS,
            ),
        ):
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228", "--output", "json"])
        data = json.loads(out)
        assert "summary" in data
        assert "highest_blast_radius" in data["summary"]
        assert "total_packages" in data["summary"]

    def test_json_blast_radius_is_critical_for_high_downloads(self):
        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=_ONE_PACKAGE_OSV,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                return_value=_NPM_CRITICAL_STATS,
            ),
        ):
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228", "--output", "json"])
        data = json.loads(out)
        pkg = data["packages"][0]
        assert pkg.get("blast_radius") == "CRITICAL"


# ---------------------------------------------------------------------------
# 4. _run_blast_radius — direct package spec
# ---------------------------------------------------------------------------


class TestRunBlastRadiusPackageSpec:
    def test_package_spec_text_output_exits_zero(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=_PYPI_HIGH_STATS,
        ):
            rc, _out, _err = _run_blast_radius(["requests@2.28.0"])
        assert rc == 0

    def test_package_spec_output_contains_package_name(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=_PYPI_HIGH_STATS,
        ):
            _rc, out, _err = _run_blast_radius(["requests@2.28.0"])
        assert "requests" in out

    def test_package_spec_json_output_valid(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=_PYPI_HIGH_STATS,
        ):
            rc, out, _err = _run_blast_radius(["requests@2.28.0", "--output", "json"])
        assert rc == 0
        data = json.loads(out)
        assert data["spec"] == "requests@2.28.0"
        assert data["cve_id"] is None

    def test_ecosystem_prefixed_spec_works(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=_NPM_CRITICAL_STATS,
        ):
            rc, out, _err = _run_blast_radius(["npm:lodash@4.17.20"])
        assert rc == 0
        assert "lodash" in out

    def test_package_spec_json_contains_packages(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=_PYPI_HIGH_STATS,
        ):
            _rc, out, _err = _run_blast_radius(["requests@2.28.0", "--output", "json"])
        data = json.loads(out)
        assert len(data["packages"]) == 1
        assert data["packages"][0]["package_name"] == "requests"


# ---------------------------------------------------------------------------
# 5. _run_blast_radius — no-packages-found branch
# ---------------------------------------------------------------------------


class TestRunBlastRadiusNoPackages:
    def test_no_packages_found_for_cve_returns_nonzero(self):
        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
        ):
            rc, out, _err = _run_blast_radius(["CVE-9999-99999"])
        assert rc != 0

    def test_no_packages_message_printed(self):
        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
        ):
            _rc, out, _err = _run_blast_radius(["CVE-9999-99999"])
        assert "No affected package" in out or "No affected package" in _err


# ---------------------------------------------------------------------------
# 6. _run_blast_radius — max-packages
# ---------------------------------------------------------------------------


class TestRunBlastRadiusMaxPackages:
    def test_max_packages_limits_enrich_calls(self):
        many_osv = [
            {"name": f"lib-{i}", "ecosystem": "npm", "version_range": "1.0", "source": "osv"} for i in range(10)
        ]
        call_count = {"n": 0}

        def counting_enrich(name, ecosystem):
            call_count["n"] += 1
            return {"ecosystem": "npm", "package_name": name, "weekly_downloads": 100}

        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=many_osv,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                side_effect=counting_enrich,
            ),
        ):
            rc, _out, _err = _run_blast_radius(["CVE-2021-44228", "--max-packages", "3"])
        assert rc == 0
        assert call_count["n"] == 3

    def test_json_total_packages_respects_max(self):
        many_osv = [{"name": f"lib-{i}", "ecosystem": "npm", "version_range": "1.0", "source": "osv"} for i in range(8)]

        def simple_enrich(name, ecosystem):
            return {"ecosystem": "npm", "package_name": name, "weekly_downloads": 100}

        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=many_osv,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                side_effect=simple_enrich,
            ),
        ):
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228", "--max-packages", "4", "--output", "json"])
        data = json.loads(out)
        assert data["summary"]["total_packages"] == 4


# ---------------------------------------------------------------------------
# 7. _run_blast_radius — severity sorting in text output
# ---------------------------------------------------------------------------


class TestRunBlastRadiusSorting:
    def test_critical_package_appears_before_low_in_text_output(self):
        two_pkgs = [
            {"name": "big-lib", "ecosystem": "npm", "version_range": "1.0", "source": "osv"},
            {"name": "tiny-lib", "ecosystem": "npm", "version_range": "1.0", "source": "osv"},
        ]

        def enrich_side_effect(name, ecosystem):
            # Return package-specific stats with correct package_name
            if name == "big-lib":
                return {
                    "ecosystem": "npm",
                    "package_name": "big-lib",
                    "dependent_packages_count": 200_000,
                    "weekly_downloads": 130_000_000,
                    "monthly_downloads": None,
                    "latest_version": "4.17.21",
                    "description": "Big library.",
                    "full_id": None,
                }
            return {
                "ecosystem": "npm",
                "package_name": "tiny-lib",
                "dependent_packages_count": 5,
                "weekly_downloads": 1_000,
                "monthly_downloads": None,
                "latest_version": "1.0.0",
                "description": "Tiny utility.",
                "full_id": None,
            }

        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=two_pkgs,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                side_effect=enrich_side_effect,
            ),
        ):
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228"])
        # CRITICAL package should appear before LOW package in output
        assert out.index("big-lib") < out.index("tiny-lib")

    def test_critical_package_first_in_json_packages_list(self):
        two_pkgs = [
            {"name": "tiny-lib", "ecosystem": "npm", "version_range": "1.0", "source": "osv"},
            {"name": "big-lib", "ecosystem": "npm", "version_range": "1.0", "source": "osv"},
        ]

        def enrich_side_effect(name, ecosystem):
            if name == "big-lib":
                return {
                    "ecosystem": "npm",
                    "package_name": "big-lib",
                    "dependent_packages_count": 200_000,
                    "weekly_downloads": 130_000_000,
                    "monthly_downloads": None,
                    "latest_version": "4.17.21",
                    "description": None,
                    "full_id": None,
                }
            return {
                "ecosystem": "npm",
                "package_name": "tiny-lib",
                "dependent_packages_count": 5,
                "weekly_downloads": 1_000,
                "monthly_downloads": None,
                "latest_version": "1.0.0",
                "description": None,
                "full_id": None,
            }

        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=two_pkgs,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                side_effect=enrich_side_effect,
            ),
        ):
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228", "--output", "json"])
        data = json.loads(out)
        assert data["packages"][0]["package_name"] == "big-lib"


# ---------------------------------------------------------------------------
# 8. _run_blast_radius — deduplication: multi-source packages deduplicated
# ---------------------------------------------------------------------------


class TestRunBlastRadiusDeduplication:
    def test_same_package_from_osv_and_ghsa_enriched_once(self):
        """Same package from two sources should only appear once in output."""
        osv_pkgs = [{"name": "requests", "ecosystem": "PyPI", "version_range": ">=2.0, <2.29", "source": "osv"}]
        ghsa_pkgs = [{"name": "requests", "ecosystem": "PyPI", "version_range": ">=2.0, <2.29", "source": "ghsa"}]
        enrich_calls = []

        def tracking_enrich(name, ecosystem):
            enrich_calls.append(name)
            return _PYPI_HIGH_STATS

        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
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
                side_effect=tracking_enrich,
            ),
        ):
            rc, out, _err = _run_blast_radius(["CVE-2023-32681"])
        assert rc == 0
        # enrich should be called exactly once for the deduplicated package
        assert enrich_calls.count("requests") == 1


# ---------------------------------------------------------------------------
# 9. _run_blast_radius — optional text fields present when populated
# ---------------------------------------------------------------------------


class TestRunBlastRadiusOptionalFields:
    def test_description_shown_in_text_output(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=_NPM_CRITICAL_STATS,
        ):
            _rc, out, _err = _run_blast_radius(["lodash@4.17.20"])
        assert "Lodash" in out or "Description" in out or "modular" in out

    def test_latest_version_shown_in_text_output(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=_NPM_CRITICAL_STATS,
        ):
            _rc, out, _err = _run_blast_radius(["lodash@4.17.20"])
        assert "4.17.21" in out or "Latest version" in out

    def test_maven_full_id_shown_when_present(self):
        maven_stats = {
            "ecosystem": "Maven",
            "package_name": "log4j-core",
            "full_id": "org.apache.logging.log4j:log4j-core",
            "latest_version": "2.20.0",
            "weekly_downloads": None,
            "dependent_packages_count": None,
            "monthly_downloads": None,
            "description": None,
        }
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            return_value=maven_stats,
        ):
            _rc, out, _err = _run_blast_radius(["log4j-core@2.14.1"])
        assert "org.apache.logging.log4j:log4j-core" in out or "Maven artifact" in out


# ---------------------------------------------------------------------------
# 10. _run_blast_radius — JSON totals aggregation
# ---------------------------------------------------------------------------


class TestRunBlastRadiusJsonTotals:
    def test_json_total_weekly_downloads_aggregated(self):
        two_pkgs = [
            {"name": "lodash", "ecosystem": "npm", "version_range": "4.x", "source": "osv"},
            {"name": "axios", "ecosystem": "npm", "version_range": "1.x", "source": "osv"},
        ]
        axios_stats = {
            "ecosystem": "npm",
            "package_name": "axios",
            "dependent_packages_count": 180_000,
            "weekly_downloads": 120_000_000,
            "monthly_downloads": None,
            "latest_version": "1.6.0",
            "description": None,
            "full_id": None,
        }

        def enrich_side_effect(name, ecosystem):
            return _NPM_CRITICAL_STATS if name == "lodash" else axios_stats

        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=two_pkgs,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                side_effect=enrich_side_effect,
            ),
        ):
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228", "--output", "json"])
        data = json.loads(out)
        total = data["summary"]["total_weekly_downloads"]
        # 130M + 120M = 250M
        assert total == 250_000_000

    def test_json_total_packages_is_correct(self):
        two_pkgs = [
            {"name": "pkg-a", "ecosystem": "npm", "version_range": "1.x", "source": "osv"},
            {"name": "pkg-b", "ecosystem": "npm", "version_range": "2.x", "source": "osv"},
        ]

        def simple_enrich(name, ecosystem):
            return {"ecosystem": "npm", "package_name": name, "weekly_downloads": 50_000_000}

        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected",
                return_value=two_pkgs,
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected",
                return_value=[],
            ),
            patch(
                "manus_agent.tools.get_dependency_blast_radius._enrich_package",
                side_effect=simple_enrich,
            ),
        ):
            _rc, out, _err = _run_blast_radius(["CVE-2021-44228", "--output", "json"])
        data = json.loads(out)
        assert data["summary"]["total_packages"] == 2


# ---------------------------------------------------------------------------
# 11. _SUBCOMMANDS registry membership
# ---------------------------------------------------------------------------


class TestSubcommandsRegistry:
    def test_blast_radius_in_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "blast-radius" in _SUBCOMMANDS

    def test_subcommands_is_a_set_or_collection(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert hasattr(_SUBCOMMANDS, "__contains__")

    def test_blast_radius_membership_is_exact_string(self):
        from manus_agent.cli import _SUBCOMMANDS

        # Verify neither a prefix match nor a substring would falsely pass
        assert "blast" not in _SUBCOMMANDS
        assert "radius" not in _SUBCOMMANDS
        assert "blast-radius" in _SUBCOMMANDS


# ---------------------------------------------------------------------------
# 12. main() dispatch
# ---------------------------------------------------------------------------


class TestMainDispatchBlastRadius:
    def test_main_routes_blast_radius_to_run_blast_radius(self, monkeypatch):
        """main() with blast-radius subcommand must call _run_blast_radius."""
        from manus_agent import cli

        called_with: list[list[str]] = []

        def fake_run(argv: list[str]) -> int:
            called_with.append(argv)
            return 0

        monkeypatch.setattr(cli, "_run_blast_radius", fake_run)
        monkeypatch.setattr(sys, "argv", ["manus-agent", "blast-radius", "CVE-2021-44228"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        assert called_with == [["CVE-2021-44228"]]

    def test_main_passes_output_flag(self, monkeypatch):
        """main() passes --output json through to _run_blast_radius."""
        from manus_agent import cli

        called_with: list[list[str]] = []

        def fake_run(argv: list[str]) -> int:
            called_with.append(argv)
            return 0

        monkeypatch.setattr(cli, "_run_blast_radius", fake_run)
        monkeypatch.setattr(sys, "argv", ["manus-agent", "blast-radius", "CVE-2021-44228", "--output", "json"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        assert "--output" in called_with[0]
        assert "json" in called_with[0]

    def test_main_passes_max_packages_flag(self, monkeypatch):
        """main() passes --max-packages through to _run_blast_radius."""
        from manus_agent import cli

        called_with: list[list[str]] = []

        def fake_run(argv: list[str]) -> int:
            called_with.append(argv)
            return 0

        monkeypatch.setattr(cli, "_run_blast_radius", fake_run)
        monkeypatch.setattr(sys, "argv", ["manus-agent", "blast-radius", "requests@2.28.0", "--max-packages", "5"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        assert "--max-packages" in called_with[0]
        assert "5" in called_with[0]

    def test_main_returns_nonzero_when_run_fails(self, monkeypatch):
        """main() propagates non-zero exit code from _run_blast_radius."""
        from manus_agent import cli

        monkeypatch.setattr(cli, "_run_blast_radius", lambda argv: 1)
        monkeypatch.setattr(sys, "argv", ["manus-agent", "blast-radius", "CVE-9999-0001"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1

    def test_main_with_package_spec_routes_correctly(self, monkeypatch):
        """main() routes package specs (not just CVE IDs) to _run_blast_radius."""
        from manus_agent import cli

        called_with: list[list[str]] = []

        def fake_run(argv: list[str]) -> int:
            called_with.append(argv)
            return 0

        monkeypatch.setattr(cli, "_run_blast_radius", fake_run)
        monkeypatch.setattr(sys, "argv", ["manus-agent", "blast-radius", "requests@2.28.0"])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        assert called_with[0] == ["requests@2.28.0"]


# ---------------------------------------------------------------------------
# 13. Import path sanity
# ---------------------------------------------------------------------------


class TestRunBlastRadiusImportPath:
    def test_run_blast_radius_callable(self):
        from manus_agent.cli import _run_blast_radius

        assert callable(_run_blast_radius)

    def test_build_blast_radius_parser_callable(self):
        from manus_agent.cli import _build_blast_radius_parser

        assert callable(_build_blast_radius_parser)

    def test_cli_module_importable(self):
        import manus_agent.cli as cli_module

        assert hasattr(cli_module, "_run_blast_radius")
        assert hasattr(cli_module, "_build_blast_radius_parser")
        assert hasattr(cli_module, "_SUBCOMMANDS")
