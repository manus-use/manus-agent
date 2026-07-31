"""Comprehensive test suite for _run_blast_radius CLI execution.

Tests cover:
- Parser construction (_build_blast_radius_parser)
- Input validation (CVE IDs, package specs, invalid inputs)
- CVE mode: concurrent NVD/OSV/GHSA fetching, deduplication, max_packages limit
- Direct package mode: ecosystem-qualified and bare specs
- Enrichment orchestration (_enrich_package + _blast_score per package)
- Severity sorting (CRITICAL > HIGH > MEDIUM > LOW > UNKNOWN)
- JSON output format (structure, summary calculations)
- Text output format (headers, per-package fields, summary line)
- Error handling (import failure, parse failure, no packages found)
- Edge cases (empty enrichment fields, long descriptions, zero downloads)
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestBuildBlastRadiusParser:
    """Tests for _build_blast_radius_parser."""

    def _get_parser(self):
        from manus_agent.cli import _build_blast_radius_parser

        return _build_blast_radius_parser()

    def test_parser_accepts_spec_positional(self):
        p = self._get_parser()
        args = p.parse_args(["requests@2.28.0"])
        assert args.spec == "requests@2.28.0"

    def test_parser_default_output_text(self):
        p = self._get_parser()
        args = p.parse_args(["lodash"])
        assert args.output == "text"

    def test_parser_output_json(self):
        p = self._get_parser()
        args = p.parse_args(["lodash", "--output", "json"])
        assert args.output == "json"

    def test_parser_default_max_packages(self):
        p = self._get_parser()
        args = p.parse_args(["CVE-2021-44228"])
        assert args.max_packages == 10

    def test_parser_custom_max_packages(self):
        p = self._get_parser()
        args = p.parse_args(["CVE-2021-44228", "--max-packages", "5"])
        assert args.max_packages == 5

    def test_parser_rejects_no_args(self):
        p = self._get_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_parser_rejects_invalid_output(self):
        p = self._get_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["lodash", "--output", "xml"])

    def test_parser_cve_spec(self):
        p = self._get_parser()
        args = p.parse_args(["CVE-2024-3094"])
        assert args.spec == "CVE-2024-3094"

    def test_parser_ecosystem_qualified_spec(self):
        p = self._get_parser()
        args = p.parse_args(["npm:axios@1.6.0"])
        assert args.spec == "npm:axios@1.6.0"


# ---------------------------------------------------------------------------
# Import error handling
# ---------------------------------------------------------------------------


class TestBlastRadiusImportError:
    """Tests for import failure path."""

    @patch(
        "manus_agent.cli._build_blast_radius_parser",
    )
    def test_import_error_returns_1(self, mock_parser, capsys):
        """When dependency_blast_radius module cannot be imported, return 1."""
        from manus_agent.cli import _run_blast_radius

        mock_args = MagicMock()
        mock_args.spec = "lodash"
        mock_args.max_packages = 10
        mock_args.output = "text"
        mock_parser.return_value.parse_args.return_value = mock_args

        with patch.dict(sys.modules, {"manus_agent.tools.get_dependency_blast_radius": None}):
            with patch(
                "manus_agent.cli._build_blast_radius_parser",
                return_value=MagicMock(parse_args=MagicMock(return_value=mock_args)),
            ):
                # Force ImportError by patching the import inside _run_blast_radius
                original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

                def fake_import(name, *args, **kwargs):
                    if name == "manus_agent.tools.get_dependency_blast_radius":
                        raise ImportError("no module")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=fake_import):
                    result = _run_blast_radius(["lodash"])

        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err


# ---------------------------------------------------------------------------
# Input validation (parse error)
# ---------------------------------------------------------------------------


class TestBlastRadiusParseError:
    """Tests for _parse_input ValueError paths."""

    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_invalid_spec_returns_1(self, mock_parse, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.side_effect = ValueError("Cannot parse package spec: ''")

        result = _run_blast_radius([""])
        assert result == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_whitespace_only_spec(self, mock_parse, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.side_effect = ValueError("Cannot parse package spec")
        result = _run_blast_radius(["   "])
        assert result == 1


# ---------------------------------------------------------------------------
# CVE mode tests
# ---------------------------------------------------------------------------


class TestBlastRadiusCVEMode:
    """Tests for CVE-based blast radius lookups."""

    def _mock_imports(self):
        """Return a dict of mocked blast-radius functions."""
        return {
            "_parse_input": MagicMock(
                return_value={
                    "kind": "cve",
                    "cve_id": "CVE-2021-44228",
                    "name": None,
                    "version": None,
                    "ecosystem": None,
                }
            ),
            "_fetch_nvd_affected": MagicMock(
                return_value=[{"name": "log4j-core", "ecosystem": "Maven", "version_range": "<2.15.0", "source": "NVD"}]
            ),
            "_fetch_osv_affected": MagicMock(
                return_value=[
                    {"name": "log4j-core", "ecosystem": "Maven", "version_range": "<2.15.0", "source": "OSV"},
                    {"name": "log4j-api", "ecosystem": "Maven", "version_range": "<2.15.0", "source": "OSV"},
                ]
            ),
            "_fetch_ghsa_affected": MagicMock(return_value=[]),
            "_enrich_package": MagicMock(
                return_value={
                    "package_name": "log4j-core",
                    "ecosystem": "Maven",
                    "weekly_downloads": None,
                    "monthly_downloads": 50000,
                    "dependent_packages_count": None,
                    "latest_version": "2.23.0",
                    "full_id": "org.apache.logging.log4j:log4j-core",
                    "description": "Apache Log4j core implementation",
                }
            ),
            "_blast_score": MagicMock(return_value="CRITICAL"),
            "_ECOSYSTEM_LABEL": {
                "PyPI": "PyPI (Python)",
                "npm": "npm (JavaScript/Node.js)",
                "Maven": "Maven (Java)",
                "Go": "Go modules",
                "crates.io": "crates.io (Rust)",
            },
        }

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", new_callable=lambda: MagicMock)
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_cve_mode_deduplicates_packages(
        self, mock_parse, mock_nvd, mock_osv, mock_ghsa, mock_enrich, mock_blast, mock_label, capsys
    ):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "cve",
            "cve_id": "CVE-2021-44228",
            "name": None,
            "version": None,
            "ecosystem": None,
        }
        mock_nvd.return_value = [
            {"name": "log4j-core", "ecosystem": "Maven", "version_range": "<2.15.0", "source": "NVD"}
        ]
        mock_osv.return_value = [
            {"name": "log4j-core", "ecosystem": "Maven", "version_range": "<2.15.0", "source": "OSV"},
            {"name": "log4j-api", "ecosystem": "Maven", "version_range": "<2.15.0", "source": "OSV"},
        ]
        mock_ghsa.return_value = []
        mock_enrich.return_value = {
            "package_name": "test-pkg",
            "ecosystem": "Maven",
            "weekly_downloads": 1000,
            "dependent_packages_count": 50,
        }
        mock_blast.return_value = "HIGH"
        mock_label.__getitem__ = lambda s, k: {"Maven": "Maven (Java)"}.get(k, k)
        mock_label.get = lambda k, d=None: {"Maven": "Maven (Java)"}.get(k, d)

        result = _run_blast_radius(["CVE-2021-44228"])
        assert result == 0
        # Dedup: log4j-core appears in both NVD and OSV, should be enriched only twice (2 unique packages)
        assert mock_enrich.call_count == 2

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"Maven": "Maven (Java)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_cve_mode_no_packages_returns_1(
        self, mock_parse, mock_nvd, mock_osv, mock_ghsa, mock_enrich, mock_blast, capsys
    ):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "cve",
            "cve_id": "CVE-9999-99999",
            "name": None,
            "version": None,
            "ecosystem": None,
        }
        mock_nvd.return_value = []
        mock_osv.return_value = []
        mock_ghsa.return_value = []

        result = _run_blast_radius(["CVE-9999-99999"])
        assert result == 1
        captured = capsys.readouterr()
        assert "No affected package records found" in captured.out

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"Maven": "Maven (Java)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_cve_mode_max_packages_limit(
        self, mock_parse, mock_nvd, mock_osv, mock_ghsa, mock_enrich, mock_blast, capsys
    ):
        """--max-packages truncates the package list."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "cve",
            "cve_id": "CVE-2021-44228",
            "name": None,
            "version": None,
            "ecosystem": None,
        }
        # 5 unique packages from OSV
        mock_nvd.return_value = []
        mock_osv.return_value = [
            {"name": f"pkg-{i}", "ecosystem": "npm", "version_range": "*", "source": "OSV"} for i in range(5)
        ]
        mock_ghsa.return_value = []
        mock_enrich.return_value = {
            "package_name": "pkg",
            "ecosystem": "npm",
            "weekly_downloads": 100,
            "dependent_packages_count": 10,
        }
        mock_blast.return_value = "LOW"

        result = _run_blast_radius(["CVE-2021-44228", "--max-packages", "3"])
        assert result == 0
        # Only 3 packages should be enriched
        assert mock_enrich.call_count == 3

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_cve_mode_osv_takes_precedence_over_nvd_in_dedup(
        self, mock_parse, mock_nvd, mock_osv, mock_ghsa, mock_enrich, mock_blast, capsys
    ):
        """OSV packages come first in the merge, so their data is kept during dedup."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "cve",
            "cve_id": "CVE-2024-1234",
            "name": None,
            "version": None,
            "ecosystem": None,
        }
        mock_nvd.return_value = [{"name": "axios", "ecosystem": "npm", "version_range": "<1.6.1", "source": "NVD"}]
        mock_osv.return_value = [
            {"name": "axios", "ecosystem": "npm", "version_range": ">=1.0.0,<1.6.2", "source": "OSV"}
        ]
        mock_ghsa.return_value = []
        mock_enrich.return_value = {
            "package_name": "axios",
            "ecosystem": "npm",
            "weekly_downloads": 5000000,
            "dependent_packages_count": 80000,
        }
        mock_blast.return_value = "CRITICAL"

        result = _run_blast_radius(["CVE-2024-1234", "--output", "json"])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        # Only 1 package (deduped)
        assert output["summary"]["total_packages"] == 1
        # OSV version_range kept (comes first in osv_pkgs + ghsa_pkgs + nvd_pkgs)
        assert output["packages"][0]["version_range"] == ">=1.0.0,<1.6.2"

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_cve_mode_case_insensitive_dedup(
        self, mock_parse, mock_nvd, mock_osv, mock_ghsa, mock_enrich, mock_blast, capsys
    ):
        """Deduplication is case-insensitive on package name and ecosystem."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "cve",
            "cve_id": "CVE-2024-1234",
            "name": None,
            "version": None,
            "ecosystem": None,
        }
        mock_nvd.return_value = [{"name": "Lodash", "ecosystem": "NPM", "version_range": "<4.17.21", "source": "NVD"}]
        mock_osv.return_value = [{"name": "lodash", "ecosystem": "npm", "version_range": "<4.17.21", "source": "OSV"}]
        mock_ghsa.return_value = []
        mock_enrich.return_value = {
            "package_name": "lodash",
            "ecosystem": "npm",
            "weekly_downloads": 40000000,
            "dependent_packages_count": 180000,
        }
        mock_blast.return_value = "CRITICAL"

        result = _run_blast_radius(["CVE-2024-1234", "--output", "json"])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["summary"]["total_packages"] == 1


# ---------------------------------------------------------------------------
# Direct package mode tests
# ---------------------------------------------------------------------------


class TestBlastRadiusDirectPackageMode:
    """Tests for direct package spec (non-CVE) lookups."""

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_bare_package_name(self, mock_parse, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "requests",
            "version": None,
            "ecosystem": None,
        }
        mock_enrich.return_value = {
            "package_name": "requests",
            "ecosystem": "PyPI",
            "weekly_downloads": 12000000,
            "dependent_packages_count": None,
            "monthly_downloads": 50000000,
            "latest_version": "2.31.0",
            "description": "HTTP for humans",
        }
        mock_blast.return_value = "CRITICAL"

        result = _run_blast_radius(["requests"])
        assert result == 0
        captured = capsys.readouterr()
        assert "requests" in captured.out
        assert "CRITICAL" in captured.out

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_ecosystem_qualified_spec(self, mock_parse, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "axios",
            "version": "1.6.0",
            "ecosystem": "npm",
        }
        mock_enrich.return_value = {
            "package_name": "axios",
            "ecosystem": "npm",
            "weekly_downloads": 50000000,
            "dependent_packages_count": 95000,
        }
        mock_blast.return_value = "CRITICAL"

        result = _run_blast_radius(["npm:axios@1.6.0", "--output", "json"])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["cve_id"] is None
        assert output["spec"] == "npm:axios@1.6.0"
        assert output["packages"][0]["version_range"] == "1.6.0"
        assert output["packages"][0]["source"] == "direct"

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_package_without_version_sets_all(self, mock_parse, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "lodash",
            "version": None,
            "ecosystem": None,
        }
        mock_enrich.return_value = {
            "package_name": "lodash",
            "ecosystem": "npm",
            "weekly_downloads": 40000000,
            "dependent_packages_count": 180000,
        }
        mock_blast.return_value = "CRITICAL"

        result = _run_blast_radius(["lodash", "--output", "json"])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["packages"][0]["version_range"] == "all"

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_direct_package_no_fetch_calls(self, mock_parse, mock_enrich, mock_blast, capsys):
        """Direct package mode should NOT call _fetch_nvd/osv/ghsa_affected."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "express",
            "version": "4.18.0",
            "ecosystem": "npm",
        }
        mock_enrich.return_value = {
            "package_name": "express",
            "ecosystem": "npm",
            "weekly_downloads": 30000000,
            "dependent_packages_count": 70000,
        }
        mock_blast.return_value = "CRITICAL"

        with (
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected") as mock_nvd,
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected") as mock_osv,
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected") as mock_ghsa,
        ):
            result = _run_blast_radius(["npm:express@4.18.0"])
            assert result == 0
            mock_nvd.assert_not_called()
            mock_osv.assert_not_called()
            mock_ghsa.assert_not_called()


# ---------------------------------------------------------------------------
# Sorting tests
# ---------------------------------------------------------------------------


class TestBlastRadiusSorting:
    """Tests for severity-based sorting."""

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_packages_sorted_by_severity(self, mock_parse, mock_nvd, mock_osv, mock_ghsa, mock_enrich, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "cve",
            "cve_id": "CVE-2021-44228",
            "name": None,
            "version": None,
            "ecosystem": None,
        }
        mock_nvd.return_value = []
        mock_osv.return_value = [
            {"name": "pkg-low", "ecosystem": "npm", "version_range": "*", "source": "OSV"},
            {"name": "pkg-critical", "ecosystem": "npm", "version_range": "*", "source": "OSV"},
            {"name": "pkg-medium", "ecosystem": "npm", "version_range": "*", "source": "OSV"},
        ]
        mock_ghsa.return_value = []

        # Return different blast scores for different packages
        call_count = [0]
        scores = ["LOW", "CRITICAL", "MEDIUM"]

        def enrich_side_effect(name, ecosystem):
            result = {
                "package_name": name,
                "ecosystem": ecosystem,
                "weekly_downloads": 1000,
                "dependent_packages_count": 10,
            }
            return result

        def blast_side_effect(stats):
            idx = call_count[0]
            call_count[0] += 1
            return scores[idx]

        mock_enrich.side_effect = enrich_side_effect

        with patch("manus_agent.tools.get_dependency_blast_radius._blast_score", side_effect=blast_side_effect):
            result = _run_blast_radius(["CVE-2021-44228", "--output", "json"])

        assert result == 0
        output = json.loads(capsys.readouterr().out)
        radii = [p["blast_radius"] for p in output["packages"]]
        assert radii == ["CRITICAL", "MEDIUM", "LOW"]

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_unknown_severity_sorts_last(
        self, mock_parse, mock_nvd, mock_osv, mock_ghsa, mock_enrich, mock_blast, capsys
    ):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "cve",
            "cve_id": "CVE-2021-44228",
            "name": None,
            "version": None,
            "ecosystem": None,
        }
        mock_nvd.return_value = []
        mock_osv.return_value = [
            {"name": "pkg-unknown", "ecosystem": "npm", "version_range": "*", "source": "OSV"},
            {"name": "pkg-high", "ecosystem": "npm", "version_range": "*", "source": "OSV"},
        ]
        mock_ghsa.return_value = []
        # Must return separate dicts — the code mutates each one (adds blast_radius)
        mock_enrich.side_effect = [
            {"package_name": "pkg-unknown", "ecosystem": "npm"},
            {"package_name": "pkg-high", "ecosystem": "npm"},
        ]
        mock_blast.side_effect = ["UNKNOWN", "HIGH"]

        result = _run_blast_radius(["CVE-2021-44228", "--output", "json"])

        assert result == 0
        output = json.loads(capsys.readouterr().out)
        radii = [p["blast_radius"] for p in output["packages"]]
        assert radii == ["HIGH", "UNKNOWN"]


# ---------------------------------------------------------------------------
# JSON output tests
# ---------------------------------------------------------------------------


class TestBlastRadiusJsonOutput:
    """Tests for --output json format."""

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="HIGH")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_json_structure_keys(self, mock_parse, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "express",
            "version": "4.18.0",
            "ecosystem": "npm",
        }
        mock_enrich.return_value = {
            "package_name": "express",
            "ecosystem": "npm",
            "weekly_downloads": 30000000,
            "dependent_packages_count": 70000,
        }

        result = _run_blast_radius(["npm:express@4.18.0", "--output", "json"])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert "spec" in output
        assert "cve_id" in output
        assert "packages" in output
        assert "summary" in output
        assert "highest_blast_radius" in output["summary"]
        assert "total_packages" in output["summary"]
        assert "total_weekly_downloads" in output["summary"]
        assert "total_dependent_packages" in output["summary"]

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="CRITICAL")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_json_cve_id_populated(self, mock_parse, mock_nvd, mock_osv, mock_ghsa, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "cve",
            "cve_id": "CVE-2024-3094",
            "name": None,
            "version": None,
            "ecosystem": None,
        }
        mock_nvd.return_value = [{"name": "xz", "ecosystem": "PyPI", "version_range": "5.6.0-5.6.1", "source": "NVD"}]
        mock_osv.return_value = []
        mock_ghsa.return_value = []
        mock_enrich.return_value = {
            "package_name": "xz",
            "ecosystem": "PyPI",
            "weekly_downloads": 500,
            "dependent_packages_count": 20,
        }

        result = _run_blast_radius(["CVE-2024-3094", "--output", "json"])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["cve_id"] == "CVE-2024-3094"

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="HIGH")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_json_summary_calculations(
        self, mock_parse, mock_nvd, mock_osv, mock_ghsa, mock_enrich, mock_blast, capsys
    ):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "cve",
            "cve_id": "CVE-2021-44228",
            "name": None,
            "version": None,
            "ecosystem": None,
        }
        mock_nvd.return_value = []
        mock_osv.return_value = [
            {"name": "pkg-a", "ecosystem": "npm", "version_range": "*", "source": "OSV"},
            {"name": "pkg-b", "ecosystem": "npm", "version_range": "*", "source": "OSV"},
        ]
        mock_ghsa.return_value = []

        call_count = [0]

        def enrich_side_effect(name, ecosystem):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "package_name": "pkg-a",
                    "ecosystem": "npm",
                    "weekly_downloads": 1000000,
                    "dependent_packages_count": 5000,
                }
            return {
                "package_name": "pkg-b",
                "ecosystem": "npm",
                "weekly_downloads": 2000000,
                "dependent_packages_count": 3000,
            }

        mock_enrich.side_effect = enrich_side_effect

        result = _run_blast_radius(["CVE-2021-44228", "--output", "json"])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["summary"]["total_packages"] == 2
        assert output["summary"]["total_weekly_downloads"] == 3000000
        assert output["summary"]["total_dependent_packages"] == 8000
        assert output["summary"]["highest_blast_radius"] == "HIGH"

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="LOW")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_json_null_weekly_downloads_excluded_from_sum(self, mock_parse, mock_enrich, mock_blast, capsys):
        """Packages with weekly_downloads=None should not contribute to total."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "some-pkg",
            "version": None,
            "ecosystem": None,
        }
        mock_enrich.return_value = {
            "package_name": "some-pkg",
            "ecosystem": "Maven",
            "weekly_downloads": None,
            "dependent_packages_count": 100,
        }

        result = _run_blast_radius(["some-pkg", "--output", "json"])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["summary"]["total_weekly_downloads"] == 0


# ---------------------------------------------------------------------------
# Text output tests
# ---------------------------------------------------------------------------


class TestBlastRadiusTextOutput:
    """Tests for default text output format."""

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"PyPI": "PyPI (Python)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="HIGH")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_text_header(self, mock_parse, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "requests",
            "version": "2.28.0",
            "ecosystem": "PyPI",
        }
        mock_enrich.return_value = {
            "package_name": "requests",
            "ecosystem": "PyPI",
            "weekly_downloads": 12000000,
            "dependent_packages_count": None,
        }

        result = _run_blast_radius(["pypi:requests@2.28.0"])
        assert result == 0
        out = capsys.readouterr().out
        assert "Dependency Blast Radius" in out
        assert "=" * 60 in out
        assert "Affected packages found: 1" in out

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="CRITICAL")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_text_per_package_fields(self, mock_parse, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "lodash",
            "version": "4.17.20",
            "ecosystem": "npm",
        }
        mock_enrich.return_value = {
            "package_name": "lodash",
            "ecosystem": "npm",
            "weekly_downloads": 40000000,
            "dependent_packages_count": 180000,
            "monthly_downloads": 170000000,
            "latest_version": "4.17.21",
            "description": "Lodash modular utilities",
        }

        result = _run_blast_radius(["npm:lodash@4.17.20"])
        assert result == 0
        out = capsys.readouterr().out
        assert "[1] lodash" in out
        assert "npm (JavaScript/Node.js)" in out
        assert "Blast radius:     CRITICAL" in out
        assert "Vulnerable range: 4.17.20" in out
        assert "npm dependents:" in out
        assert "180,000" in out
        assert "Weekly downloads:" in out
        assert "40,000,000" in out
        assert "Monthly downloads:" in out
        assert "Latest version:   4.17.21" in out
        assert "Lodash modular utilities" in out

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"Maven": "Maven (Java)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="HIGH")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_text_maven_artifact_shown(self, mock_parse, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "log4j-core",
            "version": "2.14.1",
            "ecosystem": "Maven",
        }
        mock_enrich.return_value = {
            "package_name": "log4j-core",
            "ecosystem": "Maven",
            "weekly_downloads": None,
            "dependent_packages_count": None,
            "full_id": "org.apache.logging.log4j:log4j-core",
        }

        result = _run_blast_radius(["maven:log4j-core@2.14.1"])
        assert result == 0
        out = capsys.readouterr().out
        assert "Maven artifact:   org.apache.logging.log4j:log4j-core" in out

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="MEDIUM")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_text_summary_line(self, mock_parse, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "express",
            "version": None,
            "ecosystem": "npm",
        }
        mock_enrich.return_value = {
            "package_name": "express",
            "ecosystem": "npm",
            "weekly_downloads": 30000000,
            "dependent_packages_count": 70000,
        }

        result = _run_blast_radius(["npm:express"])
        assert result == 0
        out = capsys.readouterr().out
        assert "Summary: highest blast radius is MEDIUM (express)" in out
        assert "Total weekly downloads:" in out
        assert "Total npm dependents:" in out

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="LOW")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_text_no_downloads_skips_summary_totals(self, mock_parse, mock_enrich, mock_blast, capsys):
        """When totals are 0, the summary download/dep lines are suppressed."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "obscure-pkg",
            "version": None,
            "ecosystem": None,
        }
        mock_enrich.return_value = {
            "package_name": "obscure-pkg",
            "ecosystem": "unknown",
            "weekly_downloads": None,
            "dependent_packages_count": 0,
        }

        result = _run_blast_radius(["obscure-pkg"])
        assert result == 0
        out = capsys.readouterr().out
        assert "Total weekly downloads:" not in out
        assert "Total npm dependents:" not in out

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="MEDIUM")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_text_cve_title_when_in_cve_mode(self, mock_parse, mock_enrich, mock_blast, capsys):
        """In CVE mode, the title uses the CVE ID."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "cve",
            "cve_id": "CVE-2021-44228",
            "name": None,
            "version": None,
            "ecosystem": None,
        }

        with (
            patch(
                "manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected",
                return_value=[{"name": "log4j-core", "ecosystem": "Maven", "version_range": "*", "source": "NVD"}],
            ),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected", return_value=[]),
            patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected", return_value=[]),
        ):
            mock_enrich.return_value = {
                "package_name": "log4j-core",
                "ecosystem": "Maven",
                "weekly_downloads": None,
                "dependent_packages_count": None,
            }
            result = _run_blast_radius(["CVE-2021-44228"])
            assert result == 0
            out = capsys.readouterr().out
            assert "Dependency Blast Radius — CVE-2021-44228" in out

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="LOW")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_text_long_description_truncated(self, mock_parse, mock_enrich, mock_blast, capsys):
        """Descriptions are truncated to 80 characters in text output."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "longdesc-pkg",
            "version": None,
            "ecosystem": None,
        }
        long_desc = "A" * 200
        mock_enrich.return_value = {
            "package_name": "longdesc-pkg",
            "ecosystem": "unknown",
            "weekly_downloads": None,
            "dependent_packages_count": None,
            "description": long_desc,
        }

        result = _run_blast_radius(["longdesc-pkg"])
        assert result == 0
        out = capsys.readouterr().out
        # Description line should contain at most 80 chars of the description
        assert "A" * 80 in out
        assert "A" * 81 not in out

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="LOW")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_text_missing_fields_not_printed(self, mock_parse, mock_enrich, mock_blast, capsys):
        """Optional fields that are None/missing should not appear in text output."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "minimal-pkg",
            "version": "1.0.0",  # version set so version_range is populated
            "ecosystem": None,
        }
        mock_enrich.return_value = {
            "package_name": "minimal-pkg",
            "ecosystem": "unknown",
            # All optional stat fields absent
        }

        result = _run_blast_radius(["minimal-pkg"])
        assert result == 0
        out = capsys.readouterr().out
        assert "npm dependents:" not in out
        assert "Weekly downloads:" not in out
        assert "Monthly downloads:" not in out
        assert "Latest version:" not in out
        assert "Maven artifact:" not in out
        assert "Description:" not in out


# ---------------------------------------------------------------------------
# Enrichment orchestration tests
# ---------------------------------------------------------------------------


class TestBlastRadiusEnrichment:
    """Tests for enrichment function calls."""

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="HIGH")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_enrich_called_with_correct_args(self, mock_parse, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "axios",
            "version": "1.6.0",
            "ecosystem": "npm",
        }
        mock_enrich.return_value = {
            "package_name": "axios",
            "ecosystem": "npm",
            "weekly_downloads": 50000000,
            "dependent_packages_count": 95000,
        }

        result = _run_blast_radius(["npm:axios@1.6.0"])
        assert result == 0
        mock_enrich.assert_called_once_with("axios", "npm")

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="MEDIUM")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_blast_score_called_with_enriched_stats(self, mock_parse, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "flask",
            "version": None,
            "ecosystem": "PyPI",
        }
        enrichment = {
            "package_name": "flask",
            "ecosystem": "PyPI",
            "weekly_downloads": 5000000,
            "dependent_packages_count": 80000,
        }
        mock_enrich.return_value = enrichment

        result = _run_blast_radius(["pypi:flask"])
        assert result == 0
        # _blast_score receives enriched dict with version_range + source added
        called_arg = mock_blast.call_args[0][0]
        assert called_arg["package_name"] == "flask"
        assert called_arg["version_range"] == "all"
        assert called_arg["source"] == "direct"

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="LOW")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_enriched_result_preserves_version_range_and_source(self, mock_parse, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "django",
            "version": "4.2.0",
            "ecosystem": "PyPI",
        }
        mock_enrich.return_value = {
            "package_name": "django",
            "ecosystem": "PyPI",
            "weekly_downloads": 3000000,
            "dependent_packages_count": 40000,
        }

        result = _run_blast_radius(["pypi:django@4.2.0", "--output", "json"])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        pkg = output["packages"][0]
        assert pkg["version_range"] == "4.2.0"
        assert pkg["source"] == "direct"
        assert pkg["blast_radius"] == "LOW"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBlastRadiusEdgeCases:
    """Edge cases and boundary conditions."""

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="LOW")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_package_with_empty_ecosystem(self, mock_parse, mock_enrich, mock_blast, capsys):
        """Package with no ecosystem defaults to 'Unknown' in text output."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "mystery-pkg",
            "version": None,
            "ecosystem": None,
        }
        mock_enrich.return_value = {
            "package_name": "mystery-pkg",
            "ecosystem": "",
            "weekly_downloads": None,
            "dependent_packages_count": None,
        }

        result = _run_blast_radius(["mystery-pkg"])
        assert result == 0
        out = capsys.readouterr().out
        # Falls through to raw ecosystem label when not in _ECOSYSTEM_LABEL
        assert "mystery-pkg" in out

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="MEDIUM")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_package_name_fallback_to_name_key(self, mock_parse, mock_enrich, mock_blast, capsys):
        """If package_name is absent, falls back to 'name' key."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "fallback-pkg",
            "version": None,
            "ecosystem": None,
        }
        mock_enrich.return_value = {
            "name": "fallback-pkg",
            "ecosystem": "npm",
            "weekly_downloads": 1000,
            "dependent_packages_count": 5,
        }

        result = _run_blast_radius(["fallback-pkg"])
        assert result == 0
        out = capsys.readouterr().out
        assert "fallback-pkg" in out

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="UNKNOWN")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_zero_downloads_shown_but_no_summary(self, mock_parse, mock_enrich, mock_blast, capsys):
        """Zero weekly downloads are falsy; summary skips them."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "dead-pkg",
            "version": None,
            "ecosystem": None,
        }
        mock_enrich.return_value = {
            "package_name": "dead-pkg",
            "ecosystem": "npm",
            "weekly_downloads": 0,
            "dependent_packages_count": 0,
        }

        result = _run_blast_radius(["dead-pkg"])
        assert result == 0
        out = capsys.readouterr().out
        # 0 downloads: weekly_downloads is not None, so "Weekly downloads:" line is shown
        assert "Weekly downloads:" in out
        # But summary total_weekly == 0 (falsy), so "Total weekly downloads:" is NOT shown
        assert "Total weekly downloads:" not in out

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="LOW")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_cve_spec_stripped(self, mock_parse, mock_nvd, mock_osv, mock_ghsa, mock_enrich, mock_blast, capsys):
        """Leading/trailing whitespace in spec is stripped."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "cve",
            "cve_id": "CVE-2021-44228",
            "name": None,
            "version": None,
            "ecosystem": None,
        }
        mock_nvd.return_value = [{"name": "log4j-core", "ecosystem": "Maven", "version_range": "*", "source": "NVD"}]
        mock_osv.return_value = []
        mock_ghsa.return_value = []
        mock_enrich.return_value = {
            "package_name": "log4j-core",
            "ecosystem": "Maven",
            "weekly_downloads": None,
            "dependent_packages_count": None,
        }

        # The spec arg has trailing space — argparse trims it, then the code calls .strip()
        result = _run_blast_radius(["  CVE-2021-44228  "])
        assert result == 0

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="CRITICAL")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_json_output_is_valid_json(self, mock_parse, mock_enrich, mock_blast, capsys):
        """Ensures the output is parseable JSON without trailing characters."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "test",
            "version": None,
            "ecosystem": None,
        }
        mock_enrich.return_value = {
            "package_name": "test",
            "ecosystem": "npm",
            "weekly_downloads": 100,
            "dependent_packages_count": 5,
        }

        result = _run_blast_radius(["test", "--output", "json"])
        assert result == 0
        raw = capsys.readouterr().out
        # Should parse without error
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {"npm": "npm (JavaScript/Node.js)"})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="CRITICAL")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_ghsa_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_osv_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._fetch_nvd_affected")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_ghsa_packages_included_in_merge(
        self, mock_parse, mock_nvd, mock_osv, mock_ghsa, mock_enrich, mock_blast, capsys
    ):
        """GHSA-exclusive packages appear in the final enrichment."""
        from manus_agent.cli import _run_blast_radius

        mock_parse.return_value = {
            "kind": "cve",
            "cve_id": "CVE-2024-5678",
            "name": None,
            "version": None,
            "ecosystem": None,
        }
        mock_nvd.return_value = []
        mock_osv.return_value = []
        mock_ghsa.return_value = [
            {"name": "ghsa-only-pkg", "ecosystem": "npm", "version_range": "<2.0.0", "source": "GHSA"}
        ]
        mock_enrich.return_value = {
            "package_name": "ghsa-only-pkg",
            "ecosystem": "npm",
            "weekly_downloads": 500,
            "dependent_packages_count": 10,
        }

        result = _run_blast_radius(["CVE-2024-5678", "--output", "json"])
        assert result == 0
        output = json.loads(capsys.readouterr().out)
        assert output["summary"]["total_packages"] == 1
        mock_enrich.assert_called_once_with("ghsa-only-pkg", "npm")


# ---------------------------------------------------------------------------
# Subcommand dispatch test
# ---------------------------------------------------------------------------


class TestBlastRadiusDispatch:
    """Test that 'blast-radius' subcommand dispatches to _run_blast_radius."""

    @patch("manus_agent.tools.get_dependency_blast_radius._ECOSYSTEM_LABEL", {})
    @patch("manus_agent.tools.get_dependency_blast_radius._blast_score", return_value="LOW")
    @patch("manus_agent.tools.get_dependency_blast_radius._enrich_package")
    @patch("manus_agent.tools.get_dependency_blast_radius._parse_input")
    def test_main_dispatches_blast_radius(self, mock_parse, mock_enrich, mock_blast, capsys):
        from manus_agent.cli import main

        mock_parse.return_value = {
            "kind": "package",
            "cve_id": None,
            "name": "test-pkg",
            "version": None,
            "ecosystem": None,
        }
        mock_enrich.return_value = {
            "package_name": "test-pkg",
            "ecosystem": "npm",
            "weekly_downloads": 100,
            "dependent_packages_count": 1,
        }

        with patch("sys.argv", ["manus-agent", "blast-radius", "test-pkg"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
