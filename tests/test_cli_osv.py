"""Comprehensive test suite for _run_osv CLI execution.

Tests cover:
- Argument parsing (CVE-ID positional, --output flag)
- Import failure handling (graceful degradation)
- fetch_osv_data invocation with correct CVE-ID
- JSON output mode (valid JSON, correct structure)
- Text output mode:
  - "not found" messaging
  - aliases display
  - per-record OSV id headers
  - severity lines
  - per-package ecosystem:name formatting
  - vulnerable-from / first-fixed / last-affected fields
  - multi-record merging (GHSA alias follow)
- Edge cases:
  - Empty/whitespace CVE-ID (parser error)
  - No aliases in payload
  - No packages in records
  - Multiple records with mixed package counts
  - Severity with missing score
  - Empty fixed/introduced/last_affected lists

All HTTP calls are fully mocked — no real network requests.
"""

from __future__ import annotations

import json
import sys
from unittest import mock

import pytest

from manus_agent import cli

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_real_import = __import__


def _make_failing_import(blocked_module: str):
    """Create an __import__ side_effect that blocks one specific module."""

    def _import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return _real_import(name, *args, **kwargs)

    return _import


def _make_osv_payload(
    *,
    found: bool = True,
    cve_id: str = "CVE-2021-44228",
    records: list | None = None,
    aliases: list | None = None,
    affected_ecosystems: list | None = None,
    message: str = "OSV.dev: CVE-2021-44228 affects 2 package(s) across 1 ecosystem(s): Maven.",
    error: str | None = None,
    affected_package_count: int = 2,
) -> dict:
    """Build a synthetic fetch_osv_data return value."""
    if records is None:
        records = [
            {
                "osv_id": "CVE-2021-44228",
                "summary": "Apache Log4j2 RCE",
                "aliases": ["CVE-2021-44228", "GHSA-jfh8-c2jp-5v3q"],
                "modified": "2024-01-15T00:00:00Z",
                "published": "2021-12-10T00:00:00Z",
                "withdrawn": None,
                "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}],
                "affected_ecosystems": ["Maven"],
                "affected_packages": [
                    {
                        "ecosystem": "Maven",
                        "package": "org.apache.logging.log4j:log4j-core",
                        "introduced": ["2.0-beta9"],
                        "fixed": ["2.15.0"],
                        "last_affected": [],
                        "affected_version_count": 35,
                        "affected_versions_sample": ["2.0", "2.1", "2.2"],
                    },
                    {
                        "ecosystem": "Maven",
                        "package": "org.apache.logging.log4j:log4j-api",
                        "introduced": ["2.0-beta9"],
                        "fixed": ["2.15.0"],
                        "last_affected": [],
                        "affected_version_count": 35,
                        "affected_versions_sample": ["2.0", "2.1"],
                    },
                ],
                "references": ["https://logging.apache.org/log4j/2.x/security.html"],
            }
        ]
    if aliases is None:
        aliases = ["CVE-2021-44228", "GHSA-jfh8-c2jp-5v3q"]
    if affected_ecosystems is None:
        affected_ecosystems = ["Maven"]

    payload: dict = {
        "found": found,
        "cve_id": cve_id,
        "records": records,
        "aliases": aliases,
        "affected_ecosystems": affected_ecosystems,
        "message": message,
    }
    if found:
        payload["affected_package_count"] = affected_package_count
    if error:
        payload["error"] = error
    return payload


def _not_found_payload(cve_id: str = "CVE-9999-0000") -> dict:
    return {
        "found": False,
        "cve_id": cve_id,
        "records": [],
        "aliases": [],
        "affected_ecosystems": [],
        "message": f"No OSV.dev record found for {cve_id}.",
    }


def _error_payload(cve_id: str = "CVE-2021-44228", error_msg: str = "timeout") -> dict:
    return {
        "found": False,
        "cve_id": cve_id,
        "records": [],
        "aliases": [],
        "affected_ecosystems": [],
        "error": f"OSV request failed: {error_msg}",
        "message": f"OSV.dev lookup failed for {cve_id}: {error_msg}",
    }


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestOsvParser:
    """Tests for _build_osv_parser argument parsing."""

    def test_parser_accepts_cve_id(self):
        p = cli._build_osv_parser()
        args = p.parse_args(["CVE-2021-44228"])
        assert args.cve_id == "CVE-2021-44228"

    def test_parser_default_output_text(self):
        p = cli._build_osv_parser()
        args = p.parse_args(["CVE-2024-3094"])
        assert args.output == "text"

    def test_parser_output_json(self):
        p = cli._build_osv_parser()
        args = p.parse_args(["CVE-2024-3094", "--output", "json"])
        assert args.output == "json"

    def test_parser_output_text_explicit(self):
        p = cli._build_osv_parser()
        args = p.parse_args(["CVE-2024-3094", "--output", "text"])
        assert args.output == "text"

    def test_parser_rejects_invalid_output(self):
        p = cli._build_osv_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["CVE-2024-3094", "--output", "yaml"])

    def test_parser_missing_cve_id(self):
        p = cli._build_osv_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_parser_help_flag(self):
        p = cli._build_osv_parser()
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Import failure
# ---------------------------------------------------------------------------


class TestOsvImportFailure:
    """Tests for graceful degradation when fetch_osv_data cannot be imported."""

    def test_import_error_returns_1(self, capsys):
        """When the import of fetch_osv_data fails, _run_osv returns 1."""
        with mock.patch(
            "manus_agent.tools.get_osv_data.fetch_osv_data",
            new_callable=lambda: mock.PropertyMock(side_effect=ImportError("test")),
        ):
            # We need to actually make the import inside _run_osv fail.
            # The function does: from manus_agent.tools.get_osv_data import fetch_osv_data
            # Patch the module to cause ImportError on access.
            pass

        # Better approach: temporarily remove the module from sys.modules and
        # patch __import__ to raise for the specific import.

        original_module = sys.modules.get("manus_agent.tools.get_osv_data")
        try:
            # Remove from cache so the import inside _run_osv is re-evaluated
            sys.modules.pop("manus_agent.tools.get_osv_data", None)
            with mock.patch(
                "builtins.__import__",
                side_effect=_make_failing_import("manus_agent.tools.get_osv_data"),
            ):
                rc = cli._run_osv(["CVE-2021-44228"])
            assert rc == 1
        finally:
            # Restore module
            if original_module is not None:
                sys.modules["manus_agent.tools.get_osv_data"] = original_module

    def test_import_error_prints_message(self, capsys):
        """Import failure prints an error message to stderr."""

        original_module = sys.modules.get("manus_agent.tools.get_osv_data")
        try:
            sys.modules.pop("manus_agent.tools.get_osv_data", None)
            with mock.patch(
                "builtins.__import__",
                side_effect=_make_failing_import("manus_agent.tools.get_osv_data"),
            ):
                rc = cli._run_osv(["CVE-2021-44228"])
        finally:
            if original_module is not None:
                sys.modules["manus_agent.tools.get_osv_data"] = original_module

        captured = capsys.readouterr()
        assert rc == 1
        assert "missing dependencies" in captured.err or "error" in captured.err.lower()


# ---------------------------------------------------------------------------
# JSON output mode
# ---------------------------------------------------------------------------


class TestOsvJsonOutput:
    """Tests for _run_osv with --output json."""

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_json_output_valid_json(self, mock_fetch, capsys):
        payload = _make_osv_payload()
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2021-44228", "--output", "json"])
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, dict)

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_json_output_preserves_structure(self, mock_fetch, capsys):
        payload = _make_osv_payload()
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2021-44228", "--output", "json"])
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["found"] is True
        assert parsed["cve_id"] == "CVE-2021-44228"
        assert len(parsed["records"]) == 1

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_json_output_not_found(self, mock_fetch, capsys):
        mock_fetch.return_value = _not_found_payload("CVE-9999-0000")
        rc = cli._run_osv(["CVE-9999-0000", "--output", "json"])
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["found"] is False

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_json_output_error_payload(self, mock_fetch, capsys):
        mock_fetch.return_value = _error_payload("CVE-2021-44228", "connection refused")
        rc = cli._run_osv(["CVE-2021-44228", "--output", "json"])
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["found"] is False
        assert "error" in parsed

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_json_output_indented(self, mock_fetch, capsys):
        mock_fetch.return_value = _make_osv_payload()
        cli._run_osv(["CVE-2021-44228", "--output", "json"])
        captured = capsys.readouterr()
        # json.dumps with indent=2 produces multi-line output
        assert "\n" in captured.out
        assert "  " in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_json_contains_aliases(self, mock_fetch, capsys):
        mock_fetch.return_value = _make_osv_payload()
        cli._run_osv(["CVE-2021-44228", "--output", "json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "GHSA-jfh8-c2jp-5v3q" in parsed["aliases"]

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_json_contains_affected_ecosystems(self, mock_fetch, capsys):
        mock_fetch.return_value = _make_osv_payload()
        cli._run_osv(["CVE-2021-44228", "--output", "json"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "Maven" in parsed["affected_ecosystems"]


# ---------------------------------------------------------------------------
# Text output mode — basic structure
# ---------------------------------------------------------------------------


class TestOsvTextOutput:
    """Tests for _run_osv text output rendering."""

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_prints_message(self, mock_fetch, capsys):
        payload = _make_osv_payload(
            message="OSV.dev: CVE-2021-44228 affects 2 package(s) across 1 ecosystem(s): Maven."
        )
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2021-44228"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "OSV.dev: CVE-2021-44228 affects 2 package(s)" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_prints_aliases(self, mock_fetch, capsys):
        payload = _make_osv_payload()
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2021-44228"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Aliases:" in captured.out
        assert "GHSA-jfh8-c2jp-5v3q" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_no_aliases_skips_line(self, mock_fetch, capsys):
        payload = _make_osv_payload(aliases=[])
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2021-44228"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Aliases:" not in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_osv_record_header(self, mock_fetch, capsys):
        payload = _make_osv_payload()
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2021-44228"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "OSV record: CVE-2021-44228" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_severity_displayed(self, mock_fetch, capsys):
        payload = _make_osv_payload()
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2021-44228"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Severity (CVSS_V3):" in captured.out
        assert "CVSS:3.1/AV:N/AC:L" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_package_ecosystem_name(self, mock_fetch, capsys):
        payload = _make_osv_payload()
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2021-44228"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Maven:org.apache.logging.log4j:log4j-core" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_vulnerable_from(self, mock_fetch, capsys):
        payload = _make_osv_payload()
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2021-44228"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Vulnerable from" in captured.out
        assert "2.0-beta9" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_first_fixed(self, mock_fetch, capsys):
        payload = _make_osv_payload()
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2021-44228"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "First fixed" in captured.out
        assert "2.15.0" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_not_found_message(self, mock_fetch, capsys):
        mock_fetch.return_value = _not_found_payload("CVE-9999-0000")
        rc = cli._run_osv(["CVE-9999-0000"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "No OSV.dev record found for CVE-9999-0000" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_not_found_no_aliases_line(self, mock_fetch, capsys):
        mock_fetch.return_value = _not_found_payload("CVE-9999-0000")
        rc = cli._run_osv(["CVE-9999-0000"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Aliases:" not in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_returns_0_on_success(self, mock_fetch):
        mock_fetch.return_value = _make_osv_payload()
        rc = cli._run_osv(["CVE-2021-44228"])
        assert rc == 0

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_returns_0_on_not_found(self, mock_fetch):
        mock_fetch.return_value = _not_found_payload()
        rc = cli._run_osv(["CVE-9999-0000"])
        assert rc == 0


# ---------------------------------------------------------------------------
# Text output — edge cases
# ---------------------------------------------------------------------------


class TestOsvTextEdgeCases:
    """Edge case tests for text output formatting."""

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_no_fixed_version(self, mock_fetch, capsys):
        """When no fixed version exists, display '(no fixed version listed)'."""
        payload = _make_osv_payload(
            records=[
                {
                    "osv_id": "CVE-2024-0001",
                    "summary": "Unfixed vuln",
                    "aliases": ["CVE-2024-0001"],
                    "modified": "2024-06-01T00:00:00Z",
                    "published": "2024-01-01T00:00:00Z",
                    "withdrawn": None,
                    "severity": [],
                    "affected_ecosystems": ["npm"],
                    "affected_packages": [
                        {
                            "ecosystem": "npm",
                            "package": "vulnerable-pkg",
                            "introduced": ["1.0.0"],
                            "fixed": [],
                            "last_affected": [],
                            "affected_version_count": 10,
                            "affected_versions_sample": [],
                        }
                    ],
                    "references": [],
                }
            ]
        )
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2024-0001"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "(no fixed version listed)" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_introduced_zero_default(self, mock_fetch, capsys):
        """When no introduced version, display '0' as default."""
        payload = _make_osv_payload(
            records=[
                {
                    "osv_id": "CVE-2024-0002",
                    "summary": "No introduced",
                    "aliases": ["CVE-2024-0002"],
                    "modified": "2024-06-01T00:00:00Z",
                    "published": "2024-01-01T00:00:00Z",
                    "withdrawn": None,
                    "severity": [],
                    "affected_ecosystems": ["PyPI"],
                    "affected_packages": [
                        {
                            "ecosystem": "PyPI",
                            "package": "some-lib",
                            "introduced": [],
                            "fixed": ["2.0.0"],
                            "last_affected": [],
                            "affected_version_count": 5,
                            "affected_versions_sample": [],
                        }
                    ],
                    "references": [],
                }
            ]
        )
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2024-0002"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Vulnerable from" in captured.out
        # "0" is the default when introduced list is empty
        assert ": 0" in captured.out or "0\n" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_last_affected_shown(self, mock_fetch, capsys):
        """When last_affected is set, display it."""
        payload = _make_osv_payload(
            records=[
                {
                    "osv_id": "CVE-2024-0003",
                    "summary": "Has last_affected",
                    "aliases": ["CVE-2024-0003"],
                    "modified": "2024-06-01T00:00:00Z",
                    "published": "2024-01-01T00:00:00Z",
                    "withdrawn": None,
                    "severity": [],
                    "affected_ecosystems": ["Go"],
                    "affected_packages": [
                        {
                            "ecosystem": "Go",
                            "package": "github.com/example/pkg",
                            "introduced": ["0.1.0"],
                            "fixed": [],
                            "last_affected": ["1.9.9"],
                            "affected_version_count": 20,
                            "affected_versions_sample": [],
                        }
                    ],
                    "references": [],
                }
            ]
        )
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2024-0003"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Last affected" in captured.out
        assert "1.9.9" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_no_last_affected_omitted(self, mock_fetch, capsys):
        """When last_affected is empty, the line is not printed."""
        payload = _make_osv_payload()
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2021-44228"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Last affected" not in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_multiple_records(self, mock_fetch, capsys):
        """Multiple records (GHSA follow) each get their own header."""
        payload = _make_osv_payload(
            records=[
                {
                    "osv_id": "CVE-2021-44228",
                    "summary": "Log4Shell",
                    "aliases": ["CVE-2021-44228", "GHSA-jfh8-c2jp-5v3q"],
                    "modified": "2024-01-15T00:00:00Z",
                    "published": "2021-12-10T00:00:00Z",
                    "withdrawn": None,
                    "severity": [],
                    "affected_ecosystems": [],
                    "affected_packages": [],
                    "references": [],
                },
                {
                    "osv_id": "GHSA-jfh8-c2jp-5v3q",
                    "summary": "Log4Shell GHSA",
                    "aliases": ["CVE-2021-44228", "GHSA-jfh8-c2jp-5v3q"],
                    "modified": "2024-01-15T00:00:00Z",
                    "published": "2021-12-10T00:00:00Z",
                    "withdrawn": None,
                    "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}],
                    "affected_ecosystems": ["Maven"],
                    "affected_packages": [
                        {
                            "ecosystem": "Maven",
                            "package": "org.apache.logging.log4j:log4j-core",
                            "introduced": ["2.0-beta9"],
                            "fixed": ["2.15.0"],
                            "last_affected": [],
                            "affected_version_count": 35,
                            "affected_versions_sample": [],
                        }
                    ],
                    "references": [],
                },
            ]
        )
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2021-44228"])
        assert rc == 0
        captured = capsys.readouterr()
        # The first record has no packages so should not show record header
        # The second record has packages so should show its header
        assert "OSV record: GHSA-jfh8-c2jp-5v3q" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_record_with_no_packages_skipped(self, mock_fetch, capsys):
        """Records with empty affected_packages should not produce package lines."""
        payload = _make_osv_payload(
            records=[
                {
                    "osv_id": "CVE-2024-0004",
                    "summary": "No packages",
                    "aliases": ["CVE-2024-0004"],
                    "modified": "2024-06-01T00:00:00Z",
                    "published": "2024-01-01T00:00:00Z",
                    "withdrawn": None,
                    "severity": [],
                    "affected_ecosystems": [],
                    "affected_packages": [],
                    "references": [],
                }
            ]
        )
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2024-0004"])
        assert rc == 0
        captured = capsys.readouterr()
        # No "OSV record:" header printed when packages are empty
        assert "OSV record:" not in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_severity_without_score_skipped(self, mock_fetch, capsys):
        """Severity entries with falsy score should not be printed."""
        payload = _make_osv_payload(
            records=[
                {
                    "osv_id": "CVE-2024-0005",
                    "summary": "No score severity",
                    "aliases": ["CVE-2024-0005"],
                    "modified": "2024-06-01T00:00:00Z",
                    "published": "2024-01-01T00:00:00Z",
                    "withdrawn": None,
                    "severity": [{"type": "CVSS_V3", "score": ""}],
                    "affected_ecosystems": ["npm"],
                    "affected_packages": [
                        {
                            "ecosystem": "npm",
                            "package": "test-pkg",
                            "introduced": ["1.0.0"],
                            "fixed": ["2.0.0"],
                            "last_affected": [],
                            "affected_version_count": 5,
                            "affected_versions_sample": [],
                        }
                    ],
                    "references": [],
                }
            ]
        )
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2024-0005"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Severity" not in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_multiple_introduced_versions(self, mock_fetch, capsys):
        """Multiple introduced versions are joined with ', '."""
        payload = _make_osv_payload(
            records=[
                {
                    "osv_id": "CVE-2024-0006",
                    "summary": "Multi-introduced",
                    "aliases": ["CVE-2024-0006"],
                    "modified": "2024-06-01T00:00:00Z",
                    "published": "2024-01-01T00:00:00Z",
                    "withdrawn": None,
                    "severity": [],
                    "affected_ecosystems": ["PyPI"],
                    "affected_packages": [
                        {
                            "ecosystem": "PyPI",
                            "package": "multi-range",
                            "introduced": ["1.0.0", "3.0.0"],
                            "fixed": ["2.0.0", "4.0.0"],
                            "last_affected": [],
                            "affected_version_count": 20,
                            "affected_versions_sample": [],
                        }
                    ],
                    "references": [],
                }
            ]
        )
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2024-0006"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "1.0.0, 3.0.0" in captured.out
        assert "2.0.0, 4.0.0" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_multiple_last_affected(self, mock_fetch, capsys):
        """Multiple last_affected versions are joined with ', '."""
        payload = _make_osv_payload(
            records=[
                {
                    "osv_id": "CVE-2024-0007",
                    "summary": "Multi-last-affected",
                    "aliases": ["CVE-2024-0007"],
                    "modified": "2024-06-01T00:00:00Z",
                    "published": "2024-01-01T00:00:00Z",
                    "withdrawn": None,
                    "severity": [],
                    "affected_ecosystems": ["RubyGems"],
                    "affected_packages": [
                        {
                            "ecosystem": "RubyGems",
                            "package": "vuln-gem",
                            "introduced": ["0"],
                            "fixed": [],
                            "last_affected": ["1.5.0", "2.3.1"],
                            "affected_version_count": 8,
                            "affected_versions_sample": [],
                        }
                    ],
                    "references": [],
                }
            ]
        )
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2024-0007"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "1.5.0, 2.3.1" in captured.out


# ---------------------------------------------------------------------------
# fetch_osv_data call verification
# ---------------------------------------------------------------------------


class TestOsvFetchInvocation:
    """Verify _run_osv passes the correct CVE-ID to fetch_osv_data."""

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_calls_fetch_with_stripped_cve(self, mock_fetch):
        mock_fetch.return_value = _not_found_payload("CVE-2024-1234")
        cli._run_osv(["  CVE-2024-1234  "])
        mock_fetch.assert_called_once_with("CVE-2024-1234")

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_calls_fetch_with_exact_id(self, mock_fetch):
        mock_fetch.return_value = _not_found_payload("CVE-2024-3094")
        cli._run_osv(["CVE-2024-3094"])
        mock_fetch.assert_called_once_with("CVE-2024-3094")

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_cve_id_case_preserved(self, mock_fetch):
        """CVE-ID casing is preserved (user might pass lowercase)."""
        mock_fetch.return_value = _not_found_payload("cve-2024-0001")
        cli._run_osv(["cve-2024-0001"])
        mock_fetch.assert_called_once_with("cve-2024-0001")


# ---------------------------------------------------------------------------
# Subcommand dispatch integration
# ---------------------------------------------------------------------------


class TestOsvSubcommandDispatch:
    """Test that 'osv' subcommand routes to _run_osv in main()."""

    @mock.patch("manus_agent.cli._run_osv", return_value=0)
    def test_osv_subcommand_dispatches(self, mock_run):
        with mock.patch.object(sys, "argv", ["manus-agent", "osv", "CVE-2021-44228"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(["CVE-2021-44228"])

    @mock.patch("manus_agent.cli._run_osv", return_value=0)
    def test_osv_subcommand_passes_all_args(self, mock_run):
        with mock.patch.object(sys, "argv", ["manus-agent", "osv", "CVE-2024-3094", "--output", "json"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(["CVE-2024-3094", "--output", "json"])


# ---------------------------------------------------------------------------
# Multi-ecosystem / multi-package scenarios
# ---------------------------------------------------------------------------


class TestOsvMultiEcosystem:
    """Tests for multi-ecosystem payloads."""

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_text_multiple_ecosystems(self, mock_fetch, capsys):
        """Multiple ecosystems are each displayed."""
        payload = _make_osv_payload(
            affected_ecosystems=["Maven", "PyPI"],
            records=[
                {
                    "osv_id": "GHSA-test-0001",
                    "summary": "Multi-ecosystem",
                    "aliases": ["CVE-2024-0010", "GHSA-test-0001"],
                    "modified": "2024-06-01T00:00:00Z",
                    "published": "2024-01-01T00:00:00Z",
                    "withdrawn": None,
                    "severity": [],
                    "affected_ecosystems": ["Maven", "PyPI"],
                    "affected_packages": [
                        {
                            "ecosystem": "Maven",
                            "package": "com.example:lib-a",
                            "introduced": ["1.0"],
                            "fixed": ["1.1"],
                            "last_affected": [],
                            "affected_version_count": 3,
                            "affected_versions_sample": [],
                        },
                        {
                            "ecosystem": "PyPI",
                            "package": "lib-b",
                            "introduced": ["0.9"],
                            "fixed": ["1.0"],
                            "last_affected": [],
                            "affected_version_count": 2,
                            "affected_versions_sample": [],
                        },
                    ],
                    "references": [],
                }
            ],
            message="OSV.dev: CVE-2024-0010 affects 2 package(s) across 2 ecosystem(s): Maven, PyPI.",
        )
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2024-0010"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "Maven:com.example:lib-a" in captured.out
        assert "PyPI:lib-b" in captured.out

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_json_multiple_ecosystems(self, mock_fetch, capsys):
        """JSON output preserves all ecosystems in structure."""
        payload = _make_osv_payload(
            affected_ecosystems=["Maven", "npm", "Go"],
            records=[
                {
                    "osv_id": "CVE-2024-0011",
                    "summary": "Three ecosystems",
                    "aliases": ["CVE-2024-0011"],
                    "modified": "2024-06-01T00:00:00Z",
                    "published": "2024-01-01T00:00:00Z",
                    "withdrawn": None,
                    "severity": [],
                    "affected_ecosystems": ["Maven", "npm", "Go"],
                    "affected_packages": [
                        {
                            "ecosystem": "Maven",
                            "package": "a:b",
                            "introduced": ["1.0"],
                            "fixed": ["2.0"],
                            "last_affected": [],
                            "affected_version_count": 5,
                            "affected_versions_sample": [],
                        },
                        {
                            "ecosystem": "npm",
                            "package": "c",
                            "introduced": ["0.1"],
                            "fixed": ["0.2"],
                            "last_affected": [],
                            "affected_version_count": 3,
                            "affected_versions_sample": [],
                        },
                        {
                            "ecosystem": "Go",
                            "package": "github.com/x/y",
                            "introduced": ["0"],
                            "fixed": ["1.0.0"],
                            "last_affected": [],
                            "affected_version_count": 1,
                            "affected_versions_sample": [],
                        },
                    ],
                    "references": [],
                }
            ],
            affected_package_count=3,
        )
        mock_fetch.return_value = payload
        rc = cli._run_osv(["CVE-2024-0011", "--output", "json"])
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert set(parsed["affected_ecosystems"]) == {"Maven", "npm", "Go"}


# ---------------------------------------------------------------------------
# Return code consistency
# ---------------------------------------------------------------------------


class TestOsvReturnCodes:
    """Verify return codes are consistent."""

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_returns_0_json_found(self, mock_fetch):
        mock_fetch.return_value = _make_osv_payload()
        assert cli._run_osv(["CVE-2021-44228", "--output", "json"]) == 0

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_returns_0_json_not_found(self, mock_fetch):
        mock_fetch.return_value = _not_found_payload()
        assert cli._run_osv(["CVE-9999-0000", "--output", "json"]) == 0

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_returns_0_text_found(self, mock_fetch):
        mock_fetch.return_value = _make_osv_payload()
        assert cli._run_osv(["CVE-2021-44228"]) == 0

    @mock.patch("manus_agent.tools.get_osv_data.fetch_osv_data")
    def test_returns_0_text_not_found(self, mock_fetch):
        mock_fetch.return_value = _not_found_payload()
        assert cli._run_osv(["CVE-9999-0000"]) == 0
