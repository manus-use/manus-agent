"""Tests for the `manus-agent osv` CLI subcommand.

Covers:
  - _build_osv_parser: flag definitions, defaults, choices, and help text
  - _run_osv: all output branches (text / JSON / not-found / error / no-packages
    / aliases / severity / last_affected / multi-record)
  - main() dispatch routing for the "osv" subcommand
  - _SUBCOMMANDS registry membership

All external calls to ``fetch_osv_data`` are patched — no real HTTP requests.
"""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_osv(argv: list[str]) -> tuple[int, str, str]:
    """Invoke _run_osv and capture stdout/stderr; return (rc, stdout, stderr)."""
    from manus_agent import cli  # noqa: PLC0415

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with patch("sys.stdout", out_buf), patch("sys.stderr", err_buf):
        rc = cli._run_osv(argv)
    return rc, out_buf.getvalue(), err_buf.getvalue()


def _make_found_payload(
    cve_id: str = "CVE-2021-44228",
    *,
    ecosystems: list[str] | None = None,
    packages: list[dict] | None = None,
    aliases: list[str] | None = None,
    records: list[dict] | None = None,
) -> dict:
    """Build a minimal ``found=True`` payload like ``fetch_osv_data`` returns."""
    if packages is None:
        packages = [
            {
                "ecosystem": "Maven",
                "package": "org.apache.logging.log4j:log4j-core",
                "introduced": ["2.0-beta9"],
                "fixed": ["2.15.0"],
                "last_affected": [],
                "affected_version_count": 14,
                "affected_versions_sample": ["2.14.1"],
            }
        ]
    if ecosystems is None:
        ecosystems = ["Maven"]
    if aliases is None:
        aliases = ["CVE-2021-44228", "GHSA-jfh8-c2jp-5v3q"]
    if records is None:
        records = [
            {
                "osv_id": "CVE-2021-44228",
                "summary": "Log4Shell RCE",
                "aliases": aliases,
                "modified": "2024-01-01T00:00:00Z",
                "published": "2021-12-10T00:00:00Z",
                "withdrawn": None,
                "severity": [{"type": "CVSS_V3", "score": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"}],
                "affected_ecosystems": ecosystems,
                "affected_packages": packages,
                "references": ["https://logging.apache.org/log4j/"],
            }
        ]
    total = sum(len(r.get("affected_packages", [])) for r in records)
    eco_list = sorted({e for r in records for e in r.get("affected_ecosystems", [])})
    return {
        "found": True,
        "cve_id": cve_id,
        "records": records,
        "aliases": aliases,
        "affected_ecosystems": eco_list,
        "affected_package_count": total,
        "message": (
            f"OSV.dev: {cve_id} affects {total} package(s) across {len(eco_list)} ecosystem(s): {', '.join(eco_list)}."
        ),
    }


def _make_not_found_payload(cve_id: str = "CVE-2099-9999") -> dict:
    return {
        "found": False,
        "cve_id": cve_id,
        "records": [],
        "aliases": [],
        "affected_ecosystems": [],
        "message": f"No OSV.dev record found for {cve_id}.",
    }


def _make_error_payload(cve_id: str = "CVE-2021-44228", error: str = "connection refused") -> dict:
    return {
        "found": False,
        "cve_id": cve_id,
        "records": [],
        "aliases": [],
        "affected_ecosystems": [],
        "error": f"OSV request failed: {error}",
        "message": f"OSV.dev lookup failed for {cve_id}: {error}",
    }


# ===========================================================================
# _build_osv_parser
# ===========================================================================


class TestBuildOsvParser:
    def test_parser_prog_string(self):
        from manus_agent.cli import _build_osv_parser  # noqa: PLC0415

        p = _build_osv_parser()
        assert p.prog == "manus-agent osv"

    def test_parser_cve_id_required(self):
        from manus_agent.cli import _build_osv_parser  # noqa: PLC0415

        p = _build_osv_parser()
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args([])
        assert exc_info.value.code != 0

    def test_parser_cve_id_positional(self):
        from manus_agent.cli import _build_osv_parser  # noqa: PLC0415

        p = _build_osv_parser()
        args = p.parse_args(["CVE-2021-44228"])
        assert args.cve_id == "CVE-2021-44228"

    def test_output_default_is_text(self):
        from manus_agent.cli import _build_osv_parser  # noqa: PLC0415

        p = _build_osv_parser()
        args = p.parse_args(["CVE-2021-44228"])
        assert args.output == "text"

    def test_output_json_choice(self):
        from manus_agent.cli import _build_osv_parser  # noqa: PLC0415

        p = _build_osv_parser()
        args = p.parse_args(["CVE-2021-44228", "--output", "json"])
        assert args.output == "json"

    def test_output_text_choice(self):
        from manus_agent.cli import _build_osv_parser  # noqa: PLC0415

        p = _build_osv_parser()
        args = p.parse_args(["CVE-2021-44228", "--output", "text"])
        assert args.output == "text"

    def test_output_invalid_choice_rejected(self):
        from manus_agent.cli import _build_osv_parser  # noqa: PLC0415

        p = _build_osv_parser()
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args(["CVE-2021-44228", "--output", "xml"])
        assert exc_info.value.code != 0

    def test_parser_has_help(self):
        from manus_agent.cli import _build_osv_parser  # noqa: PLC0415

        p = _build_osv_parser()
        buf = io.StringIO()
        p.print_help(buf)
        help_text = buf.getvalue()
        assert "osv" in help_text.lower()
        assert "--output" in help_text

    def test_parser_description_mentions_osv(self):
        from manus_agent.cli import _build_osv_parser  # noqa: PLC0415

        p = _build_osv_parser()
        assert "OSV" in p.description

    def test_parser_description_mentions_cve(self):
        from manus_agent.cli import _build_osv_parser  # noqa: PLC0415

        p = _build_osv_parser()
        assert "CVE" in p.description


# ===========================================================================
# _run_osv — JSON output
# ===========================================================================


class TestRunOsvJsonOutput:
    def test_json_output_found(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228", "--output", "json"])
        assert rc == 0
        parsed = json.loads(out)
        assert parsed["found"] is True
        assert parsed["cve_id"] == "CVE-2021-44228"

    def test_json_output_not_found(self):
        payload = _make_not_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2099-9999", "--output", "json"])
        assert rc == 0
        parsed = json.loads(out)
        assert parsed["found"] is False

    def test_json_output_error(self):
        payload = _make_error_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228", "--output", "json"])
        assert rc == 0
        parsed = json.loads(out)
        assert parsed["found"] is False
        assert "error" in parsed

    def test_json_output_is_valid_json(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228", "--output", "json"])
        assert rc == 0
        # Must not raise
        json.loads(out)

    def test_json_output_contains_records(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228", "--output", "json"])
        parsed = json.loads(out)
        assert "records" in parsed
        assert len(parsed["records"]) == 1

    def test_json_output_contains_aliases(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228", "--output", "json"])
        parsed = json.loads(out)
        assert "aliases" in parsed
        assert "GHSA-jfh8-c2jp-5v3q" in parsed["aliases"]

    def test_json_output_contains_affected_ecosystems(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228", "--output", "json"])
        parsed = json.loads(out)
        assert "affected_ecosystems" in parsed
        assert "Maven" in parsed["affected_ecosystems"]


# ===========================================================================
# _run_osv — text output
# ===========================================================================


class TestRunOsvTextOutput:
    def test_text_output_not_found_prints_message(self):
        payload = _make_not_found_payload("CVE-2099-9999")
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2099-9999"])
        assert rc == 0
        assert "No OSV.dev record found" in out

    def test_text_output_not_found_returns_zero(self):
        payload = _make_not_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, _out, _err = _run_osv(["CVE-2099-9999"])
        assert rc == 0

    def test_text_output_found_prints_message(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228"])
        assert rc == 0
        assert "OSV.dev" in out

    def test_text_output_prints_aliases(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228"])
        assert rc == 0
        assert "GHSA-jfh8-c2jp-5v3q" in out

    def test_text_output_prints_osv_record_id(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228"])
        assert rc == 0
        assert "CVE-2021-44228" in out

    def test_text_output_prints_package_location(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228"])
        assert rc == 0
        assert "Maven:org.apache.logging.log4j:log4j-core" in out

    def test_text_output_prints_introduced_version(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228"])
        assert rc == 0
        assert "2.0-beta9" in out

    def test_text_output_prints_fixed_version(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228"])
        assert rc == 0
        assert "2.15.0" in out

    def test_text_output_prints_severity(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228"])
        assert rc == 0
        # Severity block: "Severity (CVSS_V3): CVSS:3.1/..."
        assert "CVSS_V3" in out
        assert "CVSS:3.1" in out

    def test_text_output_no_fixed_version_shows_placeholder(self):
        """When fixed=[] the output should show the no-fix placeholder."""
        packages = [
            {
                "ecosystem": "PyPI",
                "package": "pillow",
                "introduced": ["9.0.0"],
                "fixed": [],
                "last_affected": [],
                "affected_version_count": 3,
                "affected_versions_sample": ["9.0.0", "9.1.0"],
            }
        ]
        payload = _make_found_payload(
            cve_id="CVE-2023-12345",
            ecosystems=["PyPI"],
            packages=packages,
            aliases=["CVE-2023-12345"],
        )
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2023-12345"])
        assert rc == 0
        assert "(no fixed version listed)" in out

    def test_text_output_last_affected_shown(self):
        packages = [
            {
                "ecosystem": "npm",
                "package": "lodash",
                "introduced": ["4.0.0"],
                "fixed": ["4.17.21"],
                "last_affected": ["4.17.20"],
                "affected_version_count": 5,
                "affected_versions_sample": [],
            }
        ]
        payload = _make_found_payload(
            cve_id="CVE-2021-23337",
            ecosystems=["npm"],
            packages=packages,
            aliases=["CVE-2021-23337"],
        )
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-23337"])
        assert rc == 0
        assert "4.17.20" in out
        assert "Last affected" in out

    def test_text_output_no_aliases_suppressed(self):
        """When aliases list is empty, no 'Aliases:' line should appear."""
        payload = _make_found_payload(aliases=[])
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228"])
        assert rc == 0
        assert "Aliases:" not in out

    def test_text_output_record_without_packages_skipped(self):
        """Records with no affected_packages should not produce package output."""
        records = [
            {
                "osv_id": "CVE-2021-44228",
                "summary": "Log4Shell",
                "aliases": ["GHSA-jfh8-c2jp-5v3q"],
                "modified": "2024-01-01T00:00:00Z",
                "published": "2021-12-10T00:00:00Z",
                "withdrawn": None,
                "severity": [],
                "affected_ecosystems": [],
                "affected_packages": [],  # empty
                "references": [],
            }
        ]
        payload = {
            "found": True,
            "cve_id": "CVE-2021-44228",
            "records": records,
            "aliases": ["GHSA-jfh8-c2jp-5v3q"],
            "affected_ecosystems": [],
            "affected_package_count": 0,
            "message": "OSV.dev has a record for CVE-2021-44228 but no package-level version ranges.",
        }
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228"])
        assert rc == 0
        # Message should be printed but no package table
        assert "OSV record" not in out

    def test_text_output_severity_with_empty_score_suppressed(self):
        """Severity entries with empty score string are not printed."""
        packages = [
            {
                "ecosystem": "Maven",
                "package": "org.example:lib",
                "introduced": ["1.0"],
                "fixed": ["1.1"],
                "last_affected": [],
                "affected_version_count": 1,
                "affected_versions_sample": [],
            }
        ]
        records = [
            {
                "osv_id": "CVE-2024-0001",
                "summary": "Test",
                "aliases": [],
                "modified": "2024-01-01T00:00:00Z",
                "published": "2024-01-01T00:00:00Z",
                "withdrawn": None,
                "severity": [{"type": "CVSS_V3", "score": ""}],  # empty score
                "affected_ecosystems": ["Maven"],
                "affected_packages": packages,
                "references": [],
            }
        ]
        payload = {
            "found": True,
            "cve_id": "CVE-2024-0001",
            "records": records,
            "aliases": [],
            "affected_ecosystems": ["Maven"],
            "affected_package_count": 1,
            "message": "OSV.dev: CVE-2024-0001 affects 1 package(s) across 1 ecosystem(s): Maven.",
        }
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2024-0001"])
        assert rc == 0
        # Empty score → "Severity (CVSS_V3): " should NOT appear
        assert "Severity" not in out

    def test_text_output_multiple_records(self):
        """Multiple records (e.g. CVE + GHSA alias) should all be printed."""
        rec1 = {
            "osv_id": "CVE-2021-44228",
            "summary": "Log4Shell",
            "aliases": ["GHSA-jfh8-c2jp-5v3q"],
            "modified": "2024-01-01T00:00:00Z",
            "published": "2021-12-10T00:00:00Z",
            "withdrawn": None,
            "severity": [],
            "affected_ecosystems": ["Maven"],
            "affected_packages": [
                {
                    "ecosystem": "Maven",
                    "package": "org.apache.logging.log4j:log4j-core",
                    "introduced": ["2.0-beta9"],
                    "fixed": ["2.15.0"],
                    "last_affected": [],
                    "affected_version_count": 14,
                    "affected_versions_sample": [],
                }
            ],
            "references": [],
        }
        rec2 = {
            "osv_id": "GHSA-jfh8-c2jp-5v3q",
            "summary": "Log4Shell GHSA",
            "aliases": ["CVE-2021-44228"],
            "modified": "2024-01-01T00:00:00Z",
            "published": "2021-12-10T00:00:00Z",
            "withdrawn": None,
            "severity": [],
            "affected_ecosystems": ["Maven"],
            "affected_packages": [
                {
                    "ecosystem": "Maven",
                    "package": "org.apache.logging.log4j:log4j-api",
                    "introduced": ["2.0-beta9"],
                    "fixed": ["2.15.0"],
                    "last_affected": [],
                    "affected_version_count": 14,
                    "affected_versions_sample": [],
                }
            ],
            "references": [],
        }
        payload = {
            "found": True,
            "cve_id": "CVE-2021-44228",
            "records": [rec1, rec2],
            "aliases": ["CVE-2021-44228", "GHSA-jfh8-c2jp-5v3q"],
            "affected_ecosystems": ["Maven"],
            "affected_package_count": 2,
            "message": "OSV.dev: CVE-2021-44228 affects 2 package(s) across 1 ecosystem(s): Maven.",
        }
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228"])
        assert rc == 0
        assert "CVE-2021-44228" in out
        assert "GHSA-jfh8-c2jp-5v3q" in out
        assert "log4j-core" in out
        assert "log4j-api" in out

    def test_text_output_introduced_defaults_to_zero_when_empty(self):
        """When introduced=[], the output should fall back to '0'."""
        packages = [
            {
                "ecosystem": "Go",
                "package": "github.com/example/mod",
                "introduced": [],
                "fixed": ["1.2.3"],
                "last_affected": [],
                "affected_version_count": 1,
                "affected_versions_sample": [],
            }
        ]
        payload = _make_found_payload(
            cve_id="CVE-2024-0002",
            ecosystems=["Go"],
            packages=packages,
            aliases=["CVE-2024-0002"],
        )
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2024-0002"])
        assert rc == 0
        # Introduced falls back to "0"
        assert "Vulnerable from" in out
        assert ": 0" in out


# ===========================================================================
# _run_osv — error / edge cases
# ===========================================================================


class TestRunOsvErrors:
    def test_error_payload_prints_message_and_returns_zero(self):
        """fetch_osv_data error payloads still return rc=0; message is printed."""
        payload = _make_error_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228"])
        assert rc == 0
        assert "lookup failed" in out or "OSV" in out

    def test_json_output_is_returned_even_for_error_payload(self):
        payload = _make_error_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, out, _err = _run_osv(["CVE-2021-44228", "--output", "json"])
        assert rc == 0
        parsed = json.loads(out)
        assert parsed["found"] is False
        assert "error" in parsed

    def test_missing_cve_id_exits_nonzero(self):
        from manus_agent.cli import _run_osv as run_osv  # noqa: PLC0415

        with pytest.raises(SystemExit) as exc_info:
            run_osv([])
        assert exc_info.value.code != 0

    def test_invalid_output_choice_exits_nonzero(self):
        from manus_agent.cli import _run_osv as run_osv  # noqa: PLC0415

        with pytest.raises(SystemExit) as exc_info:
            run_osv(["CVE-2021-44228", "--output", "yaml"])
        assert exc_info.value.code != 0

    def test_cve_id_is_passed_to_fetch(self):
        """Verifies the CVE ID from argv reaches fetch_osv_data unmodified."""
        payload = _make_not_found_payload("CVE-2024-3094")
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload) as mock_fetch:
            _run_osv(["CVE-2024-3094"])
        mock_fetch.assert_called_once_with("CVE-2024-3094")

    def test_cve_id_is_stripped_of_whitespace(self):
        """Trailing/leading whitespace in the positional arg is stripped."""
        payload = _make_not_found_payload("CVE-2024-3094")
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload) as mock_fetch:
            _run_osv(["  CVE-2024-3094  "])
        mock_fetch.assert_called_once_with("CVE-2024-3094")


# ===========================================================================
# _SUBCOMMANDS registry
# ===========================================================================


class TestSubcommandsRegistry:
    def test_osv_in_subcommands_set(self):
        from manus_agent.cli import _SUBCOMMANDS  # noqa: PLC0415

        assert "osv" in _SUBCOMMANDS

    def test_subcommands_is_a_set(self):
        from manus_agent.cli import _SUBCOMMANDS  # noqa: PLC0415

        assert isinstance(_SUBCOMMANDS, (set, frozenset))

    def test_osv_not_duplicated(self):
        from manus_agent.cli import _SUBCOMMANDS  # noqa: PLC0415

        # Sets have no duplicates by definition, but verify membership once
        assert list(_SUBCOMMANDS).count("osv") == 1 or isinstance(_SUBCOMMANDS, (set, frozenset))


# ===========================================================================
# main() dispatch
# ===========================================================================


class TestMainDispatchOsv:
    def test_main_routes_osv_subcommand(self):
        """main(['osv', 'CVE-2021-44228']) should call _run_osv."""
        from manus_agent import cli  # noqa: PLC0415

        payload = _make_found_payload()
        with (
            patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload),
            patch.object(sys, "argv", ["manus-agent", "osv", "CVE-2021-44228"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli.main()
        assert exc_info.value.code == 0

    def test_main_osv_json_flag(self):
        """main(['osv', 'CVE-2021-44228', '--output', 'json']) routes correctly."""
        from manus_agent import cli  # noqa: PLC0415

        payload = _make_found_payload()
        out_buf = io.StringIO()
        with (
            patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload),
            patch.object(sys, "argv", ["manus-agent", "osv", "CVE-2021-44228", "--output", "json"]),
            patch("sys.stdout", out_buf),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli.main()
        assert exc_info.value.code == 0
        parsed = json.loads(out_buf.getvalue())
        assert parsed["found"] is True

    def test_main_osv_not_found_exits_zero(self):
        from manus_agent import cli  # noqa: PLC0415

        payload = _make_not_found_payload()
        with (
            patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload),
            patch.object(sys, "argv", ["manus-agent", "osv", "CVE-2099-9999"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli.main()
        assert exc_info.value.code == 0

    def test_main_osv_does_not_route_to_single_shot(self):
        """Verifies 'osv' is intercepted before the generic single-shot handler."""
        from manus_agent import cli  # noqa: PLC0415

        payload = _make_found_payload()
        with (
            patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload),
            patch.object(sys, "argv", ["manus-agent", "osv", "CVE-2021-44228"]),
            patch.object(cli, "_run_single_shot") as mock_single,
        ):
            try:
                cli.main()
            except SystemExit:
                pass
        mock_single.assert_not_called()

    def test_main_osv_calls_fetch_osv_data(self):
        """Smoke-test: main() dispatch reaches fetch_osv_data for osv subcommand."""
        from manus_agent import cli  # noqa: PLC0415

        payload = _make_found_payload()
        with (
            patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload) as mock_fetch,
            patch.object(sys, "argv", ["manus-agent", "osv", "CVE-2021-44228"]),
        ):
            try:
                cli.main()
            except SystemExit:
                pass
        mock_fetch.assert_called_once_with("CVE-2021-44228")


# ===========================================================================
# fetch_osv_data import path in _run_osv
# ===========================================================================


class TestRunOsvImportPath:
    def test_fetch_osv_data_importable(self):
        """_run_osv imports fetch_osv_data from get_osv_data — verify the path."""
        from manus_agent.tools.get_osv_data import fetch_osv_data  # noqa: PLC0415

        assert callable(fetch_osv_data)

    def test_run_osv_returns_int(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, _out, _err = _run_osv(["CVE-2021-44228"])
        assert isinstance(rc, int)

    def test_run_osv_text_returns_zero_on_success(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, _out, _err = _run_osv(["CVE-2021-44228"])
        assert rc == 0

    def test_run_osv_json_returns_zero_on_success(self):
        payload = _make_found_payload()
        with patch("manus_agent.tools.get_osv_data.fetch_osv_data", return_value=payload):
            rc, _out, _err = _run_osv(["CVE-2021-44228", "--output", "json"])
        assert rc == 0
