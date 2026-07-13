"""
Tests for manus-agent check-kev CLI subcommand.

Covers:
- _build_check_kev_parser — argument definitions, defaults, choices, help
- _run_check_kev — text output, JSON output, not-in-kev, invalid input,
  API errors, ransomware flag, partial KEV fields, CWE rendering
- _SUBCOMMANDS registry membership
- main() dispatch routing

All tests are fully mocked — no real HTTP calls are made.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_MOCK_KEV_ENTRY_LOG4SHELL = {
    "cveID": "CVE-2021-44228",
    "vendorProject": "Apache",
    "product": "Log4j2",
    "vulnerabilityName": "Apache Log4j2 Remote Code Execution Vulnerability",
    "dateAdded": "2021-12-10",
    "shortDescription": (
        "Apache Log4j2 contains a vulnerability where JNDI features do not "
        "protect against attacker-controlled JNDI-related endpoints, allowing "
        "for remote code execution."
    ),
    "requiredAction": "Apply updates per vendor guidance.",
    "dueDate": "2021-12-24",
    "knownRansomwareCampaignUse": "Known",
    "notes": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
    "cwes": ["CWE-20", "CWE-400", "CWE-502"],
}

_MOCK_KEV_ENTRY_XZ = {
    "cveID": "CVE-2024-3094",
    "vendorProject": "XZ Utils",
    "product": "XZ Utils",
    "vulnerabilityName": "XZ Utils Backdoor",
    "dateAdded": "2024-04-01",
    "shortDescription": "XZ Utils contains a backdoor inserted by a malicious actor.",
    "requiredAction": "Remove affected versions immediately.",
    "dueDate": "2024-04-08",
    "knownRansomwareCampaignUse": "Unknown",
    "notes": "",
    "cwes": [],
}

_MOCK_KEV_CATALOG = {
    "vulnerabilities": [_MOCK_KEV_ENTRY_LOG4SHELL, _MOCK_KEV_ENTRY_XZ],
}

_NOT_IN_CATALOG_CVE = "CVE-9999-12345"


def _make_kev_data(entries=None):
    """Return a mock KEV catalog dict."""
    if entries is None:
        entries = [_MOCK_KEV_ENTRY_LOG4SHELL, _MOCK_KEV_ENTRY_XZ]
    return {"vulnerabilities": entries}


# ---------------------------------------------------------------------------
# TestBuildCheckKevParser — argument definitions
# ---------------------------------------------------------------------------


class TestBuildCheckKevParser:
    def test_parser_accepts_cve_id(self):
        from manus_agent.cli import _build_check_kev_parser

        parser = _build_check_kev_parser()
        args = parser.parse_args(["CVE-2021-44228"])
        assert args.cve_id == "CVE-2021-44228"

    def test_default_output_is_text(self):
        from manus_agent.cli import _build_check_kev_parser

        parser = _build_check_kev_parser()
        args = parser.parse_args(["CVE-2021-44228"])
        assert args.output == "text"

    def test_output_json_flag_accepted(self):
        from manus_agent.cli import _build_check_kev_parser

        parser = _build_check_kev_parser()
        args = parser.parse_args(["CVE-2021-44228", "--output", "json"])
        assert args.output == "json"

    def test_invalid_output_format_exits(self):
        from manus_agent.cli import _build_check_kev_parser

        parser = _build_check_kev_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["CVE-2021-44228", "--output", "xml"])

    def test_help_exits_zero(self):
        from manus_agent.cli import _build_check_kev_parser

        parser = _build_check_kev_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_missing_cve_id_exits_nonzero(self):
        from manus_agent.cli import _build_check_kev_parser

        parser = _build_check_kev_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([])
        assert exc.value.code != 0

    def test_prog_name(self):
        from manus_agent.cli import _build_check_kev_parser

        parser = _build_check_kev_parser()
        assert parser.prog == "manus-agent check-kev"

    def test_description_mentions_kev(self):
        from manus_agent.cli import _build_check_kev_parser

        parser = _build_check_kev_parser()
        assert "KEV" in parser.description or "Known Exploited" in parser.description


# ---------------------------------------------------------------------------
# TestRunCheckKevTextOutput — CVE in KEV, text mode
# ---------------------------------------------------------------------------


class TestRunCheckKevTextOutput:
    def test_in_kev_exits_zero(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            code = _run_check_kev(["CVE-2021-44228"])
        assert code == 0

    def test_in_kev_prints_actively_exploited_banner(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        assert "ACTIVELY EXPLOITED" in out

    def test_in_kev_prints_cve_id(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        assert "CVE-2021-44228" in out

    def test_in_kev_prints_vendor(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        assert "Apache" in out

    def test_in_kev_prints_product(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        assert "Log4j2" in out

    def test_in_kev_prints_date_added(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        assert "2021-12-10" in out

    def test_in_kev_prints_due_date(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        assert "2021-12-24" in out

    def test_ransomware_known_prints_warning(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        assert "RANSOMWARE" in out

    def test_ransomware_unknown_no_warning(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2024-3094"])
        out = capsys.readouterr().out
        assert "RANSOMWARE" not in out

    def test_cwes_printed(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        assert "CWE-20" in out

    def test_case_insensitive_cve_id(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            code = _run_check_kev(["cve-2021-44228"])
        assert code == 0
        out = capsys.readouterr().out
        assert "CVE-2021-44228" in out

    def test_long_description_truncated(self, capsys):
        from manus_agent.cli import _run_check_kev

        long_entry = dict(_MOCK_KEV_ENTRY_LOG4SHELL)
        long_entry["shortDescription"] = "x" * 300
        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value={"vulnerabilities": [long_entry]}):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        desc_lines = [line for line in out.splitlines() if "Description" in line]
        assert desc_lines, "Expected a Description line in output"
        assert "\u2026" in desc_lines[0]

    def test_long_required_action_truncated(self, capsys):
        from manus_agent.cli import _run_check_kev

        long_entry = dict(_MOCK_KEV_ENTRY_LOG4SHELL)
        long_entry["requiredAction"] = "a" * 300
        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value={"vulnerabilities": [long_entry]}):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        action_lines = [line for line in out.splitlines() if "Required action" in line]
        assert action_lines
        assert "\u2026" in action_lines[0]

    def test_notes_printed_when_present(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        assert "nvd.nist.gov" in out


# ---------------------------------------------------------------------------
# TestRunCheckKevNotFound — CVE not in catalog
# ---------------------------------------------------------------------------


class TestRunCheckKevNotFound:
    def test_not_in_kev_exits_zero(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            code = _run_check_kev([_NOT_IN_CATALOG_CVE])
        assert code == 0

    def test_not_in_kev_prints_not_found_message(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev([_NOT_IN_CATALOG_CVE])
        out = capsys.readouterr().out
        assert "NOT in CISA KEV" in out or "not found" in out.lower()

    def test_not_in_kev_no_banner(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev([_NOT_IN_CATALOG_CVE])
        out = capsys.readouterr().out
        assert "ACTIVELY EXPLOITED" not in out

    def test_not_in_kev_cve_id_in_output(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev([_NOT_IN_CATALOG_CVE])
        out = capsys.readouterr().out
        assert _NOT_IN_CATALOG_CVE in out


# ---------------------------------------------------------------------------
# TestRunCheckKevJsonOutput — JSON mode
# ---------------------------------------------------------------------------


class TestRunCheckKevJsonOutput:
    def test_json_output_exits_zero(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            code = _run_check_kev(["CVE-2021-44228", "--output", "json"])
        assert code == 0

    def test_json_output_is_valid_json(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228", "--output", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, dict)

    def test_json_in_kev_true(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["in_kev"] is True

    def test_json_in_kev_false(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev([_NOT_IN_CATALOG_CVE, "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["in_kev"] is False

    def test_json_cve_id_uppercased(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["cve-2021-44228", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["cve_id"] == "CVE-2021-44228"

    def test_json_vendor_present_when_in_kev(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["vendor"] == "Apache"

    def test_json_product_present_when_in_kev(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["product"] == "Log4j2"

    def test_json_date_added_present(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["date_added"] == "2021-12-10"

    def test_json_due_date_present(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["due_date"] == "2021-12-24"

    def test_json_ransomware_field_present(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["ransomware"] == "Known"

    def test_json_cwes_list_present(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert "CWE-20" in data["cwes"]

    def test_json_not_in_kev_has_no_vendor_field(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev([_NOT_IN_CATALOG_CVE, "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert "vendor" not in data
        assert "product" not in data

    def test_json_short_description_present(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert "short_description" in data
        assert len(data["short_description"]) > 0

    def test_json_required_action_present(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert "required_action" in data


# ---------------------------------------------------------------------------
# TestRunCheckKevErrors — error handling
# ---------------------------------------------------------------------------


class TestRunCheckKevErrors:
    def test_invalid_cve_id_exits_one(self, capsys):
        from manus_agent.cli import _run_check_kev

        code = _run_check_kev(["not-a-cve"])
        assert code == 1

    def test_invalid_cve_id_error_to_stderr(self, capsys):
        from manus_agent.cli import _run_check_kev

        _run_check_kev(["NOTACVE"])
        err = capsys.readouterr().err
        assert "error" in err.lower() or "Invalid" in err

    def test_numeric_only_id_fails(self, capsys):
        from manus_agent.cli import _run_check_kev

        code = _run_check_kev(["12345"])
        assert code == 1

    def test_api_failure_exits_one(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", side_effect=Exception("network error")):
            code = _run_check_kev(["CVE-2021-44228"])
        assert code == 1

    def test_api_failure_error_to_stderr(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", side_effect=Exception("network error")):
            _run_check_kev(["CVE-2021-44228"])
        err = capsys.readouterr().err
        assert "error" in err.lower()

    def test_empty_kev_catalog_exits_one(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value={}):
            code = _run_check_kev(["CVE-2021-44228"])
        assert code == 1

    def test_none_kev_catalog_exits_one(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=None):
            code = _run_check_kev(["CVE-2021-44228"])
        assert code == 1

    def test_cve_with_short_numeric_part_fails(self, capsys):
        from manus_agent.cli import _run_check_kev

        # CVE IDs need at least 4 numeric digits after the year
        code = _run_check_kev(["CVE-2021-123"])
        assert code == 1


# ---------------------------------------------------------------------------
# TestRunCheckKevEdgeCases — edge cases
# ---------------------------------------------------------------------------


class TestRunCheckKevEdgeCases:
    def test_entry_with_no_cwes(self, capsys):
        from manus_agent.cli import _run_check_kev

        entry = dict(_MOCK_KEV_ENTRY_XZ)
        entry["cwes"] = []
        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value={"vulnerabilities": [entry]}):
            code = _run_check_kev(["CVE-2024-3094"])
        assert code == 0

    def test_entry_with_no_notes_hides_notes_line(self, capsys):
        from manus_agent.cli import _run_check_kev

        entry = dict(_MOCK_KEV_ENTRY_LOG4SHELL)
        entry["notes"] = ""
        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value={"vulnerabilities": [entry]}):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        assert "Notes" not in out

    def test_entry_missing_vulnerability_name(self, capsys):
        from manus_agent.cli import _run_check_kev

        entry = {k: v for k, v in _MOCK_KEV_ENTRY_LOG4SHELL.items() if k != "vulnerabilityName"}
        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value={"vulnerabilities": [entry]}):
            code = _run_check_kev(["CVE-2021-44228"])
        assert code == 0

    def test_json_output_not_in_kev_minimal_fields(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-9999-12345", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data == {"cve_id": "CVE-9999-12345", "in_kev": False}

    def test_xz_entry_in_kev(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            code = _run_check_kev(["CVE-2024-3094"])
        assert code == 0
        out = capsys.readouterr().out
        assert "ACTIVELY EXPLOITED" in out
        assert "XZ Utils" in out

    def test_cve_id_normalised_to_uppercase_in_json(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["cve-2021-44228", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["cve_id"] == "CVE-2021-44228"

    def test_multiple_cwes_all_printed(self, capsys):
        from manus_agent.cli import _run_check_kev

        with patch("manus_agent.tools.check_cisa_kev._get_kev_data", return_value=_make_kev_data()):
            _run_check_kev(["CVE-2021-44228"])
        out = capsys.readouterr().out
        assert "CWE-400" in out
        assert "CWE-502" in out


# ---------------------------------------------------------------------------
# TestSubcommandsRegistry
# ---------------------------------------------------------------------------


class TestSubcommandsRegistry:
    def test_check_kev_in_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "check-kev" in _SUBCOMMANDS

    def test_subcommands_is_a_set(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert isinstance(_SUBCOMMANDS, (set, frozenset))

    def test_no_duplicate_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert len(_SUBCOMMANDS) == len(set(_SUBCOMMANDS))


# ---------------------------------------------------------------------------
# TestMainDispatchCheckKev — main() routing
# ---------------------------------------------------------------------------


class TestMainDispatchCheckKev:
    def test_main_routes_check_kev(self):
        with patch("manus_agent.cli._run_check_kev", return_value=0) as mock_run:
            with patch("sys.argv", ["manus-agent", "check-kev", "CVE-2021-44228"]):
                try:
                    from manus_agent.cli import main

                    main()
                except SystemExit as e:
                    assert e.code == 0
            mock_run.assert_called_once_with(["CVE-2021-44228"])

    def test_main_passes_output_flag(self):
        with patch("manus_agent.cli._run_check_kev", return_value=0) as mock_run:
            with patch("sys.argv", ["manus-agent", "check-kev", "CVE-2021-44228", "--output", "json"]):
                try:
                    from manus_agent.cli import main

                    main()
                except SystemExit:
                    pass
            mock_run.assert_called_once_with(["CVE-2021-44228", "--output", "json"])

    def test_main_propagates_nonzero_rc(self):
        with patch("manus_agent.cli._run_check_kev", return_value=1):
            with patch("sys.argv", ["manus-agent", "check-kev", "not-a-cve"]):
                try:
                    from manus_agent.cli import main

                    main()
                except SystemExit as e:
                    assert e.code == 1

    def test_main_check_kev_with_lowercase_cve(self):
        with patch("manus_agent.cli._run_check_kev", return_value=0) as mock_run:
            with patch("sys.argv", ["manus-agent", "check-kev", "cve-2021-44228"]):
                try:
                    from manus_agent.cli import main

                    main()
                except SystemExit:
                    pass
            mock_run.assert_called_once_with(["cve-2021-44228"])


# ---------------------------------------------------------------------------
# TestRunCheckKevImportPath — verify callable surface
# ---------------------------------------------------------------------------


class TestRunCheckKevImportPath:
    def test_run_check_kev_is_callable(self):
        from manus_agent.cli import _run_check_kev

        assert callable(_run_check_kev)

    def test_build_check_kev_parser_is_callable(self):
        from manus_agent.cli import _build_check_kev_parser

        assert callable(_build_check_kev_parser)

    def test_run_check_kev_exists_in_cli_module(self):
        import manus_agent.cli as cli_mod

        assert hasattr(cli_mod, "_run_check_kev")
