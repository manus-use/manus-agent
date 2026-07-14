"""Comprehensive tests for the `manus-agent cwe` CLI subcommand."""

from __future__ import annotations

import json
import sys
from unittest.mock import patch

import pytest

from manus_agent.cli import _SUBCOMMANDS, _build_cwe_parser, _run_cwe, main

# ---------------------------------------------------------------------------
# _build_cwe_parser tests
# ---------------------------------------------------------------------------


class TestBuildCweParser:
    """Verify _build_cwe_parser() flag definitions and defaults."""

    def test_prog_name(self):
        p = _build_cwe_parser()
        assert p.prog == "manus-agent cwe"

    def test_positional_cwe_id(self):
        p = _build_cwe_parser()
        args = p.parse_args(["CWE-79"])
        assert args.cwe_id == "CWE-79"

    def test_output_default_text(self):
        p = _build_cwe_parser()
        args = p.parse_args(["CWE-79"])
        assert args.output == "text"

    def test_output_json(self):
        p = _build_cwe_parser()
        args = p.parse_args(["CWE-79", "--output", "json"])
        assert args.output == "json"

    def test_output_invalid_choice(self):
        p = _build_cwe_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["CWE-79", "--output", "yaml"])

    def test_missing_cwe_id_fails(self):
        p = _build_cwe_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_numeric_cwe_id(self):
        p = _build_cwe_parser()
        args = p.parse_args(["79"])
        assert args.cwe_id == "79"

    def test_help_flag(self):
        p = _build_cwe_parser()
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# _run_cwe text output tests
# ---------------------------------------------------------------------------


_MOCK_SUCCESS_RESULT = {
    "toolUseId": "cli-cwe-lookup",
    "status": "success",
    "content": [
        {
            "json": {
                "cwe_id": "CWE-79",
                "description": "Improper Neutralization of Input During Web Page Generation",
                "url": "https://cwe.mitre.org/data/definitions/79.html",
            }
        }
    ],
}


class TestRunCweTextOutput:
    """Verify _run_cwe() text output mode."""

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_basic_text_output(self, mock_cwe, capsys):
        mock_cwe.return_value = _MOCK_SUCCESS_RESULT
        rc = _run_cwe(["CWE-79"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CWE-79" in out
        assert "https://cwe.mitre.org/data/definitions/79.html" in out
        assert "Improper Neutralization" in out

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_numeric_input_normalised(self, mock_cwe, capsys):
        mock_cwe.return_value = _MOCK_SUCCESS_RESULT
        rc = _run_cwe(["79"])
        assert rc == 0
        # Verify get_cwe_details was called with CWE-79 (normalised)
        call_args = mock_cwe.call_args[0][0]
        assert call_args["input"]["cwe_id"] == "CWE-79"

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_lowercase_input_normalised(self, mock_cwe, capsys):
        mock_cwe.return_value = _MOCK_SUCCESS_RESULT
        rc = _run_cwe(["cwe-79"])
        assert rc == 0
        call_args = mock_cwe.call_args[0][0]
        assert call_args["input"]["cwe_id"] == "CWE-79"

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_text_output_contains_url(self, mock_cwe, capsys):
        mock_cwe.return_value = _MOCK_SUCCESS_RESULT
        _run_cwe(["CWE-79"])
        out = capsys.readouterr().out
        assert "URL" in out
        assert "https://cwe.mitre.org/data/definitions/79.html" in out

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_text_output_contains_description(self, mock_cwe, capsys):
        mock_cwe.return_value = _MOCK_SUCCESS_RESULT
        _run_cwe(["CWE-79"])
        out = capsys.readouterr().out
        assert "Improper Neutralization" in out

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_text_output_wraps_long_description(self, mock_cwe, capsys):
        long_desc = "A" * 200
        mock_cwe.return_value = {
            "toolUseId": "cli-cwe-lookup",
            "status": "success",
            "content": [
                {
                    "json": {
                        "cwe_id": "CWE-787",
                        "description": long_desc,
                        "url": "https://cwe.mitre.org/data/definitions/787.html",
                    }
                }
            ],
        }
        rc = _run_cwe(["CWE-787"])
        assert rc == 0
        out = capsys.readouterr().out
        # Long text should be wrapped (multi-line)
        lines = [line for line in out.split("\n") if "A" in line]
        assert len(lines) > 1

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_text_output_no_description_fallback(self, mock_cwe, capsys):
        mock_cwe.return_value = {
            "toolUseId": "cli-cwe-lookup",
            "status": "success",
            "content": [
                {
                    "json": {
                        "cwe_id": "CWE-79",
                        "url": "https://cwe.mitre.org/data/definitions/79.html",
                    }
                }
            ],
        }
        rc = _run_cwe(["CWE-79"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No description available" in out


# ---------------------------------------------------------------------------
# _run_cwe JSON output tests
# ---------------------------------------------------------------------------


class TestRunCweJsonOutput:
    """Verify _run_cwe() JSON output mode."""

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_json_output_valid(self, mock_cwe, capsys):
        mock_cwe.return_value = _MOCK_SUCCESS_RESULT
        rc = _run_cwe(["CWE-79", "--output", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["cwe_id"] == "CWE-79"
        assert "description" in data
        assert "url" in data

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_json_output_includes_url(self, mock_cwe, capsys):
        mock_cwe.return_value = _MOCK_SUCCESS_RESULT
        _run_cwe(["CWE-79", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["url"] == "https://cwe.mitre.org/data/definitions/79.html"

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_json_output_description(self, mock_cwe, capsys):
        mock_cwe.return_value = _MOCK_SUCCESS_RESULT
        _run_cwe(["CWE-79", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert "Improper Neutralization" in data["description"]

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_json_numeric_id_input(self, mock_cwe, capsys):
        mock_cwe.return_value = {
            "toolUseId": "cli-cwe-lookup",
            "status": "success",
            "content": [
                {
                    "json": {
                        "cwe_id": "CWE-787",
                        "description": "Out-of-bounds Write",
                        "url": "https://cwe.mitre.org/data/definitions/787.html",
                    }
                }
            ],
        }
        rc = _run_cwe(["787", "--output", "json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["cwe_id"] == "CWE-787"

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_json_output_is_valid_json(self, mock_cwe, capsys):
        mock_cwe.return_value = _MOCK_SUCCESS_RESULT
        _run_cwe(["CWE-79", "--output", "json"])
        out = capsys.readouterr().out
        # Should not raise
        parsed = json.loads(out)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# _run_cwe error handling tests
# ---------------------------------------------------------------------------


class TestRunCweErrors:
    """Verify _run_cwe() handles errors correctly."""

    def test_invalid_cwe_id_non_numeric(self, capsys):
        rc = _run_cwe(["XSS"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid CWE ID" in err

    def test_invalid_cwe_id_empty_string(self, capsys):
        rc = _run_cwe(["   "])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid CWE ID" in err

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_tool_returns_error(self, mock_cwe, capsys):
        mock_cwe.return_value = {
            "toolUseId": "cli-cwe-lookup",
            "status": "error",
            "content": [{"text": "Could not find description for CWE-99999 on the page."}],
        }
        rc = _run_cwe(["CWE-99999"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Could not find description" in err

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_tool_network_error(self, mock_cwe, capsys):
        mock_cwe.return_value = {
            "toolUseId": "cli-cwe-lookup",
            "status": "error",
            "content": [{"text": "Request to CWE website failed: ConnectionError"}],
        }
        rc = _run_cwe(["CWE-79"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Request to CWE website failed" in err

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_tool_returns_empty_content(self, mock_cwe, capsys):
        mock_cwe.return_value = {
            "toolUseId": "cli-cwe-lookup",
            "status": "error",
            "content": [],
        }
        rc = _run_cwe(["CWE-79"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Unknown error" in err

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_tool_unexpected_exception(self, mock_cwe, capsys):
        mock_cwe.return_value = {
            "toolUseId": "cli-cwe-lookup",
            "status": "error",
            "content": [{"text": "An unexpected error occurred during CWE details fetching: timeout"}],
        }
        rc = _run_cwe(["CWE-79"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "unexpected error" in err.lower() or "timeout" in err.lower()

    def test_import_error_in_run(self, capsys, monkeypatch):
        """Simulate ImportError when importing get_cwe_details module."""
        monkeypatch.setitem(sys.modules, "manus_agent.tools.get_cwe_details", None)
        rc = _run_cwe(["CWE-79"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "missing dependencies" in err


# ---------------------------------------------------------------------------
# _SUBCOMMANDS registry tests
# ---------------------------------------------------------------------------


class TestSubcommandsRegistry:
    """Verify 'cwe' is in the _SUBCOMMANDS set."""

    def test_cwe_in_subcommands(self):
        assert "cwe" in _SUBCOMMANDS

    def test_subcommands_is_set(self):
        assert isinstance(_SUBCOMMANDS, set)

    def test_cwe_not_duplicate_entry(self):
        # Ensure no accidental double-registration
        count = list(_SUBCOMMANDS).count("cwe")
        assert count == 1


# ---------------------------------------------------------------------------
# main() dispatch tests
# ---------------------------------------------------------------------------


class TestMainDispatchCwe:
    """Verify main() routes 'cwe' to _run_cwe."""

    @patch("manus_agent.cli._run_cwe", return_value=0)
    def test_basic_dispatch(self, mock_run, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["manus-agent", "cwe", "CWE-79"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(["CWE-79"])

    @patch("manus_agent.cli._run_cwe", return_value=0)
    def test_dispatch_with_output_flag(self, mock_run, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["manus-agent", "cwe", "CWE-79", "--output", "json"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(["CWE-79", "--output", "json"])

    @patch("manus_agent.cli._run_cwe", return_value=0)
    def test_dispatch_numeric_id(self, mock_run, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["manus-agent", "cwe", "787"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(["787"])

    @patch("manus_agent.cli._run_cwe", return_value=1)
    def test_dispatch_nonzero_rc(self, mock_run, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["manus-agent", "cwe", "CWE-99999"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("manus_agent.cli._run_cwe", return_value=0)
    def test_dispatch_lowercase(self, mock_run, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["manus-agent", "cwe", "cwe-79"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(["cwe-79"])


# ---------------------------------------------------------------------------
# Import path tests
# ---------------------------------------------------------------------------


class TestRunCweImportPath:
    """Verify the internal import inside _run_cwe works correctly."""

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_calls_get_cwe_details_with_correct_payload(self, mock_cwe, capsys):
        mock_cwe.return_value = _MOCK_SUCCESS_RESULT
        _run_cwe(["CWE-79"])
        mock_cwe.assert_called_once()
        tool_use = mock_cwe.call_args[0][0]
        assert tool_use["toolUseId"] == "cli-cwe-lookup"
        assert tool_use["input"]["cwe_id"] == "CWE-79"

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_tool_use_id_is_string(self, mock_cwe, capsys):
        mock_cwe.return_value = _MOCK_SUCCESS_RESULT
        _run_cwe(["CWE-79"])
        tool_use = mock_cwe.call_args[0][0]
        assert isinstance(tool_use["toolUseId"], str)

    @patch("manus_agent.tools.get_cwe_details.get_cwe_details")
    def test_input_dict_has_cwe_id_key(self, mock_cwe, capsys):
        mock_cwe.return_value = _MOCK_SUCCESS_RESULT
        _run_cwe(["CWE-79"])
        tool_use = mock_cwe.call_args[0][0]
        assert "cwe_id" in tool_use["input"]
