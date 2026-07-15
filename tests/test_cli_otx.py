"""Comprehensive tests for the manus-agent otx CLI subcommand."""

import json
import sys
from unittest.mock import patch

import pytest

from manus_agent.cli import _SUBCOMMANDS, _build_otx_parser, _print_otx_text, _run_otx, main

# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestBuildOtxParser:
    """Tests for _build_otx_parser."""

    def test_parser_prog_name(self):
        p = _build_otx_parser()
        assert p.prog == "manus-agent otx"

    def test_parser_accepts_cve_id(self):
        p = _build_otx_parser()
        args = p.parse_args(["CVE-2024-3094"])
        assert args.cve_id == "CVE-2024-3094"

    def test_parser_output_default_text(self):
        p = _build_otx_parser()
        args = p.parse_args(["CVE-2024-3094"])
        assert args.output == "text"

    def test_parser_output_json(self):
        p = _build_otx_parser()
        args = p.parse_args(["CVE-2024-3094", "--output", "json"])
        assert args.output == "json"

    def test_parser_output_text_explicit(self):
        p = _build_otx_parser()
        args = p.parse_args(["CVE-2024-3094", "--output", "text"])
        assert args.output == "text"

    def test_parser_rejects_invalid_output(self):
        p = _build_otx_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["CVE-2024-3094", "--output", "csv"])

    def test_parser_no_args_fails(self):
        p = _build_otx_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_parser_has_help(self):
        p = _build_otx_parser()
        # Should contain description about AlienVault
        assert "AlienVault" in p.description or "OTX" in p.description


# ---------------------------------------------------------------------------
# _run_otx tests — invalid inputs
# ---------------------------------------------------------------------------


class TestRunOtxInvalidInputs:
    """Tests for _run_otx with invalid inputs."""

    def test_invalid_cve_format(self, capsys):
        rc = _run_otx(["not-a-cve"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Invalid CVE ID" in captured.err

    def test_empty_cve_id(self, capsys):
        rc = _run_otx([""])
        assert rc == 1
        captured = capsys.readouterr()
        assert "Invalid CVE ID" in captured.err

    def test_partial_cve_id(self, capsys):
        # CVE- alone still starts with CVE- so it passes format check, hits the tool
        # The tool validates more strictly — mock it
        with patch("manus_agent.tools.get_otx_cve_details.get_otx_cve_details") as mock_tool:
            mock_tool.return_value = {
                "status": "error",
                "content": [{"text": "Invalid CVE ID format."}],
            }
            rc = _run_otx(["CVE-"])
            assert rc == 1

    def test_lowercase_cve_normalised(self):
        """Lower-case input should be uppercased before calling the tool."""
        with patch("manus_agent.tools.get_otx_cve_details.get_otx_cve_details") as mock_tool:
            mock_tool.return_value = {
                "status": "success",
                "content": [{"text": "No pulses found"}],
            }
            rc = _run_otx(["cve-2024-3094"])
            assert rc == 0
            call_args = mock_tool.call_args[0][0]
            assert call_args["input"]["cve_id"] == "CVE-2024-3094"


# ---------------------------------------------------------------------------
# _run_otx tests — success with text output
# ---------------------------------------------------------------------------


class TestRunOtxTextOutput:
    """Tests for _run_otx text output mode."""

    def test_no_pulses_message(self, capsys):
        with patch("manus_agent.tools.get_otx_cve_details.get_otx_cve_details") as mock_tool:
            mock_tool.return_value = {
                "status": "success",
                "content": [{"text": "No specific threat intelligence pulses found for CVE-2024-9999"}],
            }
            rc = _run_otx(["CVE-2024-9999"])
            assert rc == 0
            captured = capsys.readouterr()
            assert "No specific threat intelligence" in captured.out

    def test_json_response_printed_as_text(self, capsys):
        pulse_data = {
            "pulse_info": {
                "count": 2,
                "pulses": [
                    {
                        "name": "APT-29 Campaign",
                        "author": {"username": "researcher1"},
                        "created": "2024-01-15T10:00:00Z",
                        "modified": "2024-02-01T15:30:00Z",
                        "tags": ["apt29", "russia", "backdoor"],
                        "indicator_count": 42,
                        "adversary": "APT-29",
                        "TLP": "white",
                        "targeted_countries": ["US", "UK"],
                        "malware_families": [{"display_name": "SunburstBackdoor"}],
                        "attack_ids": [{"display_name": "T1195.002"}],
                    },
                    {
                        "name": "Supply Chain Attack",
                        "author": {"username": "analyst2"},
                        "created": "2024-03-01T08:00:00Z",
                        "modified": "2024-03-10T12:00:00Z",
                        "tags": ["supply-chain"],
                        "indicator_count": 15,
                        "adversary": "",
                        "TLP": "",
                        "targeted_countries": [],
                        "malware_families": [],
                        "attack_ids": [],
                    },
                ],
            }
        }
        with patch("manus_agent.tools.get_otx_cve_details.get_otx_cve_details") as mock_tool:
            mock_tool.return_value = {
                "status": "success",
                "content": [{"json": pulse_data}],
            }
            rc = _run_otx(["CVE-2024-3094"])
            assert rc == 0
            captured = capsys.readouterr()
            assert "AlienVault OTX" in captured.out
            assert "CVE-2024-3094" in captured.out
            assert "APT-29 Campaign" in captured.out
            assert "researcher1" in captured.out
            assert "APT-29" in captured.out
            assert "42" in captured.out
            assert "Supply Chain Attack" in captured.out


# ---------------------------------------------------------------------------
# _run_otx tests — JSON output
# ---------------------------------------------------------------------------


class TestRunOtxJsonOutput:
    """Tests for _run_otx JSON output mode."""

    def test_json_output_structured(self, capsys):
        pulse_data = {
            "pulse_info": {
                "count": 1,
                "pulses": [{"name": "Test Pulse", "author": {"username": "tester"}}],
            }
        }
        with patch("manus_agent.tools.get_otx_cve_details.get_otx_cve_details") as mock_tool:
            mock_tool.return_value = {
                "status": "success",
                "content": [{"json": pulse_data}],
            }
            rc = _run_otx(["CVE-2024-3094", "--output", "json"])
            assert rc == 0
            captured = capsys.readouterr()
            parsed = json.loads(captured.out)
            assert parsed["pulse_info"]["count"] == 1
            assert parsed["pulse_info"]["pulses"][0]["name"] == "Test Pulse"

    def test_json_output_text_message(self, capsys):
        with patch("manus_agent.tools.get_otx_cve_details.get_otx_cve_details") as mock_tool:
            mock_tool.return_value = {
                "status": "success",
                "content": [{"text": "No pulses found for CVE-2024-9999"}],
            }
            rc = _run_otx(["CVE-2024-9999", "--output", "json"])
            assert rc == 0
            captured = capsys.readouterr()
            parsed = json.loads(captured.out)
            assert parsed["cve_id"] == "CVE-2024-9999"
            assert "No pulses found" in parsed["message"]


# ---------------------------------------------------------------------------
# _run_otx tests — error handling
# ---------------------------------------------------------------------------


class TestRunOtxErrors:
    """Tests for _run_otx error paths."""

    def test_api_key_missing(self, capsys):
        with patch("manus_agent.tools.get_otx_cve_details.get_otx_cve_details") as mock_tool:
            mock_tool.return_value = {
                "status": "error",
                "content": [{"text": "AlienVault OTX API key not found."}],
            }
            rc = _run_otx(["CVE-2024-3094"])
            assert rc == 1
            captured = capsys.readouterr()
            assert "API key not found" in captured.err

    def test_http_error(self, capsys):
        with patch("manus_agent.tools.get_otx_cve_details.get_otx_cve_details") as mock_tool:
            mock_tool.return_value = {
                "status": "error",
                "content": [{"text": "Request to AlienVault OTX API failed with HTTP error: 500"}],
            }
            rc = _run_otx(["CVE-2024-3094"])
            assert rc == 1
            captured = capsys.readouterr()
            assert "failed" in captured.err

    def test_tool_import_failure(self, capsys, monkeypatch):
        """Simulate ImportError when the tool module is missing."""
        import builtins

        real_import = builtins.__import__

        def fail_import(name, *args, **kwargs):
            if "get_otx_cve_details" in name:
                raise ImportError("No module named 'manus_agent.tools.get_otx_cve_details'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_import)
        rc = _run_otx(["CVE-2024-3094"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "missing dependencies" in captured.err


# ---------------------------------------------------------------------------
# _print_otx_text tests
# ---------------------------------------------------------------------------


class TestPrintOtxText:
    """Tests for the _print_otx_text helper."""

    def test_basic_pulse_output(self, capsys):
        data = {
            "pulse_info": {
                "count": 1,
                "pulses": [
                    {
                        "name": "Exploit Alert",
                        "author": {"username": "secteam"},
                        "created": "2024-05-10T00:00:00Z",
                        "modified": "2024-05-12T00:00:00Z",
                        "tags": ["exploit", "critical"],
                        "indicator_count": 7,
                        "adversary": "",
                        "TLP": "green",
                        "targeted_countries": [],
                        "malware_families": [],
                        "attack_ids": [],
                    }
                ],
            }
        }
        _print_otx_text("CVE-2024-1111", data)
        captured = capsys.readouterr()
        assert "CVE-2024-1111" in captured.out
        assert "Exploit Alert" in captured.out
        assert "secteam" in captured.out
        assert "TLP: green" in captured.out
        assert "Indicators: 7" in captured.out

    def test_many_pulses_truncation_message(self, capsys):
        pulses = [
            {
                "name": f"Pulse {i}",
                "author": {"username": f"user{i}"},
                "created": "2024-01-01T00:00:00Z",
                "modified": "2024-01-01T00:00:00Z",
                "tags": [],
                "indicator_count": 0,
                "adversary": "",
                "TLP": "",
                "targeted_countries": [],
                "malware_families": [],
                "attack_ids": [],
            }
            for i in range(15)
        ]
        data = {"pulse_info": {"count": 15, "pulses": pulses}}
        _print_otx_text("CVE-2024-2222", data)
        captured = capsys.readouterr()
        assert "and 5 more pulses" in captured.out

    def test_adversary_shown(self, capsys):
        data = {
            "pulse_info": {
                "count": 1,
                "pulses": [
                    {
                        "name": "Threat",
                        "author": {"username": "x"},
                        "created": "2024-01-01T00:00:00Z",
                        "modified": "2024-01-01T00:00:00Z",
                        "tags": [],
                        "indicator_count": 0,
                        "adversary": "Lazarus Group",
                        "TLP": "",
                        "targeted_countries": [],
                        "malware_families": [],
                        "attack_ids": [],
                    }
                ],
            }
        }
        _print_otx_text("CVE-2024-3333", data)
        captured = capsys.readouterr()
        assert "Lazarus Group" in captured.out

    def test_targeted_countries_shown(self, capsys):
        data = {
            "pulse_info": {
                "count": 1,
                "pulses": [
                    {
                        "name": "Campaign",
                        "author": {"username": "y"},
                        "created": "2024-01-01T00:00:00Z",
                        "modified": "2024-01-01T00:00:00Z",
                        "tags": [],
                        "indicator_count": 0,
                        "adversary": "",
                        "TLP": "",
                        "targeted_countries": ["DE", "FR", "JP"],
                        "malware_families": [],
                        "attack_ids": [],
                    }
                ],
            }
        }
        _print_otx_text("CVE-2024-4444", data)
        captured = capsys.readouterr()
        assert "DE" in captured.out
        assert "FR" in captured.out

    def test_malware_families_shown(self, capsys):
        data = {
            "pulse_info": {
                "count": 1,
                "pulses": [
                    {
                        "name": "Malware",
                        "author": {"username": "z"},
                        "created": "2024-01-01T00:00:00Z",
                        "modified": "2024-01-01T00:00:00Z",
                        "tags": [],
                        "indicator_count": 0,
                        "adversary": "",
                        "TLP": "",
                        "targeted_countries": [],
                        "malware_families": [{"display_name": "Emotet"}, {"display_name": "TrickBot"}],
                        "attack_ids": [],
                    }
                ],
            }
        }
        _print_otx_text("CVE-2024-5555", data)
        captured = capsys.readouterr()
        assert "Emotet" in captured.out
        assert "TrickBot" in captured.out

    def test_attack_ids_shown(self, capsys):
        data = {
            "pulse_info": {
                "count": 1,
                "pulses": [
                    {
                        "name": "ATT&CK",
                        "author": {"username": "w"},
                        "created": "2024-01-01T00:00:00Z",
                        "modified": "2024-01-01T00:00:00Z",
                        "tags": [],
                        "indicator_count": 0,
                        "adversary": "",
                        "TLP": "",
                        "targeted_countries": [],
                        "malware_families": [],
                        "attack_ids": [{"display_name": "T1059.001"}, {"display_name": "T1027"}],
                    }
                ],
            }
        }
        _print_otx_text("CVE-2024-6666", data)
        captured = capsys.readouterr()
        assert "T1059.001" in captured.out
        assert "T1027" in captured.out


# ---------------------------------------------------------------------------
# _SUBCOMMANDS registry
# ---------------------------------------------------------------------------


class TestSubcommandsRegistry:
    """Tests for the _SUBCOMMANDS set."""

    def test_otx_in_subcommands(self):
        assert "otx" in _SUBCOMMANDS


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------


class TestMainDispatchOtx:
    """Tests for main() routing to _run_otx."""

    def test_main_dispatches_otx(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["manus-agent", "otx", "CVE-2024-3094"])
        with patch("manus_agent.cli._run_otx", return_value=0) as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_run.assert_called_once_with(["CVE-2024-3094"])

    def test_main_dispatches_otx_with_output_flag(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["manus-agent", "otx", "CVE-2024-3094", "--output", "json"])
        with patch("manus_agent.cli._run_otx", return_value=0) as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_run.assert_called_once_with(["CVE-2024-3094", "--output", "json"])

    def test_main_otx_nonzero_exit(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["manus-agent", "otx", "invalid-id"])
        with patch("manus_agent.cli._run_otx", return_value=1):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
