"""Comprehensive tests for _run_analyze, _run_remediate, and _run_discover.

Covers execution paths not exercised by the existing test suite:
- All output modes (text, json, lark)
- ImportError (missing agent dependencies)
- Agent __init__ failure
- handle_request failure
- verify=True flag effect on _run_analyze
- min_epss boundary validation in _run_discover
- dry_run flag in _run_discover
- main() dispatch for analyze, remediate, and discover subcommands
"""

from __future__ import annotations

import json
import sys
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_vi_agent(*, handle_return="REPORT TEXT"):
    """Context manager: patch VulnerabilityIntelligenceAgent with a minimal stub."""
    from manus_agent.agents.vi_agent import VulnerabilityIntelligenceAgent

    return mock.patch.object(
        VulnerabilityIntelligenceAgent,
        "__init__",
        return_value=None,
    ), mock.patch.object(
        VulnerabilityIntelligenceAgent,
        "handle_request",
        return_value=handle_return,
    )


def _fake_remediation_agent(*, handle_return="Upgrade to 2.0"):
    """Context manager: patch RemediationAgent with a minimal stub."""
    from manus_agent.agents.remediation_agent import RemediationAgent

    return mock.patch.object(
        RemediationAgent,
        "__init__",
        return_value=None,
    ), mock.patch.object(
        RemediationAgent,
        "handle_request",
        return_value=handle_return,
    )


def _fake_discovery_agent(*, handle_return="DISCOVERY RESULTS"):
    """Context manager: patch VulnerabilityDiscoveryAgent with a minimal stub."""
    from manus_agent.agents.vd_agent import VulnerabilityDiscoveryAgent

    return mock.patch.object(
        VulnerabilityDiscoveryAgent,
        "__init__",
        return_value=None,
    ), mock.patch.object(
        VulnerabilityDiscoveryAgent,
        "handle_request",
        return_value=handle_return,
    )


# ---------------------------------------------------------------------------
# _run_analyze
# ---------------------------------------------------------------------------


class TestRunAnalyzeTextOutput:
    def test_text_output_returns_zero(self):
        """_run_analyze with output=text returns exit code 0 on success."""
        from manus_agent import cli

        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            rc = cli._run_analyze(
                cve_id="CVE-2024-3094",
                verify=False,
                output="text",
                config=mock.MagicMock(),
            )
        assert rc == 0

    def test_text_output_calls_handle_request(self):
        """_run_analyze passes a request string containing the CVE to handle_request."""
        from manus_agent import cli

        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle as m:
            cli._run_analyze(
                cve_id="CVE-2024-3094",
                verify=False,
                output="text",
                config=mock.MagicMock(),
            )
        m.assert_called_once()
        assert "CVE-2024-3094" in m.call_args.args[0]

    def test_text_output_does_not_call_print_json(self):
        """_run_analyze with output=text does NOT call console.print_json."""
        from manus_agent import cli

        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json") as m_json:
                cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=False,
                    output="text",
                    config=mock.MagicMock(),
                )
        m_json.assert_not_called()

    def test_verify_true_returns_zero(self):
        """_run_analyze with verify=True still returns 0 on success."""
        from manus_agent import cli

        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            rc = cli._run_analyze(
                cve_id="CVE-2024-3094",
                verify=True,
                output="text",
                config=mock.MagicMock(),
            )
        assert rc == 0

    def test_verify_true_prints_enabled_banner(self, capsys):
        """_run_analyze(verify=True) emits an 'Exploit verification: ENABLED' message."""
        from manus_agent import cli

        print_calls = []
        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print", side_effect=print_calls.append):
                cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=True,
                    output="text",
                    config=mock.MagicMock(),
                )
        messages = [str(c) for c in print_calls]
        assert any("ENABLED" in m for m in messages), f"No ENABLED message in: {messages}"

    def test_verify_false_does_not_print_enabled_banner(self):
        """_run_analyze(verify=False) does NOT print an exploitation verification message."""
        from manus_agent import cli

        print_calls = []
        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print", side_effect=print_calls.append):
                cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=False,
                    output="text",
                    config=mock.MagicMock(),
                )
        messages = [str(c) for c in print_calls]
        assert not any("ENABLED" in m for m in messages), f"Unexpected ENABLED message: {messages}"

    def test_result_string_rendered(self):
        """The string returned by handle_request is passed through str() and rendered."""
        from manus_agent import cli

        m_init, m_handle = _fake_vi_agent(handle_return="CUSTOM REPORT BODY")
        print_calls = []
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print", side_effect=print_calls.append):
                cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=False,
                    output="text",
                    config=mock.MagicMock(),
                )
        # The Panel containing the report text should have been printed;
        # str() on a Rich Panel doesn't include text — check .renderable instead.
        def _panel_text(obj) -> str:
            return str(getattr(obj, "renderable", obj))

        panel_found = any("CUSTOM REPORT BODY" in _panel_text(c) for c in print_calls)
        assert panel_found, f"Report text not in rendered output: {print_calls}"


class TestRunAnalyzeJsonOutput:
    def test_json_output_returns_zero(self):
        """_run_analyze with output=json returns exit code 0."""
        from manus_agent import cli

        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            rc = cli._run_analyze(
                cve_id="CVE-2024-3094",
                verify=False,
                output="json",
                config=mock.MagicMock(),
            )
        assert rc == 0

    def test_json_output_calls_print_json(self):
        """_run_analyze with output=json calls console.print_json exactly once."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=False,
                    output="json",
                    config=mock.MagicMock(),
                )
        assert len(json_calls) == 1

    def test_json_output_is_valid_json(self):
        """_run_analyze(output=json) emits a valid JSON string to print_json."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=False,
                    output="json",
                    config=mock.MagicMock(),
                )
        data = json.loads(json_calls[0])
        assert isinstance(data, dict)

    def test_json_output_contains_cve_field(self):
        """The JSON output includes the 'cve' field with the correct CVE ID."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_analyze(
                    cve_id="CVE-2021-44228",
                    verify=False,
                    output="json",
                    config=mock.MagicMock(),
                )
        data = json.loads(json_calls[0])
        assert data["cve"] == "CVE-2021-44228"

    def test_json_output_contains_report_field(self):
        """The JSON output includes the 'report' field with the agent's response."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_vi_agent(handle_return="Detailed analysis")
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=False,
                    output="json",
                    config=mock.MagicMock(),
                )
        data = json.loads(json_calls[0])
        assert data["report"] == "Detailed analysis"

    def test_json_output_does_not_call_plain_print(self):
        """_run_analyze(output=json) does NOT call console.print for the report Panel."""
        from manus_agent import cli

        json_calls = []
        print_calls = []
        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                with mock.patch.object(cli.console, "print", side_effect=print_calls.append):
                    cli._run_analyze(
                        cve_id="CVE-2024-3094",
                        verify=False,
                        output="json",
                        config=mock.MagicMock(),
                    )
        # Only the "Analyzing CVE-..." banner should be in print_calls, not a Panel
        # (Panel is printed only for text/lark output)
        panel_messages = [c for c in print_calls if hasattr(c, "renderable")]
        assert len(panel_messages) == 0, f"Unexpected Panel in print_calls: {panel_messages}"


class TestRunAnalyzeLarkOutput:
    def test_lark_output_returns_zero(self):
        """_run_analyze with output=lark returns exit code 0."""
        from manus_agent import cli

        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            rc = cli._run_analyze(
                cve_id="CVE-2024-3094",
                verify=False,
                output="lark",
                config=mock.MagicMock(),
            )
        assert rc == 0

    def test_lark_output_prints_delivered_message(self):
        """_run_analyze(output=lark) prints a 'delivered to Lark' notice."""
        from manus_agent import cli

        print_calls = []
        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print", side_effect=print_calls.append):
                cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=False,
                    output="lark",
                    config=mock.MagicMock(),
                )
        messages = [str(c) for c in print_calls]
        assert any("Lark" in m for m in messages), f"No 'Lark' message in: {messages}"

    def test_lark_output_does_not_call_print_json(self):
        """_run_analyze(output=lark) does NOT call console.print_json."""
        from manus_agent import cli

        m_init, m_handle = _fake_vi_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json") as m_json:
                cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=False,
                    output="lark",
                    config=mock.MagicMock(),
                )
        m_json.assert_not_called()


class TestRunAnalyzeErrorPaths:
    def test_import_error_returns_one(self):
        """_run_analyze returns 1 when VulnerabilityIntelligenceAgent cannot be imported."""
        from manus_agent import cli

        with mock.patch.dict(sys.modules, {"manus_agent.agents": None}):
            rc = cli._run_analyze(
                cve_id="CVE-2024-3094",
                verify=False,
                output="text",
                config=mock.MagicMock(),
            )
        assert rc == 1

    def test_import_error_does_not_call_handle_request(self):
        """When the import fails, handle_request is never called."""
        from manus_agent import cli
        from manus_agent.agents.vi_agent import VulnerabilityIntelligenceAgent

        with mock.patch.object(
            VulnerabilityIntelligenceAgent,
            "handle_request",
        ) as m_handle:
            with mock.patch.dict(sys.modules, {"manus_agent.agents": None}):
                cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=False,
                    output="text",
                    config=mock.MagicMock(),
                )
        m_handle.assert_not_called()

    def test_agent_init_failure_returns_one(self):
        """_run_analyze returns 1 when agent __init__ raises an exception."""
        from manus_agent import cli
        from manus_agent.agents.vi_agent import VulnerabilityIntelligenceAgent

        with mock.patch.object(
            VulnerabilityIntelligenceAgent,
            "__init__",
            side_effect=RuntimeError("API key not set"),
        ):
            rc = cli._run_analyze(
                cve_id="CVE-2024-3094",
                verify=False,
                output="text",
                config=mock.MagicMock(),
            )
        assert rc == 1

    def test_agent_init_failure_does_not_call_handle_request(self):
        """When agent __init__ fails, handle_request is never called."""
        from manus_agent import cli
        from manus_agent.agents.vi_agent import VulnerabilityIntelligenceAgent

        with mock.patch.object(
            VulnerabilityIntelligenceAgent,
            "__init__",
            side_effect=RuntimeError("init failed"),
        ):
            with mock.patch.object(
                VulnerabilityIntelligenceAgent,
                "handle_request",
            ) as m_handle:
                cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=False,
                    output="text",
                    config=mock.MagicMock(),
                )
        m_handle.assert_not_called()

    def test_handle_request_exception_returns_one(self):
        """_run_analyze returns 1 when handle_request raises an exception."""
        from manus_agent import cli
        from manus_agent.agents.vi_agent import VulnerabilityIntelligenceAgent

        with mock.patch.object(VulnerabilityIntelligenceAgent, "__init__", return_value=None):
            with mock.patch.object(
                VulnerabilityIntelligenceAgent,
                "handle_request",
                side_effect=RuntimeError("network timeout"),
            ):
                rc = cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=False,
                    output="text",
                    config=mock.MagicMock(),
                )
        assert rc == 1

    def test_handle_request_exception_returns_one_json_mode(self):
        """_run_analyze returns 1 when handle_request raises in json output mode."""
        from manus_agent import cli
        from manus_agent.agents.vi_agent import VulnerabilityIntelligenceAgent

        with mock.patch.object(VulnerabilityIntelligenceAgent, "__init__", return_value=None):
            with mock.patch.object(
                VulnerabilityIntelligenceAgent,
                "handle_request",
                side_effect=ConnectionError("API unreachable"),
            ):
                rc = cli._run_analyze(
                    cve_id="CVE-2024-3094",
                    verify=False,
                    output="json",
                    config=mock.MagicMock(),
                )
        assert rc == 1


# ---------------------------------------------------------------------------
# _run_remediate
# ---------------------------------------------------------------------------


class TestRunRemediateTextOutput:
    def test_text_output_returns_zero(self):
        """_run_remediate with output=text returns exit code 0 on success."""
        from manus_agent import cli

        m_init, m_handle = _fake_remediation_agent()
        with m_init, m_handle:
            rc = cli._run_remediate(
                cve_id="CVE-2024-3094",
                output="text",
                config=mock.MagicMock(),
            )
        assert rc == 0

    def test_text_output_calls_handle_request(self):
        """_run_remediate passes a request string to handle_request."""
        from manus_agent import cli

        m_init, m_handle = _fake_remediation_agent()
        with m_init, m_handle as m:
            cli._run_remediate(
                cve_id="CVE-2024-3094",
                output="text",
                config=mock.MagicMock(),
            )
        m.assert_called_once()

    def test_text_output_does_not_call_print_json(self):
        """_run_remediate with output=text does NOT call console.print_json."""
        from manus_agent import cli

        m_init, m_handle = _fake_remediation_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json") as m_json:
                cli._run_remediate(
                    cve_id="CVE-2024-3094",
                    output="text",
                    config=mock.MagicMock(),
                )
        m_json.assert_not_called()


class TestRunRemediateJsonOutput:
    def test_json_output_returns_zero(self):
        """_run_remediate with output=json returns exit code 0."""
        from manus_agent import cli

        m_init, m_handle = _fake_remediation_agent()
        with m_init, m_handle:
            rc = cli._run_remediate(
                cve_id="CVE-2024-3094",
                output="json",
                config=mock.MagicMock(),
            )
        assert rc == 0

    def test_json_output_calls_print_json(self):
        """_run_remediate(output=json) calls console.print_json exactly once."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_remediation_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_remediate(
                    cve_id="CVE-2024-3094",
                    output="json",
                    config=mock.MagicMock(),
                )
        assert len(json_calls) == 1

    def test_json_output_is_valid_json(self):
        """_run_remediate(output=json) emits a valid JSON string."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_remediation_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_remediate(
                    cve_id="CVE-2024-3094",
                    output="json",
                    config=mock.MagicMock(),
                )
        data = json.loads(json_calls[0])
        assert isinstance(data, dict)

    def test_json_output_contains_cve_field(self):
        """The JSON output includes the 'cve' field matching the input CVE ID."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_remediation_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_remediate(
                    cve_id="CVE-2021-44228",
                    output="json",
                    config=mock.MagicMock(),
                )
        data = json.loads(json_calls[0])
        assert data["cve"] == "CVE-2021-44228"

    def test_json_output_contains_report_field(self):
        """The JSON output includes the 'report' field with the agent's response."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_remediation_agent(handle_return="Apply patch CVE-2024-3094")
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_remediate(
                    cve_id="CVE-2024-3094",
                    output="json",
                    config=mock.MagicMock(),
                )
        data = json.loads(json_calls[0])
        assert data["report"] == "Apply patch CVE-2024-3094"


class TestRunRemediateErrorPaths:
    def test_import_error_returns_one(self):
        """_run_remediate returns 1 when RemediationAgent cannot be imported."""
        from manus_agent import cli

        with mock.patch.dict(sys.modules, {"manus_agent.agents": None}):
            rc = cli._run_remediate(
                cve_id="CVE-2024-3094",
                output="text",
                config=mock.MagicMock(),
            )
        assert rc == 1

    def test_agent_init_failure_returns_one(self):
        """_run_remediate returns 1 when agent __init__ raises an exception."""
        from manus_agent import cli
        from manus_agent.agents.remediation_agent import RemediationAgent

        with mock.patch.object(
            RemediationAgent,
            "__init__",
            side_effect=RuntimeError("credentials missing"),
        ):
            rc = cli._run_remediate(
                cve_id="CVE-2024-3094",
                output="text",
                config=mock.MagicMock(),
            )
        assert rc == 1

    def test_agent_init_failure_does_not_call_handle_request(self):
        """When agent __init__ fails, handle_request is never called."""
        from manus_agent import cli
        from manus_agent.agents.remediation_agent import RemediationAgent

        with mock.patch.object(
            RemediationAgent,
            "__init__",
            side_effect=RuntimeError("init failed"),
        ):
            with mock.patch.object(RemediationAgent, "handle_request") as m_handle:
                cli._run_remediate(
                    cve_id="CVE-2024-3094",
                    output="text",
                    config=mock.MagicMock(),
                )
        m_handle.assert_not_called()

    def test_handle_request_exception_returns_one(self):
        """_run_remediate returns 1 when handle_request raises an exception."""
        from manus_agent import cli
        from manus_agent.agents.remediation_agent import RemediationAgent

        with mock.patch.object(RemediationAgent, "__init__", return_value=None):
            with mock.patch.object(
                RemediationAgent,
                "handle_request",
                side_effect=ConnectionError("API down"),
            ):
                rc = cli._run_remediate(
                    cve_id="CVE-2024-3094",
                    output="text",
                    config=mock.MagicMock(),
                )
        assert rc == 1

    def test_handle_request_exception_json_mode_returns_one(self):
        """_run_remediate returns 1 when handle_request raises in json mode."""
        from manus_agent import cli
        from manus_agent.agents.remediation_agent import RemediationAgent

        with mock.patch.object(RemediationAgent, "__init__", return_value=None):
            with mock.patch.object(
                RemediationAgent,
                "handle_request",
                side_effect=TimeoutError("timed out"),
            ):
                rc = cli._run_remediate(
                    cve_id="CVE-2024-3094",
                    output="json",
                    config=mock.MagicMock(),
                )
        assert rc == 1


# ---------------------------------------------------------------------------
# _run_discover
# ---------------------------------------------------------------------------


class TestRunDiscoverTextOutput:
    def test_text_output_returns_zero(self):
        """_run_discover with output=text returns exit code 0 on success."""
        from manus_agent import cli

        m_init, m_handle = _fake_discovery_agent()
        with m_init, m_handle:
            rc = cli._run_discover(
                since=None,
                min_epss=0.5,
                output="text",
                dry_run=False,
                config=mock.MagicMock(),
            )
        assert rc == 0

    def test_text_output_calls_handle_request(self):
        """_run_discover passes a request string to handle_request."""
        from manus_agent import cli

        m_init, m_handle = _fake_discovery_agent()
        with m_init, m_handle as m:
            cli._run_discover(
                since="2025-06-01",
                min_epss=0.5,
                output="text",
                dry_run=False,
                config=mock.MagicMock(),
            )
        m.assert_called_once()

    def test_since_date_in_request(self):
        """_run_discover forwards the since date in the request to handle_request."""
        from manus_agent import cli
        from manus_agent.agents.vd_agent import VulnerabilityDiscoveryAgent

        requests_seen = []
        m_init, m_handle = _fake_discovery_agent()
        with m_init:
            with mock.patch.object(
                VulnerabilityDiscoveryAgent,
                "handle_request",
                side_effect=lambda r: requests_seen.append(r) or "R",
            ):
                cli._run_discover(
                    since="2025-01-15",
                    min_epss=0.3,
                    output="text",
                    dry_run=False,
                    config=mock.MagicMock(),
                )
        assert requests_seen, "handle_request was not called"
        assert "2025-01-15" in requests_seen[0]

    def test_dry_run_true_returns_zero(self):
        """_run_discover(dry_run=True) returns 0 on success."""
        from manus_agent import cli

        m_init, m_handle = _fake_discovery_agent()
        with m_init, m_handle:
            rc = cli._run_discover(
                since=None,
                min_epss=0.5,
                output="text",
                dry_run=True,
                config=mock.MagicMock(),
            )
        assert rc == 0


class TestRunDiscoverJsonOutput:
    def test_json_output_returns_zero(self):
        """_run_discover with output=json returns exit code 0."""
        from manus_agent import cli

        m_init, m_handle = _fake_discovery_agent()
        with m_init, m_handle:
            rc = cli._run_discover(
                since=None,
                min_epss=0.5,
                output="json",
                dry_run=False,
                config=mock.MagicMock(),
            )
        assert rc == 0

    def test_json_output_calls_print_json(self):
        """_run_discover(output=json) calls console.print_json exactly once."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_discovery_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_discover(
                    since="2025-06-01",
                    min_epss=0.5,
                    output="json",
                    dry_run=False,
                    config=mock.MagicMock(),
                )
        assert len(json_calls) == 1

    def test_json_output_contains_since_field(self):
        """The JSON output includes the 'since' field."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_discovery_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_discover(
                    since="2025-06-01",
                    min_epss=0.5,
                    output="json",
                    dry_run=False,
                    config=mock.MagicMock(),
                )
        data = json.loads(json_calls[0])
        assert data["since"] == "2025-06-01"

    def test_json_output_contains_min_epss_field(self):
        """The JSON output includes the 'min_epss' field."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_discovery_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_discover(
                    since=None,
                    min_epss=0.7,
                    output="json",
                    dry_run=False,
                    config=mock.MagicMock(),
                )
        data = json.loads(json_calls[0])
        assert data["min_epss"] == pytest.approx(0.7)

    def test_json_output_contains_dry_run_field(self):
        """The JSON output includes the 'dry_run' field."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_discovery_agent()
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_discover(
                    since=None,
                    min_epss=0.5,
                    output="json",
                    dry_run=True,
                    config=mock.MagicMock(),
                )
        data = json.loads(json_calls[0])
        assert data["dry_run"] is True

    def test_json_output_contains_result_field(self):
        """The JSON output includes the 'result' field with the agent's response."""
        from manus_agent import cli

        json_calls = []
        m_init, m_handle = _fake_discovery_agent(handle_return="5 CVEs found")
        with m_init, m_handle:
            with mock.patch.object(cli.console, "print_json", side_effect=json_calls.append):
                cli._run_discover(
                    since=None,
                    min_epss=0.5,
                    output="json",
                    dry_run=False,
                    config=mock.MagicMock(),
                )
        data = json.loads(json_calls[0])
        assert data["result"] == "5 CVEs found"


class TestRunDiscoverValidation:
    def test_min_epss_above_one_returns_one(self):
        """_run_discover returns 1 when min_epss > 1.0."""
        from manus_agent import cli

        rc = cli._run_discover(
            since=None,
            min_epss=1.5,
            output="text",
            dry_run=False,
            config=mock.MagicMock(),
        )
        assert rc == 1

    def test_min_epss_below_zero_returns_one(self):
        """_run_discover returns 1 when min_epss < 0.0."""
        from manus_agent import cli

        rc = cli._run_discover(
            since=None,
            min_epss=-0.01,
            output="text",
            dry_run=False,
            config=mock.MagicMock(),
        )
        assert rc == 1

    def test_min_epss_exactly_zero_is_valid(self):
        """_run_discover returns 0 when min_epss == 0.0 (inclusive lower bound)."""
        from manus_agent import cli

        m_init, m_handle = _fake_discovery_agent()
        with m_init, m_handle:
            rc = cli._run_discover(
                since=None,
                min_epss=0.0,
                output="text",
                dry_run=False,
                config=mock.MagicMock(),
            )
        assert rc == 0

    def test_min_epss_exactly_one_is_valid(self):
        """_run_discover returns 0 when min_epss == 1.0 (inclusive upper bound)."""
        from manus_agent import cli

        m_init, m_handle = _fake_discovery_agent()
        with m_init, m_handle:
            rc = cli._run_discover(
                since=None,
                min_epss=1.0,
                output="text",
                dry_run=False,
                config=mock.MagicMock(),
            )
        assert rc == 0

    def test_invalid_min_epss_does_not_call_handle_request(self):
        """When min_epss is out of range, handle_request is never called."""
        from manus_agent import cli
        from manus_agent.agents.vd_agent import VulnerabilityDiscoveryAgent

        with mock.patch.object(
            VulnerabilityDiscoveryAgent,
            "handle_request",
        ) as m_handle:
            cli._run_discover(
                since=None,
                min_epss=2.0,
                output="text",
                dry_run=False,
                config=mock.MagicMock(),
            )
        m_handle.assert_not_called()


class TestRunDiscoverErrorPaths:
    def test_import_error_returns_one(self):
        """_run_discover returns 1 when VulnerabilityDiscoveryAgent cannot be imported."""
        from manus_agent import cli

        with mock.patch.dict(sys.modules, {"manus_agent.agents": None}):
            rc = cli._run_discover(
                since=None,
                min_epss=0.5,
                output="text",
                dry_run=False,
                config=mock.MagicMock(),
            )
        assert rc == 1

    def test_agent_init_failure_returns_one(self):
        """_run_discover returns 1 when agent __init__ raises."""
        from manus_agent import cli
        from manus_agent.agents.vd_agent import VulnerabilityDiscoveryAgent

        with mock.patch.object(
            VulnerabilityDiscoveryAgent,
            "__init__",
            side_effect=RuntimeError("creds missing"),
        ):
            rc = cli._run_discover(
                since=None,
                min_epss=0.5,
                output="text",
                dry_run=False,
                config=mock.MagicMock(),
            )
        assert rc == 1

    def test_agent_init_failure_does_not_call_handle_request(self):
        """When agent __init__ fails, handle_request is never called."""
        from manus_agent import cli
        from manus_agent.agents.vd_agent import VulnerabilityDiscoveryAgent

        with mock.patch.object(
            VulnerabilityDiscoveryAgent,
            "__init__",
            side_effect=RuntimeError("init failed"),
        ):
            with mock.patch.object(
                VulnerabilityDiscoveryAgent,
                "handle_request",
            ) as m_handle:
                cli._run_discover(
                    since=None,
                    min_epss=0.5,
                    output="text",
                    dry_run=False,
                    config=mock.MagicMock(),
                )
        m_handle.assert_not_called()

    def test_handle_request_exception_returns_one(self):
        """_run_discover returns 1 when handle_request raises."""
        from manus_agent import cli
        from manus_agent.agents.vd_agent import VulnerabilityDiscoveryAgent

        with mock.patch.object(VulnerabilityDiscoveryAgent, "__init__", return_value=None):
            with mock.patch.object(
                VulnerabilityDiscoveryAgent,
                "handle_request",
                side_effect=RuntimeError("API timeout"),
            ):
                rc = cli._run_discover(
                    since=None,
                    min_epss=0.5,
                    output="text",
                    dry_run=False,
                    config=mock.MagicMock(),
                )
        assert rc == 1

    def test_handle_request_exception_json_mode_returns_one(self):
        """_run_discover returns 1 when handle_request raises in json mode."""
        from manus_agent import cli
        from manus_agent.agents.vd_agent import VulnerabilityDiscoveryAgent

        with mock.patch.object(VulnerabilityDiscoveryAgent, "__init__", return_value=None):
            with mock.patch.object(
                VulnerabilityDiscoveryAgent,
                "handle_request",
                side_effect=ConnectionError("network error"),
            ):
                rc = cli._run_discover(
                    since=None,
                    min_epss=0.5,
                    output="json",
                    dry_run=False,
                    config=mock.MagicMock(),
                )
        assert rc == 1


# ---------------------------------------------------------------------------
# main() dispatch for analyze
# ---------------------------------------------------------------------------


class TestMainDispatchAnalyze:
    def test_analyze_in_subcommands(self):
        """'analyze' is registered in the _SUBCOMMANDS set."""
        from manus_agent.cli import _SUBCOMMANDS

        assert "analyze" in _SUBCOMMANDS

    def test_main_routes_analyze(self):
        """main() routes 'analyze CVE-...' to _run_analyze (not single-shot)."""
        from manus_agent import cli

        with mock.patch.object(cli, "_run_analyze", return_value=0) as m_run:
            with mock.patch.object(sys, "argv", ["manus-agent", "analyze", "CVE-2024-3094"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit) as exc_info:
                        cli.main()
        assert exc_info.value.code == 0
        m_run.assert_called_once()

    def test_main_analyze_passes_cve_id(self):
        """main() forwards the CVE ID to _run_analyze."""
        from manus_agent import cli

        captured = {}

        def fake_run_analyze(*, cve_id, verify, output, config):
            captured["cve_id"] = cve_id
            return 0

        with mock.patch.object(cli, "_run_analyze", side_effect=fake_run_analyze):
            with mock.patch.object(sys, "argv", ["manus-agent", "analyze", "CVE-2021-44228"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit):
                        cli.main()

        assert captured["cve_id"] == "CVE-2021-44228"

    def test_main_analyze_verify_flag_forwarded(self):
        """main() forwards --verify=True to _run_analyze."""
        from manus_agent import cli

        captured = {}

        def fake_run_analyze(*, cve_id, verify, output, config):
            captured["verify"] = verify
            return 0

        with mock.patch.object(cli, "_run_analyze", side_effect=fake_run_analyze):
            with mock.patch.object(sys, "argv", ["manus-agent", "analyze", "CVE-2024-3094", "--verify"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit):
                        cli.main()

        assert captured["verify"] is True

    def test_main_analyze_output_json_forwarded(self):
        """main() forwards --output json to _run_analyze."""
        from manus_agent import cli

        captured = {}

        def fake_run_analyze(*, cve_id, verify, output, config):
            captured["output"] = output
            return 0

        with mock.patch.object(cli, "_run_analyze", side_effect=fake_run_analyze):
            with mock.patch.object(
                sys,
                "argv",
                ["manus-agent", "analyze", "CVE-2024-3094", "--output", "json"],
            ):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit):
                        cli.main()

        assert captured["output"] == "json"

    def test_main_analyze_default_output_is_text(self):
        """main() defaults output='text' when --output is not supplied."""
        from manus_agent import cli

        captured = {}

        def fake_run_analyze(*, cve_id, verify, output, config):
            captured["output"] = output
            return 0

        with mock.patch.object(cli, "_run_analyze", side_effect=fake_run_analyze):
            with mock.patch.object(sys, "argv", ["manus-agent", "analyze", "CVE-2024-3094"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit):
                        cli.main()

        assert captured["output"] == "text"

    def test_main_analyze_nonzero_rc_propagated(self):
        """main() exits with the same code as _run_analyze when it returns non-zero."""
        from manus_agent import cli

        with mock.patch.object(cli, "_run_analyze", return_value=1):
            with mock.patch.object(sys, "argv", ["manus-agent", "analyze", "CVE-2024-3094"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit) as exc_info:
                        cli.main()
        assert exc_info.value.code == 1

    def test_main_analyze_does_not_route_to_single_shot(self):
        """main('analyze CVE-...') does NOT invoke _run_single_shot."""
        from manus_agent import cli

        with mock.patch.object(cli, "_run_analyze", return_value=0):
            with mock.patch.object(sys, "argv", ["manus-agent", "analyze", "CVE-2024-3094"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with mock.patch.object(cli, "_run_single_shot") as m_ss:
                        with pytest.raises(SystemExit):
                            cli.main()
        m_ss.assert_not_called()


# ---------------------------------------------------------------------------
# main() dispatch for remediate
# ---------------------------------------------------------------------------


class TestMainDispatchRemediate:
    def test_remediate_in_subcommands(self):
        """'remediate' is registered in the _SUBCOMMANDS set."""
        from manus_agent.cli import _SUBCOMMANDS

        assert "remediate" in _SUBCOMMANDS

    def test_main_routes_remediate(self):
        """main() routes 'remediate CVE-...' to _run_remediate."""
        from manus_agent import cli

        with mock.patch.object(cli, "_run_remediate", return_value=0) as m_run:
            with mock.patch.object(sys, "argv", ["manus-agent", "remediate", "CVE-2024-3094"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit) as exc_info:
                        cli.main()
        assert exc_info.value.code == 0
        m_run.assert_called_once()

    def test_main_remediate_passes_cve_id(self):
        """main() forwards the CVE ID to _run_remediate."""
        from manus_agent import cli

        captured = {}

        def fake_run_remediate(*, cve_id, output, config):
            captured["cve_id"] = cve_id
            return 0

        with mock.patch.object(cli, "_run_remediate", side_effect=fake_run_remediate):
            with mock.patch.object(sys, "argv", ["manus-agent", "remediate", "CVE-2021-44228"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit):
                        cli.main()

        assert captured["cve_id"] == "CVE-2021-44228"

    def test_main_remediate_output_json_forwarded(self):
        """main() forwards --output json to _run_remediate."""
        from manus_agent import cli

        captured = {}

        def fake_run_remediate(*, cve_id, output, config):
            captured["output"] = output
            return 0

        with mock.patch.object(cli, "_run_remediate", side_effect=fake_run_remediate):
            with mock.patch.object(
                sys,
                "argv",
                ["manus-agent", "remediate", "CVE-2024-3094", "--output", "json"],
            ):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit):
                        cli.main()

        assert captured["output"] == "json"

    def test_main_remediate_default_output_is_text(self):
        """main() defaults output='text' when --output is not supplied."""
        from manus_agent import cli

        captured = {}

        def fake_run_remediate(*, cve_id, output, config):
            captured["output"] = output
            return 0

        with mock.patch.object(cli, "_run_remediate", side_effect=fake_run_remediate):
            with mock.patch.object(sys, "argv", ["manus-agent", "remediate", "CVE-2024-3094"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit):
                        cli.main()

        assert captured["output"] == "text"

    def test_main_remediate_nonzero_rc_propagated(self):
        """main() exits with the same code as _run_remediate when non-zero."""
        from manus_agent import cli

        with mock.patch.object(cli, "_run_remediate", return_value=1):
            with mock.patch.object(sys, "argv", ["manus-agent", "remediate", "CVE-2024-3094"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit) as exc_info:
                        cli.main()
        assert exc_info.value.code == 1

    def test_main_remediate_does_not_route_to_single_shot(self):
        """main('remediate CVE-...') does NOT invoke _run_single_shot."""
        from manus_agent import cli

        with mock.patch.object(cli, "_run_remediate", return_value=0):
            with mock.patch.object(sys, "argv", ["manus-agent", "remediate", "CVE-2024-3094"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with mock.patch.object(cli, "_run_single_shot") as m_ss:
                        with pytest.raises(SystemExit):
                            cli.main()
        m_ss.assert_not_called()


# ---------------------------------------------------------------------------
# main() dispatch for discover
# ---------------------------------------------------------------------------


class TestMainDispatchDiscover:
    def test_discover_in_subcommands(self):
        """'discover' is registered in the _SUBCOMMANDS set."""
        from manus_agent.cli import _SUBCOMMANDS

        assert "discover" in _SUBCOMMANDS

    def test_main_routes_discover(self):
        """main() routes 'discover' to _run_discover."""
        from manus_agent import cli

        with mock.patch.object(cli, "_run_discover", return_value=0) as m_run:
            with mock.patch.object(sys, "argv", ["manus-agent", "discover"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit) as exc_info:
                        cli.main()
        assert exc_info.value.code == 0
        m_run.assert_called_once()

    def test_main_discover_since_forwarded(self):
        """main() forwards --since to _run_discover."""
        from manus_agent import cli

        captured = {}

        def fake_run_discover(*, since, min_epss, output, dry_run, config):
            captured["since"] = since
            return 0

        with mock.patch.object(cli, "_run_discover", side_effect=fake_run_discover):
            with mock.patch.object(
                sys,
                "argv",
                ["manus-agent", "discover", "--since", "2025-06-01"],
            ):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit):
                        cli.main()

        assert captured["since"] == "2025-06-01"

    def test_main_discover_min_epss_forwarded(self):
        """main() forwards --min-epss to _run_discover."""
        from manus_agent import cli

        captured = {}

        def fake_run_discover(*, since, min_epss, output, dry_run, config):
            captured["min_epss"] = min_epss
            return 0

        with mock.patch.object(cli, "_run_discover", side_effect=fake_run_discover):
            with mock.patch.object(
                sys,
                "argv",
                ["manus-agent", "discover", "--min-epss", "0.8"],
            ):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit):
                        cli.main()

        assert captured["min_epss"] == pytest.approx(0.8)

    def test_main_discover_dry_run_forwarded(self):
        """main() forwards --dry-run to _run_discover."""
        from manus_agent import cli

        captured = {}

        def fake_run_discover(*, since, min_epss, output, dry_run, config):
            captured["dry_run"] = dry_run
            return 0

        with mock.patch.object(cli, "_run_discover", side_effect=fake_run_discover):
            with mock.patch.object(sys, "argv", ["manus-agent", "discover", "--dry-run"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit):
                        cli.main()

        assert captured["dry_run"] is True

    def test_main_discover_output_json_forwarded(self):
        """main() forwards --output json to _run_discover."""
        from manus_agent import cli

        captured = {}

        def fake_run_discover(*, since, min_epss, output, dry_run, config):
            captured["output"] = output
            return 0

        with mock.patch.object(cli, "_run_discover", side_effect=fake_run_discover):
            with mock.patch.object(
                sys,
                "argv",
                ["manus-agent", "discover", "--output", "json"],
            ):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit):
                        cli.main()

        assert captured["output"] == "json"

    def test_main_discover_nonzero_rc_propagated(self):
        """main() exits with the same code as _run_discover when non-zero."""
        from manus_agent import cli

        with mock.patch.object(cli, "_run_discover", return_value=1):
            with mock.patch.object(sys, "argv", ["manus-agent", "discover"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with pytest.raises(SystemExit) as exc_info:
                        cli.main()
        assert exc_info.value.code == 1

    def test_main_discover_does_not_route_to_single_shot(self):
        """main('discover') does NOT invoke _run_single_shot."""
        from manus_agent import cli

        with mock.patch.object(cli, "_run_discover", return_value=0):
            with mock.patch.object(sys, "argv", ["manus-agent", "discover"]):
                with mock.patch("manus_agent.cli.Config") as m_cfg:
                    m_cfg.from_file.return_value = mock.MagicMock()
                    with mock.patch.object(cli, "_run_single_shot") as m_ss:
                        with pytest.raises(SystemExit):
                            cli.main()
        m_ss.assert_not_called()
