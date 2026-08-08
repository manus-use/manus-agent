"""Comprehensive test suite for the export_sarif module.

Tests cover:
- TOOL_SPEC contract validation
- Input validation (missing/invalid findings)
- SARIF schema compliance (version, $schema, runs structure)
- Rule generation (severity mapping, CVSS scores, CWE tags, KEV tags)
- Result generation (levels, messages, locations, fixes)
- Deduplication of CVE findings
- CLI subcommand (stdin/file input, stdout/file output, error handling)
- Edge cases (empty findings, minimal findings, large inputs)

All tests are unit tests — no network calls, no filesystem side effects.
"""

from __future__ import annotations

import json
from io import StringIO
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_finding(cve_id: str = "CVE-2024-3094", **kwargs: Any) -> dict[str, Any]:
    """Create a minimal valid finding."""
    finding: dict[str, Any] = {"cve_id": cve_id}
    finding.update(kwargs)
    return finding


def _full_finding(**kwargs: Any) -> dict[str, Any]:
    """Create a fully-populated finding."""
    finding = {
        "cve_id": "CVE-2024-3094",
        "severity": "CRITICAL",
        "description": "A backdoor was discovered in xz-utils versions 5.6.0 and 5.6.1.",
        "affected_component": "xz-utils",
        "affected_versions": ">=5.6.0, <5.6.2",
        "fix_version": "5.6.2",
        "cvss_score": 10.0,
        "epss_score": 0.97,
        "cwe_id": "CWE-506",
        "in_kev": True,
        "references": [
            "https://nvd.nist.gov/vuln/detail/CVE-2024-3094",
            "https://www.openwall.com/lists/oss-security/2024/03/29/4",
        ],
        "file_path": "requirements.txt",
        "start_line": 12,
        "end_line": 12,
    }
    finding.update(kwargs)
    return finding


def _make_tool_use(findings: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Build a ToolUse dict for the export_sarif function."""
    tool_input: dict[str, Any] = {"findings": findings}
    tool_input.update(kwargs)
    return {"toolUseId": "test-sarif-001", "input": tool_input}


# ---------------------------------------------------------------------------
# TOOL_SPEC contract
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_module_imports(self):
        import manus_agent.tools.export_sarif as m

        assert hasattr(m, "export_sarif")
        assert hasattr(m, "TOOL_SPEC")
        assert hasattr(m, "findings_to_sarif")

    def test_tool_spec_name(self):
        from manus_agent.tools.export_sarif import TOOL_SPEC

        assert TOOL_SPEC["name"] == "export_sarif"

    def test_tool_spec_has_input_schema(self):
        from manus_agent.tools.export_sarif import TOOL_SPEC

        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "findings" in schema["properties"]
        assert "findings" in schema["required"]

    def test_tool_spec_findings_is_array(self):
        from manus_agent.tools.export_sarif import TOOL_SPEC

        findings_schema = TOOL_SPEC["inputSchema"]["json"]["properties"]["findings"]
        assert findings_schema["type"] == "array"
        assert "items" in findings_schema

    def test_tool_spec_finding_requires_cve_id(self):
        from manus_agent.tools.export_sarif import TOOL_SPEC

        items_schema = TOOL_SPEC["inputSchema"]["json"]["properties"]["findings"]["items"]
        assert "cve_id" in items_schema["properties"]
        assert "cve_id" in items_schema["required"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_findings_not_a_list_returns_error(self):
        from manus_agent.tools.export_sarif import export_sarif

        tool = {"toolUseId": "t1", "input": {"findings": "not-a-list"}}
        result = export_sarif(tool)
        assert result["status"] == "error"
        assert "must be a list" in result["content"][0]["text"]

    def test_finding_not_a_dict_returns_error(self):
        from manus_agent.tools.export_sarif import export_sarif

        tool = {"toolUseId": "t1", "input": {"findings": ["not-a-dict"]}}
        result = export_sarif(tool)
        assert result["status"] == "error"
        assert "index 0 must be an object" in result["content"][0]["text"]

    def test_finding_missing_cve_id_returns_error(self):
        from manus_agent.tools.export_sarif import export_sarif

        tool = {"toolUseId": "t1", "input": {"findings": [{"severity": "HIGH"}]}}
        result = export_sarif(tool)
        assert result["status"] == "error"
        assert "missing required field 'cve_id'" in result["content"][0]["text"]

    def test_empty_findings_list_succeeds(self):
        from manus_agent.tools.export_sarif import export_sarif

        tool = _make_tool_use([])
        result = export_sarif(tool)
        assert result["status"] == "success"

    def test_minimal_finding_succeeds(self):
        from manus_agent.tools.export_sarif import export_sarif

        tool = _make_tool_use([_minimal_finding()])
        result = export_sarif(tool)
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# SARIF schema compliance
# ---------------------------------------------------------------------------


class TestSarifSchema:
    def test_sarif_version(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()])
        assert sarif["version"] == "2.1.0"

    def test_sarif_schema_uri(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()])
        assert "$schema" in sarif
        assert "sarif-schema-2.1.0" in sarif["$schema"]

    def test_sarif_has_runs_array(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()])
        assert "runs" in sarif
        assert isinstance(sarif["runs"], list)
        assert len(sarif["runs"]) == 1

    def test_run_has_tool_driver(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()])
        run = sarif["runs"][0]
        assert "tool" in run
        assert "driver" in run["tool"]
        assert run["tool"]["driver"]["name"] == "manus-agent"

    def test_run_has_results(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()])
        run = sarif["runs"][0]
        assert "results" in run
        assert isinstance(run["results"], list)

    def test_run_has_invocations(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()])
        run = sarif["runs"][0]
        assert "invocations" in run
        assert run["invocations"][0]["executionSuccessful"] is True
        assert "endTimeUtc" in run["invocations"][0]

    def test_tool_version_custom(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()], tool_version="1.2.3")
        assert sarif["runs"][0]["tool"]["driver"]["version"] == "1.2.3"

    def test_tool_version_defaults_to_package(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()])
        version = sarif["runs"][0]["tool"]["driver"]["version"]
        assert isinstance(version, str)
        assert len(version) > 0

    def test_driver_information_uri(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()])
        driver = sarif["runs"][0]["tool"]["driver"]
        assert driver["informationUri"] == "https://github.com/manus-use/manus-agent"

    def test_sarif_is_valid_json(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_full_finding()])
        # Should serialize without errors
        json_str = json.dumps(sarif)
        parsed = json.loads(json_str)
        assert parsed["version"] == "2.1.0"


# ---------------------------------------------------------------------------
# Rule generation
# ---------------------------------------------------------------------------


class TestRuleGeneration:
    def test_rule_id_is_cve_id(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding("CVE-2024-3094")])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert rules[0]["id"] == "CVE-2024-3094"

    def test_rule_name_replaces_hyphens(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding("CVE-2024-3094")])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert rules[0]["name"] == "CVE_2024_3094"

    def test_rule_help_uri(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding("CVE-2024-3094")])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert rules[0]["helpUri"] == "https://nvd.nist.gov/vuln/detail/CVE-2024-3094"

    def test_rule_severity_mapping_critical(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(severity="CRITICAL")])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert rules[0]["properties"]["security-severity"] == "9.0"

    def test_rule_severity_mapping_high(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(severity="HIGH")])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert rules[0]["properties"]["security-severity"] == "7.0"

    def test_rule_severity_mapping_medium(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(severity="MEDIUM")])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert rules[0]["properties"]["security-severity"] == "4.0"

    def test_rule_severity_mapping_low(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(severity="LOW")])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert rules[0]["properties"]["security-severity"] == "1.0"

    def test_rule_severity_unknown_defaults_medium(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(severity="BOGUS")])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert rules[0]["properties"]["security-severity"] == "4.0"

    def test_rule_cvss_score_overrides_severity_map(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(severity="HIGH", cvss_score=8.5)])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert rules[0]["properties"]["security-severity"] == "8.5"
        assert rules[0]["properties"]["cvss-score"] == 8.5

    def test_rule_epss_score_in_properties(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(epss_score=0.42)])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert rules[0]["properties"]["epss-score"] == 0.42

    def test_rule_cwe_tag(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(cwe_id="CWE-79")])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert "CWE-79" in rules[0]["properties"]["tags"]

    def test_rule_kev_tag(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(in_kev=True)])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert "cisa-kev" in rules[0]["properties"]["tags"]

    def test_rule_no_kev_tag_when_false(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(in_kev=False)])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert "cisa-kev" not in rules[0]["properties"]["tags"]

    def test_rule_has_security_tag(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert "security" in rules[0]["properties"]["tags"]
        assert "vulnerability" in rules[0]["properties"]["tags"]

    def test_rule_help_contains_fix_info(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(affected_component="xz-utils", fix_version="5.6.2")])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        help_text = rules[0]["help"]["text"]
        assert "xz-utils" in help_text
        assert "5.6.2" in help_text

    def test_rule_description_truncation(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        long_desc = "A" * 2000
        sarif = findings_to_sarif([_minimal_finding(description=long_desc)])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert len(rules[0]["fullDescription"]["text"]) <= 1000


# ---------------------------------------------------------------------------
# Result generation
# ---------------------------------------------------------------------------


class TestResultGeneration:
    def test_result_rule_id(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding("CVE-2021-44228")])
        results = sarif["runs"][0]["results"]
        assert results[0]["ruleId"] == "CVE-2021-44228"

    def test_result_rule_index(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding("CVE-2021-44228"), _minimal_finding("CVE-2024-3094")])
        results = sarif["runs"][0]["results"]
        assert results[0]["ruleIndex"] == 0
        assert results[1]["ruleIndex"] == 1

    def test_result_level_error_for_critical(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(severity="CRITICAL")])
        results = sarif["runs"][0]["results"]
        assert results[0]["level"] == "error"

    def test_result_level_error_for_high(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(severity="HIGH")])
        results = sarif["runs"][0]["results"]
        assert results[0]["level"] == "error"

    def test_result_level_warning_for_medium(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(severity="MEDIUM")])
        results = sarif["runs"][0]["results"]
        assert results[0]["level"] == "warning"

    def test_result_level_note_for_low(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(severity="LOW")])
        results = sarif["runs"][0]["results"]
        assert results[0]["level"] == "note"

    def test_result_message_contains_cve_id(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding("CVE-2024-3094")])
        results = sarif["runs"][0]["results"]
        assert "CVE-2024-3094" in results[0]["message"]["text"]

    def test_result_message_contains_component(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(affected_component="openssl")])
        results = sarif["runs"][0]["results"]
        assert "openssl" in results[0]["message"]["text"]

    def test_result_message_contains_kev_warning(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(in_kev=True)])
        results = sarif["runs"][0]["results"]
        assert "ACTIVELY EXPLOITED" in results[0]["message"]["text"]

    def test_result_physical_location_with_file(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(file_path="package.json", start_line=5)])
        results = sarif["runs"][0]["results"]
        loc = results[0]["locations"][0]["physicalLocation"]
        assert loc["artifactLocation"]["uri"] == "package.json"
        assert loc["region"]["startLine"] == 5

    def test_result_physical_location_with_end_line(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(file_path="go.mod", start_line=10, end_line=12)])
        results = sarif["runs"][0]["results"]
        region = results[0]["locations"][0]["physicalLocation"]["region"]
        assert region["startLine"] == 10
        assert region["endLine"] == 12

    def test_result_logical_location_without_file(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(affected_component="lodash")])
        results = sarif["runs"][0]["results"]
        logical_locs = results[0]["locations"][0]["logicalLocations"]
        assert logical_locs[0]["name"] == "lodash"
        assert logical_locs[0]["kind"] == "module"

    def test_result_fix_information(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(affected_component="numpy", fix_version="1.26.0")])
        results = sarif["runs"][0]["results"]
        fixes = results[0]["fixes"]
        assert len(fixes) == 1
        assert "1.26.0" in fixes[0]["description"]["text"]
        assert "numpy" in fixes[0]["description"]["text"]

    def test_result_no_fix_when_no_fix_version(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()])
        results = sarif["runs"][0]["results"]
        assert "fixes" not in results[0]

    def test_result_related_locations_from_references(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        refs = ["https://example.com/advisory", "https://github.com/fix"]
        sarif = findings_to_sarif([_minimal_finding(references=refs)])
        results = sarif["runs"][0]["results"]
        related = results[0]["relatedLocations"]
        assert len(related) == 2
        assert related[0]["message"]["text"] == refs[0]

    def test_result_references_capped_at_10(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        refs = [f"https://example.com/{i}" for i in range(20)]
        sarif = findings_to_sarif([_minimal_finding(references=refs)])
        results = sarif["runs"][0]["results"]
        assert len(results[0]["relatedLocations"]) == 10


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_duplicate_cve_ids_deduplicated(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        findings = [
            _minimal_finding("CVE-2024-3094", severity="HIGH"),
            _minimal_finding("CVE-2024-3094", severity="CRITICAL"),
        ]
        sarif = findings_to_sarif(findings)
        results = sarif["runs"][0]["results"]
        assert len(results) == 1
        # First occurrence wins
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert len(rules) == 1

    def test_case_insensitive_deduplication(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        findings = [
            _minimal_finding("cve-2024-3094"),
            _minimal_finding("CVE-2024-3094"),
        ]
        sarif = findings_to_sarif(findings)
        results = sarif["runs"][0]["results"]
        assert len(results) == 1

    def test_different_cves_not_deduplicated(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        findings = [
            _minimal_finding("CVE-2024-3094"),
            _minimal_finding("CVE-2021-44228"),
        ]
        sarif = findings_to_sarif(findings)
        results = sarif["runs"][0]["results"]
        assert len(results) == 2

    def test_empty_cve_id_skipped(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        findings = [
            {"cve_id": ""},
            _minimal_finding("CVE-2024-3094"),
        ]
        sarif = findings_to_sarif(findings)
        results = sarif["runs"][0]["results"]
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Tool function integration
# ---------------------------------------------------------------------------


class TestToolFunction:
    def test_success_returns_text_and_json(self):
        from manus_agent.tools.export_sarif import export_sarif

        tool = _make_tool_use([_full_finding()])
        result = export_sarif(tool)
        assert result["status"] == "success"
        content_types = [list(c.keys())[0] for c in result["content"]]
        assert "text" in content_types
        assert "json" in content_types

    def test_success_text_mentions_count(self):
        from manus_agent.tools.export_sarif import export_sarif

        tool = _make_tool_use([_minimal_finding(), _minimal_finding("CVE-2021-44228")])
        result = export_sarif(tool)
        assert "2 finding(s)" in result["content"][0]["text"]

    def test_success_json_is_valid_sarif(self):
        from manus_agent.tools.export_sarif import export_sarif

        tool = _make_tool_use([_full_finding()])
        result = export_sarif(tool)
        sarif_json = next(c["json"] for c in result["content"] if "json" in c)
        assert sarif_json["version"] == "2.1.0"
        assert len(sarif_json["runs"][0]["results"]) == 1

    def test_tool_use_id_preserved(self):
        from manus_agent.tools.export_sarif import export_sarif

        tool = {"toolUseId": "custom-id-xyz", "input": {"findings": [_minimal_finding()]}}
        result = export_sarif(tool)
        assert result["toolUseId"] == "custom-id-xyz"

    def test_custom_tool_version_passed(self):
        from manus_agent.tools.export_sarif import export_sarif

        tool = _make_tool_use([_minimal_finding()], tool_version="2.0.0-beta")
        result = export_sarif(tool)
        sarif_json = next(c["json"] for c in result["content"] if "json" in c)
        assert sarif_json["runs"][0]["tool"]["driver"]["version"] == "2.0.0-beta"

    def test_empty_findings_returns_zero_results(self):
        from manus_agent.tools.export_sarif import export_sarif

        tool = _make_tool_use([])
        result = export_sarif(tool)
        assert result["status"] == "success"
        sarif_json = next(c["json"] for c in result["content"] if "json" in c)
        assert len(sarif_json["runs"][0]["results"]) == 0
        assert "0 finding(s)" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# CLI subcommand
# ---------------------------------------------------------------------------


class TestCLI:
    def test_subcommand_registered(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "export-sarif" in _SUBCOMMANDS

    def test_cli_help_exits_zero(self):
        from manus_agent.tools.export_sarif import run_export_sarif_cli

        with pytest.raises(SystemExit) as exc_info:
            run_export_sarif_cli(["--help"])
        assert exc_info.value.code == 0

    def test_cli_stdin_to_stdout(self, monkeypatch, capsys):
        from manus_agent.tools.export_sarif import run_export_sarif_cli

        input_data = json.dumps([{"cve_id": "CVE-2024-3094", "severity": "HIGH"}])
        monkeypatch.setattr("sys.stdin", StringIO(input_data))
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        rc = run_export_sarif_cli([])
        assert rc == 0
        out = capsys.readouterr().out
        sarif = json.loads(out)
        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"][0]["results"]) == 1

    def test_cli_input_file(self, tmp_path, capsys):
        from manus_agent.tools.export_sarif import run_export_sarif_cli

        input_file = tmp_path / "findings.json"
        input_file.write_text(json.dumps([{"cve_id": "CVE-2021-44228", "severity": "CRITICAL"}]))

        rc = run_export_sarif_cli(["--input", str(input_file)])
        assert rc == 0
        out = capsys.readouterr().out
        sarif = json.loads(out)
        assert sarif["runs"][0]["results"][0]["ruleId"] == "CVE-2021-44228"

    def test_cli_output_file(self, tmp_path, monkeypatch, capsys):
        from manus_agent.tools.export_sarif import run_export_sarif_cli

        input_data = json.dumps([{"cve_id": "CVE-2024-3094"}])
        monkeypatch.setattr("sys.stdin", StringIO(input_data))
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        output_file = tmp_path / "report.sarif"
        rc = run_export_sarif_cli(["--output", str(output_file)])
        assert rc == 0
        assert output_file.exists()
        sarif = json.loads(output_file.read_text())
        assert sarif["version"] == "2.1.0"
        # Should print confirmation to stderr
        err = capsys.readouterr().err
        assert "SARIF written" in err

    def test_cli_compact_output(self, monkeypatch, capsys):
        from manus_agent.tools.export_sarif import run_export_sarif_cli

        input_data = json.dumps([{"cve_id": "CVE-2024-3094"}])
        monkeypatch.setattr("sys.stdin", StringIO(input_data))
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        rc = run_export_sarif_cli(["--compact"])
        assert rc == 0
        out = capsys.readouterr().out
        # Compact output should be a single line
        assert "\n" == out[-1]  # ends with newline
        lines = out.strip().split("\n")
        assert len(lines) == 1

    def test_cli_custom_tool_version(self, monkeypatch, capsys):
        from manus_agent.tools.export_sarif import run_export_sarif_cli

        input_data = json.dumps([{"cve_id": "CVE-2024-3094"}])
        monkeypatch.setattr("sys.stdin", StringIO(input_data))
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        rc = run_export_sarif_cli(["--tool-version", "3.0.0"])
        assert rc == 0
        out = capsys.readouterr().out
        sarif = json.loads(out)
        assert sarif["runs"][0]["tool"]["driver"]["version"] == "3.0.0"

    def test_cli_invalid_json_returns_error(self, monkeypatch, capsys):
        from manus_agent.tools.export_sarif import run_export_sarif_cli

        monkeypatch.setattr("sys.stdin", StringIO("not json"))
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        rc = run_export_sarif_cli([])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid JSON" in err

    def test_cli_invalid_finding_returns_error(self, monkeypatch, capsys):
        from manus_agent.tools.export_sarif import run_export_sarif_cli

        input_data = json.dumps([{"no_cve_id": True}])
        monkeypatch.setattr("sys.stdin", StringIO(input_data))
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        rc = run_export_sarif_cli([])
        assert rc == 1
        err = capsys.readouterr().err
        assert "invalid" in err.lower()

    def test_cli_accepts_object_with_findings_key(self, monkeypatch, capsys):
        from manus_agent.tools.export_sarif import run_export_sarif_cli

        input_data = json.dumps({"findings": [{"cve_id": "CVE-2024-3094"}]})
        monkeypatch.setattr("sys.stdin", StringIO(input_data))
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        rc = run_export_sarif_cli([])
        assert rc == 0
        out = capsys.readouterr().out
        sarif = json.loads(out)
        assert len(sarif["runs"][0]["results"]) == 1

    def test_cli_accepts_object_with_results_key(self, monkeypatch, capsys):
        from manus_agent.tools.export_sarif import run_export_sarif_cli

        input_data = json.dumps({"results": [{"cve_id": "CVE-2024-3094"}]})
        monkeypatch.setattr("sys.stdin", StringIO(input_data))
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)

        rc = run_export_sarif_cli([])
        assert rc == 0
        out = capsys.readouterr().out
        sarif = json.loads(out)
        assert len(sarif["runs"][0]["results"]) == 1

    def test_cli_no_input_tty_returns_error(self, monkeypatch, capsys):
        from manus_agent.tools.export_sarif import run_export_sarif_cli

        monkeypatch.setattr("sys.stdin", StringIO(""))
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)

        rc = run_export_sarif_cli([])
        assert rc == 1
        err = capsys.readouterr().err
        assert "No input" in err

    def test_cli_nonexistent_input_file_returns_error(self, capsys):
        from manus_agent.tools.export_sarif import run_export_sarif_cli

        rc = run_export_sarif_cli(["--input", "/nonexistent/path/file.json"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Error reading input" in err


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_full_finding_all_fields(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_full_finding()])
        run = sarif["runs"][0]
        rules = run["tool"]["driver"]["rules"]
        results = run["results"]
        assert len(rules) == 1
        assert len(results) == 1
        # Rule has all expected properties
        assert rules[0]["properties"]["cvss-score"] == 10.0
        assert rules[0]["properties"]["epss-score"] == 0.97
        assert "CWE-506" in rules[0]["properties"]["tags"]
        assert "cisa-kev" in rules[0]["properties"]["tags"]
        # Result has location with region
        loc = results[0]["locations"][0]["physicalLocation"]
        assert loc["artifactLocation"]["uri"] == "requirements.txt"
        assert loc["region"]["startLine"] == 12

    def test_many_findings(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        findings = [_minimal_finding(f"CVE-2024-{i:04d}") for i in range(100)]
        sarif = findings_to_sarif(findings)
        assert len(sarif["runs"][0]["results"]) == 100
        assert len(sarif["runs"][0]["tool"]["driver"]["rules"]) == 100

    def test_cve_id_normalised_to_uppercase(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding("cve-2024-3094")])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert rules[0]["id"] == "CVE-2024-3094"

    def test_severity_case_insensitive(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(severity="critical")])
        results = sarif["runs"][0]["results"]
        assert results[0]["level"] == "error"

    def test_none_severity_defaults_medium(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()])  # no severity
        results = sarif["runs"][0]["results"]
        assert results[0]["level"] == "warning"

    def test_cwe_id_without_prefix_normalised(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(cwe_id="79")])
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        assert "CWE-79" in rules[0]["properties"]["tags"]

    def test_no_affected_component_no_logical_location(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding()])
        results = sarif["runs"][0]["results"]
        # Minimal finding has no component and no file_path, so no locations
        assert "locations" not in results[0]

    def test_description_in_message(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([_minimal_finding(description="A critical RCE vulnerability")])
        results = sarif["runs"][0]["results"]
        assert "A critical RCE vulnerability" in results[0]["message"]["text"]


# ---------------------------------------------------------------------------
# findings_to_sarif direct unit tests
# ---------------------------------------------------------------------------


class TestFindingsToSarif:
    def test_empty_list_returns_valid_sarif(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        sarif = findings_to_sarif([])
        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"][0]["results"]) == 0
        assert len(sarif["runs"][0]["tool"]["driver"]["rules"]) == 0

    def test_none_findings_handled(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        # The function accepts empty list; None should be handled gracefully
        sarif = findings_to_sarif(None)  # type: ignore[arg-type]
        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"][0]["results"]) == 0

    def test_multiple_findings_produce_indexed_rules(self):
        from manus_agent.tools.export_sarif import findings_to_sarif

        findings = [
            _minimal_finding("CVE-2024-3094", severity="CRITICAL"),
            _minimal_finding("CVE-2021-44228", severity="HIGH"),
            _minimal_finding("CVE-2023-44487", severity="MEDIUM"),
        ]
        sarif = findings_to_sarif(findings)
        results = sarif["runs"][0]["results"]
        assert results[0]["ruleIndex"] == 0
        assert results[1]["ruleIndex"] == 1
        assert results[2]["ruleIndex"] == 2
