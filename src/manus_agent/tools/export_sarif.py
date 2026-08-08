"""
Tool: export_sarif

Export vulnerability analysis results as SARIF (Static Analysis Results
Interchange Format) v2.1.0 — the industry-standard JSON format for security
tool output.

SARIF output integrates directly with:
  - GitHub Code Scanning (upload via ``gh api``)
  - VS Code SARIF Viewer extension
  - Azure DevOps Advanced Security
  - Any SARIF-compatible security dashboard

The tool accepts one or more CVE findings (as produced by the analysis pipeline)
and renders a spec-compliant SARIF log with rule definitions, result locations,
severity levels, and optional fix information.

CLI: ``manus-agent export-sarif --input results.json --output report.sarif``
     ``manus-agent analyze CVE-2024-3094 --format json | manus-agent export-sarif``
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any

from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

__all__ = ["export_sarif", "TOOL_SPEC", "findings_to_sarif"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SARIF_VERSION = "2.1.0"
_SARIF_SCHEMA = "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"
_TOOL_NAME = "manus-agent"
_TOOL_INFO_URI = "https://github.com/manus-use/manus-agent"

# Map CVSS severity strings to SARIF severity levels
_SEVERITY_MAP: dict[str, str] = {
    "CRITICAL": "error",
    "HIGH": "error",
    "MEDIUM": "warning",
    "LOW": "note",
    "NONE": "none",
}

# Map CVSS severity to SARIF security-severity scores (for GitHub Code Scanning)
_SECURITY_SEVERITY_MAP: dict[str, str] = {
    "CRITICAL": "9.0",
    "HIGH": "7.0",
    "MEDIUM": "4.0",
    "LOW": "1.0",
    "NONE": "0.0",
}

TOOL_SPEC = {
    "name": "export_sarif",
    "description": (
        "Exports vulnerability findings as a SARIF v2.1.0 JSON document compatible with "
        "GitHub Code Scanning, VS Code SARIF Viewer, and other security dashboards. "
        "Accepts a list of CVE findings (each with cve_id, severity, description, and "
        "optional affected_component/fix information) and produces a standards-compliant "
        "SARIF log. Use this after running analysis tools to produce machine-readable output "
        "that can be uploaded to GitHub or integrated into CI/CD pipelines."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "description": (
                        "List of vulnerability findings to export. Each finding should have: "
                        "cve_id (required), severity (CRITICAL/HIGH/MEDIUM/LOW), "
                        "description, affected_component, affected_versions, "
                        "fix_version, cvss_score, epss_score, cwe_id, references (list of URLs)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "cve_id": {
                                "type": "string",
                                "description": "CVE identifier (e.g. CVE-2024-3094).",
                            },
                            "severity": {
                                "type": "string",
                                "description": "Severity level: CRITICAL, HIGH, MEDIUM, LOW, or NONE.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Human-readable description of the vulnerability.",
                            },
                            "affected_component": {
                                "type": "string",
                                "description": "Affected package/component name (e.g. 'xz-utils').",
                            },
                            "affected_versions": {
                                "type": "string",
                                "description": "Affected version range (e.g. '>=5.6.0, <5.6.2').",
                            },
                            "fix_version": {
                                "type": "string",
                                "description": "Version that fixes the vulnerability.",
                            },
                            "cvss_score": {
                                "type": "number",
                                "description": "CVSS v3.x base score (0.0-10.0).",
                            },
                            "epss_score": {
                                "type": "number",
                                "description": "EPSS exploitation probability (0.0-1.0).",
                            },
                            "cwe_id": {
                                "type": "string",
                                "description": "CWE weakness identifier (e.g. CWE-506).",
                            },
                            "references": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of reference URLs.",
                            },
                            "in_kev": {
                                "type": "boolean",
                                "description": "Whether the CVE is in CISA KEV catalog.",
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Optional file path where the vulnerability was found (e.g. from SBOM or lockfile).",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "Optional start line in file_path.",
                            },
                            "end_line": {
                                "type": "integer",
                                "description": "Optional end line in file_path.",
                            },
                        },
                        "required": ["cve_id"],
                    },
                },
                "tool_version": {
                    "type": "string",
                    "description": "Version of manus-agent that produced the findings.",
                },
            },
            "required": ["findings"],
        }
    },
}


# ---------------------------------------------------------------------------
# Core conversion logic
# ---------------------------------------------------------------------------


def _normalise_severity(severity: str | None) -> str:
    """Normalise a severity string to uppercase; default to MEDIUM if unknown."""
    if not severity:
        return "MEDIUM"
    s = severity.strip().upper()
    if s in _SEVERITY_MAP:
        return s
    return "MEDIUM"


def _make_rule(finding: dict[str, Any]) -> dict[str, Any]:
    """Build a SARIF reportingDescriptor (rule) from a finding."""
    cve_id = finding["cve_id"].upper()
    severity = _normalise_severity(finding.get("severity"))
    description = finding.get("description", f"Vulnerability {cve_id}")
    cwe_id = finding.get("cwe_id", "")

    rule: dict[str, Any] = {
        "id": cve_id,
        "name": cve_id.replace("-", "_"),
        "shortDescription": {"text": f"{cve_id} ({severity})"},
        "fullDescription": {"text": description[:1000] if description else f"Vulnerability {cve_id}"},
        "helpUri": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
        "properties": {
            "security-severity": _SECURITY_SEVERITY_MAP.get(severity, "4.0"),
            "tags": ["security", "vulnerability"],
        },
    }

    # Add CVSS score as property
    cvss_score = finding.get("cvss_score")
    if cvss_score is not None:
        rule["properties"]["security-severity"] = str(float(cvss_score))
        rule["properties"]["cvss-score"] = float(cvss_score)

    # Add EPSS score as property
    epss_score = finding.get("epss_score")
    if epss_score is not None:
        rule["properties"]["epss-score"] = float(epss_score)

    # Add CWE tag
    if cwe_id:
        cwe_tag = cwe_id.upper() if cwe_id.upper().startswith("CWE-") else f"CWE-{cwe_id}"
        rule["properties"]["tags"].append(cwe_tag)

    # Add KEV tag
    if finding.get("in_kev"):
        rule["properties"]["tags"].append("cisa-kev")

    # Help text with remediation info
    help_lines = [f"**{cve_id}** — {severity} severity"]
    if finding.get("affected_component"):
        help_lines.append(f"Affected: {finding['affected_component']}")
    if finding.get("affected_versions"):
        help_lines.append(f"Versions: {finding['affected_versions']}")
    if finding.get("fix_version"):
        help_lines.append(f"Fix: upgrade to {finding['fix_version']}")
    if finding.get("references"):
        help_lines.append("References:")
        for ref in finding["references"][:5]:
            help_lines.append(f"  - {ref}")

    rule["help"] = {"text": "\n".join(help_lines), "markdown": "\n".join(help_lines)}

    return rule


def _make_result(finding: dict[str, Any], rule_index: int) -> dict[str, Any]:
    """Build a SARIF result object from a finding."""
    cve_id = finding["cve_id"].upper()
    severity = _normalise_severity(finding.get("severity"))
    sarif_level = _SEVERITY_MAP.get(severity, "warning")

    # Build message
    msg_parts = [cve_id]
    if finding.get("affected_component"):
        msg_parts.append(f"in {finding['affected_component']}")
    if finding.get("affected_versions"):
        msg_parts.append(f"(versions {finding['affected_versions']})")
    if finding.get("in_kev"):
        msg_parts.append("⚠️  ACTIVELY EXPLOITED (CISA KEV)")

    message_text = " ".join(msg_parts)
    if finding.get("description"):
        message_text += f": {finding['description'][:500]}"

    result: dict[str, Any] = {
        "ruleId": cve_id,
        "ruleIndex": rule_index,
        "level": sarif_level,
        "message": {"text": message_text},
    }

    # Add location if file_path is provided
    file_path = finding.get("file_path")
    if file_path:
        location: dict[str, Any] = {
            "physicalLocation": {
                "artifactLocation": {
                    "uri": file_path,
                    "uriBaseId": "%SRCROOT%",
                },
            }
        }
        start_line = finding.get("start_line")
        end_line = finding.get("end_line")
        if start_line:
            region: dict[str, Any] = {"startLine": int(start_line)}
            if end_line:
                region["endLine"] = int(end_line)
            location["physicalLocation"]["region"] = region
        result["locations"] = [location]
    else:
        # Use logical location (component-based) when no file path
        if finding.get("affected_component"):
            result["locations"] = [
                {
                    "logicalLocations": [
                        {
                            "name": finding["affected_component"],
                            "kind": "module",
                        }
                    ]
                }
            ]

    # Add fix information
    fix_version = finding.get("fix_version")
    if fix_version:
        component = finding.get("affected_component", "the affected component")
        result["fixes"] = [
            {
                "description": {
                    "text": f"Upgrade {component} to version {fix_version}",
                },
            }
        ]

    # Add related locations (references as related)
    references = finding.get("references", [])
    if references:
        result["relatedLocations"] = [
            {
                "id": i,
                "message": {"text": ref},
                "physicalLocation": {
                    "artifactLocation": {"uri": ref},
                },
            }
            for i, ref in enumerate(references[:10])
        ]

    return result


def findings_to_sarif(
    findings: list[dict[str, Any]],
    *,
    tool_version: str | None = None,
) -> dict[str, Any]:
    """Convert a list of vulnerability findings into a SARIF v2.1.0 log.

    Parameters
    ----------
    findings:
        List of finding dicts. Each must have at least ``cve_id``.
    tool_version:
        Version string for the manus-agent tool driver entry.

    Returns
    -------
    dict
        A complete SARIF log object ready for JSON serialization.
    """
    if not findings:
        findings = []

    # Deduplicate by CVE ID (keep first occurrence)
    seen_cves: set[str] = set()
    unique_findings: list[dict[str, Any]] = []
    for f in findings:
        cve_id = f.get("cve_id", "").upper()
        if not cve_id:
            continue
        if cve_id not in seen_cves:
            seen_cves.add(cve_id)
            unique_findings.append(f)

    # Build rules and results
    rules: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for i, finding in enumerate(unique_findings):
        rules.append(_make_rule(finding))
        results.append(_make_result(finding, rule_index=i))

    # Assemble the SARIF log
    version_str = tool_version or _get_package_version()

    sarif_log: dict[str, Any] = {
        "$schema": _SARIF_SCHEMA,
        "version": _SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": _TOOL_NAME,
                        "informationUri": _TOOL_INFO_URI,
                        "version": version_str,
                        "rules": rules,
                    }
                },
                "results": results,
                "invocations": [
                    {
                        "executionSuccessful": True,
                        "endTimeUtc": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                ],
            }
        ],
    }

    return sarif_log


def _get_package_version() -> str:
    """Get the manus-agent package version, falling back to 'unknown'."""
    try:
        from importlib.metadata import version

        return version("manus-agent")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------


def export_sarif(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Export vulnerability findings as SARIF v2.1.0 JSON."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]

    findings = tool_input.get("findings", [])
    tool_version = tool_input.get("tool_version")

    if not isinstance(findings, list):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "'findings' must be a list of vulnerability finding objects."}],
        }
        log_tool_output_size("export_sarif", result)
        return result

    # Validate that each finding has at least cve_id
    for i, finding in enumerate(findings):
        if not isinstance(finding, dict):
            result = {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Finding at index {i} must be an object, got {type(finding).__name__}."}],
            }
            log_tool_output_size("export_sarif", result)
            return result
        if not finding.get("cve_id"):
            result = {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": f"Finding at index {i} is missing required field 'cve_id'."}],
            }
            log_tool_output_size("export_sarif", result)
            return result

    try:
        sarif_log = findings_to_sarif(findings, tool_version=tool_version)
    except Exception as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Failed to generate SARIF output: {exc}"}],
        }
        log_tool_output_size("export_sarif", result)
        return result

    n_results = len(sarif_log.get("runs", [{}])[0].get("results", []))
    summary = f"SARIF v{_SARIF_VERSION} log generated: {n_results} finding(s) exported."

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [
            {"text": summary},
            {"json": sarif_log},
        ],
    }
    log_tool_output_size("export_sarif", result)
    return result


# ---------------------------------------------------------------------------
# CLI helper (called from cli.py)
# ---------------------------------------------------------------------------


def run_export_sarif_cli(argv: list[str]) -> int:
    """CLI entry point for ``manus-agent export-sarif``.

    Reads findings from --input (JSON file) or stdin, writes SARIF to
    --output (file) or stdout.

    Returns 0 on success, non-zero on failure.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="manus-agent export-sarif",
        description=(
            "Export vulnerability findings as SARIF v2.1.0 JSON.\n\n"
            "Reads a JSON array of findings from --input or stdin and writes\n"
            "a SARIF-compliant log to --output or stdout."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        metavar="FILE",
        help="Input JSON file containing findings array (default: stdin)",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        help="Output file for SARIF JSON (default: stdout)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print the SARIF output (default: true)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact (single-line) JSON output",
    )
    parser.add_argument(
        "--tool-version",
        metavar="VER",
        help="Override the tool version in SARIF output",
    )

    args = parser.parse_args(argv)

    # Read input
    try:
        if args.input:
            with open(args.input, encoding="utf-8") as fh:
                raw = fh.read()
        else:
            if sys.stdin.isatty():
                print(
                    "Error: No input provided. Pipe JSON findings or use --input FILE.",
                    file=sys.stderr,
                )
                return 1
            raw = sys.stdin.read()
    except OSError as exc:
        print(f"Error reading input: {exc}", file=sys.stderr)
        return 1

    # Parse input JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"Error: Invalid JSON input: {exc}", file=sys.stderr)
        return 1

    # Accept either a bare array or an object with a "findings" key
    if isinstance(data, list):
        findings = data
    elif isinstance(data, dict):
        findings = data.get("findings", data.get("results", []))
        if not isinstance(findings, list):
            print("Error: Could not find a findings array in the input.", file=sys.stderr)
            return 1
    else:
        print("Error: Input must be a JSON array or object with a 'findings' key.", file=sys.stderr)
        return 1

    # Validate findings
    for i, f in enumerate(findings):
        if not isinstance(f, dict) or not f.get("cve_id"):
            print(
                f"Error: Finding at index {i} is invalid (must be an object with 'cve_id').",
                file=sys.stderr,
            )
            return 1

    # Generate SARIF
    try:
        sarif_log = findings_to_sarif(findings, tool_version=args.tool_version)
    except Exception as exc:
        print(f"Error generating SARIF: {exc}", file=sys.stderr)
        return 1

    # Format output
    indent = None if args.compact else 2
    sarif_json = json.dumps(sarif_log, indent=indent, ensure_ascii=False)

    # Write output
    try:
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(sarif_json)
                fh.write("\n")
            n_results = len(sarif_log["runs"][0]["results"])
            print(
                f"✓ SARIF written to {args.output} ({n_results} finding(s))",
                file=sys.stderr,
            )
        else:
            sys.stdout.write(sarif_json)
            sys.stdout.write("\n")
    except OSError as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        return 1

    return 0
