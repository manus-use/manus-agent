"""
Tool: tool_inventory

Discovers, catalogues, and reports on all tools available in the manus-agent
framework.  Provides both a programmatic API and a CLI subcommand for auditing
the tool ecosystem — checking registration status, identifying which agents
wire each tool, and flagging tools that are defined but unreachable.

CLI: ``manus-agent tool-inventory``
     ``manus-agent tool-inventory --format json``
     ``manus-agent tool-inventory --agent vi``
     ``manus-agent tool-inventory --verbose``
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import pkgutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["tool_inventory", "discover_tools", "ToolEntry"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Package root for tool modules
# ---------------------------------------------------------------------------

_TOOLS_PACKAGE = "manus_agent.tools"
_TOOLS_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Agent → tool mapping (statically defined to avoid heavy agent imports)
# ---------------------------------------------------------------------------

# Maps agent short names to the tool module paths they wire up.
# Derived from reading the agent source code; kept static so that
# ``tool_inventory`` never needs to import heavy deps like strands/docker.
_AGENT_TOOL_MAP: dict[str, list[str]] = {
    "vi": [
        "manus_agent.tools.http_request",
        "manus_agent.tools.python_repl",
        "manus_agent.tools.create_lark_document",
        "manus_agent.tools.get_nvd_data",
        "manus_agent.tools.get_trickest_pocs",
        "manus_agent.tools.get_poc_week",
        "manus_agent.tools.search_for_exploits",
        "manus_agent.tools.get_cwe_details",
        "manus_agent.tools.search_exploit_db",
        "manus_agent.tools.search_packetstorm",
        "manus_agent.tools.check_cisa_kev",
        "manus_agent.tools.get_otx_cve_details",
        "manus_agent.tools.query_threat_intelligence_feeds",
        "manus_agent.tools.get_github_advisory",
        "manus_agent.tools.verify_exploit",
        "manus_agent.tools.get_epss_trend",
        "manus_agent.tools.get_osv_data",
        "manus_agent.tools.get_patch_diff",
        "manus_agent.tools.score_exploit_complexity",
        "manus_agent.tools.get_vulncheck_data",
        "manus_agent.tools.search_poc_sources",
        "manus_agent.tools.get_dependency_blast_radius",
    ],
    "manus": [
        "manus_agent.tools.http_request",
        "manus_agent.tools.python_repl",
    ],
    "data_analysis": [
        "manus_agent.tools.python_repl",
    ],
    "discovery": [
        "manus_agent.tools.obtain_cves",
        "manus_agent.tools.submit_cves",
        "manus_agent.tools.get_nvd_data",
        "manus_agent.tools.check_cisa_kev",
        "manus_agent.tools.get_epss_trend",
        "manus_agent.tools.search_for_exploits",
        "manus_agent.tools.get_otx_cve_details",
    ],
    "remediation": [
        "manus_agent.tools.get_nvd_data",
        "manus_agent.tools.get_patch_diff",
        "manus_agent.tools.get_osv_data",
        "manus_agent.tools.get_dependency_blast_radius",
        "manus_agent.tools.search_poc_sources",
    ],
    "variant": [
        "manus_agent.tools.get_nvd_data",
        "manus_agent.tools.get_cwe_details",
        "manus_agent.tools.get_osv_data",
        "manus_agent.tools.search_poc_sources",
    ],
}

# Reverse map: tool module → set of agents that use it
_TOOL_TO_AGENTS: dict[str, set[str]] = {}
for _agent, _tools in _AGENT_TOOL_MAP.items():
    for _tool_mod in _tools:
        _TOOL_TO_AGENTS.setdefault(_tool_mod, set()).add(_agent)

# Tools registered in tools/__init__.py ALL_TOOLS
_REGISTERED_TOOL_NAMES: frozenset[str] = frozenset(
    [
        "file_read",
        "file_write",
        "python_repl",
        "shell",
        "http_request",
        "editor",
        "environment",
        "generate_image",
        "current_time",
        "calculator",
    ]
)

# Modules that are utility/support, not standalone tools
_SKIP_MODULES: frozenset[str] = frozenset(
    [
        "tool_output_logger",
        "browser_utils",
        "browser_tools",
        "file_operations",
        "__init__",
        "tool_inventory",
    ]
)

# All tool modules referenced by at least one agent (flattened from _AGENT_TOOL_MAP)
_ALL_KNOWN_TOOL_MODULES: frozenset[str] = frozenset(
    tool_mod for tools in _AGENT_TOOL_MAP.values() for tool_mod in tools
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ToolEntry:
    """Describes a single discovered tool."""

    name: str
    module_path: str
    file_path: str
    style: str  # "tool_spec" | "strands_decorator" | "both"
    description: str = ""
    registered_in_all_tools: bool = False
    agents: list[str] = field(default_factory=list)
    input_params: list[str] = field(default_factory=list)
    has_tests: bool = False
    callable_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON output."""
        return {
            "name": self.name,
            "module": self.module_path,
            "file": self.file_path,
            "style": self.style,
            "description": self.description,
            "registered": self.registered_in_all_tools,
            "agents": sorted(self.agents),
            "input_params": self.input_params,
            "has_tests": self.has_tests,
            "callable": self.callable_name,
        }


# ---------------------------------------------------------------------------
# Discovery engine
# ---------------------------------------------------------------------------


def _extract_tool_spec_info(mod: Any, module_name: str) -> ToolEntry | None:
    """Extract tool info from a module with a TOOL_SPEC dict."""
    spec = getattr(mod, "TOOL_SPEC", None)
    if spec is None or not isinstance(spec, dict):
        return None

    name = spec.get("name", module_name.split(".")[-1])
    desc = spec.get("description", "")
    if len(desc) > 200:
        desc = desc[:197] + "..."

    # Extract input parameter names from the schema
    params: list[str] = []
    schema = spec.get("inputSchema", {})
    json_schema = schema.get("json", schema)
    props = json_schema.get("properties", {})
    params = list(props.keys())

    # Find the callable function name — look for a function whose name
    # matches the tool spec name or the module's last component
    callable_name = ""
    short_name = module_name.split(".")[-1]
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name, None)
        if callable(obj) and not attr_name.startswith("_"):
            if attr_name == name or attr_name == short_name:
                callable_name = attr_name
                break

    return ToolEntry(
        name=name,
        module_path=module_name,
        file_path="",
        style="tool_spec",
        description=desc,
        input_params=params,
        callable_name=callable_name,
    )


def _extract_strands_tool_info(mod: Any, module_name: str) -> list[ToolEntry]:
    """Extract tool info from @tool-decorated functions.

    Handles two patterns:
    1. Functions with a ``TOOL_SPEC`` class/instance attribute (legacy)
    2. ``DecoratedFunctionTool`` instances created by ``@strands.tool`` —
       these expose ``.tool_spec`` (property) and ``.tool_name``.
    """
    entries: list[ToolEntry] = []
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name, None)
        if obj is None:
            continue

        # Try legacy TOOL_SPEC attribute first
        tool_spec = getattr(obj, "TOOL_SPEC", None)

        # Try strands DecoratedFunctionTool .tool_spec property
        if tool_spec is None:
            tool_spec = getattr(obj, "tool_spec", None)
            # Filter out non-tool objects that happen to have a tool_spec attr
            if tool_spec is not None and not isinstance(tool_spec, dict):
                tool_spec = None
            # Only accept if the object also has tool_name (strong signal)
            if tool_spec is not None and not hasattr(obj, "tool_name"):
                tool_spec = None

        if tool_spec is None or not isinstance(tool_spec, dict):
            continue

        name = tool_spec.get("name", attr_name)
        desc = tool_spec.get("description", "")
        if len(desc) > 200:
            desc = desc[:197] + "..."
        params: list[str] = []
        schema = tool_spec.get("inputSchema", {})
        json_schema = schema.get("json", schema)
        props = json_schema.get("properties", {})
        params = list(props.keys())

        entries.append(
            ToolEntry(
                name=name,
                module_path=module_name,
                file_path="",
                style="strands_decorator",
                description=desc,
                input_params=params,
                callable_name=attr_name,
            )
        )
    return entries


def _check_test_exists(tool_name: str, tests_dir: Path | None = None) -> bool:
    """Check whether a test file exists for the given tool."""
    if tests_dir is None:
        tests_dir = _TOOLS_DIR.parents[2] / "tests"
    if not tests_dir.is_dir():
        return False

    # Common test file patterns
    patterns = [
        f"test_{tool_name}.py",
        f"test_{tool_name}s.py",
        f"tests_{tool_name}.py",
    ]
    for p in patterns:
        for root, _dirs, files in os.walk(tests_dir):
            if p in files:
                return True
    return False


def discover_tools(
    *,
    package_dir: Path | None = None,
    tests_dir: Path | None = None,
) -> list[ToolEntry]:
    """Scan the tools package and return a catalogue of all discovered tools.

    This function performs lightweight module introspection without importing
    heavy dependencies.  It catches ``ImportError`` gracefully so that tools
    with optional deps (docker, browser-use, playwright) are still catalogued.

    Args:
        package_dir: Override the tools package directory (for testing).
        tests_dir: Override the tests directory (for testing).

    Returns:
        Sorted list of :class:`ToolEntry` instances.
    """
    pkg_dir = package_dir or _TOOLS_DIR
    entries: dict[str, ToolEntry] = {}

    # Iterate over Python files in the tools directory
    for py_file in sorted(pkg_dir.glob("*.py")):
        stem = py_file.stem
        if stem in _SKIP_MODULES or stem.startswith("_"):
            continue

        module_name = f"{_TOOLS_PACKAGE}.{stem}"
        file_path = str(py_file.relative_to(pkg_dir.parents[2]))

        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            logger.debug("Could not import %s: %s", module_name, exc)
            # Still record the tool with minimal info
            entry = ToolEntry(
                name=stem,
                module_path=module_name,
                file_path=file_path,
                style="unknown",
                description=f"(import failed: {exc.__class__.__name__})",
                callable_name="",
            )
            entries[stem] = entry
            continue

        # Try TOOL_SPEC-style extraction
        spec_entry = _extract_tool_spec_info(mod, module_name)
        if spec_entry is not None:
            spec_entry.file_path = file_path
            entries[spec_entry.name] = spec_entry

        # Try @tool decorator extraction
        strands_entries = _extract_strands_tool_info(mod, module_name)
        for se in strands_entries:
            se.file_path = file_path
            if se.name in entries:
                # Merge: tool has both TOOL_SPEC and @tool decorator
                entries[se.name].style = "both"
                if not entries[se.name].callable_name:
                    entries[se.name].callable_name = se.callable_name
                if not entries[se.name].description:
                    entries[se.name].description = se.description
                if not entries[se.name].input_params and se.input_params:
                    entries[se.name].input_params = se.input_params
            else:
                entries[se.name] = se

        # Fallback: detect plain-function tools via __all__ or module
        # docstring when no TOOL_SPEC or @tool decorator was found.
        if spec_entry is None and not strands_entries:
            all_exports = getattr(mod, "__all__", None)
            if all_exports:
                for export_name in all_exports:
                    func = getattr(mod, export_name, None)
                    if func is not None and callable(func):
                        doc = (getattr(func, "__doc__", "") or "").strip()
                        desc = doc.split("\n")[0][:200] if doc else ""
                        entries[export_name] = ToolEntry(
                            name=export_name,
                            module_path=module_name,
                            file_path=file_path,
                            style="plain_function",
                            description=desc,
                            callable_name=export_name,
                        )
            elif module_name in _ALL_KNOWN_TOOL_MODULES:
                # Known tool module without TOOL_SPEC, @tool, or __all__
                func = getattr(mod, stem, None)
                if func is not None and callable(func):
                    doc = (getattr(func, "__doc__", "") or "").strip()
                    desc = doc.split("\n")[0][:200] if doc else ""
                    entries[stem] = ToolEntry(
                        name=stem,
                        module_path=module_name,
                        file_path=file_path,
                        style="plain_function",
                        description=desc,
                        callable_name=stem,
                    )

    # Enrich with agent mapping, registration status, and test coverage
    for entry in entries.values():
        entry.registered_in_all_tools = entry.name in _REGISTERED_TOOL_NAMES
        agent_set = _TOOL_TO_AGENTS.get(entry.module_path, set())
        entry.agents = sorted(agent_set)
        entry.has_tests = _check_test_exists(entry.name, tests_dir)

    return sorted(entries.values(), key=lambda e: e.name)


# ---------------------------------------------------------------------------
# Reporting / formatting
# ---------------------------------------------------------------------------


def _format_table(entries: list[ToolEntry], *, verbose: bool = False) -> str:
    """Render a human-readable text table."""
    lines: list[str] = []
    lines.append(f"manus-agent tool inventory: {len(entries)} tools discovered")
    lines.append("=" * 72)

    # Summary counts
    registered = sum(1 for e in entries if e.registered_in_all_tools)
    with_tests = sum(1 for e in entries if e.has_tests)
    orphaned = sum(1 for e in entries if not e.agents)
    lines.append(f"  Registered in ALL_TOOLS: {registered}/{len(entries)}")
    lines.append(f"  With test coverage:      {with_tests}/{len(entries)}")
    lines.append(f"  Orphaned (no agent):     {orphaned}/{len(entries)}")
    lines.append("")

    if verbose:
        for entry in entries:
            status_flags = []
            if entry.registered_in_all_tools:
                status_flags.append("REG")
            if entry.has_tests:
                status_flags.append("TEST")
            if not entry.agents:
                status_flags.append("ORPHAN")
            flags = " ".join(f"[{f}]" for f in status_flags)

            lines.append(f"  {entry.name}")
            lines.append(f"    Module:   {entry.module_path}")
            lines.append(f"    File:     {entry.file_path}")
            lines.append(f"    Style:    {entry.style}")
            lines.append(f"    Callable: {entry.callable_name or '(none)'}")
            lines.append(f"    Agents:   {', '.join(entry.agents) or '(none)'}")
            lines.append(f"    Params:   {', '.join(entry.input_params) or '(none)'}")
            lines.append(f"    Status:   {flags}")
            if entry.description:
                desc = entry.description
                if len(desc) > 100:
                    desc = desc[:97] + "..."
                lines.append(f"    Desc:     {desc}")
            lines.append("")
    else:
        # Compact table
        if not entries:
            lines.append("  (no tools found)")
            return "\n".join(lines)

        name_w = max(len(e.name) for e in entries) + 2
        header = f"  {'Name':<{name_w}} {'Style':<18} {'Agents':<25} {'Flags'}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for entry in entries:
            flags = []
            if entry.registered_in_all_tools:
                flags.append("REG")
            if entry.has_tests:
                flags.append("TEST")
            if not entry.agents:
                flags.append("ORPHAN")
            agents_str = ",".join(entry.agents) if entry.agents else "-"
            flags_str = " ".join(f"[{f}]" for f in flags) if flags else ""
            lines.append(f"  {entry.name:<{name_w}} {entry.style:<18} {agents_str:<25} {flags_str}")

    return "\n".join(lines)


def _format_json(entries: list[ToolEntry]) -> str:
    """Render JSON output."""
    data = {
        "total": len(entries),
        "registered_count": sum(1 for e in entries if e.registered_in_all_tools),
        "tested_count": sum(1 for e in entries if e.has_tests),
        "orphaned_count": sum(1 for e in entries if not e.agents),
        "tools": [e.to_dict() for e in entries],
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def tool_inventory(
    *,
    output_format: str = "table",
    agent_filter: str | None = None,
    verbose: bool = False,
    package_dir: Path | None = None,
    tests_dir: Path | None = None,
) -> str:
    """Run the tool inventory and return formatted output.

    Args:
        output_format: ``"table"`` (default) or ``"json"``
        agent_filter: If set, only show tools used by this agent
        verbose: Show detailed per-tool information (table format only)
        package_dir: Override tool package directory (for testing)
        tests_dir: Override tests directory (for testing)

    Returns:
        Formatted inventory string.
    """
    entries = discover_tools(package_dir=package_dir, tests_dir=tests_dir)

    if agent_filter:
        agent_key = agent_filter.lower()
        if agent_key not in _AGENT_TOOL_MAP:
            available = ", ".join(sorted(_AGENT_TOOL_MAP.keys()))
            return f"Error: unknown agent '{agent_filter}'. Available: {available}"
        entries = [e for e in entries if agent_key in e.agents]

    if output_format == "json":
        return _format_json(entries)
    return _format_table(entries, verbose=verbose)
