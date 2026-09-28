"""Comprehensive tests for manus_agent.tools.tool_inventory."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import pytest

from manus_agent.tools.tool_inventory import (
    ToolEntry,
    _AGENT_TOOL_MAP,
    _ALL_KNOWN_TOOL_MODULES,
    _REGISTERED_TOOL_NAMES,
    _SKIP_MODULES,
    _TOOL_TO_AGENTS,
    _check_test_exists,
    _extract_strands_tool_info,
    _extract_tool_spec_info,
    _format_json,
    _format_table,
    discover_tools,
    tool_inventory,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def sample_entry() -> ToolEntry:
    """A representative ToolEntry for testing."""
    return ToolEntry(
        name="check_cisa_kev",
        module_path="manus_agent.tools.check_cisa_kev",
        file_path="src/manus_agent/tools/check_cisa_kev.py",
        style="tool_spec",
        description="Checks if a CVE is in the CISA KEV catalog.",
        registered_in_all_tools=False,
        agents=["discovery", "vi"],
        input_params=["cve_id"],
        has_tests=True,
        callable_name="check_cisa_kev",
    )


@pytest.fixture
def empty_entry() -> ToolEntry:
    """A minimal ToolEntry with defaults."""
    return ToolEntry(
        name="empty_tool",
        module_path="manus_agent.tools.empty_tool",
        file_path="src/manus_agent/tools/empty_tool.py",
        style="unknown",
    )


@pytest.fixture
def mock_tool_spec_module() -> ModuleType:
    """A mock module with a TOOL_SPEC dict."""
    mod = ModuleType("manus_agent.tools.mock_spec")
    mod.TOOL_SPEC = {
        "name": "mock_spec_tool",
        "description": "A mock tool for testing TOOL_SPEC extraction.",
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "cve_id": {"type": "string", "description": "CVE identifier"},
                    "verbose": {"type": "boolean", "description": "Verbose output"},
                },
                "required": ["cve_id"],
            }
        },
    }

    def mock_spec_tool(tool, **kwargs):
        pass

    mod.mock_spec_tool = mock_spec_tool
    return mod


@pytest.fixture
def mock_strands_decorated_module() -> ModuleType:
    """A mock module with a strands-style DecoratedFunctionTool."""
    mod = ModuleType("manus_agent.tools.mock_strands")

    class FakeDecoratedTool:
        """Mimics strands.tools.decorator.DecoratedFunctionTool."""

        tool_name = "get_fancy_data"
        tool_spec = {
            "name": "get_fancy_data",
            "description": "Fetches fancy data from somewhere.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                }
            },
        }

    mod.get_fancy_data = FakeDecoratedTool()
    return mod


@pytest.fixture
def mock_plain_function_module() -> ModuleType:
    """A mock module with a plain function listed in __all__."""
    mod = ModuleType("manus_agent.tools.mock_plain")
    mod.__all__ = ["my_plain_tool"]

    def my_plain_tool(cve_id: str) -> dict:
        """Fetch data for a CVE from a plain source."""
        return {}

    mod.my_plain_tool = my_plain_tool
    return mod


@pytest.fixture
def tools_tmpdir(tmp_path: Path) -> Path:
    """Create a temporary tools directory with mock .py files."""
    tools_dir = tmp_path / "src" / "manus_agent" / "tools"
    tools_dir.mkdir(parents=True)

    # A TOOL_SPEC module
    (tools_dir / "mock_spec.py").write_text(
        textwrap.dedent("""\
        TOOL_SPEC = {
            "name": "mock_spec",
            "description": "Mock tool with spec.",
            "inputSchema": {"json": {"type": "object", "properties": {"x": {"type": "string"}}}}
        }
        def mock_spec(tool, **kwargs):
            pass
        """)
    )

    # A plain function module with __all__
    (tools_dir / "mock_plain.py").write_text(
        textwrap.dedent("""\
        __all__ = ["mock_plain"]
        def mock_plain(arg):
            \"\"\"A plain tool function.\"\"\"
            pass
        """)
    )

    # A utility module that should be skipped
    (tools_dir / "tool_output_logger.py").write_text("# utility\n")

    # __init__.py that should be skipped
    (tools_dir / "__init__.py").write_text("")

    return tools_dir


@pytest.fixture
def tests_tmpdir(tmp_path: Path) -> Path:
    """Create a temporary tests directory."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_mock_spec.py").write_text("# tests\n")
    (tests_dir / "test_compare_cves.py").write_text("# tests\n")
    return tests_dir


# =========================================================================
# ToolEntry tests
# =========================================================================


class TestToolEntry:
    """Tests for the ToolEntry dataclass."""

    def test_construction_with_all_fields(self, sample_entry: ToolEntry):
        assert sample_entry.name == "check_cisa_kev"
        assert sample_entry.module_path == "manus_agent.tools.check_cisa_kev"
        assert sample_entry.style == "tool_spec"
        assert sample_entry.registered_in_all_tools is False
        assert sample_entry.agents == ["discovery", "vi"]
        assert sample_entry.input_params == ["cve_id"]
        assert sample_entry.has_tests is True
        assert sample_entry.callable_name == "check_cisa_kev"

    def test_construction_with_defaults(self, empty_entry: ToolEntry):
        assert empty_entry.description == ""
        assert empty_entry.registered_in_all_tools is False
        assert empty_entry.agents == []
        assert empty_entry.input_params == []
        assert empty_entry.has_tests is False
        assert empty_entry.callable_name == ""

    def test_to_dict_all_fields(self, sample_entry: ToolEntry):
        d = sample_entry.to_dict()
        assert d["name"] == "check_cisa_kev"
        assert d["module"] == "manus_agent.tools.check_cisa_kev"
        assert d["file"] == "src/manus_agent/tools/check_cisa_kev.py"
        assert d["style"] == "tool_spec"
        assert d["description"] == "Checks if a CVE is in the CISA KEV catalog."
        assert d["registered"] is False
        assert d["agents"] == ["discovery", "vi"]
        assert d["input_params"] == ["cve_id"]
        assert d["has_tests"] is True
        assert d["callable"] == "check_cisa_kev"

    def test_to_dict_defaults(self, empty_entry: ToolEntry):
        d = empty_entry.to_dict()
        assert d["description"] == ""
        assert d["registered"] is False
        assert d["agents"] == []
        assert d["input_params"] == []
        assert d["has_tests"] is False
        assert d["callable"] == ""

    def test_to_dict_is_json_serialisable(self, sample_entry: ToolEntry):
        d = sample_entry.to_dict()
        serialised = json.dumps(d)
        assert isinstance(serialised, str)
        assert json.loads(serialised) == d

    def test_agents_sorted_in_to_dict(self):
        entry = ToolEntry(
            name="t",
            module_path="m",
            file_path="f",
            style="s",
            agents=["vi", "discovery", "manus"],
        )
        d = entry.to_dict()
        assert d["agents"] == ["discovery", "manus", "vi"]


# =========================================================================
# _extract_tool_spec_info tests
# =========================================================================


class TestExtractToolSpecInfo:
    """Tests for _extract_tool_spec_info."""

    def test_extracts_name_from_spec(self, mock_tool_spec_module: ModuleType):
        entry = _extract_tool_spec_info(mock_tool_spec_module, "manus_agent.tools.mock_spec")
        assert entry is not None
        assert entry.name == "mock_spec_tool"

    def test_extracts_description(self, mock_tool_spec_module: ModuleType):
        entry = _extract_tool_spec_info(mock_tool_spec_module, "manus_agent.tools.mock_spec")
        assert entry is not None
        assert "mock tool" in entry.description.lower()

    def test_extracts_input_params(self, mock_tool_spec_module: ModuleType):
        entry = _extract_tool_spec_info(mock_tool_spec_module, "manus_agent.tools.mock_spec")
        assert entry is not None
        assert "cve_id" in entry.input_params
        assert "verbose" in entry.input_params

    def test_finds_callable_matching_spec_name(self, mock_tool_spec_module: ModuleType):
        entry = _extract_tool_spec_info(mock_tool_spec_module, "manus_agent.tools.mock_spec")
        assert entry is not None
        assert entry.callable_name == "mock_spec_tool"

    def test_returns_none_when_no_tool_spec(self):
        mod = ModuleType("manus_agent.tools.empty")
        result = _extract_tool_spec_info(mod, "manus_agent.tools.empty")
        assert result is None

    def test_returns_none_when_tool_spec_not_dict(self):
        mod = ModuleType("manus_agent.tools.bad")
        mod.TOOL_SPEC = "not a dict"
        result = _extract_tool_spec_info(mod, "manus_agent.tools.bad")
        assert result is None

    def test_truncates_long_description(self):
        mod = ModuleType("manus_agent.tools.longdesc")
        mod.TOOL_SPEC = {
            "name": "long_desc_tool",
            "description": "A" * 300,
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
        entry = _extract_tool_spec_info(mod, "manus_agent.tools.longdesc")
        assert entry is not None
        assert len(entry.description) <= 200
        assert entry.description.endswith("...")

    def test_style_is_tool_spec(self, mock_tool_spec_module: ModuleType):
        entry = _extract_tool_spec_info(mock_tool_spec_module, "manus_agent.tools.mock_spec")
        assert entry is not None
        assert entry.style == "tool_spec"

    def test_fallback_name_from_module(self):
        mod = ModuleType("manus_agent.tools.fallback_name")
        mod.TOOL_SPEC = {
            "description": "No name key.",
            "inputSchema": {"json": {"type": "object", "properties": {}}},
        }
        entry = _extract_tool_spec_info(mod, "manus_agent.tools.fallback_name")
        assert entry is not None
        assert entry.name == "fallback_name"

    def test_empty_input_schema(self):
        mod = ModuleType("manus_agent.tools.no_schema")
        mod.TOOL_SPEC = {"name": "no_schema", "description": "Minimal."}
        entry = _extract_tool_spec_info(mod, "manus_agent.tools.no_schema")
        assert entry is not None
        assert entry.input_params == []


# =========================================================================
# _extract_strands_tool_info tests
# =========================================================================


class TestExtractStrandsToolInfo:
    """Tests for _extract_strands_tool_info."""

    def test_extracts_decorated_function_tool(self, mock_strands_decorated_module: ModuleType):
        entries = _extract_strands_tool_info(
            mock_strands_decorated_module, "manus_agent.tools.mock_strands"
        )
        assert len(entries) >= 1
        entry = entries[0]
        assert entry.name == "get_fancy_data"
        assert entry.style == "strands_decorator"

    def test_extracts_params_from_decorated_tool(self, mock_strands_decorated_module: ModuleType):
        entries = _extract_strands_tool_info(
            mock_strands_decorated_module, "manus_agent.tools.mock_strands"
        )
        assert len(entries) >= 1
        assert "query" in entries[0].input_params
        assert "limit" in entries[0].input_params

    def test_extracts_callable_name(self, mock_strands_decorated_module: ModuleType):
        entries = _extract_strands_tool_info(
            mock_strands_decorated_module, "manus_agent.tools.mock_strands"
        )
        assert len(entries) >= 1
        assert entries[0].callable_name == "get_fancy_data"

    def test_returns_empty_for_module_without_tools(self):
        mod = ModuleType("manus_agent.tools.noop")
        entries = _extract_strands_tool_info(mod, "manus_agent.tools.noop")
        assert entries == []

    def test_ignores_non_tool_spec_attributes(self):
        mod = ModuleType("manus_agent.tools.mixed")
        mod.some_dict = {"name": "not_a_tool"}  # has tool_spec-like shape but no tool_name
        entries = _extract_strands_tool_info(mod, "manus_agent.tools.mixed")
        assert entries == []

    def test_ignores_object_with_tool_spec_but_no_tool_name(self):
        mod = ModuleType("manus_agent.tools.partial")

        class FakeObj:
            tool_spec = {"name": "partial", "description": "...", "inputSchema": {"json": {"type": "object", "properties": {}}}}

        mod.partial_obj = FakeObj()
        entries = _extract_strands_tool_info(mod, "manus_agent.tools.partial")
        assert entries == []

    def test_truncates_long_description(self):
        mod = ModuleType("manus_agent.tools.longdec")

        class FakeTool:
            tool_name = "long_desc"
            tool_spec = {
                "name": "long_desc",
                "description": "B" * 300,
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

        mod.long_desc = FakeTool()
        entries = _extract_strands_tool_info(mod, "manus_agent.tools.longdec")
        assert len(entries) == 1
        assert len(entries[0].description) <= 200
        assert entries[0].description.endswith("...")

    def test_multiple_tools_in_one_module(self):
        mod = ModuleType("manus_agent.tools.multi")

        class Tool1:
            tool_name = "tool_one"
            tool_spec = {"name": "tool_one", "description": "First.", "inputSchema": {"json": {"type": "object", "properties": {}}}}

        class Tool2:
            tool_name = "tool_two"
            tool_spec = {"name": "tool_two", "description": "Second.", "inputSchema": {"json": {"type": "object", "properties": {}}}}

        mod.tool_one = Tool1()
        mod.tool_two = Tool2()
        entries = _extract_strands_tool_info(mod, "manus_agent.tools.multi")
        names = {e.name for e in entries}
        assert "tool_one" in names
        assert "tool_two" in names


# =========================================================================
# _check_test_exists tests
# =========================================================================


class TestCheckTestExists:
    """Tests for _check_test_exists."""

    def test_finds_existing_test_file(self, tests_tmpdir: Path):
        assert _check_test_exists("mock_spec", tests_tmpdir) is True

    def test_returns_false_when_no_test(self, tests_tmpdir: Path):
        assert _check_test_exists("nonexistent_tool", tests_tmpdir) is False

    def test_returns_false_when_tests_dir_missing(self, tmp_path: Path):
        assert _check_test_exists("anything", tmp_path / "no_such_dir") is False

    def test_finds_test_in_subdirectory(self, tmp_path: Path):
        subdir = tmp_path / "tests" / "tools"
        subdir.mkdir(parents=True)
        (subdir / "test_deep_tool.py").write_text("# deep\n")
        assert _check_test_exists("deep_tool", tmp_path / "tests") is True

    def test_finds_plural_test_file(self, tmp_path: Path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_some_tools.py").write_text("# plural\n")
        assert _check_test_exists("some_tool", tests_dir) is True


# =========================================================================
# discover_tools tests (with mocked filesystem)
# =========================================================================


class TestDiscoverTools:
    """Tests for discover_tools."""

    def test_discovers_tool_spec_module(self, tools_tmpdir: Path, tests_tmpdir: Path):
        with mock.patch("manus_agent.tools.tool_inventory.importlib.import_module") as mock_import:
            mod = ModuleType("manus_agent.tools.mock_spec")
            mod.TOOL_SPEC = {
                "name": "mock_spec",
                "description": "A mock.",
                "inputSchema": {"json": {"type": "object", "properties": {"x": {"type": "string"}}}},
            }

            def mock_spec(tool, **kwargs):
                pass

            mod.mock_spec = mock_spec

            # mock_plain should also be imported
            mod_plain = ModuleType("manus_agent.tools.mock_plain")
            mod_plain.__all__ = ["mock_plain"]

            def mock_plain_func(arg):
                """A plain tool."""
                pass

            mod_plain.mock_plain = mock_plain_func

            def side_effect(name):
                if name == "manus_agent.tools.mock_spec":
                    return mod
                if name == "manus_agent.tools.mock_plain":
                    return mod_plain
                raise ImportError(f"No module {name}")

            mock_import.side_effect = side_effect

            entries = discover_tools(package_dir=tools_tmpdir, tests_dir=tests_tmpdir)
            names = {e.name for e in entries}
            assert "mock_spec" in names

    def test_handles_import_failure(self, tools_tmpdir: Path, tests_tmpdir: Path):
        with mock.patch("manus_agent.tools.tool_inventory.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("boom")

            entries = discover_tools(package_dir=tools_tmpdir, tests_dir=tests_tmpdir)
            # Should still have entries for files that failed to import
            assert len(entries) > 0
            failed = [e for e in entries if "import failed" in e.description]
            assert len(failed) > 0

    def test_skips_skip_modules(self, tools_tmpdir: Path, tests_tmpdir: Path):
        with mock.patch("manus_agent.tools.tool_inventory.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("should not be called for skip modules")

            entries = discover_tools(package_dir=tools_tmpdir, tests_dir=tests_tmpdir)
            names = {e.name for e in entries}
            assert "tool_output_logger" not in names
            assert "__init__" not in names

    def test_enriches_with_agent_mapping(self, tools_tmpdir: Path, tests_tmpdir: Path):
        with mock.patch("manus_agent.tools.tool_inventory.importlib.import_module") as mock_import:
            mod = ModuleType("manus_agent.tools.mock_spec")
            mod.TOOL_SPEC = {
                "name": "mock_spec",
                "description": "test",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            def side_effect(name):
                if name == "manus_agent.tools.mock_spec":
                    return mod
                raise ImportError()

            mock_import.side_effect = side_effect

            # Agent mapping is static so mock_spec won't be in any agent's list
            entries = discover_tools(package_dir=tools_tmpdir, tests_dir=tests_tmpdir)
            spec_entries = [e for e in entries if e.name == "mock_spec"]
            assert len(spec_entries) == 1
            assert spec_entries[0].agents == []

    def test_enriches_with_test_coverage(self, tools_tmpdir: Path, tests_tmpdir: Path):
        with mock.patch("manus_agent.tools.tool_inventory.importlib.import_module") as mock_import:
            mod = ModuleType("manus_agent.tools.mock_spec")
            mod.TOOL_SPEC = {
                "name": "mock_spec",
                "description": "test",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }

            def side_effect(name):
                if name == "manus_agent.tools.mock_spec":
                    return mod
                raise ImportError()

            mock_import.side_effect = side_effect

            entries = discover_tools(package_dir=tools_tmpdir, tests_dir=tests_tmpdir)
            spec_entries = [e for e in entries if e.name == "mock_spec"]
            assert len(spec_entries) == 1
            assert spec_entries[0].has_tests is True  # test_mock_spec.py exists

    def test_results_sorted_by_name(self, tools_tmpdir: Path, tests_tmpdir: Path):
        with mock.patch("manus_agent.tools.tool_inventory.importlib.import_module") as mock_import:
            mod_spec = ModuleType("manus_agent.tools.mock_spec")
            mod_spec.TOOL_SPEC = {
                "name": "zzz_tool",
                "description": "",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }
            mod_plain = ModuleType("manus_agent.tools.mock_plain")
            mod_plain.__all__ = ["aaa_tool"]

            def aaa_tool():
                """AAA."""
                pass

            mod_plain.aaa_tool = aaa_tool

            def side_effect(name):
                if "mock_spec" in name:
                    return mod_spec
                if "mock_plain" in name:
                    return mod_plain
                raise ImportError()

            mock_import.side_effect = side_effect

            entries = discover_tools(package_dir=tools_tmpdir, tests_dir=tests_tmpdir)
            names = [e.name for e in entries]
            assert names == sorted(names)

    def test_merges_tool_spec_and_strands_decorator(self, tools_tmpdir: Path, tests_tmpdir: Path):
        with mock.patch("manus_agent.tools.tool_inventory.importlib.import_module") as mock_import:
            mod = ModuleType("manus_agent.tools.mock_spec")
            # Has both TOOL_SPEC and a DecoratedFunctionTool
            mod.TOOL_SPEC = {
                "name": "dual_tool",
                "description": "From TOOL_SPEC.",
                "inputSchema": {"json": {"type": "object", "properties": {"a": {"type": "string"}}}},
            }

            class FakeDec:
                tool_name = "dual_tool"
                tool_spec = {
                    "name": "dual_tool",
                    "description": "From decorator.",
                    "inputSchema": {"json": {"type": "object", "properties": {"b": {"type": "integer"}}}},
                }

            mod.dual_tool = FakeDec()

            def side_effect(name):
                if "mock_spec" in name:
                    return mod
                raise ImportError()

            mock_import.side_effect = side_effect

            entries = discover_tools(package_dir=tools_tmpdir, tests_dir=tests_tmpdir)
            dual = [e for e in entries if e.name == "dual_tool"]
            assert len(dual) == 1
            assert dual[0].style == "both"


# =========================================================================
# _format_table tests
# =========================================================================


class TestFormatTable:
    """Tests for _format_table."""

    def test_header_contains_count(self, sample_entry: ToolEntry):
        output = _format_table([sample_entry])
        assert "1 tools discovered" in output

    def test_compact_mode_has_columns(self, sample_entry: ToolEntry):
        output = _format_table([sample_entry])
        assert "Name" in output
        assert "Style" in output
        assert "Agents" in output
        assert "Flags" in output

    def test_compact_mode_shows_tool_name(self, sample_entry: ToolEntry):
        output = _format_table([sample_entry])
        assert "check_cisa_kev" in output

    def test_verbose_mode_shows_module(self, sample_entry: ToolEntry):
        output = _format_table([sample_entry], verbose=True)
        assert "Module:" in output
        assert "manus_agent.tools.check_cisa_kev" in output

    def test_verbose_mode_shows_params(self, sample_entry: ToolEntry):
        output = _format_table([sample_entry], verbose=True)
        assert "Params:" in output
        assert "cve_id" in output

    def test_verbose_mode_shows_description(self, sample_entry: ToolEntry):
        output = _format_table([sample_entry], verbose=True)
        assert "Desc:" in output

    def test_empty_list(self):
        output = _format_table([])
        assert "0 tools discovered" in output

    def test_shows_reg_flag(self):
        entry = ToolEntry(
            name="python_repl",
            module_path="m",
            file_path="f",
            style="tool_spec",
            registered_in_all_tools=True,
        )
        output = _format_table([entry])
        assert "[REG]" in output

    def test_shows_test_flag(self):
        entry = ToolEntry(
            name="t",
            module_path="m",
            file_path="f",
            style="tool_spec",
            has_tests=True,
        )
        output = _format_table([entry])
        assert "[TEST]" in output

    def test_shows_orphan_flag(self):
        entry = ToolEntry(
            name="t",
            module_path="m",
            file_path="f",
            style="tool_spec",
            agents=[],
        )
        output = _format_table([entry])
        assert "[ORPHAN]" in output

    def test_no_orphan_flag_when_has_agents(self, sample_entry: ToolEntry):
        output = _format_table([sample_entry])
        assert "[ORPHAN]" not in output

    def test_summary_counts(self):
        entries = [
            ToolEntry(name="a", module_path="m", file_path="f", style="s", registered_in_all_tools=True, has_tests=True, agents=["vi"]),
            ToolEntry(name="b", module_path="m", file_path="f", style="s", registered_in_all_tools=False, has_tests=False, agents=[]),
            ToolEntry(name="c", module_path="m", file_path="f", style="s", registered_in_all_tools=True, has_tests=False, agents=["manus"]),
        ]
        output = _format_table(entries)
        assert "Registered in ALL_TOOLS: 2/3" in output
        assert "With test coverage:      1/3" in output
        assert "Orphaned (no agent):     1/3" in output


# =========================================================================
# _format_json tests
# =========================================================================


class TestFormatJson:
    """Tests for _format_json."""

    def test_valid_json(self, sample_entry: ToolEntry):
        output = _format_json([sample_entry])
        data = json.loads(output)
        assert isinstance(data, dict)

    def test_total_count(self, sample_entry: ToolEntry):
        output = _format_json([sample_entry])
        data = json.loads(output)
        assert data["total"] == 1

    def test_registered_count(self):
        entries = [
            ToolEntry(name="a", module_path="m", file_path="f", style="s", registered_in_all_tools=True),
            ToolEntry(name="b", module_path="m", file_path="f", style="s", registered_in_all_tools=False),
        ]
        data = json.loads(_format_json(entries))
        assert data["registered_count"] == 1

    def test_tested_count(self):
        entries = [
            ToolEntry(name="a", module_path="m", file_path="f", style="s", has_tests=True),
            ToolEntry(name="b", module_path="m", file_path="f", style="s", has_tests=True),
            ToolEntry(name="c", module_path="m", file_path="f", style="s", has_tests=False),
        ]
        data = json.loads(_format_json(entries))
        assert data["tested_count"] == 2

    def test_orphaned_count(self):
        entries = [
            ToolEntry(name="a", module_path="m", file_path="f", style="s", agents=["vi"]),
            ToolEntry(name="b", module_path="m", file_path="f", style="s", agents=[]),
        ]
        data = json.loads(_format_json(entries))
        assert data["orphaned_count"] == 1

    def test_tools_list_present(self, sample_entry: ToolEntry):
        data = json.loads(_format_json([sample_entry]))
        assert "tools" in data
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "check_cisa_kev"

    def test_empty_list(self):
        data = json.loads(_format_json([]))
        assert data["total"] == 0
        assert data["tools"] == []


# =========================================================================
# tool_inventory (main entry point) tests
# =========================================================================


class TestToolInventory:
    """Tests for the tool_inventory main function."""

    def test_table_output(self):
        with mock.patch("manus_agent.tools.tool_inventory.discover_tools") as mock_discover:
            mock_discover.return_value = [
                ToolEntry(name="t1", module_path="m", file_path="f", style="tool_spec", agents=["vi"]),
            ]
            output = tool_inventory(output_format="table")
            assert "1 tools discovered" in output

    def test_json_output(self):
        with mock.patch("manus_agent.tools.tool_inventory.discover_tools") as mock_discover:
            mock_discover.return_value = [
                ToolEntry(name="t1", module_path="m", file_path="f", style="tool_spec"),
            ]
            output = tool_inventory(output_format="json")
            data = json.loads(output)
            assert data["total"] == 1

    def test_agent_filter(self):
        with mock.patch("manus_agent.tools.tool_inventory.discover_tools") as mock_discover:
            mock_discover.return_value = [
                ToolEntry(name="t1", module_path="m", file_path="f", style="s", agents=["vi"]),
                ToolEntry(name="t2", module_path="m", file_path="f", style="s", agents=["manus"]),
                ToolEntry(name="t3", module_path="m", file_path="f", style="s", agents=["vi", "manus"]),
            ]
            output = tool_inventory(output_format="table", agent_filter="vi")
            assert "2 tools discovered" in output

    def test_unknown_agent_filter(self):
        with mock.patch("manus_agent.tools.tool_inventory.discover_tools") as mock_discover:
            mock_discover.return_value = []
            output = tool_inventory(agent_filter="nonexistent")
            assert "Error: unknown agent" in output

    def test_verbose_mode(self):
        with mock.patch("manus_agent.tools.tool_inventory.discover_tools") as mock_discover:
            mock_discover.return_value = [
                ToolEntry(name="t1", module_path="mod.t1", file_path="f", style="tool_spec", description="Test."),
            ]
            output = tool_inventory(verbose=True)
            assert "Module:" in output
            assert "mod.t1" in output

    def test_passes_package_dir(self, tmp_path: Path):
        with mock.patch("manus_agent.tools.tool_inventory.discover_tools") as mock_discover:
            mock_discover.return_value = []
            tool_inventory(package_dir=tmp_path)
            mock_discover.assert_called_once_with(package_dir=tmp_path, tests_dir=None)

    def test_passes_tests_dir(self, tmp_path: Path):
        with mock.patch("manus_agent.tools.tool_inventory.discover_tools") as mock_discover:
            mock_discover.return_value = []
            tool_inventory(tests_dir=tmp_path)
            mock_discover.assert_called_once_with(package_dir=None, tests_dir=tmp_path)


# =========================================================================
# Constants / mapping tests
# =========================================================================


class TestConstants:
    """Tests for module-level constants and mappings."""

    def test_agent_tool_map_has_expected_agents(self):
        assert "vi" in _AGENT_TOOL_MAP
        assert "manus" in _AGENT_TOOL_MAP
        assert "discovery" in _AGENT_TOOL_MAP
        assert "remediation" in _AGENT_TOOL_MAP
        assert "variant" in _AGENT_TOOL_MAP
        assert "data_analysis" in _AGENT_TOOL_MAP

    def test_vi_agent_has_most_tools(self):
        # VI agent should have the most tools
        vi_count = len(_AGENT_TOOL_MAP["vi"])
        for agent, tools in _AGENT_TOOL_MAP.items():
            if agent != "vi":
                assert vi_count >= len(tools), f"vi ({vi_count}) should have >= {agent} ({len(tools)})"

    def test_tool_to_agents_reverse_map(self):
        # Every tool in _AGENT_TOOL_MAP should appear in _TOOL_TO_AGENTS
        for agent, tools in _AGENT_TOOL_MAP.items():
            for tool in tools:
                assert tool in _TOOL_TO_AGENTS
                assert agent in _TOOL_TO_AGENTS[tool]

    def test_registered_tool_names_are_strings(self):
        for name in _REGISTERED_TOOL_NAMES:
            assert isinstance(name, str)

    def test_registered_tools_include_basics(self):
        assert "file_read" in _REGISTERED_TOOL_NAMES
        assert "file_write" in _REGISTERED_TOOL_NAMES
        assert "python_repl" in _REGISTERED_TOOL_NAMES
        assert "shell" in _REGISTERED_TOOL_NAMES

    def test_skip_modules_include_utilities(self):
        assert "tool_output_logger" in _SKIP_MODULES
        assert "browser_utils" in _SKIP_MODULES
        assert "__init__" in _SKIP_MODULES

    def test_all_known_tool_modules_populated(self):
        assert len(_ALL_KNOWN_TOOL_MODULES) > 0
        for mod in _ALL_KNOWN_TOOL_MODULES:
            assert mod.startswith("manus_agent.tools.")

    def test_all_agent_tools_in_known_modules(self):
        for agent, tools in _AGENT_TOOL_MAP.items():
            for tool in tools:
                assert tool in _ALL_KNOWN_TOOL_MODULES, f"{tool} from {agent} not in _ALL_KNOWN_TOOL_MODULES"


# =========================================================================
# Integration test (real package scan)
# =========================================================================


class TestIntegration:
    """Integration tests that scan the actual tools package."""

    def test_discover_real_tools(self):
        """Verify discovery works against the real package."""
        entries = discover_tools()
        assert len(entries) > 20  # There should be 25+ tools
        names = {e.name for e in entries}
        # Spot-check some well-known tools
        assert "check_cisa_kev" in names
        assert "get_nvd_data" in names
        assert "python_repl" in names

    def test_real_inventory_table(self):
        """Verify table output is non-empty and well-formed."""
        output = tool_inventory(output_format="table")
        assert "tools discovered" in output
        assert "Registered in ALL_TOOLS" in output

    def test_real_inventory_json(self):
        """Verify JSON output is valid and has expected structure."""
        output = tool_inventory(output_format="json")
        data = json.loads(output)
        assert data["total"] > 20
        assert "tools" in data
        assert all("name" in t for t in data["tools"])

    def test_real_agent_filter_vi(self):
        """Verify VI agent filter returns expected tools."""
        output = tool_inventory(agent_filter="vi")
        assert "tools discovered" in output
        # Parse count from "manus-agent tool inventory: N tools discovered"
        import re

        m = re.search(r"(\d+) tools discovered", output)
        assert m is not None
        count = int(m.group(1))
        # VI agent should have 15+ tools
        assert count >= 15

    def test_real_verbose_output(self):
        """Verify verbose mode includes Module: lines."""
        output = tool_inventory(verbose=True)
        assert "Module:" in output
        assert "Callable:" in output
