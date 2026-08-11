"""Comprehensive test suite for browser_agent_tool module.

Tests the Strands tool wrapper that delegates to the browser agent.
Covers: TOOL_SPEC contract, input validation, asyncio.run delegation,
JSON parsing of results, exception handling, edge cases.

All tests are 100% mocked — no real browser or HTTP calls.

The module imports `run_browser_task` from `manus_agent.agents.browser`,
which requires the optional `browser_use` package. We mock only the
`manus_agent.agents.browser` module in sys.modules to bypass this dependency
without interfering with other packages (e.g. mcp used by strands).
"""

import asyncio
import importlib
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixture: mock only the browser module import (not mcp or other packages)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_browser_module():
    """Mock manus_agent.agents.browser before importing browser_agent_tool.

    Only mocks the single module that browser_agent_tool.py imports from,
    without touching unrelated packages like mcp.
    """
    mock_run_browser_task = AsyncMock(return_value='{"task_completed": true, "summary": "ok", "result": "done"}')

    fake_browser_mod = MagicMock()
    fake_browser_mod.run_browser_task = mock_run_browser_task

    # Save and replace only manus_agent.agents.browser
    saved_browser = sys.modules.get("manus_agent.agents.browser")
    sys.modules["manus_agent.agents.browser"] = fake_browser_mod

    # Remove cached browser_agent_tool to force re-import with mock
    bat_key = "manus_agent.tools.browser_agent_tool"
    saved_bat = sys.modules.pop(bat_key, None)

    yield mock_run_browser_task

    # Restore
    if saved_browser is not None:
        sys.modules["manus_agent.agents.browser"] = saved_browser
    else:
        sys.modules.pop("manus_agent.agents.browser", None)

    if saved_bat is not None:
        sys.modules[bat_key] = saved_bat
    else:
        sys.modules.pop(bat_key, None)


def _import_tool():
    """Import (or re-import) browser_agent_tool after mocks are in place."""
    import manus_agent.tools.browser_agent_tool as mod

    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# Module import tests
# ---------------------------------------------------------------------------


class TestModuleImport:
    """Verify the module can be imported and exposes expected symbols."""

    def test_import_tool_spec(self):
        mod = _import_tool()
        assert hasattr(mod, "TOOL_SPEC")
        assert mod.TOOL_SPEC is not None

    def test_import_browser_agent_tool_function(self):
        mod = _import_tool()
        assert hasattr(mod, "browser_agent_tool")
        assert callable(mod.browser_agent_tool)

    def test_module_has_docstring(self):
        mod = _import_tool()
        assert mod.__doc__ is not None

    def test_module_imports_asyncio(self):
        mod = _import_tool()
        assert hasattr(mod, "asyncio")

    def test_module_imports_json(self):
        mod = _import_tool()
        assert hasattr(mod, "json")


# ---------------------------------------------------------------------------
# TOOL_SPEC contract tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Verify TOOL_SPEC follows the Strands tool specification contract."""

    def test_spec_has_name(self):
        mod = _import_tool()
        assert mod.TOOL_SPEC["name"] == "browser_agent_tool"

    def test_spec_has_description(self):
        mod = _import_tool()
        assert isinstance(mod.TOOL_SPEC["description"], str)
        assert len(mod.TOOL_SPEC["description"]) > 50

    def test_spec_description_mentions_browser(self):
        mod = _import_tool()
        assert "browser" in mod.TOOL_SPEC["description"].lower()

    def test_spec_description_mentions_javascript(self):
        mod = _import_tool()
        desc = mod.TOOL_SPEC["description"]
        assert "JavaScript" in desc or "javascript" in desc

    def test_spec_description_mentions_poc(self):
        mod = _import_tool()
        desc = mod.TOOL_SPEC["description"]
        assert "PoC" in desc or "Proof-of-Concept" in desc

    def test_spec_has_input_schema(self):
        mod = _import_tool()
        assert "inputSchema" in mod.TOOL_SPEC
        assert "json" in mod.TOOL_SPEC["inputSchema"]

    def test_spec_input_schema_is_object_type(self):
        mod = _import_tool()
        schema = mod.TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"

    def test_spec_requires_task_property(self):
        mod = _import_tool()
        schema = mod.TOOL_SPEC["inputSchema"]["json"]
        assert "task" in schema["properties"]
        assert "task" in schema["required"]

    def test_spec_task_property_is_string_type(self):
        mod = _import_tool()
        task_prop = mod.TOOL_SPEC["inputSchema"]["json"]["properties"]["task"]
        assert task_prop["type"] == "string"

    def test_spec_task_property_has_description(self):
        mod = _import_tool()
        task_prop = mod.TOOL_SPEC["inputSchema"]["json"]["properties"]["task"]
        assert "description" in task_prop
        assert len(task_prop["description"]) > 20

    def test_spec_no_extra_required_fields(self):
        mod = _import_tool()
        assert mod.TOOL_SPEC["inputSchema"]["json"]["required"] == ["task"]

    def test_spec_description_mentions_http_request(self):
        mod = _import_tool()
        assert "http_request" in mod.TOOL_SPEC["description"]


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Test handling of missing or empty task input."""

    def test_missing_task_returns_error(self):
        mod = _import_tool()
        tool_use = {"toolUseId": "test-001", "input": {}}
        result = mod.browser_agent_tool(tool_use)
        assert result["status"] == "error"
        assert result["toolUseId"] == "test-001"

    def test_missing_task_error_mentions_task(self):
        mod = _import_tool()
        tool_use = {"toolUseId": "test-002", "input": {}}
        result = mod.browser_agent_tool(tool_use)
        error_text = result["content"][0]["text"]
        assert "task" in error_text.lower()

    def test_empty_string_task_returns_error(self):
        mod = _import_tool()
        tool_use = {"toolUseId": "test-003", "input": {"task": ""}}
        result = mod.browser_agent_tool(tool_use)
        assert result["status"] == "error"

    def test_none_task_returns_error(self):
        mod = _import_tool()
        tool_use = {"toolUseId": "test-004", "input": {"task": None}}
        result = mod.browser_agent_tool(tool_use)
        assert result["status"] == "error"

    def test_tool_use_id_propagated_on_validation_error(self):
        mod = _import_tool()
        tool_use = {"toolUseId": "unique-id-123", "input": {}}
        result = mod.browser_agent_tool(tool_use)
        assert result["toolUseId"] == "unique-id-123"

    def test_content_is_list_on_validation_error(self):
        mod = _import_tool()
        tool_use = {"toolUseId": "test-005", "input": {}}
        result = mod.browser_agent_tool(tool_use)
        assert isinstance(result["content"], list)
        assert len(result["content"]) >= 1

    def test_zero_task_returns_error(self):
        """Numeric zero is falsy, should be treated as missing."""
        mod = _import_tool()
        tool_use = {"toolUseId": "test-006", "input": {"task": 0}}
        result = mod.browser_agent_tool(tool_use)
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Success path tests — asyncio.run delegation
# ---------------------------------------------------------------------------


class TestSuccessPath:
    """Test successful task execution via asyncio.run."""

    def test_success_returns_success_status(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"task_completed": True})):
            result = mod.browser_agent_tool({"toolUseId": "ok-1", "input": {"task": "browse"}})
        assert result["status"] == "success"

    def test_success_propagates_tool_use_id(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"key": "value"})):
            result = mod.browser_agent_tool({"toolUseId": "my-id-456", "input": {"task": "go"}})
        assert result["toolUseId"] == "my-id-456"

    def test_success_content_is_json_block(self, mock_browser_module):
        payload = {"task_completed": True, "summary": "Found page", "result": "data"}
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps(payload)):
            result = mod.browser_agent_tool({"toolUseId": "ok-2", "input": {"task": "browse"}})
        assert "json" in result["content"][0]
        assert result["content"][0]["json"] == payload

    def test_asyncio_run_called_once(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"ok": True})) as mock_run:
            mod.browser_agent_tool({"toolUseId": "ok-3", "input": {"task": "Go to NVD"}})
        mock_run.assert_called_once()

    def test_success_with_nested_json_result(self, mock_browser_module):
        payload = {"task_completed": True, "data": {"nested": [1, 2, 3]}, "result": "complex"}
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps(payload)):
            result = mod.browser_agent_tool({"toolUseId": "ok-4", "input": {"task": "scrape"}})
        assert result["content"][0]["json"]["data"]["nested"] == [1, 2, 3]

    def test_success_with_unicode_result(self, mock_browser_module):
        payload = {"summary": "Found \u2014 exploitation details", "result": "CVE-2024-\u2026"}
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps(payload)):
            result = mod.browser_agent_tool({"toolUseId": "ok-5", "input": {"task": "check"}})
        assert "\u2014" in result["content"][0]["json"]["summary"]

    def test_success_with_empty_json_object(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({})):
            result = mod.browser_agent_tool({"toolUseId": "ok-6", "input": {"task": "go"}})
        assert result["status"] == "success"
        assert result["content"][0]["json"] == {}

    def test_success_with_boolean_fields(self, mock_browser_module):
        payload = {"task_completed": False, "summary": "failed", "result": ""}
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps(payload)):
            result = mod.browser_agent_tool({"toolUseId": "ok-7", "input": {"task": "check"}})
        assert result["content"][0]["json"]["task_completed"] is False

    def test_run_browser_task_receives_task_string(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"ok": True})):
            mod.browser_agent_tool({"toolUseId": "ok-8", "input": {"task": "Navigate to example.com"}})
        # Verify run_browser_task was called with the task
        mock_browser_module.assert_called_once_with("Navigate to example.com")

    def test_success_with_numeric_values(self, mock_browser_module):
        payload = {"count": 42, "score": 3.14, "negative": -1}
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps(payload)):
            result = mod.browser_agent_tool({"toolUseId": "ok-9", "input": {"task": "count"}})
        assert result["content"][0]["json"]["count"] == 42
        assert result["content"][0]["json"]["score"] == 3.14


# ---------------------------------------------------------------------------
# Error / exception handling tests
# ---------------------------------------------------------------------------


class TestExceptionHandling:
    """Test error paths: asyncio failures, JSON decode errors, runtime errors."""

    def test_runtime_error_returns_error_status(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=RuntimeError("Event loop already running")):
            result = mod.browser_agent_tool({"toolUseId": "err-1", "input": {"task": "go"}})
        assert result["status"] == "error"

    def test_runtime_error_propagates_tool_use_id(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=RuntimeError("boom")):
            result = mod.browser_agent_tool({"toolUseId": "err-id-789", "input": {"task": "go"}})
        assert result["toolUseId"] == "err-id-789"

    def test_runtime_error_message_in_content(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=RuntimeError("Event loop is closed")):
            result = mod.browser_agent_tool({"toolUseId": "err-2", "input": {"task": "go"}})
        error_text = result["content"][0]["text"]
        assert "Event loop is closed" in error_text

    def test_timeout_error_handled(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=asyncio.TimeoutError("Browser timed out")):
            result = mod.browser_agent_tool({"toolUseId": "err-3", "input": {"task": "go"}})
        assert result["status"] == "error"

    def test_import_error_from_missing_browser_deps(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=ImportError("No module named 'browser_use'")):
            result = mod.browser_agent_tool({"toolUseId": "err-4", "input": {"task": "go"}})
        assert result["status"] == "error"
        assert "browser_use" in result["content"][0]["text"]

    def test_json_decode_error_returns_error(self, mock_browser_module):
        """If run_browser_task returns invalid JSON, json.loads raises."""
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value="not valid json {{"):
            result = mod.browser_agent_tool({"toolUseId": "err-5", "input": {"task": "go"}})
        assert result["status"] == "error"

    def test_keyboard_interrupt_not_swallowed(self, mock_browser_module):
        """KeyboardInterrupt should propagate (not caught by generic except Exception)."""
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=KeyboardInterrupt()):
            with pytest.raises(KeyboardInterrupt):
                mod.browser_agent_tool({"toolUseId": "err-6", "input": {"task": "go"}})

    def test_system_exit_not_swallowed(self, mock_browser_module):
        """SystemExit should propagate (BaseException, not Exception)."""
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=SystemExit(1)):
            with pytest.raises(SystemExit):
                mod.browser_agent_tool({"toolUseId": "err-7", "input": {"task": "go"}})

    def test_os_error_handled(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=OSError("Network unreachable")):
            result = mod.browser_agent_tool({"toolUseId": "err-8", "input": {"task": "go"}})
        assert result["status"] == "error"
        assert "Network unreachable" in result["content"][0]["text"]

    def test_value_error_handled(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=ValueError("Invalid task format")):
            result = mod.browser_agent_tool({"toolUseId": "err-9", "input": {"task": "go"}})
        assert result["status"] == "error"

    def test_attribute_error_handled(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=AttributeError("'NoneType' has no attr")):
            result = mod.browser_agent_tool({"toolUseId": "err-10", "input": {"task": "go"}})
        assert result["status"] == "error"

    def test_error_message_prefix(self, mock_browser_module):
        """Error messages should start with the expected prefix."""
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=Exception("generic failure")):
            result = mod.browser_agent_tool({"toolUseId": "err-11", "input": {"task": "go"}})
        error_text = result["content"][0]["text"]
        assert error_text.startswith("An error occurred while running the browser agent:")

    def test_connection_error_handled(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=ConnectionError("Connection refused")):
            result = mod.browser_agent_tool({"toolUseId": "err-12", "input": {"task": "go"}})
        assert result["status"] == "error"
        assert "Connection refused" in result["content"][0]["text"]

    def test_type_error_handled(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=TypeError("expected str, got NoneType")):
            result = mod.browser_agent_tool({"toolUseId": "err-13", "input": {"task": "go"}})
        assert result["status"] == "error"

    def test_memory_error_handled(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=MemoryError("Out of memory")):
            result = mod.browser_agent_tool({"toolUseId": "err-14", "input": {"task": "go"}})
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: special characters, large payloads, extra kwargs."""

    def test_task_with_special_characters(self, mock_browser_module):
        payload = {"result": "ok"}
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps(payload)):
            result = mod.browser_agent_tool(
                {"toolUseId": "edge-1", "input": {"task": "Navigate to 'http://x.com?a=1&b=2'"}}
            )
        assert result["status"] == "success"

    def test_task_with_newlines(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"ok": True})):
            result = mod.browser_agent_tool(
                {"toolUseId": "edge-2", "input": {"task": "Step 1: go\nStep 2: click\nStep 3: read"}}
            )
        assert result["status"] == "success"

    def test_extra_kwargs_ignored(self, mock_browser_module):
        """The function signature accepts **kwargs; extra args shouldn't crash."""
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"ok": True})):
            result = mod.browser_agent_tool(
                {"toolUseId": "edge-3", "input": {"task": "do something"}},
                model="test",
                session_id="abc",
            )
        assert result["status"] == "success"

    def test_extra_input_fields_ignored(self, mock_browser_module):
        """Extra fields in tool input beyond 'task' shouldn't crash."""
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"ok": True})):
            result = mod.browser_agent_tool(
                {"toolUseId": "edge-4", "input": {"task": "browse", "timeout": 30, "headless": True}}
            )
        assert result["status"] == "success"

    def test_large_json_result(self, mock_browser_module):
        """Large result payloads are handled correctly."""
        payload = {"items": [{"id": i, "data": "x" * 100} for i in range(100)]}
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps(payload)):
            result = mod.browser_agent_tool({"toolUseId": "edge-5", "input": {"task": "scrape all"}})
        assert result["status"] == "success"
        assert len(result["content"][0]["json"]["items"]) == 100

    def test_result_with_null_values(self, mock_browser_module):
        payload = {"task_completed": True, "summary": None, "result": None}
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps(payload)):
            result = mod.browser_agent_tool({"toolUseId": "edge-6", "input": {"task": "check"}})
        assert result["status"] == "success"
        assert result["content"][0]["json"]["summary"] is None

    def test_result_with_array_at_top_level(self, mock_browser_module):
        """JSON arrays are also valid results."""
        payload = [{"url": "http://a.com"}, {"url": "http://b.com"}]
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps(payload)):
            result = mod.browser_agent_tool({"toolUseId": "edge-7", "input": {"task": "list links"}})
        assert result["status"] == "success"
        assert isinstance(result["content"][0]["json"], list)

    def test_whitespace_task_is_truthy(self, mock_browser_module):
        """Whitespace-only task is truthy in Python, so asyncio.run is invoked."""
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"ok": True})):
            result = mod.browser_agent_tool({"toolUseId": "edge-8", "input": {"task": "   "}})
        # "   " is truthy — documents current behavior (no strip validation)
        assert result["status"] == "success"

    def test_very_long_task_string(self, mock_browser_module):
        """Very long task strings don't crash the tool."""
        long_task = "Navigate and click: " + "step " * 5000
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"ok": True})):
            result = mod.browser_agent_tool({"toolUseId": "edge-9", "input": {"task": long_task}})
        assert result["status"] == "success"

    def test_task_with_unicode_emoji(self, mock_browser_module):
        """Unicode emoji in task handled correctly."""
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"ok": True})):
            result = mod.browser_agent_tool({"toolUseId": "edge-10", "input": {"task": "Find the 🔒 security page"}})
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# ToolResult structure conformance
# ---------------------------------------------------------------------------


class TestToolResultStructure:
    """Verify that returned ToolResult dicts match expected Strands schema."""

    def test_success_result_has_required_keys(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"ok": True})):
            result = mod.browser_agent_tool({"toolUseId": "struct-1", "input": {"task": "go"}})
        assert "toolUseId" in result
        assert "status" in result
        assert "content" in result

    def test_success_content_is_list_of_dicts(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"ok": True})):
            result = mod.browser_agent_tool({"toolUseId": "struct-2", "input": {"task": "go"}})
        assert isinstance(result["content"], list)
        for item in result["content"]:
            assert isinstance(item, dict)

    def test_error_result_has_required_keys(self):
        mod = _import_tool()
        result = mod.browser_agent_tool({"toolUseId": "struct-3", "input": {}})
        assert "toolUseId" in result
        assert "status" in result
        assert "content" in result

    def test_error_content_has_text_key(self):
        mod = _import_tool()
        result = mod.browser_agent_tool({"toolUseId": "struct-4", "input": {}})
        assert "text" in result["content"][0]

    def test_success_content_has_json_key(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"data": 42})):
            result = mod.browser_agent_tool({"toolUseId": "struct-5", "input": {"task": "go"}})
        assert "json" in result["content"][0]

    def test_exception_result_content_has_text_key(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", side_effect=Exception("crash")):
            result = mod.browser_agent_tool({"toolUseId": "struct-6", "input": {"task": "go"}})
        assert "text" in result["content"][0]
        assert "crash" in result["content"][0]["text"]

    def test_success_status_is_literal_string(self, mock_browser_module):
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"ok": True})):
            result = mod.browser_agent_tool({"toolUseId": "struct-7", "input": {"task": "go"}})
        assert result["status"] == "success"

    def test_error_status_is_literal_string(self):
        mod = _import_tool()
        result = mod.browser_agent_tool({"toolUseId": "struct-8", "input": {}})
        assert result["status"] == "error"

    def test_success_content_single_item(self, mock_browser_module):
        """Success result content list has exactly one item."""
        mod = _import_tool()
        with patch.object(mod.asyncio, "run", return_value=json.dumps({"ok": True})):
            result = mod.browser_agent_tool({"toolUseId": "struct-9", "input": {"task": "go"}})
        assert len(result["content"]) == 1

    def test_error_content_single_item(self):
        """Error result content list has exactly one item."""
        mod = _import_tool()
        result = mod.browser_agent_tool({"toolUseId": "struct-10", "input": {}})
        assert len(result["content"]) == 1
