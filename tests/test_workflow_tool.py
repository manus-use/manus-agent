"""Comprehensive test suite for workflow_tool module.

Tests cover:
- TOOL_SPEC structure and schema validation
- ManusWorkflowManager initialisation, agent registry, caching
- get_agent_for_task routing (all types, unknown fallback, caching, system_prompt)
- execute_task (dict results, object results, string results, None results,
  coroutine results, dependency context, error handling, stop_reason variants)
- create_workflow (success, store failure, default priority, UUID generation)
- workflow_tool entry point (create, start, list, status, delete, unknown action,
  missing args, exceptions, print output)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use(action: str, extra: dict | None = None, tool_use_id: str | None = None) -> dict:
    """Build a minimal ToolUse dict."""
    inp: dict[str, Any] = {"action": action}
    if extra:
        inp.update(extra)
    return {
        "toolUseId": tool_use_id or str(uuid.uuid4()),
        "input": inp,
    }


# ---------------------------------------------------------------------------
# TOOL_SPEC tests
# ---------------------------------------------------------------------------


class TestWorkflowToolSpec:
    """Validate the TOOL_SPEC for workflow_tool."""

    def test_tool_spec_name(self):
        from manus_agent.tools.workflow_tool import TOOL_SPEC

        assert TOOL_SPEC["name"] == "workflow_tool"

    def test_tool_spec_description_mentions_agent_types(self):
        from manus_agent.tools.workflow_tool import TOOL_SPEC

        desc = TOOL_SPEC["description"]
        for agent_type in ("manus", "browser", "data_analysis", "mcp"):
            assert agent_type in desc

    def test_tool_spec_has_input_schema(self):
        from manus_agent.tools.workflow_tool import TOOL_SPEC

        assert "inputSchema" in TOOL_SPEC

    def test_input_schema_enum(self):
        from manus_agent.tools.workflow_tool import TOOL_SPEC

        schema = TOOL_SPEC["inputSchema"]
        assert set(schema["enum"]) == {"manus", "browser", "data_analysis", "mcp"}
        assert schema["default"] == "manus"


# ---------------------------------------------------------------------------
# ManusWorkflowManager init
# ---------------------------------------------------------------------------


class TestWorkflowToolManagerInit:
    """Initialisation and agent registry for workflow_tool's ManusWorkflowManager."""

    @patch("manus_agent.tools.workflow_tool.Config.from_file")
    @patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None)
    def test_init_creates_agent_registry(self, _wm_init, _cfg):
        from manus_agent.tools.workflow_tool import ManusWorkflowManager

        mgr = ManusWorkflowManager({"system_prompt": None})
        assert "manus" in mgr.agent_registry
        assert "browser" in mgr.agent_registry
        assert "data_analysis" in mgr.agent_registry
        assert "mcp" in mgr.agent_registry

    @patch("manus_agent.tools.workflow_tool.Config.from_file")
    @patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None)
    def test_init_empty_agent_cache(self, _wm_init, _cfg):
        from manus_agent.tools.workflow_tool import ManusWorkflowManager

        mgr = ManusWorkflowManager({"system_prompt": None})
        assert mgr.agent_instances == {}

    @patch("manus_agent.tools.workflow_tool.Config.from_file")
    @patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None)
    def test_init_calls_config(self, _wm_init, mock_cfg):
        from manus_agent.tools.workflow_tool import ManusWorkflowManager

        ManusWorkflowManager({})
        mock_cfg.assert_called_once()

    @patch("manus_agent.tools.workflow_tool.Config.from_file")
    @patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None)
    def test_init_calls_super(self, mock_wm_init, _cfg):
        from manus_agent.tools.workflow_tool import ManusWorkflowManager

        ctx = {"system_prompt": "test"}
        ManusWorkflowManager(ctx)
        mock_wm_init.assert_called_once_with(ctx)

    @patch("manus_agent.tools.workflow_tool.Config.from_file")
    @patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None)
    def test_registry_maps_correct_classes(self, _wm, _cfg):
        from manus_agent.agents import BrowserUseAgent, DataAnalysisAgent, ManusAgent, MCPAgent
        from manus_agent.tools.workflow_tool import ManusWorkflowManager

        mgr = ManusWorkflowManager({})
        assert mgr.agent_registry["manus"] is ManusAgent
        assert mgr.agent_registry["browser"] is BrowserUseAgent
        assert mgr.agent_registry["data_analysis"] is DataAnalysisAgent
        assert mgr.agent_registry["mcp"] is MCPAgent


# ---------------------------------------------------------------------------
# get_agent_for_task
# ---------------------------------------------------------------------------


class TestWorkflowToolGetAgentForTask:
    """Routing tasks to the correct agent class."""

    def _make_manager(self):
        with (
            patch("manus_agent.tools.workflow_tool.Config.from_file") as mock_cfg,
            patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None),
        ):
            from manus_agent.tools.workflow_tool import ManusWorkflowManager

            mock_cfg.return_value = MagicMock()
            mgr = ManusWorkflowManager({})
            for key in mgr.agent_registry:
                mgr.agent_registry[key] = MagicMock()
            return mgr

    def test_default_agent_type(self):
        mgr = self._make_manager()
        agent = mgr.get_agent_for_task({})
        mgr.agent_registry["manus"].assert_called_once()
        assert agent is mgr.agent_registry["manus"].return_value

    @pytest.mark.parametrize("agent_type", ["manus", "browser", "data_analysis", "mcp"])
    def test_known_agent_types(self, agent_type):
        mgr = self._make_manager()
        agent = mgr.get_agent_for_task({"agent_type": agent_type})
        mgr.agent_registry[agent_type].assert_called_once()
        assert agent is mgr.agent_registry[agent_type].return_value

    def test_unknown_agent_type_falls_back_to_manus(self):
        mgr = self._make_manager()
        with patch("manus_agent.tools.workflow_tool.ManusAgent") as mock_manus_cls:
            mock_manus_cls.return_value = MagicMock()
            agent = mgr.get_agent_for_task({"agent_type": "unknown"})
            mock_manus_cls.assert_called_once()
            assert agent is mock_manus_cls.return_value

    def test_agent_caching(self):
        mgr = self._make_manager()
        first = mgr.get_agent_for_task({"agent_type": "browser"})
        second = mgr.get_agent_for_task({"agent_type": "browser"})
        assert first is second
        assert mgr.agent_registry["browser"].call_count == 1

    def test_system_prompt_passed(self):
        mgr = self._make_manager()
        mgr.get_agent_for_task({"agent_type": "manus", "system_prompt": "Be brief."})
        mgr.agent_registry["manus"].assert_called_once_with(config=mgr.config, system_prompt="Be brief.")

    def test_no_system_prompt(self):
        mgr = self._make_manager()
        mgr.get_agent_for_task({"agent_type": "manus"})
        mgr.agent_registry["manus"].assert_called_once_with(config=mgr.config)

    def test_different_types_cached_independently(self):
        mgr = self._make_manager()
        a1 = mgr.get_agent_for_task({"agent_type": "manus"})
        a2 = mgr.get_agent_for_task({"agent_type": "data_analysis"})
        assert a1 is not a2
        assert len(mgr.agent_instances) == 2


# ---------------------------------------------------------------------------
# execute_task
# ---------------------------------------------------------------------------


class TestWorkflowToolExecuteTask:
    """Task execution — covers all result type branches and edge cases."""

    def _make_manager(self):
        with (
            patch("manus_agent.tools.workflow_tool.Config.from_file") as mock_cfg,
            patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None),
        ):
            from manus_agent.tools.workflow_tool import ManusWorkflowManager

            mock_cfg.return_value = MagicMock()
            mgr = ManusWorkflowManager({})
            for key in mgr.agent_registry:
                mgr.agent_registry[key] = MagicMock()
            return mgr

    def test_dict_result_with_list_content(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        # Return a real dict — workflow_tool's execute_task handles dict results via .get()
        mock_agent.return_value = {"content": [{"text": "done"}], "stop_reason": "end_turn"}
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Hello", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")

        assert result["status"] == "success"
        assert result["content"] == [{"text": "done"}]

    def test_dict_result_with_string_content(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": "text result", "stop_reason": "end_turn"}
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["status"] == "success"
        assert result["content"] == [{"text": "text result"}]

    def test_dict_result_with_none_content(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": None, "stop_reason": "end_turn"}
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["status"] == "success"
        assert result["content"] == []

    def test_dict_result_with_other_content_type(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": 42, "stop_reason": "end_turn"}
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["content"] == [{"text": "42"}]

    def test_string_result(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = "plain text response"
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["status"] == "success"
        assert result["content"] == [{"text": "plain text response"}]

    def test_object_result_with_list_content(self):
        mgr = self._make_manager()
        mock_result = MagicMock(spec=[])  # No .get()
        mock_result.content = [{"text": "obj list"}]
        mock_result.stop_reason = "end_turn"
        mock_agent = MagicMock()
        mock_agent.return_value = mock_result
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["content"] == [{"text": "obj list"}]

    def test_object_result_with_string_content(self):
        mgr = self._make_manager()
        mock_result = MagicMock(spec=[])
        mock_result.content = "obj string"
        mock_result.stop_reason = "end_turn"
        mock_agent = MagicMock()
        mock_agent.return_value = mock_result
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["content"] == [{"text": "obj string"}]

    def test_object_result_with_none_content(self):
        mgr = self._make_manager()
        mock_result = MagicMock(spec=[])
        mock_result.content = None
        mock_result.stop_reason = "ok"
        mock_agent = MagicMock()
        mock_agent.return_value = mock_result
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["content"] == []

    def test_none_result(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = None
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["status"] == "success"
        assert result["content"] == []

    def test_unexpected_type_result(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = 12345  # unexpected type
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["content"] == [{"text": "12345"}]

    def test_error_stop_reason(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [], "stop_reason": "error"}
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Fail", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["status"] == "error"

    def test_coroutine_result_no_running_loop(self):
        """When agent returns a coroutine and no loop is running, asyncio.run is used."""
        mgr = self._make_manager()

        async def async_agent_call(prompt):
            return "async result"

        mock_agent = MagicMock()
        mock_agent.return_value = async_agent_call("test")
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Async", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["status"] == "success"
        assert result["content"] == [{"text": "async result"}]

    def test_dependency_context_injected(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = "ok"
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {
            "task_id": "t2",
            "description": "Follow up",
            "agent_type": "manus",
            "dependencies": ["t1"],
        }
        workflow = {
            "task_results": {
                "t1": {"status": "completed", "result": [{"text": "first"}]},
            },
        }
        mgr.execute_task(task, workflow, "tu-2")

        call_args = mock_agent.call_args[0][0]
        assert "Results from t1" in call_args
        assert "first" in call_args
        assert "Follow up" in call_args

    def test_dependency_not_completed_skipped(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = "ok"
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {
            "task_id": "t2",
            "description": "Follow up",
            "agent_type": "manus",
            "dependencies": ["t1"],
        }
        workflow = {"task_results": {"t1": {"status": "pending", "result": None}}}
        mgr.execute_task(task, workflow, "tu-2")

        call_args = mock_agent.call_args[0][0]
        assert "Previous task results" not in call_args

    def test_no_dependencies(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = "ok"
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Solo", "agent_type": "manus"}
        mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert mock_agent.call_args[0][0] == "Solo"

    def test_exception_returns_error(self):
        mgr = self._make_manager()
        mock_agent = MagicMock(side_effect=RuntimeError("boom"))
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Explode", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["status"] == "error"
        assert "boom" in result["content"][0]["text"]

    def test_tool_use_id_in_result(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = "ok"
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "my-tu-id")
        assert result["toolUseId"] == "my-tu-id"

    def test_multiple_dependencies(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = "ok"
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {
            "task_id": "t3",
            "description": "Combine",
            "agent_type": "manus",
            "dependencies": ["t1", "t2"],
        }
        workflow = {
            "task_results": {
                "t1": {"status": "completed", "result": [{"text": "alpha"}]},
                "t2": {"status": "completed", "result": [{"text": "beta"}]},
            },
        }
        mgr.execute_task(task, workflow, "tu-3")

        call_args = mock_agent.call_args[0][0]
        assert "alpha" in call_args
        assert "beta" in call_args


# ---------------------------------------------------------------------------
# create_workflow
# ---------------------------------------------------------------------------


class TestWorkflowToolCreateWorkflow:
    """Tests for ManusWorkflowManager.create_workflow."""

    def _make_manager(self):
        with (
            patch("manus_agent.tools.workflow_tool.Config.from_file"),
            patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None),
        ):
            from manus_agent.tools.workflow_tool import ManusWorkflowManager

            mgr = ManusWorkflowManager({})
            mgr.store_workflow = MagicMock(return_value={"status": "success"})
            return mgr

    def test_create_success(self):
        mgr = self._make_manager()
        tasks = [{"task_id": "t1", "description": "Task 1"}]
        result = mgr.create_workflow("wf-1", tasks, "tu-1")
        assert result["status"] == "success"
        assert "wf-1" in result["content"][0]["text"]

    def test_create_generates_workflow_id_if_empty(self):
        mgr = self._make_manager()
        tasks = [{"task_id": "t1", "description": "Task 1"}]
        result = mgr.create_workflow("", tasks, "tu-1")
        assert result["status"] == "success"
        # store_workflow was called with a UUID-like id
        stored_id = mgr.store_workflow.call_args[0][0]
        uuid.UUID(stored_id)

    def test_create_adds_default_priority(self):
        mgr = self._make_manager()
        tasks = [{"task_id": "t1", "description": "Task 1"}]
        mgr.create_workflow("wf-1", tasks, "tu-1")
        # The task should now have priority=3
        stored_workflow = mgr.store_workflow.call_args[0][1]
        assert stored_workflow["tasks"][0]["priority"] == 3

    def test_create_preserves_explicit_priority(self):
        mgr = self._make_manager()
        tasks = [{"task_id": "t1", "description": "Task 1", "priority": 1}]
        mgr.create_workflow("wf-1", tasks, "tu-1")
        stored_workflow = mgr.store_workflow.call_args[0][1]
        assert stored_workflow["tasks"][0]["priority"] == 1

    def test_create_workflow_structure(self):
        mgr = self._make_manager()
        tasks = [
            {"task_id": "t1", "description": "A"},
            {"task_id": "t2", "description": "B"},
        ]
        mgr.create_workflow("wf-1", tasks, "tu-1")
        stored = mgr.store_workflow.call_args[0][1]
        assert stored["workflow_id"] == "wf-1"
        assert stored["status"] == "created"
        assert stored["current_task_index"] == 0
        assert len(stored["tasks"]) == 2
        assert "t1" in stored["task_results"]
        assert "t2" in stored["task_results"]
        assert stored["task_results"]["t1"]["status"] == "pending"

    def test_create_store_failure(self):
        mgr = self._make_manager()
        mgr.store_workflow.return_value = {"status": "error", "error": "disk full"}
        tasks = [{"task_id": "t1", "description": "A"}]
        result = mgr.create_workflow("wf-1", tasks, "tu-1")
        assert result["status"] == "error"
        assert "disk full" in result["content"][0]["text"]

    def test_create_exception(self):
        mgr = self._make_manager()
        mgr.store_workflow.side_effect = RuntimeError("unexpected")
        tasks = [{"task_id": "t1", "description": "A"}]
        result = mgr.create_workflow("wf-1", tasks, "tu-1")
        assert result["status"] == "error"
        assert "unexpected" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# workflow_tool entry point
# ---------------------------------------------------------------------------


class TestWorkflowToolEntryPoint:
    """Tests for the workflow_tool() function."""

    def _call(self, action: str, extra: dict | None = None, **kwargs):
        from manus_agent.tools.workflow_tool import workflow_tool

        tool = _make_tool_use(action, extra, tool_use_id="test-id")
        return workflow_tool(tool, **kwargs)

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_create_action_success(self, _print, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.create_workflow.return_value = {
            "status": "success",
            "content": [{"text": "Created"}],
        }
        result = self._call(
            "create",
            {"workflow_id": "wf-1", "tasks": [{"task_id": "t1", "description": "X"}]},
        )
        assert result["status"] == "success"

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_create_missing_tasks(self, _print, mock_mgr_cls):
        result = self._call("create", {"workflow_id": "wf-1"})
        assert result["status"] == "error"
        assert "Tasks are required" in result["content"][0]["text"]

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_create_generates_workflow_id(self, _print, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.create_workflow.return_value = {
            "status": "success",
            "content": [{"text": "Created"}],
        }
        self._call("create", {"tasks": [{"task_id": "t1", "description": "X"}]})
        call_args = mock_mgr.create_workflow.call_args[0]
        uuid.UUID(call_args[0])

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_start_action(self, _print, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.start_workflow.return_value = {
            "status": "success",
            "content": [{"text": "Started"}],
        }
        result = self._call("start", {"workflow_id": "wf-1"})
        assert result["status"] == "success"
        mock_mgr.start_workflow.assert_called_once_with("wf-1", "test-id")

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_start_missing_workflow_id(self, _print, mock_mgr_cls):
        result = self._call("start")
        assert result["status"] == "error"
        assert "workflow_id is required" in result["content"][0]["text"]

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_list_action(self, _print, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.list_workflows.return_value = {
            "status": "success",
            "content": [{"text": "[]"}],
        }
        result = self._call("list")
        assert result["status"] == "success"
        mock_mgr.list_workflows.assert_called_once_with("test-id")

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_status_action(self, _print, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.get_workflow_status.return_value = {
            "status": "success",
            "content": [{"text": "running"}],
        }
        result = self._call("status", {"workflow_id": "wf-1"})
        assert result["status"] == "success"
        mock_mgr.get_workflow_status.assert_called_once_with("wf-1", "test-id")

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_status_missing_workflow_id(self, _print, mock_mgr_cls):
        result = self._call("status")
        assert result["status"] == "error"

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_delete_action(self, _print, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.delete_workflow.return_value = {
            "status": "success",
            "content": [{"text": "Deleted"}],
        }
        result = self._call("delete", {"workflow_id": "wf-1"})
        assert result["status"] == "success"

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_delete_missing_workflow_id(self, _print, mock_mgr_cls):
        result = self._call("delete")
        assert result["status"] == "error"

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_unknown_action(self, _print, mock_mgr_cls):
        result = self._call("nope")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    @patch(
        "manus_agent.tools.workflow_tool.ManusWorkflowManager",
        side_effect=RuntimeError("init failed"),
    )
    @patch("builtins.print")
    def test_exception_in_entry_point(self, _print, mock_mgr_cls):
        result = self._call("list")
        assert result["status"] == "error"
        assert "init failed" in result["content"][0]["text"]

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_tool_use_id_propagated(self, _print, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.list_workflows.return_value = {
            "status": "success",
            "content": [],
        }
        result = self._call("list")
        assert result["toolUseId"] == "test-id"

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_tool_use_id_generated_if_missing(self, _print, mock_mgr_cls):
        from manus_agent.tools.workflow_tool import workflow_tool

        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.list_workflows.return_value = {
            "status": "success",
            "content": [],
        }
        tool = {"input": {"action": "list"}}
        result = workflow_tool(tool)
        uuid.UUID(result["toolUseId"])

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_kwargs_passed_to_manager(self, _print, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.list_workflows.return_value = {
            "status": "success",
            "content": [],
        }
        self._call(
            "list",
            system_prompt="sp",
            inference_config={"max": 100},
            messages=[],
            tool_config={},
        )
        init_ctx = mock_mgr_cls.call_args[0][0]
        assert init_ctx["system_prompt"] == "sp"

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_print_called_with_input(self, mock_print, mock_mgr_cls):
        """The entry point prints the tool input (debug output)."""
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.list_workflows.return_value = {
            "status": "success",
            "content": [],
        }
        self._call("list")
        # Should have called print with separator and input
        assert mock_print.call_count >= 2

    @patch("manus_agent.tools.workflow_tool.ManusWorkflowManager")
    @patch("builtins.print")
    def test_create_preserves_explicit_workflow_id(self, _print, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.create_workflow.return_value = {
            "status": "success",
            "content": [{"text": "ok"}],
        }
        self._call(
            "create",
            {"workflow_id": "my-wf", "tasks": [{"task_id": "t1", "description": "x"}]},
        )
        call_args = mock_mgr.create_workflow.call_args[0]
        assert call_args[0] == "my-wf"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestWorkflowToolEdgeCases:
    """Miscellaneous edge cases for workflow_tool module."""

    def test_missing_dependency_key_no_crash(self):
        """Dependency id not in task_results should not crash."""
        with (
            patch("manus_agent.tools.workflow_tool.Config.from_file"),
            patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None),
        ):
            from manus_agent.tools.workflow_tool import ManusWorkflowManager

            mgr = ManusWorkflowManager({})
            mock_agent = MagicMock()
            mock_agent.return_value = "ok"
            for key in mgr.agent_registry:
                mgr.agent_registry[key] = MagicMock(return_value=mock_agent)

            task = {
                "task_id": "t2",
                "description": "Test",
                "agent_type": "manus",
                "dependencies": ["nonexistent"],
            }
            result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
            assert result["status"] == "success"

    def test_create_workflow_has_created_at(self):
        """Verify created_at timestamp is present in stored workflow."""
        with (
            patch("manus_agent.tools.workflow_tool.Config.from_file"),
            patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None),
        ):
            from manus_agent.tools.workflow_tool import ManusWorkflowManager

            mgr = ManusWorkflowManager({})
            mgr.store_workflow = MagicMock(return_value={"status": "success"})
            mgr.create_workflow("wf-1", [{"task_id": "t1", "description": "A"}], "tu-1")
            stored = mgr.store_workflow.call_args[0][1]
            # Should parse as ISO datetime
            dt = datetime.fromisoformat(stored["created_at"])
            assert dt.tzinfo is not None

    def test_create_workflow_parallel_execution_false(self):
        """Verify parallel_execution defaults to False."""
        with (
            patch("manus_agent.tools.workflow_tool.Config.from_file"),
            patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None),
        ):
            from manus_agent.tools.workflow_tool import ManusWorkflowManager

            mgr = ManusWorkflowManager({})
            mgr.store_workflow = MagicMock(return_value={"status": "success"})
            mgr.create_workflow("wf-1", [{"task_id": "t1", "description": "A"}], "tu-1")
            stored = mgr.store_workflow.call_args[0][1]
            assert stored["parallel_execution"] is False

    def test_object_result_with_stop_reason_attr(self):
        """Object result that has .stop_reason attribute."""
        with (
            patch("manus_agent.tools.workflow_tool.Config.from_file"),
            patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None),
        ):
            from manus_agent.tools.workflow_tool import ManusWorkflowManager

            mgr = ManusWorkflowManager({})
            mock_result = MagicMock(spec=[])
            mock_result.content = [{"text": "ok"}]
            mock_result.stop_reason = "error"
            mock_agent = MagicMock()
            mock_agent.return_value = mock_result
            for key in mgr.agent_registry:
                mgr.agent_registry[key] = MagicMock(return_value=mock_agent)

            task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
            result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
            assert result["status"] == "error"

    def test_object_result_with_other_content_type(self):
        """Object result where .content is not list/str/None."""
        with (
            patch("manus_agent.tools.workflow_tool.Config.from_file"),
            patch("manus_agent.tools.workflow_tool.WorkflowManager.__init__", return_value=None),
        ):
            from manus_agent.tools.workflow_tool import ManusWorkflowManager

            mgr = ManusWorkflowManager({})
            mock_result = MagicMock(spec=[])
            mock_result.content = 99
            mock_result.stop_reason = "end_turn"
            mock_agent = MagicMock()
            mock_agent.return_value = mock_result
            for key in mgr.agent_registry:
                mgr.agent_registry[key] = MagicMock(return_value=mock_agent)

            task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
            result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
            assert result["content"] == [{"text": "99"}]
