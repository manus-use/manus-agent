"""Comprehensive test suite for manus_workflow module.

Tests cover:
- TOOL_SPEC customisation and schema validation
- ManusWorkflowManager initialisation, agent registry, caching
- get_agent_for_task routing (all agent types, unknown fallback, caching, system_prompt)
- execute_task (success paths, dependency context, error handling, result extraction)
- manus_workflow entry point (create, start, list, status, delete, unknown action, missing args, exceptions)
"""

from __future__ import annotations

import uuid
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


class TestToolSpec:
    """Validate the customised TOOL_SPEC."""

    def test_tool_spec_name(self):
        from manus_agent.tools.manus_workflow import TOOL_SPEC

        assert TOOL_SPEC["name"] == "manus_workflow"

    def test_tool_spec_description_mentions_agent_types(self):
        from manus_agent.tools.manus_workflow import TOOL_SPEC

        desc = TOOL_SPEC["description"]
        for agent_type in ("manus", "browser", "data_analysis", "mcp"):
            assert agent_type in desc

    def test_tool_spec_has_input_schema(self):
        from manus_agent.tools.manus_workflow import TOOL_SPEC

        assert "inputSchema" in TOOL_SPEC

    def test_agent_type_enum_in_schema(self):
        from manus_agent.tools.manus_workflow import TOOL_SPEC

        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        agent_type_schema = props["tasks"]["items"]["properties"]["agent_type"]
        assert set(agent_type_schema["enum"]) == {"manus", "browser", "data_analysis", "mcp"}
        assert agent_type_schema["default"] == "manus"


# ---------------------------------------------------------------------------
# ManusWorkflowManager tests
# ---------------------------------------------------------------------------


class TestManusWorkflowManagerInit:
    """Initialisation and agent registry."""

    @patch("manus_agent.tools.manus_workflow.Config.from_file")
    @patch("manus_agent.tools.manus_workflow.WorkflowManager.__init__", return_value=None)
    def test_init_creates_agent_registry(self, _wm_init, _cfg):
        from manus_agent.tools.manus_workflow import ManusWorkflowManager

        mgr = ManusWorkflowManager({"system_prompt": None})
        assert "manus" in mgr.agent_registry
        assert "browser" in mgr.agent_registry
        assert "data_analysis" in mgr.agent_registry
        assert "mcp" in mgr.agent_registry

    @patch("manus_agent.tools.manus_workflow.Config.from_file")
    @patch("manus_agent.tools.manus_workflow.WorkflowManager.__init__", return_value=None)
    def test_init_empty_agent_cache(self, _wm_init, _cfg):
        from manus_agent.tools.manus_workflow import ManusWorkflowManager

        mgr = ManusWorkflowManager({"system_prompt": None})
        assert mgr.agent_instances == {}

    @patch("manus_agent.tools.manus_workflow.Config.from_file")
    @patch("manus_agent.tools.manus_workflow.WorkflowManager.__init__", return_value=None)
    def test_init_calls_config_from_file(self, _wm_init, mock_cfg):
        from manus_agent.tools.manus_workflow import ManusWorkflowManager

        ManusWorkflowManager({"system_prompt": None})
        mock_cfg.assert_called_once()

    @patch("manus_agent.tools.manus_workflow.Config.from_file")
    @patch("manus_agent.tools.manus_workflow.WorkflowManager.__init__", return_value=None)
    def test_init_calls_super(self, mock_wm_init, _cfg):
        from manus_agent.tools.manus_workflow import ManusWorkflowManager

        ctx = {"system_prompt": "test"}
        ManusWorkflowManager(ctx)
        mock_wm_init.assert_called_once_with(ctx)


# ---------------------------------------------------------------------------
# get_agent_for_task
# ---------------------------------------------------------------------------


class TestGetAgentForTask:
    """Routing tasks to the correct agent class."""

    def _make_manager(self):
        with (
            patch("manus_agent.tools.manus_workflow.Config.from_file") as mock_cfg,
            patch("manus_agent.tools.manus_workflow.WorkflowManager.__init__", return_value=None),
        ):
            from manus_agent.tools.manus_workflow import ManusWorkflowManager

            mock_cfg.return_value = MagicMock()
            mgr = ManusWorkflowManager({"system_prompt": None})
            # Replace agent classes with mocks so instantiation is cheap
            for key in mgr.agent_registry:
                mgr.agent_registry[key] = MagicMock()
            return mgr

    def test_default_agent_type_is_manus(self):
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
        # The fallback hardcodes ManusAgent, so we need to patch it at the module level
        with patch("manus_agent.tools.manus_workflow.ManusAgent") as mock_manus_cls:
            mock_manus_cls.return_value = MagicMock()
            agent = mgr.get_agent_for_task({"agent_type": "nonexistent"})
            mock_manus_cls.assert_called_once()
            assert agent is mock_manus_cls.return_value

    def test_agent_caching(self):
        mgr = self._make_manager()
        first = mgr.get_agent_for_task({"agent_type": "browser"})
        second = mgr.get_agent_for_task({"agent_type": "browser"})
        assert first is second
        # Should only have been instantiated once
        assert mgr.agent_registry["browser"].call_count == 1

    def test_system_prompt_passed_to_agent(self):
        mgr = self._make_manager()
        mgr.get_agent_for_task({"agent_type": "manus", "system_prompt": "Be concise."})
        mgr.agent_registry["manus"].assert_called_once_with(config=mgr.config, system_prompt="Be concise.")

    def test_no_system_prompt_creates_agent_without_it(self):
        mgr = self._make_manager()
        mgr.get_agent_for_task({"agent_type": "manus"})
        mgr.agent_registry["manus"].assert_called_once_with(config=mgr.config)

    def test_different_agent_types_cached_independently(self):
        mgr = self._make_manager()
        a1 = mgr.get_agent_for_task({"agent_type": "manus"})
        a2 = mgr.get_agent_for_task({"agent_type": "browser"})
        assert a1 is not a2
        assert "manus" in mgr.agent_instances
        assert "browser" in mgr.agent_instances


# ---------------------------------------------------------------------------
# execute_task
# ---------------------------------------------------------------------------


class TestExecuteTask:
    """Task execution, dependency resolution, and error handling."""

    def _make_manager(self):
        with (
            patch("manus_agent.tools.manus_workflow.Config.from_file") as mock_cfg,
            patch("manus_agent.tools.manus_workflow.WorkflowManager.__init__", return_value=None),
        ):
            from manus_agent.tools.manus_workflow import ManusWorkflowManager

            mock_cfg.return_value = MagicMock()
            mgr = ManusWorkflowManager({"system_prompt": None})
            for key in mgr.agent_registry:
                mgr.agent_registry[key] = MagicMock()
            return mgr

    def test_success_dict_result(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "done"}], "stop_reason": "end_turn"}
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Say hello", "agent_type": "manus"}
        workflow = {"task_results": {}}
        result = mgr.execute_task(task, workflow, "tu-1")

        assert result["status"] == "success"
        assert result["content"] == [{"text": "done"}]

    def test_error_stop_reason(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "fail"}], "stop_reason": "error"}
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Fail", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")

        assert result["status"] == "error"

    def test_object_result_with_content_attr(self):
        mgr = self._make_manager()
        mock_result = MagicMock(spec=[])
        mock_result.content = [{"text": "obj result"}]
        mock_result.stop_reason = "end_turn"
        # Make .get() raise AttributeError so the except branch is hit
        mock_agent = MagicMock()
        mock_agent.return_value = mock_result
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Test", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")

        assert result["status"] == "success"
        assert result["content"] == [{"text": "obj result"}]

    def test_dependency_context_injected(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "ok"}], "stop_reason": "end_turn"}
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {
            "task_id": "t2",
            "description": "Follow up",
            "agent_type": "manus",
            "dependencies": ["t1"],
        }
        workflow = {
            "task_results": {
                "t1": {"status": "completed", "result": [{"text": "first result"}]},
            },
        }
        mgr.execute_task(task, workflow, "tu-2")

        # Verify the prompt includes dependency context
        call_args = mock_agent.call_args[0][0]
        assert "Results from t1" in call_args
        assert "first result" in call_args
        assert "Follow up" in call_args

    def test_dependency_not_completed_skipped(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [], "stop_reason": "end_turn"}
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {
            "task_id": "t2",
            "description": "Follow up",
            "agent_type": "manus",
            "dependencies": ["t1"],
        }
        workflow = {
            "task_results": {
                "t1": {"status": "pending", "result": None},
            },
        }
        mgr.execute_task(task, workflow, "tu-2")

        # Should not include dependency context
        call_args = mock_agent.call_args[0][0]
        assert "Previous task results" not in call_args

    def test_no_dependencies(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [], "stop_reason": "end_turn"}
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Solo task", "agent_type": "manus"}
        mgr.execute_task(task, {"task_results": {}}, "tu-1")

        call_args = mock_agent.call_args[0][0]
        assert call_args == "Solo task"

    def test_exception_returns_error(self):
        mgr = self._make_manager()
        mock_agent = MagicMock(side_effect=RuntimeError("boom"))
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Blow up", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")

        assert result["status"] == "error"
        assert "boom" in result["content"][0]["text"]

    def test_multiple_dependencies_concatenated(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [], "stop_reason": "end_turn"}
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
        assert "Results from t1" in call_args
        assert "alpha" in call_args
        assert "Results from t2" in call_args
        assert "beta" in call_args

    def test_empty_content_result(self):
        mgr = self._make_manager()
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [], "stop_reason": "end_turn"}
        mgr.agent_registry["manus"] = MagicMock(return_value=mock_agent)

        task = {"task_id": "t1", "description": "Empty", "agent_type": "manus"}
        result = mgr.execute_task(task, {"task_results": {}}, "tu-1")

        assert result["status"] == "success"
        assert result["content"] == []


# ---------------------------------------------------------------------------
# manus_workflow entry point
# ---------------------------------------------------------------------------


class TestManusWorkflowEntryPoint:
    """Tests for the manus_workflow() function."""

    def _call(self, action: str, extra: dict | None = None, **kwargs):
        from manus_agent.tools.manus_workflow import manus_workflow

        tool = _make_tool_use(action, extra, tool_use_id="test-id")
        return manus_workflow(tool, **kwargs)

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_create_action_success(self, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.create_workflow.return_value = {
            "status": "success",
            "content": [{"text": "Created"}],
        }
        result = self._call(
            "create",
            {"workflow_id": "wf-1", "tasks": [{"task_id": "t1", "description": "Do stuff"}]},
        )
        assert result["status"] == "success"
        mock_mgr.create_workflow.assert_called_once()

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_create_action_missing_tasks(self, mock_mgr_cls):
        result = self._call("create", {"workflow_id": "wf-1"})
        assert result["status"] == "error"
        assert "Tasks are required" in result["content"][0]["text"]

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_create_action_generates_workflow_id_if_missing(self, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.create_workflow.return_value = {
            "status": "success",
            "content": [{"text": "Created"}],
        }
        self._call("create", {"tasks": [{"task_id": "t1", "description": "Task"}]})
        # The first arg to create_workflow should be a valid UUID string
        call_args = mock_mgr.create_workflow.call_args[0]
        uuid.UUID(call_args[0])  # Should not raise

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_start_action_success(self, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.start_workflow.return_value = {
            "status": "success",
            "content": [{"text": "Started"}],
        }
        result = self._call("start", {"workflow_id": "wf-1"})
        assert result["status"] == "success"
        mock_mgr.start_workflow.assert_called_once_with("wf-1", "test-id")

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_start_action_missing_workflow_id(self, mock_mgr_cls):
        result = self._call("start")
        assert result["status"] == "error"
        assert "workflow_id is required" in result["content"][0]["text"]

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_list_action(self, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.list_workflows.return_value = {
            "status": "success",
            "content": [{"text": "[]"}],
        }
        result = self._call("list")
        assert result["status"] == "success"
        mock_mgr.list_workflows.assert_called_once_with("test-id")

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_status_action_success(self, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.get_workflow_status.return_value = {
            "status": "success",
            "content": [{"text": "running"}],
        }
        result = self._call("status", {"workflow_id": "wf-1"})
        assert result["status"] == "success"
        mock_mgr.get_workflow_status.assert_called_once_with("wf-1", "test-id")

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_status_action_missing_workflow_id(self, mock_mgr_cls):
        result = self._call("status")
        assert result["status"] == "error"

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_delete_action_success(self, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.delete_workflow.return_value = {
            "status": "success",
            "content": [{"text": "Deleted"}],
        }
        result = self._call("delete", {"workflow_id": "wf-1"})
        assert result["status"] == "success"
        mock_mgr.delete_workflow.assert_called_once_with("wf-1", "test-id")

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_delete_action_missing_workflow_id(self, mock_mgr_cls):
        result = self._call("delete")
        assert result["status"] == "error"

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_unknown_action(self, mock_mgr_cls):
        result = self._call("explode")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    @patch(
        "manus_agent.tools.manus_workflow.ManusWorkflowManager",
        side_effect=RuntimeError("init failed"),
    )
    def test_exception_in_entry_point(self, mock_mgr_cls):
        result = self._call("list")
        assert result["status"] == "error"
        assert "init failed" in result["content"][0]["text"]

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_tool_use_id_propagated(self, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.list_workflows.return_value = {
            "status": "success",
            "content": [],
        }
        result = self._call("list")
        assert result["toolUseId"] == "test-id"

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_tool_use_id_generated_if_missing(self, mock_mgr_cls):
        from manus_agent.tools.manus_workflow import manus_workflow

        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.list_workflows.return_value = {
            "status": "success",
            "content": [],
        }
        tool = {"input": {"action": "list"}}  # No toolUseId
        result = manus_workflow(tool)
        # Should have generated a UUID
        uuid.UUID(result["toolUseId"])

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_kwargs_passed_to_manager(self, mock_mgr_cls):
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
        assert init_ctx["inference_config"] == {"max": 100}
        assert init_ctx["messages"] == []
        assert init_ctx["tool_config"] == {}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge cases."""

    @patch("manus_agent.tools.manus_workflow.Config.from_file")
    @patch("manus_agent.tools.manus_workflow.WorkflowManager.__init__", return_value=None)
    def test_agent_registry_maps_to_correct_classes(self, _wm, _cfg):
        from manus_agent.agents import BrowserUseAgent, DataAnalysisAgent, ManusAgent, MCPAgent
        from manus_agent.tools.manus_workflow import ManusWorkflowManager

        mgr = ManusWorkflowManager({})
        assert mgr.agent_registry["manus"] is ManusAgent
        assert mgr.agent_registry["browser"] is BrowserUseAgent
        assert mgr.agent_registry["data_analysis"] is DataAnalysisAgent
        assert mgr.agent_registry["mcp"] is MCPAgent

    @patch("manus_agent.tools.manus_workflow.ManusWorkflowManager")
    def test_create_preserves_explicit_workflow_id(self, mock_mgr_cls):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.create_workflow.return_value = {
            "status": "success",
            "content": [{"text": "ok"}],
        }
        from manus_agent.tools.manus_workflow import manus_workflow

        tool = _make_tool_use(
            "create",
            {"workflow_id": "my-wf", "tasks": [{"task_id": "t1", "description": "x"}]},
            tool_use_id="tu",
        )
        manus_workflow(tool)
        call_args = mock_mgr.create_workflow.call_args[0]
        assert call_args[0] == "my-wf"

    def test_execute_task_missing_dependency_key(self):
        """Dependency id not present in task_results should not crash."""
        with (
            patch("manus_agent.tools.manus_workflow.Config.from_file"),
            patch("manus_agent.tools.manus_workflow.WorkflowManager.__init__", return_value=None),
        ):
            from manus_agent.tools.manus_workflow import ManusWorkflowManager

            mgr = ManusWorkflowManager({})
            mock_agent = MagicMock()
            mock_agent.return_value = {"content": [], "stop_reason": "end_turn"}
            for key in mgr.agent_registry:
                mgr.agent_registry[key] = MagicMock(return_value=mock_agent)

            task = {
                "task_id": "t2",
                "description": "Test",
                "agent_type": "manus",
                "dependencies": ["nonexistent"],
            }
            result = mgr.execute_task(task, {"task_results": {}}, "tu-1")
            # Should succeed — missing dependency is silently ignored
            assert result["status"] == "success"
