"""Comprehensive test suite for manus_agent.tools.manus_workflow module.

Tests cover:
- TOOL_SPEC structure and customization
- ManusWorkflowManager.__init__ (agent registry, caching, config)
- ManusWorkflowManager.get_agent_for_task (routing, caching, fallback, system_prompt)
- ManusWorkflowManager.execute_task (dependency context, agent invocation, result extraction, errors)
- manus_workflow() entry-point function (all 5 actions, validation, error handling)

The module under test imports `TOOL_SPEC` from strands_tools.workflow. The installed version
stores TOOL_SPEC on the decorated tool object (workflow.tool_spec) without a `properties`
sub-key inside `tasks.items`. We inject a compatible TOOL_SPEC before importing.
"""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Inject a compatible TOOL_SPEC into strands_tools.workflow BEFORE importing
# manus_agent.tools.manus_workflow. This must happen at module-collection time.
# ---------------------------------------------------------------------------

_INJECTED_TOOL_SPEC = {
    "name": "workflow",
    "description": "Advanced workflow orchestration tool.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["create", "start", "list", "status", "delete"]},
                "workflow_id": {"type": "string"},
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "task_id": {"type": "string"},
                            "description": {"type": "string"},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                            "priority": {"type": "integer"},
                        },
                    },
                },
            },
        }
    },
}

# Inject TOOL_SPEC into the already-loaded strands_tools.workflow module
import strands_tools.workflow as _stw  # noqa: E402

_stw.TOOL_SPEC = deepcopy(_INJECTED_TOOL_SPEC)

# Now import the module under test (agents are already importable from installed package)
from manus_agent.tools.manus_workflow import (  # noqa: E402
    TOOL_SPEC,
    ManusWorkflowManager,
    manus_workflow,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_agents():
    """Patch agent classes and Config used by ManusWorkflowManager."""
    from unittest.mock import patch

    with (
        patch("manus_agent.tools.manus_workflow.ManusAgent") as manus,
        patch("manus_agent.tools.manus_workflow.BrowserUseAgent") as browser,
        patch("manus_agent.tools.manus_workflow.DataAnalysisAgent") as data,
        patch("manus_agent.tools.manus_workflow.MCPAgent") as mcp,
        patch("manus_agent.tools.manus_workflow.Config") as cfg_cls,
    ):
        cfg_cls.from_file.return_value = MagicMock(name="config_instance")
        yield {
            "manus": manus,
            "browser": browser,
            "data_analysis": data,
            "mcp": mcp,
            "config_cls": cfg_cls,
        }


@pytest.fixture
def manager(mock_agents):
    """Create a ManusWorkflowManager instance with mocked agent dependencies."""
    from unittest.mock import patch

    with (
        patch("strands_tools.workflow.WorkflowManager._start_file_watching"),
        patch("strands_tools.workflow.WorkflowManager._load_all_workflows"),
    ):
        mgr = ManusWorkflowManager(
            {"system_prompt": "test", "messages": [], "tool_config": None, "inference_config": None}
        )
    return mgr


# ---------------------------------------------------------------------------
# TOOL_SPEC Tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Tests for the customized TOOL_SPEC."""

    def test_tool_spec_name(self):
        assert TOOL_SPEC["name"] == "manus_workflow"

    def test_tool_spec_description_mentions_manus_use(self):
        assert "ManusUse agent types" in TOOL_SPEC["description"]

    def test_tool_spec_description_lists_all_agent_types(self):
        desc = TOOL_SPEC["description"]
        assert "manus" in desc
        assert "browser" in desc
        assert "data_analysis" in desc
        assert "mcp" in desc

    def test_tool_spec_agent_type_enum(self):
        agent_type_schema = TOOL_SPEC["inputSchema"]["json"]["properties"]["tasks"]["items"]["properties"]["agent_type"]
        assert agent_type_schema["type"] == "string"
        assert set(agent_type_schema["enum"]) == {"manus", "browser", "data_analysis", "mcp"}

    def test_tool_spec_agent_type_default(self):
        agent_type_schema = TOOL_SPEC["inputSchema"]["json"]["properties"]["tasks"]["items"]["properties"]["agent_type"]
        assert agent_type_schema["default"] == "manus"

    def test_tool_spec_is_not_base_reference(self):
        """TOOL_SPEC should be a copy, not the base spec object."""
        assert TOOL_SPEC["name"] != _INJECTED_TOOL_SPEC["name"]


# ---------------------------------------------------------------------------
# ManusWorkflowManager.__init__ Tests
# ---------------------------------------------------------------------------


class TestManusWorkflowManagerInit:
    """Tests for ManusWorkflowManager initialization."""

    def test_agent_registry_has_four_types(self, manager):
        assert set(manager.agent_registry.keys()) == {"manus", "browser", "data_analysis", "mcp"}

    def test_agent_instances_cache_starts_empty(self, manager):
        assert manager.agent_instances == {}

    def test_config_is_loaded(self, manager):
        assert manager.config is not None

    def test_config_from_file_called(self, mock_agents):
        from unittest.mock import patch

        with (
            patch("strands_tools.workflow.WorkflowManager._start_file_watching"),
            patch("strands_tools.workflow.WorkflowManager._load_all_workflows"),
        ):
            ManusWorkflowManager({"system_prompt": None, "messages": [], "tool_config": None, "inference_config": None})
        mock_agents["config_cls"].from_file.assert_called_once()

    def test_inherits_from_workflow_manager(self):
        from strands_tools.workflow import WorkflowManager

        assert issubclass(ManusWorkflowManager, WorkflowManager)


# ---------------------------------------------------------------------------
# get_agent_for_task Tests
# ---------------------------------------------------------------------------


class TestGetAgentForTask:
    """Tests for agent routing and caching."""

    def test_default_agent_type_is_manus(self, manager, mock_agents):
        task = {"description": "test"}
        manager.get_agent_for_task(task)
        mock_agents["manus"].assert_called_once()

    def test_explicit_manus_agent_type(self, manager, mock_agents):
        task = {"agent_type": "manus", "description": "test"}
        manager.get_agent_for_task(task)
        mock_agents["manus"].assert_called_once()

    def test_browser_agent_type(self, manager, mock_agents):
        task = {"agent_type": "browser", "description": "test"}
        manager.get_agent_for_task(task)
        mock_agents["browser"].assert_called_once()

    def test_data_analysis_agent_type(self, manager, mock_agents):
        task = {"agent_type": "data_analysis", "description": "test"}
        manager.get_agent_for_task(task)
        mock_agents["data_analysis"].assert_called_once()

    def test_mcp_agent_type(self, manager, mock_agents):
        task = {"agent_type": "mcp", "description": "test"}
        manager.get_agent_for_task(task)
        mock_agents["mcp"].assert_called_once()

    def test_unknown_agent_type_falls_back_to_manus(self, manager, mock_agents):
        task = {"agent_type": "nonexistent_type", "description": "test"}
        manager.get_agent_for_task(task)
        mock_agents["manus"].assert_called_once()

    def test_cache_hit_returns_same_instance(self, manager, mock_agents):
        mock_inst = MagicMock(name="cached_agent")
        mock_agents["manus"].return_value = mock_inst

        task = {"agent_type": "manus", "description": "test"}
        first = manager.get_agent_for_task(task)
        second = manager.get_agent_for_task(task)

        assert first is second
        mock_agents["manus"].assert_called_once()

    def test_different_types_cached_separately(self, manager, mock_agents):
        mock_agents["manus"].return_value = MagicMock(name="manus_inst")
        mock_agents["browser"].return_value = MagicMock(name="browser_inst")

        a1 = manager.get_agent_for_task({"agent_type": "manus", "description": "a"})
        a2 = manager.get_agent_for_task({"agent_type": "browser", "description": "b"})

        assert a1 is not a2
        assert "manus" in manager.agent_instances
        assert "browser" in manager.agent_instances

    def test_system_prompt_passed_to_agent(self, manager, mock_agents):
        task = {"agent_type": "manus", "description": "test", "system_prompt": "Be concise"}
        manager.get_agent_for_task(task)
        mock_agents["manus"].assert_called_once_with(config=manager.config, system_prompt="Be concise")

    def test_no_system_prompt_calls_without_it(self, manager, mock_agents):
        task = {"agent_type": "manus", "description": "test"}
        manager.get_agent_for_task(task)
        mock_agents["manus"].assert_called_once_with(config=manager.config)

    def test_cache_stores_by_agent_type_key(self, manager, mock_agents):
        mock_agents["data_analysis"].return_value = MagicMock(name="da_inst")
        task = {"agent_type": "data_analysis", "description": "test"}
        agent = manager.get_agent_for_task(task)
        assert manager.agent_instances["data_analysis"] is agent

    def test_empty_string_agent_type_falls_back(self, manager, mock_agents):
        """Empty string is falsy in registry lookup, falls back to ManusAgent."""
        task = {"agent_type": "", "description": "test"}
        manager.get_agent_for_task(task)
        mock_agents["manus"].assert_called_once()


# ---------------------------------------------------------------------------
# execute_task Tests
# ---------------------------------------------------------------------------


class TestExecuteTask:
    """Tests for task execution logic."""

    def test_successful_execution_returns_success(self, manager, mock_agents):
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "done"}], "stop_reason": "end_turn"}
        mock_agents["manus"].return_value = mock_agent

        workflow = {"task_results": {}}
        task = {"task_id": "t1", "description": "Do something", "agent_type": "manus"}

        result = manager.execute_task(task, workflow, "tool-use-123")
        assert result["status"] == "success"
        assert result["content"] == [{"text": "done"}]
        assert result["toolUseId"] == "tool-use-123"

    def test_error_stop_reason_returns_error_status(self, manager, mock_agents):
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "failed"}], "stop_reason": "error"}
        mock_agents["manus"].return_value = mock_agent

        workflow = {"task_results": {}}
        task = {"task_id": "t1", "description": "Do something", "agent_type": "manus"}

        result = manager.execute_task(task, workflow, "tool-use-err")
        assert result["status"] == "error"

    def test_exception_returns_error_with_task_id(self, manager, mock_agents):
        mock_agent = MagicMock()
        mock_agent.side_effect = RuntimeError("Agent crashed")
        mock_agents["manus"].return_value = mock_agent

        workflow = {"task_results": {}}
        task = {"task_id": "task_abc", "description": "Do something", "agent_type": "manus"}

        result = manager.execute_task(task, workflow, "tool-use-exc")
        assert result["status"] == "error"
        assert "task_abc" in result["content"][0]["text"]
        assert "Agent crashed" in result["content"][0]["text"]

    def test_dependency_context_injected_into_prompt(self, manager, mock_agents):
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "ok"}], "stop_reason": "end_turn"}
        mock_agents["manus"].return_value = mock_agent

        workflow = {
            "task_results": {
                "dep1": {"status": "completed", "result": [{"text": "dep1 output"}]},
            }
        }
        task = {
            "task_id": "t2",
            "description": "Use dep results",
            "dependencies": ["dep1"],
            "agent_type": "manus",
        }

        manager.execute_task(task, workflow, "tu-dep")
        call_args = mock_agent.call_args[0][0]
        assert "dep1 output" in call_args
        assert "Use dep results" in call_args
        assert "Results from dep1" in call_args

    def test_skips_incomplete_dependencies(self, manager, mock_agents):
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "ok"}], "stop_reason": "end_turn"}
        mock_agents["manus"].return_value = mock_agent

        workflow = {
            "task_results": {
                "dep1": {"status": "pending", "result": None},
            }
        }
        task = {
            "task_id": "t2",
            "description": "Use dep results",
            "dependencies": ["dep1"],
            "agent_type": "manus",
        }

        manager.execute_task(task, workflow, "tu-skip")
        call_args = mock_agent.call_args[0][0]
        assert call_args == "Use dep results"

    def test_no_dependencies_just_uses_description(self, manager, mock_agents):
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "ok"}], "stop_reason": "end_turn"}
        mock_agents["manus"].return_value = mock_agent

        workflow = {"task_results": {}}
        task = {"task_id": "t1", "description": "Simple task", "agent_type": "manus"}

        manager.execute_task(task, workflow, "tu-simple")
        call_args = mock_agent.call_args[0][0]
        assert call_args == "Simple task"

    def test_result_object_with_content_attribute(self, manager, mock_agents):
        """Handle result as an object with .content and .stop_reason attributes."""
        result_obj = MagicMock()
        del result_obj.get  # Force AttributeError on .get()
        result_obj.content = [{"text": "object result"}]
        result_obj.stop_reason = "end_turn"

        mock_agent = MagicMock()
        mock_agent.return_value = result_obj
        mock_agents["manus"].return_value = mock_agent

        workflow = {"task_results": {}}
        task = {"task_id": "t1", "description": "test", "agent_type": "manus"}

        result = manager.execute_task(task, workflow, "tu-obj")
        assert result["status"] == "success"
        assert result["content"] == [{"text": "object result"}]

    def test_result_object_with_error_stop_reason(self, manager, mock_agents):
        result_obj = MagicMock()
        del result_obj.get
        result_obj.content = [{"text": "err"}]
        result_obj.stop_reason = "error"

        mock_agent = MagicMock()
        mock_agent.return_value = result_obj
        mock_agents["manus"].return_value = mock_agent

        workflow = {"task_results": {}}
        task = {"task_id": "t1", "description": "test", "agent_type": "manus"}

        result = manager.execute_task(task, workflow, "tu-obj-err")
        assert result["status"] == "error"

    def test_multiple_dependencies_all_injected(self, manager, mock_agents):
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "ok"}], "stop_reason": "end_turn"}
        mock_agents["manus"].return_value = mock_agent

        workflow = {
            "task_results": {
                "dep1": {"status": "completed", "result": [{"text": "result A"}]},
                "dep2": {"status": "completed", "result": [{"text": "result B"}]},
            }
        }
        task = {
            "task_id": "t3",
            "description": "Final task",
            "dependencies": ["dep1", "dep2"],
            "agent_type": "manus",
        }

        manager.execute_task(task, workflow, "tu-multi")
        call_args = mock_agent.call_args[0][0]
        assert "result A" in call_args
        assert "result B" in call_args
        assert "Final task" in call_args

    def test_dependency_with_none_result_skipped(self, manager, mock_agents):
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "ok"}], "stop_reason": "end_turn"}
        mock_agents["manus"].return_value = mock_agent

        workflow = {
            "task_results": {
                "dep1": {"status": "completed", "result": None},
            }
        }
        task = {
            "task_id": "t2",
            "description": "Use deps",
            "dependencies": ["dep1"],
            "agent_type": "manus",
        }

        manager.execute_task(task, workflow, "tu-none-res")
        call_args = mock_agent.call_args[0][0]
        assert call_args == "Use deps"

    def test_empty_dependencies_list(self, manager, mock_agents):
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "ok"}], "stop_reason": "end_turn"}
        mock_agents["manus"].return_value = mock_agent

        workflow = {"task_results": {}}
        task = {"task_id": "t1", "description": "test", "dependencies": [], "agent_type": "manus"}

        manager.execute_task(task, workflow, "tu-empty-deps")
        call_args = mock_agent.call_args[0][0]
        assert call_args == "test"

    def test_dependency_result_with_missing_text_key(self, manager, mock_agents):
        """Messages without 'text' key produce empty strings via .get('text', '')."""
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "ok"}], "stop_reason": "end_turn"}
        mock_agents["manus"].return_value = mock_agent

        workflow = {
            "task_results": {
                "dep1": {"status": "completed", "result": [{"image": "data:..."}]},
            }
        }
        task = {
            "task_id": "t2",
            "description": "Follow up",
            "dependencies": ["dep1"],
            "agent_type": "manus",
        }

        manager.execute_task(task, workflow, "tu-notext")
        call_args = mock_agent.call_args[0][0]
        assert "Results from dep1" in call_args
        assert "Follow up" in call_args

    def test_result_dict_with_empty_content(self, manager, mock_agents):
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [], "stop_reason": "end_turn"}
        mock_agents["manus"].return_value = mock_agent

        workflow = {"task_results": {}}
        task = {"task_id": "t1", "description": "test", "agent_type": "manus"}

        result = manager.execute_task(task, workflow, "tu-empty-content")
        assert result["status"] == "success"
        assert result["content"] == []

    def test_result_dict_missing_content_key(self, manager, mock_agents):
        mock_agent = MagicMock()
        mock_agent.return_value = {"stop_reason": "end_turn"}
        mock_agents["manus"].return_value = mock_agent

        workflow = {"task_results": {}}
        task = {"task_id": "t1", "description": "test", "agent_type": "manus"}

        result = manager.execute_task(task, workflow, "tu-no-content")
        assert result["status"] == "success"
        assert result["content"] == []

    def test_result_dict_missing_stop_reason_defaults_to_success(self, manager, mock_agents):
        mock_agent = MagicMock()
        mock_agent.return_value = {"content": [{"text": "hi"}]}
        mock_agents["manus"].return_value = mock_agent

        workflow = {"task_results": {}}
        task = {"task_id": "t1", "description": "test", "agent_type": "manus"}

        result = manager.execute_task(task, workflow, "tu-no-stop")
        # Empty string != "error", so status is success
        assert result["status"] == "success"

    def test_routes_to_browser_agent(self, manager, mock_agents):
        mock_browser_agent = MagicMock()
        mock_browser_agent.return_value = {"content": [{"text": "browsed"}], "stop_reason": "end_turn"}
        mock_agents["browser"].return_value = mock_browser_agent

        workflow = {"task_results": {}}
        task = {"task_id": "b1", "description": "Browse", "agent_type": "browser"}

        result = manager.execute_task(task, workflow, "tu-browser")
        assert result["status"] == "success"
        mock_browser_agent.assert_called_once_with("Browse")

    def test_routes_to_data_analysis_agent(self, manager, mock_agents):
        mock_da = MagicMock()
        mock_da.return_value = {"content": [{"text": "analyzed"}], "stop_reason": "end_turn"}
        mock_agents["data_analysis"].return_value = mock_da

        workflow = {"task_results": {}}
        task = {"task_id": "d1", "description": "Analyze", "agent_type": "data_analysis"}

        result = manager.execute_task(task, workflow, "tu-da")
        assert result["status"] == "success"
        mock_da.assert_called_once_with("Analyze")

    def test_routes_to_mcp_agent(self, manager, mock_agents):
        mock_mcp = MagicMock()
        mock_mcp.return_value = {"content": [{"text": "mcp done"}], "stop_reason": "end_turn"}
        mock_agents["mcp"].return_value = mock_mcp

        workflow = {"task_results": {}}
        task = {"task_id": "m1", "description": "MCP task", "agent_type": "mcp"}

        result = manager.execute_task(task, workflow, "tu-mcp")
        assert result["status"] == "success"
        mock_mcp.assert_called_once_with("MCP task")


# ---------------------------------------------------------------------------
# manus_workflow() Entry-Point Tests
# ---------------------------------------------------------------------------


class TestManusWorkflowEntryPoint:
    """Tests for the manus_workflow() function."""

    @pytest.fixture(autouse=True)
    def _patch_manager(self):
        from unittest.mock import patch

        with (
            patch("manus_agent.tools.manus_workflow.ManusWorkflowManager") as mock_cls,
            patch("builtins.print"),
        ):
            self.mock_manager = MagicMock()
            mock_cls.return_value = self.mock_manager
            yield

    def _call(self, action, tool_use_id="tu-test", **extra):
        tool = {"toolUseId": tool_use_id, "input": {"action": action, **extra}}
        return manus_workflow(tool)

    # --- create ---

    def test_create_calls_create_workflow(self):
        self.mock_manager.create_workflow.return_value = {"status": "success", "content": [{"text": "Created"}]}
        result = self._call("create", workflow_id="wf1", tasks=[{"task_id": "t1", "description": "d"}])
        self.mock_manager.create_workflow.assert_called_once()
        assert result["status"] == "success"

    def test_create_without_tasks_returns_error(self):
        result = self._call("create", workflow_id="wf1")
        assert result["status"] == "error"
        assert "Tasks are required" in result["content"][0]["text"]

    def test_create_with_empty_tasks_returns_error(self):
        result = self._call("create", workflow_id="wf1", tasks=[])
        assert result["status"] == "error"
        assert "Tasks are required" in result["content"][0]["text"]

    def test_create_generates_uuid_when_no_workflow_id(self):
        self.mock_manager.create_workflow.return_value = {"status": "success", "content": [{"text": "ok"}]}
        result = self._call("create", tasks=[{"task_id": "t1", "description": "d"}])
        assert result["status"] == "success"
        # The workflow_id passed to create_workflow should be a UUID string
        import uuid

        call_args = self.mock_manager.create_workflow.call_args[0]
        uuid.UUID(call_args[0])  # should not raise

    # --- start ---

    def test_start_calls_start_workflow(self):
        self.mock_manager.start_workflow.return_value = {"status": "success", "content": [{"text": "Started"}]}
        result = self._call("start", workflow_id="wf1")
        self.mock_manager.start_workflow.assert_called_once_with("wf1", "tu-test")
        assert result["status"] == "success"

    def test_start_without_workflow_id_returns_error(self):
        result = self._call("start")
        assert result["status"] == "error"
        assert "workflow_id is required" in result["content"][0]["text"]

    # --- list ---

    def test_list_calls_list_workflows(self):
        self.mock_manager.list_workflows.return_value = {"status": "success", "content": [{"text": "[]"}]}
        result = self._call("list")
        self.mock_manager.list_workflows.assert_called_once_with("tu-test")
        assert result["status"] == "success"

    # --- status ---

    def test_status_calls_get_workflow_status(self):
        self.mock_manager.get_workflow_status.return_value = {"status": "success", "content": [{"text": "running"}]}
        result = self._call("status", workflow_id="wf1")
        self.mock_manager.get_workflow_status.assert_called_once_with("wf1", "tu-test")
        assert result["status"] == "success"

    def test_status_without_workflow_id_returns_error(self):
        result = self._call("status")
        assert result["status"] == "error"
        assert "workflow_id is required" in result["content"][0]["text"]

    # --- delete ---

    def test_delete_calls_delete_workflow(self):
        self.mock_manager.delete_workflow.return_value = {"status": "success", "content": [{"text": "Deleted"}]}
        result = self._call("delete", workflow_id="wf1")
        self.mock_manager.delete_workflow.assert_called_once_with("wf1", "tu-test")
        assert result["status"] == "success"

    def test_delete_without_workflow_id_returns_error(self):
        result = self._call("delete")
        assert result["status"] == "error"
        assert "workflow_id is required" in result["content"][0]["text"]

    # --- unknown action ---

    def test_unknown_action_returns_error(self):
        result = self._call("invalid_action")
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_none_action_returns_error(self):
        tool = {"toolUseId": "tu-none", "input": {}}
        with __import__("unittest.mock", fromlist=["patch"]).patch("builtins.print"):
            result = manus_workflow(tool)
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    # --- exception handling ---

    def test_exception_in_manager_returns_error_with_traceback(self):
        self.mock_manager.start_workflow.side_effect = RuntimeError("kaboom")
        result = self._call("start", workflow_id="wf1")
        assert result["status"] == "error"
        assert "kaboom" in result["content"][0]["text"]
        assert "Traceback" in result["content"][0]["text"]

    # --- toolUseId handling ---

    def test_missing_tool_use_id_generates_uuid(self):
        import uuid

        self.mock_manager.list_workflows.return_value = {"status": "success", "content": [{"text": "[]"}]}
        tool = {"input": {"action": "list"}}
        result = manus_workflow(tool)
        uuid.UUID(result["toolUseId"])  # should not raise

    def test_explicit_tool_use_id_preserved(self):
        self.mock_manager.list_workflows.return_value = {"status": "success", "content": [{"text": "[]"}]}
        result = self._call("list", tool_use_id="my-custom-id")
        assert result["toolUseId"] == "my-custom-id"

    # --- kwargs ---

    def test_kwargs_forwarded_to_manager_context(self):
        from unittest.mock import patch

        with (
            patch("manus_agent.tools.manus_workflow.ManusWorkflowManager") as mock_cls,
            patch("builtins.print"),
        ):
            mock_inst = MagicMock()
            mock_inst.list_workflows.return_value = {"status": "success", "content": [{"text": "[]"}]}
            mock_cls.return_value = mock_inst

            tool = {"toolUseId": "tu-kw", "input": {"action": "list"}}
            manus_workflow(
                tool,
                system_prompt="custom",
                inference_config={"x": 1},
                messages=[{"role": "user"}],
                tool_config={"tools": []},
            )
            mock_cls.assert_called_once_with(
                {
                    "system_prompt": "custom",
                    "inference_config": {"x": 1},
                    "messages": [{"role": "user"}],
                    "tool_config": {"tools": []},
                }
            )
