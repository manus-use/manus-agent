"""Comprehensive test suite for the workflow_tool module.

Tests cover:
- ManusWorkflowManager.__init__ (agent registry, config, caching)
- ManusWorkflowManager.get_agent_for_task (routing, caching, fallback)
- ManusWorkflowManager.execute_task (context building, result normalization, async handling)
- ManusWorkflowManager.create_workflow (creation, defaults, store failure)
- workflow_tool() entry point (all actions: create, start, list, status, delete, unknown)
- Error/edge cases throughout
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODULE = "manus_agent.tools.workflow_tool"


@pytest.fixture
def mock_config():
    """Mock Config.from_file() to avoid filesystem dependency."""
    with patch(f"{MODULE}.Config") as mock_cfg_cls:
        mock_cfg_cls.from_file.return_value = MagicMock(name="MockConfig")
        yield mock_cfg_cls


@pytest.fixture
def mock_agents():
    """Mock all agent classes imported in workflow_tool."""
    with (
        patch(f"{MODULE}.ManusAgent") as mock_manus,
        patch(f"{MODULE}.BrowserUseAgent") as mock_browser,
        patch(f"{MODULE}.DataAnalysisAgent") as mock_data,
        patch(f"{MODULE}.MCPAgent") as mock_mcp,
    ):
        # Each agent class returns a callable instance
        for cls in (mock_manus, mock_browser, mock_data, mock_mcp):
            cls.return_value = MagicMock(name=f"{cls._mock_name}_instance")
        yield {
            "manus": mock_manus,
            "browser": mock_browser,
            "data_analysis": mock_data,
            "mcp": mock_mcp,
        }


@pytest.fixture
def tool_context():
    """Standard tool context dict."""
    return {
        "system_prompt": "test prompt",
        "inference_config": {"maxTokens": 1000},
        "messages": [],
        "tool_config": {},
    }


@pytest.fixture
def manager(mock_config, mock_agents, tool_context):
    """Instantiate ManusWorkflowManager with mocked dependencies."""
    with patch(f"{MODULE}.WorkflowManager.__init__", return_value=None):
        from manus_agent.tools.workflow_tool import ManusWorkflowManager

        mgr = ManusWorkflowManager(tool_context)
        # Inject storage mock
        mgr._workflows = {}
        return mgr


# ===========================================================================
# ManusWorkflowManager.__init__
# ===========================================================================


class TestManusWorkflowManagerInit:
    """Tests for __init__."""

    def test_init_creates_agent_registry(self, manager):
        """Registry contains all four agent types."""
        assert set(manager.agent_registry.keys()) == {"manus", "browser", "data_analysis", "mcp"}

    def test_init_empty_agent_cache(self, manager):
        """Agent instance cache starts empty."""
        assert manager.agent_instances == {}

    def test_init_loads_config(self, mock_config, mock_agents, tool_context):
        """Config.from_file() is called during init."""
        with patch(f"{MODULE}.WorkflowManager.__init__", return_value=None):
            from manus_agent.tools.workflow_tool import ManusWorkflowManager

            ManusWorkflowManager(tool_context)
            mock_config.from_file.assert_called_once()


# ===========================================================================
# ManusWorkflowManager.get_agent_for_task
# ===========================================================================


class TestGetAgentForTask:
    """Tests for get_agent_for_task."""

    def test_default_agent_type_is_manus(self, manager, mock_agents):
        """No agent_type key defaults to 'manus'."""
        task = {"description": "test"}
        agent = manager.get_agent_for_task(task)
        mock_agents["manus"].assert_called_once()
        assert agent is mock_agents["manus"].return_value

    def test_explicit_manus(self, manager, mock_agents):
        """Explicit 'manus' routes to ManusAgent."""
        task = {"agent_type": "manus"}
        manager.get_agent_for_task(task)
        mock_agents["manus"].assert_called_once()

    def test_browser_agent(self, manager, mock_agents):
        """'browser' routes to BrowserUseAgent."""
        task = {"agent_type": "browser"}
        agent = manager.get_agent_for_task(task)
        mock_agents["browser"].assert_called_once()
        assert agent is mock_agents["browser"].return_value

    def test_data_analysis_agent(self, manager, mock_agents):
        """'data_analysis' routes to DataAnalysisAgent."""
        task = {"agent_type": "data_analysis"}
        agent = manager.get_agent_for_task(task)
        mock_agents["data_analysis"].assert_called_once()
        assert agent is mock_agents["data_analysis"].return_value

    def test_mcp_agent(self, manager, mock_agents):
        """'mcp' routes to MCPAgent."""
        task = {"agent_type": "mcp"}
        agent = manager.get_agent_for_task(task)
        mock_agents["mcp"].assert_called_once()
        assert agent is mock_agents["mcp"].return_value

    def test_unknown_agent_type_falls_back_to_manus(self, manager, mock_agents):
        """Unknown agent_type falls back to ManusAgent with a warning."""
        task = {"agent_type": "nonexistent"}
        agent = manager.get_agent_for_task(task)
        mock_agents["manus"].assert_called_once()
        assert agent is mock_agents["manus"].return_value

    def test_caching_returns_same_instance(self, manager, mock_agents):
        """Second call with same agent_type returns cached instance."""
        task = {"agent_type": "browser"}
        agent1 = manager.get_agent_for_task(task)
        agent2 = manager.get_agent_for_task(task)
        assert agent1 is agent2
        # Only one instantiation
        mock_agents["browser"].assert_called_once()

    def test_system_prompt_passed_to_agent(self, manager, mock_agents):
        """system_prompt in task is passed to agent constructor."""
        task = {"agent_type": "manus", "system_prompt": "custom prompt"}
        manager.get_agent_for_task(task)
        mock_agents["manus"].assert_called_once_with(config=manager.config, system_prompt="custom prompt")

    def test_no_system_prompt_calls_without_it(self, manager, mock_agents):
        """No system_prompt calls agent without the keyword."""
        task = {"agent_type": "manus"}
        manager.get_agent_for_task(task)
        mock_agents["manus"].assert_called_once_with(config=manager.config)


# ===========================================================================
# ManusWorkflowManager.execute_task
# ===========================================================================


class TestExecuteTask:
    """Tests for execute_task."""

    def _make_task(self, task_id="t1", description="do stuff", deps=None, agent_type="manus"):
        return {
            "task_id": task_id,
            "description": description,
            "dependencies": deps or [],
            "agent_type": agent_type,
        }

    def _make_workflow(self, task_results=None):
        return {"task_results": task_results or {}}

    def test_string_result(self, manager, mock_agents):
        """Agent returning a plain string is wrapped in content list."""
        mock_agents["manus"].return_value.return_value = "hello world"
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["status"] == "success"
        assert result["content"] == [{"text": "hello world"}]
        assert result["toolUseId"] == "tool-123"

    def test_none_result(self, manager, mock_agents):
        """Agent returning None produces empty content."""
        mock_agents["manus"].return_value.return_value = None
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["status"] == "success"
        assert result["content"] == []

    def test_dict_result_with_content_list(self, manager, mock_agents):
        """Dict result with content list is extracted properly."""
        mock_agents["manus"].return_value.return_value = {
            "content": [{"text": "analysis done"}],
            "stop_reason": "completed",
        }
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["status"] == "success"
        assert result["content"] == [{"text": "analysis done"}]

    def test_dict_result_with_content_string(self, manager, mock_agents):
        """Dict result where content is a string."""
        mock_agents["manus"].return_value.return_value = {
            "content": "just text",
            "stop_reason": "completed",
        }
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["content"] == [{"text": "just text"}]

    def test_dict_result_with_content_none(self, manager, mock_agents):
        """Dict result where content is None."""
        mock_agents["manus"].return_value.return_value = {
            "content": None,
            "stop_reason": "completed",
        }
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["content"] == []

    def test_dict_result_with_content_other_type(self, manager, mock_agents):
        """Dict result where content is an unexpected type."""
        mock_agents["manus"].return_value.return_value = {
            "content": 42,
            "stop_reason": "completed",
        }
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["content"] == [{"text": "42"}]

    def test_dict_result_error_stop_reason(self, manager, mock_agents):
        """Dict result with stop_reason='error' sets status='error'."""
        mock_agents["manus"].return_value.return_value = {
            "content": [{"text": "failed"}],
            "stop_reason": "error",
        }
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["status"] == "error"

    def test_object_result_with_content_attr_list(self, manager, mock_agents):
        """Object result with .content list attribute."""
        obj = MagicMock()
        obj.get = MagicMock(side_effect=AttributeError)  # Not dict-like
        obj.content = [{"text": "from object"}]
        obj.stop_reason = "completed"
        # Make hasattr work but .get raises
        del obj.get
        mock_agents["manus"].return_value.return_value = obj
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["status"] == "success"
        assert result["content"] == [{"text": "from object"}]

    def test_object_result_with_content_attr_string(self, manager, mock_agents):
        """Object result with .content as a string."""
        obj = MagicMock(spec=[])
        obj.content = "string content"
        obj.stop_reason = "completed"
        mock_agents["manus"].return_value.return_value = obj
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["content"] == [{"text": "string content"}]

    def test_object_result_with_content_attr_none(self, manager, mock_agents):
        """Object result with .content = None."""
        obj = MagicMock(spec=[])
        obj.content = None
        obj.stop_reason = "completed"
        mock_agents["manus"].return_value.return_value = obj
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["content"] == []

    def test_unexpected_result_type_fallback(self, manager, mock_agents):
        """Unexpected result type (e.g., int) is stringified."""
        mock_agents["manus"].return_value.return_value = 12345
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["content"] == [{"text": "12345"}]

    def test_dependency_context_included(self, manager, mock_agents):
        """Dependent task results are injected into prompt."""
        mock_agents["manus"].return_value.return_value = "ok"

        workflow = self._make_workflow(
            task_results={
                "dep1": {
                    "status": "completed",
                    "result": [{"text": "dep1 output"}],
                }
            }
        )
        task = self._make_task(task_id="t2", description="follow up", deps=["dep1"])
        manager.execute_task(task, workflow, "tool-123")

        # Verify the agent was called with context-enriched prompt
        call_args = mock_agents["manus"].return_value.call_args[0][0]
        assert "Results from dep1:" in call_args
        assert "dep1 output" in call_args
        assert "follow up" in call_args

    def test_dependency_not_completed_excluded(self, manager, mock_agents):
        """Non-completed dependencies are NOT included in context."""
        mock_agents["manus"].return_value.return_value = "ok"

        workflow = self._make_workflow(task_results={"dep1": {"status": "pending", "result": None}})
        task = self._make_task(task_id="t2", description="follow up", deps=["dep1"])
        manager.execute_task(task, workflow, "tool-123")

        call_args = mock_agents["manus"].return_value.call_args[0][0]
        assert "Results from dep1:" not in call_args

    def test_dependency_no_result_excluded(self, manager, mock_agents):
        """Completed dependency with no result field is excluded."""
        mock_agents["manus"].return_value.return_value = "ok"

        workflow = self._make_workflow(task_results={"dep1": {"status": "completed", "result": None}})
        task = self._make_task(task_id="t2", description="follow up", deps=["dep1"])
        manager.execute_task(task, workflow, "tool-123")

        call_args = mock_agents["manus"].return_value.call_args[0][0]
        assert "Results from dep1:" not in call_args

    def test_no_dependencies_key(self, manager, mock_agents):
        """Task without dependencies key executes normally."""
        mock_agents["manus"].return_value.return_value = "ok"
        task = {"task_id": "t1", "description": "no deps"}
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["status"] == "success"

    def test_exception_in_execution(self, manager, mock_agents):
        """Exception during agent call returns error result."""
        mock_agents["manus"].return_value.side_effect = RuntimeError("agent crashed")
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["status"] == "error"
        assert "agent crashed" in result["content"][0]["text"]

    def test_coroutine_result_with_running_loop(self, manager, mock_agents):
        """Agent returning a coroutine is awaited (RuntimeError path → asyncio.run)."""

        async def async_result():
            return "async output"

        mock_agents["manus"].return_value.return_value = async_result()
        task = self._make_task()

        # Since there's no running event loop in tests, it should fall through to asyncio.run()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["status"] == "success"
        assert result["content"] == [{"text": "async output"}]

    def test_coroutine_result_returning_dict(self, manager, mock_agents):
        """Async agent returning a dict result."""

        async def async_dict():
            return {"content": [{"text": "async dict"}], "stop_reason": "completed"}

        mock_agents["manus"].return_value.return_value = async_dict()
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["content"] == [{"text": "async dict"}]

    def test_coroutine_result_returning_none(self, manager, mock_agents):
        """Async agent returning None."""

        async def async_none():
            return None

        mock_agents["manus"].return_value.return_value = async_none()
        task = self._make_task()
        result = manager.execute_task(task, self._make_workflow(), "tool-123")
        assert result["content"] == []


# ===========================================================================
# ManusWorkflowManager.create_workflow
# ===========================================================================


class TestCreateWorkflow:
    """Tests for create_workflow."""

    def test_create_basic_workflow(self, manager):
        """Create a workflow with basic tasks succeeds."""
        tasks = [
            {"task_id": "t1", "description": "task one"},
            {"task_id": "t2", "description": "task two"},
        ]
        with patch.object(manager, "store_workflow", return_value={"status": "success"}):
            result = manager.create_workflow("wf-1", tasks, "tool-use-id")
        assert result["status"] == "success"
        assert "2 tasks" in result["content"][0]["text"]

    def test_create_adds_default_priority(self, manager):
        """Tasks without priority get default priority=3."""
        tasks = [{"task_id": "t1", "description": "task one"}]
        stored = {}

        def capture_store(wf_id, wf_data, *_):
            stored["data"] = wf_data
            return {"status": "success"}

        with patch.object(manager, "store_workflow", side_effect=capture_store):
            manager.create_workflow("wf-1", tasks, "tool-use-id")

        assert stored["data"]["tasks"][0]["priority"] == 3

    def test_create_generates_uuid_if_empty(self, manager):
        """Empty workflow_id generates a UUID."""
        tasks = [{"task_id": "t1", "description": "task one"}]
        stored = {}

        def capture_store(wf_id, wf_data, *_):
            stored["id"] = wf_id
            return {"status": "success"}

        with patch.object(manager, "store_workflow", side_effect=capture_store):
            manager.create_workflow("", tasks, "tool-use-id")

        # Should be a valid UUID
        uuid.UUID(stored["id"])

    def test_create_store_failure(self, manager):
        """Store failure propagates as error."""
        tasks = [{"task_id": "t1", "description": "task one"}]
        with patch.object(manager, "store_workflow", return_value={"status": "error", "error": "disk full"}):
            result = manager.create_workflow("wf-1", tasks, "tool-use-id")
        assert result["status"] == "error"
        assert "disk full" in result["content"][0]["text"]

    def test_create_exception_handling(self, manager):
        """Exception in create_workflow returns error."""
        with patch.object(manager, "store_workflow", side_effect=RuntimeError("boom")):
            tasks = [{"task_id": "t1", "description": "task one"}]
            result = manager.create_workflow("wf-1", tasks, "tool-use-id")
        assert result["status"] == "error"
        assert "boom" in result["content"][0]["text"]

    def test_create_workflow_sets_created_status(self, manager):
        """Created workflow has status='created'."""
        tasks = [{"task_id": "t1", "description": "task one"}]
        stored = {}

        def capture_store(wf_id, wf_data, *_):
            stored["data"] = wf_data
            return {"status": "success"}

        with patch.object(manager, "store_workflow", side_effect=capture_store):
            manager.create_workflow("wf-1", tasks, "tool-use-id")

        assert stored["data"]["status"] == "created"

    def test_create_workflow_task_results_initialized(self, manager):
        """task_results are initialized with pending status."""
        tasks = [
            {"task_id": "t1", "description": "one"},
            {"task_id": "t2", "description": "two"},
        ]
        stored = {}

        def capture_store(wf_id, wf_data, *_):
            stored["data"] = wf_data
            return {"status": "success"}

        with patch.object(manager, "store_workflow", side_effect=capture_store):
            manager.create_workflow("wf-1", tasks, "tool-use-id")

        assert stored["data"]["task_results"]["t1"]["status"] == "pending"
        assert stored["data"]["task_results"]["t2"]["status"] == "pending"


# ===========================================================================
# workflow_tool() entry point
# ===========================================================================


class TestWorkflowToolEntryPoint:
    """Tests for the workflow_tool() function."""

    @pytest.fixture(autouse=True)
    def patch_manager(self):
        """Patch ManusWorkflowManager to avoid real instantiation."""
        with patch(f"{MODULE}.ManusWorkflowManager") as mock_cls:
            self.mock_manager = MagicMock()
            mock_cls.return_value = self.mock_manager
            yield

    def _make_tool(self, action, **extra):
        tool_input = {"action": action, **extra}
        return {"toolUseId": "tu-1", "input": tool_input}

    def test_create_action_success(self):
        """Create action calls create_workflow and returns result."""
        from manus_agent.tools.workflow_tool import workflow_tool

        self.mock_manager.create_workflow.return_value = {
            "status": "success",
            "content": [{"text": "Created"}],
        }
        tool = self._make_tool(
            "create",
            tasks=[{"task_id": "t1", "description": "test"}],
        )
        result = workflow_tool(tool)
        assert result["status"] == "success"
        assert result["toolUseId"] == "tu-1"
        self.mock_manager.create_workflow.assert_called_once()

    def test_create_action_no_tasks(self):
        """Create action without tasks returns error."""
        from manus_agent.tools.workflow_tool import workflow_tool

        tool = self._make_tool("create")
        result = workflow_tool(tool)
        assert result["status"] == "error"
        assert "Tasks are required" in result["content"][0]["text"]

    def test_create_action_generates_workflow_id(self):
        """Create action generates workflow_id if not provided."""
        from manus_agent.tools.workflow_tool import workflow_tool

        self.mock_manager.create_workflow.return_value = {
            "status": "success",
            "content": [{"text": "ok"}],
        }
        tool = self._make_tool(
            "create",
            tasks=[{"task_id": "t1", "description": "test"}],
        )
        workflow_tool(tool)
        # First arg to create_workflow should be a UUID string
        call_args = self.mock_manager.create_workflow.call_args[0]
        uuid.UUID(call_args[0])  # Should not raise

    def test_start_action_success(self):
        """Start action calls start_workflow."""
        from manus_agent.tools.workflow_tool import workflow_tool

        self.mock_manager.start_workflow.return_value = {
            "status": "success",
            "content": [{"text": "Started"}],
        }
        tool = self._make_tool("start", workflow_id="wf-1")
        result = workflow_tool(tool)
        assert result["status"] == "success"
        self.mock_manager.start_workflow.assert_called_once_with("wf-1", "tu-1")

    def test_start_action_no_workflow_id(self):
        """Start action without workflow_id returns error."""
        from manus_agent.tools.workflow_tool import workflow_tool

        tool = self._make_tool("start")
        result = workflow_tool(tool)
        assert result["status"] == "error"
        assert "workflow_id is required" in result["content"][0]["text"]

    def test_list_action(self):
        """List action calls list_workflows."""
        from manus_agent.tools.workflow_tool import workflow_tool

        self.mock_manager.list_workflows.return_value = {
            "status": "success",
            "content": [{"text": "[]"}],
        }
        tool = self._make_tool("list")
        result = workflow_tool(tool)
        assert result["status"] == "success"
        self.mock_manager.list_workflows.assert_called_once_with("tu-1")

    def test_status_action_success(self):
        """Status action calls get_workflow_status."""
        from manus_agent.tools.workflow_tool import workflow_tool

        self.mock_manager.get_workflow_status.return_value = {
            "status": "success",
            "content": [{"text": "running"}],
        }
        tool = self._make_tool("status", workflow_id="wf-1")
        result = workflow_tool(tool)
        assert result["status"] == "success"
        self.mock_manager.get_workflow_status.assert_called_once_with("wf-1", "tu-1")

    def test_status_action_no_workflow_id(self):
        """Status action without workflow_id returns error."""
        from manus_agent.tools.workflow_tool import workflow_tool

        tool = self._make_tool("status")
        result = workflow_tool(tool)
        assert result["status"] == "error"
        assert "workflow_id is required" in result["content"][0]["text"]

    def test_delete_action_success(self):
        """Delete action calls delete_workflow."""
        from manus_agent.tools.workflow_tool import workflow_tool

        self.mock_manager.delete_workflow.return_value = {
            "status": "success",
            "content": [{"text": "Deleted"}],
        }
        tool = self._make_tool("delete", workflow_id="wf-1")
        result = workflow_tool(tool)
        assert result["status"] == "success"
        self.mock_manager.delete_workflow.assert_called_once_with("wf-1", "tu-1")

    def test_delete_action_no_workflow_id(self):
        """Delete action without workflow_id returns error."""
        from manus_agent.tools.workflow_tool import workflow_tool

        tool = self._make_tool("delete")
        result = workflow_tool(tool)
        assert result["status"] == "error"
        assert "workflow_id is required" in result["content"][0]["text"]

    def test_unknown_action(self):
        """Unknown action returns error."""
        from manus_agent.tools.workflow_tool import workflow_tool

        tool = self._make_tool("nope")
        result = workflow_tool(tool)
        assert result["status"] == "error"
        assert "Unknown action: nope" in result["content"][0]["text"]

    def test_missing_tool_use_id_generates_uuid(self):
        """Missing toolUseId generates a UUID."""
        from manus_agent.tools.workflow_tool import workflow_tool

        self.mock_manager.list_workflows.return_value = {
            "status": "success",
            "content": [{"text": "ok"}],
        }
        tool = {"input": {"action": "list"}}
        result = workflow_tool(tool)
        # Should have a valid UUID as toolUseId
        uuid.UUID(result["toolUseId"])

    def test_exception_in_entry_point(self):
        """Top-level exception in workflow_tool is caught and returns error."""
        from manus_agent.tools.workflow_tool import workflow_tool

        self.mock_manager.list_workflows.side_effect = RuntimeError("unexpected")
        tool = self._make_tool("list")
        result = workflow_tool(tool)
        assert result["status"] == "error"
        assert "unexpected" in result["content"][0]["text"]

    def test_kwargs_passed_to_manager(self):
        """system_prompt and other kwargs are passed to manager context."""
        with patch(f"{MODULE}.ManusWorkflowManager") as mock_cls:
            mock_mgr = MagicMock()
            mock_mgr.list_workflows.return_value = {
                "status": "success",
                "content": [{"text": "ok"}],
            }
            mock_cls.return_value = mock_mgr

            from manus_agent.tools.workflow_tool import workflow_tool

            tool = self._make_tool("list")
            workflow_tool(
                tool,
                system_prompt="sp",
                inference_config={"maxTokens": 500},
                messages=[{"role": "user"}],
                tool_config={"tools": []},
            )

            # Verify the context passed to ManusWorkflowManager
            ctx = mock_cls.call_args[0][0]
            assert ctx["system_prompt"] == "sp"
            assert ctx["inference_config"] == {"maxTokens": 500}


# ===========================================================================
# TOOL_SPEC validation
# ===========================================================================


class TestToolSpec:
    """Tests for TOOL_SPEC definition."""

    def test_tool_spec_name(self):
        """TOOL_SPEC has correct name."""
        from manus_agent.tools.workflow_tool import TOOL_SPEC

        assert TOOL_SPEC["name"] == "workflow_tool"

    def test_tool_spec_has_description(self):
        """TOOL_SPEC has a non-empty description."""
        from manus_agent.tools.workflow_tool import TOOL_SPEC

        assert len(TOOL_SPEC["description"]) > 10

    def test_tool_spec_input_schema_has_enum(self):
        """TOOL_SPEC inputSchema defines agent type enum."""
        from manus_agent.tools.workflow_tool import TOOL_SPEC

        schema = TOOL_SPEC["inputSchema"]
        assert "manus" in schema["enum"]
        assert "browser" in schema["enum"]
        assert "data_analysis" in schema["enum"]
        assert "mcp" in schema["enum"]


# ===========================================================================
# Edge cases / integration scenarios
# ===========================================================================


class TestEdgeCases:
    """Edge case and integration tests."""

    def test_multiple_dependencies_context(self, manager, mock_agents):
        """Multiple completed dependencies all contribute context."""
        mock_agents["manus"].return_value.return_value = "done"
        workflow = {
            "task_results": {
                "dep1": {"status": "completed", "result": [{"text": "output1"}]},
                "dep2": {"status": "completed", "result": [{"text": "output2"}]},
            }
        }
        task = {
            "task_id": "t3",
            "description": "combine",
            "dependencies": ["dep1", "dep2"],
            "agent_type": "manus",
        }
        manager.execute_task(task, workflow, "tu-1")
        call_args = mock_agents["manus"].return_value.call_args[0][0]
        assert "output1" in call_args
        assert "output2" in call_args

    def test_dependency_with_multiple_result_items(self, manager, mock_agents):
        """Dependency with multiple result texts joins them."""
        mock_agents["manus"].return_value.return_value = "done"
        workflow = {
            "task_results": {
                "dep1": {
                    "status": "completed",
                    "result": [{"text": "line1"}, {"text": "line2"}],
                }
            }
        }
        task = {
            "task_id": "t2",
            "description": "join",
            "dependencies": ["dep1"],
            "agent_type": "manus",
        }
        manager.execute_task(task, workflow, "tu-1")
        call_args = mock_agents["manus"].return_value.call_args[0][0]
        assert "line1" in call_args
        assert "line2" in call_args

    def test_dependency_missing_from_workflow(self, manager, mock_agents):
        """Missing dependency key in task_results doesn't crash."""
        mock_agents["manus"].return_value.return_value = "ok"
        workflow = {"task_results": {}}
        task = {
            "task_id": "t1",
            "description": "test",
            "dependencies": ["nonexistent"],
            "agent_type": "manus",
        }
        result = manager.execute_task(task, workflow, "tu-1")
        assert result["status"] == "success"

    def test_empty_description_task(self, manager, mock_agents):
        """Empty description still works (edge case)."""
        mock_agents["manus"].return_value.return_value = "ok"
        task = {"task_id": "t1", "description": "", "agent_type": "manus"}
        result = manager.execute_task(task, {"task_results": {}}, "tu-1")
        assert result["status"] == "success"

    def test_create_workflow_explicit_id(self):
        """Create with explicit workflow_id uses it."""
        with patch(f"{MODULE}.ManusWorkflowManager") as mock_cls:
            mock_mgr = MagicMock()
            mock_mgr.create_workflow.return_value = {
                "status": "success",
                "content": [{"text": "ok"}],
            }
            mock_cls.return_value = mock_mgr

            from manus_agent.tools.workflow_tool import workflow_tool

            tool = {
                "toolUseId": "tu-1",
                "input": {
                    "action": "create",
                    "workflow_id": "my-wf",
                    "tasks": [{"task_id": "t1", "description": "test"}],
                },
            }
            workflow_tool(tool)
            call_args = mock_mgr.create_workflow.call_args[0]
            assert call_args[0] == "my-wf"
