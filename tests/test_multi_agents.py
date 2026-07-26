"""Comprehensive test suite for manus_agent.multi_agents package.

Tests cover:
- AgentType enum (values, string behaviour, membership)
- ComplexityLevel enum (values, string behaviour, membership)
- TaskPlan dataclass (construction, defaults, immutability, equality)
- OrchestratorResult dataclass (construction, defaults, mutability)
- WorkflowAgent (init, system_prompt, handle_request delegation)
- Orchestrator (init, config handling, run success/failure paths)
- Module __all__ exports

All tests are fully mocked — no real LLM calls or network access.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.multi_agents import (
    AgentType,
    ComplexityLevel,
    Orchestrator,
    OrchestratorResult,
    TaskPlan,
    WorkflowAgent,
)

# ===========================================================================
# AgentType enum
# ===========================================================================


class TestAgentType:
    """Tests for the AgentType string enum."""

    def test_manus_value(self):
        assert AgentType.MANUS == "manus"
        assert AgentType.MANUS.value == "manus"

    def test_browser_value(self):
        assert AgentType.BROWSER == "browser"
        assert AgentType.BROWSER.value == "browser"

    def test_data_analysis_value(self):
        assert AgentType.DATA_ANALYSIS == "data_analysis"
        assert AgentType.DATA_ANALYSIS.value == "data_analysis"

    def test_mcp_value(self):
        assert AgentType.MCP == "mcp"
        assert AgentType.MCP.value == "mcp"

    def test_member_count(self):
        assert len(AgentType) == 4

    def test_is_str_subclass(self):
        """AgentType members are strings (str enum)."""
        for member in AgentType:
            assert isinstance(member, str)

    def test_string_operations(self):
        """str enum members support normal string operations."""
        assert AgentType.MANUS.upper() == "MANUS"
        assert AgentType.DATA_ANALYSIS.replace("_", "-") == "data-analysis"

    def test_lookup_by_value(self):
        assert AgentType("manus") is AgentType.MANUS
        assert AgentType("browser") is AgentType.BROWSER
        assert AgentType("data_analysis") is AgentType.DATA_ANALYSIS
        assert AgentType("mcp") is AgentType.MCP

    def test_invalid_lookup_raises(self):
        with pytest.raises(ValueError):
            AgentType("invalid_agent")

    def test_iteration_order(self):
        members = list(AgentType)
        assert members == [
            AgentType.MANUS,
            AgentType.BROWSER,
            AgentType.DATA_ANALYSIS,
            AgentType.MCP,
        ]


# ===========================================================================
# ComplexityLevel enum
# ===========================================================================


class TestComplexityLevel:
    """Tests for the ComplexityLevel string enum."""

    def test_low_value(self):
        assert ComplexityLevel.LOW == "low"
        assert ComplexityLevel.LOW.value == "low"

    def test_medium_value(self):
        assert ComplexityLevel.MEDIUM == "medium"
        assert ComplexityLevel.MEDIUM.value == "medium"

    def test_high_value(self):
        assert ComplexityLevel.HIGH == "high"
        assert ComplexityLevel.HIGH.value == "high"

    def test_member_count(self):
        assert len(ComplexityLevel) == 3

    def test_is_str_subclass(self):
        for member in ComplexityLevel:
            assert isinstance(member, str)

    def test_lookup_by_value(self):
        assert ComplexityLevel("low") is ComplexityLevel.LOW
        assert ComplexityLevel("medium") is ComplexityLevel.MEDIUM
        assert ComplexityLevel("high") is ComplexityLevel.HIGH

    def test_invalid_lookup_raises(self):
        with pytest.raises(ValueError):
            ComplexityLevel("critical")


# ===========================================================================
# TaskPlan dataclass
# ===========================================================================


class TestTaskPlan:
    """Tests for the TaskPlan frozen dataclass."""

    def test_minimal_construction(self):
        tp = TaskPlan(
            task_id="t1",
            description="Do something",
            agent_type=AgentType.MANUS,
        )
        assert tp.task_id == "t1"
        assert tp.description == "Do something"
        assert tp.agent_type == AgentType.MANUS

    def test_default_dependencies(self):
        tp = TaskPlan(task_id="t1", description="x", agent_type=AgentType.BROWSER)
        assert tp.dependencies == []

    def test_default_inputs(self):
        tp = TaskPlan(task_id="t1", description="x", agent_type=AgentType.BROWSER)
        assert tp.inputs == {}

    def test_default_expected_output(self):
        tp = TaskPlan(task_id="t1", description="x", agent_type=AgentType.BROWSER)
        assert tp.expected_output == ""

    def test_default_priority(self):
        tp = TaskPlan(task_id="t1", description="x", agent_type=AgentType.BROWSER)
        assert tp.priority == 3

    def test_default_estimated_complexity(self):
        tp = TaskPlan(task_id="t1", description="x", agent_type=AgentType.BROWSER)
        assert tp.estimated_complexity == ComplexityLevel.MEDIUM

    def test_full_construction(self):
        tp = TaskPlan(
            task_id="analyze",
            description="Analyze data",
            agent_type=AgentType.DATA_ANALYSIS,
            dependencies=["collect"],
            inputs={"file": "data.csv"},
            expected_output="report",
            priority=1,
            estimated_complexity=ComplexityLevel.HIGH,
        )
        assert tp.task_id == "analyze"
        assert tp.dependencies == ["collect"]
        assert tp.inputs == {"file": "data.csv"}
        assert tp.expected_output == "report"
        assert tp.priority == 1
        assert tp.estimated_complexity == ComplexityLevel.HIGH

    def test_frozen_immutability_task_id(self):
        tp = TaskPlan(task_id="t1", description="x", agent_type=AgentType.MANUS)
        with pytest.raises(FrozenInstanceError):
            tp.task_id = "t2"  # type: ignore[misc]

    def test_frozen_immutability_description(self):
        tp = TaskPlan(task_id="t1", description="x", agent_type=AgentType.MANUS)
        with pytest.raises(FrozenInstanceError):
            tp.description = "y"  # type: ignore[misc]

    def test_frozen_immutability_priority(self):
        tp = TaskPlan(task_id="t1", description="x", agent_type=AgentType.MANUS)
        with pytest.raises(FrozenInstanceError):
            tp.priority = 5  # type: ignore[misc]

    def test_equality_same_fields(self):
        tp1 = TaskPlan(task_id="t1", description="x", agent_type=AgentType.MANUS)
        tp2 = TaskPlan(task_id="t1", description="x", agent_type=AgentType.MANUS)
        assert tp1 == tp2

    def test_inequality_different_fields(self):
        tp1 = TaskPlan(task_id="t1", description="x", agent_type=AgentType.MANUS)
        tp2 = TaskPlan(task_id="t2", description="x", agent_type=AgentType.MANUS)
        assert tp1 != tp2

    def test_frozen_but_unhashable_due_to_mutable_fields(self):
        """Frozen dataclass with mutable fields (list, dict) is NOT hashable."""
        tp = TaskPlan(task_id="t1", description="x", agent_type=AgentType.MANUS)
        with pytest.raises(TypeError, match="unhashable"):
            hash(tp)

    def test_multiple_dependencies(self):
        tp = TaskPlan(
            task_id="final",
            description="Combine",
            agent_type=AgentType.MCP,
            dependencies=["step1", "step2", "step3"],
        )
        assert len(tp.dependencies) == 3
        assert "step2" in tp.dependencies

    def test_complex_inputs(self):
        tp = TaskPlan(
            task_id="t1",
            description="x",
            agent_type=AgentType.MANUS,
            inputs={"nested": {"key": [1, 2, 3]}, "flag": True},
        )
        assert tp.inputs["nested"]["key"] == [1, 2, 3]
        assert tp.inputs["flag"] is True


# ===========================================================================
# OrchestratorResult dataclass
# ===========================================================================


class TestOrchestratorResult:
    """Tests for the OrchestratorResult mutable dataclass."""

    def test_minimal_construction(self):
        result = OrchestratorResult(success=True)
        assert result.success is True
        assert result.output == ""
        assert result.error is None
        assert result.tasks == []

    def test_success_with_output(self):
        result = OrchestratorResult(success=True, output="done")
        assert result.output == "done"

    def test_failure_with_error(self):
        result = OrchestratorResult(success=False, error="timeout")
        assert result.success is False
        assert result.error == "timeout"

    def test_with_tasks(self):
        tp = TaskPlan(task_id="t1", description="x", agent_type=AgentType.MANUS)
        result = OrchestratorResult(success=True, tasks=[tp])
        assert len(result.tasks) == 1
        assert result.tasks[0].task_id == "t1"

    def test_mutability(self):
        """OrchestratorResult is NOT frozen — fields can be mutated."""
        result = OrchestratorResult(success=False)
        result.success = True
        result.output = "updated"
        result.error = None
        assert result.success is True
        assert result.output == "updated"

    def test_append_tasks(self):
        result = OrchestratorResult(success=True)
        tp = TaskPlan(task_id="t1", description="x", agent_type=AgentType.BROWSER)
        result.tasks.append(tp)
        assert len(result.tasks) == 1

    def test_equality(self):
        r1 = OrchestratorResult(success=True, output="ok")
        r2 = OrchestratorResult(success=True, output="ok")
        assert r1 == r2

    def test_inequality(self):
        r1 = OrchestratorResult(success=True)
        r2 = OrchestratorResult(success=False)
        assert r1 != r2


# ===========================================================================
# WorkflowAgent
# ===========================================================================


class TestWorkflowAgent:
    """Tests for the WorkflowAgent class (all LLM calls mocked)."""

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_init_default_model(self, mock_agent_cls):
        agent = WorkflowAgent()
        mock_agent_cls.assert_called_once()
        call_kwargs = mock_agent_cls.call_args
        assert call_kwargs.kwargs["model"] == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        assert agent.system_prompt is not None

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_init_custom_model(self, mock_agent_cls):
        agent = WorkflowAgent(model_name="custom-model-v2")
        call_kwargs = mock_agent_cls.call_args
        assert call_kwargs.kwargs["model"] == "custom-model-v2"
        assert agent is not None

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_system_prompt_content(self, mock_agent_cls):
        agent = WorkflowAgent()
        assert "Workflow Management Agent" in agent.system_prompt
        assert "manus" in agent.system_prompt
        assert "browser" in agent.system_prompt
        assert "data_analysis" in agent.system_prompt
        assert "mcp" in agent.system_prompt

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_system_prompt_has_task_format(self, mock_agent_cls):
        agent = WorkflowAgent()
        assert "task_id" in agent.system_prompt
        assert "description" in agent.system_prompt
        assert "agent_type" in agent.system_prompt
        assert "dependencies" in agent.system_prompt
        assert "priority" in agent.system_prompt

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_agent_created_with_workflow_tool(self, mock_agent_cls):
        import manus_agent.tools.workflow_tool as wt

        WorkflowAgent()
        call_kwargs = mock_agent_cls.call_args
        assert wt in call_kwargs.kwargs["tools"]

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_handle_request_delegates_to_agent(self, mock_agent_cls):
        mock_instance = MagicMock()
        mock_instance.return_value = "workflow created successfully"
        mock_agent_cls.return_value = mock_instance

        agent = WorkflowAgent()
        result = agent.handle_request("Create a research workflow")

        mock_instance.assert_called_once_with("Create a research workflow")
        assert result == "workflow created successfully"

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_handle_request_passes_through_response(self, mock_agent_cls):
        mock_instance = MagicMock()
        mock_instance.return_value = {"status": "done", "tasks": 3}
        mock_agent_cls.return_value = mock_instance

        agent = WorkflowAgent()
        result = agent.handle_request("complex task")

        assert result == {"status": "done", "tasks": 3}

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_handle_request_propagates_exception(self, mock_agent_cls):
        mock_instance = MagicMock()
        mock_instance.side_effect = RuntimeError("LLM unavailable")
        mock_agent_cls.return_value = mock_instance

        agent = WorkflowAgent()
        with pytest.raises(RuntimeError, match="LLM unavailable"):
            agent.handle_request("do something")

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_handle_request_empty_string(self, mock_agent_cls):
        mock_instance = MagicMock()
        mock_instance.return_value = ""
        mock_agent_cls.return_value = mock_instance

        agent = WorkflowAgent()
        result = agent.handle_request("")

        mock_instance.assert_called_once_with("")
        assert result == ""

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_agent_attribute_stored(self, mock_agent_cls):
        mock_instance = MagicMock()
        mock_agent_cls.return_value = mock_instance

        wa = WorkflowAgent()
        assert wa.agent is mock_instance


# ===========================================================================
# Orchestrator
# ===========================================================================


class TestOrchestrator:
    """Tests for the Orchestrator class (all LLM calls mocked)."""

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_init_default(self, mock_wa_cls):
        orch = Orchestrator()
        mock_wa_cls.assert_called_once_with()
        assert orch.config is None
        assert orch.model_name is None
        assert orch.agents == {}

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_init_with_model_name(self, mock_wa_cls):
        orch = Orchestrator(model_name="my-model")
        mock_wa_cls.assert_called_once_with(model_name="my-model")
        assert orch.model_name == "my-model"

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_init_with_config(self, mock_wa_cls):
        cfg = MagicMock()
        orch = Orchestrator(config=cfg)
        assert orch.config is cfg

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_run_success(self, mock_wa_cls):
        mock_wa_instance = MagicMock()
        mock_wa_instance.handle_request.return_value = "All tasks complete"
        mock_wa_cls.return_value = mock_wa_instance

        orch = Orchestrator()
        result = orch.run("Analyze CVE-2024-1234")

        mock_wa_instance.handle_request.assert_called_once_with("Analyze CVE-2024-1234")
        assert result.success is True
        assert result.output == "All tasks complete"
        assert result.error is None

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_run_converts_output_to_string(self, mock_wa_cls):
        mock_wa_instance = MagicMock()
        mock_wa_instance.handle_request.return_value = 42  # non-string
        mock_wa_cls.return_value = mock_wa_instance

        orch = Orchestrator()
        result = orch.run("request")

        assert result.output == "42"
        assert isinstance(result.output, str)

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_run_handles_none_output(self, mock_wa_cls):
        mock_wa_instance = MagicMock()
        mock_wa_instance.handle_request.return_value = None
        mock_wa_cls.return_value = mock_wa_instance

        orch = Orchestrator()
        result = orch.run("request")

        assert result.success is True
        assert result.output == "None"

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_run_failure_on_exception(self, mock_wa_cls):
        mock_wa_instance = MagicMock()
        mock_wa_instance.handle_request.side_effect = ValueError("bad input")
        mock_wa_cls.return_value = mock_wa_instance

        orch = Orchestrator()
        result = orch.run("invalid request")

        assert result.success is False
        assert result.error == "bad input"
        assert result.output == ""

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_run_failure_runtime_error(self, mock_wa_cls):
        mock_wa_instance = MagicMock()
        mock_wa_instance.handle_request.side_effect = RuntimeError("connection lost")
        mock_wa_cls.return_value = mock_wa_instance

        orch = Orchestrator()
        result = orch.run("request")

        assert result.success is False
        assert result.error == "connection lost"

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_run_failure_generic_exception(self, mock_wa_cls):
        mock_wa_instance = MagicMock()
        mock_wa_instance.handle_request.side_effect = Exception("unknown")
        mock_wa_cls.return_value = mock_wa_instance

        orch = Orchestrator()
        result = orch.run("request")

        assert result.success is False
        assert "unknown" in result.error

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_run_returns_orchestrator_result_type(self, mock_wa_cls):
        mock_wa_instance = MagicMock()
        mock_wa_instance.handle_request.return_value = "ok"
        mock_wa_cls.return_value = mock_wa_instance

        orch = Orchestrator()
        result = orch.run("request")

        assert isinstance(result, OrchestratorResult)

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_agents_dict_initially_empty(self, mock_wa_cls):
        orch = Orchestrator()
        assert isinstance(orch.agents, dict)
        assert len(orch.agents) == 0

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_agents_dict_can_be_populated(self, mock_wa_cls):
        orch = Orchestrator()
        orch.agents["custom"] = MagicMock()
        assert "custom" in orch.agents

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_workflow_agent_attribute(self, mock_wa_cls):
        mock_wa_instance = MagicMock()
        mock_wa_cls.return_value = mock_wa_instance

        orch = Orchestrator()
        assert orch._workflow_agent is mock_wa_instance

    @patch("manus_agent.multi_agents.WorkflowAgent")
    def test_multiple_run_calls(self, mock_wa_cls):
        mock_wa_instance = MagicMock()
        mock_wa_instance.handle_request.side_effect = ["result1", "result2", "result3"]
        mock_wa_cls.return_value = mock_wa_instance

        orch = Orchestrator()
        r1 = orch.run("req1")
        r2 = orch.run("req2")
        r3 = orch.run("req3")

        assert r1.output == "result1"
        assert r2.output == "result2"
        assert r3.output == "result3"
        assert mock_wa_instance.handle_request.call_count == 3


# ===========================================================================
# Module-level exports
# ===========================================================================


class TestModuleExports:
    """Tests for the module __all__ exports."""

    def test_all_exports_defined(self):
        from manus_agent import multi_agents

        expected = {
            "WorkflowAgent",
            "AgentType",
            "ComplexityLevel",
            "TaskPlan",
            "Orchestrator",
            "OrchestratorResult",
        }
        assert set(multi_agents.__all__) == expected

    def test_all_exports_importable(self):
        """Every name in __all__ is actually importable."""
        from manus_agent import multi_agents

        for name in multi_agents.__all__:
            assert hasattr(multi_agents, name)

    def test_agent_type_in_module(self):
        from manus_agent.multi_agents import AgentType as AgentTypeImported

        assert AgentTypeImported is AgentType

    def test_orchestrator_in_module(self):
        from manus_agent.multi_agents import Orchestrator as OrchestratorImported

        assert OrchestratorImported is Orchestrator


# ===========================================================================
# Integration-style tests (still fully mocked)
# ===========================================================================


class TestOrchestratorIntegration:
    """Higher-level integration scenarios testing Orchestrator + WorkflowAgent together."""

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_orchestrator_end_to_end_success(self, mock_agent_cls):
        """Orchestrator → WorkflowAgent → Agent mock chain."""
        mock_llm = MagicMock()
        mock_llm.return_value = "Workflow executed: 3 tasks completed"
        mock_agent_cls.return_value = mock_llm

        orch = Orchestrator(model_name="test-model")
        result = orch.run("Scan for vulnerabilities and generate report")

        assert result.success is True
        assert "3 tasks completed" in result.output

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_orchestrator_end_to_end_failure(self, mock_agent_cls):
        """Orchestrator catches exceptions from deep in the chain."""
        mock_llm = MagicMock()
        mock_llm.side_effect = ConnectionError("API unreachable")
        mock_agent_cls.return_value = mock_llm

        orch = Orchestrator()
        result = orch.run("impossible task")

        assert result.success is False
        assert "API unreachable" in result.error

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    def test_orchestrator_with_config_passed_through(self, mock_agent_cls):
        """Config is stored on orchestrator but doesn't affect WorkflowAgent currently."""
        mock_llm = MagicMock()
        mock_llm.return_value = "done"
        mock_agent_cls.return_value = mock_llm

        cfg = MagicMock()
        cfg.llm = MagicMock()
        cfg.llm.provider = "bedrock"

        orch = Orchestrator(config=cfg, model_name="us.anthropic.claude-3-7-sonnet-20250219-v1:0")
        result = orch.run("task")

        assert result.success is True
        assert orch.config is cfg
        assert orch.model_name == "us.anthropic.claude-3-7-sonnet-20250219-v1:0"


# ===========================================================================
# WorkflowAgent main() function
# ===========================================================================


class TestWorkflowAgentMain:
    """Tests for the workflow_agent.main() function."""

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    @patch("builtins.print")
    def test_main_prints_header(self, mock_print, mock_agent_cls):
        mock_llm = MagicMock()
        mock_llm.return_value = "response"
        mock_agent_cls.return_value = mock_llm

        from manus_agent.multi_agents.workflow_agent import main

        main()

        # Check that it printed the banner
        printed_strs = [str(call.args[0]) for call in mock_print.call_args_list if call.args]
        assert any("Workflow Agent Example" in s for s in printed_strs)

    @patch("manus_agent.multi_agents.workflow_agent.Agent")
    @patch("builtins.print")
    def test_main_calls_handle_request(self, mock_print, mock_agent_cls):
        mock_llm = MagicMock()
        mock_llm.return_value = "executed"
        mock_agent_cls.return_value = mock_llm

        from manus_agent.multi_agents.workflow_agent import main

        main()

        # The Agent callable should have been invoked
        mock_llm.assert_called_once()
        call_arg = mock_llm.call_args[0][0]
        assert "vulnerabilities" in call_arg.lower()
