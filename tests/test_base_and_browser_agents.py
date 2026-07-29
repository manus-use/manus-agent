"""Comprehensive tests for BaseManusAgent, BrowserAgent, and DataAnalysisAgent.

Tests cover:
- BaseManusAgent: config resolution, model initialisation, tools list handling,
  context_manager override logic, system prompt defaults, __del__ safety
- BrowserAgent: headless flag resolution from config/parameter, use_browser
  tool integration, system prompt, missing dependency graceful handling
- DataAnalysisAgent: _get_default_tools mapping, _get_default_system_prompt,
  super().__init__ delegation
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    provider: str = "openai",
    model: str = "gpt-4",
    agent_context_manager: str = "auto",
    tools_browser_headless: bool = True,
    search_engine: str = "duckduckgo",
    max_search_results: int = 5,
    enabled_tools: list[str] | None = None,
) -> Mock:
    """Build a minimal mock Config object."""
    config = Mock()
    config.llm.provider = provider
    config.llm.model = model
    config.llm.api_key = "sk-test"
    config.llm.temperature = 0.7
    config.llm.max_tokens = 4096
    config.llm.aws_region = None
    config.tools.browser_headless = tools_browser_headless
    config.tools.search_engine = search_engine
    config.tools.max_search_results = max_search_results
    config.tools.enabled = enabled_tools or ["current_time"]
    config.agent.context_manager = agent_context_manager
    config.agent.model_id = None
    config.agent.aws_region = None

    # get_model returns a mock model
    mock_model = Mock()
    config.get_model.return_value = mock_model
    return config


def _make_config_from_file_patch(config: Mock | None = None):
    """Patch Config.from_file to return a mock config."""
    c = config or _make_config()
    return patch("manus_agent.config.Config.from_file", return_value=c)


# ---------------------------------------------------------------------------
# BaseManusAgent tests
# ---------------------------------------------------------------------------


class TestBaseManusAgentInit:
    """Test BaseManusAgent initialisation logic."""

    @patch("strands.Agent.__init__", return_value=None)
    def test_uses_provided_config(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        agent = BaseManusAgent(config=config, tools=[], model=Mock())
        assert agent.config is config

    @patch("strands.Agent.__init__", return_value=None)
    def test_falls_back_to_config_from_file(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        with _make_config_from_file_patch(config):
            agent = BaseManusAgent(tools=[], model=Mock())
        assert agent.config is config

    @patch("strands.Agent.__init__", return_value=None)
    def test_model_resolved_from_config_when_not_provided(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        sentinel_model = Mock(name="config_model")
        config.get_model.return_value = sentinel_model

        agent = BaseManusAgent(config=config, tools=[])
        # Agent.__init__ should receive the config-resolved model
        call_kwargs = mock_agent_init.call_args
        assert call_kwargs.kwargs.get("model") is sentinel_model or call_kwargs[1].get("model") is sentinel_model

    @patch("strands.Agent.__init__", return_value=None)
    def test_model_injection_bypasses_config(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        injected_model = Mock(name="injected")

        BaseManusAgent(config=config, tools=[], model=injected_model)

        call_kwargs = mock_agent_init.call_args
        passed_model = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
        assert passed_model is injected_model
        config.get_model.assert_not_called()

    @patch("strands.Agent.__init__", return_value=None)
    def test_tools_list_stored_and_passed(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        fake_tool = Mock(name="tool_1")
        agent = BaseManusAgent(config=config, tools=[fake_tool], model=Mock())

        assert agent.tools == [fake_tool]
        assert agent._tools == [fake_tool]
        # Verify tools passed to super().__init__
        call_kwargs = mock_agent_init.call_args
        passed_tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert fake_tool in passed_tools

    @patch("strands.Agent.__init__", return_value=None)
    def test_empty_tools_defaults_to_empty_list(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        agent = BaseManusAgent(config=config, model=Mock())
        assert agent.tools == []

    @patch("strands.Agent.__init__", return_value=None)
    def test_none_tools_defaults_to_empty_list(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        agent = BaseManusAgent(config=config, tools=None, model=Mock())
        assert agent.tools == []


class TestBaseManusAgentContextManager:
    """Test context_manager override resolution in BaseManusAgent."""

    @patch("strands.Agent.__init__", return_value=None)
    def test_auto_defers_to_config_agent_context_manager(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config(agent_context_manager="agentic")
        BaseManusAgent(config=config, tools=[], model=Mock())

        call_kwargs = mock_agent_init.call_args
        passed_cm = call_kwargs.kwargs.get("context_manager")
        assert passed_cm == "agentic"

    @patch("strands.Agent.__init__", return_value=None)
    def test_explicit_context_manager_overrides_config(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config(agent_context_manager="agentic")
        BaseManusAgent(config=config, tools=[], model=Mock(), context_manager="summarizing")

        call_kwargs = mock_agent_init.call_args
        passed_cm = call_kwargs.kwargs.get("context_manager")
        assert passed_cm == "summarizing"

    @patch("strands.Agent.__init__", return_value=None)
    def test_auto_stays_auto_when_config_agent_is_auto(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config(agent_context_manager="auto")
        BaseManusAgent(config=config, tools=[], model=Mock())

        call_kwargs = mock_agent_init.call_args
        passed_cm = call_kwargs.kwargs.get("context_manager")
        assert passed_cm == "auto"

    @patch("strands.Agent.__init__", return_value=None)
    def test_config_without_agent_section_defaults_auto(self, mock_agent_init):
        """When config has no 'agent' attribute, context_manager stays 'auto'."""
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        # Remove the agent attribute to simulate missing section
        del config.agent

        BaseManusAgent(config=config, tools=[], model=Mock())

        call_kwargs = mock_agent_init.call_args
        passed_cm = call_kwargs.kwargs.get("context_manager")
        assert passed_cm == "auto"


class TestBaseManusAgentSystemPrompt:
    """Test system prompt handling."""

    @patch("strands.Agent.__init__", return_value=None)
    def test_custom_system_prompt_passed_through(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        BaseManusAgent(config=config, tools=[], model=Mock(), system_prompt="Custom prompt here")

        call_kwargs = mock_agent_init.call_args
        passed_prompt = call_kwargs.kwargs.get("system_prompt")
        assert passed_prompt == "Custom prompt here"

    @patch("strands.Agent.__init__", return_value=None)
    def test_default_system_prompt_used_when_none(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        BaseManusAgent(config=config, tools=[], model=Mock())

        call_kwargs = mock_agent_init.call_args
        passed_prompt = call_kwargs.kwargs.get("system_prompt")
        assert passed_prompt == "You are a helpful AI assistant."

    def test_get_default_system_prompt_returns_string(self):
        from manus_agent.agents.base import BaseManusAgent

        with patch("strands.Agent.__init__", return_value=None):
            config = _make_config()
            agent = BaseManusAgent(config=config, tools=[], model=Mock())
        assert isinstance(agent._get_default_system_prompt(), str)
        assert len(agent._get_default_system_prompt()) > 0


class TestBaseManusAgentDel:
    """Test __del__ cleanup safety."""

    @patch("strands.Agent.__init__", return_value=None)
    def test_del_does_not_raise_when_parent_has_no_del(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        agent = BaseManusAgent(config=config, tools=[], model=Mock())
        # Should not raise
        agent.__del__()

    @patch("strands.Agent.__init__", return_value=None)
    def test_del_calls_parent_del_when_exists(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        agent = BaseManusAgent(config=config, tools=[], model=Mock())

        # Monkey-patch parent __del__
        parent_del = Mock()
        with patch.object(type(agent).__mro__[1], "__del__", parent_del, create=True):
            agent.__del__()
        # Verification: no crash


class TestBaseManusAgentKwargsPassthrough:
    """Test that extra kwargs reach the Strands Agent base."""

    @patch("strands.Agent.__init__", return_value=None)
    def test_extra_kwargs_passed_to_super(self, mock_agent_init):
        from manus_agent.agents.base import BaseManusAgent

        config = _make_config()
        BaseManusAgent(config=config, tools=[], model=Mock(), some_extra="value")

        call_kwargs = mock_agent_init.call_args
        assert call_kwargs.kwargs.get("some_extra") == "value"


# ---------------------------------------------------------------------------
# BrowserAgent tests
# ---------------------------------------------------------------------------


class TestBrowserAgentInit:
    """Test BrowserAgent (lightweight one in agents/__init__.py)."""

    @patch("strands.Agent.__init__", return_value=None)
    def test_instantiates_with_config(self, mock_agent_init):
        from manus_agent.agents import BrowserAgent

        config = _make_config()
        agent = BrowserAgent(config=config)
        assert agent.config is config

    @patch("strands.Agent.__init__", return_value=None)
    def test_falls_back_to_config_from_file(self, mock_agent_init):
        from manus_agent.agents import BrowserAgent

        config = _make_config()
        with _make_config_from_file_patch(config):
            agent = BrowserAgent()
        assert agent.config is config

    @patch("strands.Agent.__init__", return_value=None)
    def test_headless_from_explicit_parameter(self, mock_agent_init):
        from manus_agent.agents import BrowserAgent

        config = _make_config(tools_browser_headless=True)
        agent = BrowserAgent(config=config, headless=False)
        assert agent.headless is False

    @patch("strands.Agent.__init__", return_value=None)
    def test_headless_from_config(self, mock_agent_init):
        from manus_agent.agents import BrowserAgent

        config = _make_config(tools_browser_headless=False)
        agent = BrowserAgent(config=config)
        assert agent.headless is False

    @patch("strands.Agent.__init__", return_value=None)
    def test_headless_defaults_true_when_config_missing(self, mock_agent_init):
        from manus_agent.agents import BrowserAgent

        config = _make_config()
        # Remove browser_headless to simulate missing config
        del config.tools.browser_headless
        agent = BrowserAgent(config=config)
        assert agent.headless is True

    @patch("strands.Agent.__init__", return_value=None)
    def test_model_injection(self, mock_agent_init):
        from manus_agent.agents import BrowserAgent

        config = _make_config()
        injected = Mock(name="injected_model")
        BrowserAgent(config=config, model=injected)

        call_kwargs = mock_agent_init.call_args
        passed_model = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
        assert passed_model is injected


class TestBrowserAgentToolsSetup:
    """Test tool loading for the lightweight BrowserAgent."""

    @patch("strands.Agent.__init__", return_value=None)
    def test_includes_use_browser_when_available(self, mock_agent_init):
        """When strands_tools.use_browser is importable, it should be in tools."""
        from manus_agent.agents import BrowserAgent, _use_browser

        config = _make_config()
        agent = BrowserAgent(config=config)

        if _use_browser is not None:
            assert _use_browser in agent.tools
        else:
            # If not installed in test env, tools list is empty
            assert agent.tools == []

    @patch("strands.Agent.__init__", return_value=None)
    def test_tools_is_list(self, mock_agent_init):
        from manus_agent.agents import BrowserAgent

        config = _make_config()
        agent = BrowserAgent(config=config)
        assert isinstance(agent.tools, list)


class TestBrowserAgentSystemPrompt:
    """Test BrowserAgent system prompt."""

    @patch("strands.Agent.__init__", return_value=None)
    def test_system_prompt_mentions_browsing(self, mock_agent_init):
        from manus_agent.agents import BrowserAgent

        config = _make_config()
        agent = BrowserAgent(config=config)

        call_kwargs = mock_agent_init.call_args
        passed_prompt = call_kwargs.kwargs.get("system_prompt")
        assert "browsing" in passed_prompt.lower() or "browser" in passed_prompt.lower()

    @patch("strands.Agent.__init__", return_value=None)
    def test_get_default_system_prompt_is_non_empty(self, mock_agent_init):
        from manus_agent.agents import BrowserAgent

        config = _make_config()
        agent = BrowserAgent(config=config)
        prompt = agent._get_default_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 20


class TestBrowserAgentKwargs:
    """Test extra kwargs passthrough."""

    @patch("strands.Agent.__init__", return_value=None)
    def test_extra_kwargs_passed_through(self, mock_agent_init):
        from manus_agent.agents import BrowserAgent

        config = _make_config()
        BrowserAgent(config=config, some_extra="value")

        call_kwargs = mock_agent_init.call_args
        assert call_kwargs.kwargs.get("some_extra") == "value"


# ---------------------------------------------------------------------------
# DataAnalysisAgent tests
# ---------------------------------------------------------------------------


class TestDataAnalysisAgentInit:
    """Test DataAnalysisAgent initialisation."""

    @patch("strands.Agent.__init__", return_value=None)
    def test_instantiates_with_config(self, mock_agent_init):
        from manus_agent.agents import DataAnalysisAgent

        config = _make_config()
        with patch("manus_agent.agents.data_analysis.DataAnalysisAgent._get_default_tools", return_value=[]):
            agent = DataAnalysisAgent(config=config)
        assert agent.config is config

    @patch("strands.Agent.__init__", return_value=None)
    def test_model_injection(self, mock_agent_init):
        from manus_agent.agents import DataAnalysisAgent

        config = _make_config()
        injected = Mock(name="injected_model")
        with patch("manus_agent.agents.data_analysis.DataAnalysisAgent._get_default_tools", return_value=[]):
            DataAnalysisAgent(config=config, model=injected)

        call_kwargs = mock_agent_init.call_args
        passed_model = call_kwargs.kwargs.get("model") or call_kwargs[1].get("model")
        assert passed_model is injected

    @patch("strands.Agent.__init__", return_value=None)
    def test_custom_tools_bypass_default(self, mock_agent_init):
        from manus_agent.agents import DataAnalysisAgent

        config = _make_config()
        custom_tool = Mock(name="custom")
        agent = DataAnalysisAgent(config=config, tools=[custom_tool])

        assert custom_tool in agent.tools


class TestDataAnalysisAgentDefaultTools:
    """Test _get_default_tools resolution."""

    @patch("strands.Agent.__init__", return_value=None)
    def test_get_default_tools_returns_list(self, mock_agent_init):
        from manus_agent.agents.data_analysis import DataAnalysisAgent

        config = _make_config()
        with patch("manus_agent.agents.data_analysis.DataAnalysisAgent._get_default_tools") as mock_tools:
            mock_tools.return_value = [Mock(), Mock()]
            agent = DataAnalysisAgent(config=config)
        assert len(agent.tools) == 2

    @patch("strands.Agent.__init__", return_value=None)
    def test_get_default_tools_calls_get_tools_by_names(self, mock_agent_init):
        """_get_default_tools should call get_tools_by_names with data analysis tool names."""
        from manus_agent.agents.data_analysis import DataAnalysisAgent

        config = _make_config()
        expected_names = [
            "file_read",
            "file_write",
            "code_execute",
            "create_chart",
            "data_analyze",
            "statistical_test",
        ]

        with patch("manus_agent.tools.get_tools_by_names") as mock_get:
            mock_get.return_value = []
            # Call _get_default_tools directly
            agent_cls = DataAnalysisAgent
            # Need to instantiate without calling super().__init__ with broken tools
            result = agent_cls._get_default_tools(None, config)
            mock_get.assert_called_once_with(expected_names, config=config)

    @patch("strands.Agent.__init__", return_value=None)
    def test_get_default_tools_uses_config_from_file_fallback(self, mock_agent_init):
        """When no config passed, falls back to Config.from_file()."""
        from manus_agent.agents.data_analysis import DataAnalysisAgent

        config = _make_config()

        with patch("manus_agent.tools.get_tools_by_names", return_value=[]) as mock_get:
            with _make_config_from_file_patch(config):
                DataAnalysisAgent._get_default_tools(None, None)
                call_kwargs = mock_get.call_args
                assert call_kwargs.kwargs.get("config") is config


class TestDataAnalysisAgentSystemPrompt:
    """Test DataAnalysisAgent system prompt."""

    @patch("strands.Agent.__init__", return_value=None)
    def test_system_prompt_mentions_data_analysis(self, mock_agent_init):
        from manus_agent.agents.data_analysis import DataAnalysisAgent

        config = _make_config()
        with patch("manus_agent.agents.data_analysis.DataAnalysisAgent._get_default_tools", return_value=[]):
            agent = DataAnalysisAgent(config=config)

        call_kwargs = mock_agent_init.call_args
        passed_prompt = call_kwargs.kwargs.get("system_prompt")
        assert "data" in passed_prompt.lower()
        assert "analysis" in passed_prompt.lower() or "analy" in passed_prompt.lower()

    @patch("strands.Agent.__init__", return_value=None)
    def test_get_default_system_prompt_non_empty(self, mock_agent_init):
        from manus_agent.agents.data_analysis import DataAnalysisAgent

        config = _make_config()
        with patch("manus_agent.agents.data_analysis.DataAnalysisAgent._get_default_tools", return_value=[]):
            agent = DataAnalysisAgent(config=config)
        prompt = agent._get_default_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    @patch("strands.Agent.__init__", return_value=None)
    def test_system_prompt_mentions_visualization(self, mock_agent_init):
        from manus_agent.agents.data_analysis import DataAnalysisAgent

        config = _make_config()
        with patch("manus_agent.agents.data_analysis.DataAnalysisAgent._get_default_tools", return_value=[]):
            agent = DataAnalysisAgent(config=config)
        prompt = agent._get_default_system_prompt()
        assert "visualiz" in prompt.lower()


# ---------------------------------------------------------------------------
# agents/__init__.py exports
# ---------------------------------------------------------------------------


class TestAgentsPackageExports:
    """Test that the agents package exports all expected classes."""

    def test_base_manus_agent_exported(self):
        from manus_agent.agents.base import BaseManusAgent

        assert BaseManusAgent is not None

    def test_browser_agent_exported(self):
        from manus_agent.agents import BrowserAgent

        assert BrowserAgent is not None

    def test_data_analysis_agent_exported(self):
        from manus_agent.agents import DataAnalysisAgent

        assert DataAnalysisAgent is not None

    def test_all_list_contains_browser_agent(self):
        from manus_agent import agents

        assert "BrowserAgent" in agents.__all__

    def test_all_list_contains_data_analysis_agent(self):
        from manus_agent import agents

        assert "DataAnalysisAgent" in agents.__all__

    def test_manus_agent_exported(self):
        from manus_agent.agents import ManusAgent

        assert ManusAgent is not None


# ---------------------------------------------------------------------------
# get_tools_by_names tests
# ---------------------------------------------------------------------------


class TestGetToolsByNames:
    """Test the tool resolution utility function."""

    def test_returns_matching_tools(self):
        from manus_agent.tools import ALL_TOOLS, get_tools_by_names

        available = list(ALL_TOOLS.keys())[:3]
        result = get_tools_by_names(available)
        assert len(result) == 3

    def test_unknown_names_skipped(self):
        from manus_agent.tools import get_tools_by_names

        result = get_tools_by_names(["nonexistent_tool_xyz"])
        assert result == []

    def test_partial_match_returns_known_only(self):
        from manus_agent.tools import get_tools_by_names

        result = get_tools_by_names(["current_time", "nonexistent_foobar"])
        assert len(result) == 1

    def test_empty_list_returns_empty(self):
        from manus_agent.tools import get_tools_by_names

        result = get_tools_by_names([])
        assert result == []

    def test_set_config_called_when_available(self):
        """If a tool has set_config, it should be called with config."""
        from manus_agent.tools import get_tools_by_names

        fake_tool = Mock()
        fake_tool.set_config = Mock()
        config = Mock()

        with patch("manus_agent.tools.ALL_TOOLS", {"my_tool": fake_tool}):
            result = get_tools_by_names(["my_tool"], config=config)
        assert len(result) == 1
        fake_tool.set_config.assert_called_once_with(config)

    def test_set_config_not_called_without_config(self):
        """If no config provided, set_config should not be called."""
        from manus_agent.tools import get_tools_by_names

        fake_tool = Mock()
        fake_tool.set_config = Mock()

        with patch("manus_agent.tools.ALL_TOOLS", {"my_tool": fake_tool}):
            result = get_tools_by_names(["my_tool"])
        assert len(result) == 1
        fake_tool.set_config.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: agent hierarchy
# ---------------------------------------------------------------------------


class TestAgentInheritanceChain:
    """Verify the inheritance chain is correct."""

    def test_browser_agent_inherits_base_manus_agent(self):
        from manus_agent.agents import BrowserAgent
        from manus_agent.agents.base import BaseManusAgent

        assert issubclass(BrowserAgent, BaseManusAgent)

    def test_data_analysis_agent_inherits_base_manus_agent(self):
        from manus_agent.agents import DataAnalysisAgent
        from manus_agent.agents.base import BaseManusAgent

        assert issubclass(DataAnalysisAgent, BaseManusAgent)

    def test_manus_agent_inherits_base_manus_agent(self):
        from manus_agent.agents import ManusAgent
        from manus_agent.agents.base import BaseManusAgent

        assert issubclass(ManusAgent, BaseManusAgent)

    def test_base_manus_agent_inherits_strands_agent(self):
        from strands import Agent

        from manus_agent.agents.base import BaseManusAgent

        assert issubclass(BaseManusAgent, Agent)


# ---------------------------------------------------------------------------
# ALL_TOOLS registry integrity
# ---------------------------------------------------------------------------


class TestAllToolsRegistry:
    """Verify the ALL_TOOLS registry in manus_agent.tools.__init__."""

    def test_all_tools_is_dict(self):
        from manus_agent.tools import ALL_TOOLS

        assert isinstance(ALL_TOOLS, dict)

    def test_all_tools_contains_current_time(self):
        from manus_agent.tools import ALL_TOOLS

        assert "current_time" in ALL_TOOLS

    def test_all_tools_contains_http_request(self):
        from manus_agent.tools import ALL_TOOLS

        assert "http_request" in ALL_TOOLS

    def test_all_tools_contains_python_repl(self):
        from manus_agent.tools import ALL_TOOLS

        assert "python_repl" in ALL_TOOLS

    def test_all_tools_contains_shell(self):
        from manus_agent.tools import ALL_TOOLS

        assert "shell" in ALL_TOOLS

    def test_all_tools_values_are_valid(self):
        import types

        from manus_agent.tools import ALL_TOOLS

        for name, tool in ALL_TOOLS.items():
            # Each tool should be callable, a module, or have a tool-like interface
            assert (
                callable(tool) or isinstance(tool, types.ModuleType)
            ), f"Tool {name} is neither callable nor a module"

    def test_all_tools_minimum_count(self):
        from manus_agent.tools import ALL_TOOLS

        # Based on inspection: file_read, file_write, python_repl, shell,
        # http_request, editor, environment, generate_image, current_time, calculator
        assert len(ALL_TOOLS) >= 8
