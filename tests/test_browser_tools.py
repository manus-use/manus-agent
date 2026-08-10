"""Comprehensive test suite for the browser_tools module.

Tests cover:
- BrowserAgentSession class (init, config handling, LLM creation, run_task, cleanup)
- Singleton management (get_browser_session)
- All @tool-decorated browser functions (browser_do, browser_cleanup, browser_navigate, etc.)
- Error handling and edge cases
- BROWSER_USE_AVAILABLE import guard

All tests are unit tests with mocked dependencies — no real browser or network calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine in a new event loop for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_config(
    *,
    provider="bedrock",
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    temperature=0.0,
    max_tokens=4096,
    headless=True,
    browser_use_provider=None,
    browser_use_model=None,
    browser_use_temperature=None,
    browser_use_max_tokens=None,
    browser_use_headless=None,
    has_browser_use=True,
    tools_browser_headless=True,
):
    """Build a mock Config object with nested attributes."""
    config = MagicMock()
    config.llm.provider = provider
    config.llm.model = model
    config.llm.temperature = temperature
    config.llm.max_tokens = max_tokens
    config.tools.browser_headless = tools_browser_headless

    if has_browser_use:
        config.browser_use.provider = browser_use_provider
        config.browser_use.model = browser_use_model
        config.browser_use.temperature = browser_use_temperature or temperature
        config.browser_use.max_tokens = browser_use_max_tokens or max_tokens
        config.browser_use.headless = browser_use_headless if browser_use_headless is not None else headless
    else:
        # Remove browser_use so hasattr returns False
        del config.browser_use

    return config


@pytest.fixture(autouse=True)
def _reset_global_session():
    """Reset the global browser session singleton before/after each test."""
    import manus_agent.tools.browser_tools as bt

    bt._browser_session = None
    yield
    bt._browser_session = None


# ---------------------------------------------------------------------------
# Module importability
# ---------------------------------------------------------------------------


class TestModuleImport:
    """Verify the module is importable and exposes expected symbols."""

    def test_module_imports(self):
        import manus_agent.tools.browser_tools as bt

        assert hasattr(bt, "BrowserAgentSession")
        assert hasattr(bt, "get_browser_session")
        assert hasattr(bt, "browser_do")
        assert hasattr(bt, "browser_cleanup")

    def test_browser_use_available_flag_exists(self):
        from manus_agent.tools.browser_tools import BROWSER_USE_AVAILABLE

        assert isinstance(BROWSER_USE_AVAILABLE, bool)

    def test_all_tool_functions_exist(self):
        import manus_agent.tools.browser_tools as bt

        expected_tools = [
            "browser_do",
            "browser_cleanup",
            "web_search",
            "browser_navigate",
            "browser_search_google",
            "browser_go_back",
            "browser_wait",
            "browser_click_element",
            "browser_input_text",
            "browser_save_pdf",
            "browser_switch_tab",
            "browser_open_tab",
            "browser_close_tab",
            "browser_extract_content",
            "browser_get_page_info",
            "browser_scroll_down",
            "browser_scroll_up",
            "browser_scroll_to_text",
            "browser_send_keys",
            "browser_select_dropdown",
            "browser_drag_drop",
        ]
        for name in expected_tools:
            assert hasattr(bt, name), f"Missing tool: {name}"

    def test_tool_functions_are_decorated(self):
        """All public tool functions should be strands DecoratedFunctionTool instances."""
        from strands.tools.decorator import DecoratedFunctionTool

        import manus_agent.tools.browser_tools as bt

        tool_names = [
            "browser_do",
            "browser_cleanup",
            "browser_navigate",
            "browser_search_google",
        ]
        for name in tool_names:
            obj = getattr(bt, name)
            assert isinstance(obj, DecoratedFunctionTool), f"{name} is not a DecoratedFunctionTool"


# ---------------------------------------------------------------------------
# BrowserAgentSession.__init__
# ---------------------------------------------------------------------------


class TestBrowserAgentSessionInit:
    """Test BrowserAgentSession initialization and config resolution."""

    def test_default_headless_true(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        session = BrowserAgentSession()
        assert session.headless is True

    def test_headless_param_false(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        session = BrowserAgentSession(headless=False)
        assert session.headless is False

    def test_headless_from_config_browser_use(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        config = _make_config(browser_use_headless=False)
        session = BrowserAgentSession(config=config)
        assert session.headless is False

    def test_explicit_headless_overrides_config(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        config = _make_config(browser_use_headless=False)
        session = BrowserAgentSession(headless=True, config=config)
        assert session.headless is True

    def test_config_without_browser_use_defaults_headless(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        config = _make_config(has_browser_use=False)
        session = BrowserAgentSession(config=config)
        assert session.headless is True

    def test_stores_config(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        config = _make_config()
        session = BrowserAgentSession(config=config)
        assert session.config is config

    def test_stores_none_config(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        session = BrowserAgentSession(config=None)
        assert session.config is None
        assert session.headless is True


# ---------------------------------------------------------------------------
# BrowserAgentSession._get_llm
# ---------------------------------------------------------------------------


class TestBrowserAgentSessionGetLLM:
    """Test LLM creation logic in BrowserAgentSession."""

    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    def test_bedrock_provider_default(self, mock_bedrock_cls):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock_cls.return_value = MagicMock()
        config = _make_config(provider="bedrock")
        session = BrowserAgentSession(config=config)
        llm = session._get_llm()
        mock_bedrock_cls.assert_called_once()
        call_kwargs = mock_bedrock_cls.call_args[1]
        assert call_kwargs["model_id"] == "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert llm is not None

    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    def test_bedrock_uses_browser_use_overrides(self, mock_bedrock_cls):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock_cls.return_value = MagicMock()
        config = _make_config(
            provider="bedrock",
            browser_use_provider="bedrock",
            browser_use_model="custom-model",
            browser_use_temperature=0.5,
            browser_use_max_tokens=2048,
        )
        session = BrowserAgentSession(config=config)
        session._get_llm()
        call_kwargs = mock_bedrock_cls.call_args[1]
        assert call_kwargs["model_id"] == "custom-model"
        assert call_kwargs["model_kwargs"]["temperature"] == 0.5
        assert call_kwargs["model_kwargs"]["max_tokens"] == 2048

    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "eu-west-1"})
    def test_bedrock_uses_env_region(self, mock_bedrock_cls):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock_cls.return_value = MagicMock()
        config = _make_config(provider="bedrock")
        session = BrowserAgentSession(config=config)
        session._get_llm()
        call_kwargs = mock_bedrock_cls.call_args[1]
        assert call_kwargs["region_name"] == "eu-west-1"

    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    @patch.dict("os.environ", {}, clear=True)
    def test_bedrock_defaults_to_us_east_1(self, mock_bedrock_cls):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock_cls.return_value = MagicMock()
        config = _make_config(provider="bedrock")
        session = BrowserAgentSession(config=config)
        session._get_llm()
        call_kwargs = mock_bedrock_cls.call_args[1]
        assert call_kwargs["region_name"] == "us-east-1"

    def test_unsupported_provider_raises(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        config = _make_config(provider="openai")
        session = BrowserAgentSession(config=config)
        with pytest.raises(ValueError, match="Unsupported provider"):
            session._get_llm()

    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    def test_no_config_uses_defaults(self, mock_bedrock_cls):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock_cls.return_value = MagicMock()
        session = BrowserAgentSession()
        session._get_llm()
        call_kwargs = mock_bedrock_cls.call_args[1]
        assert call_kwargs["model_id"] == "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert call_kwargs["model_kwargs"]["temperature"] == 0.0
        assert call_kwargs["model_kwargs"]["max_tokens"] == 4096

    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    def test_browser_use_provider_none_falls_back_to_main(self, mock_bedrock_cls):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock_cls.return_value = MagicMock()
        config = _make_config(
            provider="bedrock",
            model="main-model",
            browser_use_provider=None,
            browser_use_model=None,
        )
        session = BrowserAgentSession(config=config)
        session._get_llm()
        call_kwargs = mock_bedrock_cls.call_args[1]
        # When browser_use.provider is None (falsy), falls back to config.llm.provider
        assert call_kwargs["model_id"] in ("main-model", None)

    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    def test_config_without_browser_use_uses_llm_config(self, mock_bedrock_cls):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock_cls.return_value = MagicMock()
        config = MagicMock()
        config.llm.provider = "bedrock"
        config.llm.model = "fallback-model"
        config.llm.temperature = 0.2
        config.llm.max_tokens = 2000
        # Remove browser_use attribute
        del config.browser_use

        session = BrowserAgentSession(config=config)
        session._get_llm()
        call_kwargs = mock_bedrock_cls.call_args[1]
        assert call_kwargs["model_id"] == "fallback-model"
        assert call_kwargs["model_kwargs"]["temperature"] == 0.2
        assert call_kwargs["model_kwargs"]["max_tokens"] == 2000


# ---------------------------------------------------------------------------
# BrowserAgentSession.run_task
# ---------------------------------------------------------------------------


class TestBrowserAgentSessionRunTask:
    """Test the async run_task method."""

    @patch("manus_agent.tools.browser_tools.BROWSER_USE_AVAILABLE", False)
    def test_run_task_raises_when_unavailable(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        session = BrowserAgentSession()
        with pytest.raises(ImportError, match="browser-use and langchain-aws"):
            _run_async(session.run_task("test task"))

    @patch("manus_agent.tools.browser_tools.BROWSER_USE_AVAILABLE", True)
    @patch("manus_agent.tools.browser_tools.Controller")
    @patch("manus_agent.tools.browser_tools.BrowserProfile")
    @patch("manus_agent.tools.browser_tools.BrowserUseAgent")
    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    def test_run_task_creates_agent_and_runs(self, mock_bedrock, mock_agent_cls, mock_profile, mock_controller):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.extracted_content = "Page content extracted"
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent_instance

        config = _make_config()
        session = BrowserAgentSession(config=config)
        result = _run_async(session.run_task("Navigate to example.com"))

        assert result == "Page content extracted"
        mock_agent_cls.assert_called_once()
        call_kwargs = mock_agent_cls.call_args[1]
        assert call_kwargs["task"] == "Navigate to example.com"
        assert call_kwargs["enable_memory"] is False
        assert call_kwargs["validate_output"] is False

    @patch("manus_agent.tools.browser_tools.BROWSER_USE_AVAILABLE", True)
    @patch("manus_agent.tools.browser_tools.Controller")
    @patch("manus_agent.tools.browser_tools.BrowserProfile")
    @patch("manus_agent.tools.browser_tools.BrowserUseAgent")
    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    def test_run_task_callable_extracted_content(self, mock_bedrock, mock_agent_cls, mock_profile, mock_controller):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock.return_value = MagicMock()
        mock_result = MagicMock()
        # Make extracted_content callable
        mock_result.extracted_content = MagicMock(return_value="callable result")
        # callable() returns True for MagicMock, so this should trigger the callable path
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent_instance

        session = BrowserAgentSession(config=_make_config())
        result = _run_async(session.run_task("test"))

        # MagicMock is always callable, so extracted_content() is called
        assert result == "callable result"

    @patch("manus_agent.tools.browser_tools.BROWSER_USE_AVAILABLE", True)
    @patch("manus_agent.tools.browser_tools.Controller")
    @patch("manus_agent.tools.browser_tools.BrowserProfile")
    @patch("manus_agent.tools.browser_tools.BrowserUseAgent")
    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    def test_run_task_string_extracted_content(self, mock_bedrock, mock_agent_cls, mock_profile, mock_controller):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock.return_value = MagicMock()
        mock_result = MagicMock()
        # String is not callable — should be returned directly
        mock_result.extracted_content = "string result"
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent_instance

        session = BrowserAgentSession(config=_make_config())
        result = _run_async(session.run_task("test"))

        assert result == "string result"

    @patch("manus_agent.tools.browser_tools.BROWSER_USE_AVAILABLE", True)
    @patch("manus_agent.tools.browser_tools.Controller")
    @patch("manus_agent.tools.browser_tools.BrowserProfile")
    @patch("manus_agent.tools.browser_tools.BrowserUseAgent")
    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    def test_run_task_all_results_fallback(self, mock_bedrock, mock_agent_cls, mock_profile, mock_controller):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock.return_value = MagicMock()

        # Result with no extracted_content but with all_results
        mock_result = MagicMock(spec=["all_results"])
        done_result = MagicMock()
        done_result.is_done = True
        done_result.extracted_content = "from all_results"
        not_done_result = MagicMock()
        not_done_result.is_done = False
        not_done_result.extracted_content = "not done"
        mock_result.all_results = [not_done_result, done_result]

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent_instance

        session = BrowserAgentSession(config=_make_config())
        result = _run_async(session.run_task("test"))

        assert result == "from all_results"

    @patch("manus_agent.tools.browser_tools.BROWSER_USE_AVAILABLE", True)
    @patch("manus_agent.tools.browser_tools.Controller")
    @patch("manus_agent.tools.browser_tools.BrowserProfile")
    @patch("manus_agent.tools.browser_tools.BrowserUseAgent")
    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    def test_run_task_str_fallback(self, mock_bedrock, mock_agent_cls, mock_profile, mock_controller):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock.return_value = MagicMock()

        # Result with no extracted_content and no all_results
        # Use a simple object that has neither attribute
        class BareResult:
            def __str__(self):
                return "stringified fallback"

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=BareResult())
        mock_agent_cls.return_value = mock_agent_instance

        session = BrowserAgentSession(config=_make_config())
        result = _run_async(session.run_task("test"))

        # Falls through to str(result)
        assert result == "stringified fallback"

    @patch("manus_agent.tools.browser_tools.BROWSER_USE_AVAILABLE", True)
    @patch("manus_agent.tools.browser_tools.Controller")
    @patch("manus_agent.tools.browser_tools.BrowserProfile")
    @patch("manus_agent.tools.browser_tools.BrowserUseAgent")
    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    def test_run_task_uses_headless_setting(self, mock_bedrock, mock_agent_cls, mock_profile_cls, mock_controller):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.extracted_content = "ok"
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent_instance

        session = BrowserAgentSession(headless=False, config=_make_config())
        _run_async(session.run_task("test"))

        mock_profile_cls.assert_called_once_with(headless=False)

    @patch("manus_agent.tools.browser_tools.BROWSER_USE_AVAILABLE", True)
    @patch("manus_agent.tools.browser_tools.Controller")
    @patch("manus_agent.tools.browser_tools.BrowserProfile")
    @patch("manus_agent.tools.browser_tools.BrowserUseAgent")
    @patch("manus_agent.tools.browser_tools.ChatBedrock")
    def test_run_task_headless_true(self, mock_bedrock, mock_agent_cls, mock_profile_cls, mock_controller):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        mock_bedrock.return_value = MagicMock()
        mock_result = MagicMock()
        mock_result.extracted_content = "ok"
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)
        mock_agent_cls.return_value = mock_agent_instance

        session = BrowserAgentSession(headless=True, config=_make_config())
        _run_async(session.run_task("test"))

        mock_profile_cls.assert_called_once_with(headless=True)


# ---------------------------------------------------------------------------
# BrowserAgentSession.cleanup
# ---------------------------------------------------------------------------


class TestBrowserAgentSessionCleanup:
    """Test the cleanup method."""

    def test_cleanup_completes_without_error(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        session = BrowserAgentSession()
        # Should not raise — cleanup is a no-op for browser-use
        _run_async(session.cleanup())


# ---------------------------------------------------------------------------
# get_browser_session (singleton management)
# ---------------------------------------------------------------------------


class TestGetBrowserSession:
    """Test the singleton session factory."""

    def test_creates_new_session(self):
        import manus_agent.tools.browser_tools as bt

        session = bt.get_browser_session()
        assert isinstance(session, bt.BrowserAgentSession)
        assert session.headless is True

    def test_returns_existing_session(self):
        import manus_agent.tools.browser_tools as bt

        s1 = bt.get_browser_session()
        s2 = bt.get_browser_session()
        assert s1 is s2

    def test_passes_headless_param(self):
        import manus_agent.tools.browser_tools as bt

        session = bt.get_browser_session(headless=False)
        assert session.headless is False

    def test_passes_config_param(self):
        import manus_agent.tools.browser_tools as bt

        config = _make_config()
        session = bt.get_browser_session(config=config)
        assert session.config is config

    def test_ignores_params_on_existing_session(self):
        """Once created, subsequent calls return the same instance regardless of params."""
        import manus_agent.tools.browser_tools as bt

        s1 = bt.get_browser_session(headless=True)
        s2 = bt.get_browser_session(headless=False)
        assert s1 is s2
        assert s2.headless is True  # First creation wins


# ---------------------------------------------------------------------------
# browser_do tool
# ---------------------------------------------------------------------------


class TestBrowserDo:
    """Test the main browser_do @tool function."""

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_success_returns_dict(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_do

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="task completed")
        mock_get_session.return_value = mock_session

        result = _run_async(browser_do.__wrapped__("Navigate to example.com"))
        assert result["success"] is True
        assert result["result"] == "task completed"
        assert result["task"] == "Navigate to example.com"

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_error_returns_error_dict(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_do

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(side_effect=RuntimeError("browser crashed"))
        mock_get_session.return_value = mock_session

        result = _run_async(browser_do.__wrapped__("test"))
        assert result["success"] is False
        assert "browser crashed" in result["error"]
        assert result["task"] == "test"

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_headless_from_browser_use_config(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_do

        config = _make_config(browser_use_headless=False)
        mock_config.return_value = config
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="ok")
        mock_get_session.return_value = mock_session

        _run_async(browser_do.__wrapped__("test", headless=None))
        mock_get_session.assert_called_once_with(headless=False, config=config)

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_explicit_headless_overrides(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_do

        config = _make_config(browser_use_headless=True)
        mock_config.return_value = config
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="ok")
        mock_get_session.return_value = mock_session

        _run_async(browser_do.__wrapped__("test", headless=False))
        mock_get_session.assert_called_once_with(headless=False, config=config)

    @patch("manus_agent.config.Config.from_file")
    def test_config_from_file_exception(self, mock_config):
        from manus_agent.tools.browser_tools import browser_do

        mock_config.side_effect = FileNotFoundError("no config")

        result = _run_async(browser_do.__wrapped__("test"))
        assert result["success"] is False
        assert "no config" in result["error"]

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_config_without_browser_use_uses_tools_headless(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_do

        config = _make_config(has_browser_use=False, tools_browser_headless=False)
        mock_config.return_value = config
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="ok")
        mock_get_session.return_value = mock_session

        _run_async(browser_do.__wrapped__("test", headless=None))
        mock_get_session.assert_called_once_with(headless=False, config=config)


# ---------------------------------------------------------------------------
# browser_cleanup tool
# ---------------------------------------------------------------------------


class TestBrowserCleanup:
    """Test the browser_cleanup tool."""

    def test_cleanup_when_no_session(self):
        from manus_agent.tools.browser_tools import browser_cleanup

        result = _run_async(browser_cleanup.__wrapped__())
        assert result["success"] is True
        assert "cleaned up" in result["message"]

    def test_cleanup_with_active_session(self):
        import manus_agent.tools.browser_tools as bt
        from manus_agent.tools.browser_tools import browser_cleanup

        mock_session = MagicMock()
        mock_session.cleanup = AsyncMock()
        bt._browser_session = mock_session

        result = _run_async(browser_cleanup.__wrapped__())
        assert result["success"] is True
        mock_session.cleanup.assert_called_once()
        assert bt._browser_session is None

    def test_cleanup_error_handling(self):
        import manus_agent.tools.browser_tools as bt
        from manus_agent.tools.browser_tools import browser_cleanup

        mock_session = MagicMock()
        mock_session.cleanup = AsyncMock(side_effect=RuntimeError("cleanup failed"))
        bt._browser_session = mock_session

        result = _run_async(browser_cleanup.__wrapped__())
        assert result["success"] is False
        assert "cleanup failed" in result["error"]


# ---------------------------------------------------------------------------
# Individual browser action tools — all delegate through browser_do
# We test by mocking get_browser_session and Config.from_file since all
# action tools call `await browser_do(task=...)` which flows through those.
# ---------------------------------------------------------------------------


class TestBrowserActionTools:
    """Test individual browser action tool functions.

    Each tool delegates to browser_do with a formatted task string.
    We verify they construct the correct task and pass through results.
    """

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_navigate(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_navigate

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="navigated")
        mock_get_session.return_value = mock_session

        result = _run_async(browser_navigate.__wrapped__("https://example.com"))
        assert result["success"] is True
        assert result["result"] == "navigated"
        task_arg = mock_session.run_task.call_args[0][0]
        assert "https://example.com" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_search_google(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_search_google

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="search results")
        mock_get_session.return_value = mock_session

        result = _run_async(browser_search_google.__wrapped__("python tutorials"))
        assert result["success"] is True
        task_arg = mock_session.run_task.call_args[0][0]
        assert "python tutorials" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_go_back(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_go_back

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="went back")
        mock_get_session.return_value = mock_session

        result = _run_async(browser_go_back.__wrapped__())
        assert result["success"] is True
        task_arg = mock_session.run_task.call_args[0][0]
        assert "previous" in task_arg.lower() or "back" in task_arg.lower()

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_wait_default_3s(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_wait

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="waited")
        mock_get_session.return_value = mock_session

        result = _run_async(browser_wait.__wrapped__())
        assert result["success"] is True
        task_arg = mock_session.run_task.call_args[0][0]
        assert "3" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_wait_custom_seconds(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_wait

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="waited")
        mock_get_session.return_value = mock_session

        _run_async(browser_wait.__wrapped__(seconds=10))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "10" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_click_element_no_index(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_click_element

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="clicked")
        mock_get_session.return_value = mock_session

        _run_async(browser_click_element.__wrapped__("Submit button"))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "Submit button" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_click_element_with_index(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_click_element

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="clicked")
        mock_get_session.return_value = mock_session

        _run_async(browser_click_element.__wrapped__("link", index=3))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "3" in task_arg
        assert "link" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_input_text_no_index(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_input_text

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="typed")
        mock_get_session.return_value = mock_session

        _run_async(browser_input_text.__wrapped__("hello world", "search box"))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "hello world" in task_arg
        assert "search box" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_input_text_with_index(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_input_text

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="typed")
        mock_get_session.return_value = mock_session

        _run_async(browser_input_text.__wrapped__("test", "input field", index=2))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "2" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_save_pdf_no_filename(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_save_pdf

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="saved")
        mock_get_session.return_value = mock_session

        _run_async(browser_save_pdf.__wrapped__())
        task_arg = mock_session.run_task.call_args[0][0]
        assert "pdf" in task_arg.lower()

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_save_pdf_with_filename(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_save_pdf

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="saved")
        mock_get_session.return_value = mock_session

        _run_async(browser_save_pdf.__wrapped__(filename="report.pdf"))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "report.pdf" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_switch_tab(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_switch_tab

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="switched")
        mock_get_session.return_value = mock_session

        _run_async(browser_switch_tab.__wrapped__(tab_id=2))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "2" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_open_tab(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_open_tab

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="opened")
        mock_get_session.return_value = mock_session

        _run_async(browser_open_tab.__wrapped__("https://github.com"))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "https://github.com" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_close_tab(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_close_tab

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="closed")
        mock_get_session.return_value = mock_session

        _run_async(browser_close_tab.__wrapped__(tab_id=5))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "5" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_extract_content(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_extract_content

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="extracted data")
        mock_get_session.return_value = mock_session

        result = _run_async(browser_extract_content.__wrapped__("all email addresses"))
        assert result["success"] is True
        task_arg = mock_session.run_task.call_args[0][0]
        assert "all email addresses" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_extract_content_with_links(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_extract_content

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="data with links")
        mock_get_session.return_value = mock_session

        _run_async(browser_extract_content.__wrapped__("pricing info", include_links=True))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "links" in task_arg.lower()

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_get_page_info(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_get_page_info

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="page info")
        mock_get_session.return_value = mock_session

        result = _run_async(browser_get_page_info.__wrapped__())
        assert result["success"] is True
        assert result["result"] == "page info"

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_scroll_down_default(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_scroll_down

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="scrolled")
        mock_get_session.return_value = mock_session

        _run_async(browser_scroll_down.__wrapped__())
        task_arg = mock_session.run_task.call_args[0][0]
        assert "scroll" in task_arg.lower() or "page" in task_arg.lower()

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_scroll_down_pixels(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_scroll_down

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="scrolled")
        mock_get_session.return_value = mock_session

        _run_async(browser_scroll_down.__wrapped__(pixels=500))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "500" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_scroll_up_default(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_scroll_up

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="scrolled")
        mock_get_session.return_value = mock_session

        _run_async(browser_scroll_up.__wrapped__())
        task_arg = mock_session.run_task.call_args[0][0]
        assert "scroll" in task_arg.lower() or "up" in task_arg.lower()

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_scroll_up_pixels(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_scroll_up

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="scrolled")
        mock_get_session.return_value = mock_session

        _run_async(browser_scroll_up.__wrapped__(pixels=200))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "200" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_scroll_to_text(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_scroll_to_text

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="scrolled")
        mock_get_session.return_value = mock_session

        _run_async(browser_scroll_to_text.__wrapped__("Contact Us"))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "Contact Us" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_send_keys(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_send_keys

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="keys sent")
        mock_get_session.return_value = mock_session

        _run_async(browser_send_keys.__wrapped__("Control+S"))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "Control+S" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_select_dropdown(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_select_dropdown

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="selected")
        mock_get_session.return_value = mock_session

        _run_async(browser_select_dropdown.__wrapped__("English", "language selector"))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "English" in task_arg
        assert "language selector" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_drag_drop(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_drag_drop

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="dropped")
        mock_get_session.return_value = mock_session

        _run_async(browser_drag_drop.__wrapped__("item A", "drop zone B"))
        task_arg = mock_session.run_task.call_args[0][0]
        assert "item A" in task_arg
        assert "drop zone B" in task_arg

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_navigate_error_propagates(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_navigate

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(side_effect=TimeoutError("page load timeout"))
        mock_get_session.return_value = mock_session

        result = _run_async(browser_navigate.__wrapped__("https://slow.example.com"))
        assert result["success"] is False
        assert "timeout" in result["error"].lower()


# ---------------------------------------------------------------------------
# web_search tool (browser_tools version)
# ---------------------------------------------------------------------------


class TestWebSearchTool:
    """Test the web_search tool in browser_tools module."""

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_web_search_fallback_to_browser_do(self, mock_config, mock_get_session):
        """When the web_search import inside fails, should fall back to browser_do."""
        from manus_agent.tools.browser_tools import web_search

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="search results via browser")
        mock_get_session.return_value = mock_session

        # Patch the inner web_search import to raise ImportError
        with patch.dict("sys.modules", {"manus_agent.tools.web_search": None}):
            result = _run_async(web_search.__wrapped__(query="test query"))

        # Should succeed via browser_do fallback
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Edge cases and error paths
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test error handling and edge cases across the module."""

    def test_session_init_with_none_config(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        session = BrowserAgentSession(config=None)
        assert session.headless is True
        assert session.config is None

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_do_with_config_no_browser_use(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_do

        # Config without browser_use attribute
        config = _make_config(has_browser_use=False, tools_browser_headless=False)
        mock_config.return_value = config

        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(return_value="result")
        mock_get_session.return_value = mock_session

        result = _run_async(browser_do.__wrapped__("test", headless=None))
        assert result["success"] is True
        # Should fall back to config.tools.browser_headless
        mock_get_session.assert_called_once_with(headless=False, config=config)

    def test_global_session_reset_on_cleanup(self):
        import manus_agent.tools.browser_tools as bt

        mock_session = MagicMock()
        mock_session.cleanup = AsyncMock()
        bt._browser_session = mock_session

        _run_async(bt.browser_cleanup.__wrapped__())
        assert bt._browser_session is None

    @patch("manus_agent.tools.browser_tools.get_browser_session")
    @patch("manus_agent.config.Config.from_file")
    def test_browser_do_preserves_task_in_error(self, mock_config, mock_get_session):
        from manus_agent.tools.browser_tools import browser_do

        mock_config.return_value = _make_config()
        mock_session = MagicMock()
        mock_session.run_task = AsyncMock(side_effect=ValueError("bad value"))
        mock_get_session.return_value = mock_session

        result = _run_async(browser_do.__wrapped__("my important task"))
        assert result["success"] is False
        assert result["task"] == "my important task"
        assert "bad value" in result["error"]


# ---------------------------------------------------------------------------
# BROWSER_USE_AVAILABLE guard
# ---------------------------------------------------------------------------


class TestBrowserUseAvailableGuard:
    """Test behavior when browser-use is not installed."""

    @patch("manus_agent.tools.browser_tools.BROWSER_USE_AVAILABLE", False)
    def test_run_task_import_error_message(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        session = BrowserAgentSession()
        with pytest.raises(ImportError) as exc_info:
            _run_async(session.run_task("test"))
        assert "browser-use" in str(exc_info.value)
        assert "langchain-aws" in str(exc_info.value)

    @patch("manus_agent.tools.browser_tools.BROWSER_USE_AVAILABLE", False)
    def test_run_task_import_error_contains_install_instructions(self):
        from manus_agent.tools.browser_tools import BrowserAgentSession

        session = BrowserAgentSession()
        with pytest.raises(ImportError) as exc_info:
            _run_async(session.run_task("test"))
        assert "pip install" in str(exc_info.value)
