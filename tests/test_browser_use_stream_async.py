"""Comprehensive test suite for BrowserUseAgent.stream_async method.

Tests the full streaming path with callbacks, queue management, error handling,
task cancellation, and browser cleanup. All tests are 100% mocked — no real
browser or HTTP calls.

Covers:
- True streaming path (BrowserUse available): step_callback, done_callback, queue flow
- Fallback path (BrowserUse unavailable): non-streaming wrapper
- Error handling: constructor failures, LLM init failures, post-run exceptions
- Browser cleanup (keep_alive=True with timeout, keep_alive=False direct close)
- Task cancellation on error
- List input handling (message dicts)
- Output model integration with final_result()
- Step callback error resilience
- Done callback error resilience
- Empty/None extracted content handling
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Create a mock Config suitable for BrowserUseAgent initialization."""
    config = MagicMock()
    config.llm.provider = "openai"
    config.llm.model = "gpt-4"
    config.llm.temperature = 0.0
    config.llm.max_tokens = 2000
    config.llm.api_key = "test-key"

    # browser_use config section
    browser_use_cfg = MagicMock()
    browser_use_cfg.headless = True
    browser_use_cfg.enable_memory = False
    browser_use_cfg.max_steps = 10
    browser_use_cfg.max_actions_per_step = 5
    browser_use_cfg.use_vision = False
    browser_use_cfg.save_conversation_path = None
    browser_use_cfg.max_error_length = 400
    browser_use_cfg.tool_calling_method = None
    browser_use_cfg.keep_alive = False
    browser_use_cfg.disable_security = False
    browser_use_cfg.extra_chromium_args = []
    browser_use_cfg.timeout = 30.0
    browser_use_cfg.retry_count = 3
    browser_use_cfg.debug = False
    browser_use_cfg.save_screenshots = False
    browser_use_cfg.screenshot_path = None
    browser_use_cfg.provider = None
    browser_use_cfg.model = None
    browser_use_cfg.api_key = None
    browser_use_cfg.temperature = 0.0
    browser_use_cfg.max_tokens = 2000
    config.browser_use = browser_use_cfg

    # tools config for legacy headless
    config.tools.browser_headless = None

    config.get_model.return_value = MagicMock()

    return config


@pytest.fixture
def make_agent(mock_config):
    """Factory fixture to create a BrowserUseAgent with mocked internals."""

    def _make(**kwargs):
        with (
            patch("manus_agent.agents.browser_use_agent.BROWSER_USE_AVAILABLE", True),
            patch("manus_agent.agents.browser_use_agent.BrowserUse", MagicMock()),
            patch("manus_agent.agents.browser_use_agent.BrowserProfile", MagicMock()),
            patch("manus_agent.agents.browser_use_agent.Controller", MagicMock()),
            patch("manus_agent.agents.browser_use_agent.ChatOpenAI", MagicMock()),
            patch("manus_agent.agents.browser_use_agent.ChatBedrock", MagicMock()),
            patch("manus_agent.agents.browser_use_agent.BaseChatModel", MagicMock()),
            patch("manus_agent.agents.browser_use_agent.AgentHistoryList", MagicMock()),
            patch("manus_agent.agents.browser_use_agent.AgentOutput", MagicMock()),
            patch("manus_agent.agents.browser_use_agent.BrowserStateSummary", MagicMock()),
            patch("manus_agent.agents.browser_use_agent.apply_comprehensive_patch"),
        ):
            from manus_agent.agents.browser_use_agent import BrowserUseAgent

            agent = BrowserUseAgent(config=mock_config, **kwargs)
            # Mock _get_browser_llm to avoid ImportError from missing langchain
            agent._get_browser_llm = Mock(return_value=MagicMock())
            return agent

    return _make


# ---------------------------------------------------------------------------
# Helper to build standard mocks for the true streaming path
# ---------------------------------------------------------------------------


def _build_mock_history(extracted=None, final_result=None, is_successful=True):
    """Build a mock AgentHistoryList."""
    history = MagicMock()
    history.is_successful.return_value = is_successful
    history.history = [MagicMock()] * 3  # 3 steps
    if final_result is not None:
        history.final_result.return_value = final_result
    else:
        history.final_result.return_value = None
    if extracted is not None:
        history.extracted_content.return_value = extracted
    else:
        history.extracted_content.return_value = None
    return history


def _build_mock_summary(url="https://example.com", title="Example"):
    """Build a mock BrowserStateSummary."""
    summary = MagicMock()
    summary.url = url
    summary.title = title
    return summary


def _build_mock_model_output(actions=None, next_goal="do something"):
    """Build a mock AgentOutput."""
    output = MagicMock()
    if actions is None:
        action_mock = MagicMock()
        action_mock.model_dump.return_value = {"action": "click", "selector": "#btn"}
        output.action = [action_mock]
    else:
        output.action = actions
    output.current_state = MagicMock()
    output.current_state.next_goal = next_goal
    return output


def _streaming_patches(browser_use_side_effect):
    """Return patch context managers for the streaming path."""
    return (
        patch(
            "manus_agent.agents.browser_use_agent.BrowserUse",
            MagicMock(side_effect=browser_use_side_effect),
        ),
        patch("manus_agent.agents.browser_use_agent.BrowserProfile", MagicMock()),
        patch("manus_agent.agents.browser_use_agent.Controller", MagicMock()),
        patch("manus_agent.agents.browser_use_agent.apply_comprehensive_patch"),
    )


# ---------------------------------------------------------------------------
# Fallback path tests (BrowserUse unavailable — uses __call__ wrapper)
# ---------------------------------------------------------------------------


class TestStreamAsyncFallbackPath:
    """Tests for stream_async when BrowserUse/BrowserProfile/Controller is None."""

    def test_fallback_sync_result(self, make_agent):
        """When BrowserUse is None, stream_async wraps __call__ result."""

        async def _run():
            agent = make_agent()
            with patch("manus_agent.agents.browser_use_agent.BrowserUse", None):
                agent.__call__ = Mock(return_value="sync_fallback_result")
                results = [res async for res in agent.stream_async(task="test task")]

            assert len(results) == 1
            assert results[0] == {"type": "text", "text": "sync_fallback_result"}

        asyncio.run(_run())

    def test_fallback_async_result(self, make_agent):
        """When BrowserUse is None and __call__ returns a coroutine, it awaits it."""

        async def _run():
            agent = make_agent()

            async def mock_async_call(*args, **kwargs):
                return "async_fallback_result"

            with patch("manus_agent.agents.browser_use_agent.BrowserUse", None):
                agent.__call__ = Mock(side_effect=mock_async_call)
                results = [res async for res in agent.stream_async(task="test task")]

            assert len(results) == 1
            assert results[0] == {"type": "text", "text": "async_fallback_result"}

        asyncio.run(_run())

    def test_fallback_error_yields_error_event(self, make_agent):
        """When BrowserUse is None and __call__ raises, yields error event."""

        async def _run():
            agent = make_agent()
            with patch("manus_agent.agents.browser_use_agent.BrowserUse", None):
                agent.__call__ = Mock(side_effect=RuntimeError("call failed"))
                results = [res async for res in agent.stream_async(task="test task")]

            assert len(results) == 1
            assert results[0]["type"] == "error"
            assert "call failed" in results[0]["message"]

        asyncio.run(_run())

    def test_fallback_browser_profile_none(self, make_agent):
        """Fallback triggers when BrowserProfile is None (partial import failure)."""

        async def _run():
            agent = make_agent()
            with patch("manus_agent.agents.browser_use_agent.BrowserProfile", None):
                agent.__call__ = Mock(return_value="partial_fallback")
                results = [res async for res in agent.stream_async(task="test")]

            assert len(results) == 1
            assert results[0] == {"type": "text", "text": "partial_fallback"}

        asyncio.run(_run())

    def test_fallback_controller_none(self, make_agent):
        """Fallback triggers when Controller is None."""

        async def _run():
            agent = make_agent()
            with patch("manus_agent.agents.browser_use_agent.Controller", None):
                agent.__call__ = Mock(return_value="ctrl_fallback")
                results = [res async for res in agent.stream_async(task="test")]

            assert len(results) == 1
            assert results[0] == {"type": "text", "text": "ctrl_fallback"}

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# True streaming path tests (BrowserUse available)
# ---------------------------------------------------------------------------


class TestStreamAsyncTruePath:
    """Tests for the real streaming path with callbacks and queue."""

    def test_stream_yields_step_and_final_events(self, make_agent, mock_config):
        """Full streaming: step_callback fires, then done_callback fires."""

        async def _run():
            agent = make_agent()
            history = _build_mock_history(extracted=["Found it", "Done"])
            summary = _build_mock_summary()
            model_out = _build_mock_model_output()

            def fake_browser_use_init(**kwargs):
                step_cb = kwargs.get("register_new_step_callback")
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await step_cb(summary, model_out, 1)
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="Find weather")]

            assert len(results) == 2
            # First event: step update
            assert results[0]["type"] == "step_update"
            assert results[0]["step"] == 1
            assert results[0]["url"] == "https://example.com"
            assert results[0]["title"] == "Example"
            assert results[0]["next_goal"] == "do something"
            assert len(results[0]["planned_actions"]) == 1
            # Second event: final result
            assert results[1]["type"] == "final_result"
            assert results[1]["is_successful"] is True
            assert results[1]["total_steps"] == 3
            assert "Found it" in results[1]["content"]
            assert "Done" in results[1]["content"]

        asyncio.run(_run())

    def test_stream_multiple_steps(self, make_agent, mock_config):
        """Multiple step callbacks yield multiple step_update events."""

        async def _run():
            agent = make_agent()
            history = _build_mock_history(extracted=["Final"])

            def fake_browser_use_init(**kwargs):
                step_cb = kwargs.get("register_new_step_callback")
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    for i in range(1, 4):
                        s = _build_mock_summary(url=f"https://page{i}.com", title=f"Page {i}")
                        m = _build_mock_model_output(next_goal=f"goal {i}")
                        await step_cb(s, m, i)
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="multi step")]

            assert len(results) == 4  # 3 steps + 1 final
            for i in range(3):
                assert results[i]["type"] == "step_update"
                assert results[i]["step"] == i + 1
                assert results[i]["url"] == f"https://page{i + 1}.com"
                assert results[i]["next_goal"] == f"goal {i + 1}"
            assert results[3]["type"] == "final_result"

        asyncio.run(_run())

    def test_stream_with_output_model(self, make_agent, mock_config):
        """When output_model is set, final_result() provides JSON content."""

        async def _run():
            output_model = MagicMock()
            agent = make_agent(output_model=output_model)
            history = _build_mock_history(final_result='{"answer": 42}')

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="structured")]

            assert len(results) == 1
            assert results[0]["type"] == "final_result"
            assert results[0]["content"] == '{"answer": 42}'

        asyncio.run(_run())

    def test_stream_extracted_content_none(self, make_agent, mock_config):
        """When extracted_content returns None, content is empty string."""

        async def _run():
            agent = make_agent()
            history = _build_mock_history(extracted=None)

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            assert results[0]["content"] == ""

        asyncio.run(_run())

    def test_stream_extracted_content_non_list(self, make_agent, mock_config):
        """When extracted_content returns non-list, uses str()."""

        async def _run():
            agent = make_agent()
            history = _build_mock_history(extracted=12345)

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            assert results[0]["content"] == "12345"

        asyncio.run(_run())

    def test_stream_is_successful_false(self, make_agent, mock_config):
        """Final event reflects is_successful=False from history."""

        async def _run():
            agent = make_agent()
            history = _build_mock_history(extracted=["failed"], is_successful=False)

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            assert results[0]["is_successful"] is False

        asyncio.run(_run())

    def test_stream_step_callback_with_none_summary(self, make_agent, mock_config):
        """step_callback handles None summary gracefully (url/title = None)."""

        async def _run():
            agent = make_agent()
            history = _build_mock_history(extracted=["done"])
            model_out = _build_mock_model_output()

            def fake_browser_use_init(**kwargs):
                step_cb = kwargs.get("register_new_step_callback")
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await step_cb(None, model_out, 1)
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            assert results[0]["type"] == "step_update"
            assert results[0]["url"] is None
            assert results[0]["title"] is None

        asyncio.run(_run())

    def test_stream_step_callback_with_none_model_output(self, make_agent, mock_config):
        """step_callback handles None model_output (no actions, no goal)."""

        async def _run():
            agent = make_agent()
            history = _build_mock_history(extracted=["done"])
            summary = _build_mock_summary()

            def fake_browser_use_init(**kwargs):
                step_cb = kwargs.get("register_new_step_callback")
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await step_cb(summary, None, 1)
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            assert results[0]["planned_actions"] == []
            assert results[0]["next_goal"] is None

        asyncio.run(_run())

    def test_stream_step_callback_no_actions(self, make_agent, mock_config):
        """step_callback with model_output.action being empty list."""

        async def _run():
            agent = make_agent()
            history = _build_mock_history(extracted=["done"])
            summary = _build_mock_summary()
            model_out = _build_mock_model_output(actions=[])

            def fake_browser_use_init(**kwargs):
                step_cb = kwargs.get("register_new_step_callback")
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await step_cb(summary, model_out, 1)
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            assert results[0]["planned_actions"] == []

        asyncio.run(_run())

    def test_stream_step_callback_no_current_state(self, make_agent, mock_config):
        """step_callback with model_output.current_state being None."""

        async def _run():
            agent = make_agent()
            history = _build_mock_history(extracted=["done"])
            summary = _build_mock_summary()
            model_out = MagicMock()
            model_out.action = []
            model_out.current_state = None

            def fake_browser_use_init(**kwargs):
                step_cb = kwargs.get("register_new_step_callback")
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await step_cb(summary, model_out, 1)
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            assert results[0]["next_goal"] is None

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestStreamAsyncErrorHandling:
    """Tests for error paths in the true streaming execution.

    NOTE: The stream_async code uses asyncio.create_task for run().
    If run() raises AFTER calling done_callback (which puts None on queue),
    the exception propagates via `await run_task_bg` into the except block.
    If run() raises WITHOUT calling done_callback, the queue.get() loop
    hangs — that's a real limitation of the code's error handling design.

    We test realistic error paths that the code actually handles:
    1. Errors BEFORE create_task (constructor, _get_browser_llm)
    2. Errors AFTER done_callback fires (await run_task_bg raises)
    3. Errors inside callbacks (step_callback, done_callback internal exceptions)
    """

    def test_browser_use_constructor_raises(self, make_agent, mock_config):
        """If BrowserUse() constructor raises, yields error event."""

        async def _run():
            agent = make_agent()

            mock_browser_use_cls = MagicMock(side_effect=ValueError("bad config"))

            with (
                patch("manus_agent.agents.browser_use_agent.BrowserUse", mock_browser_use_cls),
                patch("manus_agent.agents.browser_use_agent.BrowserProfile", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.Controller", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.apply_comprehensive_patch"),
            ):
                results = [res async for res in agent.stream_async(task="test")]

            assert any(r["type"] == "error" for r in results)
            assert "bad config" in results[0]["message"]

        asyncio.run(_run())

    def test_get_browser_llm_raises(self, make_agent, mock_config):
        """If _get_browser_llm raises, yields error event."""

        async def _run():
            agent = make_agent()
            agent._get_browser_llm = Mock(side_effect=ImportError("no langchain"))

            with (
                patch("manus_agent.agents.browser_use_agent.BrowserUse", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.BrowserProfile", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.Controller", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.apply_comprehensive_patch"),
            ):
                results = [res async for res in agent.stream_async(task="test")]

            assert any(r["type"] == "error" for r in results)
            assert "no langchain" in results[0]["message"]

        asyncio.run(_run())

    def test_browser_profile_constructor_raises(self, make_agent, mock_config):
        """If BrowserProfile() raises, error is caught before create_task."""

        async def _run():
            agent = make_agent()

            with (
                patch("manus_agent.agents.browser_use_agent.BrowserUse", MagicMock()),
                patch(
                    "manus_agent.agents.browser_use_agent.BrowserProfile",
                    MagicMock(side_effect=TypeError("profile error")),
                ),
                patch("manus_agent.agents.browser_use_agent.Controller", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.apply_comprehensive_patch"),
            ):
                results = [res async for res in agent.stream_async(task="test")]

            assert any(r["type"] == "error" for r in results)
            assert "profile error" in results[0]["message"]

        asyncio.run(_run())

    def test_controller_constructor_raises(self, make_agent, mock_config):
        """If Controller() raises, error is caught before create_task."""

        async def _run():
            agent = make_agent()

            with (
                patch("manus_agent.agents.browser_use_agent.BrowserUse", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.BrowserProfile", MagicMock()),
                patch(
                    "manus_agent.agents.browser_use_agent.Controller",
                    MagicMock(side_effect=RuntimeError("ctrl init failed")),
                ),
                patch("manus_agent.agents.browser_use_agent.apply_comprehensive_patch"),
            ):
                results = [res async for res in agent.stream_async(task="test")]

            assert any(r["type"] == "error" for r in results)
            assert "ctrl init failed" in results[0]["message"]

        asyncio.run(_run())

    def test_run_raises_after_done_callback(self, make_agent, mock_config):
        """If run() calls done_callback then raises, error is caught via await."""

        async def _run():
            agent = make_agent()
            history = _build_mock_history(extracted=["partial"])

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await done_cb(history)
                    raise RuntimeError("post-done crash")

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            # Should get both final_result AND error event
            types = [r["type"] for r in results]
            assert "final_result" in types
            assert "error" in types

        asyncio.run(_run())

    def test_step_callback_exception_yields_error_event(self, make_agent, mock_config):
        """If step_callback itself raises, error event is put on queue."""

        async def _run():
            agent = make_agent()
            history = _build_mock_history(extracted=["done"])

            def fake_browser_use_init(**kwargs):
                step_cb = kwargs.get("register_new_step_callback")
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    # Call step_callback with args that cause model_dump to raise
                    bad_model_out = MagicMock()
                    action_mock = MagicMock()
                    action_mock.model_dump.side_effect = TypeError("serialize error")
                    bad_model_out.action = [action_mock]
                    bad_model_out.current_state = MagicMock()
                    bad_model_out.current_state.next_goal = "test"

                    await step_cb(MagicMock(), bad_model_out, 1)
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            # Should have an error event from the step callback failure
            assert any(r.get("type") == "error" for r in results)
            error_event = next(r for r in results if r.get("type") == "error")
            assert "step_callback" in error_event["message"]

        asyncio.run(_run())

    def test_done_callback_exception_still_ends_stream(self, make_agent, mock_config):
        """If done_callback raises internally, finally still puts None on queue."""

        async def _run():
            agent = make_agent()
            history = MagicMock()
            # Make is_successful() raise to trigger exception in done_callback
            history.is_successful.side_effect = AttributeError("no method")
            history.final_result.return_value = None
            history.extracted_content.side_effect = TypeError("bad content")
            history.history = []

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            # Should get an error event from done_callback's except clause
            assert any(r.get("type") == "error" for r in results)
            error_event = next(r for r in results if r.get("type") == "error")
            assert "done_callback" in error_event["message"]

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Browser cleanup tests
# ---------------------------------------------------------------------------


class TestStreamAsyncCleanup:
    """Tests for browser instance cleanup in stream_async."""

    def test_close_called_on_success(self, make_agent, mock_config):
        """Browser agent instance is closed after successful streaming."""

        async def _run():
            agent = make_agent()
            close_mock = AsyncMock()

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["result"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = close_mock
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                _ = [res async for res in agent.stream_async(task="test")]

            close_mock.assert_awaited_once()

        asyncio.run(_run())

    def test_close_with_keep_alive_timeout(self, make_agent, mock_config):
        """With keep_alive=True, close() is called with a timeout."""

        async def _run():
            mock_config.browser_use.keep_alive = True
            agent = make_agent()
            close_mock = AsyncMock()

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["done"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = close_mock
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                _ = [res async for res in agent.stream_async(task="test")]

            # Close should still be called (via wait_for)
            close_mock.assert_awaited_once()

        asyncio.run(_run())

    def test_close_timeout_with_keep_alive_does_not_crash(self, make_agent, mock_config):
        """If close() times out with keep_alive=True, does not raise."""

        async def _run():
            mock_config.browser_use.keep_alive = True
            agent = make_agent()

            async def slow_close():
                await asyncio.sleep(100)  # Will be cancelled by timeout

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["done"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = slow_close
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with (
                p1,
                p2,
                p3,
                p4,
                patch("manus_agent.agents.browser_use_agent.BROWSER_CLOSE_TIMEOUT", 0.01),
            ):
                results = [res async for res in agent.stream_async(task="test")]

            # We still get the final result
            assert any(r.get("type") == "final_result" for r in results)

        asyncio.run(_run())

    def test_close_raises_does_not_crash(self, make_agent, mock_config):
        """If close() raises an exception, stream_async still completes."""

        async def _run():
            agent = make_agent()

            async def bad_close():
                raise OSError("socket error")

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["done"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = bad_close
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            # Stream completed despite close() error
            assert any(r.get("type") == "final_result" for r in results)

        asyncio.run(_run())

    def test_no_close_attribute_does_not_crash(self, make_agent, mock_config):
        """If browser instance lacks close(), cleanup is skipped gracefully."""

        async def _run():
            agent = make_agent()

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                # spec=[] means no close attribute via hasattr
                instance = MagicMock(spec=[])
                history = _build_mock_history(extracted=["done"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            assert any(r.get("type") == "final_result" for r in results)

        asyncio.run(_run())

    def test_close_called_after_constructor_error(self, make_agent, mock_config):
        """Browser close() is NOT called when BrowserUse constructor fails
        (instance was never created)."""

        async def _run():
            agent = make_agent()

            mock_browser_use_cls = MagicMock(side_effect=ValueError("init failed"))

            with (
                patch("manus_agent.agents.browser_use_agent.BrowserUse", mock_browser_use_cls),
                patch("manus_agent.agents.browser_use_agent.BrowserProfile", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.Controller", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.apply_comprehensive_patch"),
            ):
                results = [res async for res in agent.stream_async(task="test")]

            # Error event should be yielded
            assert any(r["type"] == "error" for r in results)

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Input handling tests
# ---------------------------------------------------------------------------


class TestStreamAsyncInputHandling:
    """Tests for input parsing in stream_async."""

    def test_string_input(self, make_agent, mock_config):
        """String task is passed directly."""

        async def _run():
            agent = make_agent()
            captured_task = None

            def fake_browser_use_init(**kwargs):
                nonlocal captured_task
                captured_task = kwargs.get("task")
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["ok"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                _ = [res async for res in agent.stream_async(task="hello world")]

            assert captured_task == "hello world"

        asyncio.run(_run())

    def test_list_input_extracts_last_user_message(self, make_agent, mock_config):
        """List of message dicts extracts last user message content."""

        async def _run():
            agent = make_agent()
            captured_task = None

            def fake_browser_use_init(**kwargs):
                nonlocal captured_task
                captured_task = kwargs.get("task")
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["ok"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Find the weather"},
            ]
            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                _ = [res async for res in agent.stream_async(task=messages)]

            assert captured_task == "Find the weather"

        asyncio.run(_run())

    def test_list_input_non_user_last_message_fallback(self, make_agent, mock_config):
        """If last message isn't role=user, falls back to str(task)."""

        async def _run():
            agent = make_agent()
            captured_task = None

            def fake_browser_use_init(**kwargs):
                nonlocal captured_task
                captured_task = kwargs.get("task")
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["ok"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            messages = [{"role": "assistant", "content": "I'll help"}]
            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                _ = [res async for res in agent.stream_async(task=messages)]

            assert captured_task == str(messages)

        asyncio.run(_run())

    def test_empty_list_input(self, make_agent, mock_config):
        """Empty list produces empty task string, falls back to str(task)."""

        async def _run():
            agent = make_agent()
            captured_task = None

            def fake_browser_use_init(**kwargs):
                nonlocal captured_task
                captured_task = kwargs.get("task")
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["ok"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                _ = [res async for res in agent.stream_async(task=[])]

            # Empty list: code checks `if task and ...`, falls to str(task)
            assert captured_task == "[]"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Task cancellation tests
# ---------------------------------------------------------------------------


class TestStreamAsyncTaskCancellation:
    """Tests for background task cancellation in finally block."""

    def test_stream_completes_when_constructor_fails(self, make_agent, mock_config):
        """If BrowserUse construction fails, run_task_bg is None — no cancel."""

        async def _run():
            agent = make_agent()
            mock_cls = MagicMock(side_effect=TypeError("constructor failed"))

            with (
                patch("manus_agent.agents.browser_use_agent.BrowserUse", mock_cls),
                patch("manus_agent.agents.browser_use_agent.BrowserProfile", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.Controller", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.apply_comprehensive_patch"),
            ):
                results = [res async for res in agent.stream_async(task="test")]

            assert any(r.get("type") == "error" for r in results)

        asyncio.run(_run())

    def test_run_task_bg_done_after_normal_completion(self, make_agent, mock_config):
        """After normal completion, run_task_bg is done — no cancel in finally."""

        async def _run():
            agent = make_agent()

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["ok"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            assert results[0]["type"] == "final_result"

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Configuration integration tests
# ---------------------------------------------------------------------------


class TestStreamAsyncConfiguration:
    """Tests for proper config propagation into stream_async."""

    def test_headless_config_passed_to_browser_profile(self, make_agent, mock_config):
        """BrowserProfile receives headless setting from agent config."""

        async def _run():
            mock_config.browser_use.headless = False
            agent = make_agent(headless=False)
            profile_kwargs = {}

            def capture_profile(**kwargs):
                profile_kwargs.update(kwargs)
                return MagicMock()

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["ok"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            mock_profile = MagicMock(side_effect=capture_profile)
            with (
                patch(
                    "manus_agent.agents.browser_use_agent.BrowserUse",
                    MagicMock(side_effect=fake_browser_use_init),
                ),
                patch("manus_agent.agents.browser_use_agent.BrowserProfile", mock_profile),
                patch("manus_agent.agents.browser_use_agent.Controller", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.apply_comprehensive_patch"),
            ):
                _ = [res async for res in agent.stream_async(task="test")]

            mock_profile.assert_called_once()
            assert profile_kwargs.get("headless") is False

        asyncio.run(_run())

    def test_output_model_passed_to_controller(self, make_agent, mock_config):
        """When output_model is set, it's passed to Controller."""

        async def _run():
            output_model = MagicMock()
            agent = make_agent(output_model=output_model)
            controller_kwargs = {}

            def capture_controller(**kwargs):
                controller_kwargs.update(kwargs)
                return MagicMock()

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["ok"], final_result='{"x": 1}')

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            mock_ctrl = MagicMock(side_effect=capture_controller)
            with (
                patch(
                    "manus_agent.agents.browser_use_agent.BrowserUse",
                    MagicMock(side_effect=fake_browser_use_init),
                ),
                patch("manus_agent.agents.browser_use_agent.BrowserProfile", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.Controller", mock_ctrl),
                patch("manus_agent.agents.browser_use_agent.apply_comprehensive_patch"),
            ):
                _ = [res async for res in agent.stream_async(task="test")]

            mock_ctrl.assert_called_once()
            assert controller_kwargs.get("output_model") is output_model

        asyncio.run(_run())

    def test_enable_memory_passed_to_browser_use(self, make_agent, mock_config):
        """enable_memory flag is passed to BrowserUse constructor."""

        async def _run():
            mock_config.browser_use.enable_memory = True
            agent = make_agent(enable_memory=True)
            browser_use_kwargs = {}

            def capture_browser_use(**kwargs):
                browser_use_kwargs.update(kwargs)
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["ok"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(capture_browser_use)
            with p1, p2, p3, p4:
                _ = [res async for res in agent.stream_async(task="test")]

            assert browser_use_kwargs.get("enable_memory") is True

        asyncio.run(_run())

    def test_apply_browser_patch_config_called(self, make_agent, mock_config):
        """_apply_browser_patch_config is called before creating BrowserUse."""

        async def _run():
            agent = make_agent()
            patch_mock = MagicMock()

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["ok"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            with (
                patch(
                    "manus_agent.agents.browser_use_agent.BrowserUse",
                    MagicMock(side_effect=fake_browser_use_init),
                ),
                patch("manus_agent.agents.browser_use_agent.BrowserProfile", MagicMock()),
                patch("manus_agent.agents.browser_use_agent.Controller", MagicMock()),
                patch(
                    "manus_agent.agents.browser_use_agent.apply_comprehensive_patch",
                    patch_mock,
                ),
            ):
                _ = [res async for res in agent.stream_async(task="test")]

            patch_mock.assert_called()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestStreamAsyncEdgeCases:
    """Edge cases and boundary conditions."""

    def test_history_without_is_successful_attribute(self, make_agent, mock_config):
        """If history lacks is_successful, the field falls back gracefully."""

        async def _run():
            agent = make_agent()
            history = MagicMock(spec=["extracted_content", "history", "final_result"])
            history.extracted_content.return_value = ["data"]
            history.final_result.return_value = None
            history.history = [1, 2, 3]

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            assert results[0]["type"] == "final_result"
            assert results[0]["total_steps"] == 3

        asyncio.run(_run())

    def test_history_without_history_attribute(self, make_agent, mock_config):
        """If history object lacks .history, total_steps is None."""

        async def _run():
            agent = make_agent()
            history = MagicMock(spec=["extracted_content", "is_successful", "final_result"])
            history.extracted_content.return_value = ["data"]
            history.is_successful.return_value = True
            history.final_result.return_value = None

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            assert results[0]["type"] == "final_result"

        asyncio.run(_run())

    def test_output_model_with_none_final_result(self, make_agent, mock_config):
        """output_model set but final_result() returns None -> empty string."""

        async def _run():
            output_model = MagicMock()
            agent = make_agent(output_model=output_model)
            history = MagicMock()
            history.final_result.return_value = None
            history.extracted_content.return_value = None
            history.is_successful.return_value = True
            history.history = []

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            assert results[0]["content"] == ""

        asyncio.run(_run())

    def test_extracted_content_list_with_mixed_types(self, make_agent, mock_config):
        """extracted_content list with non-string items uses str()."""

        async def _run():
            agent = make_agent()
            history = MagicMock()
            history.extracted_content.return_value = [
                "text",
                123,
                None,
                {"key": "val"},
            ]
            history.is_successful.return_value = True
            history.final_result.return_value = None
            history.history = []

            def fake_browser_use_init(**kwargs):
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(fake_browser_use_init)
            with p1, p2, p3, p4:
                results = [res async for res in agent.stream_async(task="test")]

            content = results[0]["content"]
            assert "text" in content
            assert "123" in content
            assert "None" in content
            assert "key" in content

        asyncio.run(_run())

    def test_validate_output_always_false(self, make_agent, mock_config):
        """validate_output=False is always passed to BrowserUse."""

        async def _run():
            agent = make_agent()
            browser_use_kwargs = {}

            def capture_browser_use(**kwargs):
                browser_use_kwargs.update(kwargs)
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["ok"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(capture_browser_use)
            with p1, p2, p3, p4:
                _ = [res async for res in agent.stream_async(task="test")]

            assert browser_use_kwargs.get("validate_output") is False

        asyncio.run(_run())

    def test_callbacks_registered_in_browser_use(self, make_agent, mock_config):
        """Both step and done callbacks are registered with BrowserUse."""

        async def _run():
            agent = make_agent()
            browser_use_kwargs = {}

            def capture_browser_use(**kwargs):
                browser_use_kwargs.update(kwargs)
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["ok"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(capture_browser_use)
            with p1, p2, p3, p4:
                _ = [res async for res in agent.stream_async(task="test")]

            assert "register_new_step_callback" in browser_use_kwargs
            assert "register_done_callback" in browser_use_kwargs
            assert callable(browser_use_kwargs["register_new_step_callback"])
            assert callable(browser_use_kwargs["register_done_callback"])

        asyncio.run(_run())

    def test_llm_passed_to_browser_use(self, make_agent, mock_config):
        """The result of _get_browser_llm is passed as llm= to BrowserUse."""

        async def _run():
            agent = make_agent()
            fake_llm = MagicMock()
            agent._get_browser_llm = Mock(return_value=fake_llm)
            browser_use_kwargs = {}

            def capture_browser_use(**kwargs):
                browser_use_kwargs.update(kwargs)
                done_cb = kwargs.get("register_done_callback")
                instance = MagicMock()
                history = _build_mock_history(extracted=["ok"])

                async def fake_run():
                    await done_cb(history)
                    return history

                instance.run = fake_run
                instance.close = AsyncMock()
                return instance

            p1, p2, p3, p4 = _streaming_patches(capture_browser_use)
            with p1, p2, p3, p4:
                _ = [res async for res in agent.stream_async(task="test")]

            assert browser_use_kwargs.get("llm") is fake_llm

        asyncio.run(_run())
