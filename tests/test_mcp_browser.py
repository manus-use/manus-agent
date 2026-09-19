"""Comprehensive test suite for the manus_agent.mcp.browser module.

Tests cover:
- BrowserTaskResult and AssetMatch Pydantic models
- BrowserAgentRunner construction, queue, consumer, lifecycle
- _execute_browser_task success/failure/post-processing paths
- run_browser_task and run_browser_task_by_cve delegation
- close_browser cleanup (with and without consumer task)
- cli_entry classmethod
- MCP tool functions: asset_match_by_cve, browser
- Module-level main() entrypoint
- Bug fix regression: recerun_browser_task → run_browser_task

All browser_use imports are mocked — no real browser is started.
"""

from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixture: import manus_agent.mcp.browser with browser_use mocked out
# ---------------------------------------------------------------------------


@pytest.fixture()
def browser_module():
    """Import manus_agent.mcp.browser with browser_use deps mocked.

    The real ``mcp`` package stays intact (needed by strands); only
    ``browser_use.*`` sub-imports are faked.
    """
    # Build fake browser_use modules
    fake_browser_use = MagicMock()
    fake_agent_service = MagicMock()
    fake_browser_browser = MagicMock()
    fake_controller_service = MagicMock()
    fake_llm = MagicMock()

    # BrowserProfile / BrowserSession are instantiated at class-body time
    fake_browser_browser.BrowserProfile = MagicMock
    fake_browser_browser.BrowserSession = MagicMock

    browser_use_modules = {
        "browser_use": fake_browser_use,
        "browser_use.agent": MagicMock(),
        "browser_use.agent.service": fake_agent_service,
        "browser_use.browser": MagicMock(),
        "browser_use.browser.browser": fake_browser_browser,
        "browser_use.controller": MagicMock(),
        "browser_use.controller.service": fake_controller_service,
        "browser_use.llm": fake_llm,
    }

    mod_key = "manus_agent.mcp.browser"
    saved = sys.modules.pop(mod_key, None)

    # Only patch browser_use entries; leave mcp.* alone
    with patch.dict(sys.modules, browser_use_modules):
        if mod_key in sys.modules:
            del sys.modules[mod_key]

        import importlib

        mod = importlib.import_module(mod_key)
        # Stash references for patching inside tests
        mod._fake_agent_service = fake_agent_service
        mod._fake_controller_service = fake_controller_service
        mod._fake_llm = fake_llm
        yield mod

    # Restore original state
    if saved is not None:
        sys.modules[mod_key] = saved
    elif mod_key in sys.modules:
        del sys.modules[mod_key]


# ===================================================================
# BrowserTaskResult model tests
# ===================================================================


class TestBrowserTaskResult:
    """Tests for the BrowserTaskResult Pydantic model."""

    def test_create_with_all_fields(self, browser_module):
        result = browser_module.BrowserTaskResult(
            task_completed=True,
            summary="Task done",
            result="Some data",
        )
        assert result.task_completed is True
        assert result.summary == "Task done"
        assert result.result == "Some data"

    def test_create_failure_result(self, browser_module):
        result = browser_module.BrowserTaskResult(
            task_completed=False,
            summary="Agent failed to produce a result.",
            result="The agent did not return a valid result.",
        )
        assert result.task_completed is False
        assert "failed" in result.summary.lower()

    def test_json_roundtrip(self, browser_module):
        original = browser_module.BrowserTaskResult(
            task_completed=True,
            summary="Found the data",
            result="42 components",
        )
        json_str = original.model_dump_json()
        restored = browser_module.BrowserTaskResult.model_validate_json(json_str)
        assert restored.task_completed == original.task_completed
        assert restored.summary == original.summary
        assert restored.result == original.result

    def test_missing_required_field_raises(self, browser_module):
        with pytest.raises((TypeError, ValueError)):
            browser_module.BrowserTaskResult(task_completed=True, summary="ok")

    def test_model_dump(self, browser_module):
        result = browser_module.BrowserTaskResult(
            task_completed=True,
            summary="s",
            result="r",
        )
        d = result.model_dump()
        assert d == {"task_completed": True, "summary": "s", "result": "r"}

    def test_json_parse_from_string(self, browser_module):
        raw = '{"task_completed": false, "summary": "err", "result": "timeout"}'
        parsed = browser_module.BrowserTaskResult.model_validate_json(raw)
        assert parsed.task_completed is False
        assert parsed.result == "timeout"


# ===================================================================
# AssetMatch model tests
# ===================================================================


class TestAssetMatch:
    """Tests for the AssetMatch Pydantic model."""

    def test_create_with_all_fields(self, browser_module):
        am = browser_module.AssetMatch(
            result="ok",
            precisely_matched_assets=10,
            fuzzy_matched_asset=5,
        )
        assert am.precisely_matched_assets == 10
        assert am.fuzzy_matched_asset == 5
        assert am.result == "ok"

    def test_json_roundtrip(self, browser_module):
        original = browser_module.AssetMatch(
            result="data",
            precisely_matched_assets=3,
            fuzzy_matched_asset=7,
        )
        json_str = original.model_dump_json()
        restored = browser_module.AssetMatch.model_validate_json(json_str)
        assert restored.precisely_matched_assets == 3
        assert restored.fuzzy_matched_asset == 7

    def test_zero_counts(self, browser_module):
        am = browser_module.AssetMatch(
            result="none",
            precisely_matched_assets=0,
            fuzzy_matched_asset=0,
        )
        assert am.precisely_matched_assets == 0
        assert am.fuzzy_matched_asset == 0

    def test_large_counts(self, browser_module):
        am = browser_module.AssetMatch(
            result="big",
            precisely_matched_assets=999999,
            fuzzy_matched_asset=888888,
        )
        assert am.precisely_matched_assets == 999999

    def test_model_dump(self, browser_module):
        am = browser_module.AssetMatch(
            result="r",
            precisely_matched_assets=1,
            fuzzy_matched_asset=2,
        )
        d = am.model_dump()
        assert d["precisely_matched_assets"] == 1
        assert d["fuzzy_matched_asset"] == 2


# ===================================================================
# BrowserAgentRunner.__init__ tests
# ===================================================================


class TestBrowserAgentRunnerInit:
    """Tests for BrowserAgentRunner construction."""

    def test_default_params(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        assert runner.task_queue.maxsize == 1000
        assert hasattr(runner, "browser_session")

    def test_custom_headless_false(self, browser_module):
        runner = browser_module.BrowserAgentRunner(headless=False)
        assert runner.task_queue.maxsize == 1000

    def test_custom_keep_alive_false(self, browser_module):
        runner = browser_module.BrowserAgentRunner(keep_alive=False)
        assert hasattr(runner, "browser_session")

    def test_queue_starts_empty(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        assert runner.task_queue.empty()


# ===================================================================
# BrowserAgentRunner.receive tests
# ===================================================================


class TestBrowserAgentRunnerReceive:
    """Tests for the receive() method."""

    @pytest.mark.asyncio
    async def test_receive_adds_to_queue(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        await runner.receive("CVE-2024-1234")
        assert not runner.task_queue.empty()
        item = runner.task_queue.get_nowait()
        assert item == "CVE-2024-1234"

    @pytest.mark.asyncio
    async def test_receive_multiple(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        await runner.receive("CVE-2024-0001")
        await runner.receive("CVE-2024-0002")
        assert runner.task_queue.qsize() == 2


# ===================================================================
# BrowserAgentRunner.consumer tests
# ===================================================================


class TestBrowserAgentRunnerConsumer:
    """Tests for the consumer() coroutine."""

    @pytest.mark.asyncio
    async def test_consumer_processes_item(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        runner.run_browser_task_by_cve = AsyncMock(return_value='{"result": "ok"}')

        await runner.task_queue.put("CVE-2024-9999")

        task = asyncio.create_task(runner.consumer())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        runner.run_browser_task_by_cve.assert_called_once_with("CVE-2024-9999", browser_module.AssetMatch)

    @pytest.mark.asyncio
    async def test_consumer_handles_task_error(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        runner.run_browser_task_by_cve = AsyncMock(side_effect=RuntimeError("boom"))

        await runner.task_queue.put("CVE-2024-0001")

        task = asyncio.create_task(runner.consumer())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        runner.run_browser_task_by_cve.assert_called_once()

    @pytest.mark.asyncio
    async def test_consumer_continues_on_timeout(self, browser_module):
        """When queue is empty, consumer loops via TimeoutError — no crash."""
        runner = browser_module.BrowserAgentRunner()

        task = asyncio.create_task(runner.consumer())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_consumer_marks_task_done_even_on_error(self, browser_module):
        """task_done() must be called even when run_browser_task_by_cve raises."""
        runner = browser_module.BrowserAgentRunner()
        runner.run_browser_task_by_cve = AsyncMock(side_effect=ValueError("bad"))

        await runner.task_queue.put("CVE-2024-0001")

        task = asyncio.create_task(runner.consumer())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert runner.task_queue.qsize() == 0


# ===================================================================
# BrowserAgentRunner.start_browser tests
# ===================================================================


class TestBrowserAgentRunnerStartBrowser:
    """Tests for start_browser()."""

    @pytest.mark.asyncio
    async def test_start_browser_starts_session(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        runner.browser_session = MagicMock()
        runner.browser_session.start = AsyncMock()
        runner.browser_session.kill = AsyncMock()

        await runner.start_browser()

        runner.browser_session.start.assert_called_once()
        assert hasattr(runner, "consumer_task")

        runner.consumer_task.cancel()
        try:
            await runner.consumer_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_start_browser_registers_callback(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        runner.browser_session = MagicMock()
        runner.browser_session.start = AsyncMock()
        runner.browser_session.kill = AsyncMock()

        await runner.start_browser()
        assert hasattr(runner, "consumer_task")

        runner.consumer_task.cancel()
        try:
            await runner.consumer_task
        except asyncio.CancelledError:
            pass


# ===================================================================
# BrowserAgentRunner._consumer_done_callback tests
# ===================================================================


class TestConsumerDoneCallback:
    """Tests for _consumer_done_callback."""

    def test_callback_cancelled_task(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        mock_task = MagicMock()
        mock_task.result.side_effect = asyncio.CancelledError()
        runner._consumer_done_callback(mock_task)

    def test_callback_failed_task(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        mock_task = MagicMock()
        mock_task.result.side_effect = RuntimeError("Consumer crashed")
        runner._consumer_done_callback(mock_task)

    def test_callback_successful_task(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        mock_task = MagicMock()
        mock_task.result.return_value = None
        runner._consumer_done_callback(mock_task)


# ===================================================================
# BrowserAgentRunner._execute_browser_task tests
# ===================================================================


class TestExecuteBrowserTask:
    """Tests for _execute_browser_task()."""

    @pytest.mark.asyncio
    async def test_successful_execution(self, browser_module):
        runner = browser_module.BrowserAgentRunner()

        result_data = browser_module.BrowserTaskResult(
            task_completed=True,
            summary="Done",
            result="Found 5 items",
        )
        result_json = result_data.model_dump_json()

        mock_history = MagicMock()
        mock_history.final_result.return_value = result_json

        with (
            patch.object(
                sys.modules[browser_module.__name__],
                "Agent",
                return_value=MagicMock(run=AsyncMock(return_value=mock_history)),
            ),
            patch.object(
                sys.modules[browser_module.__name__],
                "Controller",
                return_value=MagicMock(),
            ),
            patch.object(
                sys.modules[browser_module.__name__],
                "ChatAnthropicBedrock",
                return_value=MagicMock(),
            ),
        ):
            result = await runner._execute_browser_task("Test task", browser_module.BrowserTaskResult)

        assert result == result_json

    @pytest.mark.asyncio
    async def test_no_result_returns_error(self, browser_module):
        runner = browser_module.BrowserAgentRunner()

        mock_history = MagicMock()
        mock_history.final_result.return_value = None

        with (
            patch.object(
                sys.modules[browser_module.__name__],
                "Agent",
                return_value=MagicMock(run=AsyncMock(return_value=mock_history)),
            ),
            patch.object(
                sys.modules[browser_module.__name__],
                "Controller",
                return_value=MagicMock(),
            ),
            patch.object(
                sys.modules[browser_module.__name__],
                "ChatAnthropicBedrock",
                return_value=MagicMock(),
            ),
        ):
            result_json = await runner._execute_browser_task("Test task", browser_module.BrowserTaskResult)

        parsed = json.loads(result_json)
        assert parsed["task_completed"] is False
        assert "failed" in parsed["summary"].lower()

    @pytest.mark.asyncio
    async def test_post_process_callback_called(self, browser_module):
        runner = browser_module.BrowserAgentRunner()

        result_data = browser_module.BrowserTaskResult(
            task_completed=True,
            summary="Done",
            result="data",
        )
        result_json = result_data.model_dump_json()

        mock_history = MagicMock()
        mock_history.final_result.return_value = result_json

        callback = MagicMock()

        with (
            patch.object(
                sys.modules[browser_module.__name__],
                "Agent",
                return_value=MagicMock(run=AsyncMock(return_value=mock_history)),
            ),
            patch.object(
                sys.modules[browser_module.__name__],
                "Controller",
                return_value=MagicMock(),
            ),
            patch.object(
                sys.modules[browser_module.__name__],
                "ChatAnthropicBedrock",
                return_value=MagicMock(),
            ),
        ):
            await runner._execute_browser_task("Test task", browser_module.BrowserTaskResult, callback)

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0].task_completed is True
        assert args[1] is browser_module.BrowserTaskResult

    @pytest.mark.asyncio
    async def test_post_process_not_called_on_no_result(self, browser_module):
        """When agent returns no result, post_process should NOT be called."""
        runner = browser_module.BrowserAgentRunner()

        mock_history = MagicMock()
        mock_history.final_result.return_value = None

        callback = MagicMock()

        with (
            patch.object(
                sys.modules[browser_module.__name__],
                "Agent",
                return_value=MagicMock(run=AsyncMock(return_value=mock_history)),
            ),
            patch.object(
                sys.modules[browser_module.__name__],
                "Controller",
                return_value=MagicMock(),
            ),
            patch.object(
                sys.modules[browser_module.__name__],
                "ChatAnthropicBedrock",
                return_value=MagicMock(),
            ),
        ):
            result_json = await runner._execute_browser_task("Test", browser_module.BrowserTaskResult, callback)

        callback.assert_not_called()
        parsed = json.loads(result_json)
        assert parsed["task_completed"] is False

    @pytest.mark.asyncio
    async def test_fallback_to_browser_task_result_when_no_output_model(self, browser_module):
        """When structured_output is falsy, should fallback to BrowserTaskResult."""
        runner = browser_module.BrowserAgentRunner()

        result_data = browser_module.BrowserTaskResult(
            task_completed=True,
            summary="Fallback test",
            result="ok",
        )
        result_json = result_data.model_dump_json()

        mock_history = MagicMock()
        mock_history.final_result.return_value = result_json

        with (
            patch.object(
                sys.modules[browser_module.__name__],
                "Agent",
                return_value=MagicMock(run=AsyncMock(return_value=mock_history)),
            ),
            patch.object(
                sys.modules[browser_module.__name__],
                "Controller",
                return_value=MagicMock(),
            ),
            patch.object(
                sys.modules[browser_module.__name__],
                "ChatAnthropicBedrock",
                return_value=MagicMock(),
            ),
        ):
            result = await runner._execute_browser_task("Test task", None)

        assert result == result_json


# ===================================================================
# BrowserAgentRunner.run_browser_task tests
# ===================================================================


class TestRunBrowserTask:
    """Tests for run_browser_task()."""

    @pytest.mark.asyncio
    async def test_delegates_to_execute(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        runner._execute_browser_task = AsyncMock(return_value='{"result": "ok"}')

        result = await runner.run_browser_task("Do something", browser_module.BrowserTaskResult)

        runner._execute_browser_task.assert_called_once_with("Do something", browser_module.BrowserTaskResult)
        assert result == '{"result": "ok"}'

    @pytest.mark.asyncio
    async def test_passes_through_return_value(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        expected = '{"task_completed": true, "summary": "s", "result": "r"}'
        runner._execute_browser_task = AsyncMock(return_value=expected)

        result = await runner.run_browser_task("task", browser_module.BrowserTaskResult)
        assert result == expected


# ===================================================================
# BrowserAgentRunner.run_browser_task_by_cve tests
# ===================================================================


class TestRunBrowserTaskByCve:
    """Tests for run_browser_task_by_cve()."""

    @pytest.mark.asyncio
    async def test_delegates_with_cve_task(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        runner._execute_browser_task = AsyncMock(return_value='{"result": "data"}')

        await runner.run_browser_task_by_cve("CVE-2024-3094", browser_module.AssetMatch)

        runner._execute_browser_task.assert_called_once()
        call_args = runner._execute_browser_task.call_args
        task_str = call_args[0][0]
        assert "CVE-2024-3094" in task_str
        assert "Impacted Component" in task_str
        assert call_args[0][1] is browser_module.AssetMatch
        assert callable(call_args[0][2])

    @pytest.mark.asyncio
    async def test_task_text_mentions_pagination(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        runner._execute_browser_task = AsyncMock(return_value='{"result": "ok"}')

        await runner.run_browser_task_by_cve("CVE-2024-0001", browser_module.AssetMatch)

        task_str = runner._execute_browser_task.call_args[0][0]
        assert "pagination" in task_str.lower()
        assert "all pages" in task_str.lower()

    @pytest.mark.asyncio
    async def test_post_process_callback_posts_to_api(self, browser_module):
        """The CVE post-processor should call requests.post."""
        runner = browser_module.BrowserAgentRunner()
        runner._execute_browser_task = AsyncMock(return_value='{"result": "data"}')

        await runner.run_browser_task_by_cve("CVE-2024-0001", browser_module.AssetMatch)

        callback = runner._execute_browser_task.call_args[0][2]

        mock_result = MagicMock()
        mock_result.precisely_matched_assets = 10
        mock_result.fuzzy_matched_asset = 5
        mock_result.result = "test data"

        with patch.object(sys.modules[browser_module.__name__], "requests") as mock_requests:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_requests.post.return_value = mock_response

            callback(mock_result, browser_module.AssetMatch)

            mock_requests.post.assert_called_once()
            posted_json = mock_requests.post.call_args[1].get("json") or mock_requests.post.call_args[0][1]
            assert posted_json["cve_id"] == "CVE-2024-0001"
            assert posted_json["precisely_matched_assets"] == 10
            assert posted_json["fuzzy_matched_asset"] == 5


# ===================================================================
# BrowserAgentRunner.close_browser tests
# ===================================================================


class TestCloseBrowser:
    """Tests for close_browser()."""

    @pytest.mark.asyncio
    async def test_close_cancels_consumer_task(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        runner.browser_session = MagicMock()
        runner.browser_session.kill = AsyncMock()

        runner.consumer_task = asyncio.create_task(asyncio.sleep(100))

        await runner.close_browser()

        assert runner.consumer_task.cancelled() or runner.consumer_task.done()
        runner.browser_session.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_consumer_task(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        runner.browser_session = MagicMock()
        runner.browser_session.kill = AsyncMock()

        await runner.close_browser()
        runner.browser_session.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_with_already_done_consumer_task(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        runner.browser_session = MagicMock()
        runner.browser_session.kill = AsyncMock()

        runner.consumer_task = asyncio.create_task(asyncio.sleep(0))
        await asyncio.sleep(0.01)

        await runner.close_browser()
        runner.browser_session.kill.assert_called_once()


# ===================================================================
# BrowserAgentRunner.cli_entry tests
# ===================================================================


class TestCliEntry:
    """Tests for the cli_entry() classmethod."""

    @pytest.mark.asyncio
    async def test_cli_entry_no_args(self, browser_module):
        """When no CLI args, should just log a message."""
        with patch("sys.argv", ["browser"]):
            await browser_module.BrowserAgentRunner.cli_entry()

    @pytest.mark.asyncio
    async def test_cli_entry_with_tasks(self, browser_module):
        asset_match_json = browser_module.AssetMatch(
            result="ok",
            precisely_matched_assets=3,
            fuzzy_matched_asset=2,
        ).model_dump_json()

        mock_start = AsyncMock()
        mock_close = AsyncMock()
        mock_run = AsyncMock(return_value=asset_match_json)

        with (
            patch.object(browser_module.BrowserAgentRunner, "__init__", lambda self, *a, **kw: None),
            patch.object(browser_module.BrowserAgentRunner, "start_browser", mock_start),
            patch.object(browser_module.BrowserAgentRunner, "run_browser_task", mock_run),
            patch.object(browser_module.BrowserAgentRunner, "close_browser", mock_close),
            patch.object(browser_module.BrowserAgentRunner, "browser_session", MagicMock(), create=True),
            patch.object(browser_module.BrowserAgentRunner, "task_queue", MagicMock(), create=True),
            patch("sys.argv", ["browser", "Find the top repo"]),
        ):
            await browser_module.BrowserAgentRunner.cli_entry()

        mock_start.assert_called_once()
        mock_run.assert_called_once()
        mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cli_entry_multiple_tasks(self, browser_module):
        asset_match_json = browser_module.AssetMatch(
            result="ok",
            precisely_matched_assets=1,
            fuzzy_matched_asset=1,
        ).model_dump_json()

        mock_run = AsyncMock(return_value=asset_match_json)

        with (
            patch.object(browser_module.BrowserAgentRunner, "__init__", lambda self, *a, **kw: None),
            patch.object(browser_module.BrowserAgentRunner, "start_browser", AsyncMock()),
            patch.object(browser_module.BrowserAgentRunner, "run_browser_task", mock_run),
            patch.object(browser_module.BrowserAgentRunner, "close_browser", AsyncMock()),
            patch.object(browser_module.BrowserAgentRunner, "browser_session", MagicMock(), create=True),
            patch.object(browser_module.BrowserAgentRunner, "task_queue", MagicMock(), create=True),
            patch("sys.argv", ["browser", "Task one", "Task two", "Task three"]),
        ):
            await browser_module.BrowserAgentRunner.cli_entry()

        assert mock_run.call_count == 3


# ===================================================================
# MCP tool: asset_match_by_cve tests
# ===================================================================


class TestMcpAssetMatchByCve:
    """Tests for the asset_match_by_cve MCP tool."""

    @pytest.mark.asyncio
    async def test_enqueues_cve(self, browser_module):
        browser_module.runner = MagicMock()
        browser_module.runner.receive = AsyncMock()

        result = await browser_module.asset_match_by_cve("CVE-2025-12345")

        browser_module.runner.receive.assert_called_once_with("CVE-2025-12345")
        assert "result" in result
        assert "CVE-2025-12345" in result["result"]
        assert "task queue" in result["result"].lower()

    @pytest.mark.asyncio
    async def test_returns_dict(self, browser_module):
        browser_module.runner = MagicMock()
        browser_module.runner.receive = AsyncMock()

        result = await browser_module.asset_match_by_cve("CVE-2024-0001")
        assert isinstance(result, dict)
        assert "result" in result

    @pytest.mark.asyncio
    async def test_result_message_format(self, browser_module):
        browser_module.runner = MagicMock()
        browser_module.runner.receive = AsyncMock()

        result = await browser_module.asset_match_by_cve("CVE-2024-9999")
        assert "successfully added" in result["result"].lower() or "task queue" in result["result"].lower()


# ===================================================================
# MCP tool: browser tests (the fixed function)
# ===================================================================


class TestMcpBrowserTool:
    """Tests for the browser() MCP tool — covers the bug fix."""

    @pytest.mark.asyncio
    async def test_calls_run_browser_task(self, browser_module):
        """After the fix, browser() should call runner.run_browser_task."""
        result_data = browser_module.BrowserTaskResult(
            task_completed=True,
            summary="Done",
            result="Search results found",
        )
        result_json = result_data.model_dump_json()

        browser_module.runner = MagicMock()
        browser_module.runner.run_browser_task = AsyncMock(return_value=result_json)

        result = await browser_module.browser("Find trending repos")

        browser_module.runner.run_browser_task.assert_called_once_with(
            "Find trending repos", browser_module.BrowserTaskResult
        )
        assert result["result"] == "Search results found"

    @pytest.mark.asyncio
    async def test_returns_dict_with_result_key(self, browser_module):
        result_data = browser_module.BrowserTaskResult(
            task_completed=False,
            summary="Error",
            result="Timeout occurred",
        )
        result_json = result_data.model_dump_json()

        browser_module.runner = MagicMock()
        browser_module.runner.run_browser_task = AsyncMock(return_value=result_json)

        result = await browser_module.browser("Some task")
        assert isinstance(result, dict)
        assert result["result"] == "Timeout occurred"

    @pytest.mark.asyncio
    async def test_no_attribute_error_on_recerun_regression(self, browser_module):
        """Regression: old code called runner.recerun_browser_task (typo).
        The fix should use runner.run_browser_task — verify no AttributeError."""
        result_data = browser_module.BrowserTaskResult(
            task_completed=True,
            summary="ok",
            result="ok",
        )
        browser_module.runner = MagicMock()
        browser_module.runner.run_browser_task = AsyncMock(return_value=result_data.model_dump_json())
        # Remove the typo method if it exists on the mock
        if hasattr(browser_module.runner, "recerun_browser_task"):
            del browser_module.runner.recerun_browser_task

        result = await browser_module.browser("test")
        assert result["result"] == "ok"

    @pytest.mark.asyncio
    async def test_passes_browser_task_result_as_output_model(self, browser_module):
        """browser() should pass BrowserTaskResult as the structured_output."""
        result_data = browser_module.BrowserTaskResult(
            task_completed=True,
            summary="s",
            result="r",
        )
        browser_module.runner = MagicMock()
        browser_module.runner.run_browser_task = AsyncMock(return_value=result_data.model_dump_json())

        await browser_module.browser("any task")

        call_args = browser_module.runner.run_browser_task.call_args
        assert call_args[0][1] is browser_module.BrowserTaskResult

    @pytest.mark.asyncio
    async def test_parses_json_result_correctly(self, browser_module):
        """browser() should parse the JSON from run_browser_task and extract result."""
        result_data = browser_module.BrowserTaskResult(
            task_completed=True,
            summary="summary here",
            result="the actual result text",
        )
        browser_module.runner = MagicMock()
        browser_module.runner.run_browser_task = AsyncMock(return_value=result_data.model_dump_json())

        result = await browser_module.browser("task")
        assert result["result"] == "the actual result text"


# ===================================================================
# Module-level main() tests
# ===================================================================


class TestModuleMain:
    """Tests for the module-level main() function."""

    @pytest.mark.asyncio
    async def test_main_starts_browser_and_mcp(self, browser_module):
        mock_runner = MagicMock()
        mock_runner.start_browser = AsyncMock()
        mock_runner.close_browser = AsyncMock()
        browser_module.runner = mock_runner

        mock_mcp = MagicMock()
        mock_mcp.run_streamable_http_async = AsyncMock()
        browser_module.mcp = mock_mcp

        await browser_module.main()

        mock_runner.start_browser.assert_called_once()
        mock_mcp.run_streamable_http_async.assert_called_once()
        mock_runner.close_browser.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_closes_browser_on_exception(self, browser_module):
        mock_runner = MagicMock()
        mock_runner.start_browser = AsyncMock()
        mock_runner.close_browser = AsyncMock()
        browser_module.runner = mock_runner

        mock_mcp = MagicMock()
        mock_mcp.run_streamable_http_async = AsyncMock(side_effect=RuntimeError("MCP crash"))
        browser_module.mcp = mock_mcp

        with pytest.raises(RuntimeError, match="MCP crash"):
            await browser_module.main()

        mock_runner.close_browser.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_close_browser_always_called(self, browser_module):
        """close_browser runs in finally block — always called."""
        mock_runner = MagicMock()
        mock_runner.start_browser = AsyncMock()
        mock_runner.close_browser = AsyncMock()
        browser_module.runner = mock_runner

        mock_mcp = MagicMock()
        mock_mcp.run_streamable_http_async = AsyncMock()
        browser_module.mcp = mock_mcp

        await browser_module.main()
        mock_runner.close_browser.assert_called_once()


# ===================================================================
# Edge case and additional coverage
# ===================================================================


class TestEdgeCases:
    """Additional edge cases for full coverage."""

    def test_browser_task_result_field_descriptions(self, browser_module):
        """Verify field descriptions are set (important for MCP tool schema)."""
        fields = browser_module.BrowserTaskResult.model_fields
        assert fields["task_completed"].description is not None
        assert fields["summary"].description is not None
        assert fields["result"].description is not None

    def test_asset_match_field_descriptions(self, browser_module):
        """Verify AssetMatch field descriptions are set."""
        fields = browser_module.AssetMatch.model_fields
        assert fields["precisely_matched_assets"].description is not None
        assert fields["fuzzy_matched_asset"].description is not None
        assert fields["result"].description is not None

    @pytest.mark.asyncio
    async def test_receive_preserves_fifo_order(self, browser_module):
        runner = browser_module.BrowserAgentRunner()
        await runner.receive("first")
        await runner.receive("second")
        await runner.receive("third")

        assert runner.task_queue.get_nowait() == "first"
        assert runner.task_queue.get_nowait() == "second"
        assert runner.task_queue.get_nowait() == "third"

    @pytest.mark.asyncio
    async def test_multiple_cves_processed_sequentially(self, browser_module):
        """Consumer processes items one at a time in FIFO order."""
        runner = browser_module.BrowserAgentRunner()
        call_order = []

        async def mock_run(cve, output):
            call_order.append(cve)
            return '{"result": "ok"}'

        runner.run_browser_task_by_cve = mock_run

        await runner.task_queue.put("CVE-A")
        await runner.task_queue.put("CVE-B")

        task = asyncio.create_task(runner.consumer())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert call_order == ["CVE-A", "CVE-B"]

    @pytest.mark.asyncio
    async def test_execute_browser_task_uses_max_steps_300(self, browser_module):
        """Verify the agent is configured with max_steps=300."""
        runner = browser_module.BrowserAgentRunner()

        result_data = browser_module.BrowserTaskResult(
            task_completed=True,
            summary="ok",
            result="ok",
        )
        mock_history = MagicMock()
        mock_history.final_result.return_value = result_data.model_dump_json()

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_history)
        mock_agent_cls = MagicMock(return_value=mock_agent_instance)

        with (
            patch.object(sys.modules[browser_module.__name__], "Agent", mock_agent_cls),
            patch.object(sys.modules[browser_module.__name__], "Controller", return_value=MagicMock()),
            patch.object(sys.modules[browser_module.__name__], "ChatAnthropicBedrock", return_value=MagicMock()),
        ):
            await runner._execute_browser_task("task", browser_module.BrowserTaskResult)

        mock_agent_instance.run.assert_called_once_with(max_steps=300)

    @pytest.mark.asyncio
    async def test_execute_browser_task_sets_validate_output(self, browser_module):
        """Verify Agent is constructed with validate_output=True."""
        runner = browser_module.BrowserAgentRunner()

        result_data = browser_module.BrowserTaskResult(
            task_completed=True,
            summary="ok",
            result="ok",
        )
        mock_history = MagicMock()
        mock_history.final_result.return_value = result_data.model_dump_json()

        mock_agent_cls = MagicMock(
            return_value=MagicMock(run=AsyncMock(return_value=mock_history)),
        )

        with (
            patch.object(sys.modules[browser_module.__name__], "Agent", mock_agent_cls),
            patch.object(sys.modules[browser_module.__name__], "Controller", return_value=MagicMock()),
            patch.object(sys.modules[browser_module.__name__], "ChatAnthropicBedrock", return_value=MagicMock()),
        ):
            await runner._execute_browser_task("task", browser_module.BrowserTaskResult)

        agent_kwargs = mock_agent_cls.call_args[1]
        assert agent_kwargs["validate_output"] is True

    @pytest.mark.asyncio
    async def test_error_result_json_is_parseable(self, browser_module):
        """The fallback error result should be valid JSON."""
        runner = browser_module.BrowserAgentRunner()

        mock_history = MagicMock()
        mock_history.final_result.return_value = None

        with (
            patch.object(
                sys.modules[browser_module.__name__],
                "Agent",
                return_value=MagicMock(run=AsyncMock(return_value=mock_history)),
            ),
            patch.object(sys.modules[browser_module.__name__], "Controller", return_value=MagicMock()),
            patch.object(sys.modules[browser_module.__name__], "ChatAnthropicBedrock", return_value=MagicMock()),
        ):
            result_json = await runner._execute_browser_task("task", browser_module.BrowserTaskResult)

        parsed = json.loads(result_json)
        assert "task_completed" in parsed
        assert "summary" in parsed
        assert "result" in parsed

    def test_queue_max_size(self, browser_module):
        """Queue maxsize should be 1000."""
        runner = browser_module.BrowserAgentRunner()
        assert runner.task_queue.maxsize == 1000
