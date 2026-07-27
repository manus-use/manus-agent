"""Comprehensive test suite for _run_single_shot execution logic.

Tests the internal execution paths of _run_single_shot -- the function that
actually runs tasks in non-interactive mode.
"""

import json
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_agent():
    """Return a callable mock agent that returns a string."""
    agent = mock.MagicMock()
    agent.return_value = "agent result"
    return agent


@pytest.fixture
def patch_make_agent(fake_agent):
    """Patch _make_agent to return fake_agent."""
    with mock.patch("manus_agent.cli._make_agent", return_value=fake_agent) as m:
        yield m


@pytest.fixture
def patch_history():
    """Patch _append_history to capture calls."""
    with mock.patch("manus_agent.cli._append_history") as m:
        yield m


@pytest.fixture
def patch_console():
    """Patch the cli console to suppress Rich output."""
    with mock.patch("manus_agent.cli.console") as m:
        yield m


@pytest.fixture
def patch_orchestrator():
    """Patch the Orchestrator class."""
    with mock.patch("manus_agent.cli.Orchestrator") as m:
        yield m


def _call_single_shot(
    task="test task",
    *,
    mode="single",
    agent_type="manus",
    show_plan=False,
    output=None,
    fmt="text",
    no_history=False,
    config=None,
    stream=False,
):
    """Helper to call _run_single_shot with sane defaults."""
    from manus_agent.cli import _run_single_shot

    if config is None:
        config = mock.MagicMock()
    return _run_single_shot(
        task,
        mode=mode,
        agent_type=agent_type,
        show_plan=show_plan,
        output=output,
        fmt=fmt,
        no_history=no_history,
        config=config,
        stream=stream,
    )


# ===========================================================================
# Section 1: Agent initialisation failure
# ===========================================================================


class TestAgentInitFailure:
    """Tests for when _make_agent raises an exception."""

    def test_returns_1_on_init_failure(self, patch_console):
        with mock.patch("manus_agent.cli._make_agent", side_effect=RuntimeError("boom")):
            rc = _call_single_shot()
        assert rc == 1

    def test_prints_error_on_init_failure(self, patch_console):
        with mock.patch("manus_agent.cli._make_agent", side_effect=RuntimeError("init error")):
            _call_single_shot()
        patch_console.print.assert_called()
        call_args_str = str(patch_console.print.call_args_list)
        assert "init error" in call_args_str

    def test_no_history_on_init_failure(self, patch_history, patch_console):
        with mock.patch("manus_agent.cli._make_agent", side_effect=RuntimeError("boom")):
            _call_single_shot()
        patch_history.assert_not_called()

    def test_import_error_also_handled(self, patch_console):
        with mock.patch("manus_agent.cli._make_agent", side_effect=ImportError("no module")):
            rc = _call_single_shot()
        assert rc == 1


# ===========================================================================
# Section 2: Single-agent buffered execution (default path)
# ===========================================================================


class TestSingleAgentBuffered:
    """Tests for the default non-streaming single-agent path."""

    def test_returns_0_on_success(self, patch_make_agent, patch_history, patch_console):
        rc = _call_single_shot()
        assert rc == 0

    def test_agent_called_with_task(self, patch_make_agent, fake_agent, patch_history, patch_console):
        _call_single_shot(task="analyze CVE-2024-1234")
        fake_agent.assert_called_once_with("analyze CVE-2024-1234")

    def test_result_text_from_agent_response(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = "detailed analysis"
        _call_single_shot(task="do work")
        patch_history.assert_called_once()
        args = patch_history.call_args[0]
        assert args[1] == "detailed analysis"

    def test_text_format_prints_panel(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = "some output"
        _call_single_shot(fmt="text")
        assert patch_console.print.called

    def test_history_appended_by_default(self, patch_make_agent, patch_history, patch_console):
        _call_single_shot(no_history=False)
        patch_history.assert_called_once()

    def test_history_suppressed_with_no_history(self, patch_make_agent, patch_history, patch_console):
        _call_single_shot(no_history=True)
        patch_history.assert_not_called()

    def test_history_records_success_true(self, patch_make_agent, patch_history, patch_console):
        _call_single_shot()
        kwargs = patch_history.call_args[1]
        assert kwargs["success"] is True


# ===========================================================================
# Section 3: Multi-agent orchestrator path
# ===========================================================================


class TestMultiAgentPath:
    """Tests for the multi-agent orchestrator execution path."""

    def test_mode_multi_uses_orchestrator(self, patch_make_agent, patch_orchestrator, patch_history, patch_console):
        orch_instance = patch_orchestrator.return_value
        orch_instance.run.return_value = mock.MagicMock(success=True, output="orchestrated result")
        rc = _call_single_shot(task="simple task", mode="multi")
        assert rc == 0
        orch_instance.run.assert_called_once_with("simple task")

    def test_mode_auto_complex_uses_orchestrator(
        self, patch_make_agent, patch_orchestrator, patch_history, patch_console
    ):
        orch_instance = patch_orchestrator.return_value
        orch_instance.run.return_value = mock.MagicMock(success=True, output="multi output")
        complex_task = "first analyze the data and then generate a report and finally summarize the results"
        rc = _call_single_shot(task=complex_task, mode="auto")
        assert rc == 0
        orch_instance.run.assert_called_once_with(complex_task)

    def test_mode_auto_simple_skips_orchestrator(
        self, patch_make_agent, fake_agent, patch_orchestrator, patch_history, patch_console
    ):
        fake_agent.return_value = "single result"
        rc = _call_single_shot(task="hello", mode="auto")
        assert rc == 0
        patch_orchestrator.return_value.run.assert_not_called()

    def test_mode_single_never_uses_orchestrator(
        self, patch_make_agent, fake_agent, patch_orchestrator, patch_history, patch_console
    ):
        fake_agent.return_value = "result"
        complex_task = "first do step one and then do step two and finally produce the report"
        rc = _call_single_shot(task=complex_task, mode="single")
        assert rc == 0
        patch_orchestrator.return_value.run.assert_not_called()

    def test_orchestrator_result_text_used(self, patch_make_agent, patch_orchestrator, patch_history, patch_console):
        orch_instance = patch_orchestrator.return_value
        orch_instance.run.return_value = mock.MagicMock(success=True, output="orchestrator says hello")
        _call_single_shot(mode="multi")
        patch_history.assert_called_once()
        assert patch_history.call_args[0][1] == "orchestrator says hello"

    def test_orchestrator_config_passed(self, patch_make_agent, patch_orchestrator, patch_history, patch_console):
        orch_instance = patch_orchestrator.return_value
        orch_instance.run.return_value = mock.MagicMock(success=True, output="ok")
        cfg = mock.MagicMock()
        _call_single_shot(mode="multi", config=cfg)
        patch_orchestrator.assert_called_once_with(config=cfg)


# ===========================================================================
# Section 4: Orchestrator failure path
# ===========================================================================


class TestOrchestratorFailure:
    """Tests for when the orchestrator returns success=False."""

    def test_returns_1_on_orchestrator_failure(
        self, patch_make_agent, patch_orchestrator, patch_history, patch_console
    ):
        orch_instance = patch_orchestrator.return_value
        orch_instance.run.return_value = mock.MagicMock(success=False, error="workflow exploded")
        rc = _call_single_shot(mode="multi")
        assert rc == 1

    def test_error_message_includes_error(self, patch_make_agent, patch_orchestrator, patch_history, patch_console):
        orch_instance = patch_orchestrator.return_value
        orch_instance.run.return_value = mock.MagicMock(success=False, error="network timeout")
        _call_single_shot(mode="multi")
        # The error is printed inside a Panel; check Panel.renderable
        panel_calls = [c for c in patch_console.print.call_args_list if hasattr(c[0][0], "renderable")]
        assert len(panel_calls) > 0
        panel_content = str(panel_calls[0][0][0].renderable)
        assert "network timeout" in panel_content

    def test_history_records_failure(self, patch_make_agent, patch_orchestrator, patch_history, patch_console):
        orch_instance = patch_orchestrator.return_value
        orch_instance.run.return_value = mock.MagicMock(success=False, error="bad")
        _call_single_shot(mode="multi", no_history=False)
        patch_history.assert_called_once()
        args = patch_history.call_args[0]
        assert "Task failed: bad" in args[1]

    def test_no_history_suppresses_failure_record(
        self, patch_make_agent, patch_orchestrator, patch_history, patch_console
    ):
        orch_instance = patch_orchestrator.return_value
        orch_instance.run.return_value = mock.MagicMock(success=False, error="oops")
        _call_single_shot(mode="multi", no_history=True)
        patch_history.assert_not_called()

    def test_orchestrator_none_error(self, patch_make_agent, patch_orchestrator, patch_history, patch_console):
        orch_instance = patch_orchestrator.return_value
        orch_instance.run.return_value = mock.MagicMock(success=False, error=None)
        rc = _call_single_shot(mode="multi")
        assert rc == 1


# ===========================================================================
# Section 5: Streaming -- PrintingCallbackHandler path
# ===========================================================================


class TestStreamingPrintingHandler:
    """Tests for streaming with PrintingCallbackHandler available."""

    def test_stream_uses_printing_handler(self, patch_history, patch_console):
        mock_handler_cls = mock.MagicMock()
        mock_stream_agent = mock.MagicMock()
        mock_stream_agent.return_value = "streamed result"

        with mock.patch("manus_agent.cli._make_agent") as m_make:
            m_make.side_effect = [mock.MagicMock(), mock_stream_agent]
            with mock.patch(
                "strands.handlers.PrintingCallbackHandler",
                mock_handler_cls,
                create=True,
            ):
                rc = _call_single_shot(stream=True)

        assert rc == 0
        assert m_make.call_count == 2
        second_call_kwargs = m_make.call_args_list[1][1]
        assert "callback_handler" in second_call_kwargs

    def test_stream_result_from_str_response(self, patch_history, patch_console):
        mock_stream_agent = mock.MagicMock()
        mock_stream_agent.return_value = "streaming output"

        with mock.patch("manus_agent.cli._make_agent") as m_make:
            m_make.side_effect = [mock.MagicMock(), mock_stream_agent]
            with mock.patch(
                "strands.handlers.PrintingCallbackHandler",
                mock.MagicMock(),
                create=True,
            ):
                _call_single_shot(stream=True)

        patch_history.assert_called_once()
        assert patch_history.call_args[0][1] == "streaming output"


# ===========================================================================
# Section 6: Streaming -- Generator fallback path
# ===========================================================================


class TestStreamingGeneratorFallback:
    """Tests for streaming when PrintingCallbackHandler is unavailable."""

    def test_generator_result_is_concatenated(self, patch_history, patch_console):
        def gen():
            yield "A"
            yield "B"
            yield "C"

        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("not supported")
                agent = mock.MagicMock()
                agent.return_value = gen()
                return agent

            m_make.side_effect = make_side
            _call_single_shot(stream=True)

        patch_history.assert_called_once()
        assert patch_history.call_args[0][1] == "ABC"

    def test_generator_writes_chunks_to_stdout(self, patch_history, patch_console):
        def gen():
            yield "hello"
            yield " world"

        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("not supported")
                agent = mock.MagicMock()
                agent.return_value = gen()
                return agent

            m_make.side_effect = make_side
            with mock.patch("sys.stdout") as mock_stdout:
                _call_single_shot(stream=True)
                write_calls = mock_stdout.write.call_args_list
                written_texts = [str(c[0][0]) for c in write_calls]
                assert "hello" in written_texts
                assert " world" in written_texts

    def test_generator_newline_appended(self, patch_history, patch_console):
        def gen():
            yield "x"

        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("not supported")
                agent = mock.MagicMock()
                agent.return_value = gen()
                return agent

            m_make.side_effect = make_side
            with mock.patch("sys.stdout") as mock_stdout:
                _call_single_shot(stream=True)
                last_write = mock_stdout.write.call_args_list[-1][0][0]
                assert last_write == "\n"


# ===========================================================================
# Section 7: Streaming -- Non-streamable agent fallback
# ===========================================================================


class TestStreamingNonStreamable:
    """Tests for streaming when agent returns a plain response."""

    def test_plain_response_fallback_warns(self, patch_history, patch_console):
        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("not supported")
                agent = mock.MagicMock()
                agent.return_value = "plain string result"
                return agent

            m_make.side_effect = make_side
            with mock.patch("sys.stderr") as mock_stderr:
                _call_single_shot(stream=True)
                written = "".join(str(c[0][0]) for c in mock_stderr.write.call_args_list)
                assert "streaming not supported" in written

    def test_plain_response_still_returns_0(self, patch_history, patch_console):
        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("not supported")
                agent = mock.MagicMock()
                agent.return_value = "buffered"
                return agent

            m_make.side_effect = make_side
            with mock.patch("sys.stderr"):
                rc = _call_single_shot(stream=True)
        assert rc == 0

    def test_plain_response_used_as_result(self, patch_history, patch_console):
        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("not supported")
                agent = mock.MagicMock()
                agent.return_value = "fallback text"
                return agent

            m_make.side_effect = make_side
            with mock.patch("sys.stderr"):
                _call_single_shot(stream=True)

        patch_history.assert_called_once()
        assert patch_history.call_args[0][1] == "fallback text"


# ===========================================================================
# Section 8: Streaming + JSON incompatibility
# ===========================================================================


class TestStreamingJsonWarning:
    """Tests for --stream + --format json incompatibility warning."""

    def test_stream_json_warns_on_stderr(self, patch_history, patch_console):
        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("no streaming")
                agent = mock.MagicMock()
                agent.return_value = "json result"
                return agent

            m_make.side_effect = make_side
            with mock.patch("sys.stderr") as mock_stderr:
                with mock.patch("sys.stdout"):
                    _call_single_shot(stream=True, fmt="json")
                written = "".join(str(c[0][0]) for c in mock_stderr.write.call_args_list)
                assert "--stream is incompatible with --format json" in written


# ===========================================================================
# Section 9: Output formatting -- text vs JSON
# ===========================================================================


class TestOutputFormatting:
    """Tests for text and JSON output formatting."""

    def test_json_format_writes_to_stdout(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = "the result"
        with mock.patch("sys.stdout") as mock_stdout:
            _call_single_shot(fmt="json")
            written = "".join(str(c[0][0]) for c in mock_stdout.write.call_args_list)
            data = json.loads(written)
            assert data["task"] == "test task"
            assert data["agent"] == "manus"
            assert data["mode"] == "single"
            assert data["result"] == "the result"

    def test_json_format_multi_mode_label(self, patch_make_agent, patch_orchestrator, patch_history, patch_console):
        orch_instance = patch_orchestrator.return_value
        orch_instance.run.return_value = mock.MagicMock(success=True, output="multi result")
        with mock.patch("sys.stdout") as mock_stdout:
            _call_single_shot(mode="multi", fmt="json")
            written = "".join(str(c[0][0]) for c in mock_stdout.write.call_args_list)
            data = json.loads(written)
            assert data["mode"] == "multi"

    def test_json_includes_unicode(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = "r\u00e9sultat \U0001f389"
        with mock.patch("sys.stdout") as mock_stdout:
            _call_single_shot(fmt="json")
            written = "".join(str(c[0][0]) for c in mock_stdout.write.call_args_list)
            assert "\U0001f389" in written

    def test_text_format_uses_console(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = "panel content"
        _call_single_shot(fmt="text")
        assert patch_console.print.called


# ===========================================================================
# Section 10: File output (--output)
# ===========================================================================


class TestFileOutput:
    """Tests for saving output to a file via --output."""

    def test_text_output_saved_to_file(self, patch_make_agent, fake_agent, patch_history, patch_console, tmp_path):
        fake_agent.return_value = "saved content"
        out_file = tmp_path / "result.txt"
        _call_single_shot(output=out_file, fmt="text")
        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == "saved content"

    def test_json_output_saved_to_file(self, patch_make_agent, fake_agent, patch_history, patch_console, tmp_path):
        fake_agent.return_value = "json content"
        out_file = tmp_path / "result.json"
        with mock.patch("sys.stdout"):
            _call_single_shot(output=out_file, fmt="json")
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data["result"] == "json content"

    def test_output_creates_parent_dirs(self, patch_make_agent, fake_agent, patch_history, patch_console, tmp_path):
        fake_agent.return_value = "nested"
        out_file = tmp_path / "deep" / "nested" / "result.txt"
        _call_single_shot(output=out_file, fmt="text")
        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == "nested"

    def test_stream_output_saved_to_file(self, patch_history, patch_console, tmp_path):
        out_file = tmp_path / "streamed.txt"

        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("no handler")
                agent = mock.MagicMock()
                agent.return_value = "streamed text"
                return agent

            m_make.side_effect = make_side
            with mock.patch("sys.stderr"):
                _call_single_shot(stream=True, output=out_file, fmt="text")

        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == "streamed text"

    def test_no_output_no_file_created(self, patch_make_agent, fake_agent, patch_history, patch_console, tmp_path):
        _call_single_shot(output=None)
        assert list(tmp_path.iterdir()) == []


# ===========================================================================
# Section 11: History recording details
# ===========================================================================


class TestHistoryRecording:
    """Tests for the details passed to _append_history."""

    def test_history_args_on_success(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = "good result"
        _call_single_shot(task="my task", agent_type="browser", mode="single", fmt="json")
        patch_history.assert_called_once()
        args, kwargs = patch_history.call_args
        assert args[0] == "my task"
        assert args[1] == "good result"
        assert kwargs["agent_type"] == "browser"
        assert kwargs["mode"] == "single"
        assert kwargs["success"] is True
        assert kwargs["format"] == "json"

    def test_history_mode_forwarded(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = "r"
        _call_single_shot(mode="auto", task="short")
        kwargs = patch_history.call_args[1]
        assert kwargs["mode"] == "auto"

    def test_history_on_orchestrator_failure(self, patch_make_agent, patch_orchestrator, patch_history, patch_console):
        orch_instance = patch_orchestrator.return_value
        orch_instance.run.return_value = mock.MagicMock(success=False, error="timeout")
        _call_single_shot(mode="multi")
        kwargs = patch_history.call_args[1]
        assert kwargs["success"] is False

    def test_history_agent_type_preserved(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = "x"
        _call_single_shot(agent_type="data")
        kwargs = patch_history.call_args[1]
        assert kwargs["agent_type"] == "data"

    def test_history_format_preserved(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = "x"
        with mock.patch("sys.stdout"):
            _call_single_shot(fmt="json")
        kwargs = patch_history.call_args[1]
        assert kwargs["format"] == "json"


# ===========================================================================
# Section 12: Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge-case tests for _run_single_shot."""

    def test_empty_result_text(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = ""
        rc = _call_single_shot()
        assert rc == 0

    def test_very_long_result(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = "x" * 100_000
        rc = _call_single_shot()
        assert rc == 0

    def test_result_with_newlines(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = "line1\nline2\nline3"
        rc = _call_single_shot()
        assert rc == 0
        assert patch_history.call_args[0][1] == "line1\nline2\nline3"

    def test_result_with_quotes(self, patch_make_agent, fake_agent, patch_history, patch_console):
        fake_agent.return_value = 'He said "hello" end'
        with mock.patch("sys.stdout") as mock_stdout:
            rc = _call_single_shot(fmt="json")
        assert rc == 0
        written = "".join(str(c[0][0]) for c in mock_stdout.write.call_args_list)
        data = json.loads(written)
        assert 'He said "hello"' in data["result"]

    def test_agent_type_forwarded_to_make_agent(self, patch_history, patch_console):
        with mock.patch("manus_agent.cli._make_agent") as m_make:
            m_make.return_value = mock.MagicMock(return_value="r")
            _call_single_shot(agent_type="data")
        assert m_make.call_args[0][0] == "data"

    def test_show_plan_param_accepted(self, patch_make_agent, fake_agent, patch_history, patch_console):
        rc = _call_single_shot(show_plan=True)
        assert rc == 0

    def test_config_forwarded_to_make_agent(self, patch_history, patch_console):
        cfg = mock.MagicMock()
        with mock.patch("manus_agent.cli._make_agent") as m_make:
            m_make.return_value = mock.MagicMock(return_value="r")
            _call_single_shot(config=cfg)
        assert m_make.call_args[0][1] is cfg


# ===========================================================================
# Section 13: Streaming early-return path (stream + text + non-multi)
# ===========================================================================


class TestStreamingEarlyReturn:
    """Tests for the streaming early-return path."""

    def test_stream_text_returns_0_early(self, patch_history, patch_console):
        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("no handler")
                agent = mock.MagicMock()
                agent.return_value = "already streamed"
                return agent

            m_make.side_effect = make_side
            with mock.patch("sys.stderr"):
                rc = _call_single_shot(stream=True, fmt="text")

        assert rc == 0

    def test_stream_text_appends_history(self, patch_history, patch_console):
        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("no handler")
                agent = mock.MagicMock()
                agent.return_value = "stream result"
                return agent

            m_make.side_effect = make_side
            with mock.patch("sys.stderr"):
                _call_single_shot(stream=True, fmt="text", no_history=False)

        patch_history.assert_called_once()

    def test_stream_text_no_history_flag(self, patch_history, patch_console):
        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("no handler")
                agent = mock.MagicMock()
                agent.return_value = "no history"
                return agent

            m_make.side_effect = make_side
            with mock.patch("sys.stderr"):
                _call_single_shot(stream=True, fmt="text", no_history=True)

        patch_history.assert_not_called()

    def test_stream_json_still_outputs_json(self, patch_history, patch_console):
        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("no handler")
                agent = mock.MagicMock()
                agent.return_value = "json data"
                return agent

            m_make.side_effect = make_side
            with mock.patch("sys.stderr"):
                with mock.patch("sys.stdout") as mock_stdout:
                    rc = _call_single_shot(stream=True, fmt="json")

        assert rc == 0
        written = "".join(str(c[0][0]) for c in mock_stdout.write.call_args_list)
        data = json.loads(written)
        assert data["result"] == "json data"

    def test_stream_multi_does_not_use_early_return(
        self, patch_make_agent, patch_orchestrator, patch_history, patch_console
    ):
        orch_instance = patch_orchestrator.return_value
        orch_instance.run.return_value = mock.MagicMock(success=True, output="multi streamed")
        _call_single_shot(stream=True, mode="multi", fmt="text")
        assert patch_history.called

    def test_stream_file_save_in_early_return(self, patch_history, patch_console, tmp_path):
        out_file = tmp_path / "stream_out.txt"

        with mock.patch("manus_agent.cli._make_agent") as m_make:

            def make_side(*args, **kwargs):
                if "callback_handler" in kwargs:
                    raise TypeError("no handler")
                agent = mock.MagicMock()
                agent.return_value = "early return file"
                return agent

            m_make.side_effect = make_side
            with mock.patch("sys.stderr"):
                _call_single_shot(stream=True, fmt="text", output=out_file)

        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == "early return file"
