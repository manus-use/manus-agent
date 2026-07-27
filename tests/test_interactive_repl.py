"""Comprehensive test suite for the _run_interactive REPL loop.

Tests cover:
- Welcome banner and initialisation messaging
- Agent and orchestrator creation (success + failure paths)
- Exit/quit/bye commands terminate the loop
- Single-agent mode (mode="single") routes to agent directly
- Multi-agent mode (mode="multi") always uses orchestrator
- Auto mode routes simple tasks to agent, complex tasks to orchestrator
- Orchestrator success/failure rendering (Panel output)
- show_plan=True triggers display_task_plan
- KeyboardInterrupt handling (graceful exit)
- Exception handling during agent execution (prints error, continues)
- Exception during agent initialisation (sys.exit(1))

All tests are fully mocked — no real LLM calls, no real agents.
"""

from __future__ import annotations

from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_config():
    """Return a mock Config object suitable for _run_interactive."""
    return mock.MagicMock()


def _make_mock_agent():
    """Return a mock agent that returns a response when called."""
    agent = mock.MagicMock()
    agent.return_value = "Agent response text"
    return agent


def _make_mock_orchestrator(*, success=True, output="Orchestrated result", error=None):
    """Return a mock Orchestrator whose .run() returns a result object."""
    orch = mock.MagicMock()
    result = mock.MagicMock()
    result.success = success
    result.output = output
    result.error = error
    orch.run.return_value = result
    return orch


def _invoke_interactive(
    user_inputs: list[str],
    *,
    mode: str = "single",
    agent_type: str = "manus",
    show_plan: bool = False,
    make_agent_side_effect=None,
    orchestrator=None,
    agent=None,
):
    """Run _run_interactive with mocked Prompt.ask providing user_inputs.

    Returns a dict with captured console output lines and call metadata.
    """
    from manus_agent import cli

    config = _make_mock_config()
    mock_agent = agent or _make_mock_agent()
    mock_orch = orchestrator or _make_mock_orchestrator()

    # Prompt.ask will yield from user_inputs
    input_iter = iter(user_inputs)

    printed = []
    raw_args = []  # list of tuples of positional args to console.print

    def fake_prompt_ask(*args, **kwargs):
        try:
            return next(input_iter)
        except StopIteration:
            # Safety: force exit if inputs exhausted
            return "exit"

    def capture_print(*args, **kwargs):
        printed.append(str(args))
        raw_args.append(args)

    with (
        mock.patch.object(
            cli,
            "_make_agent",
            side_effect=make_agent_side_effect if make_agent_side_effect else lambda *a, **kw: mock_agent,
        ) as m_make_agent,
        mock.patch.object(cli, "Orchestrator", return_value=mock_orch) as m_orch_cls,
        mock.patch("rich.prompt.Prompt.ask", side_effect=fake_prompt_ask),
        mock.patch.object(cli.console, "print", side_effect=capture_print),
        mock.patch.object(
            cli.console, "status", return_value=mock.MagicMock(__enter__=mock.MagicMock(), __exit__=mock.MagicMock())
        ),
    ):
        try:
            cli._run_interactive(
                mode=mode,
                agent_type=agent_type,
                show_plan=show_plan,
                config=config,
            )
        except SystemExit as exc:
            return {
                "printed": printed,
                "raw_args": raw_args,
                "exit_code": exc.code,
                "agent": mock_agent,
                "orchestrator": mock_orch,
                "make_agent": m_make_agent,
                "orch_cls": m_orch_cls,
            }

    return {
        "printed": printed,
        "raw_args": raw_args,
        "exit_code": None,
        "agent": mock_agent,
        "orchestrator": mock_orch,
        "make_agent": m_make_agent,
        "orch_cls": m_orch_cls,
    }


# ---------------------------------------------------------------------------
# Welcome banner tests
# ---------------------------------------------------------------------------


class TestWelcomeBanner:
    def test_prints_welcome_message(self):
        """REPL prints 'Welcome to ManusUse!' on start."""
        result = _invoke_interactive(["exit"])
        output = " ".join(result["printed"])
        assert "Welcome to ManusUse" in output

    def test_prints_mode_info(self):
        """REPL prints the configured mode on start."""
        result = _invoke_interactive(["exit"], mode="multi")
        output = " ".join(result["printed"])
        assert "multi" in output

    def test_prints_agent_type_info(self):
        """REPL prints the configured agent type on start."""
        result = _invoke_interactive(["exit"], agent_type="browser")
        output = " ".join(result["printed"])
        assert "browser" in output

    def test_prints_initialising_message(self):
        """REPL prints 'Initialising agents…' during setup."""
        result = _invoke_interactive(["exit"])
        output = " ".join(result["printed"])
        assert "Initialising" in output or "nitialis" in output

    def test_prints_success_after_init(self):
        """REPL prints a success marker after agent init."""
        result = _invoke_interactive(["exit"])
        output = " ".join(result["printed"])
        assert "initialised" in output.lower() or "✓" in output

    def test_prints_type_prompt_instruction(self):
        """REPL tells user to type requests."""
        result = _invoke_interactive(["exit"])
        output = " ".join(result["printed"])
        assert "exit" in output.lower() or "quit" in output.lower()


# ---------------------------------------------------------------------------
# Agent initialisation
# ---------------------------------------------------------------------------


class TestAgentInitialisation:
    def test_make_agent_called_with_type_and_config(self):
        """_make_agent is called with agent_type and config."""
        result = _invoke_interactive(["exit"], agent_type="manus")
        result["make_agent"].assert_called_once()
        args = result["make_agent"].call_args
        assert args[0][0] == "manus"

    def test_orchestrator_created_in_multi_mode(self):
        """Orchestrator is instantiated when mode='multi'."""
        result = _invoke_interactive(["exit"], mode="multi")
        result["orch_cls"].assert_called_once()

    def test_orchestrator_created_in_auto_mode(self):
        """Orchestrator is instantiated when mode='auto'."""
        result = _invoke_interactive(["exit"], mode="auto")
        result["orch_cls"].assert_called_once()

    def test_orchestrator_not_created_in_single_mode(self):
        """Orchestrator is NOT instantiated when mode='single'."""
        result = _invoke_interactive(["exit"], mode="single")
        result["orch_cls"].assert_not_called()

    def test_agent_init_failure_exits_1(self):
        """If _make_agent raises, REPL calls sys.exit(1)."""
        result = _invoke_interactive(
            ["exit"],
            make_agent_side_effect=RuntimeError("model not found"),
        )
        assert result["exit_code"] == 1

    def test_agent_init_failure_prints_error(self):
        """If _make_agent raises, error message is printed."""
        result = _invoke_interactive(
            ["exit"],
            make_agent_side_effect=RuntimeError("model not found"),
        )
        output = " ".join(result["printed"])
        assert "model not found" in output or "Failed" in output


# ---------------------------------------------------------------------------
# Exit commands
# ---------------------------------------------------------------------------


class TestExitCommands:
    @pytest.mark.parametrize("cmd", ["exit", "quit", "bye"])
    def test_exit_commands_terminate_loop(self, cmd):
        """'exit', 'quit', and 'bye' all terminate the REPL."""
        agent = _make_mock_agent()
        _invoke_interactive([cmd], agent=agent)
        # Agent should NOT have been called — loop exits before processing
        agent.assert_not_called()

    @pytest.mark.parametrize("cmd", ["Exit", "QUIT", "BYE", "Bye"])
    def test_exit_commands_case_insensitive(self, cmd):
        """Exit commands are case-insensitive."""
        agent = _make_mock_agent()
        _invoke_interactive([cmd], agent=agent)
        agent.assert_not_called()

    def test_exit_prints_goodbye(self):
        """Exit prints a goodbye message."""
        result = _invoke_interactive(["exit"])
        output = " ".join(result["printed"])
        assert "Goodbye" in output or "goodbye" in output


# ---------------------------------------------------------------------------
# Single-agent mode
# ---------------------------------------------------------------------------


class TestSingleAgentMode:
    def test_task_routed_to_agent(self):
        """In single mode, user input is passed directly to the agent."""
        agent = _make_mock_agent()
        _invoke_interactive(["hello world", "exit"], mode="single", agent=agent)
        agent.assert_called_once_with("hello world")

    def test_agent_response_printed(self):
        """Agent's response is printed to the console."""
        agent = _make_mock_agent()
        agent.return_value = "Here is my answer"
        result = _invoke_interactive(["question", "exit"], mode="single", agent=agent)
        output = " ".join(result["printed"])
        assert "Here is my answer" in output

    def test_multiple_turns_in_single_mode(self):
        """Multiple user inputs are processed sequentially."""
        agent = _make_mock_agent()
        agent.side_effect = ["response 1", "response 2"]
        _invoke_interactive(
            ["first question", "second question", "exit"],
            mode="single",
            agent=agent,
        )
        assert agent.call_count == 2
        agent.assert_any_call("first question")
        agent.assert_any_call("second question")

    def test_orchestrator_not_used_in_single_mode(self):
        """Orchestrator.run() is never called in single mode."""
        orch = _make_mock_orchestrator()
        _invoke_interactive(
            ["analyze and compare and research", "exit"],
            mode="single",
            orchestrator=orch,
        )
        orch.run.assert_not_called()


# ---------------------------------------------------------------------------
# Multi-agent mode
# ---------------------------------------------------------------------------


class TestMultiAgentMode:
    def test_task_routed_to_orchestrator(self):
        """In multi mode, user input is passed to orchestrator.run()."""
        orch = _make_mock_orchestrator()
        _invoke_interactive(["simple task", "exit"], mode="multi", orchestrator=orch)
        orch.run.assert_called_once_with("simple task")

    def test_orchestrator_success_output_printed(self):
        """Successful orchestrator output is rendered (Panel with result)."""
        from rich.panel import Panel

        orch = _make_mock_orchestrator(success=True, output="Multi-agent result here")
        result = _invoke_interactive(["do stuff", "exit"], mode="multi", orchestrator=orch)
        # Check that a Panel containing the result was printed
        panels = [a for args in result["raw_args"] for a in args if isinstance(a, Panel)]
        assert len(panels) >= 1
        # The result text should be in one of the panels' renderable
        panel_contents = [str(p.renderable) for p in panels]
        assert any("Multi-agent result here" in c for c in panel_contents)

    def test_orchestrator_failure_shows_error(self):
        """Failed orchestrator result shows the error message in a Panel."""
        from rich.panel import Panel

        orch = _make_mock_orchestrator(success=False, error="Timeout in sub-agent")
        result = _invoke_interactive(["complex task", "exit"], mode="multi", orchestrator=orch)
        panels = [a for args in result["raw_args"] for a in args if isinstance(a, Panel)]
        assert len(panels) >= 1
        panel_contents = [str(p.renderable) for p in panels]
        assert any("Timeout in sub-agent" in c for c in panel_contents)

    def test_agent_not_called_directly_in_multi_mode(self):
        """In multi mode, the agent is not called directly for user tasks."""
        agent = _make_mock_agent()
        orch = _make_mock_orchestrator()
        _invoke_interactive(["task", "exit"], mode="multi", agent=agent, orchestrator=orch)
        agent.assert_not_called()


# ---------------------------------------------------------------------------
# Auto mode
# ---------------------------------------------------------------------------


class TestAutoMode:
    def test_simple_task_routes_to_agent(self):
        """In auto mode, a simple task routes to the single agent."""
        agent = _make_mock_agent()
        orch = _make_mock_orchestrator()
        # "hello" is simple — no complex indicators
        _invoke_interactive(["hello", "exit"], mode="auto", agent=agent, orchestrator=orch)
        agent.assert_called_once_with("hello")
        orch.run.assert_not_called()

    def test_complex_task_routes_to_orchestrator(self):
        """In auto mode, a complex task routes to the orchestrator."""
        agent = _make_mock_agent()
        orch = _make_mock_orchestrator()
        # This triggers the "and...and" pattern in is_complex_task
        complex_input = "analyze the data and create a report and then summarize it"
        _invoke_interactive([complex_input, "exit"], mode="auto", agent=agent, orchestrator=orch)
        orch.run.assert_called_once_with(complex_input)
        agent.assert_not_called()

    def test_long_task_routes_to_orchestrator(self):
        """In auto mode, a task >30 words routes to the orchestrator."""
        agent = _make_mock_agent()
        orch = _make_mock_orchestrator()
        long_input = " ".join(["word"] * 35)
        _invoke_interactive([long_input, "exit"], mode="auto", agent=agent, orchestrator=orch)
        orch.run.assert_called_once_with(long_input)
        agent.assert_not_called()

    def test_multi_sentence_routes_to_orchestrator(self):
        """In auto mode, input with >2 sentences routes to orchestrator."""
        agent = _make_mock_agent()
        orch = _make_mock_orchestrator()
        multi_sentence = "Do this first. Then do that. Finally check the result."
        _invoke_interactive([multi_sentence, "exit"], mode="auto", agent=agent, orchestrator=orch)
        orch.run.assert_called_once_with(multi_sentence)
        agent.assert_not_called()

    def test_workflow_keyword_routes_to_orchestrator(self):
        """In auto mode, a task with 'workflow' routes to the orchestrator."""
        agent = _make_mock_agent()
        orch = _make_mock_orchestrator()
        _invoke_interactive(
            ["run the deployment workflow", "exit"],
            mode="auto",
            agent=agent,
            orchestrator=orch,
        )
        orch.run.assert_called_once()
        agent.assert_not_called()


# ---------------------------------------------------------------------------
# show_plan flag
# ---------------------------------------------------------------------------


class TestShowPlan:
    def test_show_plan_calls_display_task_plan(self):
        """show_plan=True in multi mode triggers display_task_plan."""
        from manus_agent import cli

        orch = _make_mock_orchestrator()
        # Give the orchestrator a planner with create_plan
        planner = mock.MagicMock()
        task_obj = mock.MagicMock()
        task_obj.task_id = "1"
        task_obj.description = "Test task"
        task_obj.agent_type = mock.MagicMock(value="manus")
        task_obj.dependencies = []
        planner.create_plan.return_value = [task_obj]
        orch.agents = {"planner": planner}

        with mock.patch.object(cli, "display_task_plan") as m_display:
            _invoke_interactive(
                ["do stuff", "exit"],
                mode="multi",
                show_plan=True,
                orchestrator=orch,
            )
            m_display.assert_called_once_with([task_obj])

    def test_show_plan_false_skips_display(self):
        """show_plan=False does NOT call display_task_plan."""
        from manus_agent import cli

        orch = _make_mock_orchestrator()

        with mock.patch.object(cli, "display_task_plan") as m_display:
            _invoke_interactive(
                ["do stuff", "exit"],
                mode="multi",
                show_plan=False,
                orchestrator=orch,
            )
            m_display.assert_not_called()

    def test_show_plan_no_planner_agent_graceful(self):
        """show_plan=True handles missing planner gracefully."""
        orch = _make_mock_orchestrator()
        orch.agents = {}  # No planner

        # Should not raise
        result = _invoke_interactive(
            ["task", "exit"],
            mode="multi",
            show_plan=True,
            orchestrator=orch,
        )
        # Just confirm we reached the exit without error
        output = " ".join(result["printed"])
        assert "Goodbye" in output


# ---------------------------------------------------------------------------
# Error handling during execution
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_agent_exception_prints_error(self):
        """Exception from agent is caught and printed, loop continues."""
        agent = _make_mock_agent()
        agent.side_effect = [RuntimeError("connection timeout"), "ok response"]
        result = _invoke_interactive(
            ["failing task", "working task", "exit"],
            mode="single",
            agent=agent,
        )
        output = " ".join(result["printed"])
        assert "connection timeout" in output or "Error" in output
        # Agent was called twice (once for each task)
        assert agent.call_count == 2

    def test_orchestrator_exception_prints_error(self):
        """Exception from orchestrator is caught, loop continues."""
        agent = _make_mock_agent()
        orch = _make_mock_orchestrator()
        orch.run.side_effect = [ValueError("bad workflow"), mock.MagicMock(success=True, output="ok")]
        result = _invoke_interactive(
            ["task1", "task2", "exit"],
            mode="multi",
            agent=agent,
            orchestrator=orch,
        )
        output = " ".join(result["printed"])
        assert "bad workflow" in output or "Error" in output
        # Orchestrator was called twice
        assert orch.run.call_count == 2

    def test_multiple_errors_dont_crash_loop(self):
        """Multiple consecutive errors don't crash the REPL."""
        agent = _make_mock_agent()
        agent.side_effect = [
            RuntimeError("error 1"),
            RuntimeError("error 2"),
            RuntimeError("error 3"),
            "finally works",
        ]
        _invoke_interactive(
            ["a", "b", "c", "d", "exit"],
            mode="single",
            agent=agent,
        )
        assert agent.call_count == 4


# ---------------------------------------------------------------------------
# KeyboardInterrupt handling
# ---------------------------------------------------------------------------


class TestKeyboardInterrupt:
    def test_keyboard_interrupt_exits_gracefully(self):
        """Ctrl+C (KeyboardInterrupt) exits the REPL with a goodbye."""
        from manus_agent import cli

        config = _make_mock_config()
        mock_agent = _make_mock_agent()

        printed = []
        call_count = [0]

        def fake_prompt_ask(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise KeyboardInterrupt()
            return "exit"

        with (
            mock.patch.object(cli, "_make_agent", return_value=mock_agent),
            mock.patch.object(cli, "Orchestrator", return_value=_make_mock_orchestrator()),
            mock.patch("rich.prompt.Prompt.ask", side_effect=fake_prompt_ask),
            mock.patch.object(cli.console, "print", side_effect=lambda *a, **kw: printed.append(str(a))),
            mock.patch.object(
                cli.console,
                "status",
                return_value=mock.MagicMock(__enter__=mock.MagicMock(), __exit__=mock.MagicMock()),
            ),
        ):
            cli._run_interactive(
                mode="single",
                agent_type="manus",
                show_plan=False,
                config=config,
            )

        output = " ".join(printed)
        assert "Goodbye" in output

    def test_keyboard_interrupt_during_agent_call(self):
        """KeyboardInterrupt during agent execution exits gracefully."""
        from manus_agent import cli

        config = _make_mock_config()
        mock_agent = _make_mock_agent()
        mock_agent.side_effect = KeyboardInterrupt()

        printed = []
        input_given = [False]

        def fake_prompt_ask(*args, **kwargs):
            if not input_given[0]:
                input_given[0] = True
                return "some task"
            return "exit"

        with (
            mock.patch.object(cli, "_make_agent", return_value=mock_agent),
            mock.patch.object(cli, "Orchestrator", return_value=_make_mock_orchestrator()),
            mock.patch("rich.prompt.Prompt.ask", side_effect=fake_prompt_ask),
            mock.patch.object(cli.console, "print", side_effect=lambda *a, **kw: printed.append(str(a))),
            mock.patch.object(
                cli.console,
                "status",
                return_value=mock.MagicMock(__enter__=mock.MagicMock(), __exit__=mock.MagicMock()),
            ),
        ):
            cli._run_interactive(
                mode="single",
                agent_type="manus",
                show_plan=False,
                config=config,
            )

        output = " ".join(printed)
        assert "Goodbye" in output


# ---------------------------------------------------------------------------
# Progress spinner in multi-agent mode
# ---------------------------------------------------------------------------


class TestProgressSpinner:
    def test_multi_mode_uses_progress_context(self):
        """Multi-agent mode wraps orchestrator.run in a Progress context."""
        from manus_agent import cli

        config = _make_mock_config()
        mock_agent = _make_mock_agent()
        orch = _make_mock_orchestrator()

        inputs = iter(["do thing", "exit"])

        with (
            mock.patch.object(cli, "_make_agent", return_value=mock_agent),
            mock.patch.object(cli, "Orchestrator", return_value=orch),
            mock.patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(inputs)),
            mock.patch.object(cli.console, "print"),
            mock.patch.object(
                cli.console,
                "status",
                return_value=mock.MagicMock(__enter__=mock.MagicMock(), __exit__=mock.MagicMock()),
            ),
            mock.patch("manus_agent.cli.Progress") as m_progress,
        ):
            m_progress.return_value.__enter__ = mock.MagicMock()
            m_progress.return_value.__exit__ = mock.MagicMock(return_value=False)
            m_progress.return_value.add_task = mock.MagicMock()
            m_progress.return_value.update = mock.MagicMock()

            cli._run_interactive(
                mode="multi",
                agent_type="manus",
                show_plan=False,
                config=config,
            )

        m_progress.assert_called()


# ---------------------------------------------------------------------------
# Console status context in single-agent mode
# ---------------------------------------------------------------------------


class TestConsoleStatus:
    def test_single_mode_uses_console_status(self):
        """Single-agent mode wraps agent() in console.status context."""
        from manus_agent import cli

        config = _make_mock_config()
        mock_agent = _make_mock_agent()

        inputs = iter(["hello", "exit"])
        status_mock = mock.MagicMock()

        with (
            mock.patch.object(cli, "_make_agent", return_value=mock_agent),
            mock.patch.object(cli, "Orchestrator", return_value=_make_mock_orchestrator()),
            mock.patch("rich.prompt.Prompt.ask", side_effect=lambda *a, **kw: next(inputs)),
            mock.patch.object(cli.console, "print"),
            mock.patch.object(cli.console, "status", return_value=status_mock) as m_status,
        ):
            cli._run_interactive(
                mode="single",
                agent_type="manus",
                show_plan=False,
                config=config,
            )

        m_status.assert_called()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_input_still_processed(self):
        """Empty string input is still passed to the agent (not exit)."""
        agent = _make_mock_agent()
        _invoke_interactive(["", "exit"], mode="single", agent=agent)
        agent.assert_called_once_with("")

    def test_whitespace_input_not_treated_as_exit(self):
        """Whitespace-only input is not treated as an exit command."""
        agent = _make_mock_agent()
        _invoke_interactive(["   ", "exit"], mode="single", agent=agent)
        agent.assert_called_once_with("   ")

    def test_exit_with_extra_chars_not_exit(self):
        """'exit!' or 'exitnow' are not treated as exit commands."""
        agent = _make_mock_agent()
        _invoke_interactive(["exit!", "exit"], mode="single", agent=agent)
        agent.assert_called_once_with("exit!")

    def test_very_long_input(self):
        """Very long input is handled without issues."""
        agent = _make_mock_agent()
        long_input = "x" * 10000
        _invoke_interactive([long_input, "exit"], mode="single", agent=agent)
        agent.assert_called_once_with(long_input)

    def test_special_characters_in_input(self):
        """Special characters in input are passed through."""
        agent = _make_mock_agent()
        special = "こんにちは 🌍 <script>alert(1)</script>"
        _invoke_interactive([special, "exit"], mode="single", agent=agent)
        agent.assert_called_once_with(special)


# ---------------------------------------------------------------------------
# Orchestrator planning display details
# ---------------------------------------------------------------------------


class TestOrchestratorPlanningDisplay:
    def test_planning_message_printed_in_multi_mode(self):
        """In multi mode, 'Planning execution…' message is printed."""
        orch = _make_mock_orchestrator()
        result = _invoke_interactive(["task", "exit"], mode="multi", orchestrator=orch)
        output = " ".join(result["printed"])
        assert "Planning" in output or "Orchestrator" in output

    def test_success_result_in_panel(self):
        """Successful result is displayed in a Panel (contains 'Result' title)."""
        from rich.panel import Panel

        orch = _make_mock_orchestrator(success=True, output="final answer")
        result = _invoke_interactive(["task", "exit"], mode="multi", orchestrator=orch)
        panels = [a for args in result["raw_args"] for a in args if isinstance(a, Panel)]
        assert len(panels) >= 1
        panel_contents = [str(p.renderable) for p in panels]
        assert any("final answer" in c for c in panel_contents)

    def test_error_result_in_panel(self):
        """Failed result displays an error Panel."""
        from rich.panel import Panel

        orch = _make_mock_orchestrator(success=False, error="Sub-agent crashed")
        result = _invoke_interactive(["task", "exit"], mode="multi", orchestrator=orch)
        panels = [a for args in result["raw_args"] for a in args if isinstance(a, Panel)]
        assert len(panels) >= 1
        panel_contents = [str(p.renderable) for p in panels]
        assert any("Sub-agent crashed" in c for c in panel_contents)


# ---------------------------------------------------------------------------
# Mode interaction with orchestrator=None edge case
# ---------------------------------------------------------------------------


class TestNullOrchestratorInSingleMode:
    def test_single_mode_never_calls_orchestrator(self):
        """Even with complex input, single mode doesn't use orchestrator."""
        agent = _make_mock_agent()
        complex_input = "analyze and compare and research the data then generate a report"
        result = _invoke_interactive(
            [complex_input, "exit"],
            mode="single",
            agent=agent,
        )
        agent.assert_called_once_with(complex_input)
        result["orchestrator"].run.assert_not_called()
