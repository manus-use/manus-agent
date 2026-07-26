"""Comprehensive test suite for CLI dispatch helpers: is_complex_task, display_task_plan, _make_agent.

These three functions form the routing core of the manus-agent CLI:
- is_complex_task: heuristic that decides single-agent vs multi-agent mode
- display_task_plan: renders the multi-agent execution plan table
- _make_agent: factory that instantiates the correct agent class

All tests are fully mocked — no real HTTP calls or agent instantiation.
"""

from __future__ import annotations

from unittest import mock

import pytest

from manus_agent.cli import _make_agent, display_task_plan, is_complex_task
from manus_agent.multi_agents import AgentType, TaskPlan

# ===========================================================================
# is_complex_task — complexity heuristic
# ===========================================================================


class TestIsComplexTaskWordCount:
    """Tasks with >30 words should be classified as complex."""

    def test_31_words_is_complex(self):
        task = " ".join(["word"] * 31)
        assert is_complex_task(task) is True

    def test_30_words_not_complex_alone(self):
        # 30 words without any other indicator
        task = " ".join(["hello"] * 30)
        assert is_complex_task(task) is False

    def test_one_word_not_complex(self):
        assert is_complex_task("hello") is False

    def test_empty_string_not_complex(self):
        assert is_complex_task("") is False

    def test_exactly_31_words_complex(self):
        task = " ".join(["x"] * 31)
        assert is_complex_task(task) is True

    def test_50_words_complex(self):
        task = " ".join(["data"] * 50)
        assert is_complex_task(task) is True


class TestIsComplexTaskSentenceCount:
    """Tasks with >2 sentences (split by . or ;) should be complex."""

    def test_three_period_sentences_complex(self):
        assert is_complex_task("Do this. Do that. All done.") is True

    def test_three_semicolon_sentences_complex(self):
        assert is_complex_task("apples; oranges; bananas") is True

    def test_two_sentences_not_complex(self):
        # Two sentences without indicator words → not complex
        assert is_complex_task("The cat sat. The dog ran") is False

    def test_mixed_periods_semicolons(self):
        assert is_complex_task("apples. oranges; bananas") is True

    def test_single_sentence_not_complex(self):
        assert is_complex_task("Write a hello world script") is False


class TestIsComplexTaskPatternAndThen:
    """The 'and ... and' pattern should trigger complexity."""

    def test_two_ands_complex(self):
        assert is_complex_task("read this and process it and output results") is True

    def test_one_and_not_complex(self):
        assert is_complex_task("read and process") is False

    def test_and_at_word_boundaries(self):
        # 'sand' should not match \band\b
        assert is_complex_task("sandbox handler") is False


class TestIsComplexTaskPatternThen:
    """The word 'then' should trigger complexity."""

    def test_then_triggers(self):
        assert is_complex_task("do this then do that") is True

    def test_then_case_insensitive(self):
        assert is_complex_task("Do this THEN do that") is True

    def test_then_at_boundaries(self):
        # 'authenticate' should not match \bthen\b
        assert is_complex_task("authenticate user") is False


class TestIsComplexTaskPatternAfter:
    """The word 'after' should trigger complexity."""

    def test_after_triggers(self):
        assert is_complex_task("after downloading, parse the file") is True

    def test_after_case_insensitive(self):
        assert is_complex_task("AFTER the build completes, deploy") is True

    def test_hereafter_no_match(self):
        # 'hereafter' — \bafter\b does NOT match because 'after' is not
        # at a word boundary (preceded by 'e' without a boundary)
        assert is_complex_task("see note hereafter") is False

    def test_afterword_no_match(self):
        # 'afterword' — \bafter\b does NOT match because 'w' follows 'r'
        # without a boundary
        assert is_complex_task("read the afterword") is False


class TestIsComplexTaskPatternAnalyzeCreate:
    """analyze ... create/generate/build should trigger."""

    def test_analyze_and_create(self):
        assert is_complex_task("analyze the data and create a report") is True

    def test_analyze_and_generate(self):
        assert is_complex_task("analyze logs and generate summary") is True

    def test_analyze_and_build(self):
        assert is_complex_task("analyze requirements and build the app") is True

    def test_analyze_alone_not_complex(self):
        assert is_complex_task("analyze the data") is False


class TestIsComplexTaskPatternCompareSummarize:
    """compare ... summarize should trigger."""

    def test_compare_summarize(self):
        assert is_complex_task("compare the options and summarize findings") is True

    def test_compare_alone_not_complex(self):
        assert is_complex_task("compare these two files") is False


class TestIsComplexTaskPatternMultiple:
    """The word 'multiple' should trigger."""

    def test_multiple_triggers(self):
        assert is_complex_task("process multiple files") is True

    def test_multiple_case_insensitive(self):
        assert is_complex_task("handle MULTIPLE requests") is True


class TestIsComplexTaskPatternSteps:
    """The word 'step' or 'steps' should trigger."""

    def test_steps_triggers(self):
        assert is_complex_task("follow these steps") is True

    def test_step_singular_triggers(self):
        assert is_complex_task("the next step is") is True

    def test_stepping_no_match(self):
        # 'stepping' — \bsteps?\b should not match 'stepping'
        assert is_complex_task("stepping stone") is False


class TestIsComplexTaskPatternWorkflow:
    """The word 'workflow' should trigger."""

    def test_workflow_triggers(self):
        assert is_complex_task("create a workflow") is True

    def test_workflow_case_insensitive(self):
        assert is_complex_task("design the WORKFLOW") is True


class TestIsComplexTaskPatternOrdinals:
    """Ordinals (first, second, third, finally) should trigger."""

    def test_first_triggers(self):
        assert is_complex_task("first do A") is True

    def test_second_triggers(self):
        assert is_complex_task("the second task is") is True

    def test_third_triggers(self):
        assert is_complex_task("third, compile") is True

    def test_finally_triggers(self):
        assert is_complex_task("finally output the result") is True


class TestIsComplexTaskPatternVisualize:
    """visualize/chart/graph ... analyze/data should trigger."""

    def test_visualize_data(self):
        assert is_complex_task("visualize the data") is True

    def test_chart_analyze(self):
        assert is_complex_task("chart the analyze results") is True

    def test_graph_data(self):
        assert is_complex_task("graph the data trends") is True

    def test_visualise_british_spelling(self):
        assert is_complex_task("visualise the data") is True


class TestIsComplexTaskPatternBrowse:
    """browse ... extract/analyze should trigger."""

    def test_browse_extract(self):
        assert is_complex_task("browse the site and extract links") is True

    def test_browse_analyze(self):
        assert is_complex_task("browse documentation and analyze usage") is True

    def test_browse_alone_not_complex(self):
        assert is_complex_task("browse the web") is False


class TestIsComplexTaskPatternResearch:
    """research ... implement/create should trigger."""

    def test_research_implement(self):
        assert is_complex_task("research the API and implement it") is True

    def test_research_create(self):
        assert is_complex_task("research best practices and create a guide") is True

    def test_research_alone_not_complex(self):
        assert is_complex_task("research quantum computing") is False


class TestIsComplexTaskSimpleTasks:
    """Simple tasks should not be detected as complex."""

    def test_simple_task(self):
        assert is_complex_task("write a hello world script") is False

    def test_short_question(self):
        assert is_complex_task("what is Python?") is False

    def test_single_action(self):
        assert is_complex_task("list all files") is False

    def test_two_word_task(self):
        assert is_complex_task("fix bug") is False


# ===========================================================================
# display_task_plan — execution plan table rendering
# ===========================================================================


class TestDisplayTaskPlanBasic:
    """Basic rendering of task plans."""

    def test_single_task_no_deps(self, capsys):
        tasks = [
            TaskPlan(
                task_id="T1",
                description="Fetch data from API",
                agent_type=AgentType.MANUS,
            )
        ]
        display_task_plan(tasks)
        captured = capsys.readouterr()
        assert "Execution Plan" in captured.out
        assert "T1" in captured.out
        assert "Fetch data from API" in captured.out
        assert "manus" in captured.out
        assert "None" in captured.out

    def test_multiple_tasks_with_deps(self, capsys):
        tasks = [
            TaskPlan(
                task_id="T1",
                description="Download dataset",
                agent_type=AgentType.BROWSER,
            ),
            TaskPlan(
                task_id="T2",
                description="Analyze results",
                agent_type=AgentType.DATA_ANALYSIS,
                dependencies=["T1"],
            ),
            TaskPlan(
                task_id="T3",
                description="Generate report",
                agent_type=AgentType.MANUS,
                dependencies=["T1", "T2"],
            ),
        ]
        display_task_plan(tasks)
        captured = capsys.readouterr()
        assert "T1" in captured.out
        assert "T2" in captured.out
        assert "T3" in captured.out
        assert "browser" in captured.out
        assert "data_analysis" in captured.out
        assert "manus" in captured.out

    def test_long_description_truncated(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")
        long_desc = "A" * 100
        tasks = [
            TaskPlan(
                task_id="T1",
                description=long_desc,
                agent_type=AgentType.MANUS,
            )
        ]
        display_task_plan(tasks)
        captured = capsys.readouterr()
        # The code truncates to 50 chars + "..." before passing to Rich
        assert "..." in captured.out
        # Full 100-char description should NOT appear
        assert long_desc not in captured.out

    def test_exactly_50_char_description_not_truncated(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")
        desc = "A" * 50
        tasks = [
            TaskPlan(
                task_id="T1",
                description=desc,
                agent_type=AgentType.MANUS,
            )
        ]
        display_task_plan(tasks)
        captured = capsys.readouterr()
        # 50-char desc is not truncated by the code (len <= 50)
        # Rich may add its own ellipsis if terminal is narrow, but with wide COLUMNS it fits
        assert desc in captured.out

    def test_51_char_description_truncated(self, capsys, monkeypatch):
        monkeypatch.setenv("COLUMNS", "200")
        desc = "B" * 51
        tasks = [
            TaskPlan(
                task_id="T1",
                description=desc,
                agent_type=AgentType.MANUS,
            )
        ]
        display_task_plan(tasks)
        captured = capsys.readouterr()
        # Code truncates to first 50 chars + "..."
        assert "B" * 50 + "..." in captured.out

    def test_empty_task_list(self, capsys):
        display_task_plan([])
        captured = capsys.readouterr()
        # Should still render the table header
        assert "Execution Plan" in captured.out

    def test_dependencies_displayed_comma_separated(self, capsys):
        tasks = [
            TaskPlan(
                task_id="T5",
                description="Final step",
                agent_type=AgentType.MANUS,
                dependencies=["T1", "T2", "T3"],
            )
        ]
        display_task_plan(tasks)
        captured = capsys.readouterr()
        assert "T1, T2, T3" in captured.out

    def test_mcp_agent_type(self, capsys):
        tasks = [
            TaskPlan(
                task_id="T1",
                description="Use MCP tools",
                agent_type=AgentType.MCP,
            )
        ]
        display_task_plan(tasks)
        captured = capsys.readouterr()
        assert "mcp" in captured.out


# ===========================================================================
# _make_agent — agent factory
# ===========================================================================


class TestMakeAgentBrowser:
    """_make_agent('browser', ...) should instantiate BrowserUseAgent."""

    def test_creates_browser_agent(self):
        mock_config = mock.MagicMock()
        with mock.patch("manus_agent.agents.BrowserUseAgent") as mock_agent_cls:
            mock_agent_cls.return_value = mock.MagicMock()
            result = _make_agent("browser", mock_config)
            mock_agent_cls.assert_called_once_with(config=mock_config)
            assert result == mock_agent_cls.return_value

    def test_forwards_kwargs(self):
        mock_config = mock.MagicMock()
        handler = mock.MagicMock()
        with mock.patch("manus_agent.agents.BrowserUseAgent") as mock_agent_cls:
            mock_agent_cls.return_value = mock.MagicMock()
            _make_agent("browser", mock_config, callback_handler=handler)
            mock_agent_cls.assert_called_once_with(config=mock_config, callback_handler=handler)


class TestMakeAgentData:
    """_make_agent('data', ...) should instantiate DataAnalysisAgent."""

    def test_creates_data_agent(self):
        mock_config = mock.MagicMock()
        with mock.patch("manus_agent.agents.DataAnalysisAgent") as mock_agent_cls:
            mock_agent_cls.return_value = mock.MagicMock()
            result = _make_agent("data", mock_config)
            mock_agent_cls.assert_called_once_with(config=mock_config)
            assert result == mock_agent_cls.return_value

    def test_forwards_kwargs(self):
        mock_config = mock.MagicMock()
        handler = mock.MagicMock()
        with mock.patch("manus_agent.agents.DataAnalysisAgent") as mock_agent_cls:
            mock_agent_cls.return_value = mock.MagicMock()
            _make_agent("data", mock_config, callback_handler=handler)
            mock_agent_cls.assert_called_once_with(config=mock_config, callback_handler=handler)


class TestMakeAgentMCP:
    """_make_agent('mcp', ...) should instantiate MCPAgent."""

    def test_creates_mcp_agent(self):
        mock_config = mock.MagicMock()
        with mock.patch("manus_agent.agents.MCPAgent") as mock_agent_cls:
            mock_agent_cls.return_value = mock.MagicMock()
            result = _make_agent("mcp", mock_config)
            mock_agent_cls.assert_called_once_with(config=mock_config)
            assert result == mock_agent_cls.return_value

    def test_forwards_kwargs(self):
        mock_config = mock.MagicMock()
        handler = mock.MagicMock()
        with mock.patch("manus_agent.agents.MCPAgent") as mock_agent_cls:
            mock_agent_cls.return_value = mock.MagicMock()
            _make_agent("mcp", mock_config, callback_handler=handler)
            mock_agent_cls.assert_called_once_with(config=mock_config, callback_handler=handler)


class TestMakeAgentManus:
    """_make_agent('manus', ...) or any unknown type should instantiate ManusAgent."""

    def test_creates_manus_agent_explicit(self):
        mock_config = mock.MagicMock()
        with mock.patch("manus_agent.agents.ManusAgent") as mock_agent_cls:
            mock_agent_cls.return_value = mock.MagicMock()
            result = _make_agent("manus", mock_config)
            mock_agent_cls.assert_called_once_with(config=mock_config)
            assert result == mock_agent_cls.return_value

    def test_creates_manus_agent_for_unknown_type(self):
        mock_config = mock.MagicMock()
        with mock.patch("manus_agent.agents.ManusAgent") as mock_agent_cls:
            mock_agent_cls.return_value = mock.MagicMock()
            result = _make_agent("nonexistent", mock_config)
            mock_agent_cls.assert_called_once_with(config=mock_config)
            assert result == mock_agent_cls.return_value

    def test_creates_manus_agent_empty_string(self):
        mock_config = mock.MagicMock()
        with mock.patch("manus_agent.agents.ManusAgent") as mock_agent_cls:
            mock_agent_cls.return_value = mock.MagicMock()
            result = _make_agent("", mock_config)
            mock_agent_cls.assert_called_once_with(config=mock_config)
            assert result == mock_agent_cls.return_value

    def test_forwards_kwargs(self):
        mock_config = mock.MagicMock()
        handler = mock.MagicMock()
        with mock.patch("manus_agent.agents.ManusAgent") as mock_agent_cls:
            mock_agent_cls.return_value = mock.MagicMock()
            _make_agent("manus", mock_config, callback_handler=handler)
            mock_agent_cls.assert_called_once_with(config=mock_config, callback_handler=handler)

    def test_forwards_multiple_kwargs(self):
        mock_config = mock.MagicMock()
        with mock.patch("manus_agent.agents.ManusAgent") as mock_agent_cls:
            mock_agent_cls.return_value = mock.MagicMock()
            _make_agent("manus", mock_config, callback_handler="h", extra="e")
            mock_agent_cls.assert_called_once_with(config=mock_config, callback_handler="h", extra="e")


class TestMakeAgentExceptions:
    """_make_agent should propagate exceptions from agent constructors."""

    def test_browser_agent_import_error(self):
        mock_config = mock.MagicMock()
        with mock.patch("manus_agent.agents.BrowserUseAgent", side_effect=ImportError("no browser")):
            with pytest.raises(ImportError, match="no browser"):
                _make_agent("browser", mock_config)

    def test_data_agent_init_error(self):
        mock_config = mock.MagicMock()
        with mock.patch("manus_agent.agents.DataAnalysisAgent", side_effect=RuntimeError("init failed")):
            with pytest.raises(RuntimeError, match="init failed"):
                _make_agent("data", mock_config)

    def test_manus_agent_type_error(self):
        mock_config = mock.MagicMock()
        with mock.patch("manus_agent.agents.ManusAgent", side_effect=TypeError("bad config")):
            with pytest.raises(TypeError, match="bad config"):
                _make_agent("manus", mock_config)


# ===========================================================================
# Integration: is_complex_task used in routing decisions
# ===========================================================================


class TestIsComplexTaskEdgeCases:
    """Edge cases and combined patterns."""

    def test_multiline_input(self):
        task = "First step.\nSecond step.\nThird step."
        # Split on . gives 4 parts (3 sentences + trailing)
        assert is_complex_task(task) is True

    def test_url_with_dots_not_complex(self):
        # URLs have dots but shouldn't make things complex unless word count/patterns match
        task = "open https://example.com"
        assert is_complex_task(task) is False

    def test_combined_patterns_short(self):
        # Has 'then' — triggers immediately
        task = "parse CSV then plot"
        assert is_complex_task(task) is True

    def test_whitespace_only(self):
        assert is_complex_task("   ") is False

    def test_special_characters_no_match(self):
        assert is_complex_task("@#$%^&*()") is False

    def test_pattern_match_case_insensitive(self):
        assert is_complex_task("FIRST do this") is True
        assert is_complex_task("create a WORKFLOW") is True
        assert is_complex_task("BROWSE the web and EXTRACT data") is True
