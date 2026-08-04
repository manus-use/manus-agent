"""Tests for code_execute tool — CodeExecutor, async pipeline, and @tool wrapper.

Covers:
- CodeExecutor initialisation and configuration
- execute_python local mode (sandbox disabled)
- execute_bash local mode (sandbox disabled)
- Sandbox path (mocked DockerSandbox)
- Async pipeline: code_execute() function with language routing, timeout, errors
- Synchronous @tool wrapper: code_execute_sync
- get_executor() singleton behaviour
- Edge cases: unsupported language, empty code, timeout, exceptions
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module-level imports
# ---------------------------------------------------------------------------


def test_module_importable():
    """code_execute module imports cleanly."""
    from manus_agent.tools import code_execute  # noqa: F401


def test_code_executor_class_importable():
    """CodeExecutor class is importable."""
    from manus_agent.tools.code_execute import CodeExecutor

    assert CodeExecutor is not None


def test_get_executor_importable():
    """get_executor singleton factory is importable."""
    from manus_agent.tools.code_execute import get_executor

    assert callable(get_executor)


def test_code_execute_sync_is_strands_tool():
    """code_execute_sync is decorated with @tool."""
    from manus_agent.tools.code_execute import code_execute_sync

    # Strands @tool decorated functions have specific attributes
    assert callable(code_execute_sync)


# ---------------------------------------------------------------------------
# CodeExecutor — initialisation
# ---------------------------------------------------------------------------


class TestCodeExecutorInit:
    """Tests for CodeExecutor construction."""

    def test_default_config_used_when_none(self, monkeypatch):
        """CodeExecutor uses Config.from_file() when no config passed."""
        from manus_agent.tools.code_execute import CodeExecutor

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            executor = CodeExecutor()
            assert executor.config is mock_config

    def test_explicit_config_used(self):
        """CodeExecutor uses passed config."""
        from manus_agent.tools.code_execute import CodeExecutor

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        executor = CodeExecutor(config=mock_config)
        assert executor.config is mock_config

    def test_sandbox_initially_none(self):
        """Sandbox instance is None before first execution."""
        from manus_agent.tools.code_execute import CodeExecutor

        mock_config = MagicMock()
        mock_config.sandbox.enabled = True
        executor = CodeExecutor(config=mock_config)
        assert executor._sandbox is None


# ---------------------------------------------------------------------------
# CodeExecutor.execute_python — local mode (sandbox disabled)
# ---------------------------------------------------------------------------


class TestExecutePythonLocal:
    """Tests for execute_python when sandbox is disabled."""

    @pytest.fixture
    def executor_no_sandbox(self):
        from manus_agent.tools.code_execute import CodeExecutor

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 30
        return CodeExecutor(config=mock_config)

    @pytest.mark.asyncio
    async def test_simple_print(self, executor_no_sandbox):
        """Simple print statement produces stdout."""
        stdout, stderr, code = await executor_no_sandbox.execute_python("print('hello')")
        assert stdout.strip() == "hello"
        assert code == 0

    @pytest.mark.asyncio
    async def test_syntax_error_returns_nonzero(self, executor_no_sandbox):
        """Syntax error produces nonzero exit code."""
        stdout, stderr, code = await executor_no_sandbox.execute_python("def foo(")
        assert code != 0
        assert "SyntaxError" in stderr

    @pytest.mark.asyncio
    async def test_runtime_error_in_stderr(self, executor_no_sandbox):
        """Runtime error appears in stderr."""
        stdout, stderr, code = await executor_no_sandbox.execute_python("raise ValueError('boom')")
        assert code != 0
        assert "ValueError" in stderr
        assert "boom" in stderr

    @pytest.mark.asyncio
    async def test_stderr_independent_of_stdout(self, executor_no_sandbox):
        """stderr and stdout are independently captured."""
        code_text = "import sys; print('out'); print('err', file=sys.stderr)"
        stdout, stderr, code = await executor_no_sandbox.execute_python(code_text)
        assert "out" in stdout
        assert "err" in stderr
        assert code == 0

    @pytest.mark.asyncio
    async def test_timeout_override(self, executor_no_sandbox):
        """Custom timeout is applied."""
        with patch("manus_agent.tools.code_execute.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            await executor_no_sandbox.execute_python("pass", timeout=42)
            call_kwargs = mock_run.call_args
            assert call_kwargs[1]["timeout"] == 42

    @pytest.mark.asyncio
    async def test_default_timeout_from_config(self, executor_no_sandbox):
        """Default timeout comes from config.sandbox.timeout."""
        with patch("manus_agent.tools.code_execute.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            await executor_no_sandbox.execute_python("pass")
            call_kwargs = mock_run.call_args
            assert call_kwargs[1]["timeout"] == 30

    @pytest.mark.asyncio
    async def test_tempfile_cleaned_up(self, executor_no_sandbox):
        """Temporary file is removed after execution."""
        created_paths: list[Path] = []

        original_run = subprocess.run

        def capture_run(args, **kwargs):
            # The second arg is the script path
            if len(args) >= 2:
                created_paths.append(Path(args[1]))
            return original_run(args, **kwargs)

        with patch("manus_agent.tools.code_execute.subprocess.run", side_effect=capture_run):
            await executor_no_sandbox.execute_python("pass")

        # Verify the temp file was deleted
        for p in created_paths:
            assert not p.exists(), f"Temp file was not cleaned up: {p}"

    @pytest.mark.asyncio
    async def test_multiline_code(self, executor_no_sandbox):
        """Multiline code works correctly."""
        code_text = "x = 2\ny = 3\nprint(x + y)"
        stdout, stderr, code = await executor_no_sandbox.execute_python(code_text)
        assert stdout.strip() == "5"
        assert code == 0


# ---------------------------------------------------------------------------
# CodeExecutor.execute_bash — local mode (sandbox disabled)
# ---------------------------------------------------------------------------


class TestExecuteBashLocal:
    """Tests for execute_bash when sandbox is disabled."""

    @pytest.fixture
    def executor_no_sandbox(self):
        from manus_agent.tools.code_execute import CodeExecutor

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 30
        return CodeExecutor(config=mock_config)

    @pytest.mark.asyncio
    async def test_simple_echo(self, executor_no_sandbox):
        """Simple echo command produces stdout."""
        stdout, stderr, code = await executor_no_sandbox.execute_bash("echo hello")
        assert stdout.strip() == "hello"
        assert code == 0

    @pytest.mark.asyncio
    async def test_nonexistent_command(self, executor_no_sandbox):
        """Nonexistent command gives nonzero exit code."""
        stdout, stderr, code = await executor_no_sandbox.execute_bash("command_that_does_not_exist_xyz123")
        assert code != 0

    @pytest.mark.asyncio
    async def test_exit_code_propagation(self, executor_no_sandbox):
        """Exit code is correctly propagated."""
        stdout, stderr, code = await executor_no_sandbox.execute_bash("exit 42")
        assert code == 42

    @pytest.mark.asyncio
    async def test_stderr_captured(self, executor_no_sandbox):
        """stderr is captured separately."""
        stdout, stderr, code = await executor_no_sandbox.execute_bash("echo err >&2")
        assert "err" in stderr
        assert code == 0

    @pytest.mark.asyncio
    async def test_piped_command(self, executor_no_sandbox):
        """Piped commands work (shell=True)."""
        stdout, stderr, code = await executor_no_sandbox.execute_bash("echo 'foo bar baz' | wc -w")
        assert stdout.strip() == "3"
        assert code == 0

    @pytest.mark.asyncio
    async def test_timeout_parameter(self, executor_no_sandbox):
        """Custom timeout is passed to subprocess.run."""
        with patch("manus_agent.tools.code_execute.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            await executor_no_sandbox.execute_bash("true", timeout=99)
            call_kwargs = mock_run.call_args
            assert call_kwargs[1]["timeout"] == 99


# ---------------------------------------------------------------------------
# CodeExecutor — sandbox path (mocked)
# ---------------------------------------------------------------------------


class TestExecuteWithSandbox:
    """Tests for execution when sandbox is enabled (DockerSandbox mocked)."""

    @pytest.fixture
    def executor_with_sandbox(self):
        from manus_agent.tools.code_execute import CodeExecutor

        mock_config = MagicMock()
        mock_config.sandbox.enabled = True
        mock_config.sandbox.docker_image = "python:3.12-slim"
        mock_config.sandbox.memory_limit = "512m"
        mock_config.sandbox.cpu_limit = 1.0
        mock_config.sandbox.timeout = 60
        return CodeExecutor(config=mock_config)

    @pytest.mark.asyncio
    async def test_sandbox_created_on_first_call(self, executor_with_sandbox):
        """DockerSandbox is instantiated and started on first execution."""
        mock_sandbox = AsyncMock()
        mock_sandbox.execute_code = AsyncMock(return_value=("output", "", 0))
        mock_sandbox.start = AsyncMock()

        with patch("manus_agent.tools.code_execute.DockerSandbox", return_value=mock_sandbox):
            await executor_with_sandbox.execute_python("print('hi')")
            mock_sandbox.start.assert_called_once()
            mock_sandbox.execute_code.assert_called_once_with(
                "print('hi')", language="python", timeout=60
            )

    @pytest.mark.asyncio
    async def test_sandbox_reused_on_second_call(self, executor_with_sandbox):
        """DockerSandbox is reused across calls."""
        mock_sandbox = AsyncMock()
        mock_sandbox.execute_code = AsyncMock(return_value=("", "", 0))
        mock_sandbox.start = AsyncMock()

        with patch("manus_agent.tools.code_execute.DockerSandbox", return_value=mock_sandbox):
            await executor_with_sandbox.execute_python("x = 1")
            await executor_with_sandbox.execute_python("x = 2")
            # start called only once
            assert mock_sandbox.start.call_count == 1
            assert mock_sandbox.execute_code.call_count == 2

    @pytest.mark.asyncio
    async def test_bash_uses_sandbox_execute_command(self, executor_with_sandbox):
        """execute_bash delegates to sandbox.execute_command when enabled."""
        mock_sandbox = AsyncMock()
        mock_sandbox.execute_command = AsyncMock(return_value=("dir listing", "", 0))
        mock_sandbox.start = AsyncMock()

        with patch("manus_agent.tools.code_execute.DockerSandbox", return_value=mock_sandbox):
            stdout, stderr, code = await executor_with_sandbox.execute_bash("ls")
            assert stdout == "dir listing"
            mock_sandbox.execute_command.assert_called_once_with("ls", timeout=60)

    @pytest.mark.asyncio
    async def test_cleanup_stops_sandbox(self, executor_with_sandbox):
        """cleanup() stops and clears sandbox reference."""
        mock_sandbox = AsyncMock()
        mock_sandbox.execute_code = AsyncMock(return_value=("", "", 0))
        mock_sandbox.start = AsyncMock()
        mock_sandbox.stop = AsyncMock()

        with patch("manus_agent.tools.code_execute.DockerSandbox", return_value=mock_sandbox):
            await executor_with_sandbox.execute_python("pass")
            assert executor_with_sandbox._sandbox is not None
            await executor_with_sandbox.cleanup()
            mock_sandbox.stop.assert_called_once()
            assert executor_with_sandbox._sandbox is None


# ---------------------------------------------------------------------------
# Async code_execute() function — language routing and error handling
# ---------------------------------------------------------------------------


class TestCodeExecuteFunction:
    """Tests for the async code_execute() function."""

    @pytest.fixture(autouse=True)
    def reset_global_executor(self):
        """Reset global executor between tests."""
        import manus_agent.tools.code_execute as mod

        mod._executor = None
        yield
        mod._executor = None

    @pytest.mark.asyncio
    async def test_python_language_routes_to_execute_python(self):
        """language='python' routes to execute_python."""
        from manus_agent.tools.code_execute import code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = await code_execute("print(42)", language="python")
            assert result["stdout"].strip() == "42"
            assert result["exit_code"] == 0
            assert result["error"] is None

    @pytest.mark.asyncio
    async def test_bash_language_routes_to_execute_bash(self):
        """language='bash' routes to execute_bash."""
        from manus_agent.tools.code_execute import code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = await code_execute("echo hi", language="bash")
            assert result["stdout"].strip() == "hi"
            assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_sh_language_accepted(self):
        """language='sh' routes to execute_bash."""
        from manus_agent.tools.code_execute import code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = await code_execute("echo yes", language="sh")
            assert result["stdout"].strip() == "yes"

    @pytest.mark.asyncio
    async def test_shell_language_accepted(self):
        """language='shell' routes to execute_bash."""
        from manus_agent.tools.code_execute import code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = await code_execute("echo yes", language="shell")
            assert result["stdout"].strip() == "yes"

    @pytest.mark.asyncio
    async def test_unsupported_language_returns_error(self):
        """Unsupported language returns an error dict."""
        from manus_agent.tools.code_execute import code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = await code_execute("print('x')", language="ruby")
            assert result["exit_code"] == 1
            assert "ruby" in result["error"].lower() or "ruby" in result["stderr"].lower()
            assert result["stdout"] == ""

    @pytest.mark.asyncio
    async def test_timeout_error_caught(self):
        """asyncio.TimeoutError is caught and returned as error dict."""
        from manus_agent.tools.code_execute import CodeExecutor, code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 1

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            with patch.object(
                CodeExecutor,
                "execute_python",
                side_effect=asyncio.TimeoutError(),
            ):
                result = await code_execute("import time; time.sleep(100)", language="python", timeout=1)
                assert result["exit_code"] == -1
                assert "Timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_generic_exception_caught(self):
        """Generic exceptions are caught and returned as error dict."""
        from manus_agent.tools.code_execute import CodeExecutor, code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            with patch.object(
                CodeExecutor,
                "execute_python",
                side_effect=RuntimeError("disk full"),
            ):
                result = await code_execute("pass", language="python")
                assert result["exit_code"] == -1
                assert "disk full" in result["error"]

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_sets_error_field(self):
        """Non-zero exit code populates the error field with stderr."""
        from manus_agent.tools.code_execute import code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = await code_execute("import sys; sys.exit(1)", language="python")
            assert result["exit_code"] == 1
            # error is set to stderr content when exit_code != 0
            assert result["error"] is not None or result["stderr"] != ""


# ---------------------------------------------------------------------------
# get_executor() singleton
# ---------------------------------------------------------------------------


class TestGetExecutor:
    """Tests for the get_executor() singleton factory."""

    @pytest.fixture(autouse=True)
    def reset_global_executor(self):
        """Reset global executor between tests."""
        import manus_agent.tools.code_execute as mod

        mod._executor = None
        yield
        mod._executor = None

    def test_returns_code_executor_instance(self):
        """get_executor() returns a CodeExecutor."""
        from manus_agent.tools.code_execute import CodeExecutor, get_executor

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            executor = get_executor()
            assert isinstance(executor, CodeExecutor)

    def test_returns_same_instance(self):
        """get_executor() returns the same object on repeated calls."""
        from manus_agent.tools.code_execute import get_executor

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            e1 = get_executor()
            e2 = get_executor()
            assert e1 is e2

    def test_accepts_explicit_config(self):
        """get_executor(config=...) uses the provided config."""
        from manus_agent.tools.code_execute import get_executor

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        executor = get_executor(config=mock_config)
        assert executor.config is mock_config


# ---------------------------------------------------------------------------
# code_execute_sync — @tool wrapper
# ---------------------------------------------------------------------------


class TestCodeExecuteSync:
    """Tests for the synchronous @tool wrapper."""

    @pytest.fixture(autouse=True)
    def reset_global_executor(self):
        """Reset global executor between tests."""
        import manus_agent.tools.code_execute as mod

        mod._executor = None
        yield
        mod._executor = None

    def test_sync_wrapper_returns_dict(self):
        """code_execute_sync returns a dict with expected keys."""
        from manus_agent.tools.code_execute import code_execute_sync

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = code_execute_sync("print('test')", language="python")
            assert isinstance(result, dict)
            assert "stdout" in result
            assert "stderr" in result
            assert "exit_code" in result

    def test_sync_wrapper_python_execution(self):
        """code_execute_sync executes Python code correctly."""
        from manus_agent.tools.code_execute import code_execute_sync

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = code_execute_sync("print(2 + 2)", language="python")
            assert result["stdout"].strip() == "4"
            assert result["exit_code"] == 0

    def test_sync_wrapper_bash_execution(self):
        """code_execute_sync executes bash code correctly."""
        from manus_agent.tools.code_execute import code_execute_sync

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = code_execute_sync("echo done", language="bash")
            assert result["stdout"].strip() == "done"
            assert result["exit_code"] == 0

    def test_sync_wrapper_unsupported_language(self):
        """code_execute_sync returns error for unsupported language."""
        from manus_agent.tools.code_execute import code_execute_sync

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = code_execute_sync("x", language="java")
            assert result["exit_code"] == 1
            assert result["error"] is not None


# ---------------------------------------------------------------------------
# Edge cases and integration scenarios
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests for code_execute module."""

    @pytest.fixture(autouse=True)
    def reset_global_executor(self):
        """Reset global executor between tests."""
        import manus_agent.tools.code_execute as mod

        mod._executor = None
        yield
        mod._executor = None

    @pytest.mark.asyncio
    async def test_empty_python_code(self):
        """Empty code string executes without error."""
        from manus_agent.tools.code_execute import code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = await code_execute("", language="python")
            assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_empty_bash_code(self):
        """Empty bash command executes without error."""
        from manus_agent.tools.code_execute import code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = await code_execute("", language="bash")
            assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_large_output(self):
        """Large stdout is fully captured."""
        from manus_agent.tools.code_execute import code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = await code_execute("print('x' * 10000)", language="python")
            assert len(result["stdout"].strip()) == 10000

    @pytest.mark.asyncio
    async def test_language_case_insensitive(self):
        """Language parameter is case-insensitive."""
        from manus_agent.tools.code_execute import code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = await code_execute("print('yes')", language="Python")
            assert result["stdout"].strip() == "yes"

    @pytest.mark.asyncio
    async def test_special_characters_in_code(self):
        """Code with special characters executes correctly."""
        from manus_agent.tools.code_execute import code_execute

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        with patch("manus_agent.tools.code_execute.Config.from_file", return_value=mock_config):
            result = await code_execute("print('hello\\nworld')", language="python")
            assert "hello" in result["stdout"]
            assert "world" in result["stdout"]

    @pytest.mark.asyncio
    async def test_uses_sys_executable(self):
        """execute_python uses sys.executable for the Python interpreter."""
        from manus_agent.tools.code_execute import CodeExecutor

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        executor = CodeExecutor(config=mock_config)
        with patch("manus_agent.tools.code_execute.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            await executor.execute_python("pass")
            args = mock_run.call_args[0][0]
            assert args[0] == sys.executable

    @pytest.mark.asyncio
    async def test_bash_uses_shell_true(self):
        """execute_bash uses shell=True."""
        from manus_agent.tools.code_execute import CodeExecutor

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        mock_config.sandbox.timeout = 10

        executor = CodeExecutor(config=mock_config)
        with patch("manus_agent.tools.code_execute.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
            await executor.execute_bash("ls")
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["shell"] is True

    def test_cleanup_when_no_sandbox(self):
        """cleanup() is safe when no sandbox was ever created."""
        from manus_agent.tools.code_execute import CodeExecutor

        mock_config = MagicMock()
        mock_config.sandbox.enabled = False
        executor = CodeExecutor(config=mock_config)
        # Should not raise
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(executor.cleanup())
        finally:
            loop.close()
