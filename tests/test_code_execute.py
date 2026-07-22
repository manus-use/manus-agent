"""Comprehensive test suite for code_execute tool module.

Tests cover:
- CodeExecutor class: init, sandbox/local paths, language dispatch, cleanup
- execute_python: sandbox path, local path, timeout handling
- execute_bash: sandbox path, local path, timeout handling
- get_executor: global singleton, config passthrough
- code_execute: async function, language routing, error handling, timeout
- code_execute_sync: synchronous wrapper via @tool decorator
- Edge cases: unsupported language, subprocess failures, exception handling
"""

from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from manus_agent.tools.code_execute import (
    CodeExecutor,
    code_execute,
    code_execute_sync,
    get_executor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config_sandbox_enabled():
    """Config with sandbox enabled."""
    config = MagicMock()
    config.sandbox.enabled = True
    config.sandbox.timeout = 30
    config.sandbox.docker_image = "python:3.12-slim"
    config.sandbox.memory_limit = "512m"
    config.sandbox.cpu_limit = 1.0
    return config


@pytest.fixture
def mock_config_sandbox_disabled():
    """Config with sandbox disabled."""
    config = MagicMock()
    config.sandbox.enabled = False
    config.sandbox.timeout = 30
    return config


@pytest.fixture(autouse=True)
def reset_global_executor():
    """Reset global executor between tests."""
    import manus_agent.tools.code_execute as mod

    mod._executor = None
    yield
    mod._executor = None


# ---------------------------------------------------------------------------
# CodeExecutor.__init__
# ---------------------------------------------------------------------------


class TestCodeExecutorInit:
    """Tests for CodeExecutor initialization."""

    @patch("manus_agent.tools.code_execute.Config.from_file")
    def test_default_config_from_file(self, mock_from_file):
        """Uses Config.from_file() when no config provided."""
        mock_cfg = MagicMock()
        mock_from_file.return_value = mock_cfg
        executor = CodeExecutor()
        assert executor.config is mock_cfg
        mock_from_file.assert_called_once()

    def test_explicit_config(self, mock_config_sandbox_enabled):
        """Uses provided config when passed."""
        executor = CodeExecutor(config=mock_config_sandbox_enabled)
        assert executor.config is mock_config_sandbox_enabled

    def test_sandbox_initially_none(self, mock_config_sandbox_enabled):
        """Sandbox is None until first use."""
        executor = CodeExecutor(config=mock_config_sandbox_enabled)
        assert executor._sandbox is None


# ---------------------------------------------------------------------------
# CodeExecutor._get_sandbox
# ---------------------------------------------------------------------------


class TestGetSandbox:
    """Tests for _get_sandbox lazy initialization."""

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self, mock_config_sandbox_disabled):
        """Returns None when sandbox is disabled in config."""
        executor = CodeExecutor(config=mock_config_sandbox_disabled)
        result = await executor._get_sandbox()
        assert result is None

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.DockerSandbox")
    async def test_creates_sandbox_when_enabled(self, mock_docker_cls, mock_config_sandbox_enabled):
        """Creates and starts DockerSandbox when enabled."""
        mock_sandbox = AsyncMock()
        mock_docker_cls.return_value = mock_sandbox
        executor = CodeExecutor(config=mock_config_sandbox_enabled)

        result = await executor._get_sandbox()

        mock_docker_cls.assert_called_once_with(
            image="python:3.12-slim",
            memory_limit="512m",
            cpu_limit=1.0,
        )
        mock_sandbox.start.assert_awaited_once()
        assert result is mock_sandbox

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.DockerSandbox")
    async def test_reuses_existing_sandbox(self, mock_docker_cls, mock_config_sandbox_enabled):
        """Reuses the same sandbox instance on subsequent calls."""
        mock_sandbox = AsyncMock()
        mock_docker_cls.return_value = mock_sandbox
        executor = CodeExecutor(config=mock_config_sandbox_enabled)

        result1 = await executor._get_sandbox()
        result2 = await executor._get_sandbox()

        assert result1 is result2
        mock_docker_cls.assert_called_once()
        mock_sandbox.start.assert_awaited_once()


# ---------------------------------------------------------------------------
# CodeExecutor.execute_python
# ---------------------------------------------------------------------------


class TestExecutePython:
    """Tests for execute_python method."""

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.DockerSandbox")
    async def test_sandbox_path(self, mock_docker_cls, mock_config_sandbox_enabled):
        """Executes Python code via sandbox when enabled."""
        mock_sandbox = AsyncMock()
        mock_sandbox.execute_code = AsyncMock(return_value=("output", "", 0))
        mock_docker_cls.return_value = mock_sandbox

        executor = CodeExecutor(config=mock_config_sandbox_enabled)
        stdout, stderr, exit_code = await executor.execute_python("print('hello')")

        mock_sandbox.execute_code.assert_awaited_once_with("print('hello')", language="python", timeout=30)
        assert stdout == "output"
        assert stderr == ""
        assert exit_code == 0

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.DockerSandbox")
    async def test_sandbox_custom_timeout(self, mock_docker_cls, mock_config_sandbox_enabled):
        """Passes custom timeout to sandbox."""
        mock_sandbox = AsyncMock()
        mock_sandbox.execute_code = AsyncMock(return_value=("", "", 0))
        mock_docker_cls.return_value = mock_sandbox

        executor = CodeExecutor(config=mock_config_sandbox_enabled)
        await executor.execute_python("pass", timeout=60)

        mock_sandbox.execute_code.assert_awaited_once_with("pass", language="python", timeout=60)

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_local_path(self, mock_run, mock_config_sandbox_disabled):
        """Executes Python code locally when sandbox disabled."""
        mock_run.return_value = MagicMock(stdout="hello\n", stderr="", returncode=0)

        executor = CodeExecutor(config=mock_config_sandbox_disabled)
        stdout, stderr, exit_code = await executor.execute_python("print('hello')")

        assert stdout == "hello\n"
        assert stderr == ""
        assert exit_code == 0
        mock_run.assert_called_once()
        # Verify python executable and timeout
        call_args = mock_run.call_args
        assert call_args.kwargs["timeout"] == 30
        assert call_args.kwargs["capture_output"] is True
        assert call_args.kwargs["text"] is True

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_local_path_custom_timeout(self, mock_run, mock_config_sandbox_disabled):
        """Passes custom timeout to subprocess."""
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        executor = CodeExecutor(config=mock_config_sandbox_disabled)
        await executor.execute_python("pass", timeout=120)

        call_args = mock_run.call_args
        assert call_args.kwargs["timeout"] == 120

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_local_path_cleans_tempfile(self, mock_run, mock_config_sandbox_disabled):
        """Temp file is cleaned up after local execution."""
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        executor = CodeExecutor(config=mock_config_sandbox_disabled)
        await executor.execute_python("x = 1")

        # The temp file should be deleted in finally block
        call_args = mock_run.call_args
        from pathlib import Path

        script_path = call_args[0][0][1]  # [sys.executable, path]
        assert not Path(script_path).exists()

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_local_path_nonzero_exit(self, mock_run, mock_config_sandbox_disabled):
        """Returns stderr and nonzero exit code on failure."""
        mock_run.return_value = MagicMock(stdout="", stderr="NameError: name 'x' is not defined", returncode=1)

        executor = CodeExecutor(config=mock_config_sandbox_disabled)
        stdout, stderr, exit_code = await executor.execute_python("print(x)")

        assert exit_code == 1
        assert "NameError" in stderr


# ---------------------------------------------------------------------------
# CodeExecutor.execute_bash
# ---------------------------------------------------------------------------


class TestExecuteBash:
    """Tests for execute_bash method."""

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.DockerSandbox")
    async def test_sandbox_path(self, mock_docker_cls, mock_config_sandbox_enabled):
        """Executes bash command via sandbox when enabled."""
        mock_sandbox = AsyncMock()
        mock_sandbox.execute_command = AsyncMock(return_value=("files", "", 0))
        mock_docker_cls.return_value = mock_sandbox

        executor = CodeExecutor(config=mock_config_sandbox_enabled)
        stdout, stderr, exit_code = await executor.execute_bash("ls -la")

        mock_sandbox.execute_command.assert_awaited_once_with("ls -la", timeout=30)
        assert stdout == "files"
        assert exit_code == 0

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_local_path(self, mock_run, mock_config_sandbox_disabled):
        """Executes bash command locally when sandbox disabled."""
        mock_run.return_value = MagicMock(stdout="/home/user\n", stderr="", returncode=0)

        executor = CodeExecutor(config=mock_config_sandbox_disabled)
        stdout, stderr, exit_code = await executor.execute_bash("pwd")

        assert stdout == "/home/user\n"
        assert exit_code == 0
        mock_run.assert_called_once_with(
            "pwd",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_local_bash_failure(self, mock_run, mock_config_sandbox_disabled):
        """Returns error output on bash command failure."""
        mock_run.return_value = MagicMock(stdout="", stderr="command not found: foo", returncode=127)

        executor = CodeExecutor(config=mock_config_sandbox_disabled)
        stdout, stderr, exit_code = await executor.execute_bash("foo")

        assert exit_code == 127
        assert "command not found" in stderr

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.DockerSandbox")
    async def test_sandbox_custom_timeout(self, mock_docker_cls, mock_config_sandbox_enabled):
        """Passes custom timeout to sandbox for bash."""
        mock_sandbox = AsyncMock()
        mock_sandbox.execute_command = AsyncMock(return_value=("", "", 0))
        mock_docker_cls.return_value = mock_sandbox

        executor = CodeExecutor(config=mock_config_sandbox_enabled)
        await executor.execute_bash("sleep 1", timeout=5)

        mock_sandbox.execute_command.assert_awaited_once_with("sleep 1", timeout=5)


# ---------------------------------------------------------------------------
# CodeExecutor.cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    """Tests for cleanup method."""

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.DockerSandbox")
    async def test_stops_sandbox(self, mock_docker_cls, mock_config_sandbox_enabled):
        """Cleanup stops the running sandbox."""
        mock_sandbox = AsyncMock()
        mock_docker_cls.return_value = mock_sandbox

        executor = CodeExecutor(config=mock_config_sandbox_enabled)
        await executor._get_sandbox()  # Start sandbox
        await executor.cleanup()

        mock_sandbox.stop.assert_awaited_once()
        assert executor._sandbox is None

    @pytest.mark.asyncio
    async def test_cleanup_without_sandbox(self, mock_config_sandbox_disabled):
        """Cleanup is a no-op when no sandbox was created."""
        executor = CodeExecutor(config=mock_config_sandbox_disabled)
        await executor.cleanup()  # Should not raise
        assert executor._sandbox is None


# ---------------------------------------------------------------------------
# get_executor
# ---------------------------------------------------------------------------


class TestGetExecutor:
    """Tests for get_executor global singleton."""

    @patch("manus_agent.tools.code_execute.Config.from_file")
    def test_creates_singleton(self, mock_from_file):
        """Creates a single global instance."""
        mock_from_file.return_value = MagicMock()
        executor1 = get_executor()
        executor2 = get_executor()
        assert executor1 is executor2

    @patch("manus_agent.tools.code_execute.Config.from_file")
    def test_uses_default_config(self, mock_from_file):
        """Uses Config.from_file when no config argument."""
        mock_cfg = MagicMock()
        mock_from_file.return_value = mock_cfg
        executor = get_executor()
        assert executor.config is mock_cfg

    def test_uses_provided_config(self, mock_config_sandbox_disabled):
        """Passes config to CodeExecutor when provided."""
        executor = get_executor(config=mock_config_sandbox_disabled)
        assert executor.config is mock_config_sandbox_disabled


# ---------------------------------------------------------------------------
# code_execute (async function)
# ---------------------------------------------------------------------------


class TestCodeExecuteAsync:
    """Tests for the async code_execute function."""

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_python_language(self, mock_get_exec):
        """Routes python code to execute_python."""
        mock_executor = AsyncMock()
        mock_executor.execute_python = AsyncMock(return_value=("result", "", 0))
        mock_get_exec.return_value = mock_executor

        result = await code_execute("print(1)", language="python", timeout=30)

        mock_executor.execute_python.assert_awaited_once_with("print(1)", 30)
        assert result["stdout"] == "result"
        assert result["exit_code"] == 0
        assert result["error"] is None

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_bash_language(self, mock_get_exec):
        """Routes bash code to execute_bash."""
        mock_executor = AsyncMock()
        mock_executor.execute_bash = AsyncMock(return_value=("output", "", 0))
        mock_get_exec.return_value = mock_executor

        result = await code_execute("echo hi", language="bash", timeout=30)

        mock_executor.execute_bash.assert_awaited_once_with("echo hi", 30)
        assert result["stdout"] == "output"
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_sh_language(self, mock_get_exec):
        """Routes 'sh' to execute_bash."""
        mock_executor = AsyncMock()
        mock_executor.execute_bash = AsyncMock(return_value=("", "", 0))
        mock_get_exec.return_value = mock_executor

        await code_execute("ls", language="sh")
        mock_executor.execute_bash.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_shell_language(self, mock_get_exec):
        """Routes 'shell' to execute_bash."""
        mock_executor = AsyncMock()
        mock_executor.execute_bash = AsyncMock(return_value=("", "", 0))
        mock_get_exec.return_value = mock_executor

        await code_execute("date", language="shell")
        mock_executor.execute_bash.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_unsupported_language(self, mock_get_exec):
        """Returns error for unsupported language."""
        mock_get_exec.return_value = AsyncMock()

        result = await code_execute("code", language="ruby")

        assert result["exit_code"] == 1
        assert "Unsupported language" in result["stderr"]
        assert "ruby" in result["error"]

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_nonzero_exit_sets_error(self, mock_get_exec):
        """Error field is populated when exit code is non-zero."""
        mock_executor = AsyncMock()
        mock_executor.execute_python = AsyncMock(return_value=("", "SyntaxError: invalid syntax", 1))
        mock_get_exec.return_value = mock_executor

        result = await code_execute("def", language="python")

        assert result["exit_code"] == 1
        assert result["error"] == "SyntaxError: invalid syntax"
        assert result["stderr"] == "SyntaxError: invalid syntax"

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_zero_exit_no_error(self, mock_get_exec):
        """Error is None when exit code is 0."""
        mock_executor = AsyncMock()
        mock_executor.execute_python = AsyncMock(return_value=("ok", "warning", 0))
        mock_get_exec.return_value = mock_executor

        result = await code_execute("pass")

        assert result["exit_code"] == 0
        assert result["error"] is None
        assert result["stderr"] == "warning"

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_timeout_error(self, mock_get_exec):
        """Handles asyncio.TimeoutError gracefully."""
        mock_executor = AsyncMock()
        mock_executor.execute_python = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_get_exec.return_value = mock_executor

        result = await code_execute("import time; time.sleep(999)", timeout=5)

        assert result["exit_code"] == -1
        assert result["error"] == "Timeout"
        assert "timed out" in result["stderr"]

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_generic_exception(self, mock_get_exec):
        """Handles unexpected exceptions gracefully."""
        mock_executor = AsyncMock()
        mock_executor.execute_python = AsyncMock(side_effect=RuntimeError("Docker daemon not running"))
        mock_get_exec.return_value = mock_executor

        result = await code_execute("pass")

        assert result["exit_code"] == -1
        assert "Docker daemon not running" in result["error"]
        assert "Docker daemon not running" in result["stderr"]

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_custom_timeout_passed(self, mock_get_exec):
        """Custom timeout is passed to executor."""
        mock_executor = AsyncMock()
        mock_executor.execute_bash = AsyncMock(return_value=("", "", 0))
        mock_get_exec.return_value = mock_executor

        await code_execute("echo x", language="bash", timeout=120)

        mock_executor.execute_bash.assert_awaited_once_with("echo x", 120)

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_none_timeout_passed(self, mock_get_exec):
        """None timeout is passed through (executor uses config default)."""
        mock_executor = AsyncMock()
        mock_executor.execute_python = AsyncMock(return_value=("", "", 0))
        mock_get_exec.return_value = mock_executor

        # When timeout=None, code_execute passes None to executor
        # which then falls back to config.sandbox.timeout
        await code_execute("pass", timeout=None)

        mock_executor.execute_python.assert_awaited_once_with("pass", None)

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_language_case_insensitive(self, mock_get_exec):
        """Language matching is case-insensitive."""
        mock_executor = AsyncMock()
        mock_executor.execute_python = AsyncMock(return_value=("", "", 0))
        mock_executor.execute_bash = AsyncMock(return_value=("", "", 0))
        mock_get_exec.return_value = mock_executor

        await code_execute("pass", language="Python")
        mock_executor.execute_python.assert_awaited_once()

        mock_executor.execute_python.reset_mock()
        await code_execute("echo", language="BASH")
        mock_executor.execute_bash.assert_awaited_once()


# ---------------------------------------------------------------------------
# code_execute_sync (@tool decorator)
# ---------------------------------------------------------------------------


class TestCodeExecuteSync:
    """Tests for the synchronous code_execute_sync wrapper."""

    def test_runs_async_in_event_loop(self):
        """Wraps async code_execute in a new event loop."""
        with patch(
            "manus_agent.tools.code_execute.code_execute",
            new=AsyncMock(return_value={"stdout": "sync output", "stderr": "", "exit_code": 0, "error": None}),
        ):
            result = code_execute_sync(code="pass", language="python")

        assert result["stdout"] == "sync output"
        assert result["exit_code"] == 0

    def test_passes_all_arguments(self):
        """Passes code, language, and timeout to async version."""

        async def fake_execute(code, language="python", timeout=None):
            return {"stdout": code, "stderr": language, "exit_code": 0, "error": None}

        with patch("manus_agent.tools.code_execute.code_execute", new=fake_execute):
            result = code_execute_sync(code="hello", language="bash", timeout=60)

        assert result["stdout"] == "hello"
        assert result["stderr"] == "bash"


# ---------------------------------------------------------------------------
# Integration-style tests (mocked at subprocess level)
# ---------------------------------------------------------------------------


class TestIntegrationLocalExecution:
    """Integration tests using local execution path (subprocess mocked)."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    @patch("manus_agent.tools.code_execute.Config.from_file")
    async def test_full_python_execution_flow(self, mock_from_file, mock_run):
        """Full flow: code_execute → get_executor → execute_python → subprocess."""
        mock_cfg = MagicMock()
        mock_cfg.sandbox.enabled = False
        mock_cfg.sandbox.timeout = 30
        mock_from_file.return_value = mock_cfg

        mock_run.return_value = MagicMock(stdout="42\n", stderr="", returncode=0)

        result = await code_execute("print(6*7)", language="python", timeout=10)

        assert result["stdout"] == "42\n"
        assert result["stderr"] == ""
        assert result["exit_code"] == 0
        assert result["error"] is None

    @pytest.mark.asyncio
    @patch("subprocess.run")
    @patch("manus_agent.tools.code_execute.Config.from_file")
    async def test_full_bash_execution_flow(self, mock_from_file, mock_run):
        """Full flow: code_execute → get_executor → execute_bash → subprocess."""
        mock_cfg = MagicMock()
        mock_cfg.sandbox.enabled = False
        mock_cfg.sandbox.timeout = 30
        mock_from_file.return_value = mock_cfg

        mock_run.return_value = MagicMock(stdout="hello world\n", stderr="", returncode=0)

        result = await code_execute("echo hello world", language="bash", timeout=10)

        assert result["stdout"] == "hello world\n"
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    @patch("subprocess.run")
    @patch("manus_agent.tools.code_execute.Config.from_file")
    async def test_subprocess_timeout_raises(self, mock_from_file, mock_run):
        """subprocess.TimeoutExpired surfaces as a general exception."""
        mock_cfg = MagicMock()
        mock_cfg.sandbox.enabled = False
        mock_cfg.sandbox.timeout = 5
        mock_from_file.return_value = mock_cfg

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="python", timeout=5)

        result = await code_execute("import time; time.sleep(999)", timeout=5)

        assert result["exit_code"] == -1
        assert "Execution failed" in result["error"]

    @pytest.mark.asyncio
    @patch("subprocess.run")
    @patch("manus_agent.tools.code_execute.Config.from_file")
    async def test_python_syntax_error(self, mock_from_file, mock_run):
        """Python syntax errors return non-zero exit code."""
        mock_cfg = MagicMock()
        mock_cfg.sandbox.enabled = False
        mock_cfg.sandbox.timeout = 30
        mock_from_file.return_value = mock_cfg

        mock_run.return_value = MagicMock(
            stdout="",
            stderr='  File "code.py", line 1\n    def\n       ^\nSyntaxError: invalid syntax\n',
            returncode=1,
        )

        result = await code_execute("def", language="python")

        assert result["exit_code"] == 1
        assert "SyntaxError" in result["error"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_empty_code(self, mock_get_exec):
        """Empty code string is passed through without error."""
        mock_executor = AsyncMock()
        mock_executor.execute_python = AsyncMock(return_value=("", "", 0))
        mock_get_exec.return_value = mock_executor

        result = await code_execute("", language="python")
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_multiline_code(self, mock_get_exec):
        """Multi-line code is handled correctly."""
        mock_executor = AsyncMock()
        mock_executor.execute_python = AsyncMock(return_value=("3\n", "", 0))
        mock_get_exec.return_value = mock_executor

        code = "x = 1\ny = 2\nprint(x + y)"
        await code_execute(code, language="python")
        mock_executor.execute_python.assert_awaited_once_with(code, None)

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_large_output(self, mock_get_exec):
        """Large stdout is returned in full."""
        big_output = "line\n" * 10000
        mock_executor = AsyncMock()
        mock_executor.execute_bash = AsyncMock(return_value=(big_output, "", 0))
        mock_get_exec.return_value = mock_executor

        result = await code_execute("seq 10000", language="bash")
        assert result["stdout"] == big_output

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_both_stdout_and_stderr(self, mock_get_exec):
        """Both stdout and stderr are captured."""
        mock_executor = AsyncMock()
        mock_executor.execute_python = AsyncMock(return_value=("output\n", "warning: deprecation\n", 0))
        mock_get_exec.return_value = mock_executor

        result = await code_execute("import warnings; warnings.warn('deprecation')")
        assert result["stdout"] == "output\n"
        assert result["stderr"] == "warning: deprecation\n"
        assert result["error"] is None  # exit_code is 0

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_permission_error(self, mock_get_exec):
        """Handles PermissionError from subprocess."""
        mock_executor = AsyncMock()
        mock_executor.execute_bash = AsyncMock(side_effect=PermissionError("Permission denied: /root/secret"))
        mock_get_exec.return_value = mock_executor

        result = await code_execute("cat /root/secret", language="bash")
        assert result["exit_code"] == -1
        assert "Permission denied" in result["error"]

    @pytest.mark.asyncio
    @patch("manus_agent.tools.code_execute.get_executor")
    async def test_os_error(self, mock_get_exec):
        """Handles OSError gracefully."""
        mock_executor = AsyncMock()
        mock_executor.execute_python = AsyncMock(side_effect=OSError("No space left on device"))
        mock_get_exec.return_value = mock_executor

        result = await code_execute("open('/tmp/big', 'w').write('x' * 10**9)")
        assert result["exit_code"] == -1
        assert "No space left" in result["error"]
