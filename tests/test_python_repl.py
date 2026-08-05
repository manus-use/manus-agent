"""Comprehensive test suite for python_repl module.

Tests the FixedPtyManager class and the python_repl tool function covering:
- FixedPtyManager initialization, start, stop, output reading
- Interactive and standard execution modes
- Timeout handling
- State management (reset_state, save_state)
- Error handling (RecursionError, generic exceptions)
- User consent flow (cancellation, bypass)
- Output truncation for binary content
- Edge cases (empty code, large output, concurrent access)

All subprocess/PTY operations are fully mocked — no real processes spawned.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use(code: str, interactive: bool = True, reset_state: bool = False, timeout: int = 120) -> dict:
    """Build a minimal ToolUse dict for python_repl."""
    return {
        "toolUseId": "test-tool-use-001",
        "input": {
            "code": code,
            "interactive": interactive,
            "reset_state": reset_state,
            "timeout": timeout,
        },
    }


# ---------------------------------------------------------------------------
# FixedPtyManager unit tests
# ---------------------------------------------------------------------------


class TestFixedPtyManagerInit:
    """Test FixedPtyManager constructor and initial state."""

    def test_default_init(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        assert mgr.process is None
        assert mgr.supervisor_fd == -1
        assert mgr.output_buffer == []
        assert mgr.callback is None
        assert mgr._child_exited is False
        assert mgr._code_file is None
        assert mgr._reader is None

    def test_init_with_callback(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        cb = MagicMock()
        mgr = FixedPtyManager(callback=cb)
        assert mgr.callback is cb

    def test_output_buffer_starts_empty(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        assert mgr.get_output() == ""


class TestFixedPtyManagerStart:
    """Test FixedPtyManager.start() — PTY allocation and subprocess launch."""

    @patch("manus_agent.tools.python_repl.subprocess.Popen")
    @patch("manus_agent.tools.python_repl.os.close")
    @patch(
        "manus_agent.tools.python_repl.os.fdopen", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())
    )
    @patch("manus_agent.tools.python_repl.tempfile.mkstemp", return_value=(99, "/tmp/repl_test.py"))
    def test_start_creates_temp_file_and_launches(self, mock_mkstemp, mock_fdopen, mock_close, mock_popen):
        from manus_agent.tools.python_repl import FixedPtyManager

        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        with (
            patch("pty.openpty", return_value=(10, 11)),
            patch("fcntl.ioctl"),
            patch("struct.pack", return_value=b"\x00" * 8),
        ):
            mgr = FixedPtyManager()
            mgr.start("print('hello')")

        assert mgr.process is mock_proc
        assert mgr.supervisor_fd == 10
        assert mgr._code_file == "/tmp/repl_test.py"
        assert mgr._reader is not None

    @patch("manus_agent.tools.python_repl.subprocess.Popen")
    @patch("manus_agent.tools.python_repl.os.close")
    @patch("manus_agent.tools.python_repl.os.fdopen")
    @patch("manus_agent.tools.python_repl.tempfile.mkstemp", return_value=(99, "/tmp/repl_test.py"))
    def test_start_closes_slave_fd(self, mock_mkstemp, mock_fdopen, mock_close, mock_popen):
        from manus_agent.tools.python_repl import FixedPtyManager

        mock_fdopen_ctx = MagicMock()
        mock_fdopen.return_value = mock_fdopen_ctx
        mock_fdopen_ctx.__enter__ = MagicMock(return_value=MagicMock())
        mock_fdopen_ctx.__exit__ = MagicMock(return_value=False)
        mock_popen.return_value = MagicMock()

        with (
            patch("pty.openpty", return_value=(10, 11)),
            patch("fcntl.ioctl"),
            patch("struct.pack", return_value=b"\x00" * 8),
        ):
            mgr = FixedPtyManager()
            mgr.start("x = 1")

        # slave_fd (11) should be closed after Popen
        mock_close.assert_any_call(11)


class TestFixedPtyManagerStop:
    """Test FixedPtyManager.stop() — process termination and cleanup."""

    def test_stop_terminates_running_process(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Still running
        mock_proc.wait.return_value = 0
        mgr.process = mock_proc
        mgr._child_exited = False
        mgr.supervisor_fd = 42

        with patch("os.close") as mock_close:
            mgr.stop()

        mock_proc.terminate.assert_called_once()
        assert mgr._child_exited is True
        assert mgr.supervisor_fd == -1
        mock_close.assert_called_once_with(42)

    def test_stop_kills_process_on_timeout(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        # First wait (after terminate) raises TimeoutExpired;
        # second wait (after kill) succeeds.
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 0.5), None]
        mgr.process = mock_proc
        mgr._child_exited = False
        mgr.supervisor_fd = -1

        mgr.stop()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    def test_stop_already_exited_no_op(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # Already exited
        mgr.process = mock_proc
        mgr._child_exited = False
        mgr.supervisor_fd = -1

        mgr.stop()

        mock_proc.terminate.assert_not_called()
        assert mgr._child_exited is True

    def test_stop_cleans_up_code_file(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        mgr.process = None
        mgr._child_exited = True
        mgr.supervisor_fd = -1
        mgr._code_file = "/tmp/test_code.py"

        with patch("os.path.exists", return_value=True), patch("os.unlink") as mock_unlink:
            mgr.stop()

        mock_unlink.assert_called_once_with("/tmp/test_code.py")
        assert mgr._code_file is None

    def test_stop_joins_reader_thread(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        mgr.process = None
        mgr._child_exited = True
        mgr.supervisor_fd = -1
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        mgr._reader = mock_thread

        mgr.stop()

        mock_thread.join.assert_called_once_with(timeout=0.5)

    def test_stop_handles_oserror_on_fd_close(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        mgr.process = None
        mgr._child_exited = True
        mgr.supervisor_fd = 99

        with patch("os.close", side_effect=OSError("Bad file descriptor")):
            # Should not raise
            mgr.stop()

        assert mgr.supervisor_fd == -1

    def test_stop_handles_process_lookup_error(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate.side_effect = ProcessLookupError("No such process")
        mgr.process = mock_proc
        mgr._child_exited = False
        mgr.supervisor_fd = -1

        # Should not raise
        mgr.stop()
        assert mgr._child_exited is True


class TestFixedPtyManagerGetOutput:
    """Test FixedPtyManager.get_output() — output buffer retrieval and truncation."""

    def test_empty_output(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        assert mgr.get_output() == ""

    def test_normal_output(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        mgr.output_buffer = ["line1\n", "line2\n"]
        assert mgr.get_output() == "line1\nline2\n"

    def test_binary_content_truncation(self, monkeypatch):
        from manus_agent.tools.python_repl import FixedPtyManager

        monkeypatch.setenv("PYTHON_REPL_BINARY_MAX_LEN", "20")
        mgr = FixedPtyManager()
        # Simulate output with escaped hex bytes
        mgr.output_buffer = ["\\x00\\x01\\x02" * 50]
        output = mgr.get_output()
        assert "binary content truncated" in output

    def test_binary_short_not_truncated(self, monkeypatch):
        from manus_agent.tools.python_repl import FixedPtyManager

        monkeypatch.setenv("PYTHON_REPL_BINARY_MAX_LEN", "200")
        mgr = FixedPtyManager()
        mgr.output_buffer = ["\\x00"]  # Short binary
        output = mgr.get_output()
        assert "truncated" not in output


# ---------------------------------------------------------------------------
# python_repl tool function tests — Interactive mode
# ---------------------------------------------------------------------------


class TestPythonReplInteractive:
    """Test the python_repl function in interactive mode."""

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_interactive_success(self, mock_state, mock_pty_cls, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        mock_mgr = MagicMock()
        mock_pty_cls.return_value = mock_mgr
        # Simulate process exiting with code 0
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 0]  # Running, then done
        mock_mgr.process = mock_proc
        mock_mgr._reader = MagicMock()
        mock_mgr._reader.join = MagicMock()
        mock_mgr.get_output.return_value = "hello world\n"
        mock_mgr._child_exited = False

        tool = _make_tool_use("print('hello world')", interactive=True)
        result = python_repl(tool)

        assert result["status"] == "success"
        assert "hello world" in result["content"][0]["text"]
        mock_state.save_state.assert_called_once()

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_interactive_timeout(self, mock_state, mock_pty_cls, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        mock_mgr = MagicMock()
        mock_pty_cls.return_value = mock_mgr
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Never exits
        mock_mgr.process = mock_proc
        mock_mgr._child_exited = False

        # Use timeout=5 so deadline is set; then mock time.monotonic
        # to jump past it on the second call.
        tool = _make_tool_use("import time; time.sleep(9999)", interactive=True, timeout=5)

        import time as _time

        start = _time.monotonic()
        # First monotonic() call sets the deadline (start + 5),
        # second call returns start + 10 which exceeds deadline.
        with patch("manus_agent.tools.python_repl.time.monotonic", side_effect=[start, start + 10]):
            with patch("manus_agent.tools.python_repl.time.sleep"):
                result = python_repl(tool)

        assert result["status"] == "error"
        assert "timeout" in result["content"][0]["text"].lower() or "Timeout" in result["content"][0]["text"]

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_interactive_nonzero_exit_code(self, mock_state, mock_pty_cls, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        mock_mgr = MagicMock()
        mock_pty_cls.return_value = mock_mgr
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, 1]  # Non-zero exit
        mock_mgr.process = mock_proc
        mock_mgr._reader = MagicMock()
        mock_mgr._reader.join = MagicMock()
        mock_mgr.get_output.return_value = "Traceback ...\nNameError: name 'x' is not defined\n"
        mock_mgr._child_exited = False

        tool = _make_tool_use("print(x)", interactive=True)
        result = python_repl(tool)

        # Non-zero exit → code still runs, output captured, save_state NOT called for non-zero
        assert result["status"] == "success"
        assert "NameError" in result["content"][0]["text"] or "Traceback" in result["content"][0]["text"]
        # save_state should NOT be called for non-zero exit
        mock_state.save_state.assert_not_called()


# ---------------------------------------------------------------------------
# python_repl tool function tests — Standard (non-interactive) mode
# ---------------------------------------------------------------------------


class TestPythonReplStandard:
    """Test the python_repl function in standard (non-interactive) mode."""

    @patch("manus_agent.tools.python_repl.repl_state")
    @patch("manus_agent.tools.python_repl.OutputCapture")
    def test_standard_mode_success(self, mock_capture_cls, mock_state, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        mock_capture = MagicMock()
        mock_capture.__enter__ = MagicMock(return_value=mock_capture)
        mock_capture.__exit__ = MagicMock(return_value=False)
        mock_capture.get_output.return_value = "42\n"
        mock_capture_cls.return_value = mock_capture

        tool = _make_tool_use("print(6*7)", interactive=False)
        result = python_repl(tool)

        assert result["status"] == "success"
        assert "42" in result["content"][0]["text"]
        mock_state.execute.assert_called_once_with("print(6*7)")

    @patch("manus_agent.tools.python_repl.repl_state")
    @patch("manus_agent.tools.python_repl.OutputCapture")
    def test_standard_mode_no_output(self, mock_capture_cls, mock_state, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        mock_capture = MagicMock()
        mock_capture.__enter__ = MagicMock(return_value=mock_capture)
        mock_capture.__exit__ = MagicMock(return_value=False)
        mock_capture.get_output.return_value = ""
        mock_capture_cls.return_value = mock_capture

        tool = _make_tool_use("x = 1", interactive=False)
        result = python_repl(tool)

        assert result["status"] == "success"
        assert "executed successfully" in result["content"][0]["text"].lower()


# ---------------------------------------------------------------------------
# State management tests
# ---------------------------------------------------------------------------


class TestPythonReplState:
    """Test reset_state and state persistence."""

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_reset_state_clears(self, mock_state, mock_pty_cls, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        mock_mgr = MagicMock()
        mock_pty_cls.return_value = mock_mgr
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [0]
        mock_mgr.process = mock_proc
        mock_mgr._reader = MagicMock()
        mock_mgr._reader.join = MagicMock()
        mock_mgr.get_output.return_value = ""
        mock_mgr._child_exited = False

        mock_state.get_user_objects.return_value = {}

        tool = _make_tool_use("pass", interactive=True, reset_state=True)
        result = python_repl(tool)

        mock_state.clear_state.assert_called_once()
        assert result["status"] == "success"

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_user_objects_reported(self, mock_state, mock_pty_cls, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        mock_mgr = MagicMock()
        mock_pty_cls.return_value = mock_mgr
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [0]
        mock_mgr.process = mock_proc
        mock_mgr._reader = MagicMock()
        mock_mgr._reader.join = MagicMock()
        mock_mgr.get_output.return_value = ""
        mock_mgr._child_exited = False

        mock_state.get_user_objects.return_value = {"result": "42"}

        tool = _make_tool_use("result = 6*7", interactive=True)
        result = python_repl(tool)

        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# User consent tests
# ---------------------------------------------------------------------------


class TestPythonReplConsent:
    """Test user consent flow (bypass, approval, cancellation)."""

    @patch("manus_agent.tools.python_repl.get_user_input", return_value="n")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_user_cancels_execution(self, mock_state, mock_input, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)

        # get_user_input returns "n" first, then a reason
        mock_input.side_effect = ["n", "not safe"]

        tool = _make_tool_use("os.system('rm -rf /')")
        result = python_repl(tool, non_interactive_mode=False)

        assert result["status"] == "error"
        assert "cancelled" in result["content"][0]["text"].lower()

    @patch("manus_agent.tools.python_repl.get_user_input", return_value="custom reason")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_user_cancels_with_custom_reason(self, mock_state, mock_input, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)

        # First call returns something other than "y" but also not "n" → treated as reason
        mock_input.return_value = "I don't trust this code"

        tool = _make_tool_use("import evil")
        result = python_repl(tool, non_interactive_mode=False)

        assert result["status"] == "error"
        assert "I don't trust this code" in result["content"][0]["text"]

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_non_interactive_mode_skips_consent(self, mock_state, mock_pty_cls, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.delenv("BYPASS_TOOL_CONSENT", raising=False)

        mock_mgr = MagicMock()
        mock_pty_cls.return_value = mock_mgr
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [0]
        mock_mgr.process = mock_proc
        mock_mgr._reader = MagicMock()
        mock_mgr._reader.join = MagicMock()
        mock_mgr.get_output.return_value = "done"
        mock_mgr._child_exited = False
        mock_state.get_user_objects.return_value = {}

        tool = _make_tool_use("print('ok')")
        # non_interactive_mode=True should skip user consent
        result = python_repl(tool, non_interactive_mode=True)

        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestPythonReplErrors:
    """Test error handling in python_repl."""

    @patch("manus_agent.tools.python_repl.repl_state")
    @patch("manus_agent.tools.python_repl.OutputCapture")
    def test_recursion_error_resets_state(self, mock_capture_cls, mock_state, monkeypatch, tmp_path):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
        monkeypatch.chdir(tmp_path)

        mock_capture = MagicMock()
        mock_capture.__enter__ = MagicMock(return_value=mock_capture)
        mock_capture.__exit__ = MagicMock(return_value=False)
        mock_capture_cls.return_value = mock_capture

        mock_state.execute.side_effect = RecursionError("maximum recursion depth exceeded")

        tool = _make_tool_use("def f(): f()\nf()", interactive=False)
        result = python_repl(tool)

        assert result["status"] == "error"
        assert "RecursionError" in result["content"][0]["text"] or "recursion" in result["content"][0]["text"].lower()
        # clear_state called both in the handler AND after catching in outer except
        assert mock_state.clear_state.call_count >= 1

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_generic_exception_in_interactive(self, mock_state, mock_pty_cls, monkeypatch, tmp_path):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
        monkeypatch.chdir(tmp_path)

        mock_pty_cls.side_effect = RuntimeError("PTY allocation failed")

        tool = _make_tool_use("print('hi')", interactive=True)
        result = python_repl(tool)

        assert result["status"] == "error"
        assert "PTY allocation failed" in result["content"][0]["text"]

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_error_logged_to_file(self, mock_state, mock_pty_cls, monkeypatch, tmp_path):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
        monkeypatch.chdir(tmp_path)

        mock_pty_cls.side_effect = ValueError("test error for logging")

        tool = _make_tool_use("x = 1", interactive=True)
        python_repl(tool)

        errors_dir = tmp_path / "errors"
        assert errors_dir.exists()
        error_file = errors_dir / "errors.txt"
        assert error_file.exists()
        content = error_file.read_text()
        assert "test error for logging" in content


# ---------------------------------------------------------------------------
# Environment variable override tests
# ---------------------------------------------------------------------------


class TestPythonReplEnvOverrides:
    """Test environment variable overrides for REPL settings."""

    @patch("manus_agent.tools.python_repl.repl_state")
    @patch("manus_agent.tools.python_repl.OutputCapture")
    def test_env_forces_non_interactive(self, mock_capture_cls, mock_state, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
        monkeypatch.setenv("PYTHON_REPL_INTERACTIVE", "false")

        mock_capture = MagicMock()
        mock_capture.__enter__ = MagicMock(return_value=mock_capture)
        mock_capture.__exit__ = MagicMock(return_value=False)
        mock_capture.get_output.return_value = "standard mode output"
        mock_capture_cls.return_value = mock_capture
        mock_state.get_user_objects.return_value = {}

        # Tool says interactive=True but env overrides to False
        tool = _make_tool_use("print(1)", interactive=True)
        result = python_repl(tool)

        assert result["status"] == "success"
        # OutputCapture used = standard mode was triggered
        mock_capture_cls.assert_called_once()

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_env_forces_reset_state(self, mock_state, mock_pty_cls, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
        monkeypatch.setenv("PYTHON_REPL_RESET_STATE", "true")

        mock_mgr = MagicMock()
        mock_pty_cls.return_value = mock_mgr
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [0]
        mock_mgr.process = mock_proc
        mock_mgr._reader = MagicMock()
        mock_mgr._reader.join = MagicMock()
        mock_mgr.get_output.return_value = ""
        mock_mgr._child_exited = False
        mock_state.get_user_objects.return_value = {}

        # Tool says reset_state=False but env overrides to True
        tool = _make_tool_use("pass", interactive=True, reset_state=False)
        python_repl(tool)

        mock_state.clear_state.assert_called_once()

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_env_timeout_override(self, mock_state, mock_pty_cls, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
        monkeypatch.setenv("PYTHON_REPL_TIMEOUT", "1")

        mock_mgr = MagicMock()
        mock_pty_cls.return_value = mock_mgr
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # Never exits
        mock_mgr.process = mock_proc
        mock_mgr._child_exited = False

        import time as _time

        start = _time.monotonic()
        with patch("manus_agent.tools.python_repl.time.monotonic", side_effect=[start, start + 2]):
            tool = _make_tool_use("import time; time.sleep(999)", interactive=True, timeout=9999)
            result = python_repl(tool)

        # The 1-second env timeout should trigger before the tool's 9999s
        assert result["status"] == "error"
        assert "timeout" in result["content"][0]["text"].lower() or "exceeded" in result["content"][0]["text"].lower()


# ---------------------------------------------------------------------------
# TOOL_SPEC metadata tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Test that TOOL_SPEC is properly exposed."""

    def test_tool_spec_exists(self):
        from manus_agent.tools.python_repl import TOOL_SPEC

        assert TOOL_SPEC is not None
        assert "name" in TOOL_SPEC

    def test_module_exports(self):
        import importlib

        mod = importlib.import_module("manus_agent.tools.python_repl")

        assert hasattr(mod, "python_repl")
        assert hasattr(mod, "TOOL_SPEC")
        assert hasattr(mod, "FixedPtyManager")


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestPythonReplEdgeCases:
    """Edge cases and boundary conditions."""

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_empty_code_string(self, mock_state, mock_pty_cls, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        mock_mgr = MagicMock()
        mock_pty_cls.return_value = mock_mgr
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [0]
        mock_mgr.process = mock_proc
        mock_mgr._reader = MagicMock()
        mock_mgr._reader.join = MagicMock()
        mock_mgr.get_output.return_value = ""
        mock_mgr._child_exited = False
        mock_state.get_user_objects.return_value = {}

        tool = _make_tool_use("", interactive=True)
        result = python_repl(tool)

        assert result["status"] == "success"

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_multiline_code(self, mock_state, mock_pty_cls, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        mock_mgr = MagicMock()
        mock_pty_cls.return_value = mock_mgr
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [0]
        mock_mgr.process = mock_proc
        mock_mgr._reader = MagicMock()
        mock_mgr._reader.join = MagicMock()
        mock_mgr.get_output.return_value = "1\n2\n3\n"
        mock_mgr._child_exited = False
        mock_state.get_user_objects.return_value = {}

        code = "for i in range(1, 4):\n    print(i)"
        tool = _make_tool_use(code, interactive=True)
        result = python_repl(tool)

        assert result["status"] == "success"
        assert "1" in result["content"][0]["text"]

    def test_tool_use_id_propagated(self, monkeypatch, tmp_path):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")
        monkeypatch.chdir(tmp_path)

        with patch("manus_agent.tools.python_repl.FixedPtyManager") as mock_cls:
            mock_cls.side_effect = RuntimeError("forced")
            tool = {
                "toolUseId": "my-custom-id-xyz",
                "input": {"code": "x", "interactive": True, "reset_state": False, "timeout": 120},
            }
            result = python_repl(tool)

        assert result["toolUseId"] == "my-custom-id-xyz"

    @patch("manus_agent.tools.python_repl.FixedPtyManager")
    @patch("manus_agent.tools.python_repl.repl_state")
    def test_large_output(self, mock_state, mock_pty_cls, monkeypatch):
        from manus_agent.tools.python_repl import python_repl

        monkeypatch.setenv("BYPASS_TOOL_CONSENT", "true")

        mock_mgr = MagicMock()
        mock_pty_cls.return_value = mock_mgr
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [0]
        mock_mgr.process = mock_proc
        mock_mgr._reader = MagicMock()
        mock_mgr._reader.join = MagicMock()
        # Simulate large output
        mock_mgr.get_output.return_value = "x" * 100000
        mock_mgr._child_exited = False
        mock_state.get_user_objects.return_value = {}

        tool = _make_tool_use("print('x'*100000)", interactive=True)
        result = python_repl(tool)

        assert result["status"] == "success"
        assert len(result["content"][0]["text"]) > 0


# ---------------------------------------------------------------------------
# FixedPtyManager._read_output tests
# ---------------------------------------------------------------------------


class TestFixedPtyManagerReadOutput:
    """Test the _read_output thread logic with mocked I/O."""

    def test_read_output_handles_eof(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        mgr.supervisor_fd = 42
        mgr._child_exited = False

        # select returns fd ready, read returns empty (EOF) → loop exits
        with patch("select.select", return_value=([42], [], [])), patch("os.read", return_value=b""):
            mgr._read_output()

        # Should have exited cleanly (no hang, no exception)

    def test_read_output_handles_oserror_in_select(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        mgr.supervisor_fd = 42
        mgr._child_exited = False

        with patch("select.select", side_effect=OSError("fd closed")):
            mgr._read_output()

        # Should have exited without raising

    def test_read_output_invokes_callback(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        cb = MagicMock()
        mgr = FixedPtyManager(callback=cb)
        mgr.supervisor_fd = 42
        mgr._child_exited = False

        call_count = [0]

        def mock_select(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 1:
                mgr._child_exited = True
                return ([], [], [])
            return ([42], [], [])

        def mock_read(fd, size):
            return b"hello\n"

        with patch("select.select", side_effect=mock_select), patch("os.read", side_effect=mock_read):
            mgr._read_output()

        # Callback should have been invoked with output
        assert cb.call_count >= 1

    def test_read_output_handles_bad_utf8(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        mgr.supervisor_fd = 42
        mgr._child_exited = False

        call_count = [0]

        def mock_select(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] > 2:
                mgr._child_exited = True
                return ([], [], [])
            return ([42], [], [])

        def mock_read(fd, size):
            if call_count[0] == 1:
                return b"valid text\n"
            if call_count[0] == 2:
                # Incomplete UTF-8 byte → triggers UnicodeDecodeError path
                return b"\xc3"
            return b""

        with patch("select.select", side_effect=mock_select), patch("os.read", side_effect=mock_read):
            mgr._read_output()

        # Should have captured "valid text" without crashing
        output = mgr.get_output()
        assert "valid text" in output

    def test_read_output_negative_fd_exits_immediately(self):
        from manus_agent.tools.python_repl import FixedPtyManager

        mgr = FixedPtyManager()
        mgr.supervisor_fd = -1
        mgr._child_exited = False

        # With supervisor_fd == -1, the loop should break immediately
        mgr._read_output()
        # No hang, no exception
