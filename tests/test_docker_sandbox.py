"""Comprehensive test suite for manus_agent.sandbox.docker_sandbox module.

Covers:
- DockerSandbox.__init__ defaults and custom params
- start() / _start_sync() — image pull, container creation, waiting
- stop() / _stop_sync() — container cleanup, client close
- execute_command() — success, timeout, runtime error
- _execute_sync() — output decoding, demux handling
- _copy_to_container() — tar archive creation
- _get_file_extension() — language to extension mapping
- _get_execution_command() — language to command mapping

All Docker interactions are fully mocked — no real Docker daemon needed.
"""

import asyncio
import io
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import ImageNotFound

from manus_agent.sandbox.docker_sandbox import DockerSandbox


# =====================================================================
# __init__ defaults
# =====================================================================


class TestDockerSandboxInit:
    """Tests for DockerSandbox initialization."""

    def test_default_values(self):
        sb = DockerSandbox()
        assert sb.image == "python:3.12-slim"
        assert sb.memory_limit == "512m"
        assert sb.cpu_limit == 1.0
        assert sb.network_disabled is False
        assert sb.client is None
        assert sb.container is None
        assert sb.container_name.startswith("manus-sandbox-")

    def test_custom_values(self):
        sb = DockerSandbox(
            image="node:20-slim",
            memory_limit="1g",
            cpu_limit=2.0,
            network_disabled=True,
        )
        assert sb.image == "node:20-slim"
        assert sb.memory_limit == "1g"
        assert sb.cpu_limit == 2.0
        assert sb.network_disabled is True

    def test_container_name_unique(self):
        sb1 = DockerSandbox()
        sb2 = DockerSandbox()
        assert sb1.container_name != sb2.container_name

    def test_container_name_format(self):
        sb = DockerSandbox()
        assert sb.container_name.startswith("manus-sandbox-")
        suffix = sb.container_name.replace("manus-sandbox-", "")
        assert len(suffix) == 8


# =====================================================================
# _start_sync
# =====================================================================


class TestDockerSandboxStartSync:
    """Tests for synchronous container start."""

    @patch("manus_agent.sandbox.docker_sandbox.wait_for_container_running")
    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    @patch("manus_agent.sandbox.docker_sandbox.get_docker_client")
    def test_start_sync_image_exists(self, mock_get_client, mock_retry, mock_wait):
        sb = DockerSandbox()
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_container = MagicMock()
        mock_retry.side_effect = [None, mock_container]

        sb._start_sync()

        assert sb.client == mock_client
        assert sb.container == mock_container
        assert mock_retry.call_count == 2
        mock_wait.assert_called_once_with(mock_container, timeout=20)

    @patch("manus_agent.sandbox.docker_sandbox.wait_for_container_running")
    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    @patch("manus_agent.sandbox.docker_sandbox.get_docker_client")
    def test_start_sync_pulls_missing_image(self, mock_get_client, mock_retry, mock_wait):
        sb = DockerSandbox()
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_container = MagicMock()
        mock_retry.side_effect = [ImageNotFound("not found"), None, mock_container]

        sb._start_sync()

        assert sb.container == mock_container
        assert mock_retry.call_count == 3

    @patch("manus_agent.sandbox.docker_sandbox.wait_for_container_running")
    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    @patch("manus_agent.sandbox.docker_sandbox.get_docker_client")
    def test_start_sync_sets_client(self, mock_get_client, mock_retry, mock_wait):
        sb = DockerSandbox()
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_retry.side_effect = [None, MagicMock()]

        sb._start_sync()
        assert sb.client is mock_client


# =====================================================================
# start (async wrapper)
# =====================================================================


class TestDockerSandboxStart:
    """Tests for async start."""

    @pytest.mark.asyncio
    async def test_start_calls_start_sync(self):
        sb = DockerSandbox()
        with patch.object(sb, "_start_sync") as mock_sync:
            await sb.start()
            mock_sync.assert_called_once()


# =====================================================================
# _stop_sync / stop
# =====================================================================


class TestDockerSandboxStop:
    """Tests for container stop and cleanup."""

    @patch("manus_agent.sandbox.docker_sandbox.safe_kill_remove_container")
    def test_stop_sync_cleans_up(self, mock_safe_kill):
        sb = DockerSandbox()
        sb.container = MagicMock()
        sb.client = MagicMock()

        sb._stop_sync()

        mock_safe_kill.assert_called_once()
        assert sb.container is None
        assert sb.client is None

    @patch("manus_agent.sandbox.docker_sandbox.safe_kill_remove_container")
    def test_stop_sync_handles_client_close_error(self, mock_safe_kill):
        sb = DockerSandbox()
        sb.container = MagicMock()
        sb.client = MagicMock()
        sb.client.close.side_effect = Exception("close error")

        sb._stop_sync()

        assert sb.container is None
        assert sb.client is None

    @patch("manus_agent.sandbox.docker_sandbox.safe_kill_remove_container")
    def test_stop_sync_client_none(self, mock_safe_kill):
        sb = DockerSandbox()
        sb.client = None
        sb._stop_sync()
        mock_safe_kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_async_calls_stop_sync_when_container_exists(self):
        sb = DockerSandbox()
        sb.container = MagicMock()
        with patch.object(sb, "_stop_sync") as mock_sync:
            await sb.stop()
            mock_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_async_noop_when_no_container(self):
        sb = DockerSandbox()
        sb.container = None
        with patch.object(sb, "_stop_sync") as mock_sync:
            await sb.stop()
            mock_sync.assert_not_called()


# =====================================================================
# _get_file_extension
# =====================================================================


class TestGetFileExtension:
    """Tests for language to file extension mapping."""

    def test_python(self):
        assert DockerSandbox()._get_file_extension("python") == "py"

    def test_javascript(self):
        assert DockerSandbox()._get_file_extension("javascript") == "js"

    def test_typescript(self):
        assert DockerSandbox()._get_file_extension("typescript") == "ts"

    def test_bash(self):
        assert DockerSandbox()._get_file_extension("bash") == "sh"

    def test_shell(self):
        assert DockerSandbox()._get_file_extension("shell") == "sh"

    def test_sh(self):
        assert DockerSandbox()._get_file_extension("sh") == "sh"

    def test_unknown_defaults_to_txt(self):
        assert DockerSandbox()._get_file_extension("rust") == "txt"

    def test_case_insensitive(self):
        assert DockerSandbox()._get_file_extension("Python") == "py"
        assert DockerSandbox()._get_file_extension("JAVASCRIPT") == "js"


# =====================================================================
# _get_execution_command
# =====================================================================


class TestGetExecutionCommand:
    """Tests for language to execution command mapping."""

    def test_python_uses_python3_fallback(self):
        cmd = DockerSandbox()._get_execution_command("python", "/tmp/code.py")
        assert "python3" in cmd
        assert "/tmp/code.py" in cmd

    def test_javascript_uses_node(self):
        assert DockerSandbox()._get_execution_command("javascript", "/tmp/c.js") == "node /tmp/c.js"

    def test_bash_uses_bash(self):
        assert DockerSandbox()._get_execution_command("bash", "/tmp/c.sh") == "bash /tmp/c.sh"

    def test_shell_uses_sh(self):
        assert DockerSandbox()._get_execution_command("shell", "/tmp/c.sh") == "sh /tmp/c.sh"

    def test_sh_uses_sh(self):
        assert DockerSandbox()._get_execution_command("sh", "/tmp/c.sh") == "sh /tmp/c.sh"

    def test_unknown_uses_cat(self):
        assert DockerSandbox()._get_execution_command("rust", "/tmp/c.txt") == "cat /tmp/c.txt"


# =====================================================================
# _execute_sync
# =====================================================================


class TestExecuteSync:
    """Tests for synchronous command execution."""

    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    def test_returns_decoded_output(self, mock_retry):
        sb = DockerSandbox()
        sb.container = MagicMock()

        mock_result = MagicMock()
        mock_result.output = (b"hello stdout", b"hello stderr")
        mock_result.exit_code = 0
        mock_retry.return_value = mock_result

        stdout, stderr, exit_code = sb._execute_sync("echo hello")
        assert stdout == "hello stdout"
        assert stderr == "hello stderr"
        assert exit_code == 0

    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    def test_handles_none_output(self, mock_retry):
        sb = DockerSandbox()
        sb.container = MagicMock()

        mock_result = MagicMock()
        mock_result.output = (None, None)
        mock_result.exit_code = 0
        mock_retry.return_value = mock_result

        stdout, stderr, exit_code = sb._execute_sync("cmd")
        assert stdout == ""
        assert stderr == ""
        assert exit_code == 0

    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    def test_handles_non_utf8_output(self, mock_retry):
        sb = DockerSandbox()
        sb.container = MagicMock()

        mock_result = MagicMock()
        mock_result.output = (b"\x80\x81\x82", b"\xff\xfe")
        mock_result.exit_code = 1
        mock_retry.return_value = mock_result

        stdout, stderr, exit_code = sb._execute_sync("cmd")
        assert isinstance(stdout, str)
        assert isinstance(stderr, str)
        assert exit_code == 1

    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    def test_nonzero_exit_code(self, mock_retry):
        sb = DockerSandbox()
        sb.container = MagicMock()

        mock_result = MagicMock()
        mock_result.output = (b"", b"error: command not found")
        mock_result.exit_code = 127
        mock_retry.return_value = mock_result

        stdout, stderr, exit_code = sb._execute_sync("nonexistent_cmd")
        assert exit_code == 127
        assert "command not found" in stderr


# =====================================================================
# execute_command
# =====================================================================


class TestExecuteCommand:
    """Tests for async command execution."""

    @pytest.mark.asyncio
    async def test_raises_when_no_container(self):
        sb = DockerSandbox()
        sb.container = None
        with pytest.raises(RuntimeError, match="not running"):
            await sb.execute_command("echo hello")

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        sb = DockerSandbox()
        sb.container = MagicMock()
        with patch.object(sb, "_execute_sync", return_value=("out", "err", 0)):
            result = await sb.execute_command("echo hello", timeout=30)
        assert result == ("out", "err", 0)

    @pytest.mark.asyncio
    async def test_timeout_returns_error_tuple(self):
        sb = DockerSandbox()
        sb.container = MagicMock()

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            result = await sb.execute_command("long_command", timeout=1)
        assert result[2] == -1
        assert "timed out" in result[1].lower()


# =====================================================================
# execute_code
# =====================================================================


class TestExecuteCode:
    """Tests for code execution in sandbox."""

    @pytest.mark.asyncio
    async def test_raises_when_no_container(self):
        sb = DockerSandbox()
        sb.container = None
        with pytest.raises(RuntimeError, match="not running"):
            await sb.execute_code("print('hello')", language="python")


# =====================================================================
# _copy_to_container
# =====================================================================


class TestCopyToContainer:
    """Tests for file copy to container via tar archive."""

    def test_creates_tar_and_copies(self):
        sb = DockerSandbox()
        sb.container = MagicMock()

        captured_data = []

        def capture_put_archive(path, data):
            captured_data.append((path, data))

        sb.container.put_archive = capture_put_archive

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')")
            temp_path = f.name

        try:
            sb._copy_to_container(temp_path, "/tmp/code.py")
        finally:
            Path(temp_path).unlink(missing_ok=True)

        assert len(captured_data) == 1
        assert str(captured_data[0][0]) == "/tmp"

    def test_tar_contains_correct_filename(self):
        sb = DockerSandbox()
        sb.container = MagicMock()

        captured_data = []

        def capture_put_archive(path, data):
            captured_data.append((path, data))

        sb.container.put_archive = capture_put_archive

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("print('hello')")
            temp_path = f.name

        try:
            sb._copy_to_container(temp_path, "/tmp/code.py")
        finally:
            Path(temp_path).unlink(missing_ok=True)

        assert len(captured_data) == 1
        tar_data = captured_data[0][1]
        with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r") as tar:
            members = tar.getnames()
            assert "code.py" in members
            member = tar.getmember("code.py")
            assert member.mode == 0o755

    def test_tar_contains_correct_content(self):
        sb = DockerSandbox()
        sb.container = MagicMock()

        captured_data = []

        def capture_put_archive(path, data):
            captured_data.append((path, data))

        sb.container.put_archive = capture_put_archive

        code = "x = 42\nprint(x)"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            sb._copy_to_container(temp_path, "/workspace/script.py")
        finally:
            Path(temp_path).unlink(missing_ok=True)

        tar_data = captured_data[0][1]
        with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r") as tar:
            extracted = tar.extractfile("script.py")
            assert extracted is not None
            assert extracted.read().decode() == code
        assert str(captured_data[0][0]) == "/workspace"
