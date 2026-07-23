"""Comprehensive test suite for manus_agent.sandbox.docker_sandbox module.

Tests cover:
- DockerSandbox initialization (defaults, custom params)
- start() / _start_sync() (image pull, container creation, waiting)
- stop() / _stop_sync() (cleanup, idempotence)
- execute_code() (language routing, temp file lifecycle, error handling)
- execute_command() (normal execution, timeout, kill fallback)
- _execute_sync() (output demuxing, None output handling)
- _copy_to_container() (tar archive creation, put_archive call)
- _get_file_extension() (known languages, unknown fallback)
- _get_execution_command() (language-specific commands, unknown fallback)

All tests are fully mocked — no real Docker connections.
"""

import asyncio
import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.sandbox.docker_sandbox import DockerSandbox

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox():
    """Create a DockerSandbox with default settings."""
    return DockerSandbox()


@pytest.fixture
def sandbox_custom():
    """Create a DockerSandbox with custom settings."""
    return DockerSandbox(
        image="node:20-slim",
        memory_limit="1g",
        cpu_limit=2.0,
        network_disabled=True,
    )


@pytest.fixture
def mock_client():
    """Create a mock Docker client."""
    client = MagicMock()
    client.images = MagicMock()
    client.containers = MagicMock()
    return client


@pytest.fixture
def mock_container():
    """Create a mock Docker container."""
    container = MagicMock()
    container.status = "running"
    container.reload = MagicMock()
    container.exec_run = MagicMock()
    container.put_archive = MagicMock()
    return container


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestDockerSandboxInit:
    """Tests for DockerSandbox.__init__."""

    def test_default_params(self, sandbox):
        """Default init uses python:3.12-slim, 512m, 1.0 CPU, network enabled."""
        assert sandbox.image == "python:3.12-slim"
        assert sandbox.memory_limit == "512m"
        assert sandbox.cpu_limit == 1.0
        assert sandbox.network_disabled is False
        assert sandbox.client is None
        assert sandbox.container is None

    def test_custom_params(self, sandbox_custom):
        """Custom init respects all provided parameters."""
        assert sandbox_custom.image == "node:20-slim"
        assert sandbox_custom.memory_limit == "1g"
        assert sandbox_custom.cpu_limit == 2.0
        assert sandbox_custom.network_disabled is True

    def test_container_name_format(self, sandbox):
        """Container name starts with 'manus-sandbox-' and has 8 hex chars."""
        assert sandbox.container_name.startswith("manus-sandbox-")
        hex_part = sandbox.container_name.replace("manus-sandbox-", "")
        assert len(hex_part) == 8
        int(hex_part, 16)  # Validates it's valid hex

    def test_unique_container_names(self):
        """Each instance gets a unique container name."""
        s1 = DockerSandbox()
        s2 = DockerSandbox()
        assert s1.container_name != s2.container_name


# ---------------------------------------------------------------------------
# Start Tests
# ---------------------------------------------------------------------------


class TestDockerSandboxStart:
    """Tests for DockerSandbox.start() and _start_sync()."""

    @patch("manus_agent.sandbox.docker_sandbox.wait_for_container_running")
    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    @patch("manus_agent.sandbox.docker_sandbox.get_docker_client")
    def test_start_pulls_missing_image(self, mock_get_client, mock_retry, mock_wait):
        """When images.get raises ImageNotFound, start pulls the image."""
        from docker.errors import ImageNotFound

        client = MagicMock()
        mock_get_client.return_value = client

        container = MagicMock()

        call_count = [0]

        def retry_side_effect(name, fn):
            call_count[0] += 1
            if call_count[0] == 1:
                # images.get — raise ImageNotFound
                raise ImageNotFound("not found")
            elif call_count[0] == 2:
                # images.pull
                return None
            else:
                # containers.run
                return container

        mock_retry.side_effect = retry_side_effect

        sandbox = DockerSandbox()
        sandbox._start_sync()

        assert mock_retry.call_count == 3
        # Verify pull was called (second call)
        assert "images.pull" in mock_retry.call_args_list[1][0][0]
        mock_wait.assert_called_once_with(container, timeout=20)

    @patch("manus_agent.sandbox.docker_sandbox.wait_for_container_running")
    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    @patch("manus_agent.sandbox.docker_sandbox.get_docker_client")
    def test_start_skips_pull_when_image_exists(self, mock_get_client, mock_retry, mock_wait):
        """When images.get succeeds, start does not pull."""
        client = MagicMock()
        mock_get_client.return_value = client

        container = MagicMock()
        image_obj = MagicMock()

        call_count = [0]

        def retry_side_effect(name, fn):
            call_count[0] += 1
            if call_count[0] == 1:
                # images.get — success
                return image_obj
            else:
                # containers.run
                return container

        mock_retry.side_effect = retry_side_effect

        sandbox = DockerSandbox()
        sandbox._start_sync()

        assert mock_retry.call_count == 2
        assert sandbox.container is container
        mock_wait.assert_called_once_with(container, timeout=20)

    @patch("manus_agent.sandbox.docker_sandbox.wait_for_container_running")
    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    @patch("manus_agent.sandbox.docker_sandbox.get_docker_client")
    def test_start_passes_correct_container_params(self, mock_get_client, mock_retry, mock_wait):
        """Container is created with correct mem_limit, cpu_quota, labels, etc."""
        client = MagicMock()
        mock_get_client.return_value = client

        container = MagicMock()

        def retry_side_effect(name, fn):
            result = fn()
            return result

        mock_retry.side_effect = retry_side_effect
        client.images.get.return_value = MagicMock()
        client.containers.run.return_value = container

        sandbox = DockerSandbox(
            image="alpine:3.19",
            memory_limit="256m",
            cpu_limit=0.5,
            network_disabled=True,
        )
        sandbox._start_sync()

        client.containers.run.assert_called_once()
        kwargs = client.containers.run.call_args
        assert kwargs[1]["mem_limit"] == "256m"
        assert kwargs[1]["cpu_quota"] == 50000  # 0.5 * 100000
        assert kwargs[1]["cpu_period"] == 100000
        assert kwargs[1]["network_disabled"] is True
        assert kwargs[1]["command"] == "sleep infinity"
        assert kwargs[1]["labels"]["manus_agent.component"] == "docker_sandbox"

    @patch("manus_agent.sandbox.docker_sandbox.wait_for_container_running")
    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    @patch("manus_agent.sandbox.docker_sandbox.get_docker_client")
    def test_start_async(self, mock_get_client, mock_retry, mock_wait):
        """start() wraps _start_sync in run_in_executor."""
        client = MagicMock()
        mock_get_client.return_value = client
        container = MagicMock()

        def retry_side_effect(name, fn):
            return fn()

        mock_retry.side_effect = retry_side_effect
        client.images.get.return_value = MagicMock()
        client.containers.run.return_value = container

        sandbox = DockerSandbox()
        asyncio.run(sandbox.start())

        assert sandbox.container is container


# ---------------------------------------------------------------------------
# Stop Tests
# ---------------------------------------------------------------------------


class TestDockerSandboxStop:
    """Tests for DockerSandbox.stop() and _stop_sync()."""

    @patch("manus_agent.sandbox.docker_sandbox.safe_kill_remove_container")
    def test_stop_sync_removes_container(self, mock_kill):
        """_stop_sync calls safe_kill_remove_container and clears state."""
        sandbox = DockerSandbox()
        mock_container = MagicMock()
        mock_client = MagicMock()
        sandbox.container = mock_container
        sandbox.client = mock_client

        sandbox._stop_sync()

        mock_kill.assert_called_once_with(mock_container)
        assert sandbox.container is None
        assert sandbox.client is None
        mock_client.close.assert_called_once()

    @patch("manus_agent.sandbox.docker_sandbox.safe_kill_remove_container")
    def test_stop_sync_handles_client_close_error(self, mock_kill):
        """_stop_sync handles exceptions from client.close() gracefully."""
        sandbox = DockerSandbox()
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.close.side_effect = Exception("connection reset")
        sandbox.container = mock_container
        sandbox.client = mock_client

        sandbox._stop_sync()  # Should not raise

        assert sandbox.container is None
        assert sandbox.client is None

    def test_stop_with_no_container(self):
        """stop() is a no-op when container is None."""
        sandbox = DockerSandbox()
        sandbox.container = None
        # Should not raise
        asyncio.run(sandbox.stop())

    @patch("manus_agent.sandbox.docker_sandbox.safe_kill_remove_container")
    def test_stop_async(self, mock_kill):
        """stop() wraps _stop_sync in run_in_executor."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()
        sandbox.client = MagicMock()

        asyncio.run(sandbox.stop())

        mock_kill.assert_called_once()
        assert sandbox.container is None


# ---------------------------------------------------------------------------
# execute_code Tests
# ---------------------------------------------------------------------------


class TestDockerSandboxExecuteCode:
    """Tests for DockerSandbox.execute_code()."""

    def test_execute_code_raises_when_not_running(self):
        """execute_code raises RuntimeError if container is None."""
        sandbox = DockerSandbox()
        sandbox.container = None

        with pytest.raises(RuntimeError, match="not running"):
            asyncio.run(sandbox.execute_code("print('hi')"))

    @patch.object(DockerSandbox, "execute_command")
    @patch.object(DockerSandbox, "_copy_to_container")
    def test_execute_code_python(self, mock_copy, mock_exec):
        """execute_code for Python copies file and runs with python3."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        async def fake_exec(cmd, timeout):
            return ("output", "", 0)

        mock_exec.side_effect = fake_exec

        result = asyncio.run(sandbox.execute_code("print('hello')", language="python", timeout=10))

        assert result == ("output", "", 0)
        mock_copy.assert_called_once()
        # Verify the container path ends with .py
        container_path = mock_copy.call_args[0][1]
        assert container_path == "/tmp/code.py"

    @patch.object(DockerSandbox, "execute_command")
    @patch.object(DockerSandbox, "_copy_to_container")
    def test_execute_code_javascript(self, mock_copy, mock_exec):
        """execute_code for JavaScript copies .js file and runs with node."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        async def fake_exec(cmd, timeout):
            assert "node" in cmd
            return ("js output", "", 0)

        mock_exec.side_effect = fake_exec

        result = asyncio.run(sandbox.execute_code("console.log('hi')", language="javascript", timeout=15))

        assert result == ("js output", "", 0)
        container_path = mock_copy.call_args[0][1]
        assert container_path == "/tmp/code.js"

    @patch.object(DockerSandbox, "execute_command")
    @patch.object(DockerSandbox, "_copy_to_container")
    def test_execute_code_bash(self, mock_copy, mock_exec):
        """execute_code for bash copies .sh file and runs with bash."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        async def fake_exec(cmd, timeout):
            assert "bash" in cmd
            return ("bash output", "", 0)

        mock_exec.side_effect = fake_exec

        result = asyncio.run(sandbox.execute_code("echo hello", language="bash"))

        assert result == ("bash output", "", 0)
        container_path = mock_copy.call_args[0][1]
        assert container_path == "/tmp/code.sh"

    @patch.object(DockerSandbox, "execute_command")
    @patch.object(DockerSandbox, "_copy_to_container")
    def test_execute_code_cleans_temp_file(self, mock_copy, mock_exec):
        """Temp file is cleaned up even on success."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        async def fake_exec(cmd, timeout):
            return ("", "", 0)

        mock_exec.side_effect = fake_exec

        # After execute_code, the temp file should be removed
        asyncio.run(sandbox.execute_code("x = 1", language="python"))

        # The local_path passed to _copy_to_container should no longer exist
        local_path = mock_copy.call_args[0][0]
        assert not Path(local_path).exists()

    @patch.object(DockerSandbox, "execute_command")
    @patch.object(DockerSandbox, "_copy_to_container")
    def test_execute_code_cleans_temp_file_on_error(self, mock_copy, mock_exec):
        """Temp file is cleaned up even when execute_command raises."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        async def fake_exec(cmd, timeout):
            raise Exception("exec failed")

        mock_exec.side_effect = fake_exec

        with pytest.raises(Exception, match="exec failed"):
            asyncio.run(sandbox.execute_code("x = 1", language="python"))

        local_path = mock_copy.call_args[0][0]
        assert not Path(local_path).exists()

    @patch.object(DockerSandbox, "execute_command")
    @patch.object(DockerSandbox, "_copy_to_container")
    def test_execute_code_default_timeout(self, mock_copy, mock_exec):
        """Default timeout is 30 seconds."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        captured_timeout = []

        async def fake_exec(cmd, timeout):
            captured_timeout.append(timeout)
            return ("", "", 0)

        mock_exec.side_effect = fake_exec

        asyncio.run(sandbox.execute_code("x = 1"))

        assert captured_timeout[0] == 30


# ---------------------------------------------------------------------------
# execute_command Tests
# ---------------------------------------------------------------------------


class TestDockerSandboxExecuteCommand:
    """Tests for DockerSandbox.execute_command()."""

    def test_execute_command_raises_when_not_running(self):
        """execute_command raises RuntimeError if container is None."""
        sandbox = DockerSandbox()
        sandbox.container = None

        with pytest.raises(RuntimeError, match="not running"):
            asyncio.run(sandbox.execute_command("ls"))

    @patch.object(DockerSandbox, "_execute_sync")
    def test_execute_command_success(self, mock_exec_sync):
        """execute_command returns (stdout, stderr, exit_code) on success."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()
        mock_exec_sync.return_value = ("file1\nfile2\n", "", 0)

        result = asyncio.run(sandbox.execute_command("ls /tmp"))

        assert result == ("file1\nfile2\n", "", 0)

    @patch.object(DockerSandbox, "_execute_sync")
    def test_execute_command_timeout(self, mock_exec_sync):
        """execute_command returns timeout message when execution exceeds timeout."""
        sandbox = DockerSandbox()
        container = MagicMock()
        sandbox.container = container

        async def slow_exec():
            await asyncio.sleep(10)
            return ("", "", 0)

        # Simulate timeout by making _execute_sync block
        mock_exec_sync.side_effect = lambda cmd: asyncio.get_event_loop().run_until_complete(slow_exec())

        # Use a very short timeout to trigger TimeoutError
        # We need to patch asyncio.wait_for to raise TimeoutError
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            result = asyncio.run(sandbox.execute_command("sleep 100", timeout=1))

        assert result[0] == ""
        assert "timed out after 1 seconds" in result[1]
        assert result[2] == -1

    @patch.object(DockerSandbox, "_execute_sync")
    def test_execute_command_timeout_kill_fallback(self, mock_exec_sync):
        """On timeout, execute_command tries to kill the process."""
        sandbox = DockerSandbox()
        container = MagicMock()
        sandbox.container = container

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            asyncio.run(sandbox.execute_command("python3 heavy.py", timeout=5))

        # Verify kill attempt was made
        container.exec_run.assert_called_once()
        kill_cmd = container.exec_run.call_args[0][0]
        assert "pkill" in kill_cmd
        assert "python3" in kill_cmd

    @patch.object(DockerSandbox, "_execute_sync")
    def test_execute_command_timeout_kill_exception_ignored(self, mock_exec_sync):
        """If kill attempt fails, exception is suppressed."""
        sandbox = DockerSandbox()
        container = MagicMock()
        container.exec_run.side_effect = Exception("container gone")
        sandbox.container = container

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
            result = asyncio.run(sandbox.execute_command("python3 heavy.py", timeout=5))

        # Should still return timeout message without raising
        assert "timed out" in result[1]
        assert result[2] == -1

    @patch.object(DockerSandbox, "_execute_sync")
    def test_execute_command_none_timeout(self, mock_exec_sync):
        """execute_command with timeout=None passes None to wait_for."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()
        mock_exec_sync.return_value = ("ok", "", 0)

        result = asyncio.run(sandbox.execute_command("echo ok", timeout=None))

        assert result == ("ok", "", 0)


# ---------------------------------------------------------------------------
# _execute_sync Tests
# ---------------------------------------------------------------------------


class TestDockerSandboxExecuteSync:
    """Tests for DockerSandbox._execute_sync()."""

    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    def test_execute_sync_normal_output(self, mock_retry):
        """_execute_sync demuxes stdout and stderr correctly."""
        sandbox = DockerSandbox()
        container = MagicMock()
        sandbox.container = container

        exec_result = MagicMock()
        exec_result.output = (b"hello stdout", b"hello stderr")
        exec_result.exit_code = 0
        mock_retry.return_value = exec_result

        result = sandbox._execute_sync("echo hello")

        assert result == ("hello stdout", "hello stderr", 0)

    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    def test_execute_sync_none_stdout(self, mock_retry):
        """_execute_sync handles None stdout (no output)."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        exec_result = MagicMock()
        exec_result.output = (None, b"some error")
        exec_result.exit_code = 1
        mock_retry.return_value = exec_result

        result = sandbox._execute_sync("bad_cmd")

        assert result == ("", "some error", 1)

    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    def test_execute_sync_none_stderr(self, mock_retry):
        """_execute_sync handles None stderr."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        exec_result = MagicMock()
        exec_result.output = (b"output", None)
        exec_result.exit_code = 0
        mock_retry.return_value = exec_result

        result = sandbox._execute_sync("ls")

        assert result == ("output", "", 0)

    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    def test_execute_sync_both_none(self, mock_retry):
        """_execute_sync handles both stdout and stderr being None."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        exec_result = MagicMock()
        exec_result.output = (None, None)
        exec_result.exit_code = 0
        mock_retry.return_value = exec_result

        result = sandbox._execute_sync("true")

        assert result == ("", "", 0)

    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    def test_execute_sync_non_utf8_output(self, mock_retry):
        """_execute_sync handles non-UTF-8 bytes with replacement."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        exec_result = MagicMock()
        exec_result.output = (b"\xff\xfe binary", b"")
        exec_result.exit_code = 0
        mock_retry.return_value = exec_result

        result = sandbox._execute_sync("cat /bin/ls")

        # Should contain replacement characters, not raise
        assert "binary" in result[0]
        assert result[2] == 0

    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    def test_execute_sync_nonzero_exit(self, mock_retry):
        """_execute_sync passes through non-zero exit codes."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        exec_result = MagicMock()
        exec_result.output = (b"", b"not found")
        exec_result.exit_code = 127
        mock_retry.return_value = exec_result

        result = sandbox._execute_sync("nonexistent_cmd")

        assert result == ("", "not found", 127)


# ---------------------------------------------------------------------------
# _copy_to_container Tests
# ---------------------------------------------------------------------------


class TestDockerSandboxCopyToContainer:
    """Tests for DockerSandbox._copy_to_container()."""

    def test_copy_to_container_creates_tar(self, tmp_path):
        """_copy_to_container creates a tar archive and calls put_archive."""
        sandbox = DockerSandbox()
        container = MagicMock()
        sandbox.container = container

        # Create a temp file to copy
        source_file = tmp_path / "test_script.py"
        source_file.write_text("print('hello')")

        sandbox._copy_to_container(str(source_file), "/tmp/code.py")

        container.put_archive.assert_called_once()
        # First arg is the parent directory
        assert container.put_archive.call_args[0][0] == Path("/tmp")

    def test_copy_to_container_tar_contents(self, tmp_path):
        """The tar archive contains the file with correct name and content."""
        sandbox = DockerSandbox()
        container = MagicMock()
        sandbox.container = container

        source_file = tmp_path / "script.py"
        source_file.write_text("x = 42")

        sandbox._copy_to_container(str(source_file), "/workspace/code.py")

        # Extract the tar data passed to put_archive
        tar_data = container.put_archive.call_args[0][1]
        tar_stream = io.BytesIO(tar_data)
        tar = tarfile.open(fileobj=tar_stream, mode="r")

        members = tar.getmembers()
        assert len(members) == 1
        assert members[0].name == "code.py"
        assert members[0].mode == 0o755

        extracted = tar.extractfile(members[0])
        assert extracted.read() == b"x = 42"

    def test_copy_to_container_binary_data(self, tmp_path):
        """_copy_to_container handles binary file content correctly."""
        sandbox = DockerSandbox()
        container = MagicMock()
        sandbox.container = container

        source_file = tmp_path / "binary.bin"
        binary_data = bytes(range(256))
        source_file.write_bytes(binary_data)

        sandbox._copy_to_container(str(source_file), "/tmp/binary.bin")

        tar_data = container.put_archive.call_args[0][1]
        tar_stream = io.BytesIO(tar_data)
        tar = tarfile.open(fileobj=tar_stream, mode="r")

        extracted = tar.extractfile(tar.getmembers()[0])
        assert extracted.read() == binary_data

    def test_copy_to_container_nested_path(self, tmp_path):
        """_copy_to_container passes correct parent directory for nested paths."""
        sandbox = DockerSandbox()
        container = MagicMock()
        sandbox.container = container

        source_file = tmp_path / "hello.js"
        source_file.write_text("console.log('hi')")

        sandbox._copy_to_container(str(source_file), "/app/src/main.js")

        # Parent directory should be /app/src
        assert container.put_archive.call_args[0][0] == Path("/app/src")


# ---------------------------------------------------------------------------
# _get_file_extension Tests
# ---------------------------------------------------------------------------


class TestDockerSandboxGetFileExtension:
    """Tests for DockerSandbox._get_file_extension()."""

    @pytest.mark.parametrize(
        "language,expected",
        [
            ("python", "py"),
            ("javascript", "js"),
            ("typescript", "ts"),
            ("bash", "sh"),
            ("shell", "sh"),
            ("sh", "sh"),
        ],
    )
    def test_known_languages(self, sandbox, language, expected):
        """Known languages return correct extensions."""
        assert sandbox._get_file_extension(language) == expected

    @pytest.mark.parametrize(
        "language",
        ["Python", "PYTHON", "JavaScript", "BASH"],
    )
    def test_case_insensitive(self, sandbox, language):
        """Language lookup is case-insensitive."""
        result = sandbox._get_file_extension(language)
        assert result != "txt"  # Should match a known language

    @pytest.mark.parametrize(
        "language",
        ["ruby", "go", "rust", "java", "unknown"],
    )
    def test_unknown_languages_return_txt(self, sandbox, language):
        """Unknown languages fall back to 'txt'."""
        assert sandbox._get_file_extension(language) == "txt"


# ---------------------------------------------------------------------------
# _get_execution_command Tests
# ---------------------------------------------------------------------------


class TestDockerSandboxGetExecutionCommand:
    """Tests for DockerSandbox._get_execution_command()."""

    def test_python_command(self, sandbox):
        """Python uses sh -lc with python3/python fallback."""
        cmd = sandbox._get_execution_command("python", "/tmp/code.py")
        assert "python3" in cmd
        assert "python" in cmd
        assert "/tmp/code.py" in cmd
        assert "sh -lc" in cmd

    def test_javascript_command(self, sandbox):
        """JavaScript uses node."""
        cmd = sandbox._get_execution_command("javascript", "/tmp/code.js")
        assert cmd == "node /tmp/code.js"

    def test_bash_command(self, sandbox):
        """Bash uses bash."""
        cmd = sandbox._get_execution_command("bash", "/tmp/code.sh")
        assert cmd == "bash /tmp/code.sh"

    def test_shell_command(self, sandbox):
        """Shell uses sh."""
        cmd = sandbox._get_execution_command("shell", "/tmp/code.sh")
        assert cmd == "sh /tmp/code.sh"

    def test_sh_command(self, sandbox):
        """'sh' language uses sh."""
        cmd = sandbox._get_execution_command("sh", "/tmp/code.sh")
        assert cmd == "sh /tmp/code.sh"

    def test_unknown_language_uses_cat(self, sandbox):
        """Unknown language falls back to cat (displays file content)."""
        cmd = sandbox._get_execution_command("ruby", "/tmp/code.rb")
        assert cmd == "cat /tmp/code.rb"

    @pytest.mark.parametrize(
        "language",
        ["Python", "PYTHON", "JavaScript", "BASH"],
    )
    def test_case_insensitive_commands(self, sandbox, language):
        """Command lookup is case-insensitive."""
        cmd = sandbox._get_execution_command(language, "/tmp/code.x")
        assert cmd != "cat /tmp/code.x"


# ---------------------------------------------------------------------------
# Integration / Lifecycle Tests
# ---------------------------------------------------------------------------


class TestDockerSandboxLifecycle:
    """Integration tests for the full start→execute→stop lifecycle."""

    @patch("manus_agent.sandbox.docker_sandbox.safe_kill_remove_container")
    @patch("manus_agent.sandbox.docker_sandbox.wait_for_container_running")
    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    @patch("manus_agent.sandbox.docker_sandbox.get_docker_client")
    def test_full_lifecycle(self, mock_get_client, mock_retry, mock_wait, mock_kill):
        """Full start → execute → stop lifecycle works correctly."""
        client = MagicMock()
        mock_get_client.return_value = client

        container = MagicMock()
        exec_result = MagicMock()
        exec_result.output = (b"42\n", None)
        exec_result.exit_code = 0

        call_count = [0]

        def retry_side_effect(name, fn):
            call_count[0] += 1
            if "images" in name:
                return MagicMock()
            elif "containers" in name:
                return container
            elif "exec_run" in name:
                return exec_result
            return fn()

        mock_retry.side_effect = retry_side_effect

        sandbox = DockerSandbox()

        # Start
        asyncio.run(sandbox.start())
        assert sandbox.container is container

        # Stop
        asyncio.run(sandbox.stop())
        mock_kill.assert_called_once_with(container)
        assert sandbox.container is None

    @patch("manus_agent.sandbox.docker_sandbox.safe_kill_remove_container")
    def test_double_stop_is_safe(self, mock_kill):
        """Calling stop() twice does not raise."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()
        sandbox.client = MagicMock()

        asyncio.run(sandbox.stop())
        asyncio.run(sandbox.stop())  # Second call should be no-op

        mock_kill.assert_called_once()

    @patch.object(DockerSandbox, "_execute_sync")
    @patch.object(DockerSandbox, "_copy_to_container")
    def test_execute_code_with_nonzero_exit(self, mock_copy, mock_exec_sync):
        """execute_code propagates non-zero exit codes from the container."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        async def fake_exec(cmd, timeout):
            return ("", "SyntaxError: invalid syntax", 1)

        with patch.object(sandbox, "execute_command", side_effect=fake_exec):
            result = asyncio.run(sandbox.execute_code("def f("))

        assert result[2] == 1
        assert "SyntaxError" in result[1]

    def test_container_name_uses_uuid_hex(self):
        """Container name uses UUID hex for uniqueness."""
        sandbox = DockerSandbox()
        parts = sandbox.container_name.split("-")
        # Format: manus-sandbox-XXXXXXXX
        assert parts[0] == "manus"
        assert parts[1] == "sandbox"
        assert len(parts[2]) == 8


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestDockerSandboxEdgeCases:
    """Edge cases and error handling tests."""

    @patch.object(DockerSandbox, "execute_command")
    @patch.object(DockerSandbox, "_copy_to_container")
    def test_execute_code_with_empty_code(self, mock_copy, mock_exec):
        """execute_code handles empty code string."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        async def fake_exec(cmd, timeout):
            return ("", "", 0)

        mock_exec.side_effect = fake_exec

        result = asyncio.run(sandbox.execute_code("", language="python"))
        assert result == ("", "", 0)

    @patch.object(DockerSandbox, "execute_command")
    @patch.object(DockerSandbox, "_copy_to_container")
    def test_execute_code_with_unicode(self, mock_copy, mock_exec):
        """execute_code handles Unicode code correctly."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        async def fake_exec(cmd, timeout):
            return ("你好世界", "", 0)

        mock_exec.side_effect = fake_exec

        result = asyncio.run(sandbox.execute_code("print('你好世界')", language="python"))
        assert result[0] == "你好世界"

    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    def test_execute_sync_calls_docker_retry(self, mock_retry):
        """_execute_sync uses docker_retry for resilience."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()

        exec_result = MagicMock()
        exec_result.output = (b"ok", b"")
        exec_result.exit_code = 0
        mock_retry.return_value = exec_result

        sandbox._execute_sync("echo ok")

        mock_retry.assert_called_once()
        assert "exec_run" in mock_retry.call_args[0][0]

    @patch("manus_agent.sandbox.docker_sandbox.wait_for_container_running")
    @patch("manus_agent.sandbox.docker_sandbox.docker_retry")
    @patch("manus_agent.sandbox.docker_sandbox.get_docker_client")
    def test_start_sets_correct_labels(self, mock_get_client, mock_retry, mock_wait):
        """Container labels include component and sandbox_uid."""
        client = MagicMock()
        mock_get_client.return_value = client

        container = MagicMock()

        def retry_side_effect(name, fn):
            return fn()

        mock_retry.side_effect = retry_side_effect
        client.images.get.return_value = MagicMock()
        client.containers.run.return_value = container

        sandbox = DockerSandbox()
        sandbox._start_sync()

        kwargs = client.containers.run.call_args[1]
        assert kwargs["labels"]["manus_agent.component"] == "docker_sandbox"
        assert kwargs["labels"]["manus_agent.sandbox_uid"] == sandbox.container_name

    @patch("manus_agent.sandbox.docker_sandbox.safe_kill_remove_container")
    def test_stop_with_none_client(self, mock_kill):
        """_stop_sync handles client already being None."""
        sandbox = DockerSandbox()
        sandbox.container = MagicMock()
        sandbox.client = None

        sandbox._stop_sync()  # Should not raise

        mock_kill.assert_called_once()
        assert sandbox.container is None
