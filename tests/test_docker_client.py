"""Comprehensive test suite for manus_agent.utils.docker_client module.

Covers:
- DockerConnectionError exception
- _get_default_docker_hosts() platform-specific socket discovery
- _check_docker_context() Docker CLI context lookup
- _is_socket_accessible() socket path validation
- get_docker_client() multi-strategy connection
- _diagnose_docker_issue() error classification & remediation
- check_docker_available() convenience wrapper
- is_transient_docker_error() error classification
- docker_retry() retry logic with backoff, jitter, deadline
- wait_for_container_running() polling with reload
- wait_for_container_healthy() healthcheck polling
- safe_kill_remove_container() idempotent cleanup
- safe_remove_network() idempotent cleanup
- safe_remove_image() idempotent cleanup

All Docker interactions are fully mocked — no real Docker daemon needed.
"""

import platform
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import docker
import pytest
from docker.errors import APIError, DockerException, NotFound

from manus_agent.utils.docker_client import (
    DockerConnectionError,
    _check_docker_context,
    _diagnose_docker_issue,
    _get_default_docker_hosts,
    _is_socket_accessible,
    check_docker_available,
    docker_retry,
    get_docker_client,
    is_transient_docker_error,
    safe_kill_remove_container,
    safe_remove_image,
    safe_remove_network,
    wait_for_container_healthy,
    wait_for_container_running,
)


# =====================================================================
# DockerConnectionError
# =====================================================================


class TestDockerConnectionError:
    """Tests for DockerConnectionError exception class."""

    def test_attributes_stored(self):
        err = DockerConnectionError(
            message="Cannot connect",
            diagnosis="Socket not found",
            remediation="Start Docker",
        )
        assert err.message == "Cannot connect"
        assert err.diagnosis == "Socket not found"
        assert err.remediation == "Start Docker"

    def test_str_contains_all_parts(self):
        err = DockerConnectionError(
            message="Cannot connect",
            diagnosis="Socket not found",
            remediation="Start Docker",
        )
        s = str(err)
        assert "Cannot connect" in s
        assert "Diagnosis: Socket not found" in s
        assert "Remediation: Start Docker" in s

    def test_is_exception(self):
        err = DockerConnectionError(message="m", diagnosis="d", remediation="r")
        assert isinstance(err, Exception)


# =====================================================================
# _get_default_docker_hosts
# =====================================================================


class TestGetDefaultDockerHosts:
    """Tests for platform-specific Docker socket discovery."""

    @patch("manus_agent.utils.docker_client.platform")
    def test_darwin_includes_desktop_orbstack_colima(self, mock_platform):
        mock_platform.system.return_value = "Darwin"
        hosts = _get_default_docker_hosts()
        descriptions = [desc for _, desc in hosts]
        assert "Docker Desktop" in descriptions
        assert "OrbStack" in descriptions
        assert "Colima" in descriptions
        assert "Docker Desktop (legacy)" in descriptions
        # Linux socket always included at the end
        assert any("Standard" in desc for _, desc in hosts)

    @patch("manus_agent.utils.docker_client.platform")
    def test_linux_includes_standard_socket_only(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        hosts = _get_default_docker_hosts()
        descriptions = [desc for _, desc in hosts]
        assert "Docker Desktop" not in descriptions
        assert "OrbStack" not in descriptions
        assert any("Standard" in desc for _, desc in hosts)

    @patch("manus_agent.utils.docker_client.platform")
    def test_all_hosts_are_unix_sockets(self, mock_platform):
        mock_platform.system.return_value = "Darwin"
        hosts = _get_default_docker_hosts()
        for url, _ in hosts:
            assert url.startswith("unix://")


# =====================================================================
# _check_docker_context
# =====================================================================


class TestCheckDockerContext:
    """Tests for Docker CLI context lookup."""

    def test_returns_none_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = _check_docker_context()
        assert result is None

    def test_returns_none_when_no_current_context(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".docker"
        config_dir.mkdir()
        (config_dir / "config.json").write_text('{}')
        result = _check_docker_context()
        assert result is None

    def test_returns_endpoint_when_context_matches(self, tmp_path, monkeypatch):
        import json

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".docker"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({"currentContext": "my-ctx"}))

        ctx_dir = config_dir / "contexts" / "meta" / "abc123"
        ctx_dir.mkdir(parents=True)
        (ctx_dir / "meta.json").write_text(
            json.dumps(
                {
                    "Name": "my-ctx",
                    "Endpoints": {"docker": {"Host": "unix:///custom/docker.sock"}},
                }
            )
        )
        result = _check_docker_context()
        assert result is not None
        assert result[0] == "unix:///custom/docker.sock"
        assert "my-ctx" in result[1]

    def test_returns_none_when_context_name_mismatch(self, tmp_path, monkeypatch):
        import json

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".docker"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({"currentContext": "my-ctx"}))

        ctx_dir = config_dir / "contexts" / "meta" / "abc123"
        ctx_dir.mkdir(parents=True)
        (ctx_dir / "meta.json").write_text(json.dumps({"Name": "other-ctx"}))
        result = _check_docker_context()
        assert result is None

    def test_returns_none_on_exception(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".docker"
        config_dir.mkdir()
        (config_dir / "config.json").write_text("invalid-json{{{")
        result = _check_docker_context()
        assert result is None

    def test_returns_none_when_endpoint_missing(self, tmp_path, monkeypatch):
        import json

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        config_dir = tmp_path / ".docker"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(json.dumps({"currentContext": "my-ctx"}))

        ctx_dir = config_dir / "contexts" / "meta" / "abc123"
        ctx_dir.mkdir(parents=True)
        (ctx_dir / "meta.json").write_text(json.dumps({"Name": "my-ctx", "Endpoints": {}}))
        result = _check_docker_context()
        assert result is None


# =====================================================================
# _is_socket_accessible
# =====================================================================


class TestIsSocketAccessible:
    """Tests for socket path validation."""

    def test_non_unix_protocol_returns_false(self):
        assert _is_socket_accessible("tcp://localhost:2375") is False

    def test_nonexistent_path_returns_false(self):
        assert _is_socket_accessible("unix:///nonexistent/docker.sock") is False

    def test_regular_file_returns_false(self, tmp_path):
        regular_file = tmp_path / "not-a-socket"
        regular_file.write_text("data")
        assert _is_socket_accessible(f"unix://{regular_file}") is False

    @patch("manus_agent.utils.docker_client.Path")
    def test_accessible_socket_returns_true(self, mock_path_cls):
        mock_instance = MagicMock()
        mock_instance.exists.return_value = True
        mock_instance.is_socket.return_value = True
        mock_path_cls.return_value = mock_instance
        assert _is_socket_accessible("unix:///var/run/docker.sock") is True


# =====================================================================
# get_docker_client
# =====================================================================


class TestGetDockerClient:
    """Tests for multi-strategy Docker client connection."""

    @patch("manus_agent.utils.docker_client.docker.DockerClient")
    def test_env_docker_host_takes_priority(self, mock_docker_cls, monkeypatch):
        monkeypatch.setenv("DOCKER_HOST", "tcp://my-host:2375")
        mock_client = MagicMock()
        mock_docker_cls.return_value = mock_client
        result = get_docker_client()
        assert result == mock_client
        mock_docker_cls.assert_called_once_with(base_url="tcp://my-host:2375", timeout=10)
        mock_client.ping.assert_called_once()

    @patch("manus_agent.utils.docker_client._check_docker_context")
    @patch("manus_agent.utils.docker_client.docker.DockerClient")
    def test_context_used_when_env_fails(self, mock_docker_cls, mock_ctx, monkeypatch):
        monkeypatch.setenv("DOCKER_HOST", "tcp://bad:1234")
        # env host fails
        env_client = MagicMock()
        env_client.ping.side_effect = Exception("connection refused")
        # context client succeeds
        ctx_client = MagicMock()
        mock_docker_cls.side_effect = [env_client, ctx_client]
        mock_ctx.return_value = ("unix:///custom/sock", "Custom context")

        result = get_docker_client()
        assert result == ctx_client

    @patch("manus_agent.utils.docker_client._check_docker_context")
    @patch("manus_agent.utils.docker_client._get_default_docker_hosts")
    @patch("manus_agent.utils.docker_client._is_socket_accessible")
    @patch("manus_agent.utils.docker_client.docker.DockerClient")
    def test_falls_back_to_default_sockets(self, mock_docker_cls, mock_accessible, mock_hosts, mock_ctx, monkeypatch):
        monkeypatch.delenv("DOCKER_HOST", raising=False)
        mock_ctx.return_value = None
        mock_hosts.return_value = [("unix:///var/run/docker.sock", "Standard")]
        mock_accessible.return_value = True
        mock_client = MagicMock()
        mock_docker_cls.return_value = mock_client

        result = get_docker_client()
        assert result == mock_client

    @patch("manus_agent.utils.docker_client._check_docker_context")
    @patch("manus_agent.utils.docker_client._get_default_docker_hosts")
    @patch("manus_agent.utils.docker_client._is_socket_accessible")
    def test_raises_connection_error_when_all_fail(self, mock_accessible, mock_hosts, mock_ctx, monkeypatch):
        monkeypatch.delenv("DOCKER_HOST", raising=False)
        mock_ctx.return_value = None
        mock_hosts.return_value = [("unix:///var/run/docker.sock", "Standard")]
        mock_accessible.return_value = False

        with pytest.raises(DockerConnectionError) as exc_info:
            get_docker_client()
        assert "Failed to connect" in str(exc_info.value)

    @patch("manus_agent.utils.docker_client._check_docker_context")
    @patch("manus_agent.utils.docker_client._get_default_docker_hosts")
    @patch("manus_agent.utils.docker_client._is_socket_accessible")
    @patch("manus_agent.utils.docker_client.docker.DockerClient")
    def test_skips_inaccessible_sockets(self, mock_docker_cls, mock_accessible, mock_hosts, mock_ctx, monkeypatch):
        monkeypatch.delenv("DOCKER_HOST", raising=False)
        mock_ctx.return_value = None
        mock_hosts.return_value = [
            ("unix:///bad/sock", "Bad"),
            ("unix:///good/sock", "Good"),
        ]
        mock_accessible.side_effect = [False, True]
        mock_client = MagicMock()
        mock_docker_cls.return_value = mock_client

        result = get_docker_client()
        assert result == mock_client
        mock_docker_cls.assert_called_once_with(base_url="unix:///good/sock", timeout=10)

    @patch("manus_agent.utils.docker_client.docker.DockerClient")
    def test_custom_timeout(self, mock_docker_cls, monkeypatch):
        monkeypatch.setenv("DOCKER_HOST", "tcp://host:2375")
        mock_client = MagicMock()
        mock_docker_cls.return_value = mock_client
        get_docker_client(timeout=30)
        mock_docker_cls.assert_called_once_with(base_url="tcp://host:2375", timeout=30)


# =====================================================================
# _diagnose_docker_issue
# =====================================================================


class TestDiagnoseDockerIssue:
    """Tests for error classification and remediation advice."""

    @patch("manus_agent.utils.docker_client.platform")
    def test_no_sockets_darwin(self, mock_platform):
        mock_platform.system.return_value = "Darwin"
        diagnosis, remediation = _diagnose_docker_issue(["some error"])
        assert "Docker Desktop" in diagnosis or "OrbStack" in diagnosis or "Colima" in diagnosis
        assert "DOCKER_HOST" in remediation

    @patch("manus_agent.utils.docker_client.platform")
    def test_no_sockets_linux(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        diagnosis, remediation = _diagnose_docker_issue(["some error"])
        assert "/var/run/docker.sock" in diagnosis
        assert "systemctl" in remediation

    def test_permission_denied(self):
        diagnosis, remediation = _diagnose_docker_issue(
            ["unix:///var/run/docker.sock: Permission denied on socket"]
        )
        assert "Permission denied" in diagnosis
        assert "docker group" in remediation

    def test_connection_refused(self):
        diagnosis, remediation = _diagnose_docker_issue(
            ["unix:///var/run/docker.sock: Connection refused"]
        )
        assert "not accepting connections" in diagnosis
        assert "Restart" in remediation

    def test_generic_errors(self):
        diagnosis, remediation = _diagnose_docker_issue(
            ["unix:///var/run/docker.sock: random error with unix://"]
        )
        assert "Errors encountered" in diagnosis
        assert "docker --version" in remediation


# =====================================================================
# check_docker_available
# =====================================================================


class TestCheckDockerAvailable:
    """Tests for the convenience availability check."""

    @patch("manus_agent.utils.docker_client.get_docker_client")
    def test_returns_true_when_available(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        available, error = check_docker_available()
        assert available is True
        assert error is None
        mock_client.close.assert_called_once()

    @patch("manus_agent.utils.docker_client.get_docker_client")
    def test_returns_false_on_connection_error(self, mock_get_client):
        mock_get_client.side_effect = DockerConnectionError(
            message="fail", diagnosis="d", remediation="r"
        )
        available, error = check_docker_available()
        assert available is False
        assert "fail" in error

    @patch("manus_agent.utils.docker_client.get_docker_client")
    def test_returns_false_on_unexpected_error(self, mock_get_client):
        mock_get_client.side_effect = RuntimeError("unexpected")
        available, error = check_docker_available()
        assert available is False
        assert "Unexpected" in error


# =====================================================================
# is_transient_docker_error
# =====================================================================


class TestIsTransientDockerError:
    """Tests for error classification as transient/retryable."""

    def test_docker_exception_is_transient(self):
        assert is_transient_docker_error(DockerException("daemon error")) is True

    def test_api_error_is_transient(self):
        err = APIError("Server Error")
        assert is_transient_docker_error(err) is True

    def test_conflict_api_error_is_not_transient(self):
        # Mock APIError with conflict message
        err = APIError("409 Conflict: name already in use")
        assert is_transient_docker_error(err) is False

    def test_already_in_use_is_not_transient(self):
        err = APIError("container name /foo is already in use")
        assert is_transient_docker_error(err) is False

    def test_connection_aborted_is_transient(self):
        err = ConnectionError("Connection aborted")
        assert is_transient_docker_error(err) is True

    def test_connection_reset_is_transient(self):
        err = OSError("Connection reset by peer")
        assert is_transient_docker_error(err) is True

    def test_timeout_string_is_transient(self):
        err = Exception("read timed out")
        assert is_transient_docker_error(err) is True

    def test_eof_is_transient(self):
        err = Exception("unexpected eof during read")
        assert is_transient_docker_error(err) is True

    def test_too_many_requests_is_transient(self):
        err = Exception("too many requests from client")
        assert is_transient_docker_error(err) is True

    def test_regular_value_error_is_not_transient(self):
        assert is_transient_docker_error(ValueError("bad value")) is False

    def test_key_error_is_not_transient(self):
        assert is_transient_docker_error(KeyError("missing")) is False

    def test_broken_pipe_is_transient(self):
        assert is_transient_docker_error(BrokenPipeError("Broken pipe")) is True

    def test_connection_refused_is_transient(self):
        assert is_transient_docker_error(ConnectionRefusedError("Connection refused")) is True

    def test_service_unavailable_is_transient(self):
        assert is_transient_docker_error(Exception("service unavailable")) is True

    def test_server_error_is_transient(self):
        assert is_transient_docker_error(Exception("server error 500")) is True

    def test_temporarily_unavailable_is_transient(self):
        assert is_transient_docker_error(Exception("resource temporarily unavailable")) is True


# =====================================================================
# docker_retry
# =====================================================================


class TestDockerRetry:
    """Tests for the retry wrapper with backoff and deadline support."""

    def test_success_on_first_attempt(self):
        fn = Mock(return_value="ok")
        result = docker_retry("test_op", fn)
        assert result == "ok"
        assert fn.call_count == 1

    def test_retries_on_transient_error(self):
        fn = Mock(side_effect=[DockerException("fail"), "ok"])
        result = docker_retry("test_op", fn, attempts=3, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 2

    def test_raises_after_all_attempts_exhausted(self):
        fn = Mock(side_effect=DockerException("keep failing"))
        with pytest.raises(DockerException, match="keep failing"):
            docker_retry("test_op", fn, attempts=3, base_delay=0.01)
        assert fn.call_count == 3

    def test_raises_immediately_on_non_transient_error(self):
        fn = Mock(side_effect=ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            docker_retry("test_op", fn, attempts=5, base_delay=0.01)
        assert fn.call_count == 1

    def test_deadline_stops_retries(self):
        call_count = 0

        def failing_fn():
            nonlocal call_count
            call_count += 1
            raise DockerException("fail")

        # Set deadline 0.05s in the future so at least one attempt runs,
        # but retries are cut short.
        with pytest.raises(DockerException):
            docker_retry(
                "test_op", failing_fn,
                attempts=100, base_delay=0.01, deadline=time.time() + 0.05,
            )
        # Should have run at least once but far fewer than 100 attempts
        assert 1 <= call_count < 20

    def test_single_attempt_with_attempts_1(self):
        fn = Mock(side_effect=DockerException("fail"))
        with pytest.raises(DockerException):
            docker_retry("test_op", fn, attempts=1, base_delay=0.01)
        assert fn.call_count == 1

    def test_retries_up_to_max_attempts(self):
        fn = Mock(
            side_effect=[
                DockerException("1"),
                DockerException("2"),
                DockerException("3"),
                "ok",
            ]
        )
        result = docker_retry("test_op", fn, attempts=4, base_delay=0.01)
        assert result == "ok"
        assert fn.call_count == 4

    def test_conflict_api_error_not_retried(self):
        fn = Mock(side_effect=APIError("409 Conflict: name already in use"))
        with pytest.raises(APIError):
            docker_retry("test_op", fn, attempts=5, base_delay=0.01)
        assert fn.call_count == 1


# =====================================================================
# wait_for_container_running
# =====================================================================


class TestWaitForContainerRunning:
    """Tests for container running state polling."""

    def test_returns_when_running(self):
        container = MagicMock()
        container.attrs = {"State": {"Running": True}}
        container.reload = MagicMock()
        wait_for_container_running(container, timeout=5)

    def test_raises_runtime_error_on_exited(self):
        container = MagicMock()
        container.attrs = {"State": {"Running": False, "Status": "exited", "ExitCode": 1}}
        container.reload = MagicMock()
        with pytest.raises(RuntimeError, match="not running"):
            wait_for_container_running(container, timeout=2)

    def test_raises_runtime_error_on_dead(self):
        container = MagicMock()
        container.attrs = {"State": {"Running": False, "Status": "dead", "ExitCode": 137}}
        container.reload = MagicMock()
        with pytest.raises(RuntimeError, match="not running"):
            wait_for_container_running(container, timeout=2)

    def test_raises_timeout_error(self):
        container = MagicMock()
        container.attrs = {"State": {"Running": False, "Status": "created"}}
        container.reload = MagicMock()
        with pytest.raises(TimeoutError, match="Timed out"):
            wait_for_container_running(container, timeout=1)

    def test_polls_until_running(self):
        container = MagicMock()
        call_count = 0

        def reload_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                container.attrs = {"State": {"Running": True}}
            else:
                container.attrs = {"State": {"Running": False, "Status": "created"}}

        container.attrs = {"State": {"Running": False, "Status": "created"}}
        container.reload = reload_side_effect
        wait_for_container_running(container, timeout=10)
        assert call_count >= 3


# =====================================================================
# wait_for_container_healthy
# =====================================================================


class TestWaitForContainerHealthy:
    """Tests for healthcheck polling."""

    def test_returns_immediately_when_no_healthcheck(self):
        container = MagicMock()
        container.attrs = {"State": {}}
        container.reload = MagicMock()
        wait_for_container_healthy(container, timeout=5)

    def test_returns_immediately_when_health_not_dict(self):
        container = MagicMock()
        container.attrs = {"State": {"Health": None}}
        container.reload = MagicMock()
        wait_for_container_healthy(container, timeout=5)

    def test_returns_when_healthy(self):
        container = MagicMock()
        container.attrs = {"State": {"Health": {"Status": "healthy"}}}
        container.reload = MagicMock()
        wait_for_container_healthy(container, timeout=5)

    def test_raises_runtime_error_on_unhealthy(self):
        container = MagicMock()
        # First reload returns health dict (triggers the polling loop entry)
        # Second reload returns unhealthy
        container.attrs = {"State": {"Health": {"Status": "unhealthy"}}}
        container.reload = MagicMock()
        with pytest.raises(RuntimeError, match="unhealthy"):
            wait_for_container_healthy(container, timeout=5)

    def test_raises_timeout_on_starting(self):
        container = MagicMock()
        container.attrs = {"State": {"Health": {"Status": "starting"}}}
        container.reload = MagicMock()
        with pytest.raises(TimeoutError, match="Timed out"):
            wait_for_container_healthy(container, timeout=1)

    def test_polls_until_healthy(self):
        container = MagicMock()
        call_count = 0

        def reload_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                container.attrs = {"State": {"Health": {"Status": "healthy"}}}
            else:
                container.attrs = {"State": {"Health": {"Status": "starting"}}}

        container.attrs = {"State": {"Health": {"Status": "starting"}}}
        container.reload = reload_side_effect
        wait_for_container_healthy(container, timeout=10)
        assert call_count >= 3


# =====================================================================
# safe_kill_remove_container
# =====================================================================


class TestSafeKillRemoveContainer:
    """Tests for idempotent container cleanup."""

    def test_none_container_is_noop(self):
        safe_kill_remove_container(None)

    def test_kills_and_removes(self):
        container = MagicMock()
        safe_kill_remove_container(container)
        container.kill.assert_called_once()
        container.remove.assert_called_once_with(force=True)

    def test_handles_not_found_on_kill(self):
        container = MagicMock()
        container.kill.side_effect = NotFound("gone")
        safe_kill_remove_container(container)
        container.remove.assert_not_called()

    def test_handles_not_found_on_remove(self):
        container = MagicMock()
        container.remove.side_effect = NotFound("gone")
        safe_kill_remove_container(container)

    def test_handles_other_exception_on_kill(self):
        container = MagicMock()
        container.kill.side_effect = RuntimeError("oops")
        safe_kill_remove_container(container)
        container.remove.assert_called_once()

    def test_handles_other_exception_on_remove(self):
        container = MagicMock()
        container.remove.side_effect = RuntimeError("oops")
        safe_kill_remove_container(container)

    def test_force_false(self):
        container = MagicMock()
        safe_kill_remove_container(container, force=False)
        container.remove.assert_called_once_with(force=False)


# =====================================================================
# safe_remove_network
# =====================================================================


class TestSafeRemoveNetwork:
    """Tests for idempotent network cleanup."""

    def test_none_network_is_noop(self):
        safe_remove_network(None)

    def test_removes_network(self):
        network = MagicMock()
        safe_remove_network(network)
        network.remove.assert_called_once()

    def test_handles_not_found(self):
        network = MagicMock()
        network.remove.side_effect = NotFound("gone")
        safe_remove_network(network)

    def test_handles_other_exception(self):
        network = MagicMock()
        network.remove.side_effect = RuntimeError("oops")
        safe_remove_network(network)


# =====================================================================
# safe_remove_image
# =====================================================================


class TestSafeRemoveImage:
    """Tests for idempotent image cleanup."""

    def test_none_client_is_noop(self):
        safe_remove_image(None, "sha256:abc")

    def test_none_image_id_is_noop(self):
        client = MagicMock()
        safe_remove_image(client, None)
        client.images.remove.assert_not_called()

    def test_empty_image_id_is_noop(self):
        client = MagicMock()
        safe_remove_image(client, "")
        client.images.remove.assert_not_called()

    def test_removes_image(self):
        client = MagicMock()
        safe_remove_image(client, "sha256:abc123")
        client.images.remove.assert_called_once_with("sha256:abc123", force=True)

    def test_handles_not_found(self):
        client = MagicMock()
        client.images.remove.side_effect = NotFound("gone")
        safe_remove_image(client, "sha256:abc123")

    def test_handles_other_exception(self):
        client = MagicMock()
        client.images.remove.side_effect = RuntimeError("oops")
        safe_remove_image(client, "sha256:abc123")
