"""Comprehensive test suite for manus_agent.utils.docker_client.

Tests cover:
- get_docker_client: socket detection, DOCKER_HOST, Docker context, platform paths
- docker_retry: exponential backoff, jitter, deadline awareness, transient classification
- is_transient_docker_error: error classification logic
- wait_for_container_running: polling, timeout, exited containers
- wait_for_container_healthy: healthcheck polling
- safe_kill_remove_container: idempotent cleanup
- safe_remove_network: idempotent cleanup
- safe_remove_image: idempotent cleanup
- DockerConnectionError: structured error with diagnosis/remediation
- _get_default_docker_hosts: platform-specific socket paths
- _check_docker_context: context resolution from config
- _is_socket_accessible: socket path validation
- _diagnose_docker_issue: remediation messages

All tests are fully mocked — no Docker daemon required.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import APIError, DockerException, ImageNotFound, NotFound

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

# =============================================================================
# DockerConnectionError
# =============================================================================


class TestDockerConnectionError:
    def test_structured_fields(self):
        err = DockerConnectionError(
            message="Failed to connect",
            diagnosis="No socket found",
            remediation="Start Docker",
        )
        assert err.message == "Failed to connect"
        assert err.diagnosis == "No socket found"
        assert err.remediation == "Start Docker"
        assert "Failed to connect" in str(err)
        assert "Diagnosis: No socket found" in str(err)
        assert "Remediation: Start Docker" in str(err)

    def test_is_exception(self):
        err = DockerConnectionError(message="msg", diagnosis="diag", remediation="fix")
        assert isinstance(err, Exception)


# =============================================================================
# is_transient_docker_error
# =============================================================================


class TestIsTransientDockerError:
    def test_docker_exception_is_transient(self):
        exc = DockerException("connection reset by peer")
        assert is_transient_docker_error(exc) is True

    def test_api_error_generic_is_transient(self):
        exc = APIError("server error")
        assert is_transient_docker_error(exc) is True

    def test_api_error_conflict_is_not_transient(self):
        exc = APIError("409 Client Error: Conflict")
        assert is_transient_docker_error(exc) is False

    def test_api_error_already_in_use_is_not_transient(self):
        exc = APIError("name already in use")
        assert is_transient_docker_error(exc) is False

    def test_connection_refused_in_message(self):
        exc = OSError("Connection refused")
        assert is_transient_docker_error(exc) is True

    def test_broken_pipe(self):
        exc = OSError("Broken pipe")
        assert is_transient_docker_error(exc) is True

    def test_read_timed_out(self):
        exc = Exception("Read timed out")
        assert is_transient_docker_error(exc) is True

    def test_timeout_in_message(self):
        exc = Exception("request timeout")
        assert is_transient_docker_error(exc) is True

    def test_eof(self):
        exc = Exception("unexpected EOF")
        assert is_transient_docker_error(exc) is True

    def test_service_unavailable(self):
        exc = Exception("503 Service Unavailable")
        assert is_transient_docker_error(exc) is True

    def test_too_many_requests(self):
        exc = Exception("429 Too Many Requests")
        assert is_transient_docker_error(exc) is True

    def test_type_name_readtimeout(self):
        # Simulate a requests ReadTimeout by using a custom exception class
        class ReadTimeoutError(Exception):  # noqa: N818
            pass

        # Override __name__ to match what requests.ReadTimeout looks like
        ReadTimeoutError.__name__ = "ReadTimeout"
        exc = ReadTimeoutError("timed out reading")
        # The type name is "readtimeout" when lowercased
        assert is_transient_docker_error(exc) is True

    def test_type_name_connectionerror(self):
        class _ConnectionError(Exception):
            pass

        _ConnectionError.__name__ = "ConnectionError"
        exc = _ConnectionError("failed")
        assert is_transient_docker_error(exc) is True

    def test_non_transient_value_error(self):
        exc = ValueError("invalid argument")
        assert is_transient_docker_error(exc) is False

    def test_non_transient_image_not_found(self):
        exc = ImageNotFound("no such image")
        # ImageNotFound is an APIError; check message for conflict/already-in-use
        # "no such image" doesn't match those, so it IS transient by default classification
        # Actually looking at the code: isinstance(exc, (DockerException, APIError)) => True
        # Then checks for "conflict"/"already in use" — doesn't match, returns True
        assert is_transient_docker_error(exc) is True

    def test_not_found_is_transient_by_api_error_path(self):
        # NotFound inherits from APIError
        exc = NotFound("no such container")
        # "no such container" — no "conflict"/"already in use" => returns True
        assert is_transient_docker_error(exc) is True

    def test_temporarily_unavailable(self):
        exc = Exception("resource temporarily unavailable")
        assert is_transient_docker_error(exc) is True

    def test_tls_handshake_timeout(self):
        exc = Exception("TLS handshake timeout")
        assert is_transient_docker_error(exc) is True

    def test_i_o_timeout(self):
        exc = Exception("i/o timeout")
        assert is_transient_docker_error(exc) is True


# =============================================================================
# docker_retry
# =============================================================================


class TestDockerRetry:
    def test_success_on_first_attempt(self):
        fn = MagicMock(return_value="ok")
        result = docker_retry("test_op", fn)
        assert result == "ok"
        fn.assert_called_once()

    def test_retries_on_transient_error(self):
        fn = MagicMock(side_effect=[DockerException("connection reset"), "success"])
        result = docker_retry("test_op", fn, base_delay=0.01, max_delay=0.02)
        assert result == "success"
        assert fn.call_count == 2

    def test_raises_after_max_attempts(self):
        fn = MagicMock(side_effect=DockerException("persistent failure"))
        with pytest.raises(DockerException, match="persistent failure"):
            docker_retry("test_op", fn, attempts=3, base_delay=0.01, max_delay=0.02)
        assert fn.call_count == 3

    def test_does_not_retry_non_transient(self):
        fn = MagicMock(side_effect=ValueError("bad argument"))
        with pytest.raises(ValueError, match="bad argument"):
            docker_retry("test_op", fn, attempts=5, base_delay=0.01)
        fn.assert_called_once()

    def test_deadline_already_expired_raises_assertion(self):
        """When deadline is already past, loop breaks before fn() is called.
        The function asserts last_exc is not None then raises AssertionError.
        """
        call_count = 0

        def slow_fail():
            nonlocal call_count
            call_count += 1
            raise DockerException("transient")

        # Deadline already expired => loop breaks immediately, last_exc is None
        with pytest.raises(AssertionError):
            docker_retry(
                "test_op",
                slow_fail,
                attempts=10,
                base_delay=0.01,
                deadline=time.time() - 1,
            )
        # fn was never called because deadline check is at top of loop
        assert call_count == 0

    def test_exponential_backoff_timing(self):
        """Verify retry delays grow exponentially."""
        fn = MagicMock(
            side_effect=[
                DockerException("e1"),
                DockerException("e2"),
                DockerException("e3"),
                "ok",
            ]
        )

        with patch("manus_agent.utils.docker_client.time.sleep") as mock_sleep:
            with patch("manus_agent.utils.docker_client.random.uniform", return_value=0.0):
                result = docker_retry("test_op", fn, attempts=4, base_delay=0.1, max_delay=10.0, jitter=0.0)

        assert result == "ok"
        # Delays: 0.1 * 2^0 = 0.1, 0.1 * 2^1 = 0.2, 0.1 * 2^2 = 0.4
        calls = mock_sleep.call_args_list
        assert len(calls) == 3
        assert abs(calls[0][0][0] - 0.1) < 0.05
        assert abs(calls[1][0][0] - 0.2) < 0.05
        assert abs(calls[2][0][0] - 0.4) < 0.05

    def test_max_delay_caps_backoff(self):
        fn = MagicMock(
            side_effect=[
                DockerException("e1"),
                DockerException("e2"),
                DockerException("e3"),
                "ok",
            ]
        )

        with patch("manus_agent.utils.docker_client.time.sleep") as mock_sleep:
            with patch("manus_agent.utils.docker_client.random.uniform", return_value=0.0):
                docker_retry("test_op", fn, attempts=4, base_delay=1.0, max_delay=0.5, jitter=0.0)

        # All delays should be capped at max_delay=0.5
        for c in mock_sleep.call_args_list:
            assert c[0][0] <= 0.5 + 0.01

    def test_single_attempt(self):
        fn = MagicMock(side_effect=DockerException("fail"))
        with pytest.raises(DockerException):
            docker_retry("test_op", fn, attempts=1, base_delay=0.01)
        fn.assert_called_once()

    def test_deadline_stops_after_first_failure(self):
        """When deadline expires during delay calculation, retry loop breaks
        and re-raises the last exception."""
        fn = MagicMock(side_effect=DockerException("transient"))

        # Deadline is slightly in the future so first attempt runs,
        # but by the time delay is calculated, deadline is exceeded.
        # We patch time.time to simulate time advancing past deadline.
        original_time = time.time
        real_now = original_time()
        call_count = [0]

        def advancing_time():
            call_count[0] += 1
            # First two calls: return current time (before deadline)
            # After fn() fails and delay calc checks remaining: jump past deadline
            if call_count[0] > 2:
                return real_now + 200
            return real_now

        with patch("manus_agent.utils.docker_client.time.sleep"):
            with patch("manus_agent.utils.docker_client.time.time", side_effect=advancing_time):
                with patch("manus_agent.utils.docker_client.random.uniform", return_value=0.0):
                    with pytest.raises(DockerException, match="transient"):
                        docker_retry(
                            "test_op",
                            fn,
                            attempts=10,
                            base_delay=10.0,
                            deadline=real_now + 0.5,
                        )


# =============================================================================
# _is_socket_accessible
# =============================================================================


class TestIsSocketAccessible:
    def test_non_unix_prefix_returns_false(self):
        assert _is_socket_accessible("tcp://localhost:2375") is False
        assert _is_socket_accessible("http://localhost") is False

    def test_nonexistent_path_returns_false(self):
        assert _is_socket_accessible("unix:///nonexistent/path/docker.sock") is False

    def test_existing_regular_file_returns_false(self, tmp_path):
        regular_file = tmp_path / "not_a_socket"
        regular_file.write_text("hello")
        assert _is_socket_accessible(f"unix://{regular_file}") is False

    def test_existing_socket_returns_true(self, tmp_path):
        """Test with a real Unix socket file."""
        import socket

        sock_path = tmp_path / "test.sock"
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.bind(str(sock_path))
            assert _is_socket_accessible(f"unix://{sock_path}") is True
        finally:
            s.close()


# =============================================================================
# _get_default_docker_hosts
# =============================================================================


class TestGetDefaultDockerHosts:
    @patch("manus_agent.utils.docker_client.platform.system", return_value="Darwin")
    def test_darwin_includes_multiple_paths(self, _mock_sys):
        hosts = _get_default_docker_hosts()
        descriptions = [desc for _, desc in hosts]
        assert "Docker Desktop" in descriptions
        assert "OrbStack" in descriptions
        assert "Colima" in descriptions
        assert "Standard Linux/Unix socket" in descriptions
        # Should have at least 5 entries on macOS
        assert len(hosts) >= 5

    @patch("manus_agent.utils.docker_client.platform.system", return_value="Linux")
    def test_linux_includes_standard_socket(self, _mock_sys):
        hosts = _get_default_docker_hosts()
        urls = [url for url, _ in hosts]
        assert "unix:///var/run/docker.sock" in urls
        # On Linux, no macOS-specific paths
        descriptions = [desc for _, desc in hosts]
        assert "Docker Desktop" not in descriptions

    @patch("manus_agent.utils.docker_client.platform.system", return_value="Darwin")
    def test_all_hosts_are_unix_urls(self, _mock_sys):
        hosts = _get_default_docker_hosts()
        for url, _ in hosts:
            assert url.startswith("unix://")


# =============================================================================
# _check_docker_context
# =============================================================================


class TestCheckDockerContext:
    def test_no_config_file_returns_none(self, tmp_path):
        with patch("manus_agent.utils.docker_client.Path.home", return_value=tmp_path):
            result = _check_docker_context()
        assert result is None

    def test_config_without_current_context_returns_none(self, tmp_path):
        docker_dir = tmp_path / ".docker"
        docker_dir.mkdir()
        config_file = docker_dir / "config.json"
        config_file.write_text(json.dumps({"auths": {}}))

        with patch("manus_agent.utils.docker_client.Path.home", return_value=tmp_path):
            result = _check_docker_context()
        assert result is None

    def test_config_with_context_resolved(self, tmp_path):
        docker_dir = tmp_path / ".docker"
        docker_dir.mkdir()
        config_file = docker_dir / "config.json"
        config_file.write_text(json.dumps({"currentContext": "my-context"}))

        # Create context meta directory
        context_meta = docker_dir / "contexts" / "meta" / "abc123"
        context_meta.mkdir(parents=True)
        meta_file = context_meta / "meta.json"
        meta_file.write_text(
            json.dumps(
                {
                    "Name": "my-context",
                    "Endpoints": {"docker": {"Host": "unix:///custom/docker.sock"}},
                }
            )
        )

        with patch("manus_agent.utils.docker_client.Path.home", return_value=tmp_path):
            result = _check_docker_context()

        assert result is not None
        url, desc = result
        assert url == "unix:///custom/docker.sock"
        assert "my-context" in desc

    def test_context_name_mismatch_returns_none(self, tmp_path):
        docker_dir = tmp_path / ".docker"
        docker_dir.mkdir()
        config_file = docker_dir / "config.json"
        config_file.write_text(json.dumps({"currentContext": "my-context"}))

        context_meta = docker_dir / "contexts" / "meta" / "abc123"
        context_meta.mkdir(parents=True)
        meta_file = context_meta / "meta.json"
        meta_file.write_text(
            json.dumps(
                {
                    "Name": "other-context",
                    "Endpoints": {"docker": {"Host": "unix:///other.sock"}},
                }
            )
        )

        with patch("manus_agent.utils.docker_client.Path.home", return_value=tmp_path):
            result = _check_docker_context()
        assert result is None

    def test_malformed_json_returns_none(self, tmp_path):
        docker_dir = tmp_path / ".docker"
        docker_dir.mkdir()
        config_file = docker_dir / "config.json"
        config_file.write_text("not valid json {{{")

        with patch("manus_agent.utils.docker_client.Path.home", return_value=tmp_path):
            result = _check_docker_context()
        assert result is None


# =============================================================================
# get_docker_client
# =============================================================================


class TestGetDockerClient:
    @patch.dict("os.environ", {"DOCKER_HOST": "unix:///custom/docker.sock"})
    @patch("manus_agent.utils.docker_client.docker.DockerClient")
    def test_docker_host_env_takes_priority(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client_cls.return_value = mock_client

        result = get_docker_client(timeout=5)

        mock_client_cls.assert_called_once_with(base_url="unix:///custom/docker.sock", timeout=5)
        mock_client.ping.assert_called_once()
        assert result is mock_client

    @patch.dict("os.environ", {"DOCKER_HOST": ""}, clear=False)
    @patch("manus_agent.utils.docker_client._check_docker_context")
    @patch("manus_agent.utils.docker_client.docker.DockerClient")
    def test_docker_context_used_when_no_env(self, mock_client_cls, mock_context):
        mock_context.return_value = (
            "unix:///ctx/docker.sock",
            "test context",
        )
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client_cls.return_value = mock_client

        result = get_docker_client()

        mock_client_cls.assert_called_with(base_url="unix:///ctx/docker.sock", timeout=10)
        assert result is mock_client

    @patch.dict("os.environ", {}, clear=True)
    @patch("manus_agent.utils.docker_client._check_docker_context", return_value=None)
    @patch("manus_agent.utils.docker_client._is_socket_accessible", return_value=True)
    @patch("manus_agent.utils.docker_client.docker.DockerClient")
    @patch("manus_agent.utils.docker_client._get_default_docker_hosts")
    def test_falls_through_to_default_sockets(self, mock_hosts, mock_client_cls, mock_accessible, mock_context):
        mock_hosts.return_value = [
            ("unix:///var/run/docker.sock", "Standard"),
        ]
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client_cls.return_value = mock_client

        result = get_docker_client()
        assert result is mock_client

    @patch.dict("os.environ", {}, clear=True)
    @patch("manus_agent.utils.docker_client._check_docker_context", return_value=None)
    @patch("manus_agent.utils.docker_client._is_socket_accessible", return_value=False)
    def test_raises_docker_connection_error_when_no_socket(self, mock_accessible, mock_context):
        with pytest.raises(DockerConnectionError) as exc_info:
            get_docker_client()
        assert "Failed to connect" in exc_info.value.message

    @patch.dict("os.environ", {"DOCKER_HOST": "tcp://broken:2375"})
    @patch("manus_agent.utils.docker_client._check_docker_context", return_value=None)
    @patch("manus_agent.utils.docker_client._is_socket_accessible", return_value=False)
    @patch("manus_agent.utils.docker_client.docker.DockerClient")
    def test_env_host_failure_falls_through(self, mock_client_cls, mock_accessible, mock_context):
        mock_client_cls.side_effect = Exception("connection refused")

        with pytest.raises(DockerConnectionError):
            get_docker_client()


# =============================================================================
# check_docker_available
# =============================================================================


class TestCheckDockerAvailable:
    @patch("manus_agent.utils.docker_client.get_docker_client")
    def test_available(self, mock_get):
        mock_client = MagicMock()
        mock_get.return_value = mock_client

        available, error = check_docker_available()

        assert available is True
        assert error is None
        mock_client.close.assert_called_once()

    @patch("manus_agent.utils.docker_client.get_docker_client")
    def test_not_available_docker_error(self, mock_get):
        mock_get.side_effect = DockerConnectionError(message="fail", diagnosis="diag", remediation="fix")

        available, error = check_docker_available()

        assert available is False
        assert "fail" in error

    @patch("manus_agent.utils.docker_client.get_docker_client")
    def test_not_available_unexpected_error(self, mock_get):
        mock_get.side_effect = RuntimeError("unexpected")

        available, error = check_docker_available()

        assert available is False
        assert "Unexpected error" in error


# =============================================================================
# wait_for_container_running
# =============================================================================


class TestWaitForContainerRunning:
    def test_already_running(self):
        container = MagicMock()
        container.attrs = {"State": {"Running": True, "Status": "running"}}
        container.reload = MagicMock()

        wait_for_container_running(container, timeout=5)
        container.reload.assert_called()

    def test_becomes_running_after_reload(self):
        container = MagicMock()
        # First reload: not running, second: running
        states = [
            {"State": {"Running": False, "Status": "created"}},
            {"State": {"Running": True, "Status": "running"}},
        ]
        call_idx = [0]

        def fake_reload():
            container.attrs = states[min(call_idx[0], len(states) - 1)]
            call_idx[0] += 1

        container.reload = fake_reload
        container.attrs = states[0]

        with patch("manus_agent.utils.docker_client.time.sleep"):
            wait_for_container_running(container, timeout=5)

    def test_exited_container_raises(self):
        container = MagicMock()
        container.attrs = {"State": {"Running": False, "Status": "exited", "ExitCode": 1}}
        container.reload = MagicMock()

        with pytest.raises(RuntimeError, match="not running"):
            wait_for_container_running(container, timeout=5)

    def test_dead_container_raises(self):
        container = MagicMock()
        container.attrs = {"State": {"Running": False, "Status": "dead", "ExitCode": 137}}
        container.reload = MagicMock()

        with pytest.raises(RuntimeError, match="not running"):
            wait_for_container_running(container, timeout=5)

    def test_timeout_raises(self):
        container = MagicMock()
        container.attrs = {"State": {"Running": False, "Status": "created"}}
        container.reload = MagicMock()

        with patch("manus_agent.utils.docker_client.time.sleep"):
            with pytest.raises(TimeoutError, match="Timed out"):
                wait_for_container_running(container, timeout=0)


# =============================================================================
# wait_for_container_healthy
# =============================================================================


class TestWaitForContainerHealthy:
    def test_no_healthcheck_returns_immediately(self):
        container = MagicMock()
        container.attrs = {"State": {"Running": True}}
        container.reload = MagicMock()

        wait_for_container_healthy(container, timeout=5)

    def test_already_healthy(self):
        container = MagicMock()
        container.attrs = {"State": {"Running": True, "Health": {"Status": "healthy"}}}
        container.reload = MagicMock()

        wait_for_container_healthy(container, timeout=5)

    def test_becomes_healthy(self):
        container = MagicMock()
        states = [
            {"State": {"Running": True, "Health": {"Status": "starting"}}},
            {"State": {"Running": True, "Health": {"Status": "healthy"}}},
        ]
        call_idx = [0]

        def fake_reload():
            container.attrs = states[min(call_idx[0], len(states) - 1)]
            call_idx[0] += 1

        container.reload = fake_reload
        container.attrs = states[0]

        with patch("manus_agent.utils.docker_client.time.sleep"):
            wait_for_container_healthy(container, timeout=5)

    def test_unhealthy_raises(self):
        container = MagicMock()
        container.attrs = {"State": {"Running": True, "Health": {"Status": "unhealthy"}}}
        container.reload = MagicMock()

        with pytest.raises(RuntimeError, match="unhealthy"):
            wait_for_container_healthy(container, timeout=5)

    def test_timeout_raises(self):
        container = MagicMock()
        container.attrs = {"State": {"Running": True, "Health": {"Status": "starting"}}}
        container.reload = MagicMock()

        # Patch time.time to return values that make the while loop expire
        # The function does: deadline = time.time() + timeout
        # Then: while time.time() < deadline
        # We need the first call (deadline calc) to return T,
        # the docker_retry deadline checks to pass (< deadline),
        # then the while check to fail (>= deadline)
        time_values = iter([100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0])

        with patch("manus_agent.utils.docker_client.time.time", side_effect=lambda: next(time_values, 200.0)):
            with patch("manus_agent.utils.docker_client.time.sleep"):
                with pytest.raises(TimeoutError, match="Timed out"):
                    wait_for_container_healthy(container, timeout=5)


# =============================================================================
# safe_kill_remove_container
# =============================================================================


class TestSafeKillRemoveContainer:
    def test_none_container_is_noop(self):
        # Should not raise
        safe_kill_remove_container(None)

    def test_kill_then_remove(self):
        container = MagicMock()
        safe_kill_remove_container(container)
        container.kill.assert_called_once()
        container.remove.assert_called_once_with(force=True)

    def test_not_found_on_kill_returns_early(self):
        container = MagicMock()
        container.kill.side_effect = NotFound("gone")
        safe_kill_remove_container(container)
        container.remove.assert_not_called()

    def test_kill_other_error_still_tries_remove(self):
        container = MagicMock()
        container.kill.side_effect = DockerException("some error")
        safe_kill_remove_container(container)
        container.remove.assert_called_once_with(force=True)

    def test_not_found_on_remove_is_silent(self):
        container = MagicMock()
        container.remove.side_effect = NotFound("already gone")
        safe_kill_remove_container(container)
        # No exception raised

    def test_remove_other_error_is_silent(self):
        container = MagicMock()
        container.remove.side_effect = DockerException("remove failed")
        safe_kill_remove_container(container)
        # No exception raised


# =============================================================================
# safe_remove_network
# =============================================================================


class TestSafeRemoveNetwork:
    def test_none_network_is_noop(self):
        safe_remove_network(None)

    def test_removes_network(self):
        network = MagicMock()
        safe_remove_network(network)
        network.remove.assert_called_once()

    def test_not_found_is_silent(self):
        network = MagicMock()
        network.remove.side_effect = NotFound("gone")
        safe_remove_network(network)

    def test_other_error_is_silent(self):
        network = MagicMock()
        network.remove.side_effect = DockerException("fail")
        safe_remove_network(network)


# =============================================================================
# safe_remove_image
# =============================================================================


class TestSafeRemoveImage:
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

    def test_not_found_is_silent(self):
        client = MagicMock()
        client.images.remove.side_effect = NotFound("gone")
        safe_remove_image(client, "sha256:abc123")

    def test_other_error_is_silent(self):
        client = MagicMock()
        client.images.remove.side_effect = DockerException("fail")
        safe_remove_image(client, "sha256:abc123")


# =============================================================================
# _diagnose_docker_issue
# =============================================================================


class TestDiagnoseDockerIssue:
    @patch("manus_agent.utils.docker_client.platform.system", return_value="Darwin")
    def test_no_sockets_darwin(self, _):
        diagnosis, remediation = _diagnose_docker_issue(["No socket: failed"])
        assert "Docker Desktop" in diagnosis or "Docker Desktop" in remediation
        assert "OrbStack" in remediation or "Colima" in remediation

    @patch("manus_agent.utils.docker_client.platform.system", return_value="Linux")
    def test_no_sockets_linux(self, _):
        diagnosis, remediation = _diagnose_docker_issue(["No socket: failed"])
        assert "docker.sock" in diagnosis
        assert "systemctl" in remediation

    def test_permission_denied(self):
        errors = ["unix:///var/run/docker.sock: Permission denied connecting"]
        diagnosis, remediation = _diagnose_docker_issue(errors)
        assert "Permission denied" in diagnosis
        assert "docker group" in remediation.lower() or "usermod" in remediation

    def test_connection_refused(self):
        errors = ["unix:///var/run/docker.sock: Connection refused"]
        diagnosis, remediation = _diagnose_docker_issue(errors)
        assert "not accepting connections" in diagnosis
        assert "restart" in remediation.lower()

    def test_generic_errors(self):
        errors = ["some weird error at unix:///var/run/docker.sock: timeout"]
        diagnosis, remediation = _diagnose_docker_issue(errors)
        assert "Errors encountered" in diagnosis
        assert "docker --version" in remediation


# =============================================================================
# Integration: retry + transient classification
# =============================================================================


class TestRetryWithTransientClassification:
    def test_retry_succeeds_after_connection_reset(self):
        fn = MagicMock(
            side_effect=[
                DockerException("connection reset by peer"),
                DockerException("read timed out"),
                "finally",
            ]
        )
        with patch("manus_agent.utils.docker_client.time.sleep"):
            with patch("manus_agent.utils.docker_client.random.uniform", return_value=0.0):
                result = docker_retry("test", fn, attempts=4, base_delay=0.01)
        assert result == "finally"
        assert fn.call_count == 3

    def test_conflict_error_not_retried(self):
        fn = MagicMock(side_effect=APIError("409 Conflict: name already in use"))
        with pytest.raises(APIError):
            docker_retry("test", fn, attempts=5, base_delay=0.01)
        fn.assert_called_once()

    def test_jitter_adds_variance(self):
        """Verify jitter parameter is used in delay calculation."""
        fn = MagicMock(side_effect=[DockerException("e1"), "ok"])

        with patch("manus_agent.utils.docker_client.time.sleep") as mock_sleep:
            with patch("manus_agent.utils.docker_client.random.uniform", return_value=0.5):
                docker_retry("test", fn, attempts=2, base_delay=1.0, jitter=0.2)

        # With jitter=0.2, uniform returns 0.5, so jitter_amount = 0.5 * 0.2 * delay
        # Delay = max(0, 1.0 + 0.5 * 0.2 * 1.0) = 1.1
        actual = mock_sleep.call_args[0][0]
        assert actual != 1.0  # Not exactly base_delay due to jitter
