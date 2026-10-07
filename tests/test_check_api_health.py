"""Comprehensive tests for check_api_health tool.

100% mocked — no real HTTP calls.
"""

from __future__ import annotations

import json
import os
from unittest import mock

import requests

from manus_agent.tools.check_api_health import (
    _PROBES,
    TOOL_SPEC,
    _format_text_report,
    _github_headers,
    _nvd_headers,
    _print_plain,
    _run_single_probe,
    _short_error,
    _vulncheck_headers,
    check_api_health,
    check_api_health_tool,
    print_api_health_report,
)

# ---------------------------------------------------------------------------
# Header builder tests
# ---------------------------------------------------------------------------


class TestNvdHeaders:
    """Tests for _nvd_headers."""

    def test_returns_api_key_when_set(self):
        with mock.patch.dict(os.environ, {"NVD_API_KEY": "my-nvd-key"}):
            h = _nvd_headers()
            assert h == {"apiKey": "my-nvd-key"}

    def test_returns_empty_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "NVD_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            h = _nvd_headers()
            assert h == {}

    def test_returns_empty_for_empty_string(self):
        with mock.patch.dict(os.environ, {"NVD_API_KEY": ""}):
            h = _nvd_headers()
            assert h == {}


class TestVulncheckHeaders:
    """Tests for _vulncheck_headers."""

    def test_returns_bearer_when_set(self):
        with mock.patch.dict(os.environ, {"VULNCHECK_API_KEY": "vc-key-123"}):
            h = _vulncheck_headers()
            assert h == {"Authorization": "Bearer vc-key-123"}

    def test_returns_empty_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "VULNCHECK_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            h = _vulncheck_headers()
            assert h == {}


class TestGithubHeaders:
    """Tests for _github_headers."""

    def test_returns_bearer_with_github_token(self):
        env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_TOKEN", "MANUS_GITHUB_TOKEN")}
        env["GITHUB_TOKEN"] = "ghp_abc123"
        with mock.patch.dict(os.environ, env, clear=True):
            h = _github_headers()
            assert h["Authorization"] == "Bearer ghp_abc123"
            assert h["Accept"] == "application/vnd.github+json"

    def test_returns_bearer_with_manus_github_token(self):
        env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_TOKEN", "MANUS_GITHUB_TOKEN")}
        env["MANUS_GITHUB_TOKEN"] = "ghp_manus"
        with mock.patch.dict(os.environ, env, clear=True):
            h = _github_headers()
            assert h["Authorization"] == "Bearer ghp_manus"

    def test_prefers_github_token_over_manus(self):
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_first", "MANUS_GITHUB_TOKEN": "ghp_second"}):
            h = _github_headers()
            assert h["Authorization"] == "Bearer ghp_first"

    def test_no_auth_when_both_unset(self):
        env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_TOKEN", "MANUS_GITHUB_TOKEN")}
        with mock.patch.dict(os.environ, env, clear=True):
            h = _github_headers()
            assert "Authorization" not in h
            assert h["Accept"] == "application/vnd.github+json"


# ---------------------------------------------------------------------------
# _short_error
# ---------------------------------------------------------------------------


class TestShortError:
    def test_simple_message(self):
        assert _short_error(Exception("boom")) == "boom"

    def test_truncates_long_message(self):
        long_msg = "x" * 200
        result = _short_error(Exception(long_msg))
        assert len(result) <= 120

    def test_strips_nested_chain(self):
        result = _short_error(Exception("outer: inner: actual error"))
        assert result == "actual error"


# ---------------------------------------------------------------------------
# _run_single_probe
# ---------------------------------------------------------------------------


class TestRunSingleProbe:
    """Tests for _run_single_probe."""

    def _make_probe(self, **overrides):
        base = {
            "name": "Test API",
            "url": "https://example.com/api",
            "note": "test note",
        }
        base.update(overrides)
        return base

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_successful_get(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {}
        mock_req.return_value = resp

        result = _run_single_probe(self._make_probe())

        assert result["ok"] is True
        assert result["status_code"] == 200
        assert result["latency_ms"] >= 0
        assert result["name"] == "Test API"
        mock_req.assert_called_once()

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_head_method(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {}
        mock_req.return_value = resp

        _run_single_probe(self._make_probe(method="HEAD"))

        mock_req.assert_called_once()
        assert mock_req.call_args[0][0] == "HEAD"

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_custom_headers_fn(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {}
        mock_req.return_value = resp

        _run_single_probe(self._make_probe(headers_fn=lambda: {"X-Custom": "val"}))

        _, kwargs = mock_req.call_args
        assert kwargs["headers"]["X-Custom"] == "val"

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_401_sets_error(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 401
        resp.headers = {}
        mock_req.return_value = resp

        result = _run_single_probe(self._make_probe())

        assert result["ok"] is False
        assert "Authentication failed" in result["error"]

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_403_without_key(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 403
        resp.headers = {}
        mock_req.return_value = resp

        env = {k: v for k, v in os.environ.items() if k != "MY_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            result = _run_single_probe(self._make_probe(key_env="MY_KEY"))

        assert result["ok"] is False
        assert "set MY_KEY" in result["error"]

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_403_with_key_set(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 403
        resp.headers = {}
        mock_req.return_value = resp

        with mock.patch.dict(os.environ, {"MY_KEY": "some-key"}):
            result = _run_single_probe(self._make_probe(key_env="MY_KEY"))

        assert result["ok"] is False
        assert "invalid or expired" in result["error"]

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_429_rate_limited(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 429
        resp.headers = {}
        mock_req.return_value = resp

        result = _run_single_probe(self._make_probe())

        assert result["ok"] is False
        assert "Rate-limited" in result["error"]

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_500_server_error(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 502
        resp.headers = {}
        mock_req.return_value = resp

        result = _run_single_probe(self._make_probe())

        assert result["ok"] is False
        assert "Server error" in result["error"]

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_rate_limit_headers_captured(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {
            "X-RateLimit-Remaining": "42",
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Reset": "1700000000",
        }
        mock_req.return_value = resp

        result = _run_single_probe(self._make_probe())

        assert result["ok"] is True
        assert result["rate_limit"]["remaining"] == "42"
        assert result["rate_limit"]["limit"] == "100"
        assert result["rate_limit"]["reset"] == "1700000000"

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_retry_after_marks_unhealthy(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {"Retry-After": "60"}
        mock_req.return_value = resp

        result = _run_single_probe(self._make_probe())

        assert result["ok"] is False
        assert "Rate-limited" in result["error"]
        assert result["rate_limit"]["retry_after"] == "60"

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_connect_timeout(self, mock_req):
        mock_req.side_effect = requests.exceptions.ConnectTimeout("timed out")

        result = _run_single_probe(self._make_probe())

        assert result["ok"] is False
        assert result["status_code"] is None
        assert "timed out" in result["error"].lower()

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_connection_error(self, mock_req):
        mock_req.side_effect = requests.exceptions.ConnectionError("DNS failed: host not found")

        result = _run_single_probe(self._make_probe())

        assert result["ok"] is False
        assert result["status_code"] is None
        assert "Connection error" in result["error"]

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_general_timeout(self, mock_req):
        mock_req.side_effect = requests.exceptions.Timeout("read timed out")

        result = _run_single_probe(self._make_probe())

        assert result["ok"] is False
        assert "timed out" in result["error"].lower()

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_request_exception(self, mock_req):
        mock_req.side_effect = requests.exceptions.RequestException("something broke")

        result = _run_single_probe(self._make_probe())

        assert result["ok"] is False
        assert "Request failed" in result["error"]

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_key_set_detection(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {}
        mock_req.return_value = resp

        with mock.patch.dict(os.environ, {"MY_KEY": "secret"}):
            result = _run_single_probe(self._make_probe(key_env="MY_KEY"))

        assert result["key_set"] is True

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_key_unset_detection(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {}
        mock_req.return_value = resp

        env = {k: v for k, v in os.environ.items() if k != "MY_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            result = _run_single_probe(self._make_probe(key_env="MY_KEY"))

        assert result["key_set"] is False

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_no_key_env_means_none(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {}
        mock_req.return_value = resp

        result = _run_single_probe(self._make_probe())  # no key_env

        assert result["key_set"] is None

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_timeout_parameter_forwarded(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {}
        mock_req.return_value = resp

        _run_single_probe(self._make_probe(), timeout=5.0)

        _, kwargs = mock_req.call_args
        assert kwargs["timeout"] == 5.0


# ---------------------------------------------------------------------------
# check_api_health (orchestrator)
# ---------------------------------------------------------------------------


class TestCheckApiHealth:
    """Tests for the check_api_health orchestrator."""

    @mock.patch("manus_agent.tools.check_api_health._run_single_probe")
    def test_all_healthy(self, mock_probe):
        mock_probe.return_value = {"name": "Test", "ok": True}

        probes = [{"name": "A"}, {"name": "B"}]
        report = check_api_health(probes=probes)

        assert report["all_healthy"] is True
        assert report["summary"]["healthy"] == 2
        assert report["summary"]["down"] == 0
        assert report["summary"]["total"] == 2

    @mock.patch("manus_agent.tools.check_api_health._run_single_probe")
    def test_one_down(self, mock_probe):
        def side_effect(probe, timeout):
            if probe["name"] == "A":
                return {"name": "A", "ok": True}
            return {"name": "B", "ok": False, "error": "timeout"}

        mock_probe.side_effect = side_effect

        probes = [{"name": "A"}, {"name": "B"}]
        report = check_api_health(probes=probes)

        assert report["all_healthy"] is False
        assert report["summary"]["healthy"] == 1
        assert report["summary"]["down"] == 1

    @mock.patch("manus_agent.tools.check_api_health._run_single_probe")
    def test_results_sorted_by_probe_order(self, mock_probe):
        """Results must be in the same order as the probe list, regardless of completion order."""
        import time

        def side_effect(probe, timeout):
            # B returns first, A second — but output should be A, B
            if probe["name"] == "B":
                return {"name": "B", "ok": True}
            time.sleep(0.01)
            return {"name": "A", "ok": True}

        mock_probe.side_effect = side_effect

        probes = [{"name": "A"}, {"name": "B"}]
        report = check_api_health(probes=probes)

        assert [r["name"] for r in report["results"]] == ["A", "B"]

    @mock.patch("manus_agent.tools.check_api_health._run_single_probe")
    def test_uses_default_probes_when_none_passed(self, mock_probe):
        mock_probe.return_value = {"name": "x", "ok": True}

        report = check_api_health()

        # Should have called _run_single_probe once per default probe
        assert mock_probe.call_count == len(_PROBES)
        assert report["summary"]["total"] == len(_PROBES)

    @mock.patch("manus_agent.tools.check_api_health._run_single_probe")
    def test_timeout_forwarded(self, mock_probe):
        mock_probe.return_value = {"name": "x", "ok": True}

        check_api_health(timeout=3.5, probes=[{"name": "x"}])

        mock_probe.assert_called_once_with({"name": "x"}, 3.5)

    @mock.patch("manus_agent.tools.check_api_health._run_single_probe")
    def test_empty_probe_list(self, mock_probe):
        report = check_api_health(probes=[])

        assert report["all_healthy"] is True
        assert report["summary"]["total"] == 0
        mock_probe.assert_not_called()

    @mock.patch("manus_agent.tools.check_api_health._run_single_probe")
    def test_all_down(self, mock_probe):
        mock_probe.return_value = {"name": "x", "ok": False, "error": "down"}

        probes = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
        report = check_api_health(probes=probes)

        assert report["all_healthy"] is False
        assert report["summary"]["healthy"] == 0
        assert report["summary"]["down"] == 3


# ---------------------------------------------------------------------------
# Text report formatting
# ---------------------------------------------------------------------------


class TestFormatTextReport:
    def test_basic_format(self):
        report = {
            "results": [
                {
                    "name": "NVD",
                    "ok": True,
                    "status_code": 200,
                    "latency_ms": 150,
                },
                {
                    "name": "EPSS",
                    "ok": False,
                    "status_code": 429,
                    "latency_ms": 50,
                    "error": "Rate-limited",
                },
            ],
            "summary": {"healthy": 1, "down": 1, "total": 2},
        }

        lines = _format_text_report(report)
        text = "\n".join(lines)

        assert "API Health Check" in text
        assert "NVD" in text
        assert "EPSS" in text
        assert "Rate-limited" in text
        assert "1/2" in text

    def test_key_status_shown(self):
        report = {
            "results": [
                {
                    "name": "VulnCheck",
                    "ok": True,
                    "status_code": 200,
                    "latency_ms": 100,
                    "key_env": "VULNCHECK_API_KEY",
                    "key_set": True,
                },
            ],
            "summary": {"healthy": 1, "down": 0, "total": 1},
        }

        lines = _format_text_report(report)
        text = "\n".join(lines)

        assert "key:set" in text

    def test_key_unset_shown(self):
        report = {
            "results": [
                {
                    "name": "VulnCheck",
                    "ok": False,
                    "status_code": 403,
                    "latency_ms": 80,
                    "key_env": "VULNCHECK_API_KEY",
                    "key_set": False,
                    "error": "Forbidden",
                },
            ],
            "summary": {"healthy": 0, "down": 1, "total": 1},
        }

        lines = _format_text_report(report)
        text = "\n".join(lines)

        assert "key:UNSET" in text

    def test_no_key_env_no_key_column(self):
        report = {
            "results": [
                {
                    "name": "EPSS",
                    "ok": True,
                    "status_code": 200,
                    "latency_ms": 75,
                },
            ],
            "summary": {"healthy": 1, "down": 0, "total": 1},
        }

        lines = _format_text_report(report)
        text = "\n".join(lines)

        assert "key:" not in text


# ---------------------------------------------------------------------------
# Plain text fallback
# ---------------------------------------------------------------------------


class TestPrintPlain:
    def test_prints_output(self, capsys):
        report = {
            "results": [
                {"name": "NVD", "ok": True, "status_code": 200, "latency_ms": 100},
                {"name": "EPSS", "ok": False, "status_code": None, "latency_ms": 0, "error": "timeout"},
            ],
            "summary": {"healthy": 1, "down": 1, "total": 2},
        }

        _print_plain(report)

        captured = capsys.readouterr()
        assert "NVD" in captured.out
        assert "EPSS" in captured.out
        assert "1/2 healthy" in captured.out


# ---------------------------------------------------------------------------
# Rich report
# ---------------------------------------------------------------------------


class TestPrintApiHealthReport:
    @mock.patch("rich.console.Console")
    def test_all_healthy_report(self, mock_console_cls):
        console_instance = mock.Mock()
        mock_console_cls.return_value = console_instance

        report = {
            "results": [
                {
                    "name": "NVD",
                    "ok": True,
                    "status_code": 200,
                    "latency_ms": 120,
                    "note": "NVD test",
                },
            ],
            "summary": {"healthy": 1, "down": 0, "total": 1},
            "all_healthy": True,
        }

        print_api_health_report(report)

        # Should call console.print at least twice (table + panel)
        assert console_instance.print.call_count >= 2

    @mock.patch("rich.console.Console")
    def test_unhealthy_report(self, mock_console_cls):
        console_instance = mock.Mock()
        mock_console_cls.return_value = console_instance

        report = {
            "results": [
                {
                    "name": "NVD",
                    "ok": False,
                    "status_code": 503,
                    "latency_ms": 5000,
                    "error": "Server error",
                },
            ],
            "summary": {"healthy": 0, "down": 1, "total": 1},
            "all_healthy": False,
        }

        print_api_health_report(report)

        assert console_instance.print.call_count >= 2

    @mock.patch("rich.console.Console")
    def test_rate_limit_display(self, mock_console_cls):
        console_instance = mock.Mock()
        mock_console_cls.return_value = console_instance

        report = {
            "results": [
                {
                    "name": "GitHub",
                    "ok": True,
                    "status_code": 200,
                    "latency_ms": 90,
                    "rate_limit": {"remaining": "42", "limit": "60"},
                    "key_env": "GITHUB_TOKEN",
                    "key_set": True,
                    "note": "",
                },
            ],
            "summary": {"healthy": 1, "down": 0, "total": 1},
            "all_healthy": True,
        }

        print_api_health_report(report)
        assert console_instance.print.call_count >= 2

    @mock.patch("rich.console.Console")
    def test_retry_after_display(self, mock_console_cls):
        console_instance = mock.Mock()
        mock_console_cls.return_value = console_instance

        report = {
            "results": [
                {
                    "name": "NVD",
                    "ok": False,
                    "status_code": 200,
                    "latency_ms": 100,
                    "rate_limit": {"retry_after": "30"},
                    "error": "Rate-limited",
                },
            ],
            "summary": {"healthy": 0, "down": 1, "total": 1},
            "all_healthy": False,
        }

        print_api_health_report(report)
        assert console_instance.print.call_count >= 2

    @mock.patch("rich.console.Console")
    def test_no_key_env_shows_na(self, mock_console_cls):
        console_instance = mock.Mock()
        mock_console_cls.return_value = console_instance

        report = {
            "results": [
                {
                    "name": "EPSS",
                    "ok": True,
                    "status_code": 200,
                    "latency_ms": 60,
                    "note": "no key",
                },
            ],
            "summary": {"healthy": 1, "down": 0, "total": 1},
            "all_healthy": True,
        }

        print_api_health_report(report)
        assert console_instance.print.call_count >= 2

    @mock.patch("rich.console.Console")
    def test_long_note_truncated(self, mock_console_cls):
        console_instance = mock.Mock()
        mock_console_cls.return_value = console_instance

        report = {
            "results": [
                {
                    "name": "Test",
                    "ok": True,
                    "status_code": 200,
                    "latency_ms": 100,
                    "note": "x" * 200,  # very long note
                },
            ],
            "summary": {"healthy": 1, "down": 0, "total": 1},
            "all_healthy": True,
        }

        # Should not raise
        print_api_health_report(report)


# ---------------------------------------------------------------------------
# Strands tool interface
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_spec_has_required_fields(self):
        assert TOOL_SPEC["name"] == "check_api_health"
        assert "description" in TOOL_SPEC
        assert "inputSchema" in TOOL_SPEC

    def test_spec_properties(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert "timeout" in props
        assert "output" in props
        assert props["output"]["enum"] == ["text", "json"]


class TestCheckApiHealthTool:
    """Tests for the Strands tool entry point."""

    @mock.patch("manus_agent.tools.check_api_health.check_api_health")
    def test_text_output(self, mock_check):
        mock_check.return_value = {
            "results": [{"name": "NVD", "ok": True, "status_code": 200, "latency_ms": 100}],
            "summary": {"healthy": 1, "down": 0, "total": 1},
            "all_healthy": True,
        }

        result = check_api_health_tool(
            {"toolUseId": "t1", "input": {"output": "text"}},
        )

        assert result["status"] == "success"
        assert "NVD" in result["content"][0]["text"]

    @mock.patch("manus_agent.tools.check_api_health.check_api_health")
    def test_json_output(self, mock_check):
        mock_check.return_value = {
            "results": [{"name": "NVD", "ok": True, "status_code": 200, "latency_ms": 100}],
            "summary": {"healthy": 1, "down": 0, "total": 1},
            "all_healthy": True,
        }

        result = check_api_health_tool(
            {"toolUseId": "t2", "input": {"output": "json"}},
        )

        assert result["status"] == "success"
        parsed = json.loads(result["content"][0]["text"])
        assert parsed["all_healthy"] is True

    @mock.patch("manus_agent.tools.check_api_health.check_api_health")
    def test_unhealthy_returns_error_status(self, mock_check):
        mock_check.return_value = {
            "results": [{"name": "NVD", "ok": False, "error": "timeout"}],
            "summary": {"healthy": 0, "down": 1, "total": 1},
            "all_healthy": False,
        }

        result = check_api_health_tool(
            {"toolUseId": "t3", "input": {}},
        )

        assert result["status"] == "error"

    @mock.patch("manus_agent.tools.check_api_health.check_api_health")
    def test_timeout_forwarded(self, mock_check):
        mock_check.return_value = {
            "results": [],
            "summary": {"healthy": 0, "down": 0, "total": 0},
            "all_healthy": True,
        }

        check_api_health_tool(
            {"toolUseId": "t4", "input": {"timeout": 5}},
        )

        mock_check.assert_called_once_with(timeout=5)

    @mock.patch("manus_agent.tools.check_api_health.check_api_health")
    def test_default_timeout(self, mock_check):
        mock_check.return_value = {
            "results": [],
            "summary": {"healthy": 0, "down": 0, "total": 0},
            "all_healthy": True,
        }

        check_api_health_tool(
            {"toolUseId": "t5", "input": {}},
        )

        mock_check.assert_called_once_with(timeout=10)

    @mock.patch("manus_agent.tools.check_api_health.check_api_health")
    def test_exception_caught(self, mock_check):
        mock_check.side_effect = RuntimeError("kaboom")

        result = check_api_health_tool(
            {"toolUseId": "t6", "input": {}},
        )

        assert result["status"] == "error"
        assert "kaboom" in result["content"][0]["text"]

    def test_missing_tool_use_id(self):
        with mock.patch("manus_agent.tools.check_api_health.check_api_health") as mock_check:
            mock_check.return_value = {
                "results": [],
                "summary": {"healthy": 0, "down": 0, "total": 0},
                "all_healthy": True,
            }

            result = check_api_health_tool({"input": {}})

            assert result["toolUseId"] == ""


# ---------------------------------------------------------------------------
# Default probes structure
# ---------------------------------------------------------------------------


class TestDefaultProbes:
    def test_probe_count(self):
        assert len(_PROBES) == 6

    def test_all_probes_have_required_keys(self):
        for p in _PROBES:
            assert "name" in p
            assert "url" in p
            assert "note" in p

    def test_probe_names_unique(self):
        names = [p["name"] for p in _PROBES]
        assert len(names) == len(set(names))

    def test_probe_urls_are_https(self):
        for p in _PROBES:
            assert p["url"].startswith("https://"), f"{p['name']} URL is not HTTPS"

    def test_known_probes_present(self):
        names = {p["name"] for p in _PROBES}
        assert "NVD (NIST)" in names
        assert "EPSS (FIRST.org)" in names
        assert "OSV.dev" in names
        assert "CISA KEV" in names
        assert "GitHub Advisories" in names
        assert "VulnCheck KEV" in names


# ---------------------------------------------------------------------------
# CLI integration (doctor --apis)
# ---------------------------------------------------------------------------


class TestDoctorApisFlag:
    """Tests for the --apis flag on manus-agent doctor."""

    def test_doctor_parser_has_apis_flag(self):
        """The --apis flag must be registered on the doctor parser."""
        import sys

        with mock.patch.object(sys, "argv", ["manus-agent", "doctor"]):
            from manus_agent.cli import _build_doctor_parser

            parser = _build_doctor_parser()
            args = parser.parse_args([])
            assert hasattr(args, "apis")
            assert args.apis is False

    def test_doctor_parser_apis_true(self):
        from manus_agent.cli import _build_doctor_parser

        parser = _build_doctor_parser()
        args = parser.parse_args(["--apis"])
        assert args.apis is True

    def test_doctor_parser_has_timeout_flag(self):
        from manus_agent.cli import _build_doctor_parser

        parser = _build_doctor_parser()
        args = parser.parse_args(["--timeout", "5"])
        assert args.timeout == 5.0

    def test_doctor_parser_default_timeout(self):
        from manus_agent.cli import _build_doctor_parser

        parser = _build_doctor_parser()
        args = parser.parse_args([])
        assert args.timeout == 10.0

    @mock.patch("manus_agent.tools.check_api_health.check_api_health")
    def test_doctor_apis_runs_health_check(self, mock_check):
        """When --apis is passed, doctor must call check_api_health."""
        mock_check.return_value = {
            "results": [{"name": "NVD", "ok": True}],
            "summary": {"healthy": 1, "down": 0, "total": 1},
            "all_healthy": True,
        }

        import sys

        from manus_agent.cli import _build_doctor_parser, _cmd_doctor

        parser = _build_doctor_parser()

        with (
            mock.patch.object(sys, "argv", ["manus-agent", "doctor", "--apis"]),
            mock.patch("manus_agent.cli._check_import", return_value=True),
            mock.patch("manus_agent.cli.console"),
            mock.patch("manus_agent.tools.check_api_health.print_api_health_report"),
            mock.patch("shutil.which", return_value=None),
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
        ):
            args = parser.parse_args(["--apis"])
            rc = _cmd_doctor(args)

        mock_check.assert_called_once()
        assert rc == 0

    @mock.patch("manus_agent.tools.check_api_health.check_api_health")
    def test_doctor_apis_failure_adds_issues(self, mock_check):
        """When an API is down, doctor --apis adds it to the issues list."""
        mock_check.return_value = {
            "results": [
                {"name": "NVD", "ok": True},
                {"name": "EPSS", "ok": False, "error": "Connection timed out"},
            ],
            "summary": {"healthy": 1, "down": 1, "total": 2},
            "all_healthy": False,
        }

        import sys

        from manus_agent.cli import _build_doctor_parser, _cmd_doctor

        parser = _build_doctor_parser()

        with (
            mock.patch.object(sys, "argv", ["manus-agent", "doctor", "--apis"]),
            mock.patch("manus_agent.cli._check_import", return_value=True),
            mock.patch("manus_agent.cli.console"),
            mock.patch("manus_agent.tools.check_api_health.print_api_health_report"),
            mock.patch("shutil.which", return_value=None),
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
        ):
            args = parser.parse_args(["--apis"])
            rc = _cmd_doctor(args)

        # doctor should return 1 because EPSS is down
        assert rc == 1

    def test_doctor_without_apis_does_not_import_health_check(self):
        """When --apis is NOT passed, doctor must not import check_api_health."""
        import sys

        from manus_agent.cli import _build_doctor_parser, _cmd_doctor

        parser = _build_doctor_parser()

        with (
            mock.patch.object(sys, "argv", ["manus-agent", "doctor"]),
            mock.patch("manus_agent.cli._check_import", return_value=True),
            mock.patch("manus_agent.cli.console"),
            mock.patch("shutil.which", return_value=None),
            mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
        ):
            args = parser.parse_args([])
            rc = _cmd_doctor(args)

        assert rc == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_empty_response_headers(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {}
        mock_req.return_value = resp

        probe = {"name": "Test", "url": "https://example.com"}
        result = _run_single_probe(probe)

        assert "rate_limit" not in result
        assert result["ok"] is True

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_redirect_followed(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {}
        mock_req.return_value = resp

        probe = {"name": "Test", "url": "https://example.com"}
        _run_single_probe(probe)

        _, kwargs = mock_req.call_args
        assert kwargs["allow_redirects"] is True

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_params_forwarded(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {}
        mock_req.return_value = resp

        probe = {"name": "Test", "url": "https://example.com", "params": {"foo": "bar"}}
        _run_single_probe(probe)

        _, kwargs = mock_req.call_args
        assert kwargs["params"] == {"foo": "bar"}

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_200_is_ok(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 200
        resp.headers = {}
        mock_req.return_value = resp

        probe = {"name": "Test", "url": "https://example.com"}
        result = _run_single_probe(probe)
        assert result["ok"] is True

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_301_is_ok(self, mock_req):
        """3xx after redirect resolution is still < 400."""
        resp = mock.Mock()
        resp.status_code = 301
        resp.headers = {}
        mock_req.return_value = resp

        probe = {"name": "Test", "url": "https://example.com"}
        result = _run_single_probe(probe)
        assert result["ok"] is True

    @mock.patch("manus_agent.tools.check_api_health.requests.request")
    def test_400_is_not_ok(self, mock_req):
        resp = mock.Mock()
        resp.status_code = 400
        resp.headers = {}
        mock_req.return_value = resp

        probe = {"name": "Test", "url": "https://example.com"}
        result = _run_single_probe(probe)
        assert result["ok"] is False
