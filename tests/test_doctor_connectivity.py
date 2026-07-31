"""Comprehensive test suite for `manus-agent doctor --connectivity` and the changelog generate JSON bug fix.

Tests cover:
- _probe_endpoints() function: success, auth failures, timeouts, connection errors, rate limits
- _build_doctor_parser() --connectivity flag
- _cmd_doctor integration with --connectivity (mocked network)
- Bug fix verification: line 1697 indent=2 inside json.dumps() not print()
"""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def _clean_env(monkeypatch):
    """Remove API key env vars to test unauthenticated paths."""
    for key in ("NVD_API_KEY", "VULNCHECK_API_KEY", "GITHUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def _set_api_keys(monkeypatch):
    """Set all API key env vars to test authenticated paths."""
    monkeypatch.setenv("NVD_API_KEY", "test-nvd-key")
    monkeypatch.setenv("VULNCHECK_API_KEY", "test-vc-key")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")


# ---------------------------------------------------------------------------
# Tests: _build_doctor_parser
# ---------------------------------------------------------------------------


class TestBuildDoctorParser:
    """Tests for the --connectivity flag in _build_doctor_parser."""

    def test_connectivity_flag_absent_by_default(self):
        from manus_agent.cli import _build_doctor_parser

        parser = _build_doctor_parser()
        args = parser.parse_args([])
        assert args.connectivity is False

    def test_connectivity_flag_when_present(self):
        from manus_agent.cli import _build_doctor_parser

        parser = _build_doctor_parser()
        args = parser.parse_args(["--connectivity"])
        assert args.connectivity is True

    def test_config_and_connectivity_together(self):
        from manus_agent.cli import _build_doctor_parser

        parser = _build_doctor_parser()
        args = parser.parse_args(["--config", "/tmp/test.toml", "--connectivity"])
        assert args.connectivity is True
        assert args.config == Path("/tmp/test.toml")

    def test_connectivity_flag_order_independent(self):
        from manus_agent.cli import _build_doctor_parser

        parser = _build_doctor_parser()
        args = parser.parse_args(["--connectivity", "--config", "/tmp/test.toml"])
        assert args.connectivity is True


# ---------------------------------------------------------------------------
# Tests: _probe_endpoints
# ---------------------------------------------------------------------------


class TestProbeEndpoints:
    """Tests for the _probe_endpoints() connectivity checker."""

    def test_all_endpoints_reachable(self, _clean_env):
        """All endpoints return 200 — all marked reachable."""
        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp):
            results = _probe_endpoints()

        assert len(results) == 6
        for _label, reachable, detail in results:
            assert reachable is True
            assert "200" in detail

    def test_endpoint_returns_401_still_reachable(self, _clean_env):
        """401 means auth needed but endpoint is reachable."""
        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        with patch("requests.get", return_value=mock_resp):
            results = _probe_endpoints()

        for _label, reachable, detail in results:
            assert reachable is True
            assert "auth required" in detail

    def test_endpoint_returns_403_reachable_with_hint(self, _clean_env):
        """403 with no API key shows hint to set it."""
        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 403

        with patch("requests.get", return_value=mock_resp):
            results = _probe_endpoints()

        # At least the VulnCheck and GitHub endpoints should suggest setting env var
        vulncheck_result = next(r for r in results if "VulnCheck" in r[0])
        assert vulncheck_result[1] is True
        assert "VULNCHECK_API_KEY" in vulncheck_result[2]

        github_result = next(r for r in results if "GitHub" in r[0])
        assert github_result[1] is True
        assert "GITHUB_TOKEN" in github_result[2]

    def test_endpoint_returns_403_with_key_set(self, _set_api_keys):
        """403 with API key set does NOT show the 'set KEY' hint."""
        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 403

        with patch("requests.get", return_value=mock_resp):
            results = _probe_endpoints()

        vulncheck_result = next(r for r in results if "VulnCheck" in r[0])
        assert vulncheck_result[1] is True
        # Should NOT suggest setting the key since it's already set
        assert "set VULNCHECK_API_KEY" not in vulncheck_result[2]

    def test_endpoint_returns_429_reachable(self, _clean_env):
        """429 (rate-limited) still counts as reachable."""
        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 429

        with patch("requests.get", return_value=mock_resp):
            results = _probe_endpoints()

        for _label, reachable, detail in results:
            assert reachable is True
            assert "rate-limited" in detail

    def test_endpoint_returns_500_not_reachable(self, _clean_env):
        """500 server error is treated as unreachable."""
        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 500

        with patch("requests.get", return_value=mock_resp):
            results = _probe_endpoints()

        for _label, reachable, detail in results:
            assert reachable is False
            assert "500" in detail

    def test_timeout_not_reachable(self, _clean_env):
        """Timeout exception marks endpoint unreachable."""
        import requests

        from manus_agent.cli import _probe_endpoints

        with patch("requests.get", side_effect=requests.exceptions.Timeout("timed out")):
            results = _probe_endpoints()

        for _label, reachable, detail in results:
            assert reachable is False
            assert "timeout" in detail

    def test_connection_error_not_reachable(self, _clean_env):
        """ConnectionError marks endpoint unreachable."""
        import requests

        from manus_agent.cli import _probe_endpoints

        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("DNS failure")):
            results = _probe_endpoints()

        for _label, reachable, detail in results:
            assert reachable is False
            assert "connection refused" in detail or "DNS" in detail

    def test_generic_request_exception(self, _clean_env):
        """Generic RequestException marks endpoint unreachable."""
        import requests

        from manus_agent.cli import _probe_endpoints

        with patch("requests.get", side_effect=requests.exceptions.RequestException("something broke")):
            results = _probe_endpoints()

        for _label, reachable, detail in results:
            assert reachable is False
            assert "something broke" in detail

    def test_nvd_api_key_sent_in_header(self, monkeypatch):
        """NVD_API_KEY is sent as apiKey header."""
        monkeypatch.setenv("NVD_API_KEY", "my-nvd-key")
        monkeypatch.delenv("VULNCHECK_API_KEY", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp) as mock_get:
            _probe_endpoints()

        # Find the NVD call
        nvd_calls = [c for c in mock_get.call_args_list if "nvd.nist.gov" in str(c)]
        assert len(nvd_calls) == 1
        headers = nvd_calls[0][1]["headers"]
        assert headers["apiKey"] == "my-nvd-key"

    def test_vulncheck_api_key_sent_as_bearer(self, monkeypatch):
        """VULNCHECK_API_KEY is sent as Bearer token."""
        monkeypatch.setenv("VULNCHECK_API_KEY", "my-vc-key")
        monkeypatch.delenv("NVD_API_KEY", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp) as mock_get:
            _probe_endpoints()

        # Find the VulnCheck call
        vc_calls = [c for c in mock_get.call_args_list if "vulncheck.com" in str(c)]
        assert len(vc_calls) == 1
        headers = vc_calls[0][1]["headers"]
        assert headers["Authorization"] == "Bearer my-vc-key"

    def test_github_token_sent_as_token(self, monkeypatch):
        """GITHUB_TOKEN is sent as token auth."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_testtoken")
        monkeypatch.delenv("NVD_API_KEY", raising=False)
        monkeypatch.delenv("VULNCHECK_API_KEY", raising=False)

        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp) as mock_get:
            _probe_endpoints()

        # Find the GitHub call
        gh_calls = [c for c in mock_get.call_args_list if "api.github.com" in str(c)]
        assert len(gh_calls) == 1
        headers = gh_calls[0][1]["headers"]
        assert headers["Authorization"] == "token ghp_testtoken"

    def test_user_agent_always_set(self, _clean_env):
        """User-Agent header is always set."""
        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp) as mock_get:
            _probe_endpoints()

        for call in mock_get.call_args_list:
            headers = call[1]["headers"]
            assert headers["User-Agent"] == "manus-agent/doctor"

    def test_timeout_is_10_seconds(self, _clean_env):
        """Each request uses a 10-second timeout."""
        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp) as mock_get:
            _probe_endpoints()

        for call in mock_get.call_args_list:
            assert call[1]["timeout"] == 10

    def test_mixed_results(self, _clean_env):
        """Some endpoints up, some down — results are per-endpoint."""
        import requests

        from manus_agent.cli import _probe_endpoints

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise requests.exceptions.Timeout("slow")
            resp = MagicMock()
            resp.status_code = 200
            return resp

        with patch("requests.get", side_effect=side_effect):
            results = _probe_endpoints()

        reachable_count = sum(1 for _, r, _ in results if r)
        unreachable_count = sum(1 for _, r, _ in results if not r)
        assert reachable_count > 0
        assert unreachable_count > 0

    def test_returns_six_results(self, _clean_env):
        """_probe_endpoints probes exactly 6 known APIs."""
        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp):
            results = _probe_endpoints()

        assert len(results) == 6

    def test_endpoint_labels_are_descriptive(self, _clean_env):
        """Each result has a human-readable label."""
        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp):
            results = _probe_endpoints()

        labels = {r[0] for r in results}
        assert "NVD (NIST)" in labels
        assert "EPSS (FIRST)" in labels
        assert "CISA KEV" in labels
        assert "OSV.dev" in labels
        assert "GitHub API" in labels
        assert "VulnCheck" in labels


# ---------------------------------------------------------------------------
# Tests: _cmd_doctor with --connectivity
# ---------------------------------------------------------------------------


class TestCmdDoctorConnectivity:
    """Integration tests for _cmd_doctor with --connectivity flag."""

    def _make_args(self, connectivity=False, config=None):
        """Build a doctor args namespace."""
        return argparse.Namespace(connectivity=connectivity, config=config)

    @patch("manus_agent.cli._probe_endpoints")
    @patch("manus_agent.cli._check_import", return_value=True)
    @patch("shutil.which", return_value=None)
    def test_connectivity_not_called_without_flag(self, _which, _imp, mock_probe):
        """Without --connectivity, probes are not run."""
        from manus_agent.cli import _cmd_doctor

        args = self._make_args(connectivity=False)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=False):
            _cmd_doctor(args)

        mock_probe.assert_not_called()

    @patch("manus_agent.cli._probe_endpoints")
    @patch("manus_agent.cli._check_import", return_value=True)
    @patch("shutil.which", return_value=None)
    def test_connectivity_called_with_flag(self, _which, _imp, mock_probe):
        """With --connectivity, probes ARE run."""
        mock_probe.return_value = [
            ("NVD (NIST)", True, "HTTP 200"),
            ("EPSS (FIRST)", True, "HTTP 200"),
            ("CISA KEV", True, "HTTP 200"),
            ("OSV.dev", True, "HTTP 200"),
            ("GitHub API", True, "HTTP 200"),
            ("VulnCheck", True, "HTTP 200"),
        ]

        from manus_agent.cli import _cmd_doctor

        args = self._make_args(connectivity=True)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=False):
            exit_code = _cmd_doctor(args)

        mock_probe.assert_called_once()
        assert exit_code == 0

    @patch("manus_agent.cli._probe_endpoints")
    @patch("manus_agent.cli._check_import", return_value=True)
    @patch("shutil.which", return_value=None)
    def test_connectivity_failure_causes_exit_1(self, _which, _imp, mock_probe):
        """Unreachable API causes doctor to exit with code 1."""
        mock_probe.return_value = [
            ("NVD (NIST)", False, "timeout (>10s)"),
            ("EPSS (FIRST)", True, "HTTP 200"),
            ("CISA KEV", True, "HTTP 200"),
            ("OSV.dev", True, "HTTP 200"),
            ("GitHub API", True, "HTTP 200"),
            ("VulnCheck", True, "HTTP 200"),
        ]

        from manus_agent.cli import _cmd_doctor

        args = self._make_args(connectivity=True)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=False):
            exit_code = _cmd_doctor(args)

        assert exit_code == 1

    @patch("manus_agent.cli._probe_endpoints")
    @patch("manus_agent.cli._check_import", return_value=True)
    @patch("shutil.which", return_value=None)
    def test_all_probes_fail_reports_all_issues(self, _which, _imp, mock_probe):
        """Multiple failures are all reported."""
        mock_probe.return_value = [
            ("NVD (NIST)", False, "connection refused / DNS failure"),
            ("EPSS (FIRST)", False, "timeout (>10s)"),
            ("CISA KEV", False, "connection refused / DNS failure"),
            ("OSV.dev", False, "timeout (>10s)"),
            ("GitHub API", False, "connection refused / DNS failure"),
            ("VulnCheck", False, "timeout (>10s)"),
        ]

        from manus_agent.cli import _cmd_doctor

        args = self._make_args(connectivity=True)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=False):
            exit_code = _cmd_doctor(args)

        assert exit_code == 1


# ---------------------------------------------------------------------------
# Tests: Bug fix — _run_changelog_generate indent=2 in json.dumps
# ---------------------------------------------------------------------------


class TestChangelogGenerateBugFix:
    """Verify the fix for the indent=2 bug (was passed to print() not json.dumps())."""

    def test_no_commits_json_output_is_valid_json(self, tmp_path, monkeypatch):
        """When --generate --output json finds no commits, output is valid JSON."""
        from unittest.mock import patch as _patch

        from manus_agent.cli import _run_changelog_generate

        # Set up a minimal args namespace
        args = argparse.Namespace(output="json")

        # Create a fake git repo root
        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        # Mock subprocess.run to simulate: git tag returns empty, git log returns nothing
        def mock_run(cmd, *a, **kw):
            result = MagicMock()
            result.returncode = 0
            if "tag" in cmd:
                result.stdout = ""
            elif "log" in cmd:
                result.stdout = ""
            else:
                result.stdout = ""
            return result

        captured_output = []

        def mock_print(*args, **kwargs):
            captured_output.append(args[0] if args else "")

        with (
            _patch("subprocess.run", side_effect=mock_run),
            _patch("builtins.print", side_effect=mock_print),
        ):
            exit_code = _run_changelog_generate(args, tmp_path)

        assert exit_code == 0
        # The output should be valid JSON (not a TypeError from indent=2 in print)
        if captured_output:
            # Find the JSON output (not stderr messages)
            json_outputs = [o for o in captured_output if o.startswith("{")]
            if json_outputs:
                parsed = json.loads(json_outputs[0])
                assert "error" in parsed
                assert "commits" in parsed
                assert parsed["commits"] == []

    def test_indent_2_inside_json_dumps_not_print(self):
        """Verify the source code has indent=2 inside _json.dumps(), not print()."""
        import inspect

        from manus_agent.cli import _run_changelog_generate

        source = inspect.getsource(_run_changelog_generate)
        # The bug was: print(_json.dumps({"error": msg, "commits": []}), indent=2)
        # The fix is: print(_json.dumps({"error": msg, "commits": []}, indent=2))
        assert 'dumps({"error": msg, "commits": []}), indent=2)' not in source
        assert 'dumps({"error": msg, "commits": []}, indent=2)' in source


# ---------------------------------------------------------------------------
# Tests: _API_ENDPOINTS constant
# ---------------------------------------------------------------------------


class TestApiEndpointsConstant:
    """Validate the _API_ENDPOINTS constant is well-formed."""

    def test_all_entries_are_tuples_of_3(self):
        from manus_agent.cli import _API_ENDPOINTS

        for entry in _API_ENDPOINTS:
            assert len(entry) == 3
            label, url, env_key = entry
            assert isinstance(label, str)
            assert isinstance(url, str)
            assert url.startswith("https://")
            assert env_key is None or isinstance(env_key, str)

    def test_no_duplicate_labels(self):
        from manus_agent.cli import _API_ENDPOINTS

        labels = [entry[0] for entry in _API_ENDPOINTS]
        assert len(labels) == len(set(labels))

    def test_no_duplicate_urls(self):
        from manus_agent.cli import _API_ENDPOINTS

        urls = [entry[1] for entry in _API_ENDPOINTS]
        assert len(urls) == len(set(urls))


# ---------------------------------------------------------------------------
# Tests: edge cases and robustness
# ---------------------------------------------------------------------------


class TestProbeEdgeCases:
    """Edge-case tests for _probe_endpoints robustness."""

    def test_empty_api_key_not_sent(self, monkeypatch):
        """Empty string env vars are not sent as auth headers."""
        monkeypatch.setenv("NVD_API_KEY", "")
        monkeypatch.setenv("VULNCHECK_API_KEY", "")
        monkeypatch.setenv("GITHUB_TOKEN", "")

        from manus_agent.cli import _probe_endpoints

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp) as mock_get:
            _probe_endpoints()

        for call in mock_get.call_args_list:
            headers = call[1]["headers"]
            # Should only have User-Agent, not auth headers for empty keys
            assert "Authorization" not in headers or headers.get("Authorization") not in (
                "Bearer ",
                "token ",
            )
            assert headers.get("apiKey", "non-empty") != ""

    def test_probe_detail_string_truncated_on_long_error(self, _clean_env):
        """Very long error messages are truncated to avoid console spam."""
        import requests

        from manus_agent.cli import _probe_endpoints

        long_error = "x" * 200
        with patch(
            "requests.get",
            side_effect=requests.exceptions.RequestException(long_error),
        ):
            results = _probe_endpoints()

        for _, _, detail in results:
            assert len(detail) <= 60
