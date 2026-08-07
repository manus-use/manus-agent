"""Comprehensive test suite for get_github_advisory tool.

Tests cover:
- TOOL_SPEC contract validation
- Input validation (invalid CVE IDs)
- Token/authentication handling (env var, config, none)
- Successful advisory fetch (single result, multiple results)
- Empty results (no advisory found)
- HTTP error handling (404, 403, 500, etc.)
- Network/request exceptions
- Unexpected response formats (KeyError, IndexError, ValueError)
- Output logging integration
- Request construction (URL, headers, timeout)
"""

from unittest.mock import MagicMock, patch

import requests

from manus_agent.tools.get_github_advisory import get_github_advisory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_ADVISORY = {
    "ghsa_id": "GHSA-xxxx-yyyy-zzzz",
    "cve_id": "CVE-2023-1234",
    "summary": "A critical vulnerability in example-lib",
    "description": "Detailed description of the vulnerability.",
    "severity": "critical",
    "cvss": {"score": 9.8, "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"},
    "vulnerabilities": [
        {
            "package": {"ecosystem": "npm", "name": "example-lib"},
            "vulnerable_version_range": "< 2.0.0",
            "first_patched_version": "2.0.0",
        }
    ],
    "references": [{"url": "https://github.com/example/example-lib/security/advisories/GHSA-xxxx-yyyy-zzzz"}],
    "published_at": "2023-06-15T00:00:00Z",
    "updated_at": "2023-06-20T12:00:00Z",
}

SAMPLE_ADVISORY_2 = {
    "ghsa_id": "GHSA-aaaa-bbbb-cccc",
    "cve_id": "CVE-2023-1234",
    "summary": "A secondary advisory for the same CVE",
    "severity": "high",
}


def _invoke(cve_id):
    """Invoke the tool function, unwrapping the @tool decorator."""
    # strands @tool wraps the function in DecoratedFunctionTool;
    # call the underlying _tool_func directly to avoid decorator overhead
    return get_github_advisory._tool_func(cve_id=cve_id)


# ---------------------------------------------------------------------------
# TOOL_SPEC contract
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Verify the tool's metadata/spec contract."""

    def test_tool_has_name(self):
        assert hasattr(get_github_advisory, "TOOL_SPEC") or hasattr(get_github_advisory, "tool_name")

    def test_tool_is_callable(self):
        assert callable(get_github_advisory)

    def test_tool_spec_structure(self):
        spec = getattr(get_github_advisory, "TOOL_SPEC", None)
        if spec:
            assert "name" in spec
            assert "description" in spec or "inputSchema" in spec

    def test_tool_spec_name_matches(self):
        spec = getattr(get_github_advisory, "TOOL_SPEC", None)
        if spec:
            assert spec["name"] == "get_github_advisory"

    def test_tool_spec_has_cve_id_parameter(self):
        spec = getattr(get_github_advisory, "TOOL_SPEC", None)
        if spec and "inputSchema" in spec:
            props = spec["inputSchema"].get("properties", {})
            assert "cve_id" in props


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Verify rejection of invalid CVE ID inputs."""

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    def test_empty_string_rejected(self, mock_log):
        result = _invoke("")
        assert "error" in result
        assert "Invalid CVE ID" in result["error"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    def test_none_rejected(self, mock_log):
        result = _invoke(None)
        assert "error" in result
        assert "Invalid CVE ID" in result["error"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    def test_non_string_rejected(self, mock_log):
        result = _invoke(12345)
        assert "error" in result
        assert "Invalid CVE ID" in result["error"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    def test_wrong_prefix_rejected(self, mock_log):
        result = _invoke("GHSA-xxxx-yyyy-zzzz")
        assert "error" in result
        assert "Invalid CVE ID" in result["error"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    def test_partial_prefix_rejected(self, mock_log):
        result = _invoke("CV-2023-1234")
        assert "error" in result
        assert "Invalid CVE ID" in result["error"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    def test_valid_cve_format_accepted(self, mock_log):
        """Valid CVE ID should not trigger input validation error (mocked request)."""
        with patch("manus_agent.tools.get_github_advisory.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = [SAMPLE_ADVISORY]
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp
            with patch("manus_agent.tools.get_github_advisory.Config.from_file") as mock_cfg:
                mock_cfg.return_value = MagicMock(github=None)
                result = _invoke("CVE-2023-1234")
        assert "error" not in result

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    def test_lowercase_cve_accepted(self, mock_log):
        """Lowercase 'cve-' prefix should pass validation (startswith uses upper())."""
        with patch("manus_agent.tools.get_github_advisory.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = [SAMPLE_ADVISORY]
            mock_resp.raise_for_status.return_value = None
            mock_get.return_value = mock_resp
            with patch("manus_agent.tools.get_github_advisory.Config.from_file") as mock_cfg:
                mock_cfg.return_value = MagicMock(github=None)
                result = _invoke("cve-2023-5678")
        assert "error" not in result

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    def test_logs_called_on_invalid_input(self, mock_log):
        _invoke("")
        mock_log.assert_called_once()
        args = mock_log.call_args[0]
        assert args[0] == "get_github_advisory"


# ---------------------------------------------------------------------------
# Token / Authentication handling
# ---------------------------------------------------------------------------


class TestTokenHandling:
    """Verify correct authentication header construction."""

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_env_token_used(self, mock_cfg, mock_get, mock_log):
        """GITHUB_TOKEN env var should be used in Authorization header."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"}):
            _invoke("CVE-2023-1234")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "token ghp_test123"

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_config_token_used_when_no_env(self, mock_cfg, mock_get, mock_log):
        """Config github.api_token should be used when GITHUB_TOKEN env is absent."""
        mock_github = MagicMock()
        mock_github.api_token = "ghp_config_token"
        mock_cfg.return_value = MagicMock(github=mock_github)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            # Ensure GITHUB_TOKEN is not set
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2023-1234")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "token ghp_config_token"

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_no_auth_header_when_no_token(self, mock_cfg, mock_get, mock_log):
        """No Authorization header when neither env nor config has a token."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2023-1234")

        call_kwargs = mock_get.call_args[1]
        assert "Authorization" not in call_kwargs["headers"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_env_token_takes_precedence_over_config(self, mock_cfg, mock_get, mock_log):
        """GITHUB_TOKEN env var should take precedence over config token."""
        mock_github = MagicMock()
        mock_github.api_token = "ghp_config_token"
        mock_cfg.return_value = MagicMock(github=mock_github)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_env_token"}):
            _invoke("CVE-2023-1234")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "token ghp_env_token"

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_config_from_file_exception_falls_back_to_env(self, mock_get, mock_log):
        """If Config.from_file() raises, fall back to GITHUB_TOKEN env var."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch("manus_agent.tools.get_github_advisory.Config.from_file", side_effect=Exception("no config")):
            with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_fallback"}):
                _invoke("CVE-2023-1234")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "token ghp_fallback"

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    def test_config_exception_no_env_token(self, mock_get, mock_log):
        """If Config.from_file() raises and no env token, no auth header."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch("manus_agent.tools.get_github_advisory.Config.from_file", side_effect=Exception("no config")):
            with patch.dict("os.environ", {}, clear=True):
                import os

                os.environ.pop("GITHUB_TOKEN", None)
                _invoke("CVE-2023-1234")

        call_kwargs = mock_get.call_args[1]
        assert "Authorization" not in call_kwargs["headers"]


# ---------------------------------------------------------------------------
# Successful advisory fetch
# ---------------------------------------------------------------------------


class TestSuccessfulFetch:
    """Verify correct behavior when advisory is found."""

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_returns_first_advisory(self, mock_cfg, mock_get, mock_log):
        """Should return the first advisory from the list."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY, SAMPLE_ADVISORY_2]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert result == SAMPLE_ADVISORY

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_single_advisory_returned(self, mock_cfg, mock_get, mock_log):
        """Single advisory in list should be returned directly."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert result["ghsa_id"] == "GHSA-xxxx-yyyy-zzzz"
        assert result["severity"] == "critical"

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_advisory_preserves_all_fields(self, mock_cfg, mock_get, mock_log):
        """All advisory fields should be preserved in the result."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert result["summary"] == "A critical vulnerability in example-lib"
        assert result["cvss"]["score"] == 9.8
        assert len(result["vulnerabilities"]) == 1
        assert result["vulnerabilities"][0]["package"]["name"] == "example-lib"

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_logs_output_on_success(self, mock_cfg, mock_get, mock_log):
        """log_tool_output_size should be called on success."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2023-1234")

        mock_log.assert_called_once()
        assert mock_log.call_args[0][0] == "get_github_advisory"


# ---------------------------------------------------------------------------
# Empty results
# ---------------------------------------------------------------------------


class TestEmptyResults:
    """Verify behavior when no advisory is found."""

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_empty_list_returns_not_found(self, mock_cfg, mock_get, mock_log):
        """Empty API response list should return 'not found' message."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2099-9999")

        assert "message" in result
        assert "No advisory found" in result["message"]
        assert "CVE-2099-9999" in result["message"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_empty_list_logs_output(self, mock_cfg, mock_get, mock_log):
        """log_tool_output_size should be called even for empty results."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2099-9999")

        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------


class TestHTTPErrors:
    """Verify handling of HTTP error responses."""

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_404_returns_not_found(self, mock_cfg, mock_get, mock_log):
        """404 should return 'not found' message (not error)."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        http_err = requests.exceptions.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_err
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-0000")

        assert "message" in result
        assert "No advisory found" in result["message"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_403_returns_error(self, mock_cfg, mock_get, mock_log):
        """403 Forbidden should return HTTP error message."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        http_err = requests.exceptions.HTTPError("403 Client Error: Forbidden", response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_err
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert "error" in result
        assert "HTTP error" in result["error"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_500_returns_error(self, mock_cfg, mock_get, mock_log):
        """500 Internal Server Error should return HTTP error message."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        http_err = requests.exceptions.HTTPError("500 Server Error: Internal Server Error", response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_err
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert "error" in result
        assert "HTTP error" in result["error"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_429_rate_limit_returns_error(self, mock_cfg, mock_get, mock_log):
        """429 Rate Limit should return HTTP error message."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        http_err = requests.exceptions.HTTPError("429 Client Error: Too Many Requests", response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_err
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert "error" in result
        assert "HTTP error" in result["error"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_http_error_logs_output(self, mock_cfg, mock_get, mock_log):
        """log_tool_output_size should be called on HTTP error."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        http_err = requests.exceptions.HTTPError("500 error", response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_err
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2023-1234")

        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# Network / Request exceptions
# ---------------------------------------------------------------------------


class TestRequestExceptions:
    """Verify handling of network-level failures."""

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_connection_error(self, mock_cfg, mock_get, mock_log):
        """ConnectionError should return a descriptive error."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert "error" in result
        assert "error occurred" in result["error"].lower() or "Error" in result["error"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_timeout_error(self, mock_cfg, mock_get, mock_log):
        """Timeout should return a descriptive error."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert "error" in result
        assert "error occurred" in result["error"].lower() or "Error" in result["error"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_ssl_error(self, mock_cfg, mock_get, mock_log):
        """SSLError should return a descriptive error."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_get.side_effect = requests.exceptions.SSLError("SSL certificate verify failed")

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert "error" in result

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_generic_request_exception(self, mock_cfg, mock_get, mock_log):
        """Generic RequestException should return error."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_get.side_effect = requests.exceptions.RequestException("Something went wrong")

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert "error" in result
        assert "GitHub Advisory API" in result["error"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_request_exception_logs_output(self, mock_cfg, mock_get, mock_log):
        """log_tool_output_size should be called on request exception."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_get.side_effect = requests.exceptions.ConnectionError("Failed")

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2023-1234")

        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# Unexpected response format (KeyError, IndexError, ValueError)
# ---------------------------------------------------------------------------


class TestUnexpectedResponseFormat:
    """Verify handling of malformed API responses."""

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_json_decode_error(self, mock_cfg, mock_get, mock_log):
        """ValueError from json() should return unexpected response error."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("No JSON object could be decoded")
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert "error" in result
        assert "unexpected response" in result["error"].lower()

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_index_error_empty_access(self, mock_cfg, mock_get, mock_log):
        """IndexError when accessing data[0] should be caught."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        # Return something truthy but not indexable properly
        mock_resp.json.return_value = MagicMock()
        mock_resp.json.return_value.__bool__ = lambda self: True
        mock_resp.json.return_value.__getitem__ = MagicMock(side_effect=IndexError("list index out of range"))
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert "error" in result
        assert "unexpected response" in result["error"].lower()

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_key_error(self, mock_cfg, mock_get, mock_log):
        """KeyError during response processing should be caught."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        # Return truthy object that raises KeyError on access
        mock_data = MagicMock()
        mock_data.__bool__ = lambda self: True
        mock_data.__getitem__ = MagicMock(side_effect=KeyError("missing key"))
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            result = _invoke("CVE-2023-1234")

        assert "error" in result
        assert "unexpected response" in result["error"].lower()

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_unexpected_format_logs_output(self, mock_cfg, mock_get, mock_log):
        """log_tool_output_size should be called on unexpected format."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("bad json")
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2023-1234")

        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


class TestRequestConstruction:
    """Verify correct URL, headers, and timeout in the request."""

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_correct_url_format(self, mock_cfg, mock_get, mock_log):
        """Request URL should use GitHub advisories API with cve_id param."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2023-1234")

        call_args = mock_get.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "https://api.github.com/advisories" in url
        assert "cve_id=CVE-2023-1234" in url

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_accept_header(self, mock_cfg, mock_get, mock_log):
        """Request should include GitHub JSON Accept header."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2023-1234")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["Accept"] == "application/vnd.github+json"

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_api_version_header(self, mock_cfg, mock_get, mock_log):
        """Request should include X-GitHub-Api-Version header."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2023-1234")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["X-GitHub-Api-Version"] == "2022-11-28"

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_timeout_set(self, mock_cfg, mock_get, mock_log):
        """Request should have a timeout of 15 seconds."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2023-1234")

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["timeout"] == 15

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_cve_id_preserved_in_url(self, mock_cfg, mock_get, mock_log):
        """CVE ID with different formats should be preserved in URL."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2024-99999")

        call_args = mock_get.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "CVE-2024-99999" in url


# ---------------------------------------------------------------------------
# Output logging integration
# ---------------------------------------------------------------------------


class TestOutputLogging:
    """Verify log_tool_output_size is always called with correct arguments."""

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_log_format_on_success(self, mock_cfg, mock_get, mock_log):
        """Log call should have tool name and content structure."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [SAMPLE_ADVISORY]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2023-1234")

        args = mock_log.call_args[0]
        assert args[0] == "get_github_advisory"
        # Second arg should be a dict with "content" key containing list with "json"
        assert "content" in args[1]
        assert isinstance(args[1]["content"], list)
        assert "json" in args[1]["content"][0]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    def test_log_format_on_input_validation(self, mock_log):
        """Log call on invalid input should have same structure."""
        _invoke("")
        args = mock_log.call_args[0]
        assert args[0] == "get_github_advisory"
        assert "content" in args[1]
        assert "json" in args[1]["content"][0]
        # The json should contain the error
        assert "error" in args[1]["content"][0]["json"]

    @patch("manus_agent.tools.get_github_advisory.log_tool_output_size")
    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_log_format_on_empty_result(self, mock_cfg, mock_get, mock_log):
        """Log call on empty result should contain the message."""
        mock_cfg.return_value = MagicMock(github=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GITHUB_TOKEN", None)
            _invoke("CVE-2099-0001")

        args = mock_log.call_args[0]
        assert "message" in args[1]["content"][0]["json"]
