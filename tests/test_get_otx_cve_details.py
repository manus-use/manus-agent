"""Comprehensive test suite for the get_otx_cve_details tool module.

Tests cover:
- Input validation (invalid CVE ID formats, missing/empty/non-string inputs)
- API key resolution (env var, config file, fallback, config exception handling)
- Successful responses (with pulses, without pulses, empty data)
- HTTP error handling (404 not found, 500 server error, other status codes)
- Request exceptions (timeout, connection error, generic request failure)
- JSON decode errors
- Unexpected exceptions
- log_tool_output_size invocation for every return path
- Header and URL construction
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from manus_agent.tools.get_otx_cve_details import TOOL_SPEC, get_otx_cve_details

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_tool_use():
    """Factory for creating ToolUse dicts."""

    def _make(cve_id, tool_use_id="test-tool-use-id"):
        return {"toolUseId": tool_use_id, "input": {"cve_id": cve_id}}

    return _make


@pytest.fixture
def mock_config():
    """Mock Config.from_file returning a config with OTX key."""
    config = MagicMock()
    config.otx.api_key = "config-otx-key-123"
    return config


@pytest.fixture
def sample_pulse_response():
    """Sample OTX response with pulses."""
    return {
        "pulse_info": {
            "count": 2,
            "pulses": [
                {
                    "id": "pulse-001",
                    "name": "APT Campaign Targeting CVE-2024-3094",
                    "description": "Threat actors exploiting XZ backdoor",
                    "created": "2024-03-30T12:00:00",
                    "modified": "2024-04-01T08:00:00",
                    "tags": ["apt", "backdoor", "xz"],
                    "adversary": "Unknown",
                    "targeted_countries": ["US", "DE"],
                    "indicators": [
                        {"type": "domain", "indicator": "evil.example.com"},
                        {"type": "IPv4", "indicator": "192.168.1.100"},
                    ],
                },
                {
                    "id": "pulse-002",
                    "name": "Supply Chain Compromise via XZ Utils",
                    "description": "Monitoring supply chain attack vectors",
                    "created": "2024-03-31T00:00:00",
                    "modified": "2024-04-02T14:30:00",
                    "tags": ["supply-chain"],
                    "adversary": "",
                    "targeted_countries": [],
                    "indicators": [],
                },
            ],
        },
        "base_indicator": {
            "id": 12345,
            "type": "CVE",
            "indicator": "CVE-2024-3094",
        },
    }


# ---------------------------------------------------------------------------
# TOOL_SPEC tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Tests for the tool specification metadata."""

    def test_tool_spec_name(self):
        assert TOOL_SPEC["name"] == "get_otx_cve_details"

    def test_tool_spec_has_description(self):
        assert "AlienVault OTX" in TOOL_SPEC["description"]
        assert "CVE" in TOOL_SPEC["description"]

    def test_tool_spec_input_schema_requires_cve_id(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_tool_spec_cve_id_is_string_type(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["properties"]["cve_id"]["type"] == "string"


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for CVE ID format validation."""

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    def test_invalid_cve_id_not_starting_with_cve(self, mock_log, make_tool_use):
        tool = make_tool_use("VULN-2024-1234")
        result = get_otx_cve_details(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID format" in result["content"][0]["text"]
        mock_log.assert_called_once_with("get_otx_cve_details", result)

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    def test_invalid_cve_id_empty_string(self, mock_log, make_tool_use):
        tool = make_tool_use("")
        result = get_otx_cve_details(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID format" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    def test_invalid_cve_id_none(self, mock_log, make_tool_use):
        tool = make_tool_use(None)
        result = get_otx_cve_details(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID format" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    def test_invalid_cve_id_integer(self, mock_log, make_tool_use):
        tool = {"toolUseId": "test-id", "input": {"cve_id": 12345}}
        result = get_otx_cve_details(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID format" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    def test_invalid_cve_id_list(self, mock_log, make_tool_use):
        tool = {"toolUseId": "test-id", "input": {"cve_id": ["CVE-2024-1234"]}}
        result = get_otx_cve_details(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID format" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    def test_invalid_cve_id_random_text(self, mock_log, make_tool_use):
        tool = make_tool_use("hello world")
        result = get_otx_cve_details(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID format" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    def test_invalid_cve_id_partial_prefix(self, mock_log, make_tool_use):
        tool = make_tool_use("CV-2024-1234")
        result = get_otx_cve_details(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID format" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    def test_valid_cve_lowercase_accepted(self, mock_log, make_tool_use):
        """Lowercase 'cve-' prefix should pass validation (uppercased internally)."""
        tool = make_tool_use("cve-2024-1234")
        with (
            patch("manus_agent.tools.get_otx_cve_details.Config.from_file") as mock_cfg,
            patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="key123"),
            patch("manus_agent.tools.get_otx_cve_details.requests.get") as mock_get,
        ):
            mock_cfg.return_value = MagicMock(otx=None)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"pulse_info": {"pulses": []}}
            mock_get.return_value = mock_resp
            result = get_otx_cve_details(tool)
            # Should NOT be an invalid format error
            assert result["status"] == "success"

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    def test_tool_use_id_preserved_in_result(self, mock_log, make_tool_use):
        tool = make_tool_use("not-a-cve", tool_use_id="custom-id-xyz")
        result = get_otx_cve_details(tool)
        assert result["toolUseId"] == "custom-id-xyz"


# ---------------------------------------------------------------------------
# API key resolution tests
# ---------------------------------------------------------------------------


class TestApiKeyResolution:
    """Tests for API key lookup from env var and config file."""

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_env_var_takes_precedence_over_config(
        self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use, sample_pulse_response
    ):
        mock_env.return_value = "env-key-456"
        config = MagicMock()
        config.otx.api_key = "config-key-789"
        mock_cfg.return_value = config

        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_pulse_response
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-3094")
        get_otx_cve_details(tool)

        # The env var should be used — check header
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["X-OTX-API-KEY"] == "env-key-456"

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_config_key_used_when_env_var_absent(
        self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use, sample_pulse_response
    ):
        mock_env.return_value = None
        config = MagicMock()
        config.otx.api_key = "config-key-789"
        mock_cfg.return_value = config

        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_pulse_response
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-3094")
        get_otx_cve_details(tool)

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["X-OTX-API-KEY"] == "config-key-789"

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_no_api_key_returns_error(self, mock_cfg, mock_env, mock_log, make_tool_use):
        mock_env.return_value = None
        config = MagicMock()
        config.otx = None
        mock_cfg.return_value = config

        tool = make_tool_use("CVE-2024-1234")
        result = get_otx_cve_details(tool)

        assert result["status"] == "error"
        assert "API key not found" in result["content"][0]["text"]
        mock_log.assert_called_once_with("get_otx_cve_details", result)

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_config_exception_falls_back_to_env_var(self, mock_cfg, mock_env, mock_log, make_tool_use):
        mock_cfg.side_effect = Exception("Config file not found")
        mock_env.return_value = None

        tool = make_tool_use("CVE-2024-1234")
        result = get_otx_cve_details(tool)

        # Should reach the "no api_key" error, not crash
        assert result["status"] == "error"
        assert "API key not found" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_config_exception_env_key_available_proceeds(
        self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use, sample_pulse_response
    ):
        mock_cfg.side_effect = Exception("Config file not found")
        mock_env.return_value = "env-fallback-key"

        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_pulse_response
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-3094")
        result = get_otx_cve_details(tool)

        assert result["status"] == "success"
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"]["X-OTX-API-KEY"] == "env-fallback-key"

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_empty_string_api_key_treated_as_missing(self, mock_cfg, mock_env, mock_log, make_tool_use):
        mock_env.return_value = ""
        config = MagicMock()
        config.otx.api_key = ""
        mock_cfg.return_value = config

        tool = make_tool_use("CVE-2024-1234")
        result = get_otx_cve_details(tool)

        assert result["status"] == "error"
        assert "API key not found" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Successful response tests
# ---------------------------------------------------------------------------


class TestSuccessfulResponses:
    """Tests for successful API responses."""

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_response_with_pulses_returns_json_data(
        self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use, sample_pulse_response
    ):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_pulse_response
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-3094")
        result = get_otx_cve_details(tool)

        assert result["status"] == "success"
        assert result["content"][0]["json"] == sample_pulse_response
        assert result["toolUseId"] == "test-tool-use-id"

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_response_without_pulses_returns_text_message(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"pulse_info": {"pulses": []}}
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-9999")
        result = get_otx_cve_details(tool)

        assert result["status"] == "success"
        assert "No specific threat intelligence pulses found" in result["content"][0]["text"]
        assert "CVE-2024-9999" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_response_empty_pulse_info(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"pulse_info": {}}
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-0001")
        result = get_otx_cve_details(tool)

        assert result["status"] == "success"
        assert "No specific threat intelligence pulses found" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_response_empty_data(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-0001")
        result = get_otx_cve_details(tool)

        assert result["status"] == "success"
        assert "No specific threat intelligence pulses found" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_response_none_data(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        """If the API returns None (unlikely but defensive)."""
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = None
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-0001")
        result = get_otx_cve_details(tool)

        assert result["status"] == "success"
        assert "No specific threat intelligence pulses found" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_response_pulse_info_none_pulses(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"pulse_info": {"pulses": None}}
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-0001")
        result = get_otx_cve_details(tool)

        assert result["status"] == "success"
        assert "No specific threat intelligence pulses found" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# URL construction tests
# ---------------------------------------------------------------------------


class TestUrlConstruction:
    """Tests for correct URL and header construction."""

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_url_uses_uppercased_cve_id(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"pulse_info": {"pulses": []}}
        mock_get.return_value = mock_resp

        tool = make_tool_use("cve-2024-3094")
        get_otx_cve_details(tool)

        call_args = mock_get.call_args
        assert call_args[0][0] == "https://otx.alienvault.com/api/v1/indicators/cve/CVE-2024-3094"

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_request_uses_correct_timeout(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"pulse_info": {"pulses": []}}
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-1234")
        get_otx_cve_details(tool)

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["timeout"] == 20

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_request_calls_raise_for_status(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"pulse_info": {"pulses": []}}
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-1234")
        get_otx_cve_details(tool)

        mock_resp.raise_for_status.assert_called_once()

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="my-api-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_api_key_header_set_correctly(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"pulse_info": {"pulses": []}}
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-1234")
        get_otx_cve_details(tool)

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"] == {"X-OTX-API-KEY": "my-api-key"}


# ---------------------------------------------------------------------------
# HTTP error handling tests
# ---------------------------------------------------------------------------


class TestHttpErrors:
    """Tests for HTTP error responses."""

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_http_404_returns_not_found_message(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-9999")
        result = get_otx_cve_details(tool)

        assert result["status"] == "success"
        assert "No information found for CVE-2024-9999" in result["content"][0]["text"]
        mock_log.assert_called_once_with("get_otx_cve_details", result)

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_http_500_returns_error(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        http_error = requests.exceptions.HTTPError("500 Server Error", response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-1234")
        result = get_otx_cve_details(tool)

        assert result["status"] == "error"
        assert "HTTP error" in result["content"][0]["text"]
        mock_log.assert_called_once_with("get_otx_cve_details", result)

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_http_403_returns_error(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        http_error = requests.exceptions.HTTPError("403 Forbidden", response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-1234")
        result = get_otx_cve_details(tool)

        assert result["status"] == "error"
        assert "HTTP error" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_http_429_returns_error(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        http_error = requests.exceptions.HTTPError("429 Too Many Requests", response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-1234")
        result = get_otx_cve_details(tool)

        assert result["status"] == "error"
        assert "HTTP error" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Request exception tests
# ---------------------------------------------------------------------------


class TestRequestExceptions:
    """Tests for network-level request failures."""

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_timeout_returns_error(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        tool = make_tool_use("CVE-2024-1234")
        result = get_otx_cve_details(tool)

        assert result["status"] == "error"
        assert "failed" in result["content"][0]["text"].lower()
        mock_log.assert_called_once_with("get_otx_cve_details", result)

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_connection_error_returns_error(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_get.side_effect = requests.exceptions.ConnectionError("DNS resolution failed")

        tool = make_tool_use("CVE-2024-1234")
        result = get_otx_cve_details(tool)

        assert result["status"] == "error"
        assert "failed" in result["content"][0]["text"].lower()

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_generic_request_exception(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_get.side_effect = requests.exceptions.RequestException("Something went wrong")

        tool = make_tool_use("CVE-2024-1234")
        result = get_otx_cve_details(tool)

        assert result["status"] == "error"
        assert "failed" in result["content"][0]["text"].lower()


# ---------------------------------------------------------------------------
# JSON decode error tests
# ---------------------------------------------------------------------------


class TestJsonDecodeErrors:
    """Tests for malformed JSON responses."""

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_json_decode_error_returns_error(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        import json

        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-1234")
        result = get_otx_cve_details(tool)

        assert result["status"] == "error"
        assert "parse JSON" in result["content"][0]["text"]
        mock_log.assert_called_once_with("get_otx_cve_details", result)


# ---------------------------------------------------------------------------
# Unexpected exception tests
# ---------------------------------------------------------------------------


class TestUnexpectedExceptions:
    """Tests for unanticipated failures."""

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_unexpected_exception_returns_error(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_get.side_effect = RuntimeError("Unexpected internal error")

        tool = make_tool_use("CVE-2024-1234")
        result = get_otx_cve_details(tool)

        assert result["status"] == "error"
        assert "unexpected error" in result["content"][0]["text"].lower()
        mock_log.assert_called_once_with("get_otx_cve_details", result)

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="test-key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_keyboard_interrupt_propagates(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        """KeyboardInterrupt should NOT be caught by the generic except."""
        mock_cfg.return_value = MagicMock(otx=None)
        mock_get.side_effect = KeyboardInterrupt()

        tool = make_tool_use("CVE-2024-1234")
        # The generic `except Exception` does NOT catch BaseException subclasses like KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            get_otx_cve_details(tool)


# ---------------------------------------------------------------------------
# log_tool_output_size invocation tests
# ---------------------------------------------------------------------------


class TestLogToolOutputSize:
    """Tests verifying log_tool_output_size is called on every code path."""

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    def test_log_called_on_invalid_input(self, mock_log, make_tool_use):
        tool = make_tool_use("not-valid")
        get_otx_cve_details(tool)
        mock_log.assert_called_once()
        assert mock_log.call_args[0][0] == "get_otx_cve_details"

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value=None)
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_log_called_on_missing_api_key(self, mock_cfg, mock_env, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        tool = make_tool_use("CVE-2024-1234")
        get_otx_cve_details(tool)
        mock_log.assert_called_once()
        assert mock_log.call_args[0][0] == "get_otx_cve_details"

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_log_called_on_success_with_pulses(
        self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use, sample_pulse_response
    ):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = sample_pulse_response
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-3094")
        get_otx_cve_details(tool)
        mock_log.assert_called_once()

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_log_called_on_success_no_pulses(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"pulse_info": {"pulses": []}}
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-1234")
        get_otx_cve_details(tool)
        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# Edge case / integration tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge cases and integration scenarios."""

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_mixed_case_cve_uppercased_in_url(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"pulse_info": {"pulses": []}}
        mock_get.return_value = mock_resp

        tool = make_tool_use("CvE-2023-44487")
        get_otx_cve_details(tool)

        url = mock_get.call_args[0][0]
        assert "CVE-2023-44487" in url

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_cve_id_with_long_number(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        """CVE IDs with 5+ digit sequence numbers are valid."""
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"pulse_info": {"pulses": []}}
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-123456")
        result = get_otx_cve_details(tool)

        assert result["status"] == "success"
        url = mock_get.call_args[0][0]
        assert "CVE-2024-123456" in url

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_extra_kwargs_do_not_crash(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        """The function accepts **kwargs — extra arguments should be harmless."""
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"pulse_info": {"pulses": []}}
        mock_get.return_value = mock_resp

        tool = make_tool_use("CVE-2024-1234")
        result = get_otx_cve_details(tool, extra_param="ignored", another=42)

        assert result["status"] == "success"

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    def test_missing_cve_id_key_in_input(self, mock_log):
        """If cve_id key is missing from input dict entirely."""
        tool = {"toolUseId": "test-id", "input": {}}
        result = get_otx_cve_details(tool)

        assert result["status"] == "error"
        assert "Invalid CVE ID format" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_cve_with_whitespace_still_validated(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        """CVE- prefix with leading whitespace doesn't match startswith."""
        tool = make_tool_use("  CVE-2024-1234")
        result = get_otx_cve_details(tool)

        # Whitespace prefix means it doesn't start with "CVE-" — should fail validation
        assert result["status"] == "error"
        assert "Invalid CVE ID format" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_otx_cve_details.log_tool_output_size")
    @patch("manus_agent.tools.get_otx_cve_details.requests.get")
    @patch("manus_agent.tools.get_otx_cve_details.os.environ.get", return_value="key")
    @patch("manus_agent.tools.get_otx_cve_details.Config.from_file")
    def test_404_message_includes_original_cve_id(self, mock_cfg, mock_env, mock_get, mock_log, make_tool_use):
        mock_cfg.return_value = MagicMock(otx=None)
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_error
        mock_get.return_value = mock_resp

        tool = make_tool_use("cve-2021-44228")
        result = get_otx_cve_details(tool)

        # The message uses the original cve_id (uppercased via variable)
        assert result["status"] == "success"
        assert "cve-2021-44228" in result["content"][0]["text"] or "CVE-2021-44228" in result["content"][0]["text"]
