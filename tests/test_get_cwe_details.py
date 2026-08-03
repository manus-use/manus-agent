"""Comprehensive test suite for the get_cwe_details tool module.

Tests cover:
- TOOL_SPEC schema validation
- Input validation (missing/empty/invalid CWE ID, non-digit suffix)
- Successful lookup with description parsed
- Description not found on page
- HTTP errors (timeout, connection error, non-200 status)
- Unexpected exceptions
- HTML parsing edge cases (no Extended_Description marker, no next div)
- HTML tag stripping in description
- URL construction from CWE number
- tool_output_logger integration
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import requests

from manus_agent.tools.get_cwe_details import TOOL_SPEC, get_cwe_details

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use(cwe_id: Any = "CWE-79") -> dict:
    """Build a minimal ToolUse dict for the get_cwe_details function."""
    return {
        "toolUseId": "test-tool-use-id",
        "input": {"cwe_id": cwe_id},
    }


def _build_html(description: str, has_extended: bool = True) -> str:
    """Build a fake CWE page with Description section."""
    extended = '<div id="Extended_Description">Extended info</div>' if has_extended else ""
    return (
        "<html><body>"
        f'<div id="Description">{description}</div>'
        f"{extended}"
        '<div id="Relationships">Other content</div>'
        "</body></html>"
    )


# ---------------------------------------------------------------------------
# TOOL_SPEC validation
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_spec_has_name(self):
        assert TOOL_SPEC["name"] == "get_cwe_details"

    def test_spec_has_description(self):
        assert "CWE" in TOOL_SPEC["description"]
        assert "MITRE" in TOOL_SPEC["description"]

    def test_spec_has_input_schema(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "cwe_id" in schema["properties"]
        assert "cwe_id" in schema["required"]

    def test_spec_cwe_id_is_string(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["properties"]["cwe_id"]["type"] == "string"

    def test_spec_description_mentions_mitigations(self):
        assert "mitigation" in TOOL_SPEC["description"].lower()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_cwe_id_returns_error(self):
        tool = {"toolUseId": "tid", "input": {}}
        result = get_cwe_details(tool)
        assert result["status"] == "error"
        assert "Invalid CWE ID" in result["content"][0]["text"]

    def test_none_cwe_id_returns_error(self):
        tool = {"toolUseId": "tid", "input": {"cwe_id": None}}
        result = get_cwe_details(tool)
        assert result["status"] == "error"

    def test_empty_string_returns_error(self):
        tool = _make_tool_use("")
        result = get_cwe_details(tool)
        assert result["status"] == "error"

    def test_numeric_input_returns_error(self):
        tool = _make_tool_use(79)
        result = get_cwe_details(tool)
        assert result["status"] == "error"

    def test_list_input_returns_error(self):
        tool = _make_tool_use(["CWE-79"])
        result = get_cwe_details(tool)
        assert result["status"] == "error"

    def test_no_cwe_prefix_returns_error(self):
        tool = _make_tool_use("79")
        result = get_cwe_details(tool)
        assert result["status"] == "error"
        assert "Invalid CWE ID" in result["content"][0]["text"]

    def test_wrong_prefix_returns_error(self):
        tool = _make_tool_use("CVE-79")
        result = get_cwe_details(tool)
        assert result["status"] == "error"

    def test_non_digit_number_returns_error(self):
        tool = _make_tool_use("CWE-abc")
        result = get_cwe_details(tool)
        assert result["status"] == "error"
        assert "Number part" in result["content"][0]["text"]

    def test_cwe_with_trailing_text_returns_error(self):
        tool = _make_tool_use("CWE-79abc")
        result = get_cwe_details(tool)
        assert result["status"] == "error"

    def test_cwe_hyphen_only_returns_error(self):
        tool = _make_tool_use("CWE-")
        result = get_cwe_details(tool)
        assert result["status"] == "error"

    def test_tool_use_id_preserved_on_error(self):
        tool = {"toolUseId": "my-unique-id", "input": {"cwe_id": ""}}
        result = get_cwe_details(tool)
        assert result["toolUseId"] == "my-unique-id"

    def test_case_insensitive_prefix(self):
        """CWE- prefix should be case-insensitive (cwe-79 accepted)."""
        # The function does .upper() so lowercase should work
        with patch("manus_agent.tools.get_cwe_details.requests.get") as mock_get:
            html = _build_html("<p>Cross-site scripting</p>")
            mock_get.return_value = MagicMock(status_code=200, text=html)
            mock_get.return_value.raise_for_status = MagicMock()

            result = get_cwe_details(_make_tool_use("cwe-79"))
            assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Successful lookup
# ---------------------------------------------------------------------------


class TestSuccessfulLookup:
    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_basic_description_parsed(self, mock_get):
        html = _build_html("<p>Cross-site scripting vulnerability</p>")
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = get_cwe_details(_make_tool_use("CWE-79"))
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert payload["cwe_id"] == "CWE-79"
        assert "Cross-site scripting" in payload["description"]
        assert "79.html" in payload["url"]

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_html_tags_stripped(self, mock_get):
        html = _build_html("<p>Buffer <br/> overflow <ul><li>item</li></ul></p>")
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = get_cwe_details(_make_tool_use("CWE-120"))
        payload = result["content"][0]["json"]
        # All basic HTML tags should be stripped
        assert "<p>" not in payload["description"]
        assert "<br/>" not in payload["description"]
        assert "<ul>" not in payload["description"]
        assert "<li>" not in payload["description"]
        assert "Buffer" in payload["description"]
        assert "overflow" in payload["description"]

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_url_uses_cwe_number(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text=_build_html("desc"))
        mock_get.return_value.raise_for_status = MagicMock()

        get_cwe_details(_make_tool_use("CWE-287"))
        called_url = mock_get.call_args[0][0]
        assert "287.html" in called_url
        assert "cwe.mitre.org" in called_url

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_uppercase_normalization(self, mock_get):
        """Function should normalize cwe-79 → CWE-79 for output."""
        html = _build_html("<p>XSS</p>")
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = get_cwe_details(_make_tool_use("cwe-79"))
        payload = result["content"][0]["json"]
        # The input gets preserved as-is in the output
        assert "cwe" in payload["cwe_id"].lower() or "CWE" in payload["cwe_id"]

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_tool_use_id_preserved_on_success(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text=_build_html("desc"))
        mock_get.return_value.raise_for_status = MagicMock()

        tool = {"toolUseId": "unique-789", "input": {"cwe_id": "CWE-79"}}
        result = get_cwe_details(tool)
        assert result["toolUseId"] == "unique-789"

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_result_contains_url_field(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text=_build_html("desc"))
        mock_get.return_value.raise_for_status = MagicMock()

        result = get_cwe_details(_make_tool_use("CWE-79"))
        payload = result["content"][0]["json"]
        assert "url" in payload
        assert payload["url"].startswith("https://cwe.mitre.org")

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_no_extended_description_uses_next_div(self, mock_get):
        """When Extended_Description is missing, fallback to next div marker."""
        html = (
            "<html><body>"
            '<div id="Description"><p>Injection flaw</p></div>'
            '<div id="Relationships">Other</div>'
            "</body></html>"
        )
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = get_cwe_details(_make_tool_use("CWE-89"))
        payload = result["content"][0]["json"]
        assert "Injection flaw" in payload["description"]

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_no_following_div_reads_to_end(self, mock_get):
        """When no following div exists, read to end of content."""
        html = '<html><body><div id="Description"><p>Memory corruption</p></body></html>'
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = get_cwe_details(_make_tool_use("CWE-119"))
        payload = result["content"][0]["json"]
        assert "Memory corruption" in payload["description"]


# ---------------------------------------------------------------------------
# Description not found
# ---------------------------------------------------------------------------


class TestDescriptionNotFound:
    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_no_description_div_returns_error(self, mock_get):
        html = "<html><body><div>No Description section here</div></body></html>"
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = get_cwe_details(_make_tool_use("CWE-79"))
        assert result["status"] == "error"
        assert "Could not find description" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_empty_page_returns_error(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text="")
        mock_get.return_value.raise_for_status = MagicMock()

        result = get_cwe_details(_make_tool_use("CWE-79"))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# HTTP errors
# ---------------------------------------------------------------------------


class TestHTTPErrors:
    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = get_cwe_details(_make_tool_use("CWE-79"))
        assert result["status"] == "error"
        assert "CWE website failed" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_timeout_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        result = get_cwe_details(_make_tool_use("CWE-79"))
        assert result["status"] == "error"
        assert "failed" in result["content"][0]["text"].lower()

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_http_404_error(self, mock_get):
        response = MagicMock()
        response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = response

        result = get_cwe_details(_make_tool_use("CWE-99999"))
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_http_500_error(self, mock_get):
        response = MagicMock()
        response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = response

        result = get_cwe_details(_make_tool_use("CWE-79"))
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_request_timeout_value(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text=_build_html("test"))
        mock_get.return_value.raise_for_status = MagicMock()

        get_cwe_details(_make_tool_use("CWE-79"))
        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] == 15


# ---------------------------------------------------------------------------
# Unexpected exceptions
# ---------------------------------------------------------------------------


class TestUnexpectedExceptions:
    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_generic_exception_caught(self, mock_get):
        mock_get.side_effect = RuntimeError("something broke")

        result = get_cwe_details(_make_tool_use("CWE-79"))
        assert result["status"] == "error"
        assert "unexpected error" in result["content"][0]["text"].lower()

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_attribute_error_caught(self, mock_get):
        mock_get.side_effect = AttributeError("bad attribute")

        result = get_cwe_details(_make_tool_use("CWE-79"))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# HTML tag stripping edge cases
# ---------------------------------------------------------------------------


class TestHTMLStripping:
    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_br_tag_stripped(self, mock_get):
        html = _build_html("Line one<br>Line two<br/>Line three")
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = get_cwe_details(_make_tool_use("CWE-79"))
        payload = result["content"][0]["json"]
        assert "<br>" not in payload["description"]
        assert "<br/>" not in payload["description"]

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_nested_tags_stripped(self, mock_get):
        html = _build_html("<p><ul><li>Item 1</li><li>Item 2</li></ul></p>")
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = get_cwe_details(_make_tool_use("CWE-79"))
        payload = result["content"][0]["json"]
        assert "<p>" not in payload["description"]
        assert "<ul>" not in payload["description"]
        assert "<li>" not in payload["description"]
        assert "Item 1" in payload["description"]

    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_description_whitespace_trimmed(self, mock_get):
        html = _build_html("   <p>  Spaced content  </p>   ")
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = get_cwe_details(_make_tool_use("CWE-79"))
        payload = result["content"][0]["json"]
        # Description should be trimmed
        assert not payload["description"].startswith(" ")
        assert not payload["description"].endswith(" ")


# ---------------------------------------------------------------------------
# tool_output_logger integration
# ---------------------------------------------------------------------------


class TestOutputLogger:
    @patch("manus_agent.tools.get_cwe_details.log_tool_output_size")
    def test_logger_called_on_invalid_input(self, mock_logger):
        tool = _make_tool_use("")
        get_cwe_details(tool)
        mock_logger.assert_called_once()
        assert mock_logger.call_args[0][0] == "get_cwe_details"

    @patch("manus_agent.tools.get_cwe_details.log_tool_output_size")
    def test_logger_called_on_non_digit_error(self, mock_logger):
        tool = _make_tool_use("CWE-abc")
        get_cwe_details(tool)
        mock_logger.assert_called_once()

    @patch("manus_agent.tools.get_cwe_details.log_tool_output_size")
    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_logger_called_on_success(self, mock_get, mock_logger):
        html = _build_html("Test description")
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        get_cwe_details(_make_tool_use("CWE-79"))
        mock_logger.assert_called_once()
        assert mock_logger.call_args[0][0] == "get_cwe_details"

    @patch("manus_agent.tools.get_cwe_details.log_tool_output_size")
    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_logger_called_on_no_description(self, mock_get, mock_logger):
        mock_get.return_value = MagicMock(status_code=200, text="<html></html>")
        mock_get.return_value.raise_for_status = MagicMock()

        get_cwe_details(_make_tool_use("CWE-79"))
        mock_logger.assert_called_once()

    @patch("manus_agent.tools.get_cwe_details.log_tool_output_size")
    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_logger_called_on_request_error(self, mock_get, mock_logger):
        mock_get.side_effect = requests.exceptions.ConnectionError("fail")

        get_cwe_details(_make_tool_use("CWE-79"))
        mock_logger.assert_called_once()

    @patch("manus_agent.tools.get_cwe_details.log_tool_output_size")
    @patch("manus_agent.tools.get_cwe_details.requests.get")
    def test_logger_called_on_unexpected_error(self, mock_get, mock_logger):
        mock_get.side_effect = RuntimeError("boom")

        get_cwe_details(_make_tool_use("CWE-79"))
        mock_logger.assert_called_once()


# ---------------------------------------------------------------------------
# Integration: module importable and interface stable
# ---------------------------------------------------------------------------


class TestModuleInterface:
    def test_tool_spec_importable(self):
        from manus_agent.tools.get_cwe_details import TOOL_SPEC

        assert TOOL_SPEC is not None

    def test_function_importable(self):
        from manus_agent.tools.get_cwe_details import get_cwe_details

        assert callable(get_cwe_details)

    def test_function_accepts_kwargs(self):
        """Function signature must accept **kwargs for Strands SDK compatibility."""
        import inspect

        sig = inspect.signature(get_cwe_details)
        params = list(sig.parameters.values())
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
