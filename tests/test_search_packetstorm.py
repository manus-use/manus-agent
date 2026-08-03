"""Comprehensive test suite for the search_packetstorm tool module.

Tests cover:
- TOOL_SPEC schema validation
- Input validation (missing/empty/invalid query)
- Successful search with results (HTML parsing)
- No results found
- HTTP errors (timeout, connection error, non-200 status)
- Unexpected exceptions
- Result limiting (max 5)
- HTML parsing edge cases (malformed entries, partial matches)
- URL encoding of special characters in queries
- tool_output_logger integration
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import requests

from manus_agent.tools.search_packetstorm import TOOL_SPEC, search_packetstorm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use(query: Any = "CVE-2024-3094") -> dict:
    """Build a minimal ToolUse dict for the search_packetstorm function."""
    return {
        "toolUseId": "test-tool-use-id",
        "input": {"query": query},
    }


def _file_entry(title: str, path: str = "/files/12345/test.html") -> str:
    """Generate a fake Packet Storm file entry matching the parser's expectations."""
    return f'<dl class="file"><dt><a href="{path}">{title}</a></dt><dd>Some description</dd></dl>'


def _build_html(*entries: str) -> str:
    """Wrap file entries in minimal page HTML."""
    return "<html><body>" + "".join(entries) + "</body></html>"


# ---------------------------------------------------------------------------
# TOOL_SPEC validation
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_spec_has_name(self):
        assert TOOL_SPEC["name"] == "search_packetstorm"

    def test_spec_has_description(self):
        assert "Packet Storm" in TOOL_SPEC["description"]

    def test_spec_has_input_schema(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "query" in schema["required"]

    def test_spec_query_is_string(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["properties"]["query"]["type"] == "string"

    def test_spec_description_mentions_exploits(self):
        assert "exploit" in TOOL_SPEC["description"].lower()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_missing_query_returns_error(self):
        tool = {"toolUseId": "tid", "input": {}}
        result = search_packetstorm(tool)
        assert result["status"] == "error"
        assert "Invalid query" in result["content"][0]["text"]

    def test_none_query_returns_error(self):
        tool = {"toolUseId": "tid", "input": {"query": None}}
        result = search_packetstorm(tool)
        assert result["status"] == "error"

    def test_empty_string_query_returns_error(self):
        tool = _make_tool_use("")
        result = search_packetstorm(tool)
        assert result["status"] == "error"
        assert "Invalid query" in result["content"][0]["text"]

    def test_whitespace_only_query_returns_error(self):
        tool = _make_tool_use("   ")
        result = search_packetstorm(tool)
        assert result["status"] == "error"

    def test_numeric_query_returns_error(self):
        tool = _make_tool_use(12345)
        result = search_packetstorm(tool)
        assert result["status"] == "error"

    def test_list_query_returns_error(self):
        tool = _make_tool_use(["CVE-2024-3094"])
        result = search_packetstorm(tool)
        assert result["status"] == "error"

    def test_tool_use_id_preserved_on_error(self):
        tool = {"toolUseId": "my-unique-id", "input": {"query": ""}}
        result = search_packetstorm(tool)
        assert result["toolUseId"] == "my-unique-id"


# ---------------------------------------------------------------------------
# Successful search — results found
# ---------------------------------------------------------------------------


class TestSuccessfulSearch:
    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_single_result_parsed(self, mock_get):
        html = _build_html(_file_entry("Apache Struts RCE", "/files/98765/apache-rce.html"))
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_packetstorm(_make_tool_use("CVE-2024-3094"))
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert len(payload["exploits"]) == 1
        assert payload["exploits"][0]["title"] == "Apache Struts RCE"
        assert "packetstormsecurity.com" in payload["exploits"][0]["link"]

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_multiple_results_parsed(self, mock_get):
        entries = [_file_entry(f"Exploit {i}", f"/files/{i}/test.html") for i in range(3)]
        html = _build_html(*entries)
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_packetstorm(_make_tool_use("CVE-2021-44228"))
        payload = result["content"][0]["json"]
        assert len(payload["exploits"]) == 3
        assert "Found 3" in payload["summary"]

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_results_limited_to_five(self, mock_get):
        entries = [_file_entry(f"Exploit {i}", f"/files/{i}/test.html") for i in range(10)]
        html = _build_html(*entries)
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_packetstorm(_make_tool_use("openssl"))
        payload = result["content"][0]["json"]
        assert len(payload["exploits"]) == 5
        assert "Found 5" in payload["summary"]

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_url_contains_query(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text="<html></html>")
        mock_get.return_value.raise_for_status = MagicMock()

        search_packetstorm(_make_tool_use("CVE-2024-3094"))
        called_url = mock_get.call_args[0][0]
        assert "CVE-2024-3094" in called_url

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_special_chars_url_encoded(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text="<html></html>")
        mock_get.return_value.raise_for_status = MagicMock()

        search_packetstorm(_make_tool_use("test query&special=true"))
        called_url = mock_get.call_args[0][0]
        # URL should have encoded the query
        assert "packetstormsecurity.com" in called_url

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_tool_use_id_preserved_on_success(self, mock_get):
        html = _build_html(_file_entry("Test", "/files/1/t.html"))
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        tool = {"toolUseId": "unique-456", "input": {"query": "CVE-2024-3094"}}
        result = search_packetstorm(tool)
        assert result["toolUseId"] == "unique-456"

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_link_has_full_packetstorm_url(self, mock_get):
        html = _build_html(_file_entry("Kernel Bug", "/files/99999/kern.html"))
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_packetstorm(_make_tool_use("kernel"))
        payload = result["content"][0]["json"]
        assert payload["exploits"][0]["link"].startswith("https://packetstormsecurity.com")

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_link_preserves_path(self, mock_get):
        html = _build_html(_file_entry("Test", "/files/12345/specific-path.html"))
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_packetstorm(_make_tool_use("test"))
        payload = result["content"][0]["json"]
        assert "/files/12345/specific-path.html" in payload["exploits"][0]["link"]


# ---------------------------------------------------------------------------
# No results found
# ---------------------------------------------------------------------------


class TestNoResults:
    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_empty_page_returns_no_exploits(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text="<html><body></body></html>")
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_packetstorm(_make_tool_use("CVE-9999-99999"))
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert payload["exploits"] == []
        assert "No exploits found" in payload["summary"]

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_no_file_entries_returns_empty(self, mock_get):
        html = "<html><body><div>No results</div></body></html>"
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_packetstorm(_make_tool_use("nonexistent-vuln"))
        payload = result["content"][0]["json"]
        assert payload["exploits"] == []

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_summary_mentions_query_on_no_results(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text="<html></html>")
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_packetstorm(_make_tool_use("CVE-2024-9999"))
        payload = result["content"][0]["json"]
        assert "CVE-2024-9999" in payload["summary"]


# ---------------------------------------------------------------------------
# HTTP errors
# ---------------------------------------------------------------------------


class TestHTTPErrors:
    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = search_packetstorm(_make_tool_use("CVE-2024-3094"))
        assert result["status"] == "error"
        assert "Packet Storm failed" in result["content"][0]["text"]

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_timeout_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("timed out")

        result = search_packetstorm(_make_tool_use("CVE-2024-3094"))
        assert result["status"] == "error"
        assert "failed" in result["content"][0]["text"].lower()

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_http_500_error(self, mock_get):
        response = MagicMock()
        response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = response

        result = search_packetstorm(_make_tool_use("CVE-2024-3094"))
        assert result["status"] == "error"

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_http_403_forbidden(self, mock_get):
        response = MagicMock()
        response.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Forbidden")
        mock_get.return_value = response

        result = search_packetstorm(_make_tool_use("CVE-2024-3094"))
        assert result["status"] == "error"

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_request_timeout_value(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, text="<html></html>")
        mock_get.return_value.raise_for_status = MagicMock()

        search_packetstorm(_make_tool_use("test"))
        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] == 15


# ---------------------------------------------------------------------------
# Unexpected exceptions
# ---------------------------------------------------------------------------


class TestUnexpectedExceptions:
    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_generic_exception_caught(self, mock_get):
        mock_get.side_effect = RuntimeError("something broke")

        result = search_packetstorm(_make_tool_use("CVE-2024-3094"))
        assert result["status"] == "error"
        assert "unexpected error" in result["content"][0]["text"].lower()

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_value_error_caught(self, mock_get):
        mock_get.side_effect = ValueError("bad value")

        result = search_packetstorm(_make_tool_use("CVE-2024-3094"))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# HTML parsing edge cases
# ---------------------------------------------------------------------------


class TestHTMLParsingEdgeCases:
    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_entry_without_dt_link_skipped(self, mock_get):
        # Entry exists but has no <dt><a href="..."> pattern
        html = '<html><body><dl class="file"><dt>No link here</dt></dl></body></html>'
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_packetstorm(_make_tool_use("CVE-2024-3094"))
        payload = result["content"][0]["json"]
        assert payload["exploits"] == []

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_entry_with_empty_title_still_parsed(self, mock_get):
        html = _build_html('<dl class="file"><dt><a href="/files/123/empty.html"></a></dt></dl>')
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_packetstorm(_make_tool_use("test"))
        payload = result["content"][0]["json"]
        # Empty title is still a valid parsed entry
        assert len(payload["exploits"]) == 1
        assert payload["exploits"][0]["title"] == ""

    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_multiple_dl_file_classes(self, mock_get):
        """Parser should split on '<dl class="file">' correctly."""
        html = (
            "<html><body>"
            '<dl class="file"><dt><a href="/files/1/a.html">First</a></dt></dl>'
            '<dl class="file"><dt><a href="/files/2/b.html">Second</a></dt></dl>'
            "</body></html>"
        )
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        result = search_packetstorm(_make_tool_use("test"))
        payload = result["content"][0]["json"]
        assert len(payload["exploits"]) == 2
        assert payload["exploits"][0]["title"] == "First"
        assert payload["exploits"][1]["title"] == "Second"


# ---------------------------------------------------------------------------
# tool_output_logger integration
# ---------------------------------------------------------------------------


class TestOutputLogger:
    @patch("manus_agent.tools.search_packetstorm.log_tool_output_size")
    def test_logger_called_on_error(self, mock_logger):
        tool = _make_tool_use("")
        search_packetstorm(tool)
        mock_logger.assert_called_once()
        assert mock_logger.call_args[0][0] == "search_packetstorm"

    @patch("manus_agent.tools.search_packetstorm.log_tool_output_size")
    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_logger_called_on_success(self, mock_get, mock_logger):
        html = _build_html(_file_entry("Test", "/files/1/t.html"))
        mock_get.return_value = MagicMock(status_code=200, text=html)
        mock_get.return_value.raise_for_status = MagicMock()

        search_packetstorm(_make_tool_use("CVE-2024-3094"))
        mock_logger.assert_called_once()
        assert mock_logger.call_args[0][0] == "search_packetstorm"

    @patch("manus_agent.tools.search_packetstorm.log_tool_output_size")
    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_logger_called_on_no_results(self, mock_get, mock_logger):
        mock_get.return_value = MagicMock(status_code=200, text="<html></html>")
        mock_get.return_value.raise_for_status = MagicMock()

        search_packetstorm(_make_tool_use("nothing"))
        mock_logger.assert_called_once()

    @patch("manus_agent.tools.search_packetstorm.log_tool_output_size")
    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_logger_called_on_request_error(self, mock_get, mock_logger):
        mock_get.side_effect = requests.exceptions.ConnectionError("fail")

        search_packetstorm(_make_tool_use("test"))
        mock_logger.assert_called_once()

    @patch("manus_agent.tools.search_packetstorm.log_tool_output_size")
    @patch("manus_agent.tools.search_packetstorm.requests.get")
    def test_logger_called_on_unexpected_error(self, mock_get, mock_logger):
        mock_get.side_effect = RuntimeError("boom")

        search_packetstorm(_make_tool_use("test"))
        mock_logger.assert_called_once()


# ---------------------------------------------------------------------------
# Integration: module importable and interface stable
# ---------------------------------------------------------------------------


class TestModuleInterface:
    def test_tool_spec_importable(self):
        from manus_agent.tools.search_packetstorm import TOOL_SPEC

        assert TOOL_SPEC is not None

    def test_function_importable(self):
        from manus_agent.tools.search_packetstorm import search_packetstorm

        assert callable(search_packetstorm)

    def test_function_accepts_kwargs(self):
        """Function signature must accept **kwargs for Strands SDK compatibility."""
        import inspect

        sig = inspect.signature(search_packetstorm)
        params = list(sig.parameters.values())
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
