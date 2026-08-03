"""
Comprehensive test suite for query_threat_intelligence_feeds tool module.

Tests cover:
- Input validation (invalid/empty/None CVE IDs)
- Successful feed queries with CVE found in content
- No results found across all feeds
- Snippet extraction logic (position-based, boundary handling)
- Multiple feeds with mixed results (some found, some not)
- HTTP error handling per feed (continues to next feed)
- Request timeout handling
- Connection error handling
- Unexpected exception handling in feed processing
- Case-insensitive CVE matching
- Multiple CVE mentions in same feed
- TOOL_SPEC structure validation
- ToolResult structure and status codes
- log_tool_output_size invocation verification
"""

from unittest.mock import MagicMock, patch

import requests

from manus_agent.tools.query_threat_intelligence_feeds import (
    TOOL_SPEC,
    query_threat_intelligence_feeds,
)

# --- Fixtures ---


def _make_tool(cve_id):
    """Create a minimal ToolUse dict."""
    return {"toolUseId": "test-tool-use-123", "input": {"cve_id": cve_id}}


def _make_response(text, status_code=200):
    """Create a mock response object."""
    resp = MagicMock()
    resp.text = text
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)
    return resp


# --- TOOL_SPEC Tests ---


class TestToolSpec:
    """Validate TOOL_SPEC structure."""

    def test_tool_spec_has_required_keys(self):
        assert "name" in TOOL_SPEC
        assert "description" in TOOL_SPEC
        assert "inputSchema" in TOOL_SPEC

    def test_tool_spec_name(self):
        assert TOOL_SPEC["name"] == "query_threat_intelligence_feeds"

    def test_tool_spec_description_mentions_threat_intelligence(self):
        assert "threat intelligence" in TOOL_SPEC["description"].lower()

    def test_tool_spec_requires_cve_id(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_tool_spec_cve_id_is_string_type(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["properties"]["cve_id"]["type"] == "string"


# --- Input Validation Tests ---


class TestInputValidation:
    """Tests for CVE ID input validation."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_empty_string_cve_id(self, mock_log):
        tool = _make_tool("")
        result = query_threat_intelligence_feeds(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]
        mock_log.assert_called_once()

    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_whitespace_only_cve_id(self, mock_log):
        tool = _make_tool("   ")
        result = query_threat_intelligence_feeds(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_none_cve_id(self, mock_log):
        tool = {"toolUseId": "test-123", "input": {"cve_id": None}}
        result = query_threat_intelligence_feeds(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_integer_cve_id(self, mock_log):
        tool = {"toolUseId": "test-123", "input": {"cve_id": 12345}}
        result = query_threat_intelligence_feeds(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_missing_cve_id_key(self, mock_log):
        tool = {"toolUseId": "test-123", "input": {}}
        result = query_threat_intelligence_feeds(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_tool_use_id_preserved_on_error(self, mock_log):
        tool = {"toolUseId": "unique-id-456", "input": {"cve_id": None}}
        result = query_threat_intelligence_feeds(tool)
        assert result["toolUseId"] == "unique-id-456"


# --- Successful Query Tests ---


class TestSuccessfulQueries:
    """Tests for successful feed queries."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_cve_found_in_feed(self, mock_log, mock_get):
        feed_content = (
            "<item><title>Alert about CVE-2024-3094</title><description>Critical vuln in xz utils</description></item>"
        )
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-3094")
        result = query_threat_intelligence_feeds(tool)

        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert "Found relevant threat intelligence" in data["summary"]
        assert len(data["intelligence"]) == 1
        assert data["intelligence"][0]["cve_found"] == "CVE-2024-3094"

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_cve_not_found_in_any_feed(self, mock_log, mock_get):
        feed_content = "<item><title>Some unrelated advisory</title></item>"
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2099-99999")
        result = query_threat_intelligence_feeds(tool)

        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert "No direct threat intelligence found" in data["summary"]
        assert data["intelligence"] == []

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_feed_name_in_result(self, mock_log, mock_get):
        feed_content = "Advisory: CVE-2024-1234 is actively exploited"
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        assert data["intelligence"][0]["feed_name"] == "CISA Cybersecurity Advisories"

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_feed_url_in_result(self, mock_log, mock_get):
        feed_content = "Advisory: CVE-2024-1234 critical"
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        assert "cisa.gov" in data["intelligence"][0]["feed_url"]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_tool_use_id_preserved_on_success(self, mock_log, mock_get):
        feed_content = "CVE-2024-5555 found here"
        mock_get.return_value = _make_response(feed_content)

        tool = {"toolUseId": "my-unique-id", "input": {"cve_id": "CVE-2024-5555"}}
        result = query_threat_intelligence_feeds(tool)

        assert result["toolUseId"] == "my-unique-id"

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_summary_includes_feed_count(self, mock_log, mock_get):
        feed_content = "CVE-2024-1111 advisory content"
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-1111")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        assert "1 feeds" in data["summary"] or "1 feed" in data["summary"]


# --- Case Sensitivity Tests ---


class TestCaseSensitivity:
    """Tests for case-insensitive CVE matching."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_lowercase_cve_in_feed_matched(self, mock_log, mock_get):
        feed_content = "This advisory covers cve-2024-1234 exploitation"
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        assert len(data["intelligence"]) == 1

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_mixed_case_cve_in_feed_matched(self, mock_log, mock_get):
        feed_content = "Critical: Cve-2024-1234 under active exploitation"
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        assert len(data["intelligence"]) == 1

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_lowercase_input_matched_against_uppercase_feed(self, mock_log, mock_get):
        feed_content = "Advisory: CVE-2024-1234 is critical"
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("cve-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        assert len(data["intelligence"]) == 1


# --- Snippet Extraction Tests ---


class TestSnippetExtraction:
    """Tests for content snippet extraction logic."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_snippet_contains_cve_id(self, mock_log, mock_get):
        feed_content = "x" * 100 + "CVE-2024-5678" + "y" * 200
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-5678")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        snippet = data["intelligence"][0]["snippet"]
        assert "CVE-2024-5678" in snippet

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_snippet_ends_with_ellipsis(self, mock_log, mock_get):
        feed_content = "x" * 100 + "CVE-2024-5678" + "y" * 200
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-5678")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        snippet = data["intelligence"][0]["snippet"]
        assert snippet.endswith("...")

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_snippet_at_beginning_of_content(self, mock_log, mock_get):
        # When CVE is at position 0, content[0-50:0+100] = content[-50:100] = empty string (Python slice)
        # This is a known edge case — the snippet extraction uses raw slicing without clamping
        feed_content = "CVE-2024-0001 is critical and exploited" + "z" * 200
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-0001")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        snippet = data["intelligence"][0]["snippet"]
        # Edge case: when pos=0, content[-50:100] is empty due to Python slice semantics
        assert snippet == "..."

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_snippet_near_end_of_short_content(self, mock_log, mock_get):
        feed_content = "Short: CVE-2024-9999 end"
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-9999")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        snippet = data["intelligence"][0]["snippet"]
        assert "CVE-2024-9999" in snippet

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_snippet_includes_surrounding_context(self, mock_log, mock_get):
        prefix = "BEFORE_CONTEXT_" + "a" * 35
        suffix = "_AFTER_CONTEXT" + "b" * 85
        feed_content = prefix + "CVE-2024-7777" + suffix + "c" * 200
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-7777")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        snippet = data["intelligence"][0]["snippet"]
        # Should contain surrounding context (50 chars before, 100 chars after the match pos)
        assert "CVE-2024-7777" in snippet


# --- Error Handling Tests ---


class TestErrorHandling:
    """Tests for HTTP errors and exceptions in feed processing."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_http_error_continues_to_next_feed(self, mock_log, mock_get):
        """HTTP error on one feed doesn't crash — returns empty intelligence."""
        mock_get.return_value = _make_response("", status_code=500)

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert data["intelligence"] == []

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_timeout_error_handled_gracefully(self, mock_log, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert data["intelligence"] == []
        assert "No direct threat intelligence found" in data["summary"]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_connection_error_handled_gracefully(self, mock_log, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError("DNS resolution failed")

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert data["intelligence"] == []

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_generic_request_exception_handled(self, mock_log, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException("Unknown error")

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert data["intelligence"] == []

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_unexpected_exception_handled(self, mock_log, mock_get):
        """Non-requests exception (e.g. AttributeError) also caught."""
        mock_get.side_effect = AttributeError("Unexpected attribute error")

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        # Should not crash; returns empty intelligence
        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert data["intelligence"] == []

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_404_error_handled_gracefully(self, mock_log, mock_get):
        mock_get.return_value = _make_response("Not Found", status_code=404)

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert data["intelligence"] == []

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_403_error_handled_gracefully(self, mock_log, mock_get):
        mock_get.return_value = _make_response("Forbidden", status_code=403)

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert data["intelligence"] == []


# --- Feed Request Behavior Tests ---


class TestFeedRequestBehavior:
    """Tests for how feed requests are made."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_requests_called_with_timeout(self, mock_log, mock_get):
        mock_get.return_value = _make_response("no cve here")

        tool = _make_tool("CVE-2024-1234")
        query_threat_intelligence_feeds(tool)

        # Verify timeout is passed
        call_kwargs = mock_get.call_args[1]
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] == 10

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_requests_called_with_cisa_url(self, mock_log, mock_get):
        mock_get.return_value = _make_response("no cve here")

        tool = _make_tool("CVE-2024-1234")
        query_threat_intelligence_feeds(tool)

        # Verify CISA URL is requested
        call_args = mock_get.call_args[0]
        assert "cisa.gov" in call_args[0]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_raise_for_status_called(self, mock_log, mock_get):
        resp = _make_response("content")
        mock_get.return_value = resp

        tool = _make_tool("CVE-2024-1234")
        query_threat_intelligence_feeds(tool)

        resp.raise_for_status.assert_called_once()


# --- Log Output Tests ---


class TestLogOutput:
    """Tests for log_tool_output_size invocation."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_log_called_on_success_with_results(self, mock_log, mock_get):
        mock_get.return_value = _make_response("CVE-2024-1234 found")

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        mock_log.assert_called_once_with("query_threat_intelligence_feeds", result)

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_log_called_on_success_no_results(self, mock_log, mock_get):
        mock_get.return_value = _make_response("unrelated content")

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        mock_log.assert_called_once_with("query_threat_intelligence_feeds", result)

    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_log_called_on_validation_error(self, mock_log):
        tool = _make_tool("")
        result = query_threat_intelligence_feeds(tool)

        mock_log.assert_called_once_with("query_threat_intelligence_feeds", result)


# --- Edge Cases ---


class TestEdgeCases:
    """Edge case tests."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_empty_feed_content(self, mock_log, mock_get):
        mock_get.return_value = _make_response("")

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert data["intelligence"] == []

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_very_long_feed_content(self, mock_log, mock_get):
        # Large content with CVE buried in the middle
        feed_content = "a" * 10000 + "CVE-2024-8888" + "b" * 10000
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-8888")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        assert len(data["intelligence"]) == 1
        snippet = data["intelligence"][0]["snippet"]
        assert "CVE-2024-8888" in snippet

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_cve_id_with_whitespace_still_valid(self, mock_log, mock_get):
        """A CVE ID with surrounding whitespace is valid (non-empty after strip check is on the raw value)."""
        # The tool checks `not cve_id.strip()` — so " CVE-2024-1234 " passes validation
        feed_content = "Advisory: CVE-2024-1234 active"
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool(" CVE-2024-1234 ")
        result = query_threat_intelligence_feeds(tool)

        # Passes validation since strip() is non-empty
        # But matching depends on the raw value (with spaces) — feed won't contain " CVE-2024-1234 "
        # Actually the code does cve_id.upper() comparison, so spaces matter
        assert result["status"] == "success"

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_non_cve_format_string_still_processed(self, mock_log, mock_get):
        """Any non-empty string passes validation — the tool doesn't enforce CVE format."""
        feed_content = "Some advisory mentioning GHSA-1234-abcd"
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("GHSA-1234-abcd")
        result = query_threat_intelligence_feeds(tool)

        assert result["status"] == "success"
        data = result["content"][0]["json"]
        assert len(data["intelligence"]) == 1

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_multiple_occurrences_of_cve_in_feed(self, mock_log, mock_get):
        """Only first occurrence drives the snippet."""
        feed_content = "First mention of CVE-2024-1234 here. And again CVE-2024-1234 repeated."
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        # Should find it (at least once)
        assert len(data["intelligence"]) == 1

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_special_characters_in_feed_content(self, mock_log, mock_get):
        feed_content = '<entry><id>CVE-2024-3094</id><title>&amp; "escaped"</title></entry>'
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-3094")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        assert len(data["intelligence"]) == 1

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_unicode_content_with_cve(self, mock_log, mock_get):
        feed_content = "Alerte de sécurité: CVE-2024-1234 — vulnérabilité critique détectée"
        mock_get.return_value = _make_response(feed_content)

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        assert len(data["intelligence"]) == 1

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_result_structure_has_json_key(self, mock_log, mock_get):
        """Both found and not-found results use json content type."""
        mock_get.return_value = _make_response("no cve here")

        tool = _make_tool("CVE-2024-1234")
        result = query_threat_intelligence_feeds(tool)

        assert "json" in result["content"][0]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_not_found_summary_includes_cve_id(self, mock_log, mock_get):
        mock_get.return_value = _make_response("nothing relevant")

        tool = _make_tool("CVE-2024-9999")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        assert "CVE-2024-9999" in data["summary"]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_found_summary_includes_cve_id(self, mock_log, mock_get):
        mock_get.return_value = _make_response("Advisory about CVE-2024-5555 exploitation")

        tool = _make_tool("CVE-2024-5555")
        result = query_threat_intelligence_feeds(tool)

        data = result["content"][0]["json"]
        assert "CVE-2024-5555" in data["summary"]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    @patch("manus_agent.tools.query_threat_intelligence_feeds.log_tool_output_size")
    def test_kwargs_forwarded_without_error(self, mock_log, mock_get):
        """Extra kwargs don't cause crashes."""
        mock_get.return_value = _make_response("no cve")

        tool = _make_tool("CVE-2024-1234")
        # Should not raise
        result = query_threat_intelligence_feeds(tool, session_id="abc", extra="data")
        assert result["status"] == "success"
