"""Comprehensive test suite for manus_agent.tools.http_request module.

Tests cover:
- Delegation to upstream strands_tools.http_request
- Per-item truncation (Phase 1)
- Proportional total truncation (Phase 2)
- Environment variable overrides for limits
- Edge cases: empty content, non-text items, mixed content
- Output size logging (before/after)
- tool_output_logger integration
- Exception handling in truncation path
- Exception handling in upstream call
"""

import os
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use(url: str = "https://example.com", method: str = "GET") -> dict:
    """Build a minimal ToolUse dict."""
    return {
        "toolUseId": "test-id-1",
        "input": {"url": url, "method": method},
    }


def _text_item(text: str) -> dict:
    """Build a content item with text."""
    return {"text": text}


def _upstream_result(content: list | None = None, **extra) -> dict:
    """Build a result dict as returned by the upstream http_request."""
    result = {"status": "success", "content": content or []}
    result.update(extra)
    return result


# ---------------------------------------------------------------------------
# Module-level constants / imports
# ---------------------------------------------------------------------------


class TestHttpRequestDelegation:
    """Verify that our wrapper delegates to the upstream tool."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_delegates_tool_and_kwargs(self, mock_logger, mock_upstream):
        """The wrapper passes tool and kwargs through to upstream."""
        from manus_agent.tools.http_request import http_request

        tool = _make_tool_use()
        mock_upstream.return_value = _upstream_result([_text_item("ok")])

        http_request(tool, system_prompt="sp", messages=[])

        mock_upstream.assert_called_once_with(tool, system_prompt="sp", messages=[])

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_returns_upstream_result_when_no_truncation(self, mock_logger, mock_upstream):
        """When content is below limits, original result is returned."""
        from manus_agent.tools.http_request import http_request

        original = _upstream_result([_text_item("short")])
        mock_upstream.return_value = original

        result = http_request(_make_tool_use())
        # No truncation → same object (identity, not just equality)
        assert result is original

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_no_kwargs(self, mock_logger, mock_upstream):
        """Works when no kwargs are provided."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("x")])
        result = http_request(_make_tool_use())
        assert result["status"] == "success"


class TestPhase1PerItemTruncation:
    """Phase 1: individual items exceeding MAX_OUTPUT_CHARS are truncated."""

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 10)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_truncates_single_oversized_item(self, mock_logger, mock_upstream):
        """A single text item longer than MAX_OUTPUT_CHARS is truncated."""
        from manus_agent.tools.http_request import http_request

        long_text = "A" * 25
        mock_upstream.return_value = _upstream_result([_text_item(long_text)])

        result = http_request(_make_tool_use())
        text = result["content"][0]["text"]

        assert text.startswith("A" * 10)
        assert "[truncated: 15 chars removed]" in text

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 10)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_no_truncation_when_under_limit(self, mock_logger, mock_upstream):
        """Items at or under the per-item limit are not truncated."""
        from manus_agent.tools.http_request import http_request

        exact_text = "B" * 10
        mock_upstream.return_value = _upstream_result([_text_item(exact_text)])

        result = http_request(_make_tool_use())
        assert result["content"][0]["text"] == exact_text

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 5)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_truncates_multiple_items_independently(self, mock_logger, mock_upstream):
        """Each item is truncated independently in Phase 1."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result(
            [
                _text_item("X" * 20),
                _text_item("Y" * 3),  # under limit
                _text_item("Z" * 15),
            ]
        )

        result = http_request(_make_tool_use())
        items = result["content"]

        assert items[0]["text"].startswith("X" * 5)
        assert "[truncated: 15 chars removed]" in items[0]["text"]

        assert items[1]["text"] == "Y" * 3  # untouched

        assert items[2]["text"].startswith("Z" * 5)
        assert "[truncated: 10 chars removed]" in items[2]["text"]

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 10)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_preserves_extra_keys_in_text_item(self, mock_logger, mock_upstream):
        """Non-text keys in the item dict are preserved after truncation."""
        from manus_agent.tools.http_request import http_request

        item = {"text": "C" * 20, "type": "response", "encoding": "utf-8"}
        mock_upstream.return_value = _upstream_result([item])

        result = http_request(_make_tool_use())
        out_item = result["content"][0]

        assert out_item["type"] == "response"
        assert out_item["encoding"] == "utf-8"
        assert "[truncated:" in out_item["text"]


class TestPhase2TotalTruncation:
    """Phase 2: proportional reduction when total exceeds MAX_TOTAL_OUTPUT_CHARS."""

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 20)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_proportional_reduction(self, mock_logger, mock_upstream):
        """When total exceeds the limit, items are proportionally reduced."""
        from manus_agent.tools.http_request import http_request

        # Two items: 30 chars each = 60 total, limit is 20
        mock_upstream.return_value = _upstream_result(
            [
                _text_item("A" * 30),
                _text_item("B" * 30),
            ]
        )

        result = http_request(_make_tool_use())
        items = result["content"]

        # Both should be truncated; ratio = 20/60 = 0.333...
        for item in items:
            assert "[truncated:" in item["text"]

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 50)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_phase2_not_triggered_when_under_total(self, mock_logger, mock_upstream):
        """Phase 2 does nothing when total is under MAX_TOTAL_OUTPUT_CHARS."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result(
            [
                _text_item("A" * 20),
                _text_item("B" * 20),
            ]
        )

        result = http_request(_make_tool_use())
        items = result["content"]

        # 40 total < 50 limit → no Phase 2 truncation
        assert items[0]["text"] == "A" * 20
        assert items[1]["text"] == "B" * 20

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 20)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 15)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_both_phases_triggered(self, mock_logger, mock_upstream):
        """Phase 1 truncates per-item, then Phase 2 further reduces total."""
        from manus_agent.tools.http_request import http_request

        # Items: 50 chars each. Phase 1 limit 20 → each becomes ~20 + tag.
        # Phase 2 limit 15 → further proportional cut.
        mock_upstream.return_value = _upstream_result(
            [
                _text_item("X" * 50),
                _text_item("Y" * 50),
            ]
        )

        result = http_request(_make_tool_use())
        items = result["content"]

        # Both items should be truncated
        for item in items:
            assert "[truncated:" in item["text"]

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 10)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_phase2_preserves_non_text_items(self, mock_logger, mock_upstream):
        """Non-text items pass through Phase 2 unmodified."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result(
            [
                _text_item("A" * 30),
                {"image": "data:image/png;base64,abc"},
                _text_item("B" * 30),
            ]
        )

        result = http_request(_make_tool_use())
        items = result["content"]

        assert items[1] == {"image": "data:image/png;base64,abc"}

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 5)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_phase2_item_limit_floors_at_zero(self, mock_logger, mock_upstream):
        """When ratio makes item_limit=0, truncation still works (max(int(...), 0))."""
        from manus_agent.tools.http_request import http_request

        # Very tiny total limit vs large items → ratio near 0
        mock_upstream.return_value = _upstream_result(
            [
                _text_item("A" * 1000),
                _text_item("B" * 1000),
            ]
        )

        result = http_request(_make_tool_use())
        items = result["content"]

        for item in items:
            assert "[truncated:" in item["text"]


class TestNonTextContent:
    """Edge cases with non-text content items."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_non_dict_items_pass_through(self, mock_logger, mock_upstream):
        """Non-dict items in the content list are passed through unchanged."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result(["raw_string", 42, None])

        result = http_request(_make_tool_use())
        assert result["content"] == ["raw_string", 42, None]

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_dict_without_text_key(self, mock_logger, mock_upstream):
        """Dict items without a 'text' key pass through Phase 1."""
        from manus_agent.tools.http_request import http_request

        item = {"json": {"key": "value"}}
        mock_upstream.return_value = _upstream_result([item])

        result = http_request(_make_tool_use())
        assert result["content"][0] == item

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_dict_with_non_string_text(self, mock_logger, mock_upstream):
        """Dict items with non-string 'text' values pass through."""
        from manus_agent.tools.http_request import http_request

        item = {"text": 12345}  # text is int, not str
        mock_upstream.return_value = _upstream_result([item])

        result = http_request(_make_tool_use())
        assert result["content"][0] == {"text": 12345}

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_empty_content_list(self, mock_logger, mock_upstream):
        """Empty content list produces no truncation."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([])

        result = http_request(_make_tool_use())
        assert result["content"] == []

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_missing_content_key(self, mock_logger, mock_upstream):
        """Result without 'content' key defaults to empty list."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = {"status": "success"}

        result = http_request(_make_tool_use())
        assert result["status"] == "success"

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_mixed_text_and_non_text_items(self, mock_logger, mock_upstream):
        """Mixed content with text and non-text items is handled correctly."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result(
            [
                _text_item("hello"),
                {"image": "png-data"},
                _text_item("world"),
                42,
            ]
        )

        result = http_request(_make_tool_use())
        assert len(result["content"]) == 4


class TestEmptyTextItems:
    """Edge cases with empty or zero-length text."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_empty_string_text(self, mock_logger, mock_upstream):
        """Items with empty text strings pass through unchanged."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("")])

        result = http_request(_make_tool_use())
        assert result["content"][0]["text"] == ""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_total_before_zero_skips_print(self, mock_logger, mock_upstream, capsys):
        """When total text length is 0, the 'before truncation' print is skipped."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("")])

        http_request(_make_tool_use())
        captured = capsys.readouterr()
        assert "Output size before truncation" not in captured.out


class TestOutputSizeLogging:
    """Verify the print statements for output size logging."""

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_prints_before_truncation_size(self, mock_logger, mock_upstream, capsys):
        """Prints output size before truncation when there is text content."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("hello")])

        http_request(_make_tool_use())
        captured = capsys.readouterr()
        assert "[http_request] Output size before truncation: 5 chars" in captured.out

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 3)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_prints_after_truncation_size(self, mock_logger, mock_upstream, capsys):
        """Prints output size after truncation when text content exists."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("ABCDEF")])

        http_request(_make_tool_use())
        captured = capsys.readouterr()
        assert "[http_request] Output size before truncation: 6 chars" in captured.out
        assert "[http_request] Output size after truncation:" in captured.out

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_no_after_print_when_zero_text(self, mock_logger, mock_upstream, capsys):
        """No 'after' print when there is no text content left."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([{"image": "data"}])

        http_request(_make_tool_use())
        captured = capsys.readouterr()
        assert "Output size after truncation" not in captured.out


class TestToolOutputLoggerIntegration:
    """Verify log_tool_output_size is called."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_logger_called_with_result(self, mock_logger, mock_upstream):
        """log_tool_output_size is called with tool name and final result."""
        from manus_agent.tools.http_request import http_request

        upstream_result = _upstream_result([_text_item("data")])
        mock_upstream.return_value = upstream_result

        result = http_request(_make_tool_use())

        mock_logger.assert_called_once_with("http_request", result)

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 5)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_logger_called_with_truncated_result(self, mock_logger, mock_upstream):
        """When truncation happens, logger is called with the truncated result."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("A" * 100)])

        http_request(_make_tool_use())

        mock_logger.assert_called_once()
        logged_result = mock_logger.call_args[0][1]
        assert "[truncated:" in logged_result["content"][0]["text"]


class TestExceptionHandling:
    """Exception handling in the truncation/logging path."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_truncation_exception_returns_original(self, mock_logger, mock_upstream, capsys):
        """If truncation logic raises, the original upstream result is returned."""
        from manus_agent.tools.http_request import http_request

        original = _upstream_result([_text_item("data")])
        mock_upstream.return_value = original

        # Make result.get raise by returning a non-dict
        # We need to simulate an exception in the try block
        mock_upstream.return_value = MagicMock()
        mock_upstream.return_value.get.side_effect = TypeError("boom")

        result = http_request(_make_tool_use())
        captured = capsys.readouterr()

        assert "[http_request] Truncation/logging failed:" in captured.out
        # Returns the original result despite the error
        assert result is mock_upstream.return_value

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_logger_exception_still_returns_result(self, mock_logger, mock_upstream, capsys):
        """If log_tool_output_size raises, the result is still returned."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("data")])
        mock_logger.side_effect = RuntimeError("logging broke")

        http_request(_make_tool_use())
        captured = capsys.readouterr()

        assert "[http_request] Truncation/logging failed:" in captured.out

    @patch("manus_agent.tools.http_request._http_request")
    def test_upstream_exception_propagates(self, mock_upstream):
        """Exceptions from the upstream call propagate (not caught by wrapper)."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.side_effect = ConnectionError("network down")

        with pytest.raises(ConnectionError, match="network down"):
            http_request(_make_tool_use())


class TestEnvironmentVariableOverrides:
    """Verify that environment variables control truncation limits."""

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 8)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_custom_per_item_limit(self, mock_logger, mock_upstream):
        """HTTP_REQUEST_MAX_OUTPUT_CHARS env var controls per-item limit."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("A" * 20)])

        result = http_request(_make_tool_use())
        text = result["content"][0]["text"]

        assert text.startswith("A" * 8)
        assert "[truncated: 12 chars removed]" in text

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 10)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_custom_total_limit(self, mock_logger, mock_upstream):
        """HTTP_REQUEST_MAX_TOTAL_OUTPUT_CHARS env var controls total limit."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result(
            [
                _text_item("A" * 30),
                _text_item("B" * 30),
            ]
        )

        result = http_request(_make_tool_use())
        for item in result["content"]:
            assert "[truncated:" in item["text"]


class TestResultImmutability:
    """Verify original result is not mutated when truncation occurs."""

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 5)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_original_result_not_mutated(self, mock_logger, mock_upstream):
        """Truncation creates a new result dict, not mutating the original."""
        from manus_agent.tools.http_request import http_request

        original_content = [_text_item("A" * 50)]
        original_result = _upstream_result(original_content)
        mock_upstream.return_value = original_result

        result = http_request(_make_tool_use())

        # Original should be unchanged
        assert original_content[0]["text"] == "A" * 50
        # Result should be a new dict
        assert result is not original_result
        assert "[truncated:" in result["content"][0]["text"]

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_returns_same_object_when_no_truncation(self, mock_logger, mock_upstream):
        """When no truncation, the original result dict is returned (identity)."""
        from manus_agent.tools.http_request import http_request

        original = _upstream_result([_text_item("short")])
        mock_upstream.return_value = original

        result = http_request(_make_tool_use())
        assert result is original


class TestTruncationMessageFormat:
    """Verify the exact format of truncation messages."""

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 10)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_phase1_truncation_format(self, mock_logger, mock_upstream):
        """Phase 1 appends '\\n[truncated: N chars removed]'."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("A" * 25)])

        result = http_request(_make_tool_use())
        text = result["content"][0]["text"]

        assert text == "A" * 10 + "\n[truncated: 15 chars removed]"

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 8)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_phase2_truncation_format(self, mock_logger, mock_upstream):
        """Phase 2 also appends '\\n[truncated: N chars removed]'."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("A" * 20)])

        result = http_request(_make_tool_use())
        text = result["content"][0]["text"]

        assert "\n[truncated:" in text
        assert "chars removed]" in text


class TestExtraResultKeys:
    """Verify non-content keys in the result dict are preserved."""

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 5)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_preserves_status_and_tool_use_id(self, mock_logger, mock_upstream):
        """Result keys beyond 'content' (status, toolUseId) are preserved after truncation."""
        from manus_agent.tools.http_request import http_request

        original = {
            "status": "success",
            "toolUseId": "xyz-123",
            "content": [_text_item("A" * 50)],
            "headers": {"content-type": "text/plain"},
        }
        mock_upstream.return_value = original

        result = http_request(_make_tool_use())

        assert result["status"] == "success"
        assert result["toolUseId"] == "xyz-123"
        assert result["headers"] == {"content-type": "text/plain"}
        assert "[truncated:" in result["content"][0]["text"]


class TestBoundaryConditions:
    """Boundary and edge cases."""

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 10)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_exactly_at_per_item_limit(self, mock_logger, mock_upstream):
        """Text exactly at MAX_OUTPUT_CHARS is not truncated."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("A" * 10)])

        result = http_request(_make_tool_use())
        assert result["content"][0]["text"] == "A" * 10

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 10)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_one_char_over_per_item_limit(self, mock_logger, mock_upstream):
        """Text one char over MAX_OUTPUT_CHARS is truncated."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("A" * 11)])

        result = http_request(_make_tool_use())
        text = result["content"][0]["text"]
        assert text.startswith("A" * 10)
        assert "[truncated: 1 chars removed]" in text

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 20)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_exactly_at_total_limit(self, mock_logger, mock_upstream):
        """Total text exactly at MAX_TOTAL_OUTPUT_CHARS — no Phase 2 truncation."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result(
            [
                _text_item("A" * 10),
                _text_item("B" * 10),
            ]
        )

        result = http_request(_make_tool_use())
        assert result["content"][0]["text"] == "A" * 10
        assert result["content"][1]["text"] == "B" * 10

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 19)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_one_char_over_total_limit(self, mock_logger, mock_upstream):
        """Total text one char over MAX_TOTAL_OUTPUT_CHARS triggers Phase 2."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result(
            [
                _text_item("A" * 10),
                _text_item("B" * 10),
            ]
        )

        result = http_request(_make_tool_use())
        items = result["content"]
        # At least one item should be truncated
        has_truncation = any("[truncated:" in item.get("text", "") for item in items if isinstance(item, dict))
        assert has_truncation

    @patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 100_000)
    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_single_char_text(self, mock_logger, mock_upstream):
        """Single-character text items pass through fine."""
        from manus_agent.tools.http_request import http_request

        mock_upstream.return_value = _upstream_result([_text_item("X")])

        result = http_request(_make_tool_use())
        assert result["content"][0]["text"] == "X"

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_many_small_items(self, mock_logger, mock_upstream):
        """Many small items below all limits pass through unchanged."""
        from manus_agent.tools.http_request import http_request

        items = [_text_item(f"item-{i}") for i in range(100)]
        mock_upstream.return_value = _upstream_result(items)

        result = http_request(_make_tool_use())
        assert len(result["content"]) == 100
        for i, item in enumerate(result["content"]):
            assert item["text"] == f"item-{i}"


class TestDefaultLimits:
    """Verify the default module-level constants."""

    def test_default_max_output_chars(self):
        """Default MAX_OUTPUT_CHARS is 20,000."""
        from manus_agent.tools.http_request import MAX_OUTPUT_CHARS

        # Check the default (unless overridden by env)
        if "HTTP_REQUEST_MAX_OUTPUT_CHARS" not in os.environ:
            assert MAX_OUTPUT_CHARS == 20_000

    def test_default_max_total_output_chars(self):
        """Default MAX_TOTAL_OUTPUT_CHARS is 30,000."""
        from manus_agent.tools.http_request import MAX_TOTAL_OUTPUT_CHARS

        if "HTTP_REQUEST_MAX_TOTAL_OUTPUT_CHARS" not in os.environ:
            assert MAX_TOTAL_OUTPUT_CHARS == 30_000


class TestToolOutputLoggerUnit:
    """Unit tests for the log_tool_output_size helper itself."""

    def test_logs_text_content_size(self, capsys):
        """Logs total text size from content items."""
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {"content": [{"text": "hello"}, {"text": "world"}]}
        log_tool_output_size("test_tool", result)
        captured = capsys.readouterr()
        assert "[test_tool] Output size: 10 chars" in captured.out

    def test_logs_json_content_size(self, capsys):
        """Logs size of JSON content items."""
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {"content": [{"json": {"key": "value"}}]}
        log_tool_output_size("test_tool", result)
        captured = capsys.readouterr()
        assert "[test_tool] Output size:" in captured.out

    def test_empty_content(self, capsys):
        """No output for empty content."""
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {"content": []}
        log_tool_output_size("test_tool", result)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_content_key(self, capsys):
        """No output when content key is missing."""
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        log_tool_output_size("test_tool", {"status": "ok"})
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_non_dict_result(self, capsys):
        """Non-dict result does not crash."""
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        log_tool_output_size("test_tool", "not a dict")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_none_result(self, capsys):
        """None result does not crash."""
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        log_tool_output_size("test_tool", None)
        # Should not raise

    def test_mixed_text_and_json(self, capsys):
        """Mixed text and json items both contribute to size."""
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        result = {"content": [{"text": "abc"}, {"json": {"x": 1}}]}
        log_tool_output_size("test_tool", result)
        captured = capsys.readouterr()
        assert "[test_tool] Output size:" in captured.out

    def test_json_with_exception(self, capsys):
        """JSON item that fails str() doesn't crash."""
        from manus_agent.tools.tool_output_logger import log_tool_output_size

        class BadStr:
            def __str__(self):
                raise ValueError("bad")

        result = {"content": [{"json": BadStr()}, {"text": "ok"}]}
        log_tool_output_size("test_tool", result)
        # Should not raise; "ok" still counted
        captured = capsys.readouterr()
        assert "[test_tool] Output size: 2 chars" in captured.out
