"""Comprehensive test suite for the http_request truncation wrapper.

Tests cover:
- Passthrough behavior (no truncation needed)
- Per-item truncation (Phase 1: individual items exceeding MAX_OUTPUT_CHARS)
- Total truncation (Phase 2: proportional reduction when total exceeds MAX_TOTAL_OUTPUT_CHARS)
- Combined Phase 1 + Phase 2 truncation
- Edge cases: empty content, non-text items, missing keys, exception handling
- Environment variable overrides for limits
- Logging behavior (log_tool_output_size integration)
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_result(content):
    """Build a minimal ToolResult-like dict."""
    return {"status": "success", "content": content}


def _text_item(text):
    """Build a text content item."""
    return {"text": text}


def _image_item():
    """Build a non-text content item (image placeholder)."""
    return {"image": {"format": "png", "source": {"bytes": b"fake"}}}


# ---------------------------------------------------------------------------
# Passthrough — no truncation required
# ---------------------------------------------------------------------------


class TestPassthrough:
    """When output is within limits, it should pass through unchanged."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_small_single_item_passes_through(self, mock_log, mock_inner):
        """A single small text item should not be truncated."""
        from manus_agent.tools.http_request import http_request

        content = [_text_item("hello world")]
        mock_inner.return_value = _make_tool_result(content)
        tool_use = MagicMock()

        result = http_request(tool_use)

        assert result["content"][0]["text"] == "hello world"
        mock_log.assert_called_once()

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_multiple_small_items_pass_through(self, mock_log, mock_inner):
        """Multiple small text items should not be truncated."""
        from manus_agent.tools.http_request import http_request

        items = [_text_item(f"item {i}") for i in range(5)]
        mock_inner.return_value = _make_tool_result(items)
        tool_use = MagicMock()

        result = http_request(tool_use)

        for i in range(5):
            assert result["content"][i]["text"] == f"item {i}"

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_empty_content_passes_through(self, mock_log, mock_inner):
        """Empty content list should pass through."""
        from manus_agent.tools.http_request import http_request

        mock_inner.return_value = _make_tool_result([])
        tool_use = MagicMock()

        result = http_request(tool_use)

        assert result["content"] == []

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_non_text_items_pass_through(self, mock_log, mock_inner):
        """Non-text items (images, etc.) should pass through untouched."""
        from manus_agent.tools.http_request import http_request

        content = [_image_item(), _text_item("short")]
        mock_inner.return_value = _make_tool_result(content)
        tool_use = MagicMock()

        result = http_request(tool_use)

        assert result["content"][0] == _image_item()
        assert result["content"][1]["text"] == "short"

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_exactly_at_per_item_limit_not_truncated(self, mock_log, mock_inner):
        """Text at exactly MAX_OUTPUT_CHARS should not be truncated."""
        from manus_agent.tools.http_request import MAX_OUTPUT_CHARS, http_request

        text = "x" * MAX_OUTPUT_CHARS
        mock_inner.return_value = _make_tool_result([_text_item(text)])
        tool_use = MagicMock()

        result = http_request(tool_use)

        assert result["content"][0]["text"] == text
        assert "[truncated" not in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Phase 1 — Per-item truncation
# ---------------------------------------------------------------------------


class TestPerItemTruncation:
    """Phase 1: individual items exceeding MAX_OUTPUT_CHARS are truncated."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_single_item_exceeding_limit_truncated(self, mock_log, mock_inner):
        """A single item exceeding MAX_OUTPUT_CHARS should be truncated with marker."""
        from manus_agent.tools.http_request import MAX_OUTPUT_CHARS, http_request

        overflow = 500
        text = "A" * (MAX_OUTPUT_CHARS + overflow)
        mock_inner.return_value = _make_tool_result([_text_item(text)])
        tool_use = MagicMock()

        result = http_request(tool_use)

        output_text = result["content"][0]["text"]
        assert output_text.startswith("A" * MAX_OUTPUT_CHARS)
        assert f"[truncated: {overflow} chars removed]" in output_text

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_one_item_over_one_under(self, mock_log, mock_inner):
        """Only the oversized item should be truncated; small one untouched."""
        from manus_agent.tools.http_request import MAX_OUTPUT_CHARS, http_request

        big_text = "B" * (MAX_OUTPUT_CHARS + 100)
        small_text = "s" * 50
        mock_inner.return_value = _make_tool_result([_text_item(big_text), _text_item(small_text)])
        tool_use = MagicMock()

        result = http_request(tool_use)

        assert "[truncated: 100 chars removed]" in result["content"][0]["text"]
        assert result["content"][1]["text"] == small_text

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_multiple_items_over_limit(self, mock_log, mock_inner):
        """Multiple oversized items should each be independently truncated."""
        from manus_agent.tools.http_request import MAX_OUTPUT_CHARS, http_request

        text1 = "X" * (MAX_OUTPUT_CHARS + 200)
        text2 = "Y" * (MAX_OUTPUT_CHARS + 300)
        mock_inner.return_value = _make_tool_result([_text_item(text1), _text_item(text2)])
        tool_use = MagicMock()

        # Use a very high total limit so Phase 2 doesn't trigger
        with patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", MAX_OUTPUT_CHARS * 10):
            result = http_request(tool_use)

        assert "[truncated: 200 chars removed]" in result["content"][0]["text"]
        assert "[truncated: 300 chars removed]" in result["content"][1]["text"]

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_truncation_marker_format(self, mock_log, mock_inner):
        """Truncation marker should be on a newline with exact char count."""
        from manus_agent.tools.http_request import MAX_OUTPUT_CHARS, http_request

        overflow = 1234
        text = "Z" * (MAX_OUTPUT_CHARS + overflow)
        mock_inner.return_value = _make_tool_result([_text_item(text)])
        tool_use = MagicMock()

        result = http_request(tool_use)

        output_text = result["content"][0]["text"]
        lines = output_text.split("\n")
        assert lines[-1] == f"[truncated: {overflow} chars removed]"


# ---------------------------------------------------------------------------
# Phase 2 — Proportional total truncation
# ---------------------------------------------------------------------------


class TestTotalTruncation:
    """Phase 2: when total exceeds MAX_TOTAL_OUTPUT_CHARS, proportional reduction."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_total_exceeds_limit_items_reduced(self, mock_log, mock_inner):
        """When sum of items > MAX_TOTAL_OUTPUT_CHARS, all items reduced proportionally."""
        from manus_agent.tools.http_request import (
            MAX_OUTPUT_CHARS,
            MAX_TOTAL_OUTPUT_CHARS,
            http_request,
        )

        # Create items that are individually under per-item limit but
        # collectively exceed total limit.
        # Use items of size MAX_OUTPUT_CHARS (at the per-item cap exactly)
        # so Phase 1 doesn't trigger, but total = N * MAX_OUTPUT_CHARS > MAX_TOTAL_OUTPUT_CHARS
        item_size = MAX_OUTPUT_CHARS  # 20000
        num_items = (MAX_TOTAL_OUTPUT_CHARS // item_size) + 1  # 2 items = 40000 > 30000
        items = [_text_item("A" * item_size) for _ in range(num_items)]
        mock_inner.return_value = _make_tool_result(items)
        tool_use = MagicMock()

        result = http_request(tool_use)

        # Every item should have a truncation marker
        for item in result["content"]:
            if isinstance(item, dict) and "text" in item:
                assert "[truncated:" in item["text"]

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_proportional_reduction_maintains_ratio(self, mock_log, mock_inner):
        """Items should be reduced proportionally (larger items lose more)."""
        from manus_agent.tools.http_request import (
            http_request,
        )

        # Two items: 15000 + 20000 = 35000 > 30000 total limit
        # After Phase 1 (both under 20000 per-item limit), Phase 2 applies
        text1 = "A" * 15000
        text2 = "B" * 20000
        mock_inner.return_value = _make_tool_result([_text_item(text1), _text_item(text2)])
        tool_use = MagicMock()

        result = http_request(tool_use)

        # Extract actual text lengths (before the truncation marker line)
        content_0 = result["content"][0]["text"]
        content_1 = result["content"][1]["text"]

        # Both should be truncated since total exceeds limit
        assert "[truncated:" in content_0
        assert "[truncated:" in content_1

        # The ratio of kept text should roughly preserve the original ratio
        # item1 was 15000/35000 ≈ 43%, item2 was 20000/35000 ≈ 57%
        kept_0 = len(content_0.split("\n[truncated:")[0])
        kept_1 = len(content_1.split("\n[truncated:")[0])
        # Larger item should keep more absolute chars
        assert kept_1 > kept_0

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_exactly_at_total_limit_not_reduced(self, mock_log, mock_inner):
        """Content at exactly MAX_TOTAL_OUTPUT_CHARS should not trigger Phase 2."""
        from manus_agent.tools.http_request import http_request

        # Single item at exactly total limit (also under per-item limit of 20000)
        # Total limit is 30000, per-item is 20000, so use 2 items of 15000 each = 30000
        text1 = "A" * 15000
        text2 = "B" * 15000
        mock_inner.return_value = _make_tool_result([_text_item(text1), _text_item(text2)])
        tool_use = MagicMock()

        result = http_request(tool_use)

        # Neither should be truncated
        assert result["content"][0]["text"] == text1
        assert result["content"][1]["text"] == text2


# ---------------------------------------------------------------------------
# Combined Phase 1 + Phase 2
# ---------------------------------------------------------------------------


class TestCombinedTruncation:
    """Phase 1 truncates individual items, then Phase 2 reduces total."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_phase1_then_phase2(self, mock_log, mock_inner):
        """An oversized item is first capped at per-item limit, then total is reduced."""
        from manus_agent.tools.http_request import (
            MAX_OUTPUT_CHARS,
            http_request,
        )

        # Item 1: way over per-item limit → Phase 1 caps at MAX_OUTPUT_CHARS
        # Item 2: under per-item limit but makes total exceed total limit
        # After Phase 1: item1=20000, item2=15000 → total=35000 > 30000
        text1 = "X" * (MAX_OUTPUT_CHARS + 5000)  # 25000
        text2 = "Y" * 15000
        mock_inner.return_value = _make_tool_result([_text_item(text1), _text_item(text2)])
        tool_use = MagicMock()

        result = http_request(tool_use)

        # Both items should be truncated (Phase 1 first, then Phase 2 proportional)
        assert "[truncated:" in result["content"][0]["text"]
        assert "[truncated:" in result["content"][1]["text"]

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_phase1_sufficient_no_phase2(self, mock_log, mock_inner):
        """If Phase 1 truncation brings total under limit, Phase 2 should not apply."""
        from manus_agent.tools.http_request import (
            MAX_OUTPUT_CHARS,
            http_request,
        )

        # Item 1: slightly over per-item limit, but after Phase 1 total is under total limit
        # Phase 1 caps item1 at 20000. Total = 20000 + 5000 = 25000 < 30000
        text1 = "X" * (MAX_OUTPUT_CHARS + 100)  # 20100
        text2 = "Y" * 5000
        mock_inner.return_value = _make_tool_result([_text_item(text1), _text_item(text2)])
        tool_use = MagicMock()

        result = http_request(tool_use)

        # Item 1 truncated by Phase 1
        assert "[truncated: 100 chars removed]" in result["content"][0]["text"]
        # Item 2 should NOT be truncated (no Phase 2 needed)
        assert result["content"][1]["text"] == text2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and unusual inputs."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_missing_content_key(self, mock_log, mock_inner):
        """Result without 'content' key should not crash."""
        from manus_agent.tools.http_request import http_request

        mock_inner.return_value = {"status": "success"}
        tool_use = MagicMock()

        result = http_request(tool_use)

        assert result == {"status": "success"}

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_none_text_in_item(self, mock_log, mock_inner):
        """Item with text=None should be treated as non-text."""
        from manus_agent.tools.http_request import http_request

        content = [{"text": None}, _text_item("hello")]
        mock_inner.return_value = _make_tool_result(content)
        tool_use = MagicMock()

        result = http_request(tool_use)

        assert result["content"][0] == {"text": None}
        assert result["content"][1]["text"] == "hello"

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_non_dict_items_in_content(self, mock_log, mock_inner):
        """Non-dict items in content list should pass through."""
        from manus_agent.tools.http_request import http_request

        content = ["string_item", 42, _text_item("real")]
        mock_inner.return_value = _make_tool_result(content)
        tool_use = MagicMock()

        result = http_request(tool_use)

        assert result["content"][0] == "string_item"
        assert result["content"][1] == 42
        assert result["content"][2]["text"] == "real"

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_empty_text_items(self, mock_log, mock_inner):
        """Items with empty string text should pass through unchanged."""
        from manus_agent.tools.http_request import http_request

        content = [_text_item(""), _text_item("")]
        mock_inner.return_value = _make_tool_result(content)
        tool_use = MagicMock()

        result = http_request(tool_use)

        assert result["content"][0]["text"] == ""
        assert result["content"][1]["text"] == ""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_item_with_extra_keys_preserved(self, mock_log, mock_inner):
        """Extra keys in text items (like 'type') should be preserved."""
        from manus_agent.tools.http_request import http_request

        content = [{"text": "hello", "type": "text", "annotations": []}]
        mock_inner.return_value = _make_tool_result(content)
        tool_use = MagicMock()

        result = http_request(tool_use)

        item = result["content"][0]
        assert item["text"] == "hello"
        assert item["type"] == "text"
        assert item["annotations"] == []

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_result_other_keys_preserved(self, mock_log, mock_inner):
        """Keys besides 'content' in the result should be preserved."""
        from manus_agent.tools.http_request import MAX_OUTPUT_CHARS, http_request

        big_text = "Z" * (MAX_OUTPUT_CHARS + 10)
        mock_inner.return_value = {
            "status": "success",
            "content": [_text_item(big_text)],
            "metadata": {"url": "https://example.com"},
        }
        tool_use = MagicMock()

        result = http_request(tool_use)

        assert result["status"] == "success"
        assert result["metadata"] == {"url": "https://example.com"}
        assert "[truncated:" in result["content"][0]["text"]

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_single_char_overflow(self, mock_log, mock_inner):
        """Even 1 char over the limit should trigger truncation."""
        from manus_agent.tools.http_request import MAX_OUTPUT_CHARS, http_request

        text = "A" * (MAX_OUTPUT_CHARS + 1)
        mock_inner.return_value = _make_tool_result([_text_item(text)])
        tool_use = MagicMock()

        result = http_request(tool_use)

        assert "[truncated: 1 chars removed]" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Exception handling (graceful degradation)
# ---------------------------------------------------------------------------


class TestExceptionHandling:
    """The wrapper should never crash; on internal error it returns raw result."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_log_exception_returns_raw_result(self, mock_log, mock_inner):
        """If log_tool_output_size raises, the result should still be returned."""
        from manus_agent.tools.http_request import http_request

        mock_log.side_effect = RuntimeError("logging exploded")
        content = [_text_item("hi")]
        mock_inner.return_value = _make_tool_result(content)
        tool_use = MagicMock()

        # Should not raise
        result = http_request(tool_use)

        # Still get the result (either original or truncated)
        assert "content" in result

    @patch("manus_agent.tools.http_request._http_request")
    def test_content_not_iterable_returns_raw(self, mock_inner):
        """If content is not iterable (e.g., int), wrapper should gracefully degrade."""
        from manus_agent.tools.http_request import http_request

        mock_inner.return_value = {"status": "success", "content": 12345}
        tool_use = MagicMock()

        result = http_request(tool_use)

        # Should return the raw result without crashing
        assert result["status"] == "success"

    @patch("manus_agent.tools.http_request._http_request")
    def test_content_none_returns_raw(self, mock_inner):
        """If content is None, wrapper should gracefully degrade."""
        from manus_agent.tools.http_request import http_request

        mock_inner.return_value = {"status": "success", "content": None}
        tool_use = MagicMock()

        result = http_request(tool_use)

        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvVarOverrides:
    """Truncation limits can be overridden via patching module constants."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_custom_per_item_limit(self, mock_log, mock_inner):
        """Patching MAX_OUTPUT_CHARS should change per-item truncation threshold."""
        from manus_agent.tools.http_request import http_request

        text = "A" * 150  # Over a custom 100-char limit
        mock_inner.return_value = _make_tool_result([_text_item(text)])
        tool_use = MagicMock()

        with patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 100):
            with patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 10000):
                result = http_request(tool_use)

        assert "[truncated: 50 chars removed]" in result["content"][0]["text"]

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_custom_total_limit(self, mock_log, mock_inner):
        """Patching MAX_TOTAL_OUTPUT_CHARS should change total truncation threshold."""
        from manus_agent.tools.http_request import http_request

        # Two items: 100 each = 200 total > 150 limit
        text1 = "A" * 100
        text2 = "B" * 100
        mock_inner.return_value = _make_tool_result([_text_item(text1), _text_item(text2)])
        tool_use = MagicMock()

        with patch("manus_agent.tools.http_request.MAX_OUTPUT_CHARS", 200):
            with patch("manus_agent.tools.http_request.MAX_TOTAL_OUTPUT_CHARS", 150):
                result = http_request(tool_use)

        # Both should be proportionally reduced
        assert "[truncated:" in result["content"][0]["text"]
        assert "[truncated:" in result["content"][1]["text"]

    def test_env_vars_read_at_import_time(self):
        """Module constants are read from env vars at import time."""
        import sys

        mod = sys.modules["manus_agent.tools.http_request"]

        # Verify the module reads from env vars (default values)
        assert mod.MAX_OUTPUT_CHARS == 20_000
        assert mod.MAX_TOTAL_OUTPUT_CHARS == 30_000


# ---------------------------------------------------------------------------
# kwargs and tool_use passthrough
# ---------------------------------------------------------------------------


class TestInnerFunctionCall:
    """The wrapper should pass tool_use and kwargs to the inner function."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_tool_use_passed_to_inner(self, mock_log, mock_inner):
        """The ToolUse object should be forwarded to _http_request."""
        from manus_agent.tools.http_request import http_request

        mock_inner.return_value = _make_tool_result([_text_item("ok")])
        tool_use = MagicMock()
        tool_use.name = "http_request"

        http_request(tool_use)

        mock_inner.assert_called_once_with(tool_use)

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_kwargs_passed_to_inner(self, mock_log, mock_inner):
        """Extra kwargs should be forwarded to _http_request."""
        from manus_agent.tools.http_request import http_request

        mock_inner.return_value = _make_tool_result([_text_item("ok")])
        tool_use = MagicMock()

        http_request(tool_use, session="test_session", extra_param=42)

        mock_inner.assert_called_once_with(tool_use, session="test_session", extra_param=42)


# ---------------------------------------------------------------------------
# Logging behavior
# ---------------------------------------------------------------------------


class TestLogging:
    """Verify print/log output for observability."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_log_tool_output_size_called_with_name_and_result(self, mock_log, mock_inner):
        """log_tool_output_size should be called with 'http_request' and the final result."""
        from manus_agent.tools.http_request import http_request

        content = [_text_item("data")]
        mock_inner.return_value = _make_tool_result(content)
        tool_use = MagicMock()

        result = http_request(tool_use)

        mock_log.assert_called_once_with("http_request", result)

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    @patch("builtins.print")
    def test_print_output_sizes(self, mock_print, mock_log, mock_inner):
        """Should print before/after truncation sizes."""
        from manus_agent.tools.http_request import http_request

        content = [_text_item("A" * 100)]
        mock_inner.return_value = _make_tool_result(content)
        tool_use = MagicMock()

        http_request(tool_use)

        # Should print at least the before and after sizes
        print_calls = [str(c) for c in mock_print.call_args_list]
        before_printed = any("before truncation: 100" in c for c in print_calls)
        after_printed = any("after truncation: 100" in c for c in print_calls)
        assert before_printed
        assert after_printed

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    @patch("builtins.print")
    def test_no_text_no_size_print(self, mock_print, mock_log, mock_inner):
        """When total text is 0 (no text items), size prints should be skipped."""
        from manus_agent.tools.http_request import http_request

        content = [_image_item()]
        mock_inner.return_value = _make_tool_result(content)
        tool_use = MagicMock()

        http_request(tool_use)

        # Should NOT print size lines (total_before is 0)
        print_calls = [str(c) for c in mock_print.call_args_list]
        assert not any("Output size" in c for c in print_calls)


# ---------------------------------------------------------------------------
# Immutability of original result
# ---------------------------------------------------------------------------


class TestImmutability:
    """Original result dict should not be mutated when not truncating."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_no_mutation_when_no_truncation(self, mock_log, mock_inner):
        """When nothing is truncated, original result object should be returned as-is."""
        from manus_agent.tools.http_request import http_request

        content = [_text_item("short")]
        original_result = _make_tool_result(content)
        mock_inner.return_value = original_result
        tool_use = MagicMock()

        result = http_request(tool_use)

        # Should be the same object (no copy made)
        assert result is original_result

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_new_dict_when_truncated(self, mock_log, mock_inner):
        """When truncation occurs, a new dict should be returned (not mutated original)."""
        from manus_agent.tools.http_request import MAX_OUTPUT_CHARS, http_request

        content = [_text_item("A" * (MAX_OUTPUT_CHARS + 10))]
        original_result = _make_tool_result(content)
        mock_inner.return_value = original_result
        tool_use = MagicMock()

        result = http_request(tool_use)

        # Should be a different object
        assert result is not original_result
        # Original should be unchanged
        assert len(original_result["content"][0]["text"]) == MAX_OUTPUT_CHARS + 10


# ---------------------------------------------------------------------------
# Phase 2 proportional math accuracy
# ---------------------------------------------------------------------------


class TestProportionalMath:
    """Verify the proportional reduction math is correct."""

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_ratio_calculation(self, mock_log, mock_inner):
        """Verify ratio = MAX_TOTAL_OUTPUT_CHARS / total_after_phase1."""
        from manus_agent.tools.http_request import MAX_TOTAL_OUTPUT_CHARS, http_request

        # 3 items of 12000 each = 36000 > 30000
        # ratio = 30000/36000 ≈ 0.8333
        # Each item kept: int(12000 * 0.8333) = 10000
        items = [_text_item("A" * 12000) for _ in range(3)]
        mock_inner.return_value = _make_tool_result(items)
        tool_use = MagicMock()

        result = http_request(tool_use)

        for item in result["content"]:
            text = item["text"]
            kept_part = text.split("\n[truncated:")[0]
            expected_kept = int(12000 * (MAX_TOTAL_OUTPUT_CHARS / 36000))
            assert len(kept_part) == expected_kept

    @patch("manus_agent.tools.http_request._http_request")
    @patch("manus_agent.tools.http_request.log_tool_output_size")
    def test_zero_length_item_in_proportional(self, mock_log, mock_inner):
        """Zero-length text items should survive proportional reduction without error."""
        from manus_agent.tools.http_request import MAX_TOTAL_OUTPUT_CHARS, http_request

        # Mix of empty and large items where total > limit
        items = [_text_item(""), _text_item("X" * (MAX_TOTAL_OUTPUT_CHARS + 100))]
        mock_inner.return_value = _make_tool_result(items)
        tool_use = MagicMock()

        # Should not crash (empty item * ratio = 0, which is fine)
        result = http_request(tool_use)

        assert result["content"][0]["text"] == ""
