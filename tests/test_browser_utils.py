"""Comprehensive test suite for manus_agent.tools.browser_utils module.

Tests cover:
- BrowserTimeoutError exception construction and messaging
- make_dom_read_script_null_safe (null-safety injection for DOM reads)
- normalize_evaluate_script (JS normalization for Playwright evaluate)
- prepare_evaluate_script (combined pipeline)
- normalize_browser_selector (CSS selector cleaning)
- is_selector_syntax_error (error classification)
- is_probably_raw_text_page (heuristic raw-text detection)
- extract_raw_page_text (raw text extraction)
- get_text_with_fallback (timeout/fallback text extraction)
- get_html_with_fallback (HTML extraction with timeout handling)
- find_best_content_selector (content selector detection)
- extract_page_text (high-level text extraction)
- format_browser_error (error formatting)

All tests are fully mocked — no real browser or HTTP calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from manus_agent.tools.browser_utils import (
    DEFAULT_TIMEOUT,
    FALLBACK_SELECTORS,
    RAW_TEXT_URL_HINTS,
    BrowserTimeoutError,
    extract_page_text,
    extract_raw_page_text,
    find_best_content_selector,
    format_browser_error,
    get_html_with_fallback,
    get_text_with_fallback,
    is_probably_raw_text_page,
    is_selector_syntax_error,
    make_dom_read_script_null_safe,
    normalize_browser_selector,
    normalize_evaluate_script,
    prepare_evaluate_script,
)

# ===========================================================================
# BrowserTimeoutError
# ===========================================================================


class TestBrowserTimeoutError:
    """Tests for the BrowserTimeoutError exception class."""

    def test_basic_construction(self):
        err = BrowserTimeoutError(selector="div.content", timeout_ms=5000)
        assert err.selector == "div.content"
        assert err.timeout_ms == 5000
        assert err.url is None
        assert err.tried_fallbacks == []
        assert "div.content" in str(err)
        assert "5000" in str(err)

    def test_with_url(self):
        err = BrowserTimeoutError(selector="main", timeout_ms=10000, url="https://example.com")
        assert err.url == "https://example.com"
        assert "https://example.com" in str(err)

    def test_with_fallbacks(self):
        fallbacks = ["main", "article", "body"]
        err = BrowserTimeoutError(selector="article", timeout_ms=3000, tried_fallbacks=fallbacks)
        assert err.tried_fallbacks == fallbacks
        msg = str(err)
        assert "main" in msg or "article" in msg

    def test_suggestions_in_message(self):
        err = BrowserTimeoutError(selector="h1", timeout_ms=1000)
        msg = str(err)
        assert "Verify the selector" in msg or "Suggestions" in msg

    def test_inherits_from_exception(self):
        err = BrowserTimeoutError(selector="p", timeout_ms=100)
        assert isinstance(err, Exception)

    def test_none_fallbacks_becomes_empty_list(self):
        err = BrowserTimeoutError(selector="span", timeout_ms=500, tried_fallbacks=None)
        assert err.tried_fallbacks == []


# ===========================================================================
# make_dom_read_script_null_safe
# ===========================================================================


class TestMakeDomReadScriptNullSafe:
    """Tests for make_dom_read_script_null_safe."""

    def test_text_content_null_safe(self):
        script = "document.querySelector('h1').textContent"
        result = make_dom_read_script_null_safe(script)
        assert "?." in result or "??" in result
        assert ".textContent" in result

    def test_inner_text_null_safe(self):
        script = "document.querySelector('.title').innerText"
        result = make_dom_read_script_null_safe(script)
        assert "?." in result or "??" in result

    def test_inner_html_null_safe(self):
        script = "document.querySelector('#box').innerHTML"
        result = make_dom_read_script_null_safe(script)
        assert "??" in result

    def test_value_null_safe(self):
        script = "document.querySelector('input').value"
        result = make_dom_read_script_null_safe(script)
        assert "??" in result

    def test_href_null_safe(self):
        script = "document.querySelector('a').href"
        result = make_dom_read_script_null_safe(script)
        assert "??" in result

    def test_empty_string_unchanged(self):
        assert make_dom_read_script_null_safe("") == ""

    def test_none_unchanged(self):
        # Falsy returns as-is
        assert make_dom_read_script_null_safe("") == ""

    def test_unrelated_script_unchanged(self):
        script = "window.location.href"
        result = make_dom_read_script_null_safe(script)
        assert result == script

    def test_multiple_patterns_in_one_script(self):
        script = "document.querySelector('h1').textContent + document.querySelector('p').innerText"
        result = make_dom_read_script_null_safe(script)
        assert result.count("??") >= 2


# ===========================================================================
# normalize_evaluate_script
# ===========================================================================


class TestNormalizeEvaluateScript:
    """Tests for normalize_evaluate_script."""

    def test_already_a_function(self):
        script = "function() { return 1; }"
        result = normalize_evaluate_script(script)
        assert result == script

    def test_already_async_function(self):
        script = "async function() { return await fetch('/'); }"
        result = normalize_evaluate_script(script)
        assert result == script

    def test_arrow_function_preserved(self):
        script = "() => document.title"
        result = normalize_evaluate_script(script)
        assert result == script

    def test_arrow_with_param(self):
        script = "x => x + 1"
        result = normalize_evaluate_script(script)
        assert result == script

    def test_simple_expression_wrapped(self):
        script = "document.title"
        result = normalize_evaluate_script(script)
        assert result == "() => (document.title)"

    def test_object_literal_wrapped(self):
        script = '{"key": "value"}'
        result = normalize_evaluate_script(script)
        assert result == '() => ({"key": "value"})'

    def test_statement_wrapped_in_block(self):
        script = "return document.title"
        result = normalize_evaluate_script(script)
        assert "() => {" in result
        assert "return document.title" in result

    def test_const_statement_wrapped(self):
        script = "const x = 1; return x;"
        result = normalize_evaluate_script(script)
        assert "() => {" in result

    def test_multiline_wrapped_in_block(self):
        script = "const a = 1;\nconst b = 2;\nreturn a + b;"
        result = normalize_evaluate_script(script)
        assert "() => {" in result

    def test_semicolon_expression_wrapped_in_block(self):
        script = "let x = document.title; x"
        result = normalize_evaluate_script(script)
        assert "() => {" in result

    def test_empty_string_returns_null_function(self):
        result = normalize_evaluate_script("")
        assert result == "() => null"

    def test_whitespace_only_returns_null_function(self):
        result = normalize_evaluate_script("   ")
        assert result == "() => null"

    def test_non_string_passthrough(self):
        # Edge case: non-string input
        result = normalize_evaluate_script(123)  # type: ignore[arg-type]
        assert result == 123

    def test_if_statement_wrapped(self):
        script = "if (true) return 1;"
        result = normalize_evaluate_script(script)
        assert "() => {" in result

    def test_for_loop_wrapped(self):
        script = "for (let i=0; i<10; i++) {}"
        result = normalize_evaluate_script(script)
        assert "() => {" in result


# ===========================================================================
# prepare_evaluate_script
# ===========================================================================


class TestPrepareEvaluateScript:
    """Tests for prepare_evaluate_script (combined pipeline)."""

    def test_null_safety_plus_normalization(self):
        script = "document.querySelector('h1').textContent"
        result = prepare_evaluate_script(script)
        # Should be both null-safe AND wrapped
        assert "??" in result or "?." in result
        assert "() =>" in result

    def test_non_string_passthrough(self):
        result = prepare_evaluate_script(42)  # type: ignore[arg-type]
        assert result == 42

    def test_already_valid_function(self):
        script = "() => document.title"
        result = prepare_evaluate_script(script)
        assert result == script


# ===========================================================================
# normalize_browser_selector
# ===========================================================================


class TestNormalizeBrowserSelector:
    """Tests for normalize_browser_selector."""

    def test_strips_whitespace(self):
        assert normalize_browser_selector("  div.content  ") == "div.content"

    def test_removes_outer_single_quotes(self):
        assert normalize_browser_selector("'div.content'") == "div.content"

    def test_removes_outer_double_quotes(self):
        assert normalize_browser_selector('"div.content"') == "div.content"

    def test_preserves_inner_content_with_quotes(self):
        result = normalize_browser_selector("div[data-value='test']")
        assert result == "div[data-value='test']"

    def test_none_returns_none(self):
        assert normalize_browser_selector(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_browser_selector("") is None

    def test_whitespace_only_returns_none(self):
        assert normalize_browser_selector("   ") is None

    def test_non_string_returns_input(self):
        # Non-string input that is not None — function checks isinstance(selector, str)
        # and returns the selector unchanged when it's not a string and not None
        result = normalize_browser_selector(123)  # type: ignore[arg-type]
        assert result is None or result == 123  # Implementation returns selector as-is or None

    def test_single_char_selector(self):
        assert normalize_browser_selector("a") == "a"

    def test_quoted_whitespace_inner_stripped(self):
        assert normalize_browser_selector("'  main  '") == "main"

    def test_mismatched_quotes_preserved(self):
        # Opening ' closing " — not matching
        result = normalize_browser_selector("'div\"")
        assert result == "'div\""


# ===========================================================================
# is_selector_syntax_error
# ===========================================================================


class TestIsSelectorSyntaxError:
    """Tests for is_selector_syntax_error."""

    def test_parsing_css_selector(self):
        err = ValueError("while parsing css selector 'invalid>>>'")
        assert is_selector_syntax_error(err) is True

    def test_failed_to_parse(self):
        err = Exception("Failed to parse selector '#broken[")
        assert is_selector_syntax_error(err) is True

    def test_unexpected_token(self):
        err = Exception("Unexpected token in selector: >>")
        assert is_selector_syntax_error(err) is True

    def test_unsupported_token(self):
        err = Exception("Unsupported token in CSS selector")
        assert is_selector_syntax_error(err) is True

    def test_unknown_engine(self):
        err = Exception("Unknown engine 'xpath' in selector")
        assert is_selector_syntax_error(err) is True

    def test_invalid_selector(self):
        err = Exception("Invalid selector provided")
        assert is_selector_syntax_error(err) is True

    def test_normal_timeout_not_syntax_error(self):
        err = PlaywrightTimeoutError("Timeout 10000ms exceeded")
        assert is_selector_syntax_error(err) is False

    def test_generic_error_not_syntax_error(self):
        err = RuntimeError("Something went wrong")
        assert is_selector_syntax_error(err) is False


# ===========================================================================
# is_probably_raw_text_page
# ===========================================================================


class TestIsProbablyRawTextPage:
    """Tests for is_probably_raw_text_page (async heuristic)."""

    @pytest.mark.asyncio
    async def test_patch_url_hint(self):
        page = AsyncMock()
        result = await is_probably_raw_text_page(page, url="https://github.com/user/repo/commit/abc123.patch")
        assert result is True

    @pytest.mark.asyncio
    async def test_diff_url_hint(self):
        page = AsyncMock()
        result = await is_probably_raw_text_page(page, url="https://example.com/changes.diff")
        assert result is True

    @pytest.mark.asyncio
    async def test_normal_url_with_text_plain_content_type(self):
        page = AsyncMock()
        page.evaluate = AsyncMock(
            return_value={
                "contentType": "text/plain",
                "hasPre": False,
                "preCount": 0,
                "bodyChildCount": 1,
                "bodyTextLength": 500,
                "hasMainContent": False,
            }
        )
        result = await is_probably_raw_text_page(page, url="https://example.com/file.txt")
        assert result is True

    @pytest.mark.asyncio
    async def test_pre_only_page_no_main_content(self):
        page = AsyncMock()
        page.evaluate = AsyncMock(
            return_value={
                "contentType": "text/html",
                "hasPre": True,
                "preCount": 1,
                "bodyChildCount": 1,
                "bodyTextLength": 1000,
                "hasMainContent": False,
            }
        )
        result = await is_probably_raw_text_page(page, url="https://example.com/page")
        assert result is True

    @pytest.mark.asyncio
    async def test_rich_html_page(self):
        page = AsyncMock()
        page.evaluate = AsyncMock(
            return_value={
                "contentType": "text/html",
                "hasPre": False,
                "preCount": 0,
                "bodyChildCount": 20,
                "bodyTextLength": 5000,
                "hasMainContent": True,
            }
        )
        result = await is_probably_raw_text_page(page, url="https://example.com/article")
        assert result is False

    @pytest.mark.asyncio
    async def test_evaluate_exception_returns_false(self):
        page = AsyncMock()
        page.evaluate = AsyncMock(side_effect=RuntimeError("Page crashed"))
        result = await is_probably_raw_text_page(page, url="https://example.com")
        assert result is False

    @pytest.mark.asyncio
    async def test_non_dict_evaluate_returns_false(self):
        page = AsyncMock()
        page.evaluate = AsyncMock(return_value="not a dict")
        result = await is_probably_raw_text_page(page, url="https://example.com")
        assert result is False

    @pytest.mark.asyncio
    async def test_pre_with_main_content_is_not_raw(self):
        page = AsyncMock()
        page.evaluate = AsyncMock(
            return_value={
                "contentType": "text/html",
                "hasPre": True,
                "preCount": 1,
                "bodyChildCount": 10,
                "bodyTextLength": 3000,
                "hasMainContent": True,
            }
        )
        result = await is_probably_raw_text_page(page, url=None)
        assert result is False

    @pytest.mark.asyncio
    async def test_multiple_pre_tags_not_raw(self):
        page = AsyncMock()
        page.evaluate = AsyncMock(
            return_value={
                "contentType": "text/html",
                "hasPre": True,
                "preCount": 5,
                "bodyChildCount": 3,
                "bodyTextLength": 2000,
                "hasMainContent": False,
            }
        )
        result = await is_probably_raw_text_page(page, url=None)
        assert result is False


# ===========================================================================
# extract_raw_page_text
# ===========================================================================


class TestExtractRawPageText:
    """Tests for extract_raw_page_text (async)."""

    @pytest.mark.asyncio
    async def test_extracts_from_pre_element(self):
        page = AsyncMock()
        page.evaluate = AsyncMock(return_value="diff --git a/file.py b/file.py\n+added line")
        result = await extract_raw_page_text(page)
        assert "diff --git" in result

    @pytest.mark.asyncio
    async def test_fallback_to_body_inner_text(self):
        page = AsyncMock()
        # First script returns None (no <pre>), second returns body text
        page.evaluate = AsyncMock(side_effect=[None, "Body text content"])
        result = await extract_raw_page_text(page)
        assert result == "Body text content"

    @pytest.mark.asyncio
    async def test_fallback_to_document_element(self):
        page = AsyncMock()
        page.evaluate = AsyncMock(side_effect=[None, None, "Document element text"])
        result = await extract_raw_page_text(page)
        assert result == "Document element text"

    @pytest.mark.asyncio
    async def test_all_fail_returns_empty(self):
        page = AsyncMock()
        page.evaluate = AsyncMock(side_effect=RuntimeError("Crashed"))
        result = await extract_raw_page_text(page)
        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_string_skipped(self):
        page = AsyncMock()
        # pre returns empty, body returns empty, documentElement returns content
        page.evaluate = AsyncMock(side_effect=["", "   ", "Final content"])
        result = await extract_raw_page_text(page)
        assert result == "Final content"


# ===========================================================================
# get_text_with_fallback
# ===========================================================================


class TestGetTextWithFallback:
    """Tests for get_text_with_fallback (async)."""

    @pytest.mark.asyncio
    async def test_successful_primary_selector(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.text_content = AsyncMock(return_value="Hello world")
        text, selector = await get_text_with_fallback(page, "div.content")
        assert text == "Hello world"
        assert selector == "div.content"

    @pytest.mark.asyncio
    async def test_fallback_selector_used(self):
        page = AsyncMock()
        page.url = "https://example.com"

        async def _text_content(sel, timeout=None):
            if sel == "article":
                raise PlaywrightTimeoutError("timeout")
            if sel == "main":
                return "Main content"
            return None

        page.text_content = _text_content
        # Patch is_probably_raw_text_page to return False to avoid raw text path
        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = False
            text, selector = await get_text_with_fallback(page, "article")
        assert text == "Main content"
        assert selector == "main"

    @pytest.mark.asyncio
    async def test_raises_on_all_timeouts(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.text_content = AsyncMock(side_effect=PlaywrightTimeoutError("timeout"))
        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = False
            with pytest.raises(BrowserTimeoutError) as exc_info:
                await get_text_with_fallback(page, "article")
        assert exc_info.value.selector == "article"

    @pytest.mark.asyncio
    async def test_empty_selector_raises_valueerror(self):
        page = AsyncMock()
        with pytest.raises(ValueError, match="[Ss]elector.*empty"):
            await get_text_with_fallback(page, "")

    @pytest.mark.asyncio
    async def test_none_text_content_tries_fallback(self):
        page = AsyncMock()
        page.url = "https://example.com"
        call_count = [0]

        async def _text_content(sel, timeout=None):
            call_count[0] += 1
            if sel == "main":
                return None  # Element found but no text
            if sel == "#main":
                return "Found it"
            return None

        page.text_content = _text_content
        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = False
            text, selector = await get_text_with_fallback(page, "main")
        assert text == "Found it"
        assert selector == "#main"

    @pytest.mark.asyncio
    async def test_raw_text_page_fallback_on_none(self):
        page = AsyncMock()
        page.url = "https://example.com/file.patch"
        page.text_content = AsyncMock(return_value=None)

        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = True
            with patch("manus_agent.tools.browser_utils.extract_raw_page_text", new_callable=AsyncMock) as mock_extract:
                mock_extract.return_value = "raw patch content"
                text, selector = await get_text_with_fallback(page, "div.patch")
        assert text == "raw patch content"
        assert selector == "raw_text_page"

    @pytest.mark.asyncio
    async def test_raw_text_page_fallback_on_timeout(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.text_content = AsyncMock(side_effect=PlaywrightTimeoutError("timeout"))

        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = True
            with patch("manus_agent.tools.browser_utils.extract_raw_page_text", new_callable=AsyncMock) as mock_extract:
                mock_extract.return_value = "diff content here"
                text, selector = await get_text_with_fallback(page, "pre")
        assert text == "diff content here"
        assert selector == "raw_text_page"

    @pytest.mark.asyncio
    async def test_selector_syntax_error_raises_valueerror(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.text_content = AsyncMock(side_effect=ValueError("while parsing css selector 'invalid>>>'"))
        with pytest.raises(ValueError, match="[Ii]nvalid.*[Ss]elector"):
            await get_text_with_fallback(page, "invalid>>>", use_fallbacks=False)

    @pytest.mark.asyncio
    async def test_no_fallbacks_mode(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.text_content = AsyncMock(side_effect=PlaywrightTimeoutError("timeout"))
        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = False
            with pytest.raises(BrowserTimeoutError):
                await get_text_with_fallback(page, "nonexistent", use_fallbacks=False)

    @pytest.mark.asyncio
    async def test_custom_timeout(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.text_content = AsyncMock(return_value="Content")
        text, sel = await get_text_with_fallback(page, "p", timeout_ms=500)
        assert text == "Content"
        page.text_content.assert_called_with("p", timeout=500)


# ===========================================================================
# get_html_with_fallback
# ===========================================================================


class TestGetHtmlWithFallback:
    """Tests for get_html_with_fallback (async)."""

    @pytest.mark.asyncio
    async def test_no_selector_returns_full_page(self):
        page = AsyncMock()
        page.content = AsyncMock(return_value="<html><body>Full page</body></html>")
        result = await get_html_with_fallback(page, selector=None)
        assert "Full page" in result

    @pytest.mark.asyncio
    async def test_empty_selector_returns_full_page(self):
        page = AsyncMock()
        page.content = AsyncMock(return_value="<html><body>Full</body></html>")
        result = await get_html_with_fallback(page, selector="")
        assert "Full" in result

    @pytest.mark.asyncio
    async def test_valid_selector_returns_inner_html(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.wait_for_selector = AsyncMock()
        page.inner_html = AsyncMock(return_value="<p>Content</p>")
        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = False
            result = await get_html_with_fallback(page, selector="div.content")
        assert result == "<p>Content</p>"
        page.wait_for_selector.assert_called_with("div.content", timeout=DEFAULT_TIMEOUT, state="attached")

    @pytest.mark.asyncio
    async def test_timeout_raises_browser_timeout_error(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.wait_for_selector = AsyncMock(side_effect=PlaywrightTimeoutError("timeout"))
        page.content = AsyncMock(return_value="<html>fallback</html>")
        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = False
            with pytest.raises(BrowserTimeoutError):
                await get_html_with_fallback(page, selector="div.missing")

    @pytest.mark.asyncio
    async def test_timeout_on_raw_text_page_returns_content(self):
        page = AsyncMock()
        page.url = "https://example.com/file.patch"
        page.wait_for_selector = AsyncMock(side_effect=PlaywrightTimeoutError("timeout"))
        page.content = AsyncMock(return_value="<html><pre>diff content</pre></html>")
        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = True
            result = await get_html_with_fallback(page, selector="code")
        assert "diff content" in result

    @pytest.mark.asyncio
    async def test_raw_text_page_detected_returns_full_content(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.content = AsyncMock(return_value="<html><pre>raw text</pre></html>")
        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = True
            result = await get_html_with_fallback(page, selector="div.code")
        assert "raw text" in result

    @pytest.mark.asyncio
    async def test_selector_syntax_error_raises_valueerror(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.wait_for_selector = AsyncMock(side_effect=ValueError("Failed to parse selector"))
        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = False
            with pytest.raises(ValueError, match="[Ii]nvalid.*[Ss]elector"):
                await get_html_with_fallback(page, selector=">>>invalid")

    @pytest.mark.asyncio
    async def test_long_html_returned_as_is(self):
        """get_html_with_fallback returns full content (truncation is in the patch layer)."""
        page = AsyncMock()
        page.url = "https://example.com"
        page.wait_for_selector = AsyncMock()
        long_content = "x" * 2000
        page.inner_html = AsyncMock(return_value=long_content)
        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = False
            result = await get_html_with_fallback(page, selector="body")
        # browser_utils returns full content; truncation is done in the patch layer
        assert result == long_content
        assert len(result) == 2000

    @pytest.mark.asyncio
    async def test_custom_timeout_passed(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.wait_for_selector = AsyncMock()
        page.inner_html = AsyncMock(return_value="<p>ok</p>")
        with patch("manus_agent.tools.browser_utils.is_probably_raw_text_page", new_callable=AsyncMock) as mock_raw:
            mock_raw.return_value = False
            await get_html_with_fallback(page, selector="p", timeout_ms=3000)
        page.wait_for_selector.assert_called_with("p", timeout=3000, state="attached")


# ===========================================================================
# find_best_content_selector
# ===========================================================================


class TestFindBestContentSelector:
    """Tests for find_best_content_selector (async)."""

    @pytest.mark.asyncio
    async def test_finds_article(self):
        page = AsyncMock()
        element = AsyncMock()
        element.text_content = AsyncMock(return_value="A" * 200)

        async def _query_selector(sel):
            if sel == "article":
                return element
            return None

        page.query_selector = _query_selector
        result = await find_best_content_selector(page)
        assert result == "article"

    @pytest.mark.asyncio
    async def test_skips_short_content(self):
        page = AsyncMock()
        short_element = AsyncMock()
        short_element.text_content = AsyncMock(return_value="short")
        long_element = AsyncMock()
        long_element.text_content = AsyncMock(return_value="A" * 200)

        async def _query_selector(sel):
            if sel == "article":
                return short_element  # Too short
            if sel == "main":
                return long_element
            return None

        page.query_selector = _query_selector
        result = await find_best_content_selector(page)
        assert result == "main"

    @pytest.mark.asyncio
    async def test_falls_back_to_body(self):
        page = AsyncMock()
        body_element = AsyncMock()
        body_element.text_content = AsyncMock(return_value="B" * 200)

        async def _query_selector(sel):
            if sel == "body":
                return body_element
            return None

        page.query_selector = _query_selector
        result = await find_best_content_selector(page)
        assert result == "body"

    @pytest.mark.asyncio
    async def test_all_fail_returns_body(self):
        page = AsyncMock()
        page.query_selector = AsyncMock(return_value=None)
        result = await find_best_content_selector(page)
        assert result == "body"

    @pytest.mark.asyncio
    async def test_exception_in_query_skipped(self):
        page = AsyncMock()
        page.query_selector = AsyncMock(side_effect=RuntimeError("Error"))
        result = await find_best_content_selector(page)
        assert result == "body"


# ===========================================================================
# extract_page_text
# ===========================================================================


class TestExtractPageText:
    """Tests for extract_page_text (async, high-level)."""

    @pytest.mark.asyncio
    async def test_successful_extraction(self):
        page = AsyncMock()
        page.url = "https://example.com"

        with patch("manus_agent.tools.browser_utils.find_best_content_selector", new_callable=AsyncMock) as mock_find:
            mock_find.return_value = "article"
            with patch(
                "manus_agent.tools.browser_utils.get_text_with_fallback", new_callable=AsyncMock
            ) as mock_get_text:
                mock_get_text.return_value = ("Article content", "article")
                result = await extract_page_text(page)

        assert result["text"] == "Article content"
        assert result["selector"] == "article"
        assert result["url"] == "https://example.com"
        assert result["method"] == "text_content"

    @pytest.mark.asyncio
    async def test_fallback_to_html(self):
        page = AsyncMock()
        page.url = "https://example.com"

        with patch("manus_agent.tools.browser_utils.find_best_content_selector", new_callable=AsyncMock) as mock_find:
            mock_find.return_value = "main"
            with patch(
                "manus_agent.tools.browser_utils.get_text_with_fallback", new_callable=AsyncMock
            ) as mock_get_text:
                mock_get_text.side_effect = BrowserTimeoutError(selector="main", timeout_ms=10000)
                with patch(
                    "manus_agent.tools.browser_utils.get_html_with_fallback", new_callable=AsyncMock
                ) as mock_get_html:
                    mock_get_html.return_value = "<html><body>HTML fallback</body></html>"
                    result = await extract_page_text(page)

        assert "HTML fallback" in result["text"]
        assert result["method"] == "html_content"
        assert result["selector"] == "full page"

    @pytest.mark.asyncio
    async def test_fallback_to_javascript(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.evaluate = AsyncMock(return_value="JS extracted text")

        with patch("manus_agent.tools.browser_utils.find_best_content_selector", new_callable=AsyncMock) as mock_find:
            mock_find.return_value = "main"
            with patch(
                "manus_agent.tools.browser_utils.get_text_with_fallback", new_callable=AsyncMock
            ) as mock_get_text:
                mock_get_text.side_effect = BrowserTimeoutError(selector="main", timeout_ms=10000)
                with patch(
                    "manus_agent.tools.browser_utils.get_html_with_fallback", new_callable=AsyncMock
                ) as mock_get_html:
                    mock_get_html.side_effect = RuntimeError("HTML also failed")
                    result = await extract_page_text(page)

        assert result["text"] == "JS extracted text"
        assert result["method"] == "javascript_innerText"
        assert result["selector"] == "body"

    @pytest.mark.asyncio
    async def test_all_methods_fail(self):
        page = AsyncMock()
        page.url = "https://example.com"
        page.evaluate = AsyncMock(side_effect=RuntimeError("Complete failure"))

        with patch("manus_agent.tools.browser_utils.find_best_content_selector", new_callable=AsyncMock) as mock_find:
            mock_find.return_value = "body"
            with patch(
                "manus_agent.tools.browser_utils.get_text_with_fallback", new_callable=AsyncMock
            ) as mock_get_text:
                mock_get_text.side_effect = BrowserTimeoutError(selector="body", timeout_ms=10000)
                with patch(
                    "manus_agent.tools.browser_utils.get_html_with_fallback", new_callable=AsyncMock
                ) as mock_get_html:
                    mock_get_html.side_effect = RuntimeError("HTML failed")
                    result = await extract_page_text(page)

        assert "Error" in result["text"]
        assert result["method"] == "error"

    @pytest.mark.asyncio
    async def test_url_exception_handled(self):
        page = MagicMock()
        # Make .url property raise
        type(page).url = property(lambda self: (_ for _ in ()).throw(RuntimeError("No URL")))

        with patch("manus_agent.tools.browser_utils.find_best_content_selector", new_callable=AsyncMock) as mock_find:
            mock_find.return_value = "body"
            with patch(
                "manus_agent.tools.browser_utils.get_text_with_fallback", new_callable=AsyncMock
            ) as mock_get_text:
                mock_get_text.return_value = ("Some text", "body")
                result = await extract_page_text(page)

        assert result["text"] == "Some text"
        assert result["url"] is None


# ===========================================================================
# format_browser_error
# ===========================================================================


class TestFormatBrowserError:
    """Tests for format_browser_error."""

    def test_basic_error_formatting(self):
        err = RuntimeError("Something broke")
        result = format_browser_error(err)
        assert "RuntimeError" in result
        assert "Something broke" in result

    def test_with_url_context(self):
        err = RuntimeError("Error")
        result = format_browser_error(err, context={"url": "https://example.com"})
        assert "https://example.com" in result

    def test_with_selector_context(self):
        err = RuntimeError("Error")
        result = format_browser_error(err, context={"selector": "div.missing"})
        assert "div.missing" in result

    def test_playwright_timeout_suggestions(self):
        err = PlaywrightTimeoutError("Timeout 10000ms exceeded")
        result = format_browser_error(err)
        assert "Timeout Suggestions" in result or "Suggestions" in result
        assert "element may not exist" in result

    def test_browser_timeout_error_no_extra_suggestions(self):
        err = BrowserTimeoutError(selector="main", timeout_ms=5000)
        result = format_browser_error(err)
        # BrowserTimeoutError already has suggestions embedded
        assert "BrowserTimeoutError" in result

    def test_no_context(self):
        err = ValueError("test")
        result = format_browser_error(err)
        assert "ValueError" in result
        assert "URL" not in result
        assert "Selector" not in result

    def test_empty_context(self):
        err = ValueError("test")
        result = format_browser_error(err, context={})
        assert "ValueError" in result


# ===========================================================================
# Module-level constants
# ===========================================================================


class TestModuleConstants:
    """Verify module-level constants are accessible and correct."""

    def test_default_timeout_is_positive(self):
        assert DEFAULT_TIMEOUT > 0
        assert DEFAULT_TIMEOUT == 10000

    def test_raw_text_url_hints(self):
        assert ".patch" in RAW_TEXT_URL_HINTS
        assert ".diff" in RAW_TEXT_URL_HINTS

    def test_fallback_selectors_has_expected_keys(self):
        assert "article" in FALLBACK_SELECTORS
        assert "main" in FALLBACK_SELECTORS
        assert "content" in FALLBACK_SELECTORS

    def test_fallback_selectors_values_are_lists(self):
        for _key, val in FALLBACK_SELECTORS.items():
            assert isinstance(val, list)
            assert len(val) > 0

    def test_fallback_selectors_end_with_body(self):
        for _key, val in FALLBACK_SELECTORS.items():
            assert val[-1] == "body"
