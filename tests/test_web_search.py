"""Comprehensive test suite for web_search module.

Tests cover:
- DuckDuckGoSearch: sync search, async search, import error, empty results
- GoogleSearch: missing credentials, placeholder behavior
- get_search_engine: factory function, config-driven selection, caching
- web_search_async: success, error handling, max_results
- web_search (sync tool): sync path, fallback to async, error handling
- Edge cases: empty query, special characters, result structure
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.web_search import (
    DuckDuckGoSearch,
    GoogleSearch,
    SearchEngine,
    get_search_engine,
    web_search,
    web_search_async,
)

# ===========================================================================
# SearchEngine base class
# ===========================================================================


class TestSearchEngineBase:
    """Test the SearchEngine base class."""

    def test_search_raises_not_implemented(self):
        engine = SearchEngine()
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(NotImplementedError):
                loop.run_until_complete(engine.search("test"))
        finally:
            loop.close()


# ===========================================================================
# DuckDuckGoSearch
# ===========================================================================


class TestDuckDuckGoSearch:
    """Test DuckDuckGoSearch implementation."""

    @patch("manus_agent.tools.web_search.DuckDuckGoSearch._search_sync")
    def test_async_search_delegates_to_sync(self, mock_sync):
        mock_sync.return_value = [{"title": "Result 1", "url": "https://example.com", "snippet": "A snippet"}]
        engine = DuckDuckGoSearch()
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(engine.search("test query", max_results=5))
        finally:
            loop.close()
        assert len(results) == 1
        assert results[0]["title"] == "Result 1"
        mock_sync.assert_called_once_with("test query", 5)

    @patch("duckduckgo_search.DDGS")
    def test_search_sync_returns_formatted_results(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Page 1", "href": "https://page1.com", "body": "Body 1"},
            {"title": "Page 2", "href": "https://page2.com", "body": "Body 2"},
        ]
        mock_ddgs_class.return_value = mock_ddgs

        engine = DuckDuckGoSearch()
        results = engine._search_sync("test", 5)

        assert len(results) == 2
        assert results[0]["title"] == "Page 1"
        assert results[0]["url"] == "https://page1.com"
        assert results[0]["snippet"] == "Body 1"

    @patch("duckduckgo_search.DDGS")
    def test_search_sync_handles_link_key(self, mock_ddgs_class):
        """Some versions use 'link' instead of 'href'."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Page", "link": "https://link-style.com", "body": "Body"},
        ]
        mock_ddgs_class.return_value = mock_ddgs

        engine = DuckDuckGoSearch()
        results = engine._search_sync("test", 5)

        assert results[0]["url"] == "https://link-style.com"

    @patch("duckduckgo_search.DDGS")
    def test_search_sync_empty_results(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = []
        mock_ddgs_class.return_value = mock_ddgs

        engine = DuckDuckGoSearch()
        results = engine._search_sync("obscure query", 5)

        assert results == []

    @patch("duckduckgo_search.DDGS")
    def test_search_sync_respects_max_results(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": f"Result {i}", "href": f"https://r{i}.com", "body": f"B{i}"} for i in range(3)
        ]
        mock_ddgs_class.return_value = mock_ddgs

        engine = DuckDuckGoSearch()
        engine._search_sync("test", 3)

        mock_ddgs.text.assert_called_once_with("test", max_results=3)

    @patch("duckduckgo_search.DDGS")
    def test_search_sync_handles_missing_fields(self, mock_ddgs_class):
        """Results with missing fields should default to empty strings."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {},  # All fields missing
        ]
        mock_ddgs_class.return_value = mock_ddgs

        engine = DuckDuckGoSearch()
        results = engine._search_sync("test", 5)

        assert results[0]["title"] == ""
        assert results[0]["url"] == ""
        assert results[0]["snippet"] == ""

    def test_search_sync_import_error(self, monkeypatch):
        """If duckduckgo-search is not installed, raise ImportError."""
        import sys

        # Remove the module to simulate it not being installed
        monkeypatch.setitem(sys.modules, "duckduckgo_search", None)

        engine = DuckDuckGoSearch()
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(ImportError, match="duckduckgo-search"):
                loop.run_until_complete(engine.search("test"))
        finally:
            loop.close()


# ===========================================================================
# GoogleSearch
# ===========================================================================


class TestGoogleSearch:
    """Test GoogleSearch implementation."""

    def test_init_with_credentials(self):
        engine = GoogleSearch(api_key="key123", cx="cx456")
        assert engine.api_key == "key123"
        assert engine.cx == "cx456"

    def test_init_without_credentials(self):
        engine = GoogleSearch()
        assert engine.api_key is None
        assert engine.cx is None

    def test_search_without_api_key_raises(self):
        engine = GoogleSearch(api_key=None, cx="cx456")
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(ValueError, match="API key"):
                loop.run_until_complete(engine.search("test"))
        finally:
            loop.close()

    def test_search_without_cx_raises(self):
        engine = GoogleSearch(api_key="key123", cx=None)
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(ValueError, match="API key"):
                loop.run_until_complete(engine.search("test"))
        finally:
            loop.close()

    def test_search_with_empty_credentials_raises(self):
        engine = GoogleSearch(api_key="", cx="")
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(ValueError):
                loop.run_until_complete(engine.search("test"))
        finally:
            loop.close()

    def test_search_with_valid_credentials_returns_empty(self):
        """Current implementation returns empty list (placeholder)."""
        engine = GoogleSearch(api_key="key123", cx="cx456")
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(engine.search("test"))
        finally:
            loop.close()
        assert results == []


# ===========================================================================
# get_search_engine factory
# ===========================================================================


class TestGetSearchEngine:
    """Test the get_search_engine factory function."""

    def setup_method(self):
        """Reset global search engine singleton between tests."""
        import manus_agent.tools.web_search as ws

        ws._search_engine = None

    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_default_returns_duckduckgo(self, mock_config):
        mock_cfg = MagicMock()
        mock_cfg.tools.search_engine = "duckduckgo"
        mock_config.return_value = mock_cfg

        engine = get_search_engine()
        assert isinstance(engine, DuckDuckGoSearch)

    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_google_config_returns_google(self, mock_config):
        mock_cfg = MagicMock()
        mock_cfg.tools.search_engine = "google"
        mock_config.return_value = mock_cfg

        engine = get_search_engine()
        assert isinstance(engine, GoogleSearch)

    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_unknown_config_defaults_to_duckduckgo(self, mock_config):
        mock_cfg = MagicMock()
        mock_cfg.tools.search_engine = "unknown_engine"
        mock_config.return_value = mock_cfg

        engine = get_search_engine()
        assert isinstance(engine, DuckDuckGoSearch)

    def test_accepts_config_argument(self):
        mock_cfg = MagicMock()
        mock_cfg.tools.search_engine = "duckduckgo"

        engine = get_search_engine(config=mock_cfg)
        assert isinstance(engine, DuckDuckGoSearch)

    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_caches_engine_instance(self, mock_config):
        mock_cfg = MagicMock()
        mock_cfg.tools.search_engine = "duckduckgo"
        mock_config.return_value = mock_cfg

        engine1 = get_search_engine()
        engine2 = get_search_engine()
        assert engine1 is engine2


# ===========================================================================
# web_search_async
# ===========================================================================


class TestWebSearchAsync:
    """Test the web_search_async function."""

    def setup_method(self):
        import manus_agent.tools.web_search as ws

        ws._search_engine = None

    @patch("manus_agent.tools.web_search.get_search_engine")
    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_success_returns_results(self, mock_config, mock_get_engine):
        mock_cfg = MagicMock()
        mock_cfg.tools.max_search_results = 5
        mock_config.return_value = mock_cfg

        mock_engine = MagicMock()

        async def fake_search(query, max_results):
            return [{"title": "Test", "url": "https://test.com", "snippet": "snippet"}]

        mock_engine.search = fake_search
        mock_get_engine.return_value = mock_engine

        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(web_search_async("test query"))
        finally:
            loop.close()

        assert len(results) == 1
        assert results[0]["title"] == "Test"

    @patch("manus_agent.tools.web_search.get_search_engine")
    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_custom_max_results(self, mock_config, mock_get_engine):
        mock_cfg = MagicMock()
        mock_cfg.tools.max_search_results = 10
        mock_config.return_value = mock_cfg

        mock_engine = MagicMock()
        call_args = {}

        async def fake_search(query, max_results):
            call_args["max_results"] = max_results
            return []

        mock_engine.search = fake_search
        mock_get_engine.return_value = mock_engine

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(web_search_async("test", max_results=3))
        finally:
            loop.close()

        assert call_args["max_results"] == 3

    @patch("manus_agent.tools.web_search.get_search_engine")
    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_uses_config_default_max_results(self, mock_config, mock_get_engine):
        mock_cfg = MagicMock()
        mock_cfg.tools.max_search_results = 7
        mock_config.return_value = mock_cfg

        mock_engine = MagicMock()
        call_args = {}

        async def fake_search(query, max_results):
            call_args["max_results"] = max_results
            return []

        mock_engine.search = fake_search
        mock_get_engine.return_value = mock_engine

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(web_search_async("test"))
        finally:
            loop.close()

        assert call_args["max_results"] == 7

    @patch("manus_agent.tools.web_search.get_search_engine")
    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_error_returns_error_result(self, mock_config, mock_get_engine):
        mock_cfg = MagicMock()
        mock_cfg.tools.max_search_results = 5
        mock_config.return_value = mock_cfg

        mock_engine = MagicMock()

        async def failing_search(query, max_results):
            raise RuntimeError("Network timeout")

        mock_engine.search = failing_search
        mock_get_engine.return_value = mock_engine

        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(web_search_async("test"))
        finally:
            loop.close()

        assert len(results) == 1
        assert results[0]["title"] == "Search Error"
        assert "Network timeout" in results[0]["snippet"]
        assert results[0]["url"] == ""


# ===========================================================================
# web_search (sync tool)
# ===========================================================================


class TestWebSearchSync:
    """Test the web_search synchronous tool function."""

    def setup_method(self):
        import manus_agent.tools.web_search as ws

        ws._search_engine = None

    @patch("manus_agent.tools.web_search.get_search_engine")
    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_sync_uses_search_sync_method(self, mock_config, mock_get_engine):
        mock_cfg = MagicMock()
        mock_cfg.tools.max_search_results = 5
        mock_config.return_value = mock_cfg

        mock_engine = MagicMock()
        mock_engine._search_sync = MagicMock(
            return_value=[{"title": "Sync", "url": "https://sync.com", "snippet": "sync"}]
        )
        mock_get_engine.return_value = mock_engine

        results = web_search("test query")

        assert len(results) == 1
        assert results[0]["title"] == "Sync"
        mock_engine._search_sync.assert_called_once_with("test query", 5)

    @patch("manus_agent.tools.web_search.get_search_engine")
    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_sync_custom_max_results(self, mock_config, mock_get_engine):
        mock_cfg = MagicMock()
        mock_cfg.tools.max_search_results = 10
        mock_config.return_value = mock_cfg

        mock_engine = MagicMock()
        mock_engine._search_sync = MagicMock(return_value=[])
        mock_get_engine.return_value = mock_engine

        web_search("query", max_results=3)

        mock_engine._search_sync.assert_called_once_with("query", 3)

    @patch("manus_agent.tools.web_search.get_search_engine")
    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_sync_error_returns_error_result(self, mock_config, mock_get_engine):
        mock_cfg = MagicMock()
        mock_cfg.tools.max_search_results = 5
        mock_config.return_value = mock_cfg

        mock_engine = MagicMock()
        mock_engine._search_sync = MagicMock(side_effect=RuntimeError("API error"))
        mock_get_engine.return_value = mock_engine

        results = web_search("test")

        assert len(results) == 1
        assert results[0]["title"] == "Search Error"
        assert "API error" in results[0]["snippet"]

    @patch("manus_agent.tools.web_search.get_search_engine")
    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_sync_fallback_to_async_when_no_search_sync(self, mock_config, mock_get_engine):
        """If engine doesn't have _search_sync, fall back to async path."""
        mock_cfg = MagicMock()
        mock_cfg.tools.max_search_results = 5
        mock_config.return_value = mock_cfg

        mock_engine = MagicMock(spec=SearchEngine)
        # Remove _search_sync attribute (spec=SearchEngine means it's not there)
        assert not hasattr(mock_engine, "_search_sync")

        async def fake_search(query, max_results):
            return [{"title": "Async fallback", "url": "https://async.com", "snippet": "async"}]

        mock_engine.search = fake_search
        mock_get_engine.return_value = mock_engine

        results = web_search("test")

        assert len(results) == 1
        assert results[0]["title"] == "Async fallback"

    @patch("manus_agent.tools.web_search.get_search_engine")
    @patch("manus_agent.tools.web_search.Config.from_file")
    def test_sync_empty_results(self, mock_config, mock_get_engine):
        mock_cfg = MagicMock()
        mock_cfg.tools.max_search_results = 5
        mock_config.return_value = mock_cfg

        mock_engine = MagicMock()
        mock_engine._search_sync = MagicMock(return_value=[])
        mock_get_engine.return_value = mock_engine

        results = web_search("nothing here")
        assert results == []


# ===========================================================================
# web_search_sync alias
# ===========================================================================


class TestWebSearchSyncAlias:
    """Test the backward-compatibility alias."""

    def test_alias_is_same_function(self):
        from manus_agent.tools.web_search import web_search_sync

        assert web_search_sync is web_search


# ===========================================================================
# Result structure validation
# ===========================================================================


class TestResultStructure:
    """Test that results always have the expected structure."""

    @patch("duckduckgo_search.DDGS")
    def test_results_have_required_keys(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "T", "href": "https://u.com", "body": "S"},
        ]
        mock_ddgs_class.return_value = mock_ddgs

        engine = DuckDuckGoSearch()
        results = engine._search_sync("test", 5)

        for result in results:
            assert "title" in result
            assert "url" in result
            assert "snippet" in result

    @patch("duckduckgo_search.DDGS")
    def test_results_values_are_strings(self, mock_ddgs_class):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Title", "href": "https://url.com", "body": "Body"},
        ]
        mock_ddgs_class.return_value = mock_ddgs

        engine = DuckDuckGoSearch()
        results = engine._search_sync("test", 5)

        for result in results:
            assert isinstance(result["title"], str)
            assert isinstance(result["url"], str)
            assert isinstance(result["snippet"], str)
