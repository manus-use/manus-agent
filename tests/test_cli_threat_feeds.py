"""Tests for manus-agent threat-feeds CLI subcommand."""

import json
import sys
from unittest.mock import patch

import pytest

from manus_agent.cli import _build_threat_feeds_parser, _run_threat_feeds


class TestBuildThreatFeedsParser:
    """Tests for _build_threat_feeds_parser."""

    def test_parser_prog(self):
        p = _build_threat_feeds_parser()
        assert p.prog == "manus-agent threat-feeds"

    def test_parser_requires_cve_id(self):
        p = _build_threat_feeds_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_parser_accepts_cve_id(self):
        p = _build_threat_feeds_parser()
        args = p.parse_args(["CVE-2024-3094"])
        assert args.cve_id == "CVE-2024-3094"

    def test_parser_default_output_is_text(self):
        p = _build_threat_feeds_parser()
        args = p.parse_args(["CVE-2024-3094"])
        assert args.output == "text"

    def test_parser_accepts_json_output(self):
        p = _build_threat_feeds_parser()
        args = p.parse_args(["CVE-2024-3094", "--output", "json"])
        assert args.output == "json"

    def test_parser_rejects_invalid_output(self):
        p = _build_threat_feeds_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["CVE-2024-3094", "--output", "xml"])


class TestRunThreatFeedsValidation:
    """Tests for CVE ID validation in _run_threat_feeds."""

    def test_invalid_cve_id_exits_with_error(self):
        with pytest.raises(SystemExit) as exc_info:
            _run_threat_feeds(["not-a-cve"])
        assert exc_info.value.code == 2

    def test_empty_string_exits_with_error(self):
        with pytest.raises(SystemExit) as exc_info:
            _run_threat_feeds([""])
        assert exc_info.value.code == 2

    def test_partial_cve_exits_with_error(self):
        with pytest.raises(SystemExit) as exc_info:
            _run_threat_feeds(["CVE-2024"])
        assert exc_info.value.code == 2

    def test_no_args_exits_with_error(self):
        with pytest.raises(SystemExit) as exc_info:
            _run_threat_feeds([])
        assert exc_info.value.code == 2


class TestRunThreatFeedsNoResults:
    """Tests for _run_threat_feeds when no intelligence is found."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.fetch_threat_intelligence")
    def test_no_results_text_output(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "summary": "No direct threat intelligence found for CVE-2024-9999 in curated feeds.",
            "intelligence": [],
            "errors": [],
        }
        exit_code = _run_threat_feeds(["CVE-2024-9999"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No direct threat intelligence found" in captured.out
        assert "CVE-2024-9999" in captured.out

    @patch("manus_agent.tools.query_threat_intelligence_feeds.fetch_threat_intelligence")
    def test_no_results_json_output(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "summary": "No direct threat intelligence found for CVE-2024-9999 in curated feeds.",
            "intelligence": [],
            "errors": [],
        }
        exit_code = _run_threat_feeds(["CVE-2024-9999", "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["intelligence"] == []
        assert "No direct threat intelligence found" in data["summary"]


class TestRunThreatFeedsWithResults:
    """Tests for _run_threat_feeds when intelligence is found."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.fetch_threat_intelligence")
    def test_results_text_output(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "summary": "Found relevant threat intelligence for CVE-2024-3094 in 1 feed(s).",
            "intelligence": [
                {
                    "feed_name": "CISA Cybersecurity Advisories",
                    "feed_url": "https://www.cisa.gov/cybersecurity-advisories/all.xml",
                    "cve_found": "CVE-2024-3094",
                    "snippet": "...advisory about CVE-2024-3094 affecting xz-utils...",
                }
            ],
            "errors": [],
        }
        exit_code = _run_threat_feeds(["CVE-2024-3094"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "CISA Cybersecurity Advisories" in captured.out
        assert "CVE-2024-3094" in captured.out
        assert "Threat Intelligence Feeds" in captured.out

    @patch("manus_agent.tools.query_threat_intelligence_feeds.fetch_threat_intelligence")
    def test_results_json_output(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "summary": "Found relevant threat intelligence for CVE-2024-3094 in 1 feed(s).",
            "intelligence": [
                {
                    "feed_name": "CISA Cybersecurity Advisories",
                    "feed_url": "https://www.cisa.gov/cybersecurity-advisories/all.xml",
                    "cve_found": "CVE-2024-3094",
                    "snippet": "...advisory about CVE-2024-3094...",
                }
            ],
            "errors": [],
        }
        exit_code = _run_threat_feeds(["CVE-2024-3094", "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["intelligence"]) == 1
        assert data["intelligence"][0]["feed_name"] == "CISA Cybersecurity Advisories"

    @patch("manus_agent.tools.query_threat_intelligence_feeds.fetch_threat_intelligence")
    def test_multiple_feeds_text_output(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "summary": "Found relevant threat intelligence for CVE-2024-3094 in 2 feed(s).",
            "intelligence": [
                {
                    "feed_name": "CISA Cybersecurity Advisories",
                    "feed_url": "https://www.cisa.gov/cybersecurity-advisories/all.xml",
                    "cve_found": "CVE-2024-3094",
                    "snippet": "snippet1",
                },
                {
                    "feed_name": "Another Feed",
                    "feed_url": "https://example.com/feed.xml",
                    "cve_found": "CVE-2024-3094",
                    "snippet": "snippet2",
                },
            ],
            "errors": [],
        }
        exit_code = _run_threat_feeds(["CVE-2024-3094"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "[1]" in captured.out
        assert "[2]" in captured.out
        assert "Another Feed" in captured.out


class TestRunThreatFeedsWithErrors:
    """Tests for _run_threat_feeds when feeds have errors."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.fetch_threat_intelligence")
    def test_errors_displayed_in_text_output(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "summary": "No direct threat intelligence found for CVE-2024-9999 in curated feeds.",
            "intelligence": [],
            "errors": [
                {"feed_name": "Bad Feed", "error": "Connection timeout"},
            ],
        }
        exit_code = _run_threat_feeds(["CVE-2024-9999"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Errors encountered" in captured.out
        assert "Bad Feed" in captured.out
        assert "Connection timeout" in captured.out

    @patch("manus_agent.tools.query_threat_intelligence_feeds.fetch_threat_intelligence")
    def test_errors_included_in_json_output(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "summary": "No direct threat intelligence found for CVE-2024-9999 in curated feeds.",
            "intelligence": [],
            "errors": [
                {"feed_name": "Bad Feed", "error": "Connection timeout"},
            ],
        }
        exit_code = _run_threat_feeds(["CVE-2024-9999", "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["errors"]) == 1
        assert data["errors"][0]["feed_name"] == "Bad Feed"

    @patch("manus_agent.tools.query_threat_intelligence_feeds.fetch_threat_intelligence")
    def test_partial_success_with_errors(self, mock_fetch, capsys):
        mock_fetch.return_value = {
            "summary": "Found relevant threat intelligence for CVE-2024-3094 in 1 feed(s).",
            "intelligence": [
                {
                    "feed_name": "CISA Cybersecurity Advisories",
                    "feed_url": "https://www.cisa.gov/feed.xml",
                    "cve_found": "CVE-2024-3094",
                    "snippet": "found it here...",
                }
            ],
            "errors": [
                {"feed_name": "Failing Feed", "error": "HTTP 503"},
            ],
        }
        exit_code = _run_threat_feeds(["CVE-2024-3094"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "CISA Cybersecurity Advisories" in captured.out
        assert "Failing Feed" in captured.out
        assert "HTTP 503" in captured.out


class TestRunThreatFeedsSnippetTruncation:
    """Tests for snippet display truncation."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.fetch_threat_intelligence")
    def test_long_snippet_is_truncated(self, mock_fetch, capsys):
        long_snippet = "A" * 300
        mock_fetch.return_value = {
            "summary": "Found relevant threat intelligence for CVE-2024-3094 in 1 feed(s).",
            "intelligence": [
                {
                    "feed_name": "Test Feed",
                    "feed_url": "https://example.com/feed.xml",
                    "cve_found": "CVE-2024-3094",
                    "snippet": long_snippet,
                }
            ],
            "errors": [],
        }
        exit_code = _run_threat_feeds(["CVE-2024-3094"])
        assert exit_code == 0
        captured = capsys.readouterr()
        # The displayed snippet should be truncated to 200 chars + "..."
        assert "A" * 200 + "..." in captured.out
        # Should NOT contain the full 300 chars
        assert "A" * 300 not in captured.out

    @patch("manus_agent.tools.query_threat_intelligence_feeds.fetch_threat_intelligence")
    def test_short_snippet_is_not_truncated(self, mock_fetch, capsys):
        short_snippet = "Short CVE mention"
        mock_fetch.return_value = {
            "summary": "Found relevant threat intelligence for CVE-2024-3094 in 1 feed(s).",
            "intelligence": [
                {
                    "feed_name": "Test Feed",
                    "feed_url": "https://example.com/feed.xml",
                    "cve_found": "CVE-2024-3094",
                    "snippet": short_snippet,
                }
            ],
            "errors": [],
        }
        exit_code = _run_threat_feeds(["CVE-2024-3094"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Short CVE mention" in captured.out


class TestFetchThreatIntelligence:
    """Tests for the fetch_threat_intelligence helper function."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    def test_feed_match(self, mock_get):
        from manus_agent.tools.query_threat_intelligence_feeds import (
            fetch_threat_intelligence,
        )

        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.text = "This advisory covers CVE-2024-3094 affecting xz-utils"

        result = fetch_threat_intelligence("CVE-2024-3094")
        assert len(result["intelligence"]) == 1
        assert result["intelligence"][0]["cve_found"] == "CVE-2024-3094"
        assert result["errors"] == []

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    def test_no_match(self, mock_get):
        from manus_agent.tools.query_threat_intelligence_feeds import (
            fetch_threat_intelligence,
        )

        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.text = "No CVEs mentioned here at all"

        result = fetch_threat_intelligence("CVE-2024-9999")
        assert result["intelligence"] == []
        assert "No direct threat intelligence found" in result["summary"]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    def test_case_insensitive_match(self, mock_get):
        from manus_agent.tools.query_threat_intelligence_feeds import (
            fetch_threat_intelligence,
        )

        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.text = "Contains cve-2024-3094 in lower case"

        result = fetch_threat_intelligence("CVE-2024-3094")
        assert len(result["intelligence"]) == 1

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    def test_request_error_captured(self, mock_get):
        import requests

        from manus_agent.tools.query_threat_intelligence_feeds import (
            fetch_threat_intelligence,
        )

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = fetch_threat_intelligence("CVE-2024-3094")
        assert result["intelligence"] == []
        assert len(result["errors"]) == 1
        assert "Connection refused" in result["errors"][0]["error"]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    def test_timeout_error_captured(self, mock_get):
        import requests

        from manus_agent.tools.query_threat_intelligence_feeds import (
            fetch_threat_intelligence,
        )

        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        result = fetch_threat_intelligence("CVE-2024-3094")
        assert result["intelligence"] == []
        assert len(result["errors"]) == 1
        assert "timed out" in result["errors"][0]["error"]

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    def test_custom_feeds(self, mock_get):
        from manus_agent.tools.query_threat_intelligence_feeds import (
            fetch_threat_intelligence,
        )

        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.text = "Advisory: CVE-2024-3094 is critical"

        custom_feeds = [
            {"name": "Custom Feed", "url": "https://custom.example.com/feed.xml", "type": "rss"},
        ]
        result = fetch_threat_intelligence("CVE-2024-3094", feeds=custom_feeds)
        assert len(result["intelligence"]) == 1
        assert result["intelligence"][0]["feed_name"] == "Custom Feed"

    @patch("manus_agent.tools.query_threat_intelligence_feeds.requests.get")
    def test_snippet_extraction_bounds(self, mock_get):
        from manus_agent.tools.query_threat_intelligence_feeds import (
            fetch_threat_intelligence,
        )

        # CVE at the very start of content (no 50 chars before it)
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = lambda: None
        mock_get.return_value.text = "CVE-2024-3094 is critical"

        result = fetch_threat_intelligence("CVE-2024-3094")
        assert len(result["intelligence"]) == 1
        # Snippet should start from beginning since CVE is at index 0
        assert "CVE-2024-3094" in result["intelligence"][0]["snippet"]


class TestMainDispatch:
    """Tests for threat-feeds dispatch in main()."""

    @patch("manus_agent.tools.query_threat_intelligence_feeds.fetch_threat_intelligence")
    def test_main_dispatches_threat_feeds(self, mock_fetch):
        mock_fetch.return_value = {
            "summary": "No results",
            "intelligence": [],
            "errors": [],
        }
        with patch.object(sys, "argv", ["manus-agent", "threat-feeds", "CVE-2024-3094"]):
            from manus_agent.cli import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
