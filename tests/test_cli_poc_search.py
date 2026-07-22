"""Comprehensive test suite for `manus-agent poc-search` CLI subcommand.

Tests cover:
- Parser construction and flag validation
- Routing from main() dispatcher
- CVE ID validation (invalid formats rejected)
- Text output formatting (table, exploited-in-wild banner, recent activity)
- JSON output mode
- --sources filtering
- Error/edge cases: no results, all sources failed, partial failures
- Exit codes

All HTTP calls are fully mocked — no real network traffic.
"""

import json
import sys
from io import StringIO
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke_poc_search(argv: list[str]) -> tuple[int, str, str]:
    """Call the poc-search CLI and capture stdout/stderr.

    Returns (exit_code, stdout, stderr).
    """
    from manus_agent import cli  # noqa: PLC0415

    exit_code = 0
    captured_out = StringIO()
    captured_err = StringIO()

    with (
        mock.patch.object(sys, "argv", ["manus-agent"] + argv),
        mock.patch.object(sys, "stdout", captured_out),
        mock.patch.object(sys, "stderr", captured_err),
    ):
        try:
            cli.main()
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else 0

    return exit_code, captured_out.getvalue(), captured_err.getvalue()


def _make_aggregate_result(
    cve_id: str = "CVE-2024-3094",
    total_found: int = 0,
    exploited_in_wild: bool = False,
    recent_activity: bool = False,
    sources_checked: list | None = None,
    sources_failed: list | None = None,
    results: list | None = None,
) -> dict:
    """Build a mock aggregate_poc_results return value."""
    return {
        "cve_id": cve_id.upper(),
        "total_found": total_found,
        "exploited_in_wild": exploited_in_wild,
        "recent_activity": recent_activity,
        "sources_checked": sources_checked or ["exploitdb", "github", "nvd", "trickest", "vulncheck_kev"],
        "sources_failed": sources_failed or [],
        "results": results or [],
    }


def _make_poc_result(
    source: str = "github",
    url: str = "https://github.com/example/poc",
    title: str = "example/poc",
    published: str | None = "2024-03-01",
    author: str | None = "hacker",
    tags: list | None = None,
    exploited_in_wild: bool = False,
) -> dict:
    return {
        "source": source,
        "url": url,
        "title": title,
        "published": published,
        "author": author,
        "tags": tags or ["github-repo"],
        "exploited_in_wild": exploited_in_wild,
    }


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestPocSearchParser:
    """Tests for _build_poc_search_parser."""

    def test_parser_requires_cve_id(self):
        from manus_agent.cli import _build_poc_search_parser

        p = _build_poc_search_parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_parser_accepts_cve_id(self):
        from manus_agent.cli import _build_poc_search_parser

        p = _build_poc_search_parser()
        args = p.parse_args(["CVE-2024-3094"])
        assert args.cve_id == "CVE-2024-3094"

    def test_parser_output_default_text(self):
        from manus_agent.cli import _build_poc_search_parser

        p = _build_poc_search_parser()
        args = p.parse_args(["CVE-2024-1234"])
        assert args.output == "text"

    def test_parser_output_json(self):
        from manus_agent.cli import _build_poc_search_parser

        p = _build_poc_search_parser()
        args = p.parse_args(["CVE-2024-1234", "--output", "json"])
        assert args.output == "json"

    def test_parser_output_invalid_choice(self):
        from manus_agent.cli import _build_poc_search_parser

        p = _build_poc_search_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["CVE-2024-1234", "--output", "csv"])

    def test_parser_sources_default_empty(self):
        from manus_agent.cli import _build_poc_search_parser

        p = _build_poc_search_parser()
        args = p.parse_args(["CVE-2024-1234"])
        assert args.sources == ""

    def test_parser_sources_flag(self):
        from manus_agent.cli import _build_poc_search_parser

        p = _build_poc_search_parser()
        args = p.parse_args(["CVE-2024-1234", "--sources", "trickest,github"])
        assert args.sources == "trickest,github"

    def test_parser_help_exits_zero(self):
        from manus_agent.cli import _build_poc_search_parser

        p = _build_poc_search_parser()
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


class TestPocSearchRouting:
    """Tests that main() dispatches to _run_poc_search correctly."""

    @mock.patch("manus_agent.cli._run_poc_search", return_value=0)
    def test_main_routes_poc_search(self, mock_run):
        exit_code, _, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        mock_run.assert_called_once()

    @mock.patch("manus_agent.cli._run_poc_search", return_value=0)
    def test_main_routes_poc_search_with_flags(self, mock_run):
        exit_code, _, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094", "--output", "json", "--sources", "github"])
        assert exit_code == 0
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# CVE ID validation tests
# ---------------------------------------------------------------------------


class TestPocSearchValidation:
    """Tests for CVE ID format validation."""

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_valid_cve_id_accepted(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result()
        exit_code, _, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        mock_agg.assert_called_once()

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_lowercase_cve_id_accepted(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result()
        exit_code, _, _ = _invoke_poc_search(["poc-search", "cve-2024-3094"])
        assert exit_code == 0

    def test_invalid_cve_id_not_cve_prefix(self):
        exit_code, _, stderr = _invoke_poc_search(["poc-search", "NOTCVE-2024-1234"])
        assert exit_code == 2  # argparse error

    def test_invalid_cve_id_no_digits(self):
        exit_code, _, stderr = _invoke_poc_search(["poc-search", "CVE-ABCD-XYZ"])
        assert exit_code == 2

    def test_invalid_cve_id_empty_string(self):
        exit_code, _, stderr = _invoke_poc_search(["poc-search", ""])
        assert exit_code == 2

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_cve_id_with_whitespace_stripped(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result()
        exit_code, _, _ = _invoke_poc_search(["poc-search", "  CVE-2024-3094  "])
        assert exit_code == 0
        # The function should have been called with stripped ID
        call_args = mock_agg.call_args
        assert call_args[0][0] == "CVE-2024-3094"


# ---------------------------------------------------------------------------
# JSON output tests
# ---------------------------------------------------------------------------


class TestPocSearchJsonOutput:
    """Tests for --output json mode."""

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_json_output_is_valid_json(self, mock_agg):
        result = _make_aggregate_result(
            total_found=2,
            results=[
                _make_poc_result(source="github", title="repo1"),
                _make_poc_result(source="exploitdb", title="exploit1"),
            ],
        )
        mock_agg.return_value = result
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094", "--output", "json"])
        assert exit_code == 0
        parsed = json.loads(stdout)
        assert parsed["cve_id"] == "CVE-2024-3094"
        assert parsed["total_found"] == 2
        assert len(parsed["results"]) == 2

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_json_output_empty_results(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(total_found=0, results=[])
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094", "--output", "json"])
        assert exit_code == 0
        parsed = json.loads(stdout)
        assert parsed["total_found"] == 0
        assert parsed["results"] == []

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_json_output_exploited_in_wild(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            exploited_in_wild=True,
            total_found=1,
            results=[_make_poc_result(exploited_in_wild=True, source="vulncheck_kev")],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094", "--output", "json"])
        assert exit_code == 0
        parsed = json.loads(stdout)
        assert parsed["exploited_in_wild"] is True

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_json_output_recent_activity(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(recent_activity=True)
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094", "--output", "json"])
        assert exit_code == 0
        parsed = json.loads(stdout)
        assert parsed["recent_activity"] is True

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_json_output_sources_failed(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            sources_failed=["exploitdb", "nvd"],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094", "--output", "json"])
        assert exit_code == 0
        parsed = json.loads(stdout)
        assert "exploitdb" in parsed["sources_failed"]
        assert "nvd" in parsed["sources_failed"]


# ---------------------------------------------------------------------------
# Text output tests
# ---------------------------------------------------------------------------


class TestPocSearchTextOutput:
    """Tests for default text output mode."""

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_header_shows_cve_id(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(cve_id="CVE-2024-3094")
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "CVE-2024-3094" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_shows_sources_checked(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            sources_checked=["github", "trickest", "nvd"],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "github" in stdout
        assert "trickest" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_shows_sources_failed(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            sources_failed=["exploitdb"],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "exploitdb" in stdout
        assert "failed" in stdout.lower()

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_no_results_message(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(total_found=0, results=[])
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "No PoC results found" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_exploited_in_wild_banner(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            exploited_in_wild=True,
            total_found=1,
            results=[_make_poc_result(exploited_in_wild=True)],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "EXPLOITED IN WILD" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_no_exploited_banner_when_false(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            exploited_in_wild=False,
            total_found=1,
            results=[_make_poc_result()],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "EXPLOITED IN WILD" not in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_recent_activity_indicator(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            recent_activity=True,
            total_found=1,
            results=[_make_poc_result()],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "Recent activity" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_no_recent_activity_when_false(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            recent_activity=False,
            total_found=1,
            results=[_make_poc_result()],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "Recent activity" not in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_results_table_header(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            total_found=1,
            results=[_make_poc_result()],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "Source" in stdout
        assert "Exploited?" in stdout
        assert "Title" in stdout
        assert "Date" in stdout
        assert "URL" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_result_row_contains_source(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            total_found=1,
            results=[_make_poc_result(source="exploitdb", title="EDB-12345")],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "exploitdb" in stdout
        assert "EDB-12345" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_result_row_exploited_yes(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            exploited_in_wild=True,
            total_found=1,
            results=[_make_poc_result(exploited_in_wild=True)],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "YES" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_result_row_exploited_no(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            total_found=1,
            results=[_make_poc_result(exploited_in_wild=False)],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        # "no" should appear in exploited column
        lines = stdout.split("\n")
        data_lines = [line for line in lines if "github" in line.lower() or "exploitdb" in line.lower()]
        assert any("no" in line for line in data_lines)

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_multiple_results(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            total_found=3,
            results=[
                _make_poc_result(source="trickest", title="trickest/cve index"),
                _make_poc_result(source="github", title="poc-exploit-repo"),
                _make_poc_result(source="nvd", title="https://nvd.example.com"),
            ],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "trickest" in stdout
        assert "github" in stdout
        assert "nvd" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_shows_result_count(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(total_found=7)
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "7" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_url_displayed(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            total_found=1,
            results=[_make_poc_result(url="https://github.com/hacker/cve-2024-3094-poc")],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "https://github.com/hacker/cve-2024-3094-poc" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_date_displayed(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            total_found=1,
            results=[_make_poc_result(published="2024-06-15")],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "2024-06-15" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_text_output_null_published_handled(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            total_found=1,
            results=[_make_poc_result(published=None)],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        # Should not crash and should still show the row


# ---------------------------------------------------------------------------
# Sources filter tests
# ---------------------------------------------------------------------------


class TestPocSearchSourcesFilter:
    """Tests for --sources flag."""

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_sources_empty_passes_none(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result()
        _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        call_args = mock_agg.call_args
        assert call_args[0][1] is None  # source_list is None = all

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_sources_single_value(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result()
        _invoke_poc_search(["poc-search", "CVE-2024-3094", "--sources", "github"])
        call_args = mock_agg.call_args
        assert call_args[0][1] == ["github"]

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_sources_multiple_values(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result()
        _invoke_poc_search(["poc-search", "CVE-2024-3094", "--sources", "trickest,exploitdb,nvd"])
        call_args = mock_agg.call_args
        assert call_args[0][1] == ["trickest", "exploitdb", "nvd"]

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_sources_with_spaces_trimmed(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result()
        _invoke_poc_search(["poc-search", "CVE-2024-3094", "--sources", " github , nvd "])
        call_args = mock_agg.call_args
        assert call_args[0][1] == ["github", "nvd"]

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_sources_whitespace_only_passes_none(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result()
        _invoke_poc_search(["poc-search", "CVE-2024-3094", "--sources", "   "])
        call_args = mock_agg.call_args
        assert call_args[0][1] is None


# ---------------------------------------------------------------------------
# Exit code tests
# ---------------------------------------------------------------------------


class TestPocSearchExitCodes:
    """Tests for correct exit codes."""

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_exit_zero_on_success(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result()
        exit_code, _, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_exit_zero_no_results(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(total_found=0, results=[])
        exit_code, _, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_exit_zero_with_results(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            total_found=1,
            results=[_make_poc_result()],
        )
        exit_code, _, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0

    def test_exit_two_on_invalid_cve(self):
        exit_code, _, _ = _invoke_poc_search(["poc-search", "INVALID"])
        assert exit_code == 2

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_exit_zero_json_mode(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result()
        exit_code, _, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094", "--output", "json"])
        assert exit_code == 0


# ---------------------------------------------------------------------------
# Edge cases and integration tests
# ---------------------------------------------------------------------------


class TestPocSearchEdgeCases:
    """Edge cases and real-world scenarios."""

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_all_sources_failed(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            total_found=0,
            sources_checked=["github", "nvd", "trickest"],
            sources_failed=["github", "nvd", "trickest"],
            results=[],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "No PoC results found" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_mixed_exploited_and_not(self, mock_agg):
        mock_agg.return_value = _make_aggregate_result(
            exploited_in_wild=True,
            total_found=3,
            results=[
                _make_poc_result(source="vulncheck_kev", exploited_in_wild=True, title="KEV entry"),
                _make_poc_result(source="github", exploited_in_wild=False, title="PoC repo"),
                _make_poc_result(source="exploitdb", exploited_in_wild=False, title="EDB entry"),
            ],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        assert "EXPLOITED IN WILD" in stdout
        assert "YES" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_very_long_title_truncated_in_table(self, mock_agg):
        long_title = "A" * 100
        mock_agg.return_value = _make_aggregate_result(
            total_found=1,
            results=[_make_poc_result(title=long_title)],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        # Title column is capped at 30 chars
        assert long_title not in stdout  # Full title shouldn't appear
        assert "A" * 30 in stdout  # Truncated version should

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_very_long_url_truncated_in_table(self, mock_agg):
        long_url = "https://example.com/" + "x" * 200
        mock_agg.return_value = _make_aggregate_result(
            total_found=1,
            results=[_make_poc_result(url=long_url)],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
        # URL column is capped at 70 chars
        assert long_url not in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_cve_2021_44228_example(self, mock_agg):
        """Integration test mimicking the README example for Log4Shell."""
        mock_agg.return_value = _make_aggregate_result(
            cve_id="CVE-2021-44228",
            exploited_in_wild=True,
            recent_activity=True,
            total_found=5,
            results=[
                _make_poc_result(
                    source="vulncheck_kev",
                    exploited_in_wild=True,
                    title="CVE-2021-44228 — VulnCheck KEV",
                    url="https://vulncheck.com/browse/kev/cve-2021-44228",
                ),
                _make_poc_result(
                    source="trickest",
                    title="CVE-2021-44228 — trickest/cve index",
                    url="https://github.com/trickest/cve/tree/main/2021/CVE-2021-44228",
                ),
                _make_poc_result(
                    source="github",
                    title="tangxiaofeng7/CVE-2021-44228-Apache-Log4j-Rce",
                    url="https://github.com/tangxiaofeng7/CVE-2021-44228-Apache-Log4j-Rce",
                    published="2021-12-10",
                ),
                _make_poc_result(
                    source="exploitdb",
                    title="Apache Log4j RCE",
                    url="https://www.exploit-db.com/exploits/50592",
                    published="2021-12-12",
                ),
                _make_poc_result(
                    source="nvd",
                    title="https://github.com/apache/logging-log4j2/pull/608",
                    url="https://github.com/apache/logging-log4j2/pull/608",
                ),
            ],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2021-44228"])
        assert exit_code == 0
        assert "CVE-2021-44228" in stdout
        assert "EXPLOITED IN WILD" in stdout
        assert "Recent activity" in stdout
        assert "5" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_json_output_preserves_all_fields(self, mock_agg):
        """Ensure JSON output preserves all aggregate fields faithfully."""
        full_result = _make_aggregate_result(
            cve_id="CVE-2024-1234",
            total_found=2,
            exploited_in_wild=True,
            recent_activity=True,
            sources_checked=["github", "nvd"],
            sources_failed=["exploitdb"],
            results=[
                _make_poc_result(source="github", author="researcher1", tags=["github-repo"]),
                _make_poc_result(source="nvd", author=None, tags=["exploit"]),
            ],
        )
        mock_agg.return_value = full_result
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-1234", "--output", "json"])
        assert exit_code == 0
        parsed = json.loads(stdout)
        assert parsed == full_result

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_sources_only_vulncheck_kev(self, mock_agg):
        """Test querying only vulncheck_kev source."""
        mock_agg.return_value = _make_aggregate_result(
            sources_checked=["vulncheck_kev"],
            total_found=1,
            exploited_in_wild=True,
            results=[_make_poc_result(source="vulncheck_kev", exploited_in_wild=True)],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094", "--sources", "vulncheck_kev"])
        assert exit_code == 0
        assert "vulncheck_kev" in stdout

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_result_with_empty_source_field(self, mock_agg):
        """Gracefully handle empty/None fields in results."""
        mock_agg.return_value = _make_aggregate_result(
            total_found=1,
            results=[
                {
                    "source": "",
                    "url": "",
                    "title": "",
                    "published": None,
                    "author": None,
                    "tags": [],
                    "exploited_in_wild": False,
                }
            ],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0  # Should not crash

    @mock.patch("manus_agent.tools.search_poc_sources.aggregate_poc_results")
    def test_sources_checked_empty_list(self, mock_agg):
        """Handle edge case where no sources were actually checked."""
        mock_agg.return_value = _make_aggregate_result(
            sources_checked=[],
            total_found=0,
            results=[],
        )
        exit_code, stdout, _ = _invoke_poc_search(["poc-search", "CVE-2024-3094"])
        assert exit_code == 0
