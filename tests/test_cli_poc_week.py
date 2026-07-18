"""Tests for the `manus-agent poc-week` CLI subcommand."""

import json
import sys
from unittest import mock

import pytest

from manus_agent import cli

# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestPocWeekParser:
    """Tests for _build_poc_week_parser."""

    def test_parser_exists(self):
        assert callable(cli._build_poc_week_parser)

    def test_parser_prog_name(self):
        p = cli._build_poc_week_parser()
        assert p.prog == "manus-agent poc-week"

    def test_parser_date_positional_optional(self):
        p = cli._build_poc_week_parser()
        args = p.parse_args([])
        assert args.date is None

    def test_parser_date_positional_provided(self):
        p = cli._build_poc_week_parser()
        args = p.parse_args(["2026-07-13"])
        assert args.date == "2026-07-13"

    def test_parser_output_default_text(self):
        p = cli._build_poc_week_parser()
        args = p.parse_args([])
        assert args.output == "text"

    def test_parser_output_json(self):
        p = cli._build_poc_week_parser()
        args = p.parse_args(["--output", "json"])
        assert args.output == "json"

    def test_parser_limit_default_zero(self):
        p = cli._build_poc_week_parser()
        args = p.parse_args([])
        assert args.limit == 0

    def test_parser_limit_flag(self):
        p = cli._build_poc_week_parser()
        args = p.parse_args(["--limit", "5"])
        assert args.limit == 5

    def test_parser_new_only_default_false(self):
        p = cli._build_poc_week_parser()
        args = p.parse_args([])
        assert args.new_only is False

    def test_parser_new_only_flag(self):
        p = cli._build_poc_week_parser()
        args = p.parse_args(["--new-only"])
        assert args.new_only is True

    def test_parser_all_flags_combined(self):
        p = cli._build_poc_week_parser()
        args = p.parse_args(["2026-06-01", "--output", "json", "--limit", "3", "--new-only"])
        assert args.date == "2026-06-01"
        assert args.output == "json"
        assert args.limit == 3
        assert args.new_only is True

    def test_parser_help_exits_zero(self):
        p = cli._build_poc_week_parser()
        with pytest.raises(SystemExit) as exc_info:
            p.parse_args(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


class TestPocWeekRouting:
    """Test that poc-week is registered and dispatched."""

    def test_poc_week_in_subcommands(self):
        assert "poc-week" in cli._SUBCOMMANDS

    def test_poc_week_dispatch_in_main(self):
        """main() routes 'poc-week' to _run_poc_week."""
        with mock.patch.object(cli, "_run_poc_week", return_value=0) as mocked:
            with mock.patch.object(sys, "argv", ["manus-agent", "poc-week", "2026-07-13"]):
                with pytest.raises(SystemExit) as exc_info:
                    cli.main()
                assert exc_info.value.code == 0
            mocked.assert_called_once_with(["2026-07-13"])


# ---------------------------------------------------------------------------
# Execution tests — _run_poc_week
# ---------------------------------------------------------------------------


_SAMPLE_RESULT = {
    "week_date": "2026-07-12",
    "url": "https://tonyharris.io/poc-week/poc-week-20260712/",
    "total": 3,
    "cves": [
        {
            "cve_id": "CVE-2026-1111",
            "mention_rank": 1,
            "severity": "CRITICAL",
            "products": "Apache HTTP Server",
            "description": "Remote code execution via crafted request",
            "poc_urls": ["https://github.com/example/poc-1111"],
            "is_new": True,
        },
        {
            "cve_id": "CVE-2026-2222",
            "mention_rank": 2,
            "severity": "HIGH",
            "products": "OpenSSL",
            "description": "Buffer overflow in TLS handshake",
            "poc_urls": ["https://github.com/example/poc-2222", "https://exploit-db.com/exploits/99999"],
            "is_new": False,
        },
        {
            "cve_id": "CVE-2026-3333",
            "mention_rank": 3,
            "severity": None,
            "products": None,
            "description": None,
            "poc_urls": [],
            "is_new": True,
        },
    ],
}


class TestRunPocWeek:
    """Tests for _run_poc_week execution logic."""

    def test_text_output_default(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "PoC Week Digest" in out
        assert "2026-07-12" in out
        assert "CVE-2026-1111" in out
        assert "CVE-2026-2222" in out
        assert "CVE-2026-3333" in out

    def test_text_output_with_date(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT) as mocked:
            rc = cli._run_poc_week(["2026-07-10"])
        assert rc == 0
        mocked.assert_called_once_with("2026-07-10")

    def test_json_output(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week(["--output", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["week_date"] == "2026-07-12"
        assert data["total"] == 3
        assert data["shown"] == 3
        assert len(data["cves"]) == 3

    def test_json_output_with_limit(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week(["--output", "json", "--limit", "2"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["shown"] == 2
        assert len(data["cves"]) == 2
        assert data["total"] == 3  # original total preserved

    def test_limit_flag_truncates(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week(["--limit", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2026-1111" in out
        assert "CVE-2026-2222" not in out

    def test_new_only_filter(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week(["--new-only"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2026-1111" in out
        assert "CVE-2026-3333" in out
        assert "CVE-2026-2222" not in out  # not new

    def test_new_only_with_limit(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week(["--new-only", "--limit", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2026-1111" in out
        assert "CVE-2026-3333" not in out  # limited to 1

    def test_error_from_tool(self, capsys):
        error_result = {"error": "No PoC Week issue found within 3 weeks of 2026-07-12."}
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=error_result):
            rc = cli._run_poc_week(["2020-01-01"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "No PoC Week issue found" in err

    def test_empty_cves_list(self, capsys):
        empty_result = {
            "week_date": "2026-07-12",
            "url": "https://tonyharris.io/poc-week/poc-week-20260712/",
            "total": 0,
            "cves": [],
        }
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=empty_result):
            rc = cli._run_poc_week([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No entries match" in out

    def test_new_only_with_no_new_entries(self, capsys):
        no_new_result = {
            "week_date": "2026-07-12",
            "url": "https://tonyharris.io/poc-week/poc-week-20260712/",
            "total": 1,
            "cves": [
                {
                    "cve_id": "CVE-2026-9999",
                    "mention_rank": 1,
                    "severity": "HIGH",
                    "products": "nginx",
                    "description": "Some vuln",
                    "poc_urls": [],
                    "is_new": False,
                },
            ],
        }
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=no_new_result):
            rc = cli._run_poc_week(["--new-only"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No entries match" in out

    def test_text_displays_poc_urls(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "https://github.com/example/poc-1111" in out
        assert "https://github.com/example/poc-2222" in out
        assert "https://exploit-db.com/exploits/99999" in out

    def test_text_displays_new_badge(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "\U0001f195" in out  # 🆕 emoji

    def test_text_handles_none_fields(self, capsys):
        """Entries with None severity/products/description display gracefully."""
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week([])
        assert rc == 0
        out = capsys.readouterr().out
        # CVE-2026-3333 has None fields — should show "—" placeholders
        assert "CVE-2026-3333" in out

    def test_no_date_passes_none(self):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT) as mocked:
            cli._run_poc_week([])
        mocked.assert_called_once_with(None)

    def test_json_output_includes_url(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week(["--output", "json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["url"] == "https://tonyharris.io/poc-week/poc-week-20260712/"

    def test_text_shows_source_url(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "tonyharris.io" in out

    def test_text_showing_count_with_limit(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week(["--limit", "2"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "top 2 of 3" in out

    def test_text_showing_new_only_count(self, capsys):
        with mock.patch("manus_agent.tools.get_poc_week.get_poc_week", return_value=_SAMPLE_RESULT):
            rc = cli._run_poc_week(["--new-only"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "NEW entries only" in out
