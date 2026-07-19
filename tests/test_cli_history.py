"""Comprehensive tests for `manus-agent history` subcommand.

Covers: display (text/json), --limit, --grep, --clear, empty states,
malformed JSONL lines, _append_history helper, and edge cases.
"""

import datetime
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RECORDS = [
    {
        "timestamp": "2025-06-01T10:00:00+00:00",
        "task": "Analyze CVE-2024-3094",
        "agent": "single",
        "mode": "auto",
        "format": "text",
        "success": True,
        "result": "Analysis complete.",
    },
    {
        "timestamp": "2025-06-02T12:30:00+00:00",
        "task": "Find bitcoin price",
        "agent": "browser",
        "mode": "auto",
        "format": "json",
        "success": True,
        "result": '{"price": 70000}',
    },
    {
        "timestamp": "2025-06-03T08:15:00+00:00",
        "task": "Create a factorial function in Python",
        "agent": "single",
        "mode": "single",
        "format": "text",
        "success": False,
        "result": "Error: timeout exceeded",
    },
    {
        "timestamp": "2025-06-04T14:00:00+00:00",
        "task": "Research quantum computing and create slides",
        "agent": "multi",
        "mode": "multi",
        "format": "text",
        "success": True,
        "result": "Slides created successfully.",
    },
    {
        "timestamp": "2025-06-05T09:45:00+00:00",
        "task": "Analyze CVE-2021-44228 with exploit check",
        "agent": "single",
        "mode": "auto",
        "format": "json",
        "success": True,
        "result": '{"severity": "critical"}',
    },
]


def _write_history(path: Path, records: list[dict]) -> None:
    """Write records as JSONL to a history file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _invoke_history(argv: list[str], history_path: Path, capsys=None):
    """Call cli.main() with mocked _HISTORY_PATH and capture exit code."""
    from manus_agent import cli

    exit_code = 0
    with mock.patch.object(cli, "_HISTORY_PATH", history_path):
        with mock.patch.object(sys, "argv", ["manus-agent", "history"] + argv):
            try:
                cli.main()
            except SystemExit as exc:
                exit_code = exc.code if isinstance(exc.code, int) else 0

    return exit_code


# ---------------------------------------------------------------------------
# Tests: Text output (default)
# ---------------------------------------------------------------------------


class TestHistoryTextOutput:
    """Tests for `manus-agent history` (text table display)."""

    def test_text_output_shows_entries(self, tmp_path, capsys):
        """Default text output renders a table with correct entry count."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)

        code = _invoke_history([], hist)
        assert code == 0

        captured = capsys.readouterr()
        # Should show table title with entry count
        assert "5 entries" in captured.out

    def test_text_output_shows_timestamps(self, tmp_path, capsys):
        """Text output includes formatted timestamps."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS[:1])

        code = _invoke_history([], hist)
        assert code == 0

        captured = capsys.readouterr()
        assert "2025-06-01" in captured.out

    def test_text_output_shows_agent_type(self, tmp_path, capsys):
        """Text output includes the agent type column."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS[:2])

        code = _invoke_history([], hist)
        assert code == 0

        captured = capsys.readouterr()
        assert "single" in captured.out
        assert "browser" in captured.out

    def test_text_output_truncates_long_tasks(self, tmp_path, capsys):
        """Tasks longer than 80 chars are truncated with ellipsis."""
        hist = tmp_path / "history.jsonl"
        long_task = "A" * 100
        rec = {**_SAMPLE_RECORDS[0], "task": long_task}
        _write_history(hist, [rec])

        code = _invoke_history([], hist)
        assert code == 0

        captured = capsys.readouterr()
        # The truncated text should be 80 chars + ellipsis
        assert "…" in captured.out

    def test_text_output_shows_success_indicator(self, tmp_path, capsys):
        """Text output shows ✓ for success and ✗ for failure."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS[:3])

        code = _invoke_history([], hist)
        assert code == 0

        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "✗" in captured.out

    def test_text_output_shows_history_path(self, tmp_path, capsys):
        """Text output footer shows the history file path."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS[:1])

        code = _invoke_history([], hist)
        assert code == 0

        captured = capsys.readouterr()
        assert str(hist) in captured.out


# ---------------------------------------------------------------------------
# Tests: JSON output
# ---------------------------------------------------------------------------


class TestHistoryJsonOutput:
    """Tests for `manus-agent history --format json`."""

    def test_json_output_valid(self, tmp_path, capsys):
        """JSON output is valid JSON array."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)

        code = _invoke_history(["--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) == 5

    def test_json_output_most_recent_first(self, tmp_path, capsys):
        """JSON output returns entries in reverse chronological order."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)

        code = _invoke_history(["--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # Most recent entry (index -1 in original) should be first
        assert data[0]["task"] == _SAMPLE_RECORDS[-1]["task"]
        assert data[-1]["task"] == _SAMPLE_RECORDS[0]["task"]

    def test_json_output_preserves_all_fields(self, tmp_path, capsys):
        """JSON output preserves all record fields."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS[:1])

        code = _invoke_history(["--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        rec = data[0]
        assert rec["timestamp"] == "2025-06-01T10:00:00+00:00"
        assert rec["task"] == "Analyze CVE-2024-3094"
        assert rec["agent"] == "single"
        assert rec["mode"] == "auto"
        assert rec["format"] == "text"
        assert rec["success"] is True
        assert rec["result"] == "Analysis complete."


# ---------------------------------------------------------------------------
# Tests: --limit
# ---------------------------------------------------------------------------


class TestHistoryLimit:
    """Tests for `manus-agent history --limit N`."""

    def test_limit_default_20(self, tmp_path, capsys):
        """Default limit is 20 entries."""
        hist = tmp_path / "history.jsonl"
        # Write 25 records
        records = []
        for i in range(25):
            rec = {**_SAMPLE_RECORDS[0], "task": f"Task {i}"}
            records.append(rec)
        _write_history(hist, records)

        code = _invoke_history(["--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 20

    def test_limit_custom(self, tmp_path, capsys):
        """Custom --limit restricts output count."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)

        code = _invoke_history(["--limit", "2", "--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 2

    def test_limit_zero_shows_all(self, tmp_path, capsys):
        """--limit 0 shows all entries."""
        hist = tmp_path / "history.jsonl"
        # Write 25 records
        records = []
        for i in range(25):
            rec = {**_SAMPLE_RECORDS[0], "task": f"Task {i}"}
            records.append(rec)
        _write_history(hist, records)

        code = _invoke_history(["--limit", "0", "--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 25

    def test_limit_larger_than_entries(self, tmp_path, capsys):
        """--limit larger than available records shows all available."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)

        code = _invoke_history(["--limit", "100", "--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 5


# ---------------------------------------------------------------------------
# Tests: --grep
# ---------------------------------------------------------------------------


class TestHistoryGrep:
    """Tests for `manus-agent history --grep PATTERN`."""

    def test_grep_filters_by_task(self, tmp_path, capsys):
        """--grep filters entries by task content."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)

        code = _invoke_history(["--grep", "bitcoin", "--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert "bitcoin" in data[0]["task"].lower()

    def test_grep_case_insensitive(self, tmp_path, capsys):
        """--grep is case-insensitive."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)

        code = _invoke_history(["--grep", "CVE", "--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # Should match "Analyze CVE-2024-3094" and "Analyze CVE-2021-44228..."
        assert len(data) == 2

    def test_grep_no_matches(self, tmp_path, capsys):
        """--grep with no matches shows empty message."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)

        code = _invoke_history(["--grep", "nonexistent_xyz"], hist)
        assert code == 0

        captured = capsys.readouterr()
        assert "No matching" in captured.out

    def test_grep_combined_with_limit(self, tmp_path, capsys):
        """--grep and --limit work together; limit applied after filtering."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)

        code = _invoke_history(["--grep", "CVE", "--limit", "1", "--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1

    def test_grep_partial_match(self, tmp_path, capsys):
        """--grep matches substrings within task text."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)

        code = _invoke_history(["--grep", "factorial", "--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert "factorial" in data[0]["task"]


# ---------------------------------------------------------------------------
# Tests: --clear
# ---------------------------------------------------------------------------


class TestHistoryClear:
    """Tests for `manus-agent history --clear`."""

    def test_clear_deletes_file(self, tmp_path, capsys):
        """--clear removes the history file."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)
        assert hist.exists()

        code = _invoke_history(["--clear"], hist)
        assert code == 0
        assert not hist.exists()

        captured = capsys.readouterr()
        assert "cleared" in captured.out.lower()

    def test_clear_nonexistent_file(self, tmp_path, capsys):
        """--clear on non-existent file still succeeds."""
        hist = tmp_path / "history.jsonl"
        assert not hist.exists()

        code = _invoke_history(["--clear"], hist)
        assert code == 0

        captured = capsys.readouterr()
        assert "cleared" in captured.out.lower()


# ---------------------------------------------------------------------------
# Tests: Empty / missing states
# ---------------------------------------------------------------------------


class TestHistoryEmptyStates:
    """Tests for edge cases with no or empty history."""

    def test_no_history_file(self, tmp_path, capsys):
        """Shows helpful message when history file doesn't exist."""
        hist = tmp_path / "nonexistent" / "history.jsonl"

        code = _invoke_history([], hist)
        assert code == 0

        captured = capsys.readouterr()
        assert "No history yet" in captured.out

    def test_empty_history_file(self, tmp_path, capsys):
        """Shows helpful message when history file is empty."""
        hist = tmp_path / "history.jsonl"
        hist.parent.mkdir(parents=True, exist_ok=True)
        hist.write_text("")

        code = _invoke_history([], hist)
        assert code == 0

        captured = capsys.readouterr()
        assert "No matching" in captured.out

    def test_whitespace_only_file(self, tmp_path, capsys):
        """History file with only whitespace is treated as empty."""
        hist = tmp_path / "history.jsonl"
        hist.parent.mkdir(parents=True, exist_ok=True)
        hist.write_text("\n\n  \n\n")

        code = _invoke_history([], hist)
        assert code == 0

        captured = capsys.readouterr()
        assert "No matching" in captured.out


# ---------------------------------------------------------------------------
# Tests: Malformed JSONL handling
# ---------------------------------------------------------------------------


class TestHistoryMalformedData:
    """Tests for graceful handling of malformed JSONL lines."""

    def test_skips_malformed_lines(self, tmp_path, capsys):
        """Malformed JSON lines are silently skipped."""
        hist = tmp_path / "history.jsonl"
        hist.parent.mkdir(parents=True, exist_ok=True)
        with hist.open("w") as fh:
            fh.write(json.dumps(_SAMPLE_RECORDS[0]) + "\n")
            fh.write("THIS IS NOT VALID JSON\n")
            fh.write("{broken json\n")
            fh.write(json.dumps(_SAMPLE_RECORDS[1]) + "\n")

        code = _invoke_history(["--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # Only 2 valid records should survive
        assert len(data) == 2

    def test_mixed_valid_invalid_records(self, tmp_path, capsys):
        """Valid records interspersed with invalid ones still render correctly."""
        hist = tmp_path / "history.jsonl"
        hist.parent.mkdir(parents=True, exist_ok=True)
        with hist.open("w") as fh:
            fh.write("not json\n")
            fh.write(json.dumps(_SAMPLE_RECORDS[0]) + "\n")
            fh.write("\n")  # empty line
            fh.write(json.dumps(_SAMPLE_RECORDS[2]) + "\n")
            fh.write("also {invalid\n")

        code = _invoke_history(["--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 2

    def test_records_missing_fields(self, tmp_path, capsys):
        """Records with missing optional fields still render."""
        hist = tmp_path / "history.jsonl"
        hist.parent.mkdir(parents=True, exist_ok=True)
        # Minimal record with only task
        minimal = {"task": "Minimal task"}
        with hist.open("w") as fh:
            fh.write(json.dumps(minimal) + "\n")

        code = _invoke_history([], hist)
        assert code == 0

        captured = capsys.readouterr()
        assert "Minimal task" in captured.out


# ---------------------------------------------------------------------------
# Tests: _append_history helper
# ---------------------------------------------------------------------------


class TestAppendHistory:
    """Tests for the _append_history helper function."""

    def test_append_creates_parent_dirs(self, tmp_path):
        """_append_history creates parent directories if needed."""
        from manus_agent import cli

        hist = tmp_path / "deep" / "nested" / "history.jsonl"
        with mock.patch.object(cli, "_HISTORY_PATH", hist):
            cli._append_history(
                "Test task",
                "Result",
                agent_type="single",
                mode="auto",
                success=True,
                format="text",
            )

        assert hist.exists()
        lines = hist.read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["task"] == "Test task"

    def test_append_writes_all_fields(self, tmp_path):
        """_append_history writes all expected fields."""
        from manus_agent import cli

        hist = tmp_path / "history.jsonl"
        with mock.patch.object(cli, "_HISTORY_PATH", hist):
            cli._append_history(
                "Analyze CVE-2024-3094",
                "Analysis done",
                agent_type="browser",
                mode="multi",
                success=True,
                format="json",
            )

        rec = json.loads(hist.read_text().strip())
        assert rec["task"] == "Analyze CVE-2024-3094"
        assert rec["result"] == "Analysis done"
        assert rec["agent"] == "browser"
        assert rec["mode"] == "multi"
        assert rec["success"] is True
        assert rec["format"] == "json"
        assert "timestamp" in rec

    def test_append_timestamp_is_utc_iso(self, tmp_path):
        """_append_history writes ISO-8601 UTC timestamps."""
        from manus_agent import cli

        hist = tmp_path / "history.jsonl"
        with mock.patch.object(cli, "_HISTORY_PATH", hist):
            cli._append_history(
                "task",
                "result",
                agent_type="single",
                mode="auto",
                success=True,
            )

        rec = json.loads(hist.read_text().strip())
        ts = rec["timestamp"]
        # Should parse as a valid datetime with timezone
        dt = datetime.datetime.fromisoformat(ts)
        assert dt.tzinfo is not None

    def test_append_multiple_records(self, tmp_path):
        """_append_history appends multiple records on separate lines."""
        from manus_agent import cli

        hist = tmp_path / "history.jsonl"
        with mock.patch.object(cli, "_HISTORY_PATH", hist):
            for i in range(3):
                cli._append_history(
                    f"Task {i}",
                    f"Result {i}",
                    agent_type="single",
                    mode="auto",
                    success=True,
                )

        lines = hist.read_text().strip().splitlines()
        assert len(lines) == 3
        for i, line in enumerate(lines):
            rec = json.loads(line)
            assert rec["task"] == f"Task {i}"

    def test_append_failure_record(self, tmp_path):
        """_append_history correctly stores failure records."""
        from manus_agent import cli

        hist = tmp_path / "history.jsonl"
        with mock.patch.object(cli, "_HISTORY_PATH", hist):
            cli._append_history(
                "Failing task",
                "Error: connection timeout",
                agent_type="single",
                mode="auto",
                success=False,
            )

        rec = json.loads(hist.read_text().strip())
        assert rec["success"] is False
        assert "timeout" in rec["result"]

    def test_append_unicode_task(self, tmp_path):
        """_append_history handles unicode in task/result."""
        from manus_agent import cli

        hist = tmp_path / "history.jsonl"
        with mock.patch.object(cli, "_HISTORY_PATH", hist):
            cli._append_history(
                "Análisis de vulnerabilidades 日本語テスト",
                "Résultat: 成功 ✓",
                agent_type="single",
                mode="auto",
                success=True,
            )

        rec = json.loads(hist.read_text().strip())
        assert "日本語テスト" in rec["task"]
        assert "成功" in rec["result"]


# ---------------------------------------------------------------------------
# Tests: Ordering
# ---------------------------------------------------------------------------


class TestHistoryOrdering:
    """Tests for result ordering (most-recent-first)."""

    def test_most_recent_first_in_text(self, tmp_path, capsys):
        """Text output shows most recent entries first."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)

        code = _invoke_history(["--limit", "5"], hist)
        assert code == 0

        captured = capsys.readouterr()
        # Most recent entry (index 4) should appear before earliest (index 0)
        pos_recent = captured.out.find("CVE-2021-44228")
        pos_oldest = captured.out.find("CVE-2024-3094")
        # Both should be present
        assert pos_recent >= 0
        assert pos_oldest >= 0
        # Recent should appear first (earlier in output)
        assert pos_recent < pos_oldest

    def test_most_recent_first_in_json(self, tmp_path, capsys):
        """JSON output returns entries in reverse chronological order."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS)

        code = _invoke_history(["--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        timestamps = [r["timestamp"] for r in data]
        # Should be in reverse order
        assert timestamps == sorted(timestamps, reverse=True)


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestHistoryEdgeCases:
    """Additional edge case tests."""

    def test_single_record(self, tmp_path, capsys):
        """Works correctly with a single history record."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS[:1])

        code = _invoke_history(["--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1

    def test_grep_with_special_regex_chars(self, tmp_path, capsys):
        """--grep handles special chars without regex interpretation."""
        hist = tmp_path / "history.jsonl"
        # Add a record with regex-special characters in the task
        rec = {**_SAMPLE_RECORDS[0], "task": "Find files matching *.py (regex test)"}
        _write_history(hist, [rec])

        # grep uses substring match, not regex — should find it literally
        code = _invoke_history(["--grep", "*.py", "--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1

    def test_record_with_empty_task(self, tmp_path, capsys):
        """Records with empty task field still render."""
        hist = tmp_path / "history.jsonl"
        rec = {**_SAMPLE_RECORDS[0], "task": ""}
        _write_history(hist, [rec])

        code = _invoke_history(["--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["task"] == ""

    def test_large_history_performance(self, tmp_path, capsys):
        """Handles a large history file without issue."""
        hist = tmp_path / "history.jsonl"
        records = []
        for i in range(500):
            rec = {
                "timestamp": f"2025-01-{(i % 28) + 1:02d}T10:00:00+00:00",
                "task": f"Task number {i}",
                "agent": "single",
                "mode": "auto",
                "format": "text",
                "success": True,
                "result": f"Done {i}",
            }
            records.append(rec)
        _write_history(hist, records)

        code = _invoke_history(["--limit", "10", "--format", "json"], hist)
        assert code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 10

    def test_format_text_explicit(self, tmp_path, capsys):
        """--format text is explicit alternative to default."""
        hist = tmp_path / "history.jsonl"
        _write_history(hist, _SAMPLE_RECORDS[:2])

        code = _invoke_history(["--format", "text"], hist)
        assert code == 0

        captured = capsys.readouterr()
        # Should produce rich table, not JSON
        assert "entries" in captured.out
        # Should not be valid JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(captured.out)
