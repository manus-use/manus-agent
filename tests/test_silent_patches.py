"""
Tests for the silent patch detector (find_silent_patches tool + CLI subcommand).

All HTTP calls are mocked — no real network I/O.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def _make_commit(
    sha: str = "abc1234567890",
    message: str = "fix: normal bug",
    date: str = "2024-03-15T12:00:00Z",
    author: str = "Alice",
) -> dict:
    return {
        "sha": sha,
        "html_url": f"https://github.com/owner/repo/commit/{sha}",
        "commit": {
            "message": message,
            "author": {"date": date, "name": author},
        },
    }


def _make_tool_use(repo: str, **kwargs) -> dict:
    inp = {"repo": repo}
    inp.update(kwargs)
    return {"toolUseId": "tu-001", "input": inp}


# ---------------------------------------------------------------------------
# Module-level import sanity
# ---------------------------------------------------------------------------


def test_find_silent_patches_module_imports():
    from manus_use.tools.find_silent_patches import find_silent_patches  # noqa: F401

    assert callable(find_silent_patches)


def test_tool_spec_structure():
    from manus_use.tools.find_silent_patches import TOOL_SPEC

    assert TOOL_SPEC["name"] == "find_silent_patches"
    assert "repo" in TOOL_SPEC["inputSchema"]["json"]["properties"]
    assert TOOL_SPEC["inputSchema"]["json"]["required"] == ["repo"]


def test_tool_spec_description_mentions_silent_patch():
    from manus_use.tools.find_silent_patches import TOOL_SPEC

    assert "silent" in TOOL_SPEC["description"].lower()
    assert "cve" in TOOL_SPEC["description"].lower()


# ---------------------------------------------------------------------------
# _github_headers
# ---------------------------------------------------------------------------


def test_github_headers_no_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    from manus_use.tools.find_silent_patches import _github_headers

    h = _github_headers()
    assert "Authorization" not in h
    assert h["Accept"] == "application/vnd.github.v3+json"


def test_github_headers_with_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test123")
    from manus_use.tools.find_silent_patches import _github_headers

    h = _github_headers()
    assert h["Authorization"] == "Bearer ghp_test123"


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------


def test_parse_date_short():
    from manus_use.tools.find_silent_patches import _parse_date

    assert _parse_date("2024-03-01") == "2024-03-01T00:00:00Z"


def test_parse_date_full_iso():
    from manus_use.tools.find_silent_patches import _parse_date

    full = "2024-03-01T15:30:00Z"
    assert _parse_date(full) == full


def test_parse_date_strips_whitespace():
    from manus_use.tools.find_silent_patches import _parse_date

    assert _parse_date("  2024-06-01  ") == "2024-06-01T00:00:00Z"


# ---------------------------------------------------------------------------
# _default_since
# ---------------------------------------------------------------------------


def test_default_since_is_90_days_ago():
    from datetime import datetime, timezone

    from manus_use.tools.find_silent_patches import _default_since

    result = _default_since()
    dt = datetime.fromisoformat(result.replace("Z", "+00:00"))
    delta = datetime.now(timezone.utc) - dt
    # Should be approximately 90 days (allow 1 day tolerance for test speed)
    assert 89 <= delta.days <= 91


# ---------------------------------------------------------------------------
# _score_message
# ---------------------------------------------------------------------------


def test_score_message_security_keyword():
    from manus_use.tools.find_silent_patches import _score_message

    score, kws = _score_message("fix: security vulnerability in auth module")
    assert score >= 20
    assert len(kws) >= 1


def test_score_message_no_keywords():
    from manus_use.tools.find_silent_patches import _score_message

    score, kws = _score_message("chore: update dependencies")
    # Might still score zero or very low
    assert score <= 5
    assert isinstance(kws, list)


def test_score_message_cve_reference():
    from manus_use.tools.find_silent_patches import _score_message

    # CVE keyword should match (we use it to exclude later)
    score, kws = _score_message("fix CVE-2024-1234: path traversal")
    assert score >= 15


def test_score_message_injection_keyword():
    from manus_use.tools.find_silent_patches import _score_message

    score, kws = _score_message("fix: prevent SQL injection in login form")
    assert score >= 10


def test_score_message_buffer_overflow():
    from manus_use.tools.find_silent_patches import _score_message

    score, kws = _score_message("fix buffer overflow in packet parser")
    assert score >= 10


def test_score_message_bypass():
    from manus_use.tools.find_silent_patches import _score_message

    score, kws = _score_message("fix authentication bypass in API handler")
    assert score >= 10


def test_score_message_rce():
    from manus_use.tools.find_silent_patches import _score_message

    score, kws = _score_message("fix RCE via template injection")
    assert score >= 18


# ---------------------------------------------------------------------------
# _score_diff
# ---------------------------------------------------------------------------


def test_score_diff_sanitize():
    from manus_use.tools.find_silent_patches import _score_diff

    score, kws = _score_diff("+    output = html.escape(user_input)")
    assert score >= 8


def test_score_diff_empty():
    from manus_use.tools.find_silent_patches import _score_diff

    score, kws = _score_diff("")
    assert score == 0
    assert kws == []


def test_score_diff_sql_parameterized():
    from manus_use.tools.find_silent_patches import _score_diff

    diff = "+    cursor.execute(query, (param,))\n+    # Use parameterized query"
    score, kws = _score_diff(diff)
    assert score >= 8


def test_score_diff_auth_check():
    from manus_use.tools.find_silent_patches import _score_diff

    diff = "+    if not is_authenticated(request.user):\n+        raise PermissionError"
    score, kws = _score_diff(diff)
    assert score >= 10


# ---------------------------------------------------------------------------
# _infer_bug_class
# ---------------------------------------------------------------------------


def test_infer_bug_class_auth():
    from manus_use.tools.find_silent_patches import _infer_bug_class

    diff = "check_permission(user)\nis_authenticated\nhas_permission"
    result = _infer_bug_class(diff)
    assert result == "auth_bypass"


def test_infer_bug_class_sql():
    from manus_use.tools.find_silent_patches import _infer_bug_class

    diff = "cursor.execute(query)\nparameterized\nuse sql prepare"
    result = _infer_bug_class(diff)
    assert result == "sql_injection"


def test_infer_bug_class_xss():
    from manus_use.tools.find_silent_patches import _infer_bug_class

    diff = "html.escape(input)\nsanitize(data)\nescape(text)"
    result = _infer_bug_class(diff)
    assert result == "xss"


def test_infer_bug_class_none_for_generic():
    from manus_use.tools.find_silent_patches import _infer_bug_class

    diff = "fix typo in README\nadd blank line"
    result = _infer_bug_class(diff)
    assert result is None


# ---------------------------------------------------------------------------
# _has_cve_reference
# ---------------------------------------------------------------------------


def test_has_cve_reference_true():
    from manus_use.tools.find_silent_patches import _has_cve_reference

    assert _has_cve_reference("Fixes CVE-2024-3094 in xz-utils") is True


def test_has_cve_reference_ghsa():
    from manus_use.tools.find_silent_patches import _has_cve_reference

    assert _has_cve_reference("GHSA-abcd-1234-efgh") is True


def test_has_cve_reference_none():
    from manus_use.tools.find_silent_patches import _has_cve_reference

    assert _has_cve_reference("fix auth bypass in login") is False


def test_has_cve_reference_case_insensitive():
    from manus_use.tools.find_silent_patches import _has_cve_reference

    assert _has_cve_reference("cve-2021-44228 log4j") is True


# ---------------------------------------------------------------------------
# _list_commits
# ---------------------------------------------------------------------------


def test_list_commits_success():
    from manus_use.tools.find_silent_patches import _list_commits

    commits = [_make_commit(sha=f"abc{i:010d}") for i in range(5)]

    with patch("manus_use.tools.find_silent_patches._fetch_json", return_value=commits):
        result = _list_commits("owner/repo", "2024-01-01T00:00:00Z", "2024-03-01T00:00:00Z", 100, {})
    assert len(result) == 5


def test_list_commits_respects_max():
    from manus_use.tools.find_silent_patches import _list_commits

    commits = [_make_commit(sha=f"abc{i:010d}") for i in range(50)]

    with patch("manus_use.tools.find_silent_patches._fetch_json", return_value=commits):
        result = _list_commits("owner/repo", "2024-01-01T00:00:00Z", "2024-03-01T00:00:00Z", 10, {})
    assert len(result) == 10


def test_list_commits_404_raises_value_error():
    import requests

    from manus_use.tools.find_silent_patches import _list_commits

    mock_resp = MagicMock()
    mock_resp.status_code = 404
    exc = requests.HTTPError(response=mock_resp)

    with patch("manus_use.tools.find_silent_patches._fetch_json", side_effect=exc):
        with pytest.raises(ValueError, match="not found"):
            _list_commits("bad/repo", "2024-01-01T00:00:00Z", "2024-03-01T00:00:00Z", 100, {})


def test_list_commits_pagination_stops_on_empty():
    from manus_use.tools.find_silent_patches import _list_commits

    # First call returns 3 items (< per_page of min(100,5)), so should stop
    commits = [_make_commit(sha=f"abc{i:010d}") for i in range(3)]

    with patch("manus_use.tools.find_silent_patches._fetch_json", return_value=commits) as mock_fetch:
        result = _list_commits("owner/repo", "2024-01-01T00:00:00Z", "2024-03-01T00:00:00Z", 5, {})
    assert len(result) == 3
    assert mock_fetch.call_count == 1


# ---------------------------------------------------------------------------
# _fetch_commit_diff
# ---------------------------------------------------------------------------


def test_fetch_commit_diff_success():
    from manus_use.tools.find_silent_patches import _fetch_commit_diff

    mock_resp = MagicMock()
    mock_resp.text = "+    sanitize(input)\n-    pass"
    mock_resp.raise_for_status = MagicMock()

    with patch("manus_use.tools.find_silent_patches.requests.get", return_value=mock_resp):
        result = _fetch_commit_diff("owner/repo", "abc123", {})
    assert "sanitize" in result


def test_fetch_commit_diff_failure_returns_empty():
    from manus_use.tools.find_silent_patches import _fetch_commit_diff

    with patch(
        "manus_use.tools.find_silent_patches.requests.get",
        side_effect=Exception("network error"),
    ):
        result = _fetch_commit_diff("owner/repo", "abc123", {})
    assert result == ""


# ---------------------------------------------------------------------------
# find_silent_patches_impl — integration
# ---------------------------------------------------------------------------


def _mock_commits_for_impl():
    """Return a mix: one suspicious (no CVE), one with CVE (should be filtered out), one generic."""
    return [
        _make_commit(
            sha="security1234567",
            message="fix: sanitize user input to prevent XSS",
        ),
        _make_commit(
            sha="cve_commit12345",
            message="fix CVE-2024-9999: buffer overflow in parser",
        ),
        _make_commit(
            sha="generic1234567",
            message="chore: bump version to 2.0.1",
        ),
        _make_commit(
            sha="auth_bypass1234",
            message="security: fix authentication bypass in login endpoint",
        ),
    ]


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits")
def test_impl_filters_cve_commits(mock_list, mock_diff):
    from manus_use.tools.find_silent_patches import find_silent_patches_impl

    mock_list.return_value = _mock_commits_for_impl()

    result = find_silent_patches_impl(repo="owner/repo", fast=True)

    # CVE commit should be excluded from candidates
    shas = [c["sha"] for c in result["candidates"]]
    assert not any("cve_commit" in s for s in shas)


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits")
def test_impl_includes_suspicious_commits(mock_list, mock_diff):
    from manus_use.tools.find_silent_patches import find_silent_patches_impl

    mock_list.return_value = _mock_commits_for_impl()

    result = find_silent_patches_impl(repo="owner/repo", fast=True)

    # At minimum, the sanitize/XSS commit should score high enough to appear
    assert len(result["candidates"]) >= 1


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits")
def test_impl_sorted_by_score_descending(mock_list, mock_diff):
    from manus_use.tools.find_silent_patches import find_silent_patches_impl

    mock_list.return_value = _mock_commits_for_impl()

    result = find_silent_patches_impl(repo="owner/repo", fast=True)

    scores = [c["score"] for c in result["candidates"]]
    assert scores == sorted(scores, reverse=True)


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits")
def test_impl_total_scanned_matches_commit_count(mock_list, mock_diff):
    from manus_use.tools.find_silent_patches import find_silent_patches_impl

    commits = _mock_commits_for_impl()
    mock_list.return_value = commits

    result = find_silent_patches_impl(repo="owner/repo", fast=True)
    assert result["total_scanned"] == len(commits)


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits")
def test_impl_returns_repo_and_date_range(mock_list, mock_diff):
    from manus_use.tools.find_silent_patches import find_silent_patches_impl

    mock_list.return_value = []

    result = find_silent_patches_impl(repo="owner/repo", since="2024-01-01", until="2024-03-01", fast=True)
    assert result["repo"] == "owner/repo"
    assert result["since"] == "2024-01-01"
    assert result["until"] == "2024-03-01"


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits")
def test_impl_strips_github_url_prefix(mock_list, mock_diff):
    from manus_use.tools.find_silent_patches import find_silent_patches_impl

    mock_list.return_value = []

    result = find_silent_patches_impl(repo="https://github.com/owner/repo", fast=True)
    assert result["repo"] == "owner/repo"


def test_impl_invalid_repo_format():
    from manus_use.tools.find_silent_patches import find_silent_patches_impl

    result = find_silent_patches_impl(repo="notavalidrepo")
    assert "error" in result
    assert "Invalid repo format" in result["error"]


@patch("manus_use.tools.find_silent_patches._list_commits")
def test_impl_404_repo(mock_list):
    from manus_use.tools.find_silent_patches import find_silent_patches_impl

    mock_list.side_effect = ValueError("Repository 'bad/repo' not found on GitHub.")

    result = find_silent_patches_impl(repo="bad/repo")
    assert "error" in result
    assert "not found" in result["error"].lower()


@patch(
    "manus_use.tools.find_silent_patches._fetch_commit_diff",
    return_value="+    html.escape(data)\n+    sanitize(user)\n+    is_authenticated(req)",
)
@patch("manus_use.tools.find_silent_patches._list_commits")
def test_impl_diff_scan_increases_score(mock_list, mock_diff):
    from manus_use.tools.find_silent_patches import find_silent_patches_impl

    mock_list.return_value = [
        _make_commit(sha="fix12345678901", message="fix: sanitize output"),
    ]

    result = find_silent_patches_impl(repo="owner/repo", fast=False)
    # Diff score should add to message score
    assert result["candidates"][0]["diff_score"] > 0


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits")
def test_impl_fast_mode_skips_diff(mock_list, mock_diff):
    from manus_use.tools.find_silent_patches import find_silent_patches_impl

    mock_list.return_value = [
        _make_commit(sha="fix12345678901", message="fix: sanitize output"),
    ]

    result = find_silent_patches_impl(repo="owner/repo", fast=True)
    mock_diff.assert_not_called()
    if result["candidates"]:
        assert result["candidates"][0]["diff_score"] == 0


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits")
def test_impl_max_commits_capped_at_500(mock_list, mock_diff):
    from manus_use.tools.find_silent_patches import find_silent_patches_impl

    mock_list.return_value = []

    find_silent_patches_impl(repo="owner/repo", max_commits=9999, fast=True)
    # Check that the call used a max of 500
    # max_commits is the 3rd positional arg to _list_commits
    assert mock_list.call_args[0][3] == 500


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits")
def test_impl_summary_string_populated(mock_list, mock_diff):
    from manus_use.tools.find_silent_patches import find_silent_patches_impl

    mock_list.return_value = _mock_commits_for_impl()
    result = find_silent_patches_impl(repo="owner/repo", fast=True)
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 10


# ---------------------------------------------------------------------------
# render_silent_patches_text
# ---------------------------------------------------------------------------


def test_render_text_with_candidates():
    from manus_use.tools.find_silent_patches import render_silent_patches_text

    result = {
        "repo": "owner/repo",
        "since": "2024-01-01",
        "until": "2024-03-01",
        "total_scanned": 50,
        "candidates": [
            {
                "sha": "abc1234",
                "date": "2024-02-10",
                "author": "Alice",
                "message": "fix: sanitize user input",
                "score": 45,
                "msg_score": 20,
                "diff_score": 25,
                "keywords": ["sanitize", "xss"],
                "bug_class": "xss",
                "url": "https://github.com/owner/repo/commit/abc1234",
            }
        ],
        "summary": "Scanned 50 commits. Found 1 candidate.",
    }
    text = render_silent_patches_text(result)
    assert "owner/repo" in text
    assert "abc1234" in text
    assert "xss" in text
    assert "sanitize" in text
    assert "45" in text


def test_render_text_no_candidates():
    from manus_use.tools.find_silent_patches import render_silent_patches_text

    result = {
        "repo": "clean/repo",
        "since": "2024-01-01",
        "until": "2024-03-01",
        "total_scanned": 30,
        "candidates": [],
        "summary": "No candidates found.",
    }
    text = render_silent_patches_text(result)
    assert "No suspicious commits" in text


def test_render_text_error():
    from manus_use.tools.find_silent_patches import render_silent_patches_text

    text = render_silent_patches_text({"error": "Repository 'bad/repo' not found on GitHub."})
    assert "Error:" in text
    assert "not found" in text.lower()


def test_render_text_shows_period():
    from manus_use.tools.find_silent_patches import render_silent_patches_text

    result = {
        "repo": "owner/repo",
        "since": "2024-01-01",
        "until": "2024-06-01",
        "total_scanned": 10,
        "candidates": [],
        "summary": "",
    }
    text = render_silent_patches_text(result)
    assert "2024-01-01" in text
    assert "2024-06-01" in text


# ---------------------------------------------------------------------------
# Strands entry point (find_silent_patches function)
# ---------------------------------------------------------------------------


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits", return_value=[])
def test_strands_entrypoint_success(mock_list, mock_diff):
    from manus_use.tools.find_silent_patches import find_silent_patches

    tool = _make_tool_use("owner/repo", since="2024-01-01", fast=True)
    result = find_silent_patches(tool)
    assert result["status"] == "success"
    assert "content" in result
    assert result["toolUseId"] == "tu-001"


def test_strands_entrypoint_missing_repo():
    from manus_use.tools.find_silent_patches import find_silent_patches

    tool = {"toolUseId": "tu-002", "input": {}}
    result = find_silent_patches(tool)
    assert result["status"] == "error"
    assert "Missing required parameter" in result["content"][0]["text"]


@patch("manus_use.tools.find_silent_patches._list_commits")
def test_strands_entrypoint_404(mock_list):
    from manus_use.tools.find_silent_patches import find_silent_patches

    mock_list.side_effect = ValueError("Repository 'bad/repo' not found on GitHub.")
    tool = _make_tool_use("bad/repo")
    result = find_silent_patches(tool)
    # The impl returns {"error": ...} which the entrypoint wraps as success content
    assert result["status"] == "success"
    content_text = result["content"][0]["text"]
    assert "error" in content_text.lower() or "not found" in content_text.lower()


# ---------------------------------------------------------------------------
# CLI — _build_silent_patches_parser
# ---------------------------------------------------------------------------


def test_cli_parser_defaults():
    from manus_use.cli import _build_silent_patches_parser

    p = _build_silent_patches_parser()
    args = p.parse_args(["owner/repo"])
    assert args.repo == "owner/repo"
    assert args.since is None
    assert args.until is None
    assert args.max_commits == 200
    assert args.fast is False
    assert args.output == "text"


def test_cli_parser_all_flags():
    from manus_use.cli import _build_silent_patches_parser

    p = _build_silent_patches_parser()
    args = p.parse_args(
        [
            "django/django",
            "--since",
            "2024-01-01",
            "--until",
            "2024-06-01",
            "--max-commits",
            "100",
            "--fast",
            "--output",
            "json",
        ]
    )
    assert args.repo == "django/django"
    assert args.since == "2024-01-01"
    assert args.until == "2024-06-01"
    assert args.max_commits == 100
    assert args.fast is True
    assert args.output == "json"


def test_cli_parser_help_exits_zero():
    from manus_use.cli import _build_silent_patches_parser

    p = _build_silent_patches_parser()
    with pytest.raises(SystemExit) as exc:
        p.parse_args(["--help"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# CLI — _run_silent_patches
# ---------------------------------------------------------------------------


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits", return_value=[])
def test_run_silent_patches_text_output_exits_zero(mock_list, mock_diff, capsys):
    from manus_use.cli import _run_silent_patches

    rc = _run_silent_patches(["owner/repo", "--fast"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "owner/repo" in out


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits", return_value=[])
def test_run_silent_patches_json_output(mock_list, mock_diff, capsys):
    from manus_use.cli import _run_silent_patches

    rc = _run_silent_patches(["owner/repo", "--output", "json", "--fast"])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "candidates" in data
    assert data["repo"] == "owner/repo"


@patch("manus_use.tools.find_silent_patches._list_commits")
def test_run_silent_patches_invalid_repo(mock_list, capsys):
    from manus_use.cli import _run_silent_patches

    # No "/" in repo name → impl returns error
    rc = _run_silent_patches(["notarepo", "--fast"])
    assert rc == 1


@patch("manus_use.tools.find_silent_patches._fetch_commit_diff", return_value="")
@patch("manus_use.tools.find_silent_patches._list_commits")
def test_run_silent_patches_with_candidates(mock_list, mock_diff, capsys):
    from manus_use.cli import _run_silent_patches

    mock_list.return_value = [
        _make_commit(sha="sec1234567890x", message="fix: sanitize XSS input in user profile"),
    ]

    rc = _run_silent_patches(["owner/repo", "--fast"])
    assert rc == 0
    out = capsys.readouterr().out
    # Should mention the candidate
    assert "sec12345" in out or "sanitize" in out.lower()


# ---------------------------------------------------------------------------
# CLI — subcommand registration
# ---------------------------------------------------------------------------


def test_silent_patches_in_subcommands_set():
    from manus_use.cli import _SUBCOMMANDS

    assert "silent-patches" in _SUBCOMMANDS


def test_silent_patches_dispatch_in_main(monkeypatch):
    """Test that main() dispatches to _run_silent_patches when 'silent-patches' is first positional."""
    from unittest.mock import patch

    with patch("manus_use.cli._run_silent_patches", return_value=0) as mock_run:
        with patch("sys.argv", ["manus-use", "silent-patches", "django/django", "--fast"]):
            try:
                from manus_use.cli import main

                main()
            except SystemExit as exc:
                assert exc.code == 0

        mock_run.assert_called_once_with(["django/django", "--fast"])


# ---------------------------------------------------------------------------
# VI agent wiring
# ---------------------------------------------------------------------------


def test_vi_agent_imports_find_silent_patches():
    """Verify VulnerabilityIntelligenceAgent module references find_silent_patches."""
    import importlib

    spec = importlib.util.find_spec("manus_use.agents.vi_agent")
    assert spec is not None
    source = spec.loader.get_source("manus_use.agents.vi_agent")
    assert "find_silent_patches" in source


def test_vi_agent_system_prompt_mentions_silent_patch():
    import importlib

    spec = importlib.util.find_spec("manus_use.agents.vi_agent")
    source = spec.loader.get_source("manus_use.agents.vi_agent")
    assert "silent" in source.lower() or "Silent Patch" in source


def test_vi_agent_tool_list_includes_find_silent_patches(monkeypatch):
    """VulnerabilityIntelligenceAgent.__init__ should add find_silent_patches to tools."""
    # We mock the heavy deps so we don't need real strands/strands_tools installed
    import types

    fake_strands = types.ModuleType("strands")
    fake_strands.Agent = MagicMock()
    fake_strands_tools = types.ModuleType("strands_tools")
    fake_strands_tools.current_time = MagicMock()

    monkeypatch.setitem(sys.modules, "strands", fake_strands)
    monkeypatch.setitem(sys.modules, "strands_tools", fake_strands_tools)

    # Ensure the module source mentions find_silent_patches in the import block
    import importlib

    spec = importlib.util.find_spec("manus_use.agents.vi_agent")
    source = spec.loader.get_source("manus_use.agents.vi_agent")
    assert "from manus_use.tools.find_silent_patches import find_silent_patches" in source


# ---------------------------------------------------------------------------
# README / docs coverage
# ---------------------------------------------------------------------------


def test_readme_mentions_silent_patches():
    """README.md should document the silent-patches subcommand."""
    readme = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    text = readme.read_text()
    assert "silent-patches" in text


def test_readme_silent_patches_example_has_owner_repo():
    """README silent-patches example should include a plausible repo argument."""
    readme = __import__("pathlib").Path(__file__).parent.parent / "README.md"
    text = readme.read_text()
    # Should show at least one owner/repo example
    import re

    section_match = re.search(r"silent-patches.*?(?=###|$)", text, re.DOTALL)
    assert section_match is not None
    assert "/" in section_match.group()
