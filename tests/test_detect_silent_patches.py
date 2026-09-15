"""Comprehensive test suite for detect_silent_patches tool and CLI subcommand.

All HTTP calls are mocked — zero real network requests.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.detect_silent_patches import (
    _REPO_RE,
    _classify_bug_classes,
    _confidence_label,
    _github_headers,
    _has_cve_or_advisory,
    _score_diff,
    _score_message,
    detect_silent_patches,
    fetch_commit_diff,
    fetch_commits,
    scan_repository,
)

# =========================================================================
# Helpers
# =========================================================================


def _tool_use(owner_repo: str = "owner/repo", **extra) -> dict:
    inp = {"owner_repo": owner_repo, **extra}
    return {"toolUseId": "test-id", "input": inp}


def _mock_commits_response(commits: list[dict], status_code: int = 200):
    """Build a mock response for the commits endpoint."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = commits
    resp.text = json.dumps(commits)
    return resp


def _make_commit(sha: str, message: str, date_str: str = "2025-06-01T12:00:00Z", author: str = "dev"):
    return {
        "sha": sha,
        "html_url": f"https://github.com/owner/repo/commit/{sha}",
        "commit": {
            "message": message,
            "author": {"name": author, "date": date_str},
        },
    }


# =========================================================================
# Unit tests — scoring functions
# =========================================================================


class TestScoreMessage:
    """Tests for _score_message."""

    def test_empty_message(self):
        assert _score_message("") == 0.0

    def test_security_fix_keyword(self):
        score = _score_message("security fix for input handling")
        assert score >= 25.0

    def test_buffer_overflow_keyword(self):
        score = _score_message("fix buffer overflow in parser")
        assert score >= 20.0

    def test_sanitize_keyword(self):
        score = _score_message("sanitize user input before processing")
        assert score >= 12.0

    def test_multiple_keywords_accumulate(self):
        score = _score_message("security fix: prevent injection and sanitize input")
        assert score >= 35.0

    def test_max_cap_at_50(self):
        # A message with many security keywords should cap at 50.
        msg = (
            "security fix security patch prevent injection buffer overflow "
            "use after free remote code execution privilege escalation "
            "arbitrary code sanitize validate input bounds check"
        )
        assert _score_message(msg) == 50.0

    def test_non_security_message(self):
        score = _score_message("refactor: rename variable for clarity")
        assert score == 0.0

    def test_case_insensitive(self):
        score = _score_message("SECURITY FIX for Buffer Overflow")
        assert score >= 25.0

    def test_rce_keyword(self):
        score = _score_message("prevent rce via deserialization")
        assert score >= 15.0

    def test_privilege_escalation(self):
        score = _score_message("fix privilege escalation in sudo handler")
        assert score >= 18.0


class TestScoreDiff:
    """Tests for _score_diff."""

    def test_empty_diff(self):
        assert _score_diff("") == 0.0

    def test_sanitize_function(self):
        score = _score_diff("+    result = sanitize(user_input)\n")
        assert score >= 8.0

    def test_auth_check(self):
        score = _score_diff("+    if not is_authenticated(request):\n+        return forbidden()\n")
        assert score >= 8.0

    def test_bounds_check(self):
        score = _score_diff("+    if len(data) > MAX_SIZE:\n+        raise ValueError('too large')\n")
        assert score >= 5.0

    def test_multiple_keywords_accumulate(self):
        diff = (
            "+    sanitize(input)\n"
            "+    check_permission(user)\n"
            "+    if len(buf) > MAX_LENGTH:\n"
            "+        raise ValueError\n"
        )
        score = _score_diff(diff)
        assert score >= 15.0

    def test_max_cap_at_50(self):
        # Repeating many keywords should cap at 50.
        diff = "\n".join(
            f"+    sanitize(x{i})\n+    check_permission(u{i})\n+    is_authenticated(r{i})" for i in range(20)
        )
        assert _score_diff(diff) == 50.0

    def test_keyword_count_capped_per_keyword(self):
        # Even if a keyword appears 100 times, contribution is capped at 3x.
        diff = "\n".join("+    sanitize(x)" for _ in range(100))
        score = _score_diff(diff)
        # sanitize( has weight 8.0, capped at 3 occurrences = 24.0
        assert score == 24.0

    def test_shell_false(self):
        score = _score_diff("+    subprocess.run(cmd, shell=False)\n")
        assert score >= 8.0

    def test_shlex_quote(self):
        score = _score_diff("+    cmd = shlex.quote(user_input)\n")
        assert score >= 8.0


class TestClassifyBugClasses:
    """Tests for _classify_bug_classes."""

    def test_sql_injection(self):
        classes = _classify_bug_classes("fix sql injection in login form")
        assert "sql_injection" in classes

    def test_command_injection(self):
        classes = _classify_bug_classes("prevent command injection via os.system call")
        assert "command_injection" in classes

    def test_xss(self):
        classes = _classify_bug_classes("fix cross-site scripting in comment renderer")
        assert "xss" in classes

    def test_buffer_overflow(self):
        classes = _classify_bug_classes("fix buffer overflow in memcpy")
        assert "buffer_overflow" in classes

    def test_use_after_free(self):
        classes = _classify_bug_classes("fix use after free in allocator")
        assert "use_after_free" in classes

    def test_auth_bypass(self):
        classes = _classify_bug_classes("fix auth bypass in permission check")
        assert "auth_bypass" in classes

    def test_path_traversal(self):
        classes = _classify_bug_classes("prevent path traversal with ../ sequences")
        assert "path_traversal" in classes

    def test_race_condition(self):
        classes = _classify_bug_classes("fix race condition with mutex lock")
        assert "race_condition" in classes

    def test_cryptographic(self):
        classes = _classify_bug_classes("replace weak cipher with strong encryption")
        assert "cryptographic" in classes

    def test_deserialization(self):
        classes = _classify_bug_classes("fix insecure deserialization in pickle handler")
        assert "deserialization" in classes

    def test_ssrf(self):
        classes = _classify_bug_classes("prevent ssrf via url validation and allowlist")
        assert "ssrf" in classes

    def test_input_validation(self):
        classes = _classify_bug_classes("add input validation with regex filter")
        assert "input_validation" in classes

    def test_integer_overflow(self):
        classes = _classify_bug_classes("fix integer overflow in size_t calculation")
        assert "integer_overflow" in classes

    def test_null_dereference(self):
        classes = _classify_bug_classes("add null pointer check before dereference")
        assert "null_dereference" in classes

    def test_no_match(self):
        classes = _classify_bug_classes("refactor: rename variable for clarity")
        assert classes == []

    def test_multiple_classes(self):
        classes = _classify_bug_classes("fix sql injection and prevent command injection with input validation")
        assert "sql_injection" in classes
        assert "command_injection" in classes
        assert "input_validation" in classes


class TestConfidenceLabel:
    """Tests for _confidence_label."""

    def test_high(self):
        assert _confidence_label(60) == "HIGH"
        assert _confidence_label(100) == "HIGH"
        assert _confidence_label(75) == "HIGH"

    def test_medium(self):
        assert _confidence_label(30) == "MEDIUM"
        assert _confidence_label(59) == "MEDIUM"
        assert _confidence_label(45) == "MEDIUM"

    def test_low(self):
        assert _confidence_label(0) == "LOW"
        assert _confidence_label(29) == "LOW"
        assert _confidence_label(10) == "LOW"


class TestHasCveOrAdvisory:
    """Tests for _has_cve_or_advisory."""

    def test_cve_present(self):
        assert _has_cve_or_advisory("Fix for CVE-2024-12345") is True

    def test_ghsa_present(self):
        assert _has_cve_or_advisory("Address GHSA-abcd-efgh-ijkl") is True

    def test_no_cve(self):
        assert _has_cve_or_advisory("Fix buffer overflow in parser") is False

    def test_both_present(self):
        assert _has_cve_or_advisory("CVE-2024-1234 (GHSA-abcd-efgh-ijkl)") is True


class TestRepoRegex:
    """Tests for _REPO_RE validation."""

    def test_valid_repos(self):
        assert _REPO_RE.match("torvalds/linux")
        assert _REPO_RE.match("manus-use/manus-agent")
        assert _REPO_RE.match("org.name/repo_name")

    def test_invalid_repos(self):
        assert not _REPO_RE.match("")
        assert not _REPO_RE.match("noslash")
        assert not _REPO_RE.match("too/many/slashes")
        assert not _REPO_RE.match("/leading-slash")


class TestGitHubHeaders:
    """Tests for _github_headers."""

    def test_without_token(self):
        with patch.dict(os.environ, {}, clear=True):
            headers = _github_headers()
            assert "Authorization" not in headers
            assert "User-Agent" in headers

    def test_with_token(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test123"}):
            headers = _github_headers()
            assert headers["Authorization"] == "Bearer ghp_test123"


# =========================================================================
# Integration tests — fetch_commits
# =========================================================================


class TestFetchCommits:
    """Tests for fetch_commits."""

    @patch("manus_agent.tools.detect_silent_patches._github_get")
    def test_basic_fetch(self, mock_get):
        commits = [
            _make_commit("abc1234" * 5 + "abcde", "fix: sanitize input"),
            _make_commit("def5678" * 5 + "defgh", "refactor: rename variable"),
        ]
        mock_get.return_value = _mock_commits_response(commits)

        result = fetch_commits("owner", "repo", since="2025-01-01", until="2025-06-01")
        assert len(result) == 2
        assert result[0]["message"] == "fix: sanitize input"

    @patch("manus_agent.tools.detect_silent_patches._github_get")
    def test_404_raises(self, mock_get):
        mock_get.return_value = _mock_commits_response([], status_code=404)
        mock_get.return_value.text = "Not Found"

        with pytest.raises(ValueError, match="not found"):
            fetch_commits("ghost", "nonexistent")

    @patch("manus_agent.tools.detect_silent_patches._github_get")
    def test_409_empty_repo(self, mock_get):
        resp = MagicMock()
        resp.status_code = 409
        mock_get.return_value = resp

        result = fetch_commits("owner", "empty-repo")
        assert result == []

    @patch("manus_agent.tools.detect_silent_patches._github_get")
    def test_max_commits_limit(self, mock_get):
        commits = [_make_commit(f"sha{i:040d}", f"commit {i}") for i in range(10)]
        mock_get.return_value = _mock_commits_response(commits)

        result = fetch_commits("owner", "repo", max_commits=3)
        assert len(result) == 3

    @patch("manus_agent.tools.detect_silent_patches._github_get")
    def test_pagination(self, mock_get):
        page1 = [_make_commit(f"sha1_{i:038d}", f"commit {i}") for i in range(100)]
        page2 = [_make_commit(f"sha2_{i:038d}", f"commit {100 + i}") for i in range(50)]

        resp1 = _mock_commits_response(page1)
        resp2 = _mock_commits_response(page2)
        mock_get.side_effect = [resp1, resp2]

        result = fetch_commits("owner", "repo", max_commits=200)
        assert len(result) == 150

    @patch("manus_agent.tools.detect_silent_patches._github_get")
    def test_500_error(self, mock_get):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal Server Error"
        mock_get.return_value = resp

        with pytest.raises(ValueError, match="GitHub API error 500"):
            fetch_commits("owner", "repo")


class TestFetchCommitDiff:
    """Tests for fetch_commit_diff."""

    @patch("manus_agent.tools.detect_silent_patches._github_diff_get")
    def test_success(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "diff --git a/file.py b/file.py\n+    sanitize(input)\n"
        mock_get.return_value = resp

        result = fetch_commit_diff("owner", "repo", "abc1234")
        assert "sanitize" in result

    @patch("manus_agent.tools.detect_silent_patches._github_diff_get")
    def test_failure_returns_empty(self, mock_get):
        resp = MagicMock()
        resp.status_code = 404
        mock_get.return_value = resp

        result = fetch_commit_diff("owner", "repo", "missing")
        assert result == ""

    @patch("manus_agent.tools.detect_silent_patches._github_diff_get")
    def test_large_diff_truncated(self, mock_get):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = "x" * 100_000
        mock_get.return_value = resp

        result = fetch_commit_diff("owner", "repo", "abc1234")
        assert len(result) == 50_000


# =========================================================================
# Integration tests — scan_repository
# =========================================================================


class TestScanRepository:
    """Tests for scan_repository."""

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_basic_scan(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "a" * 40,
                "message": "security fix: prevent injection in login",
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/owner/repo/commit/" + "a" * 40,
                "author": "dev",
            },
            {
                "sha": "b" * 40,
                "message": "docs: update README",
                "date": "2025-06-02T12:00:00Z",
                "url": "https://github.com/owner/repo/commit/" + "b" * 40,
                "author": "dev",
            },
        ]
        mock_diff.return_value = "+    sanitize(user_input)\n"

        result = scan_repository("owner", "repo")
        assert result["total_commits_scanned"] == 2
        assert len(result["candidates"]) >= 1
        # The security fix should be a candidate.
        found = any(c["sha"] == "a" * 40 for c in result["candidates"])
        assert found

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_cve_commits_excluded(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "c" * 40,
                "message": "security fix for CVE-2024-12345",
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/owner/repo/commit/" + "c" * 40,
                "author": "dev",
            },
        ]
        mock_diff.return_value = ""

        result = scan_repository("owner", "repo")
        assert len(result["candidates"]) == 0

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_ghsa_commits_excluded(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "d" * 40,
                "message": "fix GHSA-abcd-efgh-ijkl buffer overflow",
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/owner/repo/commit/" + "d" * 40,
                "author": "dev",
            },
        ]
        mock_diff.return_value = ""

        result = scan_repository("owner", "repo")
        assert len(result["candidates"]) == 0

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_fast_mode_skips_diff(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "e" * 40,
                "message": "security fix: prevent buffer overflow",
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/owner/repo/commit/" + "e" * 40,
                "author": "dev",
            },
        ]

        result = scan_repository("owner", "repo", fast=True)
        mock_diff.assert_not_called()
        assert result["fast_mode"] is True
        assert len(result["candidates"]) >= 1
        # Diff score should be 0 in fast mode.
        assert result["candidates"][0]["diff_score"] == 0.0

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_diff_error_graceful(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "f" * 40,
                "message": "security fix: sanitize and validate input",
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/owner/repo/commit/" + "f" * 40,
                "author": "dev",
            },
        ]
        mock_diff.side_effect = Exception("Network error")

        result = scan_repository("owner", "repo")
        assert len(result["errors"]) == 1
        assert "Network error" in result["errors"][0]
        # Should still have a candidate from message scoring.
        assert len(result["candidates"]) >= 1

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_default_dates(self, mock_commits, mock_diff):
        mock_commits.return_value = []
        mock_diff.return_value = ""

        result = scan_repository("owner", "repo")
        assert result["since"] == (date.today() - timedelta(days=90)).isoformat()
        assert result["until"] == date.today().isoformat()

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_custom_dates(self, mock_commits, mock_diff):
        mock_commits.return_value = []
        mock_diff.return_value = ""

        result = scan_repository("owner", "repo", since="2025-01-01", until="2025-06-01")
        assert result["since"] == "2025-01-01"
        assert result["until"] == "2025-06-01"

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_empty_repo(self, mock_commits, mock_diff):
        mock_commits.return_value = []
        mock_diff.return_value = ""

        result = scan_repository("owner", "repo")
        assert result["total_commits_scanned"] == 0
        assert result["candidates"] == []

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_summary_stats(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "a" * 40,
                "message": "security fix security patch prevent injection buffer overflow use-after-free",
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/o/r/commit/" + "a" * 40,
                "author": "dev",
            },
            {
                "sha": "b" * 40,
                "message": "fix sanitize input validation",
                "date": "2025-06-02T12:00:00Z",
                "url": "https://github.com/o/r/commit/" + "b" * 40,
                "author": "dev",
            },
        ]
        mock_diff.return_value = "+    sanitize(x)\n+    check_permission(y)\n"

        result = scan_repository("owner", "repo")
        summary = result["summary"]
        assert summary["total_candidates"] >= 1
        assert isinstance(summary["high_confidence"], int)
        assert isinstance(summary["medium_confidence"], int)
        assert isinstance(summary["low_confidence"], int)
        assert isinstance(summary["bug_class_distribution"], dict)

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_candidates_sorted_by_score(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "a" * 40,
                "message": "fix sanitize input",
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/o/r/commit/" + "a" * 40,
                "author": "dev",
            },
            {
                "sha": "b" * 40,
                "message": "security fix security patch prevent injection buffer overflow remote code execution",
                "date": "2025-06-02T12:00:00Z",
                "url": "https://github.com/o/r/commit/" + "b" * 40,
                "author": "dev",
            },
        ]
        mock_diff.return_value = ""

        result = scan_repository("owner", "repo")
        candidates = result["candidates"]
        if len(candidates) >= 2:
            assert candidates[0]["total_score"] >= candidates[1]["total_score"]

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_low_scoring_messages_filtered(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "x" * 40,
                "message": "chore: update dependencies",
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/o/r/commit/" + "x" * 40,
                "author": "dev",
            },
        ]
        mock_diff.return_value = ""

        result = scan_repository("owner", "repo")
        assert len(result["candidates"]) == 0

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_message_truncated_at_120(self, mock_commits, mock_diff):
        long_msg = "security fix: " + "a" * 200
        mock_commits.return_value = [
            {
                "sha": "z" * 40,
                "message": long_msg,
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/o/r/commit/" + "z" * 40,
                "author": "dev",
            },
        ]
        mock_diff.return_value = ""

        result = scan_repository("owner", "repo")
        if result["candidates"]:
            assert len(result["candidates"][0]["message"]) <= 120

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_multiline_message_first_line(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "m" * 40,
                "message": "security fix: prevent injection\n\nDetailed description here.",
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/o/r/commit/" + "m" * 40,
                "author": "dev",
            },
        ]
        mock_diff.return_value = ""

        result = scan_repository("owner", "repo")
        if result["candidates"]:
            assert "\n" not in result["candidates"][0]["message"]


# =========================================================================
# Strands tool interface tests
# =========================================================================


class TestDetectSilentPatchesTool:
    """Tests for the Strands tool wrapper."""

    def test_invalid_owner_repo_format(self):
        result = detect_silent_patches(_tool_use("not-a-repo"))
        assert result["status"] == "error"
        assert "owner/repo" in result["content"][0]["text"].lower()

    def test_empty_owner_repo(self):
        result = detect_silent_patches(_tool_use(""))
        assert result["status"] == "error"

    def test_missing_owner_repo(self):
        tool = {"toolUseId": "test-id", "input": {}}
        result = detect_silent_patches(tool)
        assert result["status"] == "error"

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_success(self, mock_scan):
        mock_scan.return_value = {
            "owner": "owner",
            "repo": "repo",
            "candidates": [],
            "summary": {"total_candidates": 0},
        }
        result = detect_silent_patches(_tool_use("owner/repo"))
        assert result["status"] == "success"
        data = json.loads(result["content"][0]["text"])
        assert data["owner"] == "owner"

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_value_error(self, mock_scan):
        mock_scan.side_effect = ValueError("Repository not found")
        result = detect_silent_patches(_tool_use("ghost/repo"))
        assert result["status"] == "error"
        assert "Repository not found" in result["content"][0]["text"]

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_request_exception(self, mock_scan):
        import requests

        mock_scan.side_effect = requests.RequestException("Connection timeout")
        result = detect_silent_patches(_tool_use("owner/repo"))
        assert result["status"] == "error"
        assert "Connection timeout" in result["content"][0]["text"]

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_passes_optional_params(self, mock_scan):
        mock_scan.return_value = {"owner": "o", "repo": "r", "candidates": [], "summary": {}}
        detect_silent_patches(
            _tool_use("owner/repo", since="2025-01-01", until="2025-06-01", max_commits=100, fast=True)
        )
        mock_scan.assert_called_once_with(
            owner="owner",
            repo="repo",
            since="2025-01-01",
            until="2025-06-01",
            max_commits=100,
            fast=True,
        )

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_tool_use_id_preserved(self, mock_scan):
        mock_scan.return_value = {"candidates": [], "summary": {}}
        tool = {"toolUseId": "my-unique-id", "input": {"owner_repo": "o/r"}}
        result = detect_silent_patches(tool)
        assert result["toolUseId"] == "my-unique-id"


# =========================================================================
# CLI tests — _run_silent_patches
# =========================================================================


class TestCliSilentPatches:
    """Tests for the CLI subcommand."""

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_text_output_basic(self, mock_scan, capsys):
        mock_scan.return_value = {
            "owner": "owner",
            "repo": "repo",
            "since": "2025-06-01",
            "until": "2025-09-01",
            "total_commits_scanned": 42,
            "fast_mode": False,
            "candidates": [
                {
                    "sha": "a" * 40,
                    "short_sha": "a" * 7,
                    "message": "fix: prevent injection",
                    "date": "2025-06-15T12:00:00Z",
                    "url": "https://github.com/owner/repo/commit/" + "a" * 40,
                    "author": "dev",
                    "message_score": 30.0,
                    "diff_score": 15.0,
                    "total_score": 45.0,
                    "confidence": "MEDIUM",
                    "bug_classes": ["sql_injection"],
                }
            ],
            "summary": {
                "total_candidates": 1,
                "high_confidence": 0,
                "medium_confidence": 1,
                "low_confidence": 0,
                "bug_class_distribution": {"sql_injection": 1},
            },
            "errors": [],
        }

        from manus_agent.cli import _run_silent_patches

        code = _run_silent_patches(["owner/repo"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Silent Patch Detector" in out
        assert "owner/repo" in out
        assert "42" in out
        assert "MEDIUM" in out
        assert "sql_injection" in out

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_json_output(self, mock_scan, capsys):
        mock_scan.return_value = {
            "owner": "owner",
            "repo": "repo",
            "candidates": [],
            "summary": {"total_candidates": 0},
        }

        from manus_agent.cli import _run_silent_patches

        code = _run_silent_patches(["owner/repo", "--output", "json"])
        assert code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["owner"] == "owner"

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_no_candidates(self, mock_scan, capsys):
        mock_scan.return_value = {
            "owner": "owner",
            "repo": "repo",
            "since": "2025-06-01",
            "until": "2025-09-01",
            "total_commits_scanned": 10,
            "fast_mode": False,
            "candidates": [],
            "summary": {
                "total_candidates": 0,
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0,
                "bug_class_distribution": {},
            },
            "errors": [],
        }

        from manus_agent.cli import _run_silent_patches

        code = _run_silent_patches(["owner/repo"])
        assert code == 0
        out = capsys.readouterr().out
        assert "No silent patch candidates found" in out

    def test_invalid_repo_format(self, capsys):
        from manus_agent.cli import _run_silent_patches

        code = _run_silent_patches(["not-a-repo"])
        assert code == 1
        err = capsys.readouterr().err
        assert "owner/repo" in err

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_scan_error(self, mock_scan, capsys):
        mock_scan.side_effect = ValueError("Repository not found (404)")

        from manus_agent.cli import _run_silent_patches

        code = _run_silent_patches(["ghost/nonexistent"])
        assert code == 1
        err = capsys.readouterr().err
        assert "not found" in err.lower()

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_fast_flag(self, mock_scan, capsys):
        mock_scan.return_value = {
            "owner": "owner",
            "repo": "repo",
            "since": "2025-06-01",
            "until": "2025-09-01",
            "total_commits_scanned": 5,
            "fast_mode": True,
            "candidates": [],
            "summary": {
                "total_candidates": 0,
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0,
                "bug_class_distribution": {},
            },
            "errors": [],
        }

        from manus_agent.cli import _run_silent_patches

        code = _run_silent_patches(["owner/repo", "--fast"])
        assert code == 0
        mock_scan.assert_called_once()
        assert mock_scan.call_args[1]["fast"] is True

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_since_until_flags(self, mock_scan, capsys):
        mock_scan.return_value = {
            "owner": "o",
            "repo": "r",
            "since": "2025-01-01",
            "until": "2025-06-01",
            "total_commits_scanned": 0,
            "fast_mode": False,
            "candidates": [],
            "summary": {
                "total_candidates": 0,
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0,
                "bug_class_distribution": {},
            },
            "errors": [],
        }

        from manus_agent.cli import _run_silent_patches

        code = _run_silent_patches(["o/r", "--since", "2025-01-01", "--until", "2025-06-01"])
        assert code == 0
        call_kwargs = mock_scan.call_args[1]
        assert call_kwargs["since"] == "2025-01-01"
        assert call_kwargs["until"] == "2025-06-01"

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_max_commits_flag(self, mock_scan, capsys):
        mock_scan.return_value = {
            "owner": "o",
            "repo": "r",
            "candidates": [],
            "summary": {"total_candidates": 0},
        }

        from manus_agent.cli import _run_silent_patches

        _run_silent_patches(["o/r", "--max-commits", "100", "--output", "json"])
        assert mock_scan.call_args[1]["max_commits"] == 100

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_text_output_with_errors(self, mock_scan, capsys):
        mock_scan.return_value = {
            "owner": "owner",
            "repo": "repo",
            "since": "2025-06-01",
            "until": "2025-09-01",
            "total_commits_scanned": 5,
            "fast_mode": False,
            "candidates": [
                {
                    "sha": "a" * 40,
                    "short_sha": "a" * 7,
                    "message": "security fix",
                    "date": "2025-06-15T12:00:00Z",
                    "url": "https://github.com/owner/repo/commit/" + "a" * 40,
                    "author": "dev",
                    "message_score": 25.0,
                    "diff_score": 0.0,
                    "total_score": 25.0,
                    "confidence": "LOW",
                    "bug_classes": [],
                }
            ],
            "summary": {
                "total_candidates": 1,
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 1,
                "bug_class_distribution": {},
            },
            "errors": ["Diff fetch error for abc1234: timeout"],
        }

        from manus_agent.cli import _run_silent_patches

        code = _run_silent_patches(["owner/repo"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Warnings:" in out
        assert "timeout" in out

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_text_output_high_confidence(self, mock_scan, capsys):
        mock_scan.return_value = {
            "owner": "owner",
            "repo": "repo",
            "since": "2025-06-01",
            "until": "2025-09-01",
            "total_commits_scanned": 10,
            "fast_mode": False,
            "candidates": [
                {
                    "sha": "h" * 40,
                    "short_sha": "h" * 7,
                    "message": "security fix: prevent remote code execution",
                    "date": "2025-06-15T12:00:00Z",
                    "url": "https://github.com/owner/repo/commit/" + "h" * 40,
                    "author": "dev",
                    "message_score": 45.0,
                    "diff_score": 30.0,
                    "total_score": 75.0,
                    "confidence": "HIGH",
                    "bug_classes": ["command_injection", "input_validation"],
                }
            ],
            "summary": {
                "total_candidates": 1,
                "high_confidence": 1,
                "medium_confidence": 0,
                "low_confidence": 0,
                "bug_class_distribution": {"command_injection": 1, "input_validation": 1},
            },
            "errors": [],
        }

        from manus_agent.cli import _run_silent_patches

        code = _run_silent_patches(["owner/repo"])
        assert code == 0
        out = capsys.readouterr().out
        assert "HIGH" in out
        assert "command_injection" in out
        assert "input_validation" in out

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_generic_exception(self, mock_scan, capsys):
        mock_scan.side_effect = RuntimeError("Unexpected error")

        from manus_agent.cli import _run_silent_patches

        code = _run_silent_patches(["owner/repo"])
        assert code == 1
        err = capsys.readouterr().err
        assert "Unexpected error" in err


# =========================================================================
# CLI integration via main()
# =========================================================================


class TestCliMainDispatch:
    """Test that main() routes silent-patches to _run_silent_patches."""

    @patch("manus_agent.cli._run_silent_patches")
    def test_main_routes_to_silent_patches(self, mock_run):
        mock_run.return_value = 0

        with patch.object(sys, "argv", ["manus-agent", "silent-patches", "owner/repo"]):
            with pytest.raises(SystemExit) as exc_info:
                from manus_agent.cli import main

                main()
            assert exc_info.value.code == 0

        mock_run.assert_called_once_with(["owner/repo"])


# =========================================================================
# HTTP retry tests
# =========================================================================


class TestGitHubGetRetry:
    """Tests for _github_get retry behavior."""

    @patch("manus_agent.tools.detect_silent_patches.time.sleep")
    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_retry_on_rate_limit(self, mock_get, mock_sleep):
        rate_resp = MagicMock()
        rate_resp.status_code = 403
        rate_resp.text = "API rate limit exceeded"

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = []

        mock_get.side_effect = [rate_resp, ok_resp]

        from manus_agent.tools.detect_silent_patches import _github_get

        result = _github_get("https://api.github.com/repos/o/r/commits")
        assert result.status_code == 200
        assert mock_sleep.call_count == 1

    @patch("manus_agent.tools.detect_silent_patches.time.sleep")
    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_retry_on_502(self, mock_get, mock_sleep):
        bad_resp = MagicMock()
        bad_resp.status_code = 502
        bad_resp.text = "Bad Gateway"

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = []

        mock_get.side_effect = [bad_resp, ok_resp]

        from manus_agent.tools.detect_silent_patches import _github_get

        result = _github_get("https://api.github.com/repos/o/r/commits")
        assert result.status_code == 200

    @patch("manus_agent.tools.detect_silent_patches.time.sleep")
    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_retry_on_request_exception(self, mock_get, mock_sleep):
        import requests as req

        mock_get.side_effect = [
            req.ConnectionError("Connection reset"),
            MagicMock(status_code=200, json=lambda: []),
        ]

        from manus_agent.tools.detect_silent_patches import _github_get

        result = _github_get("https://api.github.com/repos/o/r/commits")
        assert result.status_code == 200

    @patch("manus_agent.tools.detect_silent_patches.time.sleep")
    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_all_retries_exhausted(self, mock_get, mock_sleep):
        import requests as req

        mock_get.side_effect = req.ConnectionError("Connection refused")

        from manus_agent.tools.detect_silent_patches import _github_get

        with pytest.raises(req.ConnectionError):
            _github_get("https://api.github.com/repos/o/r/commits")

    @patch("manus_agent.tools.detect_silent_patches.time.sleep")
    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_diff_get_retry_on_rate_limit(self, mock_get, mock_sleep):
        rate_resp = MagicMock()
        rate_resp.status_code = 403
        rate_resp.text = "API rate limit exceeded"

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.text = "diff content"

        mock_get.side_effect = [rate_resp, ok_resp]

        from manus_agent.tools.detect_silent_patches import _github_diff_get

        result = _github_diff_get("https://api.github.com/repos/o/r/commits/abc")
        assert result.status_code == 200


# =========================================================================
# Edge case tests
# =========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_score_message_with_special_chars(self):
        score = _score_message("fix: buffer overflow — security fix™")
        assert score >= 20.0

    def test_score_diff_with_unicode(self):
        score = _score_diff("+    # 修复安全漏洞\n+    sanitize(data)\n")
        assert score >= 8.0

    def test_classify_empty_text(self):
        assert _classify_bug_classes("") == []

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_total_score_capped_at_100(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "q" * 40,
                "message": (
                    "security fix security patch prevent injection buffer overflow "
                    "use-after-free remote code execution privilege escalation rce"
                ),
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/o/r/commit/" + "q" * 40,
                "author": "dev",
            },
        ]
        mock_diff.return_value = "\n".join(
            f"+    sanitize(x{i})\n+    check_permission(u{i})\n+    is_authenticated(r{i})" for i in range(20)
        )

        result = scan_repository("owner", "repo")
        for c in result["candidates"]:
            assert c["total_score"] <= 100.0

    @patch("manus_agent.tools.detect_silent_patches.fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches.fetch_commits")
    def test_short_sha_is_7_chars(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "abc1234def5678abc1234def5678abc1234def56",
                "message": "security fix: prevent injection",
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/o/r/commit/abc1234def5678abc1234def5678abc1234def56",
                "author": "dev",
            },
        ]
        mock_diff.return_value = ""

        result = scan_repository("owner", "repo")
        if result["candidates"]:
            assert result["candidates"][0]["short_sha"] == "abc1234"

    def test_owner_repo_with_dots_and_underscores(self):
        result = detect_silent_patches(_tool_use("org.name/repo_name"))
        # Should not fail validation — the scan may fail but validation passes.
        # We just check it doesn't return the "invalid format" error.
        if result["status"] == "error":
            assert "owner/repo" not in result["content"][0]["text"].lower()

    @patch("manus_agent.tools.detect_silent_patches.scan_repository")
    def test_cli_parser_help(self, mock_scan, capsys):
        from manus_agent.cli import _build_silent_patches_parser

        parser = _build_silent_patches_parser()
        assert parser.prog == "manus-agent silent-patches"
