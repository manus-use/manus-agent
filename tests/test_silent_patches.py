"""Comprehensive test suite for detect_silent_patches module.

100 % mocked — no real HTTP calls.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.detect_silent_patches import (
    _BUG_CLASSES,
    _CANDIDATE_THRESHOLD,
    _CVE_RE,
    _MSG_SCORE_THRESHOLD,
    _SECURITY_MSG_KEYWORDS,
    TOOL_SPEC,
    _classify_commit,
    _fetch_commit_diff,
    _fetch_commits,
    _github_headers,
    _parse_repo_spec,
    _score_diff,
    _score_message,
    _score_text_against_bug_classes,
    detect_silent_patches,
    scan_silent_patches,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_tool_use(input_dict: dict[str, Any]) -> dict[str, Any]:
    return {"toolUseId": "test-id-123", "input": input_dict}


def _make_commit(
    sha: str = "abc1234567890",
    message: str = "fix: some change",
    date: str = "2025-06-10T10:00:00Z",
    author: str = "dev",
    url: str = "https://github.com/owner/repo/commit/abc1234567890",
) -> dict[str, Any]:
    return {
        "sha": sha,
        "commit": {
            "message": message,
            "author": {"name": author, "date": date},
            "committer": {"date": date},
        },
        "html_url": url,
    }


def _mock_response(
    status_code: int = 200,
    json_data: Any = None,
    text: str = "",
    headers: dict[str, str] | None = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = text
    resp.headers = headers or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    # For streaming diffs.
    resp.iter_content = MagicMock(return_value=[text.encode("utf-8")] if text else [])
    return resp


# ===================================================================
# TOOL_SPEC validation
# ===================================================================


class TestToolSpec:
    def test_has_required_keys(self) -> None:
        assert "name" in TOOL_SPEC
        assert "description" in TOOL_SPEC
        assert "inputSchema" in TOOL_SPEC

    def test_name(self) -> None:
        assert TOOL_SPEC["name"] == "detect_silent_patches"

    def test_input_schema_has_repo_required(self) -> None:
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "repo" in schema["properties"]
        assert "repo" in schema["required"]

    def test_input_schema_has_optional_fields(self) -> None:
        schema = TOOL_SPEC["inputSchema"]["json"]
        for field in ("since", "until", "max_commits", "fast"):
            assert field in schema["properties"]

    def test_description_mentions_silent_patches(self) -> None:
        assert "silent" in TOOL_SPEC["description"].lower()


# ===================================================================
# _parse_repo_spec
# ===================================================================


class TestParseRepoSpec:
    def test_owner_repo(self) -> None:
        assert _parse_repo_spec("torvalds/linux") == ("torvalds", "linux")

    def test_full_https_url(self) -> None:
        assert _parse_repo_spec("https://github.com/torvalds/linux") == ("torvalds", "linux")

    def test_full_http_url(self) -> None:
        assert _parse_repo_spec("http://github.com/torvalds/linux") == ("torvalds", "linux")

    def test_no_scheme_url(self) -> None:
        assert _parse_repo_spec("github.com/torvalds/linux") == ("torvalds", "linux")

    def test_trailing_slash(self) -> None:
        assert _parse_repo_spec("torvalds/linux/") == ("torvalds", "linux")

    def test_url_trailing_slash(self) -> None:
        assert _parse_repo_spec("https://github.com/torvalds/linux/") == ("torvalds", "linux")

    def test_invalid_single_word(self) -> None:
        with pytest.raises(ValueError, match="Invalid repo spec"):
            _parse_repo_spec("linux")

    def test_invalid_empty(self) -> None:
        with pytest.raises(ValueError, match="Invalid repo spec"):
            _parse_repo_spec("")

    def test_whitespace_stripped(self) -> None:
        assert _parse_repo_spec("  torvalds/linux  ") == ("torvalds", "linux")


# ===================================================================
# _github_headers
# ===================================================================


class TestGithubHeaders:
    def test_no_token(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            headers = _github_headers()
            assert "Authorization" not in headers
            assert "Accept" in headers

    def test_with_token(self) -> None:
        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"}):
            headers = _github_headers()
            assert headers["Authorization"] == "token ghp_test123"


# ===================================================================
# _score_text_against_bug_classes
# ===================================================================


class TestScoreTextAgainstBugClasses:
    def test_empty_text(self) -> None:
        hits, classes = _score_text_against_bug_classes("")
        assert hits == 0
        assert classes == []

    def test_sql_injection_match(self) -> None:
        hits, classes = _score_text_against_bug_classes("fix sql injection in query executor")
        assert "sql_injection" in classes
        assert hits >= 1

    def test_buffer_overflow_match(self) -> None:
        hits, classes = _score_text_against_bug_classes("fix memcpy overflow in parser")
        assert "buffer_overflow" in classes

    def test_xss_match(self) -> None:
        hits, classes = _score_text_against_bug_classes("sanitize user input to prevent xss")
        assert "xss" in classes

    def test_auth_bypass_match(self) -> None:
        hits, classes = _score_text_against_bug_classes("fix authentication bypass in login")
        assert "auth_bypass" in classes

    def test_multiple_classes(self) -> None:
        text = "fix sql injection and xss vulnerability with proper sanitize"
        hits, classes = _score_text_against_bug_classes(text)
        assert len(classes) >= 2

    def test_case_insensitive(self) -> None:
        hits, classes = _score_text_against_bug_classes("Fix MEMCPY OVERFLOW")
        assert "buffer_overflow" in classes

    def test_no_match(self) -> None:
        hits, classes = _score_text_against_bug_classes("bump version to 2.0.0")
        assert hits == 0
        assert classes == []


# ===================================================================
# _score_message
# ===================================================================


class TestScoreMessage:
    def test_empty_message(self) -> None:
        assert _score_message("") == 0

    def test_security_fix_message(self) -> None:
        score = _score_message("fix: security vulnerability in auth module")
        assert score >= 2

    def test_plain_refactor_message(self) -> None:
        score = _score_message("refactor: rename variable x to y")
        assert score == 0

    def test_exploit_keyword(self) -> None:
        score = _score_message("fix exploit in deserialization handler")
        assert score >= 2

    def test_overflow_keyword(self) -> None:
        score = _score_message("fix: buffer overflow in packet parser")
        assert score >= 2

    def test_multiple_keywords(self) -> None:
        score = _score_message("fix: security vuln — sql injection bypass in auth")
        assert score >= 4

    def test_cve_mention_still_scored(self) -> None:
        # _score_message doesn't filter CVEs — that's done in the pipeline.
        score = _score_message("fix CVE-2024-1234 sql injection")
        assert score >= 2


# ===================================================================
# _score_diff
# ===================================================================


class TestScoreDiff:
    def test_empty_diff(self) -> None:
        score, classes = _score_diff("")
        assert score == 0
        assert classes == []

    def test_added_sanitize_line(self) -> None:
        diff = "+    user_input = sanitize(user_input)\n-    # no validation"
        score, classes = _score_diff(diff)
        assert score >= 1

    def test_header_lines_ignored(self) -> None:
        diff = "--- a/file.py\n+++ b/file.py\n+sanitize(input)"
        score, classes = _score_diff(diff)
        # Only the +sanitize line should contribute.
        assert score >= 1

    def test_diff_with_security_fix(self) -> None:
        diff = (
            "+    if not user.is_authenticated():\n"
            "+        raise PermissionError('Unauthorized')\n"
            "-    # TODO: add auth check\n"
        )
        score, classes = _score_diff(diff)
        assert score >= 1
        assert "auth_bypass" in classes

    def test_diff_with_memcpy_fix(self) -> None:
        diff = "+    memcpy(dst, src, MIN(len, sizeof(dst)));\n-    memcpy(dst, src, len);"
        score, classes = _score_diff(diff)
        assert "buffer_overflow" in classes

    def test_non_security_diff(self) -> None:
        diff = "+    print('hello world')\n-    print('goodbye')"
        score, classes = _score_diff(diff)
        assert score == 0


# ===================================================================
# _classify_commit
# ===================================================================


class TestClassifyCommit:
    def test_low_score_not_candidate(self) -> None:
        result = _classify_commit("bump version to 2.0.0", None)
        assert result["is_candidate"] is False
        assert result["total_score"] == 0

    def test_high_message_score_candidate(self) -> None:
        msg = "fix: security vulnerability sql injection bypass in authentication"
        result = _classify_commit(msg, None)
        assert result["is_candidate"] is True
        assert result["total_score"] >= _CANDIDATE_THRESHOLD

    def test_fast_mode_skips_diff(self) -> None:
        msg = "fix security vuln"
        diff = "+sanitize(input)\n-eval(input)"
        result = _classify_commit(msg, diff, fast=True)
        assert result["diff_score"] == 0

    def test_diff_boosts_score(self) -> None:
        msg = "fix: input handling"
        diff = "+    sanitize(user_input)\n-    eval(user_input)"
        result_no_diff = _classify_commit(msg, None)
        result_with_diff = _classify_commit(msg, diff, fast=False)
        assert result_with_diff["total_score"] >= result_no_diff["total_score"]

    def test_classifications_from_both_sources(self) -> None:
        msg = "fix buffer overflow"
        diff = "+    if (!check_permission(user)):\n-    // skip auth"
        result = _classify_commit(msg, diff)
        # Should have buffer_overflow from message and auth_bypass from diff.
        assert "buffer_overflow" in result["classifications"]

    def test_fast_mode_lower_threshold(self) -> None:
        msg = "fix security vuln"
        result = _classify_commit(msg, None, fast=True)
        # In fast mode, threshold is _MSG_SCORE_THRESHOLD (2).
        assert result["total_score"] >= _MSG_SCORE_THRESHOLD
        assert result["is_candidate"] is True


# ===================================================================
# _CVE_RE
# ===================================================================


class TestCveRegex:
    def test_matches_standard_cve(self) -> None:
        assert _CVE_RE.search("fix CVE-2024-1234")

    def test_matches_long_cve(self) -> None:
        assert _CVE_RE.search("patch for CVE-2024-12345678")

    def test_no_match_without_cve(self) -> None:
        assert _CVE_RE.search("fix buffer overflow") is None

    def test_case_insensitive(self) -> None:
        assert _CVE_RE.search("fix cve-2024-1234")

    def test_embedded_in_message(self) -> None:
        assert _CVE_RE.search("Fixes CVE-2024-1234: buffer overflow in parser")


# ===================================================================
# _fetch_commits (mocked)
# ===================================================================


class TestFetchCommits:
    @patch("manus_agent.tools.detect_silent_patches._github_get_json")
    def test_empty_response(self, mock_get: MagicMock) -> None:
        mock_get.return_value = []
        result = _fetch_commits("owner", "repo")
        assert result == []

    @patch("manus_agent.tools.detect_silent_patches._github_get_json")
    def test_none_response(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        result = _fetch_commits("owner", "repo")
        assert result == []

    @patch("manus_agent.tools.detect_silent_patches._github_get_json")
    def test_single_commit(self, mock_get: MagicMock) -> None:
        mock_get.return_value = [_make_commit()]
        result = _fetch_commits("owner", "repo")
        assert len(result) == 1
        assert result[0]["sha"] == "abc1234567890"
        assert result[0]["message"] == "fix: some change"

    @patch("manus_agent.tools.detect_silent_patches._github_get_json")
    def test_max_commits_respected(self, mock_get: MagicMock) -> None:
        commits = [_make_commit(sha=f"sha{i:040d}", message=f"commit {i}") for i in range(10)]
        mock_get.return_value = commits
        result = _fetch_commits("owner", "repo", max_commits=3)
        assert len(result) == 3

    @patch("manus_agent.tools.detect_silent_patches._github_get_json")
    def test_since_until_params(self, mock_get: MagicMock) -> None:
        mock_get.return_value = []
        _fetch_commits("owner", "repo", since="2025-01-01T00:00:00Z", until="2025-06-01T00:00:00Z")
        # Should have been called at least once.
        mock_get.assert_called_once()


# ===================================================================
# _fetch_commit_diff (mocked)
# ===================================================================


class TestFetchCommitDiff:
    @patch("manus_agent.tools.detect_silent_patches._github_get_text")
    def test_returns_diff_text(self, mock_get: MagicMock) -> None:
        mock_get.return_value = "+added line\n-removed line"
        result = _fetch_commit_diff("owner", "repo", "abc123")
        assert result == "+added line\n-removed line"

    @patch("manus_agent.tools.detect_silent_patches._github_get_text")
    def test_returns_none_on_error(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        result = _fetch_commit_diff("owner", "repo", "abc123")
        assert result is None


# ===================================================================
# scan_silent_patches (integration, mocked HTTP)
# ===================================================================


class TestScanSilentPatches:
    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_empty_repo(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = []
        result = scan_silent_patches("owner/repo", now=NOW)
        assert result["repo"] == "owner/repo"
        assert result["commits_scanned"] == 0
        assert result["candidates_found"] == 0
        assert result["candidates"] == []

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_skips_cve_commits(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = [
            {
                "sha": "abc123",
                "message": "fix CVE-2024-1234 sql injection",
                "date": "2025-06-10T10:00:00Z",
                "author": "dev",
                "url": "https://github.com/owner/repo/commit/abc123",
            }
        ]
        result = scan_silent_patches("owner/repo", now=NOW)
        assert result["commits_skipped_cve"] == 1
        assert result["candidates_found"] == 0

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_detects_silent_patch(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = [
            {
                "sha": "abc1234567890abcdef",
                "message": "fix: security vulnerability — sql injection bypass in authentication handler",
                "date": "2025-06-10T10:00:00Z",
                "author": "dev",
                "url": "https://github.com/owner/repo/commit/abc1234567890abcdef",
            }
        ]
        mock_diff.return_value = "+    query = sanitize(query)\n-    cursor.execute(query)"
        result = scan_silent_patches("owner/repo", now=NOW)
        assert result["candidates_found"] >= 1
        candidate = result["candidates"][0]
        assert candidate["sha"] == "abc123456789"  # truncated to 12

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_fast_mode(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = [
            {
                "sha": "abc1234567890abcdef",
                "message": "fix: security vulnerability sql injection bypass authentication",
                "date": "2025-06-10T10:00:00Z",
                "author": "dev",
                "url": "https://github.com/owner/repo/commit/abc1234567890abcdef",
            }
        ]
        result = scan_silent_patches("owner/repo", fast=True, now=NOW)
        assert result["fast_mode"] is True
        # Should NOT have fetched any diffs.
        mock_diff.assert_not_called()

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_default_window_90_days(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = []
        result = scan_silent_patches("owner/repo", now=NOW)
        window = result["scan_window"]
        assert "2025-03-17" in window["since"]  # 90 days before June 15
        assert "2025-06-15" in window["until"]

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_custom_since_until(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = []
        result = scan_silent_patches("owner/repo", since="2025-01-01", until="2025-03-01", now=NOW)
        window = result["scan_window"]
        assert "2025-01-01" in window["since"]
        assert "2025-03-01" in window["until"]

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_url_format_accepted(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = []
        result = scan_silent_patches("https://github.com/torvalds/linux", now=NOW)
        assert result["repo"] == "torvalds/linux"

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_classification_breakdown(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = [
            {
                "sha": f"sha{i:040d}",
                "message": msg,
                "date": "2025-06-10T10:00:00Z",
                "author": "dev",
                "url": f"https://github.com/owner/repo/commit/sha{i:040d}",
            }
            for i, msg in enumerate(
                [
                    "fix: security vuln sql injection in query executor bypass",
                    "fix: buffer overflow exploit in memcpy handler vulnerability",
                    "fix: xss sanitize security vulnerability in escape html",
                ]
            )
        ]
        mock_diff.return_value = None
        result = scan_silent_patches("owner/repo", fast=True, now=NOW)
        summary = result["summary"]
        assert isinstance(summary["classification_breakdown"], dict)

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_candidates_sorted_by_score(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = [
            {
                "sha": "low_score_sha_0000000",
                "message": "fix security vuln",
                "date": "2025-06-10T10:00:00Z",
                "author": "dev",
                "url": "",
            },
            {
                "sha": "high_score_sha_000000",
                "message": "fix: security vulnerability sql injection bypass authentication overflow exploit",
                "date": "2025-06-10T10:00:00Z",
                "author": "dev",
                "url": "",
            },
        ]
        mock_diff.return_value = None
        result = scan_silent_patches("owner/repo", fast=True, now=NOW)
        candidates = result["candidates"]
        if len(candidates) >= 2:
            assert candidates[0]["total_score"] >= candidates[1]["total_score"]

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_max_commits_clamped(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = []
        # max_commits should be clamped to 5000.
        scan_silent_patches("owner/repo", max_commits=99999, now=NOW)
        # Just verify it doesn't crash.

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_non_security_commits_filtered(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = [
            {
                "sha": "abc1234567890abcdef",
                "message": "docs: update readme with new badge",
                "date": "2025-06-10T10:00:00Z",
                "author": "dev",
                "url": "",
            },
            {
                "sha": "def1234567890abcdef",
                "message": "chore: bump version to 2.0.0",
                "date": "2025-06-10T10:00:00Z",
                "author": "dev",
                "url": "",
            },
        ]
        mock_diff.return_value = None
        result = scan_silent_patches("owner/repo", fast=True, now=NOW)
        assert result["candidates_found"] == 0


# ===================================================================
# detect_silent_patches (Strands tool interface)
# ===================================================================


class TestDetectSilentPatchesTool:
    @patch("manus_agent.tools.detect_silent_patches.scan_silent_patches")
    def test_success(self, mock_scan: MagicMock) -> None:
        mock_scan.return_value = {
            "repo": "owner/repo",
            "scan_window": {"since": "2025-03-17", "until": "2025-06-15"},
            "commits_scanned": 100,
            "commits_skipped_cve": 5,
            "candidates_found": 2,
            "fast_mode": False,
            "candidates": [],
            "summary": {"total_candidates": 2, "classification_breakdown": {}, "top_classifications": []},
        }
        tool_use = _make_tool_use({"repo": "owner/repo"})
        result = detect_silent_patches(tool_use)
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-id-123"
        payload = result["content"][0]["json"]
        assert payload["repo"] == "owner/repo"

    def test_missing_repo(self) -> None:
        tool_use = _make_tool_use({})
        result = detect_silent_patches(tool_use)
        assert result["status"] == "error"
        assert "required" in result["content"][0]["text"].lower()

    def test_empty_repo(self) -> None:
        tool_use = _make_tool_use({"repo": ""})
        result = detect_silent_patches(tool_use)
        assert result["status"] == "error"

    def test_invalid_repo_spec(self) -> None:
        tool_use = _make_tool_use({"repo": "just-one-word"})
        result = detect_silent_patches(tool_use)
        assert result["status"] == "error"
        assert "invalid" in result["content"][0]["text"].lower()

    @patch("manus_agent.tools.detect_silent_patches.scan_silent_patches")
    def test_with_all_options(self, mock_scan: MagicMock) -> None:
        mock_scan.return_value = {
            "repo": "owner/repo",
            "scan_window": {},
            "commits_scanned": 0,
            "commits_skipped_cve": 0,
            "candidates_found": 0,
            "fast_mode": True,
            "candidates": [],
            "summary": {"total_candidates": 0, "classification_breakdown": {}, "top_classifications": []},
        }
        tool_use = _make_tool_use(
            {
                "repo": "owner/repo",
                "since": "2025-01-01",
                "until": "2025-06-01",
                "max_commits": 200,
                "fast": True,
            }
        )
        result = detect_silent_patches(tool_use)
        assert result["status"] == "success"
        mock_scan.assert_called_once_with(
            "owner/repo",
            since="2025-01-01",
            until="2025-06-01",
            max_commits=200,
            fast=True,
        )

    @patch("manus_agent.tools.detect_silent_patches.scan_silent_patches")
    def test_max_commits_coerced_to_int(self, mock_scan: MagicMock) -> None:
        mock_scan.return_value = {
            "repo": "owner/repo",
            "scan_window": {},
            "commits_scanned": 0,
            "commits_skipped_cve": 0,
            "candidates_found": 0,
            "fast_mode": False,
            "candidates": [],
            "summary": {"total_candidates": 0, "classification_breakdown": {}, "top_classifications": []},
        }
        tool_use = _make_tool_use({"repo": "owner/repo", "max_commits": "300"})
        result = detect_silent_patches(tool_use)
        assert result["status"] == "success"
        mock_scan.assert_called_once()
        call_kwargs = mock_scan.call_args
        assert call_kwargs[1]["max_commits"] == 300

    @patch("manus_agent.tools.detect_silent_patches.scan_silent_patches")
    def test_invalid_max_commits_defaults(self, mock_scan: MagicMock) -> None:
        mock_scan.return_value = {
            "repo": "owner/repo",
            "scan_window": {},
            "commits_scanned": 0,
            "commits_skipped_cve": 0,
            "candidates_found": 0,
            "fast_mode": False,
            "candidates": [],
            "summary": {"total_candidates": 0, "classification_breakdown": {}, "top_classifications": []},
        }
        tool_use = _make_tool_use({"repo": "owner/repo", "max_commits": "not_a_number"})
        result = detect_silent_patches(tool_use)
        assert result["status"] == "success"
        call_kwargs = mock_scan.call_args
        assert call_kwargs[1]["max_commits"] == 500


# ===================================================================
# _github_get_json (retry/back-off)
# ===================================================================


class TestGithubGetJson:
    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_success(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.detect_silent_patches import _github_get_json

        mock_get.return_value = _mock_response(200, json_data={"key": "value"})
        result = _github_get_json("https://api.github.com/test")
        assert result == {"key": "value"}

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_404_returns_none(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.detect_silent_patches import _github_get_json

        mock_get.return_value = _mock_response(404)
        result = _github_get_json("https://api.github.com/test")
        assert result is None

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_422_returns_none(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.detect_silent_patches import _github_get_json

        mock_get.return_value = _mock_response(422)
        result = _github_get_json("https://api.github.com/test")
        assert result is None

    @patch("manus_agent.tools.detect_silent_patches.time.sleep")
    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_429_retries(self, mock_get: MagicMock, mock_sleep: MagicMock) -> None:
        from manus_agent.tools.detect_silent_patches import _github_get_json

        resp_429 = _mock_response(429, headers={"Retry-After": "1"})
        resp_429.raise_for_status.side_effect = None
        resp_200 = _mock_response(200, json_data={"ok": True})
        mock_get.side_effect = [resp_429, resp_200]
        result = _github_get_json("https://api.github.com/test", max_retries=1)
        assert result == {"ok": True}

    @patch("manus_agent.tools.detect_silent_patches.time.sleep")
    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_network_error_retries(self, mock_get: MagicMock, mock_sleep: MagicMock) -> None:
        import requests as req

        from manus_agent.tools.detect_silent_patches import _github_get_json

        mock_get.side_effect = [req.ConnectionError("fail"), _mock_response(200, json_data={"ok": True})]
        result = _github_get_json("https://api.github.com/test", max_retries=1)
        assert result == {"ok": True}

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_all_retries_exhausted(self, mock_get: MagicMock) -> None:
        import requests as req

        from manus_agent.tools.detect_silent_patches import _github_get_json

        mock_get.side_effect = req.ConnectionError("fail")
        result = _github_get_json("https://api.github.com/test", max_retries=0)
        assert result is None


# ===================================================================
# _github_get_text (diff fetching)
# ===================================================================


class TestGithubGetText:
    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_returns_diff(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.detect_silent_patches import _github_get_text

        resp = _mock_response(200, text="+added\n-removed")
        mock_get.return_value = resp
        result = _github_get_text("https://api.github.com/test")
        assert "+added" in result

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_404_returns_none(self, mock_get: MagicMock) -> None:
        from manus_agent.tools.detect_silent_patches import _github_get_text

        mock_get.return_value = _mock_response(404)
        result = _github_get_text("https://api.github.com/test")
        assert result is None

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_exception_returns_none(self, mock_get: MagicMock) -> None:
        import requests as req

        from manus_agent.tools.detect_silent_patches import _github_get_text

        mock_get.side_effect = req.ConnectionError("fail")
        result = _github_get_text("https://api.github.com/test")
        assert result is None


# ===================================================================
# CLI subcommand (_run_silent_patches)
# ===================================================================


class TestCliSilentPatches:
    @patch("manus_agent.tools.detect_silent_patches.scan_silent_patches")
    def test_text_output(self, mock_scan: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        from manus_agent.cli import _run_silent_patches

        mock_scan.return_value = {
            "repo": "owner/repo",
            "scan_window": {"since": "2025-03-17T00:00:00Z", "until": "2025-06-15T12:00:00Z"},
            "commits_scanned": 50,
            "commits_skipped_cve": 3,
            "candidates_found": 1,
            "fast_mode": False,
            "candidates": [
                {
                    "sha": "abc123456789",
                    "full_sha": "abc1234567890abcdef",
                    "date": "2025-06-10T10:00:00Z",
                    "author": "dev",
                    "message": "fix: sanitize input in query handler",
                    "url": "https://github.com/owner/repo/commit/abc1234567890abcdef",
                    "message_score": 3,
                    "diff_score": 2,
                    "total_score": 5,
                    "classification": ["sql_injection", "input_validation"],
                }
            ],
            "summary": {
                "total_candidates": 1,
                "classification_breakdown": {"sql_injection": 1, "input_validation": 1},
                "top_classifications": ["sql_injection", "input_validation"],
            },
        }

        exit_code = _run_silent_patches(["owner/repo"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Silent Patch Scan" in captured.out
        assert "owner/repo" in captured.out
        assert "abc123456789" in captured.out
        assert "sql_injection" in captured.out

    @patch("manus_agent.tools.detect_silent_patches.scan_silent_patches")
    def test_json_output(self, mock_scan: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        from manus_agent.cli import _run_silent_patches

        mock_scan.return_value = {
            "repo": "owner/repo",
            "scan_window": {},
            "commits_scanned": 0,
            "commits_skipped_cve": 0,
            "candidates_found": 0,
            "fast_mode": False,
            "candidates": [],
            "summary": {"total_candidates": 0, "classification_breakdown": {}, "top_classifications": []},
        }

        exit_code = _run_silent_patches(["owner/repo", "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["repo"] == "owner/repo"

    @patch("manus_agent.tools.detect_silent_patches.scan_silent_patches")
    def test_fast_flag_passed(self, mock_scan: MagicMock) -> None:
        from manus_agent.cli import _run_silent_patches

        mock_scan.return_value = {
            "repo": "owner/repo",
            "scan_window": {},
            "commits_scanned": 0,
            "commits_skipped_cve": 0,
            "candidates_found": 0,
            "fast_mode": True,
            "candidates": [],
            "summary": {"total_candidates": 0, "classification_breakdown": {}, "top_classifications": []},
        }

        _run_silent_patches(["owner/repo", "--fast"])
        mock_scan.assert_called_once()
        assert mock_scan.call_args[1]["fast"] is True

    @patch("manus_agent.tools.detect_silent_patches.scan_silent_patches")
    def test_since_until_flags(self, mock_scan: MagicMock) -> None:
        from manus_agent.cli import _run_silent_patches

        mock_scan.return_value = {
            "repo": "owner/repo",
            "scan_window": {},
            "commits_scanned": 0,
            "commits_skipped_cve": 0,
            "candidates_found": 0,
            "fast_mode": False,
            "candidates": [],
            "summary": {"total_candidates": 0, "classification_breakdown": {}, "top_classifications": []},
        }

        _run_silent_patches(["owner/repo", "--since", "2025-01-01", "--until", "2025-06-01"])
        mock_scan.assert_called_once()
        assert mock_scan.call_args[1]["since"] == "2025-01-01"
        assert mock_scan.call_args[1]["until"] == "2025-06-01"

    @patch("manus_agent.tools.detect_silent_patches.scan_silent_patches")
    def test_max_commits_flag(self, mock_scan: MagicMock) -> None:
        from manus_agent.cli import _run_silent_patches

        mock_scan.return_value = {
            "repo": "owner/repo",
            "scan_window": {},
            "commits_scanned": 0,
            "commits_skipped_cve": 0,
            "candidates_found": 0,
            "fast_mode": False,
            "candidates": [],
            "summary": {"total_candidates": 0, "classification_breakdown": {}, "top_classifications": []},
        }

        _run_silent_patches(["owner/repo", "--max-commits", "100"])
        assert mock_scan.call_args[1]["max_commits"] == 100

    @patch("manus_agent.tools.detect_silent_patches.scan_silent_patches")
    def test_no_candidates_text(self, mock_scan: MagicMock, capsys: pytest.CaptureFixture[str]) -> None:
        from manus_agent.cli import _run_silent_patches

        mock_scan.return_value = {
            "repo": "owner/repo",
            "scan_window": {"since": "2025-03-17T00:00:00Z", "until": "2025-06-15T12:00:00Z"},
            "commits_scanned": 50,
            "commits_skipped_cve": 0,
            "candidates_found": 0,
            "fast_mode": False,
            "candidates": [],
            "summary": {"total_candidates": 0, "classification_breakdown": {}, "top_classifications": []},
        }

        exit_code = _run_silent_patches(["owner/repo"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No silent-patch candidates" in captured.out


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_bug_classes_has_14_entries(self) -> None:
        assert len(_BUG_CLASSES) == 14

    def test_security_keywords_non_empty(self) -> None:
        assert len(_SECURITY_MSG_KEYWORDS) > 10

    def test_thresholds_reasonable(self) -> None:
        assert _MSG_SCORE_THRESHOLD >= 1
        assert _CANDIDATE_THRESHOLD >= _MSG_SCORE_THRESHOLD

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_message_truncated_in_output(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        long_msg = "fix: security vulnerability " + "x" * 300
        mock_commits.return_value = [
            {
                "sha": "abc1234567890abcdef",
                "message": long_msg,
                "date": "2025-06-10T10:00:00Z",
                "author": "dev",
                "url": "",
            }
        ]
        mock_diff.return_value = None
        result = scan_silent_patches("owner/repo", fast=True, now=NOW)
        if result["candidates"]:
            assert len(result["candidates"][0]["message"]) <= 200

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_diff_fetch_only_for_high_msg_score(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = [
            {
                "sha": "low_score_sha_000000",
                "message": "update readme",
                "date": "2025-06-10T10:00:00Z",
                "author": "dev",
                "url": "",
            }
        ]
        scan_silent_patches("owner/repo", fast=False, now=NOW)
        # Low score message should NOT trigger diff fetch.
        mock_diff.assert_not_called()

    @patch("manus_agent.tools.detect_silent_patches._fetch_commit_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_diff_fetched_for_high_msg_score(self, mock_commits: MagicMock, mock_diff: MagicMock) -> None:
        mock_commits.return_value = [
            {
                "sha": "high_score_sha_00000",
                "message": "fix: security vulnerability in authentication bypass handler",
                "date": "2025-06-10T10:00:00Z",
                "author": "dev",
                "url": "",
            }
        ]
        mock_diff.return_value = "+check_permission(user)"
        scan_silent_patches("owner/repo", fast=False, now=NOW)
        mock_diff.assert_called_once()


# ===================================================================
# Bug-class coverage (ensure all 14 classes can be detected)
# ===================================================================


class TestBugClassCoverage:
    """Verify each of the 14 bug classes can be detected from a message."""

    @pytest.mark.parametrize(
        "bug_class,message",
        [
            ("sql_injection", "fix sql injection in query builder"),
            ("command_injection", "fix: prevent shell=True command injection via subprocess"),
            ("path_traversal", "fix: prevent ../ path traversal in file upload"),
            ("buffer_overflow", "fix: memcpy overflow in packet handler"),
            ("integer_overflow", "fix: integer overflow in size_t calculation"),
            ("use_after_free", "fix: use-after-free in kfree handler"),
            ("null_dereference", "fix: null pointer dereference check"),
            ("auth_bypass", "fix: authentication bypass in permission check"),
            ("deserialization", "fix: unsafe pickle deserialization"),
            ("xss", "fix: xss via unsanitized innerhtml"),
            ("ssrf", "fix: ssrf via urlopen to localhost"),
            ("input_validation", "fix: input validation with allowlist regex"),
            ("race_condition", "fix: race condition with mutex lock"),
            ("cryptographic", "fix: weak cipher in tls encryption"),
        ],
    )
    def test_class_detected(self, bug_class: str, message: str) -> None:
        _, classes = _score_text_against_bug_classes(message)
        assert bug_class in classes, f"Expected {bug_class} in {classes} for message: {message}"
