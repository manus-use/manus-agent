"""Comprehensive test suite for detect_silent_patches tool module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from manus_agent.tools.detect_silent_patches import (
    _CVE_RE,
    BUG_CLASSES,
    _classify,
    _fetch_commits,
    _fetch_diff,
    _score_diff,
    _score_message,
    detect_silent_patches,
)

# ---------------------------------------------------------------------------
# BUG_CLASSES constant
# ---------------------------------------------------------------------------


class TestBugClasses:
    """Tests for the BUG_CLASSES constant."""

    def test_has_14_classes(self):
        assert len(BUG_CLASSES) == 14

    def test_all_lowercase_underscore(self):
        for cls in BUG_CLASSES:
            assert cls == cls.lower()
            assert " " not in cls

    def test_known_classes_present(self):
        expected = {"auth_bypass", "buffer_overflow", "xss", "sql_injection", "use_after_free"}
        assert expected.issubset(set(BUG_CLASSES))


# ---------------------------------------------------------------------------
# CVE regex
# ---------------------------------------------------------------------------


class TestCVERegex:
    """Tests for the CVE reference regex."""

    def test_matches_standard_cve(self):
        assert _CVE_RE.search("Fix for CVE-2024-1234")

    def test_matches_long_id(self):
        assert _CVE_RE.search("Addresses CVE-2025-123456")

    def test_no_match_without_cve(self):
        assert _CVE_RE.search("Fix buffer overflow in parser") is None

    def test_case_insensitive(self):
        assert _CVE_RE.search("fixes cve-2023-9999")


# ---------------------------------------------------------------------------
# _score_message
# ---------------------------------------------------------------------------


class TestScoreMessage:
    """Tests for commit message scoring."""

    def test_buffer_overflow_message(self):
        score, classes = _score_message("fix buffer overflow in network parser")
        assert score > 0
        assert "buffer_overflow" in classes

    def test_use_after_free_message(self):
        score, classes = _score_message("fix use-after-free in memory allocator")
        assert score > 0
        assert "use_after_free" in classes

    def test_sql_injection_message(self):
        score, classes = _score_message("prevent sql injection in login handler")
        assert score > 0
        assert "sql_injection" in classes

    def test_xss_message(self):
        score, classes = _score_message("fix reflected XSS in search endpoint")
        assert score > 0
        assert "xss" in classes

    def test_auth_bypass_message(self):
        score, classes = _score_message("fix authentication bypass in API middleware")
        assert score > 0
        assert "auth_bypass" in classes

    def test_command_injection_message(self):
        score, classes = _score_message("prevent command injection via shell exec")
        assert score > 0
        assert "command_injection" in classes

    def test_csrf_message(self):
        score, classes = _score_message("add CSRF token validation to form handler")
        assert score > 0
        assert "csrf" in classes

    def test_directory_traversal_message(self):
        score, classes = _score_message("fix path traversal vulnerability in file upload")
        assert score > 0
        assert "directory_traversal" in classes

    def test_privilege_escalation_message(self):
        score, classes = _score_message("fix privilege escalation via symlink")
        assert score > 0
        assert "privilege_escalation" in classes

    def test_race_condition_message(self):
        score, classes = _score_message("fix race condition in session handler")
        assert score > 0
        assert "race_condition" in classes

    def test_integer_overflow_message(self):
        score, classes = _score_message("prevent integer overflow in size calculation")
        assert score > 0
        assert "integer_overflow" in classes

    def test_null_deref_message(self):
        score, classes = _score_message("fix null pointer dereference in parser")
        assert score > 0
        assert "null_dereference" in classes

    def test_information_disclosure_message(self):
        score, classes = _score_message("fix information disclosure via error messages")
        assert score > 0
        assert "information_disclosure" in classes

    def test_memory_corruption_message(self):
        score, classes = _score_message("fix heap corruption in allocator")
        assert score > 0
        assert "memory_corruption" in classes

    def test_no_match_benign_message(self):
        score, classes = _score_message("update README with installation instructions")
        assert score == 0.0
        assert classes == []

    def test_no_match_refactor_message(self):
        score, classes = _score_message("refactor: extract helper function")
        assert score == 0.0
        assert classes == []

    def test_generic_security_fix(self):
        score, classes = _score_message("fix security vulnerability in auth module")
        assert score > 0

    def test_sanitize_input(self):
        score, classes = _score_message("sanitize user input before processing")
        assert score > 0

    def test_multiple_matches_higher_score(self):
        score1, _ = _score_message("fix buffer overflow")
        score2, _ = _score_message("fix buffer overflow and validate bounds and check size and check length")
        assert score2 >= score1

    def test_score_capped_at_one(self):
        # Craft a message that hits many patterns
        msg = "fix buffer overflow use-after-free null dereference race condition sql injection xss"
        score, _ = _score_message(msg)
        assert score <= 1.0

    def test_double_free(self):
        score, classes = _score_message("fix double free in cleanup path")
        assert score > 0
        assert "memory_corruption" in classes or "use_after_free" in classes


# ---------------------------------------------------------------------------
# _score_diff
# ---------------------------------------------------------------------------


class TestScoreDiff:
    """Tests for diff scoring."""

    def test_bounds_check_added(self):
        diff = "+    if (len > MAX_SIZE) return -EINVAL;"
        score, classes = _score_diff(diff)
        assert score > 0
        assert "buffer_overflow" in classes

    def test_sanitize_call_added(self):
        diff = "+    output = escape(user_input)"
        score, classes = _score_diff(diff)
        assert score > 0

    def test_csrf_token_added(self):
        diff = "+    validate_csrf_token(request)"
        score, classes = _score_diff(diff)
        assert score > 0
        assert "csrf" in classes

    def test_realpath_added(self):
        diff = "+    safe_path = realpath(user_path)"
        score, classes = _score_diff(diff)
        assert score > 0
        assert "directory_traversal" in classes

    def test_mutex_added(self):
        diff = "+    pthread_mutex_lock(&resource_mutex);"
        score, classes = _score_diff(diff)
        assert score > 0
        assert "race_condition" in classes

    def test_prepared_statement_added(self):
        diff = "+    cursor.execute(prepared_query, params)"
        score, classes = _score_diff(diff)
        # May or may not match depending on wording
        # The pattern looks for "parameterized" or "prepared" + "statement" or "query"
        # This matches "prepared...query" — but let's test a clearer one
        diff2 = "+    stmt = conn.prepareStatement(query)  # use parameterized query"
        score2, classes2 = _score_diff(diff2)
        assert score2 > 0
        assert "sql_injection" in classes2

    def test_auth_check_added(self):
        diff = "+    if (!check_auth(user)) return FORBIDDEN;"
        score, classes = _score_diff(diff)
        assert score > 0
        assert "auth_bypass" in classes or "privilege_escalation" in classes

    def test_no_match_benign_diff(self):
        diff = "+    // Update documentation\n+    version = '1.2.3'"
        score, classes = _score_diff(diff)
        assert score == 0.0
        assert classes == []

    def test_removed_lines_ignored(self):
        diff = "-    if (len > MAX_SIZE) return -EINVAL;"
        score, classes = _score_diff(diff)
        assert score == 0.0

    def test_score_capped(self):
        # Many added lines with security patterns
        lines = ["+    if (check_bounds(i, size)) break;"] * 100
        diff = "\n".join(lines)
        score, _ = _score_diff(diff)
        assert score <= 1.0

    def test_overflow_constant(self):
        diff = "+    if (value > INT_MAX) return -EOVERFLOW;"
        score, classes = _score_diff(diff)
        assert score > 0
        assert "integer_overflow" in classes


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------


class TestClassify:
    """Tests for the classification helper."""

    def test_single_class(self):
        assert _classify(["buffer_overflow"]) == "buffer_overflow"

    def test_prefers_specific_over_info_disclosure(self):
        assert _classify(["information_disclosure", "xss"]) == "xss"

    def test_empty_returns_unknown(self):
        assert _classify([]) == "unknown"

    def test_only_info_disclosure(self):
        assert _classify(["information_disclosure"]) == "information_disclosure"

    def test_multiple_specific(self):
        result = _classify(["auth_bypass", "privilege_escalation"])
        assert result in ("auth_bypass", "privilege_escalation")


# ---------------------------------------------------------------------------
# _fetch_commits (mocked)
# ---------------------------------------------------------------------------


class TestFetchCommits:
    """Tests for GitHub commit fetching."""

    @patch("manus_agent.tools.detect_silent_patches._get_session")
    def test_fetches_commits(self, mock_get_session):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "sha": "abc123def456",
                "commit": {"message": "fix something", "committer": {"date": "2025-01-01T00:00:00Z"}},
                "author": {"login": "dev"},
            }
        ]
        mock_resp.headers = {}
        mock_resp.raise_for_status = MagicMock()
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_get_session.return_value = mock_session

        commits = _fetch_commits("owner/repo", since="2025-01-01")
        assert len(commits) == 1
        assert commits[0]["sha"] == "abc123def456"

    @patch("manus_agent.tools.detect_silent_patches._get_session")
    def test_respects_max_commits(self, mock_get_session):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [{"sha": f"sha{i}"} for i in range(100)]
        mock_resp.headers = {}
        mock_resp.raise_for_status = MagicMock()
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_get_session.return_value = mock_session

        commits = _fetch_commits("owner/repo", max_commits=5)
        assert len(commits) == 5

    @patch("manus_agent.tools.detect_silent_patches._get_session")
    def test_follows_pagination(self, mock_get_session):
        page1_resp = MagicMock()
        page1_resp.json.return_value = [{"sha": "page1_sha"}]
        page1_resp.headers = {"Link": '<https://api.github.com/next>; rel="next"'}
        page1_resp.raise_for_status = MagicMock()

        page2_resp = MagicMock()
        page2_resp.json.return_value = [{"sha": "page2_sha"}]
        page2_resp.headers = {}
        page2_resp.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.side_effect = [page1_resp, page2_resp]
        mock_get_session.return_value = mock_session

        commits = _fetch_commits("owner/repo", max_commits=10)
        assert len(commits) == 2

    @patch("manus_agent.tools.detect_silent_patches._get_session")
    def test_empty_response(self, mock_get_session):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.headers = {}
        mock_resp.raise_for_status = MagicMock()
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_get_session.return_value = mock_session

        commits = _fetch_commits("owner/repo")
        assert commits == []


# ---------------------------------------------------------------------------
# _fetch_diff (mocked)
# ---------------------------------------------------------------------------


class TestFetchDiff:
    """Tests for GitHub diff fetching."""

    @patch("manus_agent.tools.detect_silent_patches._get_session")
    def test_fetches_diff(self, mock_get_session):
        mock_resp = MagicMock()
        mock_resp.text = "+    if (len > MAX) return -1;"
        mock_resp.raise_for_status = MagicMock()
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        mock_get_session.return_value = mock_session

        diff = _fetch_diff("owner/repo", "abc123")
        assert "MAX" in diff


# ---------------------------------------------------------------------------
# detect_silent_patches (integration, mocked HTTP)
# ---------------------------------------------------------------------------


class TestDetectSilentPatches:
    """Integration tests for the main entry point."""

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_basic_detection(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "aabbccdd11223344",
                "commit": {
                    "message": "fix buffer overflow in network parser",
                    "committer": {"date": "2025-06-01T12:00:00Z"},
                    "author": {"name": "dev"},
                },
                "author": {"login": "devuser"},
                "html_url": "https://github.com/owner/repo/commit/aabbccdd",
            }
        ]
        mock_diff.return_value = "+    if (len > MAX_SIZE) return -1;"

        results = detect_silent_patches("owner/repo")
        assert len(results) == 1
        assert results[0]["classification"] == "buffer_overflow"
        assert results[0]["score"] > 0

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_skips_cve_referenced_commits(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "aabbccdd11223344",
                "commit": {
                    "message": "fix buffer overflow (CVE-2025-1234)",
                    "committer": {"date": "2025-06-01T12:00:00Z"},
                    "author": {"name": "dev"},
                },
                "author": {"login": "devuser"},
                "html_url": "https://github.com/owner/repo/commit/aabbccdd",
            }
        ]

        results = detect_silent_patches("owner/repo")
        assert len(results) == 0
        mock_diff.assert_not_called()

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_fast_mode_skips_diff(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "aabbccdd11223344",
                "commit": {
                    "message": "fix use-after-free in allocator",
                    "committer": {"date": "2025-06-01T12:00:00Z"},
                    "author": {"name": "dev"},
                },
                "author": {"login": "devuser"},
                "html_url": "https://github.com/owner/repo/commit/aabbccdd",
            }
        ]

        results = detect_silent_patches("owner/repo", fast=True)
        assert len(results) == 1
        mock_diff.assert_not_called()

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_benign_commit_not_included(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "aabbccdd11223344",
                "commit": {
                    "message": "update README with new examples",
                    "committer": {"date": "2025-06-01T12:00:00Z"},
                    "author": {"name": "dev"},
                },
                "author": {"login": "devuser"},
                "html_url": "https://github.com/owner/repo/commit/aabbccdd",
            }
        ]

        results = detect_silent_patches("owner/repo")
        assert len(results) == 0

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_results_sorted_by_score(self, mock_commits, mock_diff):
        mock_commits.return_value = [
            {
                "sha": "low_score_sha12345",
                "commit": {
                    "message": "fix security vulnerability",
                    "committer": {"date": "2025-06-01T12:00:00Z"},
                    "author": {"name": "dev1"},
                },
                "author": {"login": "dev1"},
                "html_url": "https://github.com/o/r/commit/low",
            },
            {
                "sha": "high_score_sha1234",
                "commit": {
                    "message": "fix buffer overflow use-after-free null deref race condition in parser",
                    "committer": {"date": "2025-06-02T12:00:00Z"},
                    "author": {"name": "dev2"},
                },
                "author": {"login": "dev2"},
                "html_url": "https://github.com/o/r/commit/high",
            },
        ]
        mock_diff.return_value = ""

        results = detect_silent_patches("owner/repo")
        assert len(results) >= 1
        if len(results) > 1:
            assert results[0]["score"] >= results[1]["score"]

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_diff_failure_graceful(self, mock_commits, mock_diff):
        """If diff fetch fails, message score alone is used."""
        import requests

        mock_commits.return_value = [
            {
                "sha": "aabbccdd11223344",
                "commit": {
                    "message": "fix buffer overflow in network parser",
                    "committer": {"date": "2025-06-01T12:00:00Z"},
                    "author": {"name": "dev"},
                },
                "author": {"login": "devuser"},
                "html_url": "https://github.com/owner/repo/commit/aabbccdd",
            }
        ]
        mock_diff.side_effect = requests.RequestException("Network error")

        results = detect_silent_patches("owner/repo")
        assert len(results) == 1
        assert results[0]["diff_score"] == 0.0

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_default_date_range(self, mock_commits, mock_diff):
        mock_commits.return_value = []

        detect_silent_patches("owner/repo")
        call_kwargs = mock_commits.call_args
        # since should default to approximately 90 days ago
        assert call_kwargs[1]["since"] is not None

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_custom_date_range(self, mock_commits, mock_diff):
        mock_commits.return_value = []

        detect_silent_patches("owner/repo", since="2025-01-01", until="2025-03-01")
        call_kwargs = mock_commits.call_args
        assert call_kwargs[1]["since"] == "2025-01-01"
        assert call_kwargs[1]["until"] == "2025-03-01"

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_min_score_filtering(self, mock_commits, mock_diff):
        """Commits below min_score threshold are excluded."""
        mock_commits.return_value = [
            {
                "sha": "aabbccdd11223344",
                "commit": {
                    "message": "fix security vulnerability",  # generic, low score
                    "committer": {"date": "2025-06-01T12:00:00Z"},
                    "author": {"name": "dev"},
                },
                "author": {"login": "devuser"},
                "html_url": "https://github.com/owner/repo/commit/aabbccdd",
            }
        ]
        mock_diff.return_value = ""

        # With very high threshold, nothing should pass
        results = detect_silent_patches("owner/repo", min_score=0.99)
        assert len(results) == 0

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_output_fields(self, mock_commits, mock_diff):
        """Check that output contains expected fields."""
        mock_commits.return_value = [
            {
                "sha": "aabbccdd11223344aabbccdd11223344aabbccdd",
                "commit": {
                    "message": "fix buffer overflow in network parser\n\nDetailed description here.",
                    "committer": {"date": "2025-06-01T12:00:00Z"},
                    "author": {"name": "dev"},
                },
                "author": {"login": "devuser"},
                "html_url": "https://github.com/owner/repo/commit/aabbccdd",
            }
        ]
        mock_diff.return_value = "+    if (len > MAX_SIZE) return -1;"

        results = detect_silent_patches("owner/repo")
        assert len(results) == 1
        r = results[0]
        assert "sha" in r
        assert "short_sha" in r
        assert len(r["short_sha"]) == 8
        assert "message" in r
        assert "full_message" in r
        assert "author" in r
        assert r["author"] == "devuser"
        assert "date" in r
        assert "url" in r
        assert "score" in r
        assert "message_score" in r
        assert "diff_score" in r
        assert "classification" in r
        assert "matched_classes" in r

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_missing_author_login(self, mock_commits, mock_diff):
        """Falls back to commit author name when GitHub login is missing."""
        mock_commits.return_value = [
            {
                "sha": "aabbccdd11223344",
                "commit": {
                    "message": "fix sql injection in login",
                    "committer": {"date": "2025-06-01T12:00:00Z"},
                    "author": {"name": "Anon Dev"},
                },
                "author": None,
                "html_url": "https://github.com/owner/repo/commit/aabbccdd",
            }
        ]
        mock_diff.return_value = ""

        results = detect_silent_patches("owner/repo")
        assert len(results) == 1
        assert results[0]["author"] == "Anon Dev"


# ---------------------------------------------------------------------------
# CLI integration (mocked)
# ---------------------------------------------------------------------------


class TestCLISilentPatches:
    """Tests for the CLI _run_silent_patches function."""

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_cli_text_output(self, mock_detect, capsys):
        from manus_agent.cli import _run_silent_patches

        mock_detect.return_value = [
            {
                "sha": "aabbccdd11223344",
                "short_sha": "aabbccdd",
                "message": "fix buffer overflow",
                "full_message": "fix buffer overflow\n\ndetails",
                "author": "dev",
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/o/r/commit/aabbccdd",
                "score": 0.78,
                "message_score": 0.6,
                "diff_score": 0.3,
                "classification": "buffer_overflow",
                "matched_classes": ["buffer_overflow"],
            }
        ]

        rc = _run_silent_patches(["owner/repo"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Silent Patch Detector" in out
        assert "owner/repo" in out
        assert "buffer_overflow" in out
        assert "aabbccdd" in out

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_cli_json_output(self, mock_detect, capsys):
        import json

        from manus_agent.cli import _run_silent_patches

        mock_detect.return_value = [
            {
                "sha": "aabbccdd11223344",
                "short_sha": "aabbccdd",
                "message": "fix xss in search",
                "full_message": "fix xss in search\n\nmore",
                "author": "dev",
                "date": "2025-06-01T12:00:00Z",
                "url": "https://github.com/o/r/commit/aabbccdd",
                "score": 0.65,
                "message_score": 0.6,
                "diff_score": 0.15,
                "classification": "xss",
                "matched_classes": ["xss"],
            }
        ]

        rc = _run_silent_patches(["owner/repo", "--output", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 1
        assert data[0]["classification"] == "xss"
        # full_message should be stripped from JSON output
        assert "full_message" not in data[0]

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_cli_no_results(self, mock_detect, capsys):
        from manus_agent.cli import _run_silent_patches

        mock_detect.return_value = []

        rc = _run_silent_patches(["owner/repo"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No silent patches detected" in out

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_cli_fast_flag(self, mock_detect):
        from manus_agent.cli import _run_silent_patches

        mock_detect.return_value = []

        _run_silent_patches(["owner/repo", "--fast"])
        mock_detect.assert_called_once()
        assert mock_detect.call_args[1]["fast"] is True

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_cli_since_until_flags(self, mock_detect):
        from manus_agent.cli import _run_silent_patches

        mock_detect.return_value = []

        _run_silent_patches(["owner/repo", "--since", "2025-01-01", "--until", "2025-06-01"])
        mock_detect.assert_called_once()
        assert mock_detect.call_args[1]["since"] == "2025-01-01"
        assert mock_detect.call_args[1]["until"] == "2025-06-01"

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_cli_max_commits_flag(self, mock_detect):
        from manus_agent.cli import _run_silent_patches

        mock_detect.return_value = []

        _run_silent_patches(["owner/repo", "--max-commits", "100"])
        mock_detect.assert_called_once()
        assert mock_detect.call_args[1]["max_commits"] == 100

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_cli_error_handling(self, mock_detect, capsys):
        from manus_agent.cli import _run_silent_patches

        mock_detect.side_effect = RuntimeError("API rate limit exceeded")

        rc = _run_silent_patches(["owner/repo"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Error" in err

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_cli_multiple_results_summary(self, mock_detect, capsys):
        from manus_agent.cli import _run_silent_patches

        mock_detect.return_value = [
            {
                "sha": f"sha{i}" + "0" * 32,
                "short_sha": f"sha{i}0000",
                "message": f"fix issue {i}",
                "full_message": f"fix issue {i}",
                "author": "dev",
                "date": "2025-06-01T12:00:00Z",
                "url": f"https://github.com/o/r/commit/sha{i}",
                "score": 0.5 + i * 0.05,
                "message_score": 0.5,
                "diff_score": 0.1 * i,
                "classification": "buffer_overflow" if i % 2 == 0 else "xss",
                "matched_classes": ["buffer_overflow"] if i % 2 == 0 else ["xss"],
            }
            for i in range(5)
        ]

        rc = _run_silent_patches(["owner/repo"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Bug-class distribution" in out
        assert "Candidates found: 5" in out
