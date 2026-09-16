"""
Tests for src/manus_agent/tools/detect_silent_patches.py and the silent-patches CLI subcommand.

All network calls are mocked — no real HTTP requests are made.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_FAKE_COMMIT_SECURITY = {
    "sha": "abc12345deadbeef",
    "commit": {
        "message": "Fix SQL injection in user lookup endpoint",
        "author": {"date": "2026-08-15T10:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/abc12345deadbeef",
}

_FAKE_COMMIT_BUFFER_OVERFLOW = {
    "sha": "def67890cafebabe",
    "commit": {
        "message": "Fix buffer overflow in packet parser — bounds check added",
        "author": {"date": "2026-08-16T12:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/def67890cafebabe",
}

_FAKE_COMMIT_CVE = {
    "sha": "1111111122222222",
    "commit": {
        "message": "Fix CVE-2026-1234: auth bypass in admin panel",
        "author": {"date": "2026-08-17T14:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/1111111122222222",
}

_FAKE_COMMIT_BENIGN = {
    "sha": "aaaa0000bbbb1111",
    "commit": {
        "message": "Add unit tests for config parser",
        "author": {"date": "2026-08-18T16:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/aaaa0000bbbb1111",
}

_FAKE_COMMIT_XSS = {
    "sha": "cccc2222dddd3333",
    "commit": {
        "message": "Sanitize HTML to prevent cross-site scripting in comments",
        "author": {"date": "2026-08-19T09:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/cccc2222dddd3333",
}

_FAKE_COMMIT_AUTH_BYPASS = {
    "sha": "eeee4444ffff5555",
    "commit": {
        "message": "Fix authentication bypass in API middleware — add permission check",
        "author": {"date": "2026-08-20T11:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/eeee4444ffff5555",
}

_FAKE_COMMIT_SSRF = {
    "sha": "6666777788889999",
    "commit": {
        "message": "Prevent SSRF via URL validation for webhook endpoints",
        "author": {"date": "2026-08-21T08:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/6666777788889999",
}

_FAKE_COMMIT_DESERIALIZATION = {
    "sha": "aabbccdd11223344",
    "commit": {
        "message": "Replace pickle.loads with yaml.safe_load for untrusted input — insecure deserialization fix",
        "author": {"date": "2026-08-22T07:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/aabbccdd11223344",
}

_FAKE_COMMIT_DOS = {
    "sha": "ddddeeeeffffaaaa",
    "commit": {
        "message": "Fix denial of service via regex DOS in input validator",
        "author": {"date": "2026-08-23T15:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/ddddeeeeffffaaaa",
}

_FAKE_COMMIT_INFO_LEAK = {
    "sha": "0000111122223333",
    "commit": {
        "message": "Fix information disclosure — redact credentials from error logs",
        "author": {"date": "2026-08-24T10:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/0000111122223333",
}

_FAKE_COMMIT_GENERIC_SECURITY = {
    "sha": "4444555566667777",
    "commit": {
        "message": "Security hardening: sanitize and validate input across endpoints",
        "author": {"date": "2026-08-25T13:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/4444555566667777",
}

_FAKE_COMMIT_PATH_TRAVERSAL = {
    "sha": "8888999900001111",
    "commit": {
        "message": "Fix path traversal in file upload — use realpath to prevent directory traversal",
        "author": {"date": "2026-08-26T09:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/8888999900001111",
}

_FAKE_COMMIT_UAF = {
    "sha": "2222333344445555",
    "commit": {
        "message": "Fix use-after-free in connection handler — dangling pointer on close",
        "author": {"date": "2026-08-27T11:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/2222333344445555",
}

_FAKE_COMMIT_CSRF = {
    "sha": "6666777788880000",
    "commit": {
        "message": "Add CSRF token validation to form submission endpoint",
        "author": {"date": "2026-08-28T14:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/6666777788880000",
}

_FAKE_COMMIT_CMD_INJECTION = {
    "sha": "aaaa1111bbbb2222",
    "commit": {
        "message": "Fix command injection via shell=true in subprocess call — security",
        "author": {"date": "2026-08-29T16:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/aaaa1111bbbb2222",
}

_FAKE_COMMIT_INT_OVERFLOW = {
    "sha": "cccc3333dddd4444",
    "commit": {
        "message": "Fix integer overflow in size calculation leading to wraparound",
        "author": {"date": "2026-08-30T08:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/cccc3333dddd4444",
}

_FAKE_COMMIT_NULL_DEREF = {
    "sha": "eeee5555ffff6666",
    "commit": {
        "message": "Add null check to prevent null pointer dereference in parser",
        "author": {"date": "2026-08-31T10:00:00Z"},
    },
    "html_url": "https://github.com/owner/repo/commit/eeee5555ffff6666",
}


_FAKE_DIFF_SQL = """\
diff --git a/db/queries.py b/db/queries.py
--- a/db/queries.py
+++ b/db/queries.py
@@ -10,5 +10,6 @@ def get_user(user_id):
-    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
+    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
"""

_FAKE_DIFF_AUTH = """\
diff --git a/src/middleware.py b/src/middleware.py
--- a/src/middleware.py
+++ b/src/middleware.py
@@ -20,2 +20,5 @@ def check_access(request):
+    if not request.user.is_authenticated:
+        raise PermissionError("Login required")
+    if not request.user.has_perm(resource):
+        raise PermissionError("Access denied")
"""

_FAKE_DIFF_XSS = """\
diff --git a/templates/comment.py b/templates/comment.py
--- a/templates/comment.py
+++ b/templates/comment.py
@@ -5,3 +5,4 @@ def render_comment(text):
-    return f"<p>{text}</p>"
+    from markupsafe import escape
+    return f"<p>{escape(text)}</p>"
"""


def _mock_response(json_data: Any = None, text: str = "", status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.text = text
    resp.raise_for_status.return_value = None
    if status_code >= 400:
        import requests

        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


# ===========================================================================
# _parse_repo
# ===========================================================================


class TestParseRepo:
    """Tests for _parse_repo helper."""

    def test_simple_owner_repo(self):
        from manus_agent.tools.detect_silent_patches import _parse_repo

        assert _parse_repo("owner/repo") == ("owner", "repo")

    def test_full_https_url(self):
        from manus_agent.tools.detect_silent_patches import _parse_repo

        assert _parse_repo("https://github.com/torvalds/linux") == ("torvalds", "linux")

    def test_full_http_url(self):
        from manus_agent.tools.detect_silent_patches import _parse_repo

        assert _parse_repo("http://github.com/torvalds/linux") == ("torvalds", "linux")

    def test_github_com_prefix(self):
        from manus_agent.tools.detect_silent_patches import _parse_repo

        assert _parse_repo("github.com/owner/repo") == ("owner", "repo")

    def test_trailing_slash(self):
        from manus_agent.tools.detect_silent_patches import _parse_repo

        assert _parse_repo("owner/repo/") == ("owner", "repo")

    def test_trailing_paths_stripped(self):
        from manus_agent.tools.detect_silent_patches import _parse_repo

        assert _parse_repo("https://github.com/owner/repo/tree/main") == ("owner", "repo")

    def test_invalid_format_raises(self):
        from manus_agent.tools.detect_silent_patches import _parse_repo

        with pytest.raises(ValueError, match="Invalid repository format"):
            _parse_repo("just-a-name")

    def test_empty_parts_raises(self):
        from manus_agent.tools.detect_silent_patches import _parse_repo

        with pytest.raises(ValueError, match="Invalid repository format"):
            _parse_repo("/repo")

    def test_whitespace_stripped(self):
        from manus_agent.tools.detect_silent_patches import _parse_repo

        assert _parse_repo("  owner/repo  ") == ("owner", "repo")


# ===========================================================================
# _score_message
# ===========================================================================


class TestScoreMessage:
    """Tests for _score_message (stage 1 scoring)."""

    def test_sql_injection_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Fix SQL injection in user lookup")
        assert score > 0
        assert cls == "sql_injection"
        assert any("sql injection" in k for k in kws)

    def test_buffer_overflow_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Fix buffer overflow in parser with bounds check")
        assert score > 0
        assert cls == "buffer_overflow"

    def test_auth_bypass_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Fix authentication bypass in admin endpoint")
        assert score > 0
        assert cls == "auth_bypass"

    def test_xss_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Prevent cross-site scripting via sanitize html")
        assert score > 0
        assert cls == "xss"

    def test_path_traversal_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Fix path traversal in file upload")
        assert score > 0
        assert cls == "path_traversal"

    def test_use_after_free_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Fix use-after-free bug in connection pool")
        assert score > 0
        assert cls == "use_after_free"

    def test_command_injection_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Fix command injection via shell=true in subprocess")
        assert score > 0
        assert cls == "command_injection"

    def test_csrf_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Add CSRF token validation")
        assert score > 0
        assert cls == "csrf"

    def test_ssrf_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Prevent SSRF via URL validation")
        assert score > 0
        assert cls == "ssrf"

    def test_deserialization_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Fix insecure deserialization via pickle")
        assert score > 0
        assert cls == "deserialization"

    def test_dos_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Fix denial of service via regex dos")
        assert score > 0
        assert cls == "denial_of_service"

    def test_info_disclosure_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Fix information disclosure in error logs")
        assert score > 0
        assert cls == "information_disclosure"

    def test_integer_overflow_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Fix integer overflow in size calculation")
        assert score > 0
        assert cls == "integer_overflow"

    def test_null_deref_keywords(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Add null pointer check to prevent dereference")
        assert score > 0
        assert cls == "null_dereference"

    def test_benign_message_zero_score(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Add unit tests for config parser")
        assert score == 0
        assert cls == ""

    def test_generic_security_signal(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, kws = _score_message("Security hardening: sanitize and validate input")
        assert score > 0
        assert cls == "unknown_security"

    def test_multiple_keywords_boost_score(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score_single, _, _ = _score_message("Fix SQL injection")
        score_multi, _, _ = _score_message("Fix SQL injection with parameterized query and sanitize sql")
        assert score_multi > score_single

    def test_security_signal_bonus(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score_base, _, _ = _score_message("Fix buffer overflow")
        score_bonus, _, _ = _score_message("Fix buffer overflow — security vulnerability exploit")
        assert score_bonus > score_base

    def test_score_capped_at_70(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        # Stuff many keywords to try hitting the cap
        msg = (
            "Fix SQL injection sqli parameterized query prepared statement "
            "sanitize sql escape sql security vulnerability exploit attack"
        )
        score, _, _ = _score_message(msg)
        assert score <= 70

    def test_case_insensitive(self):
        from manus_agent.tools.detect_silent_patches import _score_message

        score, cls, _ = _score_message("FIX BUFFER OVERFLOW IN PARSER")
        assert score > 0
        assert cls == "buffer_overflow"


# ===========================================================================
# _score_diff
# ===========================================================================


class TestScoreDiff:
    """Tests for _score_diff (stage 2 scoring)."""

    def test_sql_diff_confirms_class(self):
        from manus_agent.tools.detect_silent_patches import _score_diff

        adj, cls, kws = _score_diff(_FAKE_DIFF_SQL, "sql_injection")
        assert adj > 0
        assert cls == "sql_injection"

    def test_auth_diff_confirms_class(self):
        from manus_agent.tools.detect_silent_patches import _score_diff

        adj, cls, kws = _score_diff(_FAKE_DIFF_AUTH, "auth_bypass")
        assert adj > 0
        assert cls == "auth_bypass"

    def test_xss_diff_confirms_class(self):
        from manus_agent.tools.detect_silent_patches import _score_diff

        adj, cls, kws = _score_diff(_FAKE_DIFF_XSS, "xss")
        assert adj > 0
        assert cls == "xss"

    def test_no_match_zero_adjustment(self):
        from manus_agent.tools.detect_silent_patches import _score_diff

        adj, cls, kws = _score_diff("just a normal diff with nothing interesting", "sql_injection")
        assert adj == 0
        assert cls == "sql_injection"  # initial class preserved

    def test_different_class_in_diff(self):
        from manus_agent.tools.detect_silent_patches import _score_diff

        # Diff looks like auth_bypass but initial class was sql_injection
        adj, cls, kws = _score_diff(_FAKE_DIFF_AUTH, "sql_injection")
        assert adj > 0
        # The class may be refined to auth_bypass since diff evidence is stronger
        assert cls in ("auth_bypass", "sql_injection")

    def test_adjustment_capped(self):
        from manus_agent.tools.detect_silent_patches import _score_diff

        # Maximum adjustment is 30
        adj, _, _ = _score_diff(_FAKE_DIFF_SQL, "sql_injection")
        assert adj <= 30

    def test_empty_diff(self):
        from manus_agent.tools.detect_silent_patches import _score_diff

        adj, cls, kws = _score_diff("", "buffer_overflow")
        assert adj == 0
        assert cls == "buffer_overflow"


# ===========================================================================
# _confidence_label
# ===========================================================================


class TestConfidenceLabel:
    """Tests for _confidence_label."""

    def test_high(self):
        from manus_agent.tools.detect_silent_patches import _confidence_label

        assert _confidence_label(75) == "high"
        assert _confidence_label(100) == "high"

    def test_medium(self):
        from manus_agent.tools.detect_silent_patches import _confidence_label

        assert _confidence_label(50) == "medium"
        assert _confidence_label(74) == "medium"

    def test_low(self):
        from manus_agent.tools.detect_silent_patches import _confidence_label

        assert _confidence_label(30) == "low"
        assert _confidence_label(49) == "low"

    def test_informational(self):
        from manus_agent.tools.detect_silent_patches import _confidence_label

        assert _confidence_label(0) == "informational"
        assert _confidence_label(29) == "informational"


# ===========================================================================
# _fetch_commits
# ===========================================================================


class TestFetchCommits:
    """Tests for _fetch_commits."""

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_single_page(self, mock_get):
        from manus_agent.tools.detect_silent_patches import _fetch_commits

        mock_get.return_value = _mock_response(json_data=[_FAKE_COMMIT_SECURITY, _FAKE_COMMIT_BENIGN])
        result = _fetch_commits("owner", "repo", "2026-08-01", "2026-09-01", 500)
        assert len(result) == 2
        mock_get.assert_called_once()

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_pagination(self, mock_get):
        from manus_agent.tools.detect_silent_patches import _fetch_commits

        # First page returns 100 commits (full page), second returns 50 (partial)
        page1 = [_FAKE_COMMIT_SECURITY] * 100
        page2 = [_FAKE_COMMIT_BENIGN] * 50
        mock_get.side_effect = [
            _mock_response(json_data=page1),
            _mock_response(json_data=page2),
        ]
        result = _fetch_commits("owner", "repo", "2026-08-01", "2026-09-01", 500)
        assert len(result) == 150
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_max_commits_respected(self, mock_get):
        from manus_agent.tools.detect_silent_patches import _fetch_commits

        mock_get.return_value = _mock_response(json_data=[_FAKE_COMMIT_SECURITY] * 100)
        result = _fetch_commits("owner", "repo", "2026-08-01", "2026-09-01", 10)
        assert len(result) == 10

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_empty_response(self, mock_get):
        from manus_agent.tools.detect_silent_patches import _fetch_commits

        mock_get.return_value = _mock_response(json_data=[])
        result = _fetch_commits("owner", "repo", "2026-08-01", "2026-09-01", 500)
        assert result == []

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_api_params(self, mock_get):
        from manus_agent.tools.detect_silent_patches import _fetch_commits

        mock_get.return_value = _mock_response(json_data=[])
        _fetch_commits("owner", "repo", "2026-08-01", "2026-09-01", 500)
        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
        assert params["since"] == "2026-08-01"
        assert params["until"] == "2026-09-01"


# ===========================================================================
# _fetch_diff
# ===========================================================================


class TestFetchDiff:
    """Tests for _fetch_diff."""

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_success(self, mock_get):
        from manus_agent.tools.detect_silent_patches import _fetch_diff

        mock_get.return_value = _mock_response(text=_FAKE_DIFF_SQL, status_code=200)
        result = _fetch_diff("owner", "repo", "abc123")
        assert result == _FAKE_DIFF_SQL

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_404_returns_none(self, mock_get):
        from manus_agent.tools.detect_silent_patches import _fetch_diff

        mock_get.return_value = _mock_response(status_code=404)
        result = _fetch_diff("owner", "repo", "abc123")
        assert result is None

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_network_error_returns_none(self, mock_get):
        import requests

        from manus_agent.tools.detect_silent_patches import _fetch_diff

        mock_get.side_effect = requests.ConnectionError("timeout")
        result = _fetch_diff("owner", "repo", "abc123")
        assert result is None

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_diff_accept_header(self, mock_get):
        from manus_agent.tools.detect_silent_patches import _fetch_diff

        mock_get.return_value = _mock_response(text="diff", status_code=200)
        _fetch_diff("owner", "repo", "abc123")
        call_kwargs = mock_get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers["Accept"] == "application/vnd.github.diff"


# ===========================================================================
# detect_silent_patches (integration)
# ===========================================================================


class TestDetectSilentPatches:
    """Integration tests for the main detect_silent_patches function."""

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_basic_detection(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [
            _FAKE_COMMIT_SECURITY,
            _FAKE_COMMIT_BENIGN,
        ]
        mock_diff.return_value = _FAKE_DIFF_SQL
        result = detect_silent_patches("owner/repo")

        assert result["repo"] == "owner/repo"
        assert result["total_commits_scanned"] == 2
        assert result["candidates_found"] == 1
        assert result["candidates"][0]["classification"] == "sql_injection"

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_cve_commits_excluded(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [
            _FAKE_COMMIT_CVE,
            _FAKE_COMMIT_BENIGN,
        ]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")

        # CVE commit should be excluded (it has a CVE reference)
        assert result["candidates_found"] == 0

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_benign_commits_excluded(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [_FAKE_COMMIT_BENIGN]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")
        assert result["candidates_found"] == 0

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_fast_mode_skips_diff(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [_FAKE_COMMIT_SECURITY]
        result = detect_silent_patches("owner/repo", fast=True)

        # _fetch_diff should NOT be called in fast mode
        mock_diff.assert_not_called()
        assert result["summary"]["fast_mode"] is True
        assert result["candidates_found"] == 1

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_multiple_candidates_sorted_by_score(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [
            _FAKE_COMMIT_SECURITY,
            _FAKE_COMMIT_BUFFER_OVERFLOW,
            _FAKE_COMMIT_XSS,
            _FAKE_COMMIT_AUTH_BYPASS,
            _FAKE_COMMIT_BENIGN,
        ]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")

        candidates = result["candidates"]
        assert len(candidates) == 4  # benign excluded
        # Should be sorted by score descending
        scores = [c["score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_since_until_params(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = []
        result = detect_silent_patches(
            "owner/repo",
            since="2026-01-01",
            until="2026-06-01",
        )
        assert result["scan_window"]["since"] == "2026-01-01"
        assert result["scan_window"]["until"] == "2026-06-01"

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_max_commits_passed_through(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = []
        detect_silent_patches("owner/repo", max_commits=50)
        mock_commits.assert_called_once()
        call_args = mock_commits.call_args
        assert call_args[0][4] == 50  # max_commits arg

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_diff_scoring_applied(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [_FAKE_COMMIT_SECURITY]
        mock_diff.return_value = _FAKE_DIFF_SQL

        result_with_diff = detect_silent_patches("owner/repo", fast=False)

        mock_commits.return_value = [_FAKE_COMMIT_SECURITY]
        result_fast = detect_silent_patches("owner/repo", fast=True)

        # With diff scoring, the score should be higher (or at least equal)
        score_with = result_with_diff["candidates"][0]["score"]
        score_fast = result_fast["candidates"][0]["score"]
        assert score_with >= score_fast

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_summary_by_classification(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [
            _FAKE_COMMIT_SECURITY,
            _FAKE_COMMIT_XSS,
        ]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")

        summary = result["summary"]
        assert "sql_injection" in summary["by_classification"]
        assert "xss" in summary["by_classification"]

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_summary_by_confidence(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [_FAKE_COMMIT_SECURITY]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")

        summary = result["summary"]
        assert any(k in summary["by_confidence"] for k in ("high", "medium", "low", "informational"))

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_candidate_fields(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [_FAKE_COMMIT_SECURITY]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")

        c = result["candidates"][0]
        assert "sha" in c
        assert "short_sha" in c
        assert "message" in c
        assert "full_message" in c
        assert "date" in c
        assert "url" in c
        assert "score" in c
        assert "confidence" in c
        assert "classification" in c
        assert "message_keywords" in c
        assert "diff_keywords" in c
        assert "stage2_applied" in c

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_short_sha_is_8_chars(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [_FAKE_COMMIT_SECURITY]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")

        assert len(result["candidates"][0]["short_sha"]) == 8

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_no_commits_returns_empty(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = []
        result = detect_silent_patches("owner/repo")

        assert result["candidates_found"] == 0
        assert result["candidates"] == []
        assert result["summary"]["highest_score"] == 0

    def test_invalid_repo_raises(self):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        with pytest.raises(ValueError, match="Invalid repository format"):
            detect_silent_patches("not-a-repo")

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_all_14_bug_classes_detected(self, mock_commits, mock_diff):
        """Each bug class commit should be correctly classified."""
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [
            _FAKE_COMMIT_SECURITY,  # sql_injection
            _FAKE_COMMIT_BUFFER_OVERFLOW,  # buffer_overflow
            _FAKE_COMMIT_XSS,  # xss
            _FAKE_COMMIT_AUTH_BYPASS,  # auth_bypass
            _FAKE_COMMIT_SSRF,  # ssrf
            _FAKE_COMMIT_DESERIALIZATION,  # deserialization
            _FAKE_COMMIT_DOS,  # denial_of_service
            _FAKE_COMMIT_INFO_LEAK,  # information_disclosure
            _FAKE_COMMIT_PATH_TRAVERSAL,  # path_traversal
            _FAKE_COMMIT_UAF,  # use_after_free
            _FAKE_COMMIT_CSRF,  # csrf
            _FAKE_COMMIT_CMD_INJECTION,  # command_injection
            _FAKE_COMMIT_INT_OVERFLOW,  # integer_overflow
            _FAKE_COMMIT_NULL_DEREF,  # null_dereference
        ]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")

        found_classes = {c["classification"] for c in result["candidates"]}
        expected = {
            "sql_injection",
            "buffer_overflow",
            "xss",
            "auth_bypass",
            "ssrf",
            "deserialization",
            "denial_of_service",
            "information_disclosure",
            "path_traversal",
            "use_after_free",
            "csrf",
            "command_injection",
            "integer_overflow",
            "null_dereference",
        }
        assert found_classes == expected

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_score_capped_at_100(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [_FAKE_COMMIT_SECURITY]
        # Return a diff that would maximally boost the score
        mock_diff.return_value = "cursor.execute parameterize bind_param %s ? select insert update delete sanitize sql"
        result = detect_silent_patches("owner/repo")
        assert result["candidates"][0]["score"] <= 100

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_full_url_repo_input(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = []
        result = detect_silent_patches("https://github.com/torvalds/linux")
        assert result["repo"] == "torvalds/linux"

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_generic_security_commit(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [_FAKE_COMMIT_GENERIC_SECURITY]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")

        assert result["candidates_found"] >= 1
        c = result["candidates"][0]
        assert c["classification"] == "unknown_security"

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_diff_failure_graceful(self, mock_commits, mock_diff):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [_FAKE_COMMIT_SECURITY]
        mock_diff.return_value = None  # diff fetch failed
        result = detect_silent_patches("owner/repo", fast=False)

        # Should still produce a candidate with message-only scoring
        assert result["candidates_found"] == 1
        assert result["candidates"][0]["stage2_applied"] is False


# ===========================================================================
# detect_silent_patches_tool (Strands interface)
# ===========================================================================


class TestDetectSilentPatchesTool:
    """Tests for the Strands-compatible tool handler."""

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_success(self, mock_detect):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches_tool

        mock_detect.return_value = {
            "repo": "owner/repo",
            "candidates_found": 1,
            "candidates": [{"sha": "abc"}],
            "summary": {},
        }
        tool_use = {
            "toolUseId": "test-123",
            "input": {"repo": "owner/repo"},
        }
        result = detect_silent_patches_tool(tool_use)
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_missing_repo(self, mock_detect):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches_tool

        tool_use = {
            "toolUseId": "test-456",
            "input": {},
        }
        result = detect_silent_patches_tool(tool_use)
        assert result["status"] == "error"
        mock_detect.assert_not_called()

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_value_error(self, mock_detect):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches_tool

        mock_detect.side_effect = ValueError("Invalid repo")
        tool_use = {
            "toolUseId": "test-789",
            "input": {"repo": "bad"},
        }
        result = detect_silent_patches_tool(tool_use)
        assert result["status"] == "error"
        assert "Invalid repo" in result["content"][0]["text"]

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_http_error(self, mock_detect):
        import requests

        from manus_agent.tools.detect_silent_patches import detect_silent_patches_tool

        resp = MagicMock()
        resp.status_code = 404
        mock_detect.side_effect = requests.HTTPError(response=resp)
        tool_use = {
            "toolUseId": "test-http",
            "input": {"repo": "owner/repo"},
        }
        result = detect_silent_patches_tool(tool_use)
        assert result["status"] == "error"
        assert "GitHub API error" in result["content"][0]["text"]

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_network_error(self, mock_detect):
        import requests

        from manus_agent.tools.detect_silent_patches import detect_silent_patches_tool

        mock_detect.side_effect = requests.ConnectionError("timeout")
        tool_use = {
            "toolUseId": "test-net",
            "input": {"repo": "owner/repo"},
        }
        result = detect_silent_patches_tool(tool_use)
        assert result["status"] == "error"
        assert "Network error" in result["content"][0]["text"]

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_optional_params_passed(self, mock_detect):
        from manus_agent.tools.detect_silent_patches import detect_silent_patches_tool

        mock_detect.return_value = {"candidates": [], "summary": {}}
        tool_use = {
            "toolUseId": "test-params",
            "input": {
                "repo": "owner/repo",
                "since": "2026-01-01",
                "until": "2026-06-01",
                "max_commits": 100,
                "fast": True,
            },
        }
        detect_silent_patches_tool(tool_use)
        mock_detect.assert_called_once_with(
            repo="owner/repo",
            since="2026-01-01",
            until="2026-06-01",
            max_commits=100,
            fast=True,
        )

    def test_tool_spec_schema(self):
        from manus_agent.tools.detect_silent_patches import TOOL_SPEC

        assert TOOL_SPEC["name"] == "detect_silent_patches"
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "repo" in schema["properties"]
        assert "repo" in schema["required"]
        assert "since" in schema["properties"]
        assert "until" in schema["properties"]
        assert "max_commits" in schema["properties"]
        assert "fast" in schema["properties"]


# ===========================================================================
# _gh_headers
# ===========================================================================


class TestGhHeaders:
    """Tests for _gh_headers helper."""

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"}, clear=False)
    def test_with_github_token(self):
        from manus_agent.tools.detect_silent_patches import _gh_headers

        headers = _gh_headers()
        assert headers["Authorization"] == "Bearer ghp_test123"
        assert "Accept" in headers

    @patch.dict("os.environ", {"GH_TOKEN": "ghp_alt456"}, clear=False)
    def test_with_gh_token(self):
        # Clear GITHUB_TOKEN to test GH_TOKEN fallback
        import os

        from manus_agent.tools.detect_silent_patches import _gh_headers

        old = os.environ.pop("GITHUB_TOKEN", None)
        try:
            headers = _gh_headers()
            assert headers["Authorization"] == "Bearer ghp_alt456"
        finally:
            if old is not None:
                os.environ["GITHUB_TOKEN"] = old

    @patch.dict("os.environ", {}, clear=True)
    def test_no_token(self):
        from manus_agent.tools.detect_silent_patches import _gh_headers

        headers = _gh_headers()
        assert "Authorization" not in headers
        assert headers["Accept"] == "application/vnd.github+json"


# ===========================================================================
# CLI: _build_silent_patches_parser
# ===========================================================================


class TestSilentPatchesParser:
    """Tests for CLI parser."""

    def test_defaults(self):
        from manus_agent.cli import _build_silent_patches_parser

        parser = _build_silent_patches_parser()
        args = parser.parse_args(["torvalds/linux"])
        assert args.repo == "torvalds/linux"
        assert args.since is None
        assert args.until is None
        assert args.max_commits == 500
        assert args.fast is False
        assert args.output == "text"

    def test_all_flags(self):
        from manus_agent.cli import _build_silent_patches_parser

        parser = _build_silent_patches_parser()
        args = parser.parse_args(
            [
                "owner/repo",
                "--since",
                "2026-01-01",
                "--until",
                "2026-06-01",
                "--max-commits",
                "100",
                "--fast",
                "--output",
                "json",
            ]
        )
        assert args.repo == "owner/repo"
        assert args.since == "2026-01-01"
        assert args.until == "2026-06-01"
        assert args.max_commits == 100
        assert args.fast is True
        assert args.output == "json"


# ===========================================================================
# CLI: _run_silent_patches
# ===========================================================================


class TestRunSilentPatches:
    """Tests for _run_silent_patches CLI runner."""

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_json_output(self, mock_detect, capsys):
        from manus_agent.cli import _run_silent_patches

        mock_detect.return_value = {
            "repo": "owner/repo",
            "scan_window": {"since": "2026-06-18T00:00:00Z", "until": "2026-09-16T00:00:00Z"},
            "total_commits_scanned": 42,
            "candidates_found": 1,
            "candidates": [
                {
                    "sha": "abc12345deadbeef",
                    "short_sha": "abc12345",
                    "message": "Fix SQL injection in user lookup",
                    "full_message": "Fix SQL injection in user lookup",
                    "date": "2026-08-15T10:00:00Z",
                    "url": "https://github.com/owner/repo/commit/abc12345",
                    "score": 45,
                    "confidence": "low",
                    "classification": "sql_injection",
                    "message_keywords": ["sql injection"],
                    "diff_keywords": [],
                    "stage2_applied": False,
                }
            ],
            "summary": {
                "by_classification": {"sql_injection": 1},
                "by_confidence": {"low": 1},
                "highest_score": 45,
                "fast_mode": False,
            },
        }
        exit_code = _run_silent_patches(["owner/repo", "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["candidates_found"] == 1

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_text_output(self, mock_detect, capsys):
        from manus_agent.cli import _run_silent_patches

        mock_detect.return_value = {
            "repo": "owner/repo",
            "scan_window": {"since": "2026-06-18T00:00:00Z", "until": "2026-09-16T00:00:00Z"},
            "total_commits_scanned": 10,
            "candidates_found": 1,
            "candidates": [
                {
                    "sha": "abc12345deadbeef",
                    "short_sha": "abc12345",
                    "message": "Fix auth bypass in admin panel",
                    "full_message": "Fix auth bypass in admin panel",
                    "date": "2026-08-20T11:00:00Z",
                    "url": "https://github.com/owner/repo/commit/abc12345",
                    "score": 55,
                    "confidence": "medium",
                    "classification": "auth_bypass",
                    "message_keywords": ["auth bypass"],
                    "diff_keywords": [],
                    "stage2_applied": False,
                }
            ],
            "summary": {
                "by_classification": {"auth_bypass": 1},
                "by_confidence": {"medium": 1},
                "highest_score": 55,
                "fast_mode": False,
            },
        }
        exit_code = _run_silent_patches(["owner/repo"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Silent Patch Detector" in captured.out
        assert "auth_bypass" in captured.out

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_no_candidates_text(self, mock_detect, capsys):
        from manus_agent.cli import _run_silent_patches

        mock_detect.return_value = {
            "repo": "owner/repo",
            "scan_window": {"since": "2026-06-18T00:00:00Z", "until": "2026-09-16T00:00:00Z"},
            "total_commits_scanned": 10,
            "candidates_found": 0,
            "candidates": [],
            "summary": {
                "by_classification": {},
                "by_confidence": {},
                "highest_score": 0,
                "fast_mode": False,
            },
        }
        exit_code = _run_silent_patches(["owner/repo"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "No silent patch candidates detected" in captured.out

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_value_error_exit_1(self, mock_detect, capsys):
        from manus_agent.cli import _run_silent_patches

        mock_detect.side_effect = ValueError("Invalid repo")
        exit_code = _run_silent_patches(["bad-repo"])
        assert exit_code == 1

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_fast_mode_label(self, mock_detect, capsys):
        from manus_agent.cli import _run_silent_patches

        mock_detect.return_value = {
            "repo": "owner/repo",
            "scan_window": {"since": "2026-06-18T00:00:00Z", "until": "2026-09-16T00:00:00Z"},
            "total_commits_scanned": 5,
            "candidates_found": 0,
            "candidates": [],
            "summary": {
                "by_classification": {},
                "by_confidence": {},
                "highest_score": 0,
                "fast_mode": True,
            },
        }
        exit_code = _run_silent_patches(["owner/repo", "--fast"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "fast" in captured.out.lower()

    @patch("manus_agent.tools.detect_silent_patches.detect_silent_patches")
    def test_stage2_applied_shown(self, mock_detect, capsys):
        from manus_agent.cli import _run_silent_patches

        mock_detect.return_value = {
            "repo": "owner/repo",
            "scan_window": {"since": "2026-06-18T00:00:00Z", "until": "2026-09-16T00:00:00Z"},
            "total_commits_scanned": 5,
            "candidates_found": 1,
            "candidates": [
                {
                    "sha": "abc12345deadbeef",
                    "short_sha": "abc12345",
                    "message": "Fix SQL injection",
                    "full_message": "Fix SQL injection",
                    "date": "2026-08-15T10:00:00Z",
                    "url": "https://github.com/owner/repo/commit/abc12345",
                    "score": 60,
                    "confidence": "medium",
                    "classification": "sql_injection",
                    "message_keywords": ["sql injection"],
                    "diff_keywords": ["cursor."],
                    "stage2_applied": True,
                }
            ],
            "summary": {
                "by_classification": {"sql_injection": 1},
                "by_confidence": {"medium": 1},
                "highest_score": 60,
                "fast_mode": False,
            },
        }
        exit_code = _run_silent_patches(["owner/repo"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "diff scoring applied" in captured.out


# ===========================================================================
# CLI dispatch
# ===========================================================================


class TestCLIDispatch:
    """Tests for CLI dispatch routing to silent-patches."""

    def test_silent_patches_in_known_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "silent-patches" in _SUBCOMMANDS

    @patch("manus_agent.cli._run_silent_patches")
    def test_dispatch_calls_runner(self, mock_run, monkeypatch):
        from manus_agent.cli import main

        mock_run.return_value = 0
        monkeypatch.setattr(sys, "argv", ["manus-agent", "silent-patches", "owner/repo"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_run.assert_called_once_with(["owner/repo"])


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge-case and regression tests."""

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_commit_missing_fields(self, mock_commits, mock_diff):
        """Commit with minimal/missing fields should not crash."""
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [
            {
                "sha": "0000000000000000",
                "commit": {"message": "", "author": {}},
                "html_url": "",
            }
        ]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")
        assert result["candidates_found"] == 0

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_long_message_truncated(self, mock_commits, mock_diff):
        """Commit message field should be truncated to 200 chars."""
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        long_msg = "Fix SQL injection " + "x" * 500
        mock_commits.return_value = [
            {
                "sha": "abc12345deadbeef",
                "commit": {
                    "message": long_msg,
                    "author": {"date": "2026-08-15T10:00:00Z"},
                },
                "html_url": "https://github.com/owner/repo/commit/abc",
            }
        ]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")
        assert len(result["candidates"][0]["message"]) <= 200

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_multiline_message_first_line(self, mock_commits, mock_diff):
        """Only first line of commit message should be in 'message' field."""
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        msg = "Fix buffer overflow in parser\n\nThis is the body\nWith more details"
        mock_commits.return_value = [
            {
                "sha": "def67890cafebabe",
                "commit": {
                    "message": msg,
                    "author": {"date": "2026-08-16T12:00:00Z"},
                },
                "html_url": "https://github.com/owner/repo/commit/def",
            }
        ]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")
        assert "\n" not in result["candidates"][0]["message"]

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_mixed_cve_and_silent(self, mock_commits, mock_diff):
        """Commits with CVE refs should be excluded, silent ones kept."""
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = [
            _FAKE_COMMIT_CVE,  # has CVE ref → excluded
            _FAKE_COMMIT_SECURITY,  # no CVE ref → candidate
        ]
        mock_diff.return_value = None
        result = detect_silent_patches("owner/repo")
        assert result["candidates_found"] == 1
        assert result["candidates"][0]["sha"] == _FAKE_COMMIT_SECURITY["sha"]

    @patch("manus_agent.tools.detect_silent_patches._fetch_diff")
    @patch("manus_agent.tools.detect_silent_patches._fetch_commits")
    def test_default_since_is_90_days(self, mock_commits, mock_diff):
        """Default since should be ~90 days ago."""
        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        mock_commits.return_value = []
        result = detect_silent_patches("owner/repo")
        # The since field should be set (not None)
        assert result["scan_window"]["since"] is not None
        assert "T" in result["scan_window"]["since"]

    @patch("manus_agent.tools.detect_silent_patches.requests.get")
    def test_http_error_propagates(self, mock_get):
        """HTTP errors from GitHub should propagate."""
        import requests

        from manus_agent.tools.detect_silent_patches import detect_silent_patches

        resp = MagicMock()
        resp.status_code = 403
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
        mock_get.return_value = resp

        with pytest.raises(requests.HTTPError):
            detect_silent_patches("owner/repo")
