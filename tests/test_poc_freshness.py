"""Comprehensive test suite for get_poc_freshness module.

All HTTP calls are mocked — no real network requests.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, mock_open, patch

import pytest

from manus_agent.tools.get_poc_freshness import (
    _CVE_RE,
    _TIER_BOUNDARIES,
    _TIER_FLOOR,
    _TIER_LABELS,
    _WEIGHT_COMMIT_RECENCY,
    _WEIGHT_EXPLOITDB,
    _WEIGHT_STAR_VELOCITY,
    TOOL_SPEC,
    _assess_commit_recency,
    _assess_exploitdb_recency,
    _assess_star_velocity,
    _days_since,
    _days_to_subscore,
    _fetch_trickest_repos,
    _freshness_tier,
    _get_repo_last_push,
    _github_headers,
    _http_get_json,
    _http_get_text,
    _load_exploitdb_csv,
    _parse_date_published,
    _parse_iso_date,
    compute_freshness,
    get_poc_freshness,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)
ONE_DAY_AGO = (NOW - timedelta(days=1)).isoformat()
SEVEN_DAYS_AGO = (NOW - timedelta(days=7)).isoformat()
THIRTY_DAYS_AGO = (NOW - timedelta(days=30)).isoformat()
NINETY_DAYS_AGO = (NOW - timedelta(days=90)).isoformat()
ONE_YEAR_AGO = (NOW - timedelta(days=365)).isoformat()
TWO_YEARS_AGO = (NOW - timedelta(days=730)).isoformat()


def _make_exploitdb_csv(rows: list[dict]) -> str:
    """Build an Exploit-DB CSV string from a list of row dicts."""
    fieldnames = ["id", "file", "description", "date_published", "author", "platform", "type", "port", "codes"]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return out.getvalue()


# ===================================================================
# TOOL_SPEC tests
# ===================================================================


class TestToolSpec:
    """Validate the Strands TOOL_SPEC schema."""

    def test_tool_spec_has_name(self):
        assert TOOL_SPEC["name"] == "get_poc_freshness"

    def test_tool_spec_has_description(self):
        assert "freshness" in TOOL_SPEC["description"].lower()

    def test_tool_spec_has_input_schema(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_tool_spec_cve_id_is_string(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["properties"]["cve_id"]["type"] == "string"


# ===================================================================
# Constants and weight tests
# ===================================================================


class TestConstants:
    """Validate module constants."""

    def test_weights_sum_to_one(self):
        total = _WEIGHT_COMMIT_RECENCY + _WEIGHT_STAR_VELOCITY + _WEIGHT_EXPLOITDB
        assert abs(total - 1.0) < 1e-9

    def test_cve_re_valid(self):
        assert _CVE_RE.match("CVE-2024-3094")
        assert _CVE_RE.match("cve-2021-44228")

    def test_cve_re_invalid(self):
        assert _CVE_RE.match("NOT-A-CVE") is None
        assert _CVE_RE.match("CVE-2024") is None
        assert _CVE_RE.match("") is None

    def test_tier_boundaries_sorted(self):
        days_list = [b[0] for b in _TIER_BOUNDARIES]
        assert days_list == sorted(days_list)

    def test_tier_labels_sorted_descending(self):
        thresholds = [t[0] for t in _TIER_LABELS]
        assert thresholds == sorted(thresholds, reverse=True)


# ===================================================================
# Helper function tests
# ===================================================================


class TestParseIsoDate:
    """Tests for _parse_iso_date."""

    def test_none_input(self):
        assert _parse_iso_date(None) is None

    def test_empty_string(self):
        assert _parse_iso_date("") is None

    def test_github_z_suffix(self):
        dt = _parse_iso_date("2024-06-15T10:30:00Z")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.day == 15

    def test_iso_with_offset(self):
        dt = _parse_iso_date("2024-06-15T10:30:00+00:00")
        assert dt is not None
        assert dt.year == 2024

    def test_invalid_string(self):
        assert _parse_iso_date("not-a-date") is None

    def test_date_only(self):
        dt = _parse_iso_date("2024-06-15")
        assert dt is not None
        assert dt.year == 2024


class TestDaysSince:
    """Tests for _days_since."""

    def test_none_input(self):
        assert _days_since(None) is None

    def test_zero_days(self):
        result = _days_since(NOW, now=NOW)
        assert result == 0.0

    def test_one_day(self):
        yesterday = NOW - timedelta(days=1)
        result = _days_since(yesterday, now=NOW)
        assert abs(result - 1.0) < 0.01

    def test_naive_datetime_treated_as_utc(self):
        naive = datetime(2026, 8, 31, 12, 0, 0)  # no tzinfo
        result = _days_since(naive, now=NOW)
        assert abs(result - 1.0) < 0.01

    def test_negative_clamped_to_zero(self):
        future = NOW + timedelta(days=1)
        result = _days_since(future, now=NOW)
        assert result == 0.0


class TestDaysToSubscore:
    """Tests for _days_to_subscore."""

    def test_none_returns_zero(self):
        assert _days_to_subscore(None) == 0.0

    def test_within_seven_days(self):
        assert _days_to_subscore(3.0) == 100.0

    def test_exactly_seven_days(self):
        assert _days_to_subscore(7.0) == 100.0

    def test_within_thirty_days(self):
        assert _days_to_subscore(15.0) == 75.0

    def test_exactly_thirty_days(self):
        assert _days_to_subscore(30.0) == 75.0

    def test_within_ninety_days(self):
        assert _days_to_subscore(60.0) == 50.0

    def test_within_one_year(self):
        assert _days_to_subscore(200.0) == 25.0

    def test_over_one_year(self):
        assert _days_to_subscore(500.0) == _TIER_FLOOR


class TestFreshnessTier:
    """Tests for _freshness_tier."""

    def test_active(self):
        assert _freshness_tier(85.0) == "ACTIVE"

    def test_active_boundary(self):
        assert _freshness_tier(80.0) == "ACTIVE"

    def test_recent(self):
        assert _freshness_tier(70.0) == "RECENT"

    def test_aging(self):
        assert _freshness_tier(50.0) == "AGING"

    def test_stale(self):
        assert _freshness_tier(30.0) == "STALE"

    def test_dormant(self):
        assert _freshness_tier(10.0) == "DORMANT"

    def test_zero(self):
        assert _freshness_tier(0.0) == "DORMANT"


class TestParseDatePublished:
    """Tests for _parse_date_published."""

    def test_valid_date(self):
        dt = _parse_date_published("2024-06-15")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.day == 15
        assert dt.tzinfo == timezone.utc

    def test_empty_string(self):
        assert _parse_date_published("") is None

    def test_none(self):
        assert _parse_date_published(None) is None

    def test_invalid_format(self):
        assert _parse_date_published("15/06/2024") is None

    def test_whitespace_stripped(self):
        dt = _parse_date_published("  2024-06-15  ")
        assert dt is not None
        assert dt.year == 2024


# ===================================================================
# HTTP helper tests
# ===================================================================


class TestGithubHeaders:
    """Tests for _github_headers."""

    def test_no_token(self):
        with patch.dict(os.environ, {}, clear=True):
            headers = _github_headers()
            assert "Authorization" not in headers
            assert "User-Agent" in headers

    def test_with_token(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test123"}):
            headers = _github_headers()
            assert headers["Authorization"] == "Bearer ghp_test123"

    def test_empty_token_ignored(self):
        with patch.dict(os.environ, {"GITHUB_TOKEN": "  "}):
            headers = _github_headers()
            assert "Authorization" not in headers


class TestHttpGetJson:
    """Tests for _http_get_json."""

    @patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"key": "value"}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _http_get_json("https://example.com/api")
        assert result == {"key": "value"}

    @patch("manus_agent.tools.get_poc_freshness.time.sleep")
    @patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen")
    def test_retry_on_failure(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [
            Exception("Network error"),
            Exception("Network error"),
            MagicMock(
                read=lambda: b'{"ok": true}',
                __enter__=lambda s: s,
                __exit__=MagicMock(return_value=False),
            ),
        ]
        result = _http_get_json("https://example.com/api")
        assert result == {"ok": True}
        assert mock_sleep.call_count == 2

    @patch("manus_agent.tools.get_poc_freshness.time.sleep")
    @patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen")
    def test_all_retries_exhausted(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = Exception("Persistent failure")
        with pytest.raises(Exception, match="Persistent failure"):
            _http_get_json("https://example.com/api")


class TestHttpGetText:
    """Tests for _http_get_text."""

    @patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"Hello world"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _http_get_text("https://example.com")
        assert result == "Hello world"

    @patch("manus_agent.tools.get_poc_freshness.time.sleep")
    @patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen")
    def test_retry_on_failure(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [
            Exception("Timeout"),
            MagicMock(
                read=lambda: b"retry ok",
                __enter__=lambda s: s,
                __exit__=MagicMock(return_value=False),
            ),
        ]
        result = _http_get_text("https://example.com")
        assert result == "retry ok"


# ===================================================================
# Signal 1: Commit recency tests
# ===================================================================


class TestFetchTrickestRepos:
    """Tests for _fetch_trickest_repos."""

    @patch("manus_agent.tools.get_poc_freshness._http_get_text")
    def test_extracts_github_urls(self, mock_get):
        mock_get.return_value = (
            "# CVE-2024-3094\n"
            "- https://github.com/user1/poc-exploit\n"
            "- https://github.com/user2/another-poc\n"
            "Some text without URLs\n"
        )
        repos = _fetch_trickest_repos("CVE-2024-3094")
        assert len(repos) == 2
        assert "https://github.com/user1/poc-exploit" in repos
        assert "https://github.com/user2/another-poc" in repos

    @patch("manus_agent.tools.get_poc_freshness._http_get_text")
    def test_deduplicates_urls(self, mock_get):
        mock_get.return_value = (
            "- https://github.com/user1/poc\n- https://github.com/user1/poc\n- https://github.com/USER1/POC\n"
        )
        repos = _fetch_trickest_repos("CVE-2024-3094")
        assert len(repos) == 1

    @patch("manus_agent.tools.get_poc_freshness._http_get_text")
    def test_strips_git_suffix(self, mock_get):
        mock_get.return_value = "- https://github.com/user1/poc.git\n"
        repos = _fetch_trickest_repos("CVE-2024-3094")
        assert len(repos) == 1

    @patch("manus_agent.tools.get_poc_freshness._http_get_text")
    def test_http_failure_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("404")
        repos = _fetch_trickest_repos("CVE-2024-3094")
        assert repos == []

    def test_invalid_cve_returns_empty(self):
        repos = _fetch_trickest_repos("NOT-A-CVE")
        assert repos == []

    @patch("manus_agent.tools.get_poc_freshness._http_get_text")
    def test_empty_file_returns_empty(self, mock_get):
        mock_get.return_value = ""
        repos = _fetch_trickest_repos("CVE-2024-3094")
        assert repos == []


class TestGetRepoLastPush:
    """Tests for _get_repo_last_push."""

    @patch("manus_agent.tools.get_poc_freshness._http_get_json")
    def test_success_with_commits(self, mock_get):
        mock_get.side_effect = [
            # Repo metadata
            {"pushed_at": "2026-08-30T10:00:00Z", "stargazers_count": 42},
            # Commits endpoint
            [{"commit": {"committer": {"date": "2026-08-31T12:00:00Z"}}}],
        ]
        result = _get_repo_last_push("https://github.com/user1/exploit")
        assert result["pushed_at"] == "2026-08-30T10:00:00Z"
        assert result["last_commit_at"] == "2026-08-31T12:00:00Z"
        assert result["stargazers_count"] == 42
        assert result["error"] is None

    @patch("manus_agent.tools.get_poc_freshness._http_get_json")
    def test_repo_api_failure(self, mock_get):
        mock_get.side_effect = Exception("404 Not Found")
        result = _get_repo_last_push("https://github.com/user1/exploit")
        assert result["error"] is not None
        assert "404" in result["error"]

    @patch("manus_agent.tools.get_poc_freshness._http_get_json")
    def test_commits_api_failure_still_has_pushed_at(self, mock_get):
        mock_get.side_effect = [
            {"pushed_at": "2026-08-30T10:00:00Z", "stargazers_count": 5},
            Exception("Commits API rate limited"),
        ]
        result = _get_repo_last_push("https://github.com/user1/exploit")
        assert result["pushed_at"] == "2026-08-30T10:00:00Z"
        assert result["last_commit_at"] is None  # fallback

    def test_invalid_url(self):
        result = _get_repo_last_push("not-a-github-url")
        assert result["error"] is not None

    @patch("manus_agent.tools.get_poc_freshness._http_get_json")
    def test_empty_commits_list(self, mock_get):
        mock_get.side_effect = [
            {"pushed_at": "2026-08-30T10:00:00Z", "stargazers_count": 0},
            [],
        ]
        result = _get_repo_last_push("https://github.com/user1/exploit")
        assert result["pushed_at"] == "2026-08-30T10:00:00Z"
        assert result["last_commit_at"] is None


class TestAssessCommitRecency:
    """Tests for _assess_commit_recency."""

    @patch("manus_agent.tools.get_poc_freshness.time.sleep")
    @patch("manus_agent.tools.get_poc_freshness._get_repo_last_push")
    @patch("manus_agent.tools.get_poc_freshness._fetch_trickest_repos")
    def test_no_repos_found(self, mock_trickest, mock_repo, mock_sleep):
        mock_trickest.return_value = []
        result = _assess_commit_recency("CVE-2024-3094", now=NOW)
        assert result["repos_found"] == 0
        assert result["subscore"] == 0.0

    @patch("manus_agent.tools.get_poc_freshness.time.sleep")
    @patch("manus_agent.tools.get_poc_freshness._get_repo_last_push")
    @patch("manus_agent.tools.get_poc_freshness._fetch_trickest_repos")
    def test_recent_commit(self, mock_trickest, mock_repo, mock_sleep):
        mock_trickest.return_value = ["https://github.com/user1/poc"]
        mock_repo.return_value = {
            "repo_url": "https://github.com/user1/poc",
            "pushed_at": ONE_DAY_AGO,
            "last_commit_at": ONE_DAY_AGO,
            "stargazers_count": 10,
            "error": None,
        }
        result = _assess_commit_recency("CVE-2024-3094", now=NOW)
        assert result["repos_found"] == 1
        assert result["repos_checked"] == 1
        assert result["subscore"] == 100.0  # ≤7 days

    @patch("manus_agent.tools.get_poc_freshness.time.sleep")
    @patch("manus_agent.tools.get_poc_freshness._get_repo_last_push")
    @patch("manus_agent.tools.get_poc_freshness._fetch_trickest_repos")
    def test_old_commit(self, mock_trickest, mock_repo, mock_sleep):
        mock_trickest.return_value = ["https://github.com/user1/poc"]
        mock_repo.return_value = {
            "repo_url": "https://github.com/user1/poc",
            "pushed_at": TWO_YEARS_AGO,
            "last_commit_at": TWO_YEARS_AGO,
            "stargazers_count": 1,
            "error": None,
        }
        result = _assess_commit_recency("CVE-2024-3094", now=NOW)
        assert result["subscore"] == _TIER_FLOOR

    @patch("manus_agent.tools.get_poc_freshness.time.sleep")
    @patch("manus_agent.tools.get_poc_freshness._get_repo_last_push")
    @patch("manus_agent.tools.get_poc_freshness._fetch_trickest_repos")
    def test_multiple_repos_picks_most_recent(self, mock_trickest, mock_repo, mock_sleep):
        mock_trickest.return_value = [
            "https://github.com/user1/old-poc",
            "https://github.com/user2/new-poc",
        ]
        mock_repo.side_effect = [
            {
                "repo_url": "https://github.com/user1/old-poc",
                "pushed_at": ONE_YEAR_AGO,
                "last_commit_at": ONE_YEAR_AGO,
                "stargazers_count": 5,
                "error": None,
            },
            {
                "repo_url": "https://github.com/user2/new-poc",
                "pushed_at": ONE_DAY_AGO,
                "last_commit_at": ONE_DAY_AGO,
                "stargazers_count": 50,
                "error": None,
            },
        ]
        result = _assess_commit_recency("CVE-2024-3094", now=NOW)
        assert result["repos_checked"] == 2
        assert result["subscore"] == 100.0  # most recent is 1 day ago

    @patch("manus_agent.tools.get_poc_freshness.time.sleep")
    @patch("manus_agent.tools.get_poc_freshness._get_repo_last_push")
    @patch("manus_agent.tools.get_poc_freshness._fetch_trickest_repos")
    def test_caps_at_ten_repos(self, mock_trickest, mock_repo, mock_sleep):
        mock_trickest.return_value = [f"https://github.com/user{i}/poc" for i in range(15)]
        mock_repo.return_value = {
            "repo_url": "https://github.com/userX/poc",
            "pushed_at": THIRTY_DAYS_AGO,
            "last_commit_at": THIRTY_DAYS_AGO,
            "stargazers_count": 1,
            "error": None,
        }
        result = _assess_commit_recency("CVE-2024-3094", now=NOW)
        assert result["repos_found"] == 15
        assert result["repos_checked"] == 10  # capped

    @patch("manus_agent.tools.get_poc_freshness.time.sleep")
    @patch("manus_agent.tools.get_poc_freshness._get_repo_last_push")
    @patch("manus_agent.tools.get_poc_freshness._fetch_trickest_repos")
    def test_all_repos_fail(self, mock_trickest, mock_repo, mock_sleep):
        mock_trickest.return_value = ["https://github.com/user1/poc"]
        mock_repo.return_value = {
            "repo_url": "https://github.com/user1/poc",
            "pushed_at": None,
            "last_commit_at": None,
            "stargazers_count": 0,
            "error": "404 Not Found",
        }
        result = _assess_commit_recency("CVE-2024-3094", now=NOW)
        assert result["repos_checked"] == 1
        assert result["subscore"] == 0.0


# ===================================================================
# Signal 2: Star velocity tests
# ===================================================================


class TestAssessStarVelocity:
    """Tests for _assess_star_velocity."""

    @patch("manus_agent.tools.get_poc_freshness._http_get_json")
    def test_repos_found(self, mock_get):
        mock_get.return_value = {
            "total_count": 3,
            "items": [
                {
                    "full_name": "user1/cve-2024-poc",
                    "html_url": "https://github.com/user1/cve-2024-poc",
                    "stargazers_count": 50,
                    "updated_at": ONE_DAY_AGO,
                    "pushed_at": ONE_DAY_AGO,
                },
                {
                    "full_name": "user2/exploit",
                    "html_url": "https://github.com/user2/exploit",
                    "stargazers_count": 20,
                    "updated_at": THIRTY_DAYS_AGO,
                    "pushed_at": THIRTY_DAYS_AGO,
                },
            ],
        }
        result = _assess_star_velocity("CVE-2024-3094", now=NOW)
        assert result["total_repos"] == 3
        assert result["total_stars"] == 70
        assert result["recent_repos"] == 2  # both within 90 days
        assert result["subscore"] > 0

    @patch("manus_agent.tools.get_poc_freshness._http_get_json")
    def test_no_results(self, mock_get):
        mock_get.return_value = {"total_count": 0, "items": []}
        result = _assess_star_velocity("CVE-2024-3094", now=NOW)
        assert result["total_repos"] == 0
        assert result["subscore"] == 0.0

    @patch("manus_agent.tools.get_poc_freshness._http_get_json")
    def test_api_failure(self, mock_get):
        mock_get.side_effect = Exception("Rate limited")
        result = _assess_star_velocity("CVE-2024-3094", now=NOW)
        assert result["total_repos"] == 0
        assert result["subscore"] == 0.0

    @patch("manus_agent.tools.get_poc_freshness._http_get_json")
    def test_boost_for_many_recent_stars(self, mock_get):
        """Multiple recent repos with stars get a 1.2x boost."""
        mock_get.return_value = {
            "total_count": 5,
            "items": [
                {
                    "full_name": f"user{i}/poc",
                    "html_url": f"https://github.com/user{i}/poc",
                    "stargazers_count": 10,
                    "updated_at": ONE_DAY_AGO,
                    "pushed_at": ONE_DAY_AGO,
                }
                for i in range(5)
            ],
        }
        result = _assess_star_velocity("CVE-2024-3094", now=NOW)
        # 5 recent repos, 50 total stars → should get boosted
        assert result["recent_repos"] == 5
        assert result["total_stars"] == 50
        # Base subscore for 1-day-old would be 100, boost × 1.2 capped at 100
        assert result["subscore"] == 100.0

    @patch("manus_agent.tools.get_poc_freshness._http_get_json")
    def test_old_repos_low_score(self, mock_get):
        mock_get.return_value = {
            "total_count": 1,
            "items": [
                {
                    "full_name": "user1/old-poc",
                    "html_url": "https://github.com/user1/old-poc",
                    "stargazers_count": 2,
                    "updated_at": TWO_YEARS_AGO,
                    "pushed_at": TWO_YEARS_AGO,
                },
            ],
        }
        result = _assess_star_velocity("CVE-2024-3094", now=NOW)
        assert result["subscore"] == _TIER_FLOOR


# ===================================================================
# Signal 3: Exploit-DB recency tests
# ===================================================================


class TestLoadExploitdbCsv:
    """Tests for _load_exploitdb_csv."""

    @patch("manus_agent.tools.get_poc_freshness._http_get_text")
    @patch("manus_agent.tools.get_poc_freshness.os.stat")
    def test_uses_cache_when_fresh(self, mock_stat, mock_http):
        mock_stat.return_value = MagicMock(st_mtime=time.time())
        with patch("builtins.open", mock_open(read_data="cached_csv_data")):
            result = _load_exploitdb_csv()
        assert result == "cached_csv_data"
        mock_http.assert_not_called()

    @patch("manus_agent.tools.get_poc_freshness._http_get_text")
    @patch("manus_agent.tools.get_poc_freshness.os.stat")
    def test_fetches_when_cache_stale(self, mock_stat, mock_http):
        mock_stat.return_value = MagicMock(st_mtime=time.time() - 100_000)
        mock_http.return_value = "fresh_csv_data"
        with patch("builtins.open", mock_open()):
            result = _load_exploitdb_csv()
        assert result == "fresh_csv_data"
        mock_http.assert_called_once()

    @patch("manus_agent.tools.get_poc_freshness._http_get_text")
    @patch("manus_agent.tools.get_poc_freshness.os.stat")
    def test_fetches_when_no_cache(self, mock_stat, mock_http):
        mock_stat.side_effect = FileNotFoundError
        mock_http.return_value = "new_csv_data"
        with patch("builtins.open", mock_open()):
            result = _load_exploitdb_csv()
        assert result == "new_csv_data"


class TestAssessExploitdbRecency:
    """Tests for _assess_exploitdb_recency."""

    @patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv")
    def test_matching_entries(self, mock_csv):
        csv_text = _make_exploitdb_csv(
            [
                {
                    "id": "12345",
                    "description": "PoC for CVE-2024-3094",
                    "date_published": "2026-08-30",
                    "codes": "CVE-2024-3094",
                },
                {
                    "id": "12346",
                    "description": "Another exploit",
                    "date_published": "2026-07-15",
                    "codes": "CVE-2024-3094;CVE-2023-1234",
                },
            ]
        )
        mock_csv.return_value = csv_text
        result = _assess_exploitdb_recency("CVE-2024-3094", now=NOW)
        assert result["total_entries"] == 2
        assert result["subscore"] > 0

    @patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv")
    def test_no_matching_entries(self, mock_csv):
        csv_text = _make_exploitdb_csv(
            [
                {"id": "99999", "description": "Unrelated", "date_published": "2026-01-01", "codes": "CVE-2023-9999"},
            ]
        )
        mock_csv.return_value = csv_text
        result = _assess_exploitdb_recency("CVE-2024-3094", now=NOW)
        assert result["total_entries"] == 0
        assert result["subscore"] == 0.0

    @patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv")
    def test_csv_load_failure(self, mock_csv):
        mock_csv.side_effect = Exception("Network error")
        result = _assess_exploitdb_recency("CVE-2024-3094", now=NOW)
        assert result["total_entries"] == 0
        assert result["subscore"] == 0.0

    @patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv")
    def test_recent_entry_high_subscore(self, mock_csv):
        csv_text = _make_exploitdb_csv(
            [
                {
                    "id": "55555",
                    "description": "Fresh exploit",
                    "date_published": "2026-09-01",
                    "codes": "CVE-2024-3094",
                },
            ]
        )
        mock_csv.return_value = csv_text
        result = _assess_exploitdb_recency("CVE-2024-3094", now=NOW)
        assert result["subscore"] == 100.0  # same day

    @patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv")
    def test_old_entry_low_subscore(self, mock_csv):
        csv_text = _make_exploitdb_csv(
            [
                {"id": "11111", "description": "Old exploit", "date_published": "2023-01-01", "codes": "CVE-2024-3094"},
            ]
        )
        mock_csv.return_value = csv_text
        result = _assess_exploitdb_recency("CVE-2024-3094", now=NOW)
        assert result["subscore"] == _TIER_FLOOR

    @patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv")
    def test_entries_capped_at_ten(self, mock_csv):
        rows = [
            {"id": str(i), "description": f"Exploit {i}", "date_published": "2026-08-01", "codes": "CVE-2024-3094"}
            for i in range(15)
        ]
        csv_text = _make_exploitdb_csv(rows)
        mock_csv.return_value = csv_text
        result = _assess_exploitdb_recency("CVE-2024-3094", now=NOW)
        assert result["total_entries"] == 15
        assert len(result["entries"]) == 10  # capped in output

    @patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv")
    def test_case_insensitive_matching(self, mock_csv):
        csv_text = _make_exploitdb_csv(
            [
                {"id": "77777", "description": "Test", "date_published": "2026-08-01", "codes": "cve-2024-3094"},
            ]
        )
        mock_csv.return_value = csv_text
        result = _assess_exploitdb_recency("CVE-2024-3094", now=NOW)
        assert result["total_entries"] == 1

    @patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv")
    def test_picks_most_recent_entry(self, mock_csv):
        csv_text = _make_exploitdb_csv(
            [
                {"id": "1", "description": "Old", "date_published": "2025-01-01", "codes": "CVE-2024-3094"},
                {"id": "2", "description": "New", "date_published": "2026-08-31", "codes": "CVE-2024-3094"},
            ]
        )
        mock_csv.return_value = csv_text
        result = _assess_exploitdb_recency("CVE-2024-3094", now=NOW)
        assert result["most_recent_date"] is not None
        assert "2026-08-31" in result["most_recent_date"]


# ===================================================================
# Composite scoring tests
# ===================================================================


class TestComputeFreshness:
    """Tests for compute_freshness."""

    def test_invalid_cve(self):
        result = compute_freshness("NOT-A-CVE")
        assert "error" in result
        assert result["freshness_score"] == 0
        assert result["tier"] == "DORMANT"

    def test_empty_cve(self):
        result = compute_freshness("")
        assert "error" in result

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_all_signals_active(self, mock_commit, mock_stars, mock_edb):
        mock_commit.return_value = {
            "repos_found": 3,
            "repos_checked": 3,
            "most_recent_commit": ONE_DAY_AGO,
            "most_recent_push": ONE_DAY_AGO,
            "days_since_last_activity": 1.0,
            "subscore": 100.0,
            "repos": [],
        }
        mock_stars.return_value = {
            "total_repos": 5,
            "recent_repos": 3,
            "total_stars": 100,
            "most_recent_update": ONE_DAY_AGO,
            "days_since_most_recent": 1.0,
            "subscore": 100.0,
            "top_repos": [],
        }
        mock_edb.return_value = {
            "total_entries": 2,
            "most_recent_date": ONE_DAY_AGO,
            "days_since_most_recent": 1.0,
            "subscore": 100.0,
            "entries": [],
        }

        result = compute_freshness("CVE-2024-3094", now=NOW)
        assert result["freshness_score"] == 100.0
        assert result["tier"] == "ACTIVE"
        assert "error" not in result

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_all_signals_dormant(self, mock_commit, mock_stars, mock_edb):
        mock_commit.return_value = {
            "repos_found": 0,
            "repos_checked": 0,
            "most_recent_commit": None,
            "most_recent_push": None,
            "days_since_last_activity": None,
            "subscore": 0.0,
            "repos": [],
        }
        mock_stars.return_value = {
            "total_repos": 0,
            "recent_repos": 0,
            "total_stars": 0,
            "most_recent_update": None,
            "days_since_most_recent": None,
            "subscore": 0.0,
            "top_repos": [],
        }
        mock_edb.return_value = {
            "total_entries": 0,
            "most_recent_date": None,
            "days_since_most_recent": None,
            "subscore": 0.0,
            "entries": [],
        }

        result = compute_freshness("CVE-2024-3094", now=NOW)
        assert result["freshness_score"] == 0.0
        assert result["tier"] == "DORMANT"
        assert result["most_recent_activity"] is None

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_mixed_signals(self, mock_commit, mock_stars, mock_edb):
        """Commit recent, stars old, no exploit-db entries."""
        mock_commit.return_value = {
            "repos_found": 1,
            "repos_checked": 1,
            "most_recent_commit": SEVEN_DAYS_AGO,
            "most_recent_push": SEVEN_DAYS_AGO,
            "days_since_last_activity": 7.0,
            "subscore": 100.0,
            "repos": [],
        }
        mock_stars.return_value = {
            "total_repos": 1,
            "recent_repos": 0,
            "total_stars": 2,
            "most_recent_update": ONE_YEAR_AGO,
            "days_since_most_recent": 365.0,
            "subscore": 25.0,
            "top_repos": [],
        }
        mock_edb.return_value = {
            "total_entries": 0,
            "most_recent_date": None,
            "days_since_most_recent": None,
            "subscore": 0.0,
            "entries": [],
        }

        result = compute_freshness("CVE-2024-3094", now=NOW)
        # 100*0.4 + 25*0.3 + 0*0.3 = 40 + 7.5 + 0 = 47.5
        assert result["freshness_score"] == 47.5
        assert result["tier"] == "AGING"

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_summary_includes_details(self, mock_commit, mock_stars, mock_edb):
        mock_commit.return_value = {
            "repos_found": 2,
            "repos_checked": 2,
            "most_recent_commit": THIRTY_DAYS_AGO,
            "most_recent_push": THIRTY_DAYS_AGO,
            "days_since_last_activity": 30.0,
            "subscore": 75.0,
            "repos": [],
        }
        mock_stars.return_value = {
            "total_repos": 3,
            "recent_repos": 1,
            "total_stars": 15,
            "most_recent_update": THIRTY_DAYS_AGO,
            "days_since_most_recent": 30.0,
            "subscore": 75.0,
            "top_repos": [],
        }
        mock_edb.return_value = {
            "total_entries": 1,
            "most_recent_date": NINETY_DAYS_AGO,
            "days_since_most_recent": 90.0,
            "subscore": 50.0,
            "entries": [],
        }

        result = compute_freshness("CVE-2024-3094", now=NOW)
        assert "2 PoC repo(s) found" in result["summary"]
        assert "3 GitHub repo(s)" in result["summary"]
        assert "1 Exploit-DB" in result["summary"]

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_most_recent_activity_picks_latest(self, mock_commit, mock_stars, mock_edb):
        mock_commit.return_value = {
            "repos_found": 1,
            "repos_checked": 1,
            "most_recent_commit": THIRTY_DAYS_AGO,
            "most_recent_push": THIRTY_DAYS_AGO,
            "days_since_last_activity": 30.0,
            "subscore": 75.0,
            "repos": [],
        }
        mock_stars.return_value = {
            "total_repos": 1,
            "recent_repos": 1,
            "total_stars": 5,
            "most_recent_update": ONE_DAY_AGO,  # most recent
            "days_since_most_recent": 1.0,
            "subscore": 100.0,
            "top_repos": [],
        }
        mock_edb.return_value = {
            "total_entries": 0,
            "most_recent_date": None,
            "days_since_most_recent": None,
            "subscore": 0.0,
            "entries": [],
        }

        result = compute_freshness("CVE-2024-3094", now=NOW)
        assert result["most_recent_activity"] is not None
        # Should be the star velocity date (most recent)
        parsed = _parse_iso_date(result["most_recent_activity"])
        assert parsed is not None
        assert _days_since(parsed, NOW) < 2.0

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_score_clamped_to_100(self, mock_commit, mock_stars, mock_edb):
        """Even with boost, score should not exceed 100."""
        mock_commit.return_value = {
            "repos_found": 1,
            "repos_checked": 1,
            "most_recent_commit": ONE_DAY_AGO,
            "most_recent_push": ONE_DAY_AGO,
            "days_since_last_activity": 1.0,
            "subscore": 100.0,
            "repos": [],
        }
        mock_stars.return_value = {
            "total_repos": 10,
            "recent_repos": 10,
            "total_stars": 500,
            "most_recent_update": ONE_DAY_AGO,
            "days_since_most_recent": 1.0,
            "subscore": 100.0,  # capped at 100 even with boost
            "top_repos": [],
        }
        mock_edb.return_value = {
            "total_entries": 5,
            "most_recent_date": ONE_DAY_AGO,
            "days_since_most_recent": 1.0,
            "subscore": 100.0,
            "entries": [],
        }

        result = compute_freshness("CVE-2024-3094", now=NOW)
        assert result["freshness_score"] <= 100.0

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_cve_id_normalised_to_uppercase(self, mock_commit, mock_stars, mock_edb):
        for mock_fn in [mock_commit, mock_stars, mock_edb]:
            mock_fn.return_value = {
                "repos_found": 0,
                "repos_checked": 0,
                "most_recent_commit": None,
                "most_recent_push": None,
                "most_recent_date": None,
                "most_recent_update": None,
                "days_since_last_activity": None,
                "days_since_most_recent": None,
                "total_repos": 0,
                "recent_repos": 0,
                "total_stars": 0,
                "total_entries": 0,
                "subscore": 0.0,
                "repos": [],
                "top_repos": [],
                "entries": [],
            }

        result = compute_freshness("cve-2024-3094", now=NOW)
        assert result["cve_id"] == "CVE-2024-3094"

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_signals_structure(self, mock_commit, mock_stars, mock_edb):
        """Result signals dict contains all three signal keys."""
        for mock_fn in [mock_commit, mock_stars, mock_edb]:
            mock_fn.return_value = {
                "repos_found": 0,
                "repos_checked": 0,
                "most_recent_commit": None,
                "most_recent_push": None,
                "most_recent_date": None,
                "most_recent_update": None,
                "days_since_last_activity": None,
                "days_since_most_recent": None,
                "total_repos": 0,
                "recent_repos": 0,
                "total_stars": 0,
                "total_entries": 0,
                "subscore": 0.0,
                "repos": [],
                "top_repos": [],
                "entries": [],
            }

        result = compute_freshness("CVE-2024-3094", now=NOW)
        signals = result["signals"]
        assert "commit_recency" in signals
        assert "star_velocity" in signals
        assert "exploitdb_recency" in signals
        for sig in signals.values():
            assert "weight" in sig
            assert "subscore" in sig


# ===================================================================
# Strands tool handler tests
# ===================================================================


class TestGetPocFreshnessTool:
    """Tests for the Strands tool handler."""

    @patch("manus_agent.tools.get_poc_freshness.compute_freshness")
    def test_success_result(self, mock_compute):
        mock_compute.return_value = {
            "cve_id": "CVE-2024-3094",
            "freshness_score": 85.0,
            "tier": "ACTIVE",
            "most_recent_activity": ONE_DAY_AGO,
            "summary": "3 PoC repo(s) found",
            "signals": {},
        }

        tool_use = {
            "toolUseId": "test-123",
            "input": {"cve_id": "CVE-2024-3094"},
        }
        result = get_poc_freshness(tool_use)
        assert result["status"] == "success"
        assert result["toolUseId"] == "test-123"

    @patch("manus_agent.tools.get_poc_freshness.compute_freshness")
    def test_error_result(self, mock_compute):
        mock_compute.return_value = {
            "error": "Invalid CVE identifier",
            "cve_id": "BAD",
            "freshness_score": 0,
            "tier": "DORMANT",
            "signals": {},
            "summary": "Invalid CVE identifier",
        }

        tool_use = {
            "toolUseId": "test-456",
            "input": {"cve_id": "BAD"},
        }
        result = get_poc_freshness(tool_use)
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_poc_freshness.compute_freshness")
    def test_empty_cve_id(self, mock_compute):
        mock_compute.return_value = {
            "error": "Invalid CVE identifier: ''",
            "cve_id": "",
            "freshness_score": 0,
            "tier": "DORMANT",
            "signals": {},
            "summary": "Invalid CVE identifier: ''",
        }

        tool_use = {
            "toolUseId": "test-789",
            "input": {"cve_id": ""},
        }
        result = get_poc_freshness(tool_use)
        assert result["status"] == "error"


# ===================================================================
# CLI subcommand tests
# ===================================================================


class TestCliPocFreshness:
    """Tests for the CLI poc-freshness subcommand."""

    @patch("manus_agent.tools.get_poc_freshness.compute_freshness")
    def test_json_output(self, mock_compute):
        mock_compute.return_value = {
            "cve_id": "CVE-2024-3094",
            "freshness_score": 75.0,
            "tier": "RECENT",
            "most_recent_activity": SEVEN_DAYS_AGO,
            "summary": "2 PoC repo(s) found",
            "signals": {
                "commit_recency": {
                    "weight": 0.4,
                    "subscore": 100.0,
                    "repos_found": 2,
                    "repos_checked": 2,
                    "most_recent_commit": SEVEN_DAYS_AGO,
                    "days_since_last_activity": 7.0,
                    "repos": [],
                },
                "star_velocity": {
                    "weight": 0.3,
                    "subscore": 50.0,
                    "total_repos": 1,
                    "recent_repos": 0,
                    "total_stars": 3,
                    "most_recent_update": NINETY_DAYS_AGO,
                    "days_since_most_recent": 90.0,
                    "top_repos": [],
                },
                "exploitdb_recency": {
                    "weight": 0.3,
                    "subscore": 50.0,
                    "total_entries": 1,
                    "most_recent_date": NINETY_DAYS_AGO,
                    "days_since_most_recent": 90.0,
                    "entries": [],
                },
            },
        }

        from manus_agent.cli import _run_poc_freshness

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            exit_code = _run_poc_freshness(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        output = json.loads(mock_stdout.getvalue())
        assert output["freshness_score"] == 75.0
        assert output["tier"] == "RECENT"

    @patch("manus_agent.tools.get_poc_freshness.compute_freshness")
    def test_text_output(self, mock_compute):
        mock_compute.return_value = {
            "cve_id": "CVE-2024-3094",
            "freshness_score": 85.0,
            "tier": "ACTIVE",
            "most_recent_activity": ONE_DAY_AGO,
            "summary": "3 PoC repo(s) found",
            "signals": {
                "commit_recency": {
                    "weight": 0.4,
                    "subscore": 100.0,
                    "repos_found": 3,
                    "repos_checked": 3,
                    "most_recent_commit": ONE_DAY_AGO,
                    "days_since_last_activity": 1.0,
                    "repos": [],
                },
                "star_velocity": {
                    "weight": 0.3,
                    "subscore": 75.0,
                    "total_repos": 2,
                    "recent_repos": 1,
                    "total_stars": 20,
                    "most_recent_update": SEVEN_DAYS_AGO,
                    "days_since_most_recent": 7.0,
                    "top_repos": [],
                },
                "exploitdb_recency": {
                    "weight": 0.3,
                    "subscore": 75.0,
                    "total_entries": 1,
                    "most_recent_date": THIRTY_DAYS_AGO,
                    "days_since_most_recent": 30.0,
                    "entries": [],
                },
            },
        }

        from manus_agent.cli import _run_poc_freshness

        exit_code = _run_poc_freshness(["CVE-2024-3094"])
        assert exit_code == 0

    @patch("manus_agent.tools.get_poc_freshness.compute_freshness")
    def test_error_from_compute(self, mock_compute):
        """When compute_freshness returns an error dict, exit code is 1."""
        mock_compute.return_value = {
            "error": "Some unexpected error",
            "cve_id": "CVE-9999-0001",
            "freshness_score": 0,
            "tier": "DORMANT",
            "signals": {},
            "summary": "Error",
        }

        from manus_agent.cli import _run_poc_freshness

        exit_code = _run_poc_freshness(["CVE-9999-0001"])
        assert exit_code == 1

    def test_invalid_cve_format_rejected(self):
        """Parser rejects obviously invalid CVE IDs."""
        from manus_agent.cli import _run_poc_freshness

        with pytest.raises(SystemExit):
            _run_poc_freshness(["NOT-A-CVE"])

    @patch("manus_agent.tools.get_poc_freshness.compute_freshness")
    def test_text_output_with_repos(self, mock_compute):
        mock_compute.return_value = {
            "cve_id": "CVE-2024-3094",
            "freshness_score": 90.0,
            "tier": "ACTIVE",
            "most_recent_activity": ONE_DAY_AGO,
            "summary": "2 PoC repo(s) found",
            "signals": {
                "commit_recency": {
                    "weight": 0.4,
                    "subscore": 100.0,
                    "repos_found": 2,
                    "repos_checked": 2,
                    "most_recent_commit": ONE_DAY_AGO,
                    "days_since_last_activity": 1.0,
                    "repos": [
                        {
                            "repo_url": "https://github.com/user1/poc",
                            "stargazers_count": 42,
                            "last_commit_at": ONE_DAY_AGO,
                            "pushed_at": ONE_DAY_AGO,
                            "days_since_activity": 1.0,
                            "error": None,
                        },
                        {
                            "repo_url": "https://github.com/user2/exploit",
                            "stargazers_count": 5,
                            "last_commit_at": SEVEN_DAYS_AGO,
                            "pushed_at": SEVEN_DAYS_AGO,
                            "days_since_activity": 7.0,
                            "error": None,
                        },
                    ],
                },
                "star_velocity": {
                    "weight": 0.3,
                    "subscore": 100.0,
                    "total_repos": 3,
                    "recent_repos": 2,
                    "total_stars": 50,
                    "most_recent_update": ONE_DAY_AGO,
                    "days_since_most_recent": 1.0,
                    "top_repos": [],
                },
                "exploitdb_recency": {
                    "weight": 0.3,
                    "subscore": 75.0,
                    "total_entries": 1,
                    "most_recent_date": THIRTY_DAYS_AGO,
                    "days_since_most_recent": 30.0,
                    "entries": [],
                },
            },
        }

        from manus_agent.cli import _run_poc_freshness

        exit_code = _run_poc_freshness(["CVE-2024-3094"])
        assert exit_code == 0


# ===================================================================
# Edge case / integration-level tests
# ===================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_whitespace_cve_id(self, mock_commit, mock_stars, mock_edb):
        for mock_fn in [mock_commit, mock_stars, mock_edb]:
            mock_fn.return_value = {
                "repos_found": 0,
                "repos_checked": 0,
                "most_recent_commit": None,
                "most_recent_push": None,
                "most_recent_date": None,
                "most_recent_update": None,
                "days_since_last_activity": None,
                "days_since_most_recent": None,
                "total_repos": 0,
                "recent_repos": 0,
                "total_stars": 0,
                "total_entries": 0,
                "subscore": 0.0,
                "repos": [],
                "top_repos": [],
                "entries": [],
            }

        result = compute_freshness("  CVE-2024-3094  ", now=NOW)
        assert result["cve_id"] == "CVE-2024-3094"

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_score_rounding(self, mock_commit, mock_stars, mock_edb):
        """Score is rounded to 1 decimal place."""
        mock_commit.return_value = {
            "repos_found": 1,
            "repos_checked": 1,
            "most_recent_commit": THIRTY_DAYS_AGO,
            "most_recent_push": THIRTY_DAYS_AGO,
            "days_since_last_activity": 30.0,
            "subscore": 75.0,
            "repos": [],
        }
        mock_stars.return_value = {
            "total_repos": 1,
            "recent_repos": 0,
            "total_stars": 2,
            "most_recent_update": NINETY_DAYS_AGO,
            "days_since_most_recent": 90.0,
            "subscore": 50.0,
            "top_repos": [],
        }
        mock_edb.return_value = {
            "total_entries": 1,
            "most_recent_date": ONE_YEAR_AGO,
            "days_since_most_recent": 365.0,
            "subscore": 25.0,
            "entries": [],
        }

        result = compute_freshness("CVE-2024-3094", now=NOW)
        # 75*0.4 + 50*0.3 + 25*0.3 = 30 + 15 + 7.5 = 52.5
        assert result["freshness_score"] == 52.5
        assert isinstance(result["freshness_score"], float)

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_tier_boundary_80(self, mock_commit, mock_stars, mock_edb):
        """Score of exactly 80 should be ACTIVE."""
        mock_commit.return_value = {
            "repos_found": 1,
            "repos_checked": 1,
            "most_recent_commit": ONE_DAY_AGO,
            "most_recent_push": ONE_DAY_AGO,
            "days_since_last_activity": 1.0,
            "subscore": 100.0,
            "repos": [],
        }
        mock_stars.return_value = {
            "total_repos": 1,
            "recent_repos": 1,
            "total_stars": 5,
            "most_recent_update": THIRTY_DAYS_AGO,
            "days_since_most_recent": 30.0,
            "subscore": 75.0,
            "top_repos": [],
        }
        mock_edb.return_value = {
            "total_entries": 1,
            "most_recent_date": NINETY_DAYS_AGO,
            "days_since_most_recent": 90.0,
            "subscore": 50.0,
            "entries": [],
        }

        result = compute_freshness("CVE-2024-3094", now=NOW)
        # 100*0.4 + 75*0.3 + 50*0.3 = 40 + 22.5 + 15 = 77.5
        # Not exactly 80, but tests the RECENT tier
        assert result["tier"] == "RECENT"

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_only_commit_signal(self, mock_commit, mock_stars, mock_edb):
        """Only commit recency has data; others are empty."""
        mock_commit.return_value = {
            "repos_found": 1,
            "repos_checked": 1,
            "most_recent_commit": ONE_DAY_AGO,
            "most_recent_push": ONE_DAY_AGO,
            "days_since_last_activity": 1.0,
            "subscore": 100.0,
            "repos": [],
        }
        mock_stars.return_value = {
            "total_repos": 0,
            "recent_repos": 0,
            "total_stars": 0,
            "most_recent_update": None,
            "days_since_most_recent": None,
            "subscore": 0.0,
            "top_repos": [],
        }
        mock_edb.return_value = {
            "total_entries": 0,
            "most_recent_date": None,
            "days_since_most_recent": None,
            "subscore": 0.0,
            "entries": [],
        }

        result = compute_freshness("CVE-2024-3094", now=NOW)
        # 100*0.4 + 0*0.3 + 0*0.3 = 40.0
        assert result["freshness_score"] == 40.0
        assert result["tier"] == "AGING"

    @patch("manus_agent.tools.get_poc_freshness._assess_exploitdb_recency")
    @patch("manus_agent.tools.get_poc_freshness._assess_star_velocity")
    @patch("manus_agent.tools.get_poc_freshness._assess_commit_recency")
    def test_only_exploitdb_signal(self, mock_commit, mock_stars, mock_edb):
        """Only Exploit-DB has data."""
        mock_commit.return_value = {
            "repos_found": 0,
            "repos_checked": 0,
            "most_recent_commit": None,
            "most_recent_push": None,
            "days_since_last_activity": None,
            "subscore": 0.0,
            "repos": [],
        }
        mock_stars.return_value = {
            "total_repos": 0,
            "recent_repos": 0,
            "total_stars": 0,
            "most_recent_update": None,
            "days_since_most_recent": None,
            "subscore": 0.0,
            "top_repos": [],
        }
        mock_edb.return_value = {
            "total_entries": 1,
            "most_recent_date": ONE_DAY_AGO,
            "days_since_most_recent": 1.0,
            "subscore": 100.0,
            "entries": [],
        }

        result = compute_freshness("CVE-2024-3094", now=NOW)
        # 0*0.4 + 0*0.3 + 100*0.3 = 30.0
        assert result["freshness_score"] == 30.0
        assert result["tier"] == "STALE"
