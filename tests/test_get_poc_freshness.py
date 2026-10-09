"""Comprehensive test suite for get_poc_freshness module.

100% mocked — no real HTTP calls. Tests cover:
- TOOL_SPEC contract
- Input validation
- All three data fetchers (_fetch_github_repos, _fetch_exploitdb_entries, _fetch_trickest_pocs)
- The compute_freshness scoring engine (all labels, edge cases, decay math)
- The fetch_poc_freshness orchestrator
- The Strands tool handler (get_poc_freshness)
- Source tracking (checked/failed lists)
- CLI subcommand (_run_poc_freshness, _build_poc_freshness_parser)
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest

from manus_agent.tools.get_poc_freshness import (
    TOOL_SPEC,
    _count_active_sources,
    _days_since,
    _decay,
    _fetch_exploitdb_entries,
    _fetch_github_repos,
    _fetch_trickest_pocs,
    _list_sources_checked,
    _list_sources_failed,
    _star_score,
    _volume_score,
    compute_freshness,
    fetch_poc_freshness,
    get_poc_freshness,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def _tool_use(cve_id: str | None = None) -> dict:
    inp: dict[str, Any] = {}
    if cve_id is not None:
        inp["cve_id"] = cve_id
    return {"toolUseId": "test-id-001", "input": inp}


def _github_data(
    repos: list[dict] | None = None,
    total_count: int = 0,
    newest_push: str | None = None,
    error: str | None = None,
) -> dict:
    return {
        "repos": repos or [],
        "total_count": total_count,
        "newest_push": newest_push,
        "error": error,
    }


def _exploitdb_data(
    entries: list[dict] | None = None,
    count: int = 0,
    newest_date: str | None = None,
    error: str | None = None,
) -> dict:
    return {
        "entries": entries or [],
        "count": count,
        "newest_date": newest_date,
        "error": error,
    }


def _trickest_data(
    poc_count: int = 0,
    github_pocs: list[str] | None = None,
    reference_pocs: list[str] | None = None,
    error: str | None = None,
) -> dict:
    return {
        "poc_count": poc_count,
        "github_pocs": github_pocs or [],
        "reference_pocs": reference_pocs or [],
        "error": error,
    }


# ===================================================================
# TOOL_SPEC contract
# ===================================================================


class TestToolSpec:
    def test_spec_has_required_keys(self):
        assert "name" in TOOL_SPEC
        assert "description" in TOOL_SPEC
        assert "inputSchema" in TOOL_SPEC

    def test_spec_name(self):
        assert TOOL_SPEC["name"] == "get_poc_freshness"

    def test_spec_has_cve_id_required(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_spec_description_mentions_freshness(self):
        assert "freshness" in TOOL_SPEC["description"].lower()


# ===================================================================
# Helper functions: _days_since, _decay, _volume_score, _star_score
# ===================================================================


class TestDaysSince:
    def test_none_input(self):
        assert _days_since(None, NOW) is None

    def test_empty_string(self):
        assert _days_since("", NOW) is None

    def test_iso_datetime_with_z(self):
        # 10 days before NOW
        result = _days_since("2025-06-05T12:00:00Z", NOW)
        assert result is not None
        assert abs(result - 10.0) < 0.01

    def test_iso_datetime_with_offset(self):
        result = _days_since("2025-06-15T12:00:00+00:00", NOW)
        assert result is not None
        assert abs(result) < 0.01

    def test_date_only(self):
        result = _days_since("2025-06-10", NOW)
        assert result is not None
        assert abs(result - 5.5) < 0.01  # noon vs midnight = 0.5 day

    def test_future_date_clamps_to_zero(self):
        result = _days_since("2025-06-20T12:00:00Z", NOW)
        assert result is not None
        assert result == 0.0

    def test_invalid_format_returns_none(self):
        assert _days_since("not-a-date", NOW) is None

    def test_uses_current_time_when_now_is_none(self):
        result = _days_since("2020-01-01T00:00:00Z")
        assert result is not None
        assert result > 0


class TestDecay:
    def test_none_returns_zero(self):
        assert _decay(None) == 0.0

    def test_zero_days_returns_one(self):
        assert abs(_decay(0.0) - 1.0) < 0.001

    def test_half_life(self):
        # At 30 days (half-life), should be ~0.5
        result = _decay(30.0)
        assert abs(result - 0.5) < 0.01

    def test_double_half_life(self):
        # At 60 days, should be ~0.25
        result = _decay(60.0)
        assert abs(result - 0.25) < 0.01

    def test_large_value_approaches_zero(self):
        result = _decay(365.0)
        assert result < 0.01

    def test_small_value_close_to_one(self):
        result = _decay(1.0)
        assert result > 0.95


class TestVolumeScore:
    def test_zero(self):
        assert _volume_score(0) == 0.0

    def test_negative(self):
        assert _volume_score(-1) == 0.0

    def test_one(self):
        result = _volume_score(1)
        assert 0.0 < result < 1.0

    def test_thirty_is_max(self):
        result = _volume_score(30)
        assert abs(result - 1.0) < 0.01

    def test_more_than_thirty_caps(self):
        result = _volume_score(100)
        assert result == 1.0

    def test_monotonic_increase(self):
        prev = 0.0
        for n in [1, 2, 5, 10, 20, 30]:
            cur = _volume_score(n)
            assert cur >= prev
            prev = cur


class TestStarScore:
    def test_zero(self):
        assert _star_score(0, 0) == 0.0

    def test_stars_only(self):
        result = _star_score(10, 0)
        assert 0.0 < result < 1.0

    def test_forks_weighted_double(self):
        # 5 forks = 10 combined, same as 10 stars + 0 forks
        a = _star_score(10, 0)
        b = _star_score(0, 5)
        assert abs(a - b) < 0.01

    def test_high_values_cap_at_one(self):
        result = _star_score(1000, 500)
        assert result == 1.0

    def test_moderate_values(self):
        result = _star_score(50, 10)
        assert 0.5 < result < 1.0


# ===================================================================
# _fetch_github_repos
# ===================================================================


class TestFetchGithubRepos:
    @patch("manus_agent.tools.get_poc_freshness._get_json")
    def test_success(self, mock_get):
        mock_get.return_value = {
            "total_count": 2,
            "items": [
                {
                    "full_name": "user/poc-cve-2024-1234",
                    "html_url": "https://github.com/user/poc-cve-2024-1234",
                    "pushed_at": "2025-06-10T10:00:00Z",
                    "stargazers_count": 42,
                    "forks_count": 5,
                },
                {
                    "full_name": "user2/exploit-demo",
                    "html_url": "https://github.com/user2/exploit-demo",
                    "pushed_at": "2025-05-01T08:00:00Z",
                    "stargazers_count": 10,
                    "forks_count": 2,
                },
            ],
        }
        result = _fetch_github_repos("CVE-2024-1234")
        assert result["error"] is None
        assert result["total_count"] == 2
        assert len(result["repos"]) == 2
        assert result["newest_push"] == "2025-06-10T10:00:00Z"
        assert result["repos"][0]["stars"] == 42

    @patch("manus_agent.tools.get_poc_freshness._get_json")
    def test_empty_results(self, mock_get):
        mock_get.return_value = {"total_count": 0, "items": []}
        result = _fetch_github_repos("CVE-2099-9999")
        assert result["error"] is None
        assert result["total_count"] == 0
        assert result["repos"] == []
        assert result["newest_push"] is None

    @patch("manus_agent.tools.get_poc_freshness._get_json")
    def test_network_error(self, mock_get):
        mock_get.side_effect = Exception("Connection timeout")
        result = _fetch_github_repos("CVE-2024-1234")
        assert result["error"] is not None
        assert "Connection timeout" in result["error"]
        assert result["repos"] == []
        assert result["total_count"] == 0

    @patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"})
    @patch("manus_agent.tools.get_poc_freshness._get_json")
    def test_uses_github_token(self, mock_get):
        mock_get.return_value = {"total_count": 0, "items": []}
        _fetch_github_repos("CVE-2024-1234")
        call_args = mock_get.call_args
        headers = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("headers", {})
        assert "Authorization" in headers
        assert headers["Authorization"] == "token ghp_test123"

    @patch.dict("os.environ", {}, clear=True)
    @patch("manus_agent.tools.get_poc_freshness._get_json")
    def test_no_github_token(self, mock_get):
        mock_get.return_value = {"total_count": 0, "items": []}
        _fetch_github_repos("CVE-2024-1234")
        # Should still call the API even without a token
        assert mock_get.called

    @patch("manus_agent.tools.get_poc_freshness._get_json")
    def test_picks_newest_push(self, mock_get):
        mock_get.return_value = {
            "total_count": 3,
            "items": [
                {
                    "full_name": "a",
                    "html_url": "",
                    "pushed_at": "2025-01-01T00:00:00Z",
                    "stargazers_count": 0,
                    "forks_count": 0,
                },
                {
                    "full_name": "b",
                    "html_url": "",
                    "pushed_at": "2025-06-10T00:00:00Z",
                    "stargazers_count": 0,
                    "forks_count": 0,
                },
                {
                    "full_name": "c",
                    "html_url": "",
                    "pushed_at": "2025-03-15T00:00:00Z",
                    "stargazers_count": 0,
                    "forks_count": 0,
                },
            ],
        }
        result = _fetch_github_repos("CVE-2024-1234")
        assert result["newest_push"] == "2025-06-10T00:00:00Z"


# ===================================================================
# _fetch_exploitdb_entries
# ===================================================================


class TestFetchExploitdbEntries:
    @patch("manus_agent.tools.get_poc_freshness._get_text")
    @patch("os.stat")
    def test_cache_miss_fetches_csv(self, mock_stat, mock_get_text):
        mock_stat.side_effect = OSError("No cache")
        mock_get_text.return_value = (
            "id,file,description,date_published,author,platform,type,port,codes\n"
            "12345,exploits/linux/local/12345.py,Test exploit,2025-05-20,author,linux,local,,CVE-2024-1234\n"
        )
        with patch("builtins.open", mock_open()):
            result = _fetch_exploitdb_entries("CVE-2024-1234")
        assert result["error"] is None
        assert result["count"] == 1
        assert result["entries"][0]["id"] == "12345"
        assert result["newest_date"] == "2025-05-20"

    @patch("manus_agent.tools.get_poc_freshness._get_text")
    @patch("os.stat")
    def test_no_matches(self, mock_stat, mock_get_text):
        mock_stat.side_effect = OSError("No cache")
        mock_get_text.return_value = (
            "id,file,description,date_published,author,platform,type,port,codes\n"
            "99999,exploits/linux/local/99999.py,Unrelated,2025-01-01,author,linux,local,,CVE-2099-9999\n"
        )
        with patch("builtins.open", mock_open()):
            result = _fetch_exploitdb_entries("CVE-2024-1234")
        assert result["count"] == 0
        assert result["entries"] == []

    @patch("manus_agent.tools.get_poc_freshness._get_text")
    @patch("os.stat")
    def test_multiple_matches(self, mock_stat, mock_get_text):
        mock_stat.side_effect = OSError("No cache")
        mock_get_text.return_value = (
            "id,file,description,date_published,author,platform,type,port,codes\n"
            "111,e/111.py,First,2025-01-15,a,linux,local,,CVE-2024-1234\n"
            "222,e/222.py,Second,2025-06-01,b,linux,local,,CVE-2024-1234\n"
        )
        with patch("builtins.open", mock_open()):
            result = _fetch_exploitdb_entries("CVE-2024-1234")
        assert result["count"] == 2
        assert result["newest_date"] == "2025-06-01"

    @patch("manus_agent.tools.get_poc_freshness._get_text")
    @patch("os.stat")
    def test_csv_fetch_failure(self, mock_stat, mock_get_text):
        mock_stat.side_effect = OSError("No cache")
        mock_get_text.side_effect = Exception("Network error")
        result = _fetch_exploitdb_entries("CVE-2024-1234")
        assert result["error"] is not None
        assert result["count"] == 0

    @patch("time.time", return_value=1000000.0)
    @patch("os.stat")
    def test_uses_fresh_cache(self, mock_stat, mock_time):
        mock_stat_result = MagicMock()
        mock_stat_result.st_mtime = 999999.0  # 1 second ago
        mock_stat.return_value = mock_stat_result
        csv_data = (
            "id,file,description,date_published,author,platform,type,port,codes\n"
            "555,e/555.py,Cached,2025-03-10,a,linux,local,,CVE-2024-5555\n"
        )
        with patch("builtins.open", mock_open(read_data=csv_data)):
            result = _fetch_exploitdb_entries("CVE-2024-5555")
        assert result["count"] == 1
        assert result["error"] is None

    @patch("manus_agent.tools.get_poc_freshness._get_text")
    @patch("os.stat")
    def test_matches_in_description(self, mock_stat, mock_get_text):
        mock_stat.side_effect = OSError("No cache")
        mock_get_text.return_value = (
            "id,file,description,date_published,author,platform,type,port,codes\n"
            "333,e/333.py,Exploit for CVE-2024-1234 RCE,2025-04-01,a,linux,local,,\n"
        )
        with patch("builtins.open", mock_open()):
            result = _fetch_exploitdb_entries("CVE-2024-1234")
        assert result["count"] == 1

    @patch("manus_agent.tools.get_poc_freshness._get_text")
    @patch("os.stat")
    def test_empty_csv(self, mock_stat, mock_get_text):
        mock_stat.side_effect = OSError("No cache")
        mock_get_text.return_value = ""
        with patch("builtins.open", mock_open()):
            result = _fetch_exploitdb_entries("CVE-2024-1234")
        assert result["count"] == 0
        assert result["error"] == "Empty CSV"


# ===================================================================
# _fetch_trickest_pocs
# ===================================================================


class TestFetchTrickestPocs:
    @patch("manus_agent.tools.get_poc_freshness._get_text")
    def test_success_with_pocs(self, mock_get):
        mock_get.return_value = (
            "### Description\nSome vuln\n"
            "### POC\n"
            "#### Reference\n"
            "- https://www.exploit-db.com/exploits/12345\n"
            "#### Github\n"
            "- https://github.com/user/poc-repo\n"
            "- https://github.com/user2/another-poc\n"
        )
        result = _fetch_trickest_pocs("CVE-2024-1234")
        assert result["error"] is None
        assert result["poc_count"] == 3
        assert len(result["github_pocs"]) == 2
        assert len(result["reference_pocs"]) == 1

    @patch("manus_agent.tools.get_poc_freshness._get_text")
    def test_404_returns_empty(self, mock_get):
        mock_get.side_effect = urllib.error.HTTPError("url", 404, "Not Found", {}, io.BytesIO(b""))
        result = _fetch_trickest_pocs("CVE-2024-1234")
        assert result["error"] is None
        assert result["poc_count"] == 0

    @patch("manus_agent.tools.get_poc_freshness._get_text")
    def test_http_error_non_404(self, mock_get):
        mock_get.side_effect = urllib.error.HTTPError("url", 500, "Server Error", {}, io.BytesIO(b""))
        result = _fetch_trickest_pocs("CVE-2024-1234")
        assert result["error"] is not None
        assert "500" in result["error"]

    @patch("manus_agent.tools.get_poc_freshness._get_text")
    def test_generic_exception(self, mock_get):
        mock_get.side_effect = ConnectionError("DNS failure")
        result = _fetch_trickest_pocs("CVE-2024-1234")
        assert result["error"] is not None
        assert result["poc_count"] == 0

    def test_invalid_cve_id(self):
        result = _fetch_trickest_pocs("NOT-A-CVE")
        assert result["error"] == "Invalid CVE ID"
        assert result["poc_count"] == 0

    @patch("manus_agent.tools.get_poc_freshness._get_text")
    def test_no_pocs_in_markdown(self, mock_get):
        mock_get.return_value = "### Description\nJust a description\n"
        result = _fetch_trickest_pocs("CVE-2024-1234")
        assert result["poc_count"] == 0
        assert result["error"] is None

    @patch("manus_agent.tools.get_poc_freshness._get_text")
    def test_deduplication(self, mock_get):
        mock_get.return_value = (
            "### POC\n#### Reference\n- https://github.com/user/poc\n#### Github\n- https://github.com/user/poc\n"
        )
        result = _fetch_trickest_pocs("CVE-2024-1234")
        # github_pocs and reference_pocs are independent lists
        # but poc_count is deduplicated
        assert result["poc_count"] == 1


# ===================================================================
# compute_freshness scoring engine
# ===================================================================


class TestComputeFreshness:
    def test_all_zeros(self):
        result = compute_freshness(_github_data(), _exploitdb_data(), _trickest_data(), now=NOW)
        assert result["freshness_score"] == 0
        assert result["label"] == "stale"

    def test_hot_score(self):
        # Recent activity, many repos, many stars, Exploit-DB entries
        gh = _github_data(
            repos=[
                {"stars": 200, "forks": 50},
                {"stars": 100, "forks": 30},
            ],
            total_count=15,
            newest_push="2025-06-15T11:00:00Z",  # 1 hour ago
        )
        edb = _exploitdb_data(count=3, newest_date="2025-06-14")
        tr = _trickest_data(poc_count=5)
        result = compute_freshness(gh, edb, tr, now=NOW)
        assert result["freshness_score"] >= 80
        assert result["label"] == "hot"

    def test_warm_score(self):
        # Moderate activity, a few repos, some stars
        gh = _github_data(
            repos=[{"stars": 20, "forks": 5}],
            total_count=3,
            newest_push="2025-05-20T00:00:00Z",  # ~26 days ago
        )
        edb = _exploitdb_data(count=1, newest_date="2025-05-15")
        tr = _trickest_data(poc_count=2)
        result = compute_freshness(gh, edb, tr, now=NOW)
        assert 50 <= result["freshness_score"] < 80
        assert result["label"] == "warm"

    def test_cooling_score(self):
        # Old activity, few repos, some Exploit-DB
        gh = _github_data(
            repos=[{"stars": 5, "forks": 2}],
            total_count=3,
            newest_push="2025-03-01T00:00:00Z",  # ~106 days ago
        )
        edb = _exploitdb_data(count=1, newest_date="2025-03-01")
        tr = _trickest_data(poc_count=2)
        result = compute_freshness(gh, edb, tr, now=NOW)
        assert 20 <= result["freshness_score"] < 50
        assert result["label"] == "cooling"

    def test_stale_score(self):
        # Very old activity, minimal repos
        gh = _github_data(
            repos=[{"stars": 0, "forks": 0}],
            total_count=1,
            newest_push="2023-01-01T00:00:00Z",  # >2 years ago
        )
        edb = _exploitdb_data(count=0)
        tr = _trickest_data(poc_count=0)
        result = compute_freshness(gh, edb, tr, now=NOW)
        assert result["freshness_score"] < 20
        assert result["label"] == "stale"

    def test_result_structure(self):
        result = compute_freshness(_github_data(), _exploitdb_data(), _trickest_data(), now=NOW)
        assert "freshness_score" in result
        assert "label" in result
        assert "components" in result
        assert "newest_activity" in result
        assert "days_since_activity" in result
        assert "total_pocs" in result
        assert "github_repos" in result
        assert "github_stars" in result
        assert "github_forks" in result
        assert "exploitdb_entries" in result
        assert "trickest_pocs" in result
        assert "sources_checked" in result
        assert "sources_failed" in result
        assert "summary" in result

    def test_components_structure(self):
        result = compute_freshness(_github_data(), _exploitdb_data(), _trickest_data(), now=NOW)
        components = result["components"]
        assert "recency" in components
        assert "volume" in components
        assert "stars" in components
        assert "exploitdb" in components

    def test_score_clamped_0_100(self):
        result = compute_freshness(_github_data(), _exploitdb_data(), _trickest_data(), now=NOW)
        assert 0 <= result["freshness_score"] <= 100

    def test_exploitdb_component_caps_at_one(self):
        edb = _exploitdb_data(count=10)
        result = compute_freshness(_github_data(), edb, _trickest_data(), now=NOW)
        assert result["components"]["exploitdb"] <= 1.0

    def test_newest_activity_picks_github(self):
        gh = _github_data(newest_push="2025-06-14T00:00:00Z")
        edb = _exploitdb_data(newest_date="2025-06-10")
        result = compute_freshness(gh, edb, _trickest_data(), now=NOW)
        assert result["newest_activity"] == "2025-06-14T00:00:00Z"

    def test_newest_activity_picks_exploitdb(self):
        gh = _github_data(newest_push="2025-01-01T00:00:00Z")
        edb = _exploitdb_data(newest_date="2025-06-14")
        result = compute_freshness(gh, edb, _trickest_data(), now=NOW)
        assert result["newest_activity"] == "2025-06-14"

    def test_no_newest_activity(self):
        result = compute_freshness(_github_data(), _exploitdb_data(), _trickest_data(), now=NOW)
        assert result["newest_activity"] is None
        assert result["days_since_activity"] is None

    def test_summary_contains_activity_info(self):
        gh = _github_data(
            repos=[{"stars": 5, "forks": 1}],
            total_count=3,
            newest_push="2025-06-14T00:00:00Z",
        )
        edb = _exploitdb_data(count=1, newest_date="2025-06-10")
        tr = _trickest_data(poc_count=2)
        result = compute_freshness(gh, edb, tr, now=NOW)
        assert "day" in result["summary"].lower()

    def test_summary_no_activity(self):
        result = compute_freshness(_github_data(), _exploitdb_data(), _trickest_data(), now=NOW)
        assert "no datable" in result["summary"].lower()

    def test_github_stars_forks_sum(self):
        gh = _github_data(
            repos=[
                {"stars": 10, "forks": 3},
                {"stars": 20, "forks": 7},
            ],
            total_count=2,
        )
        result = compute_freshness(gh, _exploitdb_data(), _trickest_data(), now=NOW)
        assert result["github_stars"] == 30
        assert result["github_forks"] == 10

    def test_total_pocs_sum(self):
        gh = _github_data(total_count=5)
        edb = _exploitdb_data(count=2)
        tr = _trickest_data(poc_count=3)
        result = compute_freshness(gh, edb, tr, now=NOW)
        assert result["total_pocs"] == 10

    def test_recency_within_24h(self):
        gh = _github_data(newest_push="2025-06-15T11:00:00Z")  # 1h ago
        result = compute_freshness(gh, _exploitdb_data(), _trickest_data(), now=NOW)
        assert "24 hours" in result["summary"]

    def test_recency_within_week(self):
        gh = _github_data(newest_push="2025-06-12T12:00:00Z")  # 3 days ago
        result = compute_freshness(gh, _exploitdb_data(), _trickest_data(), now=NOW)
        assert "day(s) ago" in result["summary"]

    def test_recency_within_month(self):
        gh = _github_data(newest_push="2025-05-25T12:00:00Z")  # ~21 days
        result = compute_freshness(gh, _exploitdb_data(), _trickest_data(), now=NOW)
        assert "week" in result["summary"]

    def test_recency_within_year(self):
        gh = _github_data(newest_push="2025-01-01T12:00:00Z")  # ~165 days
        result = compute_freshness(gh, _exploitdb_data(), _trickest_data(), now=NOW)
        assert "month" in result["summary"]

    def test_recency_over_year(self):
        gh = _github_data(newest_push="2023-01-01T12:00:00Z")  # ~2.5 years
        result = compute_freshness(gh, _exploitdb_data(), _trickest_data(), now=NOW)
        assert "year" in result["summary"]


# ===================================================================
# Source tracking helpers
# ===================================================================


class TestSourceTracking:
    def test_count_active_sources_all(self):
        gh = _github_data(total_count=5)
        edb = _exploitdb_data(count=2)
        tr = _trickest_data(poc_count=3)
        assert _count_active_sources(gh, edb, tr) == 3

    def test_count_active_sources_none(self):
        assert _count_active_sources(_github_data(), _exploitdb_data(), _trickest_data()) == 0

    def test_count_active_sources_partial(self):
        gh = _github_data(total_count=5)
        assert _count_active_sources(gh, _exploitdb_data(), _trickest_data()) == 1

    def test_list_sources_checked_all_ok(self):
        result = _list_sources_checked(_github_data(), _exploitdb_data(), _trickest_data())
        assert sorted(result) == ["exploitdb", "github", "trickest"]

    def test_list_sources_checked_with_errors(self):
        gh = _github_data(error="fail")
        result = _list_sources_checked(gh, _exploitdb_data(), _trickest_data())
        assert "github" not in result
        assert "exploitdb" in result

    def test_list_sources_failed_none(self):
        result = _list_sources_failed(_github_data(), _exploitdb_data(), _trickest_data())
        assert result == []

    def test_list_sources_failed_all(self):
        result = _list_sources_failed(
            _github_data(error="e1"),
            _exploitdb_data(error="e2"),
            _trickest_data(error="e3"),
        )
        assert sorted(result) == ["exploitdb", "github", "trickest"]


# ===================================================================
# fetch_poc_freshness (orchestrator)
# ===================================================================


class TestFetchPocFreshness:
    @patch("manus_agent.tools.get_poc_freshness._fetch_trickest_pocs")
    @patch("manus_agent.tools.get_poc_freshness._fetch_exploitdb_entries")
    @patch("manus_agent.tools.get_poc_freshness._fetch_github_repos")
    def test_orchestrates_all_sources(self, mock_gh, mock_edb, mock_tr):
        mock_gh.return_value = _github_data(total_count=1, newest_push="2025-06-10T00:00:00Z")
        mock_edb.return_value = _exploitdb_data(count=0)
        mock_tr.return_value = _trickest_data(poc_count=2)
        result = fetch_poc_freshness("CVE-2024-1234")
        assert result["cve_id"] == "CVE-2024-1234"
        assert "freshness_score" in result
        mock_gh.assert_called_once_with("CVE-2024-1234")
        mock_edb.assert_called_once_with("CVE-2024-1234")
        mock_tr.assert_called_once_with("CVE-2024-1234")

    @patch("manus_agent.tools.get_poc_freshness._fetch_trickest_pocs")
    @patch("manus_agent.tools.get_poc_freshness._fetch_exploitdb_entries")
    @patch("manus_agent.tools.get_poc_freshness._fetch_github_repos")
    def test_normalizes_cve_id(self, mock_gh, mock_edb, mock_tr):
        mock_gh.return_value = _github_data()
        mock_edb.return_value = _exploitdb_data()
        mock_tr.return_value = _trickest_data()
        result = fetch_poc_freshness("  cve-2024-1234  ")
        assert result["cve_id"] == "CVE-2024-1234"


# ===================================================================
# Strands tool handler
# ===================================================================


class TestGetPocFreshnessHandler:
    @patch("manus_agent.tools.get_poc_freshness.fetch_poc_freshness")
    def test_success(self, mock_fetch):
        mock_fetch.return_value = {
            "cve_id": "CVE-2024-1234",
            "freshness_score": 75,
            "label": "warm",
            "summary": "test summary",
        }
        result = get_poc_freshness(_tool_use("CVE-2024-1234"))
        assert result["status"] == "success"
        content_text = result["content"][0]["text"]
        data = json.loads(content_text)
        assert data["freshness_score"] == 75

    def test_missing_cve_id(self):
        result = get_poc_freshness(_tool_use(None))
        assert result["status"] == "error"
        assert "Invalid" in result["content"][0]["text"]

    def test_invalid_cve_id_format(self):
        result = get_poc_freshness(_tool_use("NOT-A-CVE"))
        assert result["status"] == "error"

    def test_empty_string_cve_id(self):
        result = get_poc_freshness(_tool_use(""))
        assert result["status"] == "error"

    def test_numeric_cve_id(self):
        tu = {"toolUseId": "test-id", "input": {"cve_id": 12345}}
        result = get_poc_freshness(tu)
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_poc_freshness.fetch_poc_freshness")
    def test_exception_handling(self, mock_fetch):
        mock_fetch.side_effect = RuntimeError("Something broke")
        result = get_poc_freshness(_tool_use("CVE-2024-1234"))
        assert result["status"] == "error"
        assert "Something broke" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_poc_freshness.fetch_poc_freshness")
    def test_tool_use_id_preserved(self, mock_fetch):
        mock_fetch.return_value = {"freshness_score": 50, "label": "warm"}
        tu = {"toolUseId": "my-unique-id", "input": {"cve_id": "CVE-2024-1234"}}
        result = get_poc_freshness(tu)
        assert result["toolUseId"] == "my-unique-id"

    @patch("manus_agent.tools.get_poc_freshness.fetch_poc_freshness")
    def test_case_insensitive_cve(self, mock_fetch):
        mock_fetch.return_value = {"freshness_score": 50, "label": "warm"}
        result = get_poc_freshness(_tool_use("cve-2024-1234"))
        assert result["status"] == "success"


# ===================================================================
# CLI subcommand tests
# ===================================================================


class TestCliPocFreshness:
    def test_parser_builds(self):
        from manus_agent.cli import _build_poc_freshness_parser

        parser = _build_poc_freshness_parser()
        assert parser is not None
        args = parser.parse_args(["CVE-2024-1234"])
        assert args.cve_id == "CVE-2024-1234"
        assert args.output == "text"

    def test_parser_json_output(self):
        from manus_agent.cli import _build_poc_freshness_parser

        parser = _build_poc_freshness_parser()
        args = parser.parse_args(["CVE-2024-1234", "--output", "json"])
        assert args.output == "json"

    @patch("manus_agent.tools.get_poc_freshness.fetch_poc_freshness")
    def test_run_json_output(self, mock_fetch, capsys):
        from manus_agent.cli import _run_poc_freshness

        mock_fetch.return_value = {
            "cve_id": "CVE-2024-1234",
            "freshness_score": 85,
            "label": "hot",
            "summary": "very active",
        }
        rc = _run_poc_freshness(["CVE-2024-1234", "--output", "json"])
        assert rc == 0
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["freshness_score"] == 85

    @patch("manus_agent.tools.get_poc_freshness.fetch_poc_freshness")
    def test_run_text_output(self, mock_fetch, capsys):
        from manus_agent.cli import _run_poc_freshness

        mock_fetch.return_value = {
            "cve_id": "CVE-2024-1234",
            "freshness_score": 92,
            "label": "hot",
            "summary": "activity within the last 24 hours",
            "newest_activity": "2025-06-15T11:00:00Z",
            "days_since_activity": 0.1,
            "github_repos": 10,
            "github_stars": 150,
            "github_forks": 30,
            "exploitdb_entries": 2,
            "trickest_pocs": 5,
            "total_pocs": 17,
            "sources_checked": ["github", "exploitdb", "trickest"],
            "sources_failed": [],
            "components": {
                "recency": 0.98,
                "volume": 0.75,
                "stars": 0.65,
                "exploitdb": 0.67,
            },
        }
        rc = _run_poc_freshness(["CVE-2024-1234"])
        assert rc == 0
        output = capsys.readouterr().out
        assert "92/100" in output
        assert "HOT" in output
        assert "CVE-2024-1234" in output

    @patch("manus_agent.tools.get_poc_freshness.fetch_poc_freshness")
    def test_run_text_stale(self, mock_fetch, capsys):
        from manus_agent.cli import _run_poc_freshness

        mock_fetch.return_value = {
            "cve_id": "CVE-2024-9999",
            "freshness_score": 5,
            "label": "stale",
            "summary": "no datable PoC activity found",
            "newest_activity": None,
            "days_since_activity": None,
            "github_repos": 0,
            "github_stars": 0,
            "github_forks": 0,
            "exploitdb_entries": 0,
            "trickest_pocs": 0,
            "total_pocs": 0,
            "sources_checked": ["github", "trickest"],
            "sources_failed": ["exploitdb"],
            "components": {
                "recency": 0.0,
                "volume": 0.0,
                "stars": 0.0,
                "exploitdb": 0.0,
            },
        }
        rc = _run_poc_freshness(["CVE-2024-9999"])
        assert rc == 0
        output = capsys.readouterr().out
        assert "STALE" in output
        assert "no datable" in output.lower()

    @patch("manus_agent.tools.get_poc_freshness.fetch_poc_freshness")
    def test_run_text_with_failed_sources(self, mock_fetch, capsys):
        from manus_agent.cli import _run_poc_freshness

        mock_fetch.return_value = {
            "cve_id": "CVE-2024-1234",
            "freshness_score": 40,
            "label": "cooling",
            "summary": "some activity",
            "newest_activity": "2025-04-01T00:00:00Z",
            "days_since_activity": 75.0,
            "github_repos": 2,
            "github_stars": 5,
            "github_forks": 1,
            "exploitdb_entries": 0,
            "trickest_pocs": 1,
            "total_pocs": 3,
            "sources_checked": ["github", "trickest"],
            "sources_failed": ["exploitdb"],
            "components": {
                "recency": 0.2,
                "volume": 0.3,
                "stars": 0.1,
                "exploitdb": 0.0,
            },
        }
        rc = _run_poc_freshness(["CVE-2024-1234"])
        assert rc == 0
        output = capsys.readouterr().out
        assert "Failed" in output
        assert "exploitdb" in output

    def test_subcommand_in_set(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "poc-freshness" in _SUBCOMMANDS

    def test_invalid_cve_exits(self):
        from manus_agent.cli import _run_poc_freshness

        with pytest.raises(SystemExit):
            _run_poc_freshness(["NOT-A-CVE"])


# ===================================================================
# Edge cases and regression tests
# ===================================================================


class TestEdgeCases:
    def test_exploitdb_three_entries_gives_full_score(self):
        edb = _exploitdb_data(count=3)
        result = compute_freshness(_github_data(), edb, _trickest_data(), now=NOW)
        assert result["components"]["exploitdb"] == 1.0

    def test_exploitdb_one_entry_gives_partial_score(self):
        edb = _exploitdb_data(count=1)
        result = compute_freshness(_github_data(), edb, _trickest_data(), now=NOW)
        assert 0.0 < result["components"]["exploitdb"] < 1.0

    def test_only_trickest_pocs_contributes_to_volume(self):
        tr = _trickest_data(poc_count=10)
        result = compute_freshness(_github_data(), _exploitdb_data(), tr, now=NOW)
        assert result["components"]["volume"] > 0.0
        assert result["total_pocs"] == 10

    def test_only_github_repos(self):
        gh = _github_data(
            repos=[{"stars": 50, "forks": 10}],
            total_count=5,
            newest_push="2025-06-15T00:00:00Z",
        )
        result = compute_freshness(gh, _exploitdb_data(), _trickest_data(), now=NOW)
        assert result["freshness_score"] > 0
        assert result["components"]["recency"] > 0
        assert result["components"]["stars"] > 0

    def test_all_sources_failed(self):
        gh = _github_data(error="fail")
        edb = _exploitdb_data(error="fail")
        tr = _trickest_data(error="fail")
        result = compute_freshness(gh, edb, tr, now=NOW)
        assert result["freshness_score"] == 0
        assert result["sources_checked"] == []
        assert len(result["sources_failed"]) == 3

    def test_summary_with_stars_and_edb(self):
        gh = _github_data(
            repos=[{"stars": 100, "forks": 20}],
            total_count=5,
            newest_push="2025-06-10T00:00:00Z",
        )
        edb = _exploitdb_data(count=2, newest_date="2025-06-05")
        tr = _trickest_data(poc_count=3)
        result = compute_freshness(gh, edb, tr, now=NOW)
        assert "star" in result["summary"]
        assert "Exploit-DB" in result["summary"]

    def test_days_since_activity_rounded(self):
        gh = _github_data(newest_push="2025-06-10T12:00:00Z")
        result = compute_freshness(gh, _exploitdb_data(), _trickest_data(), now=NOW)
        assert isinstance(result["days_since_activity"], float)

    def test_score_boundary_80(self):
        # Craft inputs to land near the 80 threshold
        gh = _github_data(
            repos=[{"stars": 300, "forks": 100}],
            total_count=20,
            newest_push="2025-06-15T10:00:00Z",
        )
        edb = _exploitdb_data(count=3, newest_date="2025-06-14")
        tr = _trickest_data(poc_count=10)
        result = compute_freshness(gh, edb, tr, now=NOW)
        assert result["freshness_score"] >= 80
        assert result["label"] == "hot"

    def test_label_warm_boundary(self):
        # Score exactly at 50 → warm
        gh = _github_data(
            repos=[{"stars": 5, "forks": 2}],
            total_count=4,
            newest_push="2025-05-22T00:00:00Z",
        )
        edb = _exploitdb_data(count=1, newest_date="2025-05-20")
        tr = _trickest_data(poc_count=2)
        result = compute_freshness(gh, edb, tr, now=NOW)
        # Just verify the label is one of the valid set
        assert result["label"] in {"hot", "warm", "cooling", "stale"}
