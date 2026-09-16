"""Tests for get_poc_freshness — PoC freshness checker.

Comprehensive test suite covering:
- _http_get / _http_get_json (HTTP helper with retry)
- _github_headers (token handling)
- _search_github_repos (GitHub repo search)
- _load_exploitdb_csv / _search_exploitdb (Exploit-DB CSV index)
- _check_trickest (trickest/cve lookup)
- _parse_iso_date / _days_ago / _decay_score (date & scoring helpers)
- _compute_freshness (composite scoring engine)
- get_poc_freshness (public API — end-to-end)
- CLI: _build_poc_freshness_parser / _run_poc_freshness
"""

from __future__ import annotations

import json
import os
import urllib.error
from datetime import datetime, timezone
from io import BytesIO
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_url_response(payload, status=200):
    """Create a mock urllib response returning *payload* bytes or dict."""
    resp = MagicMock()
    if isinstance(payload, dict):
        resp.read.return_value = json.dumps(payload).encode()
    elif isinstance(payload, str):
        resp.read.return_value = payload.encode()
    else:
        resp.read.return_value = payload
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _github_repo(
    name="user/cve-poc",
    stars=10,
    forks=2,
    pushed_at="2026-09-10T12:00:00Z",
    created_at="2026-08-01T00:00:00Z",
    updated_at="2026-09-10T12:00:00Z",
    description="PoC for CVE-2024-1234",
):
    return {
        "full_name": name,
        "html_url": f"https://github.com/{name}",
        "description": description,
        "stargazers_count": stars,
        "forks_count": forks,
        "pushed_at": pushed_at,
        "created_at": created_at,
        "updated_at": updated_at,
    }


# ---------------------------------------------------------------------------
# _http_get
# ---------------------------------------------------------------------------


class TestHttpGet:
    def test_success(self):
        from manus_agent.tools.get_poc_freshness import _http_get

        mock_resp = _make_url_response(b"hello")
        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", return_value=mock_resp):
            result = _http_get("https://example.com")
        assert result == b"hello"

    def test_retry_on_429(self):
        from manus_agent.tools.get_poc_freshness import _http_get

        exc_429 = urllib.error.HTTPError("url", 429, "Too Many", {}, BytesIO(b""))
        ok_resp = _make_url_response(b"ok")

        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", side_effect=[exc_429, ok_resp]):
            with patch("manus_agent.tools.get_poc_freshness.time.sleep"):
                result = _http_get("https://example.com", max_retries=1)
        assert result == b"ok"

    def test_retry_on_503(self):
        from manus_agent.tools.get_poc_freshness import _http_get

        exc_503 = urllib.error.HTTPError("url", 503, "Unavailable", {}, BytesIO(b""))
        ok_resp = _make_url_response(b"ok")

        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", side_effect=[exc_503, ok_resp]):
            with patch("manus_agent.tools.get_poc_freshness.time.sleep"):
                result = _http_get("https://example.com", max_retries=1)
        assert result == b"ok"

    def test_no_retry_on_404(self):
        from manus_agent.tools.get_poc_freshness import _http_get

        exc_404 = urllib.error.HTTPError("url", 404, "Not Found", {}, BytesIO(b""))

        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", side_effect=exc_404):
            with pytest.raises(urllib.error.HTTPError):
                _http_get("https://example.com", max_retries=2)

    def test_retry_exhaustion_raises(self):
        from manus_agent.tools.get_poc_freshness import _http_get

        exc_429 = urllib.error.HTTPError("url", 429, "Too Many", {}, BytesIO(b""))

        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", side_effect=exc_429):
            with patch("manus_agent.tools.get_poc_freshness.time.sleep"):
                with pytest.raises(urllib.error.HTTPError):
                    _http_get("https://example.com", max_retries=1)

    def test_url_error_retry(self):
        from manus_agent.tools.get_poc_freshness import _http_get

        url_err = urllib.error.URLError("timeout")
        ok_resp = _make_url_response(b"ok")

        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", side_effect=[url_err, ok_resp]):
            with patch("manus_agent.tools.get_poc_freshness.time.sleep"):
                result = _http_get("https://example.com", max_retries=1)
        assert result == b"ok"

    def test_custom_headers(self):
        from manus_agent.tools.get_poc_freshness import _http_get

        mock_resp = _make_url_response(b"ok")
        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", return_value=mock_resp):
            with patch("manus_agent.tools.get_poc_freshness.urllib.request.Request") as mock_req:
                _http_get("https://example.com", headers={"X-Custom": "val"})
                # Just verify it was called
                assert mock_req.called


class TestHttpGetJson:
    def test_parses_json(self):
        from manus_agent.tools.get_poc_freshness import _http_get_json

        mock_resp = _make_url_response({"key": "value"})
        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", return_value=mock_resp):
            result = _http_get_json("https://example.com")
        assert result == {"key": "value"}


# ---------------------------------------------------------------------------
# _github_headers
# ---------------------------------------------------------------------------


class TestGithubHeaders:
    def test_with_token(self):
        from manus_agent.tools.get_poc_freshness import _github_headers

        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test123"}):
            hdrs = _github_headers()
        assert hdrs["Authorization"] == "Bearer ghp_test123"
        assert "Accept" in hdrs

    def test_without_token(self):
        from manus_agent.tools.get_poc_freshness import _github_headers

        with patch.dict(os.environ, {}, clear=True):
            hdrs = _github_headers()
        assert "Authorization" not in hdrs
        assert "Accept" in hdrs


# ---------------------------------------------------------------------------
# _search_github_repos
# ---------------------------------------------------------------------------


class TestSearchGithubRepos:
    def test_success_with_results(self):
        from manus_agent.tools.get_poc_freshness import _search_github_repos

        github_response = {
            "items": [
                {
                    "full_name": "attacker/CVE-2024-1234-poc",
                    "html_url": "https://github.com/attacker/CVE-2024-1234-poc",
                    "description": "PoC exploit",
                    "stargazers_count": 42,
                    "forks_count": 7,
                    "pushed_at": "2026-09-10T12:00:00Z",
                    "created_at": "2026-08-01T00:00:00Z",
                    "updated_at": "2026-09-10T12:00:00Z",
                },
            ]
        }
        mock_resp = _make_url_response(github_response)
        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", return_value=mock_resp):
            repos = _search_github_repos("CVE-2024-1234")
        assert len(repos) == 1
        assert repos[0]["full_name"] == "attacker/CVE-2024-1234-poc"
        assert repos[0]["stargazers_count"] == 42

    def test_empty_results(self):
        from manus_agent.tools.get_poc_freshness import _search_github_repos

        mock_resp = _make_url_response({"items": []})
        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", return_value=mock_resp):
            repos = _search_github_repos("CVE-2099-99999")
        assert repos == []

    def test_http_error_returns_empty(self):
        from manus_agent.tools.get_poc_freshness import _search_github_repos

        exc = urllib.error.HTTPError("url", 403, "Forbidden", {}, BytesIO(b""))
        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", side_effect=exc):
            repos = _search_github_repos("CVE-2024-1234")
        assert repos == []

    def test_url_error_returns_empty(self):
        from manus_agent.tools.get_poc_freshness import _search_github_repos

        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            repos = _search_github_repos("CVE-2024-1234")
        assert repos == []

    def test_unexpected_error_returns_empty(self):
        from manus_agent.tools.get_poc_freshness import _search_github_repos

        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", side_effect=RuntimeError("oops")):
            repos = _search_github_repos("CVE-2024-1234")
        assert repos == []

    def test_truncates_long_description(self):
        from manus_agent.tools.get_poc_freshness import _search_github_repos

        github_response = {
            "items": [
                {
                    "full_name": "user/repo",
                    "html_url": "https://github.com/user/repo",
                    "description": "A" * 500,
                    "stargazers_count": 1,
                    "forks_count": 0,
                    "pushed_at": "2026-01-01T00:00:00Z",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                },
            ]
        }
        mock_resp = _make_url_response(github_response)
        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", return_value=mock_resp):
            repos = _search_github_repos("CVE-2024-1234")
        assert len(repos[0]["description"]) == 200

    def test_handles_null_description(self):
        from manus_agent.tools.get_poc_freshness import _search_github_repos

        github_response = {
            "items": [
                {
                    "full_name": "user/repo",
                    "html_url": "https://github.com/user/repo",
                    "description": None,
                    "stargazers_count": 0,
                    "forks_count": 0,
                    "pushed_at": "",
                    "created_at": "",
                    "updated_at": "",
                },
            ]
        }
        mock_resp = _make_url_response(github_response)
        with patch("manus_agent.tools.get_poc_freshness.urllib.request.urlopen", return_value=mock_resp):
            repos = _search_github_repos("CVE-2024-1234")
        assert repos[0]["description"] == ""


# ---------------------------------------------------------------------------
# _load_exploitdb_csv / _search_exploitdb
# ---------------------------------------------------------------------------


class TestSearchExploitdb:
    def test_match_found(self):
        from manus_agent.tools.get_poc_freshness import _search_exploitdb

        fake_rows = [
            {"id": "12345", "description": "Buffer overflow", "date_published": "2026-08-15", "platform": "linux", "type": "local", "codes": "CVE-2024-1234;OSVDB-99999"},
        ]
        with patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv", return_value=fake_rows):
            results = _search_exploitdb("CVE-2024-1234")
        assert len(results) == 1
        assert results[0]["id"] == "12345"
        assert "exploit-db.com" in results[0]["url"]

    def test_no_match(self):
        from manus_agent.tools.get_poc_freshness import _search_exploitdb

        fake_rows = [
            {"id": "99999", "description": "Other vuln", "date_published": "2026-01-01", "platform": "windows", "type": "remote", "codes": "CVE-2099-99999"},
        ]
        with patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv", return_value=fake_rows):
            results = _search_exploitdb("CVE-2024-1234")
        assert results == []

    def test_multiple_matches(self):
        from manus_agent.tools.get_poc_freshness import _search_exploitdb

        fake_rows = [
            {"id": "111", "description": "Exploit A", "date_published": "2026-07-01", "platform": "linux", "type": "remote", "codes": "CVE-2024-1234"},
            {"id": "222", "description": "Exploit B", "date_published": "2026-08-01", "platform": "linux", "type": "local", "codes": "CVE-2024-1234;CVE-2024-5678"},
            {"id": "333", "description": "Unrelated", "date_published": "2026-01-01", "platform": "windows", "type": "dos", "codes": "CVE-2099-1111"},
        ]
        with patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv", return_value=fake_rows):
            results = _search_exploitdb("CVE-2024-1234")
        assert len(results) == 2

    def test_case_insensitive_match(self):
        from manus_agent.tools.get_poc_freshness import _search_exploitdb

        fake_rows = [
            {"id": "444", "description": "Exploit", "date_published": "2026-06-01", "platform": "multi", "type": "webapps", "codes": "cve-2024-1234"},
        ]
        with patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv", return_value=fake_rows):
            results = _search_exploitdb("CVE-2024-1234")
        assert len(results) == 1

    def test_empty_csv(self):
        from manus_agent.tools.get_poc_freshness import _search_exploitdb

        with patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv", return_value=[]):
            results = _search_exploitdb("CVE-2024-1234")
        assert results == []

    def test_truncates_long_description(self):
        from manus_agent.tools.get_poc_freshness import _search_exploitdb

        fake_rows = [
            {"id": "555", "description": "X" * 500, "date_published": "2026-01-01", "platform": "linux", "type": "local", "codes": "CVE-2024-1234"},
        ]
        with patch("manus_agent.tools.get_poc_freshness._load_exploitdb_csv", return_value=fake_rows):
            results = _search_exploitdb("CVE-2024-1234")
        assert len(results[0]["description"]) == 200


class TestLoadExploitdbCsv:
    def test_uses_cache_when_fresh(self):
        import time as _time

        from manus_agent.tools.get_poc_freshness import _load_exploitdb_csv

        fake_stat = MagicMock()
        fake_stat.st_mtime = _time.time()  # fresh cache

        csv_content = "id,description,codes\n1,test,CVE-2024-1234\n"

        with patch("manus_agent.tools.get_poc_freshness.os.stat", return_value=fake_stat):
            with patch("builtins.open", mock_open(read_data=csv_content)):
                rows = _load_exploitdb_csv()
        assert len(rows) == 1

    def test_fetches_when_cache_stale(self):
        from manus_agent.tools.get_poc_freshness import _load_exploitdb_csv

        # Cache doesn't exist
        csv_content = "id,description,codes\n1,test,CVE-2024-1234\n"
        csv_bytes = csv_content.encode()

        with patch("manus_agent.tools.get_poc_freshness.os.stat", side_effect=FileNotFoundError):
            with patch("manus_agent.tools.get_poc_freshness._http_get", return_value=csv_bytes):
                with patch("builtins.open", mock_open(read_data=csv_content)):
                    rows = _load_exploitdb_csv()
        assert len(rows) == 1

    def test_returns_empty_on_fetch_failure(self):
        from manus_agent.tools.get_poc_freshness import _load_exploitdb_csv

        with patch("manus_agent.tools.get_poc_freshness.os.stat", side_effect=FileNotFoundError):
            with patch("manus_agent.tools.get_poc_freshness._http_get", side_effect=RuntimeError("network down")):
                rows = _load_exploitdb_csv()
        assert rows == []


# ---------------------------------------------------------------------------
# _check_trickest
# ---------------------------------------------------------------------------


class TestCheckTrickest:
    def test_found_with_pocs(self):
        from manus_agent.tools.get_poc_freshness import _check_trickest

        md = """### Description
Buffer overflow in libfoo.

### POC
#### Reference
- https://exploit-db.com/exploits/12345
#### Github
- https://github.com/attacker/cve-2024-1234-poc
"""
        with patch("manus_agent.tools.get_poc_freshness._http_get", return_value=md.encode()):
            result = _check_trickest("CVE-2024-1234")
        assert result["found"] is True
        assert result["poc_count"] == 2
        assert len(result["poc_urls"]) == 2

    def test_not_found_404(self):
        from manus_agent.tools.get_poc_freshness import _check_trickest

        exc = urllib.error.HTTPError("url", 404, "Not Found", {}, BytesIO(b""))
        with patch("manus_agent.tools.get_poc_freshness._http_get", side_effect=exc):
            result = _check_trickest("CVE-2099-99999")
        assert result["found"] is False
        assert result["poc_count"] == 0

    def test_server_error_returns_not_found(self):
        from manus_agent.tools.get_poc_freshness import _check_trickest

        exc = urllib.error.HTTPError("url", 500, "Server Error", {}, BytesIO(b""))
        with patch("manus_agent.tools.get_poc_freshness._http_get", side_effect=exc):
            result = _check_trickest("CVE-2024-1234")
        assert result["found"] is False

    def test_invalid_cve_format(self):
        from manus_agent.tools.get_poc_freshness import _check_trickest

        result = _check_trickest("not-a-cve")
        assert result["found"] is False

    def test_found_no_pocs(self):
        from manus_agent.tools.get_poc_freshness import _check_trickest

        md = """### Description
Some vulnerability with no PoC links.
"""
        with patch("manus_agent.tools.get_poc_freshness._http_get", return_value=md.encode()):
            result = _check_trickest("CVE-2024-1234")
        assert result["found"] is True
        assert result["poc_count"] == 0

    def test_caps_poc_urls_at_20(self):
        from manus_agent.tools.get_poc_freshness import _check_trickest

        urls = "\n".join(f"- https://github.com/user/repo{i}" for i in range(30))
        md = f"### POC\n#### Github\n{urls}\n"
        with patch("manus_agent.tools.get_poc_freshness._http_get", return_value=md.encode()):
            result = _check_trickest("CVE-2024-1234")
        assert len(result["poc_urls"]) <= 20

    def test_network_error_returns_not_found(self):
        from manus_agent.tools.get_poc_freshness import _check_trickest

        with patch("manus_agent.tools.get_poc_freshness._http_get", side_effect=RuntimeError("timeout")):
            result = _check_trickest("CVE-2024-1234")
        assert result["found"] is False


# ---------------------------------------------------------------------------
# _parse_iso_date / _days_ago / _decay_score
# ---------------------------------------------------------------------------


class TestParseIsoDate:
    def test_github_format(self):
        from manus_agent.tools.get_poc_freshness import _parse_iso_date

        dt = _parse_iso_date("2026-09-10T12:00:00Z")
        assert dt is not None
        assert dt.year == 2026
        assert dt.month == 9
        assert dt.day == 10

    def test_date_only(self):
        from manus_agent.tools.get_poc_freshness import _parse_iso_date

        dt = _parse_iso_date("2026-08-15")
        assert dt is not None
        assert dt.year == 2026

    def test_empty_string(self):
        from manus_agent.tools.get_poc_freshness import _parse_iso_date

        assert _parse_iso_date("") is None

    def test_none_input(self):
        from manus_agent.tools.get_poc_freshness import _parse_iso_date

        # Empty string fallback
        assert _parse_iso_date("") is None

    def test_invalid_format(self):
        from manus_agent.tools.get_poc_freshness import _parse_iso_date

        assert _parse_iso_date("not-a-date") is None

    def test_iso_with_offset(self):
        from manus_agent.tools.get_poc_freshness import _parse_iso_date

        dt = _parse_iso_date("2026-09-10T12:00:00+00:00")
        assert dt is not None
        assert dt.tzinfo is not None


class TestDaysAgo:
    def test_same_day(self):
        from manus_agent.tools.get_poc_freshness import _days_ago

        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        assert _days_ago(now, now) == 0.0

    def test_one_day_ago(self):
        from manus_agent.tools.get_poc_freshness import _days_ago

        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        yesterday = datetime(2026, 9, 15, tzinfo=timezone.utc)
        assert _days_ago(yesterday, now) == pytest.approx(1.0)

    def test_future_clamps_to_zero(self):
        from manus_agent.tools.get_poc_freshness import _days_ago

        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        future = datetime(2026, 9, 17, tzinfo=timezone.utc)
        assert _days_ago(future, now) == 0.0


class TestDecayScore:
    def test_zero_days(self):
        from manus_agent.tools.get_poc_freshness import _decay_score

        assert _decay_score(0, base=100.0) == pytest.approx(100.0)

    def test_positive_days_decays(self):
        from manus_agent.tools.get_poc_freshness import _decay_score

        score = _decay_score(30, base=100.0)
        assert 0 < score < 100

    def test_large_days_near_zero(self):
        from manus_agent.tools.get_poc_freshness import _decay_score

        score = _decay_score(365, base=100.0)
        assert score < 1.0

    def test_custom_base(self):
        from manus_agent.tools.get_poc_freshness import _decay_score

        score = _decay_score(0, base=50.0)
        assert score == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# _compute_freshness
# ---------------------------------------------------------------------------


class TestComputeFreshness:
    def test_no_activity(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        result = _compute_freshness([], [], {"found": False, "poc_count": 0})
        assert result["freshness_score"] == 0
        assert result["label"] == "None"
        assert result["most_recent_activity"] is None

    def test_github_only_recent(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        repos = [_github_repo(pushed_at="2026-09-15T12:00:00Z", stars=50, forks=10)]
        result = _compute_freshness(repos, [], {"found": False, "poc_count": 0}, now=now)
        assert result["freshness_score"] > 40
        assert result["label"] in ("Active", "Recent", "Moderate")

    def test_exploitdb_only(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        edb = [{"date_published": "2026-09-10"}]
        result = _compute_freshness([], edb, {"found": False, "poc_count": 0}, now=now)
        assert result["freshness_score"] > 10
        assert result["signals"]["exploit_db"]["entries_found"] == 1

    def test_trickest_only(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        result = _compute_freshness([], [], {"found": True, "poc_count": 5})
        assert result["freshness_score"] > 0
        assert result["signals"]["trickest"]["indexed"] is True

    def test_all_sources_combined(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        repos = [_github_repo(pushed_at="2026-09-15T00:00:00Z", stars=100, forks=20)]
        edb = [{"date_published": "2026-09-14"}]
        trickest = {"found": True, "poc_count": 8}
        result = _compute_freshness(repos, edb, trickest, now=now)
        assert result["freshness_score"] > 70
        assert result["most_recent_activity"] is not None

    def test_stale_activity(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        # Activity from ~200 days ago
        repos = [_github_repo(pushed_at="2026-03-01T00:00:00Z", stars=2, forks=0)]
        result = _compute_freshness(repos, [], {"found": False, "poc_count": 0}, now=now)
        assert result["freshness_score"] < 20
        assert result["label"] in ("Stale", "None")

    def test_score_capped_at_100(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        # Max everything: many recent repos + EDB + trickest
        repos = [_github_repo(pushed_at="2026-09-16T00:00:00Z", stars=1000, forks=500)] * 10
        edb = [{"date_published": "2026-09-16"}] * 5
        trickest = {"found": True, "poc_count": 100}
        result = _compute_freshness(repos, edb, trickest, now=now)
        assert result["freshness_score"] == 100

    def test_github_popularity_bonus(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        # Low stars
        repos_low = [_github_repo(pushed_at="2026-09-15T00:00:00Z", stars=1, forks=0)]
        result_low = _compute_freshness(repos_low, [], {"found": False, "poc_count": 0}, now=now)
        # High stars
        repos_high = [_github_repo(pushed_at="2026-09-15T00:00:00Z", stars=500, forks=100)]
        result_high = _compute_freshness(repos_high, [], {"found": False, "poc_count": 0}, now=now)
        assert result_high["freshness_score"] > result_low["freshness_score"]

    def test_multiple_exploitdb_entries_bonus(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        edb_one = [{"date_published": "2026-09-10"}]
        edb_many = [
            {"date_published": "2026-09-10"},
            {"date_published": "2026-09-08"},
            {"date_published": "2026-09-05"},
        ]
        r1 = _compute_freshness([], edb_one, {"found": False, "poc_count": 0}, now=now)
        r2 = _compute_freshness([], edb_many, {"found": False, "poc_count": 0}, now=now)
        # More entries should give a higher or equal score
        assert r2["freshness_score"] >= r1["freshness_score"]

    def test_trickest_poc_count_bonus(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        r_few = _compute_freshness([], [], {"found": True, "poc_count": 1})
        r_many = _compute_freshness([], [], {"found": True, "poc_count": 20})
        assert r_many["freshness_score"] >= r_few["freshness_score"]

    def test_days_since_last_activity(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        repos = [_github_repo(pushed_at="2026-09-10T00:00:00Z")]
        result = _compute_freshness(repos, [], {"found": False, "poc_count": 0}, now=now)
        assert result["days_since_last_activity"] == pytest.approx(6.0, abs=0.5)

    def test_label_active(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        repos = [_github_repo(pushed_at="2026-09-16T00:00:00Z", stars=200, forks=50)] * 5
        edb = [{"date_published": "2026-09-16"}] * 3
        trickest = {"found": True, "poc_count": 15}
        result = _compute_freshness(repos, edb, trickest, now=now)
        assert result["freshness_score"] == 100
        assert result["label"] == "Active"

    def test_label_none(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        result = _compute_freshness([], [], {"found": False, "poc_count": 0})
        assert result["label"] == "None"

    def test_unparseable_dates_ignored(self):
        from manus_agent.tools.get_poc_freshness import _compute_freshness

        repos = [_github_repo(pushed_at="invalid-date")]
        edb = [{"date_published": "not-a-date"}]
        result = _compute_freshness(repos, edb, {"found": False, "poc_count": 0})
        # Should not crash, just treat as no activity dates
        assert result["most_recent_activity"] is None


# ---------------------------------------------------------------------------
# get_poc_freshness (public API)
# ---------------------------------------------------------------------------


class TestGetPocFreshness:
    def test_invalid_cve_id(self):
        from manus_agent.tools.get_poc_freshness import get_poc_freshness

        result = get_poc_freshness("not-a-cve")
        assert "error" in result
        assert "Invalid" in result["error"]

    def test_valid_cve_all_sources(self):
        from manus_agent.tools.get_poc_freshness import get_poc_freshness

        github_repos = [{"full_name": "u/r", "html_url": "https://github.com/u/r", "description": "poc", "stargazers_count": 10, "forks_count": 2, "pushed_at": "2026-09-10T00:00:00Z", "created_at": "2026-08-01T00:00:00Z", "updated_at": "2026-09-10T00:00:00Z"}]
        edb_results = [{"id": "1", "url": "https://exploit-db.com/exploits/1", "description": "exploit", "date_published": "2026-09-05", "platform": "linux", "type": "local"}]
        trickest_result = {"found": True, "poc_count": 1, "poc_urls": ["https://github.com/user/poc"]}

        with patch("manus_agent.tools.get_poc_freshness._search_github_repos", return_value=github_repos):
            with patch("manus_agent.tools.get_poc_freshness._search_exploitdb", return_value=edb_results):
                with patch("manus_agent.tools.get_poc_freshness._check_trickest", return_value=trickest_result):
                    result = get_poc_freshness("CVE-2024-1234")

        assert result["cve_id"] == "CVE-2024-1234"
        assert "freshness_score" in result
        assert "label" in result
        assert "signals" in result
        assert "github_repos" in result
        assert "exploitdb_entries" in result
        assert "trickest" in result
        assert "error" not in result

    def test_normalises_cve_id(self):
        from manus_agent.tools.get_poc_freshness import get_poc_freshness

        with patch("manus_agent.tools.get_poc_freshness._search_github_repos", return_value=[]):
            with patch("manus_agent.tools.get_poc_freshness._search_exploitdb", return_value=[]):
                with patch("manus_agent.tools.get_poc_freshness._check_trickest", return_value={"found": False, "poc_count": 0, "poc_urls": []}):
                    result = get_poc_freshness("  cve-2024-1234  ")
        assert result["cve_id"] == "CVE-2024-1234"

    def test_no_activity_found(self):
        from manus_agent.tools.get_poc_freshness import get_poc_freshness

        with patch("manus_agent.tools.get_poc_freshness._search_github_repos", return_value=[]):
            with patch("manus_agent.tools.get_poc_freshness._search_exploitdb", return_value=[]):
                with patch("manus_agent.tools.get_poc_freshness._check_trickest", return_value={"found": False, "poc_count": 0, "poc_urls": []}):
                    result = get_poc_freshness("CVE-2099-99999")

        assert result["freshness_score"] == 0
        assert result["label"] == "None"
        assert result["most_recent_activity"] is None

    def test_whitespace_in_cve_id(self):
        from manus_agent.tools.get_poc_freshness import get_poc_freshness

        result = get_poc_freshness("   ")
        assert "error" in result


# ---------------------------------------------------------------------------
# CLI: _build_poc_freshness_parser
# ---------------------------------------------------------------------------


class TestBuildPocFreshnessParser:
    def test_parses_cve_id(self):
        from manus_agent.cli import _build_poc_freshness_parser

        parser = _build_poc_freshness_parser()
        args = parser.parse_args(["CVE-2024-1234"])
        assert args.cve_id == "CVE-2024-1234"
        assert args.output == "text"

    def test_json_output(self):
        from manus_agent.cli import _build_poc_freshness_parser

        parser = _build_poc_freshness_parser()
        args = parser.parse_args(["CVE-2024-1234", "--output", "json"])
        assert args.output == "json"

    def test_missing_cve_id_errors(self):
        from manus_agent.cli import _build_poc_freshness_parser

        parser = _build_poc_freshness_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# CLI: _run_poc_freshness
# ---------------------------------------------------------------------------


class TestRunPocFreshness:
    def test_json_output(self, capsys):
        from manus_agent.cli import _run_poc_freshness

        mock_result = {
            "cve_id": "CVE-2024-1234",
            "freshness_score": 75,
            "label": "Recent",
            "most_recent_activity": "2026-09-10T00:00:00+00:00",
            "days_since_last_activity": 6.0,
            "signals": {},
            "github_repos": [],
            "exploitdb_entries": [],
            "trickest": {"found": False, "poc_count": 0, "poc_urls": []},
        }
        with patch("manus_agent.tools.get_poc_freshness.get_poc_freshness", return_value=mock_result):
            rc = _run_poc_freshness(["CVE-2024-1234", "--output", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["freshness_score"] == 75

    def test_text_output(self, capsys):
        from manus_agent.cli import _run_poc_freshness

        mock_result = {
            "cve_id": "CVE-2024-1234",
            "freshness_score": 42,
            "label": "Moderate",
            "most_recent_activity": "2026-09-01T00:00:00+00:00",
            "days_since_last_activity": 15.0,
            "signals": {
                "github": {"repos_found": 2, "total_stars": 20, "total_forks": 5, "most_recent_push": "2026-09-01T00:00:00+00:00", "score_contribution": 25.0},
                "exploit_db": {"entries_found": 1, "most_recent_date": "2026-08-20T00:00:00+00:00", "score_contribution": 12.0},
                "trickest": {"indexed": True, "poc_count": 3, "score_contribution": 10.0},
            },
            "github_repos": [
                {"full_name": "attacker/poc", "stargazers_count": 15, "pushed_at": "2026-09-01T00:00:00Z"},
            ],
            "exploitdb_entries": [],
            "trickest": {"found": True, "poc_count": 3, "poc_urls": []},
        }
        with patch("manus_agent.tools.get_poc_freshness.get_poc_freshness", return_value=mock_result):
            rc = _run_poc_freshness(["CVE-2024-1234"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Moderate" in out
        assert "42/100" in out

    def test_error_result(self):
        from manus_agent.cli import _run_poc_freshness

        # "BAD" is not a valid CVE format, parser.error raises SystemExit
        with pytest.raises(SystemExit):
            _run_poc_freshness(["BAD"])

    def test_invalid_cve_format(self):
        from manus_agent.cli import _run_poc_freshness

        with pytest.raises(SystemExit):
            _run_poc_freshness(["not-a-cve"])

    def test_text_no_activity(self, capsys):
        from manus_agent.cli import _run_poc_freshness

        mock_result = {
            "cve_id": "CVE-2099-99999",
            "freshness_score": 0,
            "label": "None",
            "most_recent_activity": None,
            "days_since_last_activity": None,
            "signals": {
                "github": {"repos_found": 0, "total_stars": 0, "total_forks": 0, "most_recent_push": None, "score_contribution": 0.0},
                "exploit_db": {"entries_found": 0, "most_recent_date": None, "score_contribution": 0.0},
                "trickest": {"indexed": False, "poc_count": 0, "score_contribution": 0.0},
            },
            "github_repos": [],
            "exploitdb_entries": [],
            "trickest": {"found": False, "poc_count": 0, "poc_urls": []},
        }
        with patch("manus_agent.tools.get_poc_freshness.get_poc_freshness", return_value=mock_result):
            rc = _run_poc_freshness(["CVE-2099-99999"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No PoC activity found" in out

    def test_text_with_github_repos(self, capsys):
        from manus_agent.cli import _run_poc_freshness

        mock_result = {
            "cve_id": "CVE-2024-1234",
            "freshness_score": 80,
            "label": "Recent",
            "most_recent_activity": "2026-09-15T00:00:00+00:00",
            "days_since_last_activity": 1.0,
            "signals": {
                "github": {"repos_found": 3, "total_stars": 50, "total_forks": 10, "most_recent_push": "2026-09-15T00:00:00+00:00", "score_contribution": 55.0},
                "exploit_db": {"entries_found": 0, "most_recent_date": None, "score_contribution": 0.0},
                "trickest": {"indexed": True, "poc_count": 2, "score_contribution": 8.0},
            },
            "github_repos": [
                {"full_name": "a/poc1", "stargazers_count": 30, "pushed_at": "2026-09-15T00:00:00Z", "forks_count": 5},
                {"full_name": "b/poc2", "stargazers_count": 15, "pushed_at": "2026-09-14T00:00:00Z", "forks_count": 3},
                {"full_name": "c/poc3", "stargazers_count": 5, "pushed_at": "2026-09-10T00:00:00Z", "forks_count": 2},
            ],
            "exploitdb_entries": [],
            "trickest": {"found": True, "poc_count": 2, "poc_urls": []},
        }
        with patch("manus_agent.tools.get_poc_freshness.get_poc_freshness", return_value=mock_result):
            rc = _run_poc_freshness(["CVE-2024-1234"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Top GitHub PoC repos" in out
        assert "a/poc1" in out


# ---------------------------------------------------------------------------
# CLI dispatch integration
# ---------------------------------------------------------------------------


class TestCliDispatch:
    def test_poc_freshness_in_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "poc-freshness" in _SUBCOMMANDS
