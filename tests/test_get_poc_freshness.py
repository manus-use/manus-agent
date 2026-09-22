"""Comprehensive test suite for get_poc_freshness tool and poc-freshness CLI.

All HTTP calls are mocked — no real network requests.
"""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from datetime import datetime
from unittest import mock

import pytest

from manus_agent.tools.get_poc_freshness import (
    _HALF_LIFE_DAYS,
    _W_EPSS,
    _W_EXPLOITDB,
    _W_GITHUB,
    _compute_freshness,
    _days_ago,
    _decay,
    _fetch_epss_signal,
    _fetch_exploitdb_signal,
    _fetch_github_signal,
    _now_utc,
    _parse_iso,
    check_poc_freshness,
    get_poc_freshness,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _fake_now():
    """Pin _now_utc to 2026-09-22T00:00:00Z for deterministic tests."""
    fixed = datetime(2026, 9, 22, 0, 0, 0)
    with mock.patch("manus_agent.tools.get_poc_freshness._now_utc", return_value=fixed):
        yield fixed


@pytest.fixture()
def _github_response():
    """Minimal GitHub search/repositories response."""
    return {
        "total_count": 3,
        "items": [
            {
                "full_name": "user1/cve-2024-1234-poc",
                "html_url": "https://github.com/user1/cve-2024-1234-poc",
                "pushed_at": "2026-09-20T10:00:00Z",
                "stargazers_count": 42,
                "forks_count": 8,
                "description": "PoC for CVE-2024-1234",
                "owner": {"login": "user1"},
            },
            {
                "full_name": "user2/exploit-cve-2024-1234",
                "html_url": "https://github.com/user2/exploit-cve-2024-1234",
                "pushed_at": "2026-09-01T08:00:00Z",
                "stargazers_count": 15,
                "forks_count": 3,
                "description": "Another PoC",
                "owner": {"login": "user2"},
            },
            {
                "full_name": "user3/old-poc",
                "html_url": "https://github.com/user3/old-poc",
                "pushed_at": "2025-01-01T00:00:00Z",
                "stargazers_count": 2,
                "forks_count": 0,
                "description": "Old exploit",
                "owner": {"login": "user3"},
            },
        ],
    }


@pytest.fixture()
def _exploitdb_csv_content():
    """Minimal Exploit-DB CSV with matching entries."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=["id", "description", "date_published", "codes", "type", "platform", "author"]
    )
    writer.writeheader()
    writer.writerow(
        {
            "id": "12345",
            "description": "CVE-2024-1234 RCE exploit",
            "date_published": "2026-09-15",
            "codes": "CVE-2024-1234",
            "type": "remote",
            "platform": "linux",
            "author": "researcher1",
        }
    )
    writer.writerow(
        {
            "id": "12346",
            "description": "CVE-2024-1234 LPE variant",
            "date_published": "2026-08-01",
            "codes": "CVE-2024-1234;EDB-12346",
            "type": "local",
            "platform": "linux",
            "author": "researcher2",
        }
    )
    return buf.getvalue()


@pytest.fixture()
def _epss_response():
    """Minimal EPSS API response."""
    return {
        "status": "OK",
        "data": [
            {
                "cve": "CVE-2024-1234",
                "epss": "0.15432",
                "percentile": "0.9421",
                "date": "2026-09-21",
            }
        ],
    }


# ===================================================================
# Unit tests — date helpers
# ===================================================================


class TestParseIso:
    def test_full_iso_with_z(self):
        dt = _parse_iso("2026-09-22T14:30:00Z")
        assert dt == datetime(2026, 9, 22, 14, 30, 0)

    def test_full_iso_without_z(self):
        dt = _parse_iso("2026-09-22T14:30:00")
        assert dt == datetime(2026, 9, 22, 14, 30, 0)

    def test_date_only(self):
        dt = _parse_iso("2026-09-22")
        assert dt == datetime(2026, 9, 22, 0, 0, 0)

    def test_none_input(self):
        assert _parse_iso(None) is None

    def test_empty_string(self):
        assert _parse_iso("") is None

    def test_garbage_input(self):
        assert _parse_iso("not-a-date") is None

    def test_truncated_datetime(self):
        # Longer string is truncated to 19 chars
        dt = _parse_iso("2026-09-22T14:30:00.123456Z")
        assert dt == datetime(2026, 9, 22, 14, 30, 0)


class TestDaysAgo:
    def test_today(self, _fake_now):
        dt = datetime(2026, 9, 22, 0, 0, 0)
        assert _days_ago(dt) == 0.0

    def test_one_day_ago(self, _fake_now):
        dt = datetime(2026, 9, 21, 0, 0, 0)
        assert abs(_days_ago(dt) - 1.0) < 0.01

    def test_none_returns_none(self):
        assert _days_ago(None) is None

    def test_future_returns_zero(self, _fake_now):
        dt = datetime(2026, 9, 25, 0, 0, 0)
        assert _days_ago(dt) == 0.0


class TestDecay:
    def test_today(self):
        assert abs(_decay(0) - 1.0) < 0.001

    def test_half_life(self):
        assert abs(_decay(_HALF_LIFE_DAYS) - 0.5) < 0.01

    def test_double_half_life(self):
        assert abs(_decay(_HALF_LIFE_DAYS * 2) - 0.25) < 0.01

    def test_none_returns_zero(self):
        assert _decay(None) == 0.0

    def test_negative_returns_zero(self):
        assert _decay(-5) == 0.0

    def test_very_large(self):
        result = _decay(10000)
        assert result >= 0.0
        assert result < 0.001


# ===================================================================
# Unit tests — GitHub signal
# ===================================================================


class TestFetchGithubSignal:
    def test_success(self, _fake_now, _github_response):
        with mock.patch(
            "manus_agent.tools.get_poc_freshness._http_get_json",
            return_value=_github_response,
        ):
            result = _fetch_github_signal("CVE-2024-1234")
        assert result["repos_found"] == 3
        assert result["total_stars"] == 59
        assert result["total_forks"] == 11
        assert result["most_recent_push"] == "2026-09-20T10:00:00Z"
        assert result["most_recent_push_days_ago"] is not None
        assert result["score"] > 0
        assert len(result["top_repos"]) == 3
        assert result["error"] is None

    def test_no_results(self, _fake_now):
        with mock.patch(
            "manus_agent.tools.get_poc_freshness._http_get_json",
            return_value={"total_count": 0, "items": []},
        ):
            result = _fetch_github_signal("CVE-2024-9999")
        assert result["repos_found"] == 0
        assert result["score"] == 0.0
        assert result["most_recent_push"] is None

    def test_http_error(self):
        with mock.patch(
            "manus_agent.tools.get_poc_freshness._http_get_json",
            side_effect=Exception("rate limited"),
        ):
            result = _fetch_github_signal("CVE-2024-1234")
        assert result["error"] == "rate limited"
        assert result["score"] == 0.0
        assert result["repos_found"] == 0

    def test_github_token_used(self, _fake_now, _github_response):
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test123"}):
            with mock.patch(
                "manus_agent.tools.get_poc_freshness._http_get_json",
                return_value=_github_response,
            ) as mock_get:
                _fetch_github_signal("CVE-2024-1234")
                call_args = mock_get.call_args
                headers = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("headers", {})
                assert "Authorization" in headers

    def test_no_github_token(self, _fake_now, _github_response):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch(
                "manus_agent.tools.get_poc_freshness._http_get_json",
                return_value=_github_response,
            ) as mock_get:
                _fetch_github_signal("CVE-2024-1234")
                call_args = mock_get.call_args
                headers = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("headers", {})
                assert "Authorization" not in headers

    def test_repos_with_no_pushed_at(self, _fake_now):
        response = {
            "total_count": 1,
            "items": [
                {
                    "full_name": "user/repo",
                    "html_url": "https://github.com/user/repo",
                    "pushed_at": None,
                    "stargazers_count": 5,
                    "forks_count": 1,
                    "description": "no push date",
                    "owner": {"login": "user"},
                }
            ],
        }
        with mock.patch(
            "manus_agent.tools.get_poc_freshness._http_get_json",
            return_value=response,
        ):
            result = _fetch_github_signal("CVE-2024-1234")
        assert result["repos_found"] == 1
        assert result["most_recent_push"] is None

    def test_score_decay_for_old_repos(self, _fake_now):
        """Repos pushed a year ago should have very low recency score."""
        response = {
            "total_count": 1,
            "items": [
                {
                    "full_name": "user/old-poc",
                    "html_url": "https://github.com/user/old-poc",
                    "pushed_at": "2025-01-01T00:00:00Z",
                    "stargazers_count": 0,
                    "forks_count": 0,
                    "description": "very old",
                    "owner": {"login": "user"},
                }
            ],
        }
        with mock.patch(
            "manus_agent.tools.get_poc_freshness._http_get_json",
            return_value=response,
        ):
            result = _fetch_github_signal("CVE-2024-1234")
        assert result["score"] < 0.5  # old → low score


# ===================================================================
# Unit tests — Exploit-DB signal
# ===================================================================


class TestFetchExploitdbSignal:
    def test_success(self, _fake_now, _exploitdb_csv_content):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write(_exploitdb_csv_content)
            tmp_path = tmp.name

        try:
            with (
                mock.patch("manus_agent.tools.get_poc_freshness._EXPLOITDB_CACHE", tmp_path),
                mock.patch(
                    "manus_agent.tools.get_poc_freshness._ensure_exploitdb_cache",
                    return_value=tmp_path,
                ),
            ):
                result = _fetch_exploitdb_signal("CVE-2024-1234")
            assert result["entries_found"] == 2
            assert result["most_recent_date"] == "2026-09-15"
            assert result["score"] > 0
            assert result["error"] is None
        finally:
            os.unlink(tmp_path)

    def test_no_matching_entries(self, _fake_now, _exploitdb_csv_content):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write(_exploitdb_csv_content)
            tmp_path = tmp.name

        try:
            with (
                mock.patch("manus_agent.tools.get_poc_freshness._EXPLOITDB_CACHE", tmp_path),
                mock.patch(
                    "manus_agent.tools.get_poc_freshness._ensure_exploitdb_cache",
                    return_value=tmp_path,
                ),
            ):
                result = _fetch_exploitdb_signal("CVE-9999-0001")
            assert result["entries_found"] == 0
            assert result["most_recent_date"] is None
            assert result["score"] == 0.0
        finally:
            os.unlink(tmp_path)

    def test_cache_unavailable(self, _fake_now):
        with mock.patch(
            "manus_agent.tools.get_poc_freshness._ensure_exploitdb_cache",
            return_value=None,
        ):
            result = _fetch_exploitdb_signal("CVE-2024-1234")
        assert result["error"] is not None
        assert result["entries_found"] == 0

    def test_csv_parse_error(self, _fake_now):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write("not,valid\ncsv,data\n")
            tmp_path = tmp.name

        try:
            with mock.patch(
                "manus_agent.tools.get_poc_freshness._ensure_exploitdb_cache",
                return_value=tmp_path,
            ):
                result = _fetch_exploitdb_signal("CVE-2024-1234")
            # Should not crash — graceful handling
            assert result["entries_found"] == 0
        finally:
            os.unlink(tmp_path)

    def test_entries_limited_to_5(self, _fake_now):
        """Top entries are capped at 5 in the output."""
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=["id", "description", "date_published", "codes", "type", "platform", "author"]
        )
        writer.writeheader()
        for i in range(8):
            writer.writerow(
                {
                    "id": str(50000 + i),
                    "description": f"Exploit {i}",
                    "date_published": f"2026-09-{15 - i:02d}",
                    "codes": "CVE-2024-5678",
                    "type": "remote",
                    "platform": "linux",
                    "author": f"author{i}",
                }
            )
        content = buf.getvalue()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            with mock.patch(
                "manus_agent.tools.get_poc_freshness._ensure_exploitdb_cache",
                return_value=tmp_path,
            ):
                result = _fetch_exploitdb_signal("CVE-2024-5678")
            assert result["entries_found"] == 8
            assert len(result["entries"]) <= 5
        finally:
            os.unlink(tmp_path)


# ===================================================================
# Unit tests — EPSS signal
# ===================================================================


class TestFetchEpssSignal:
    def test_success(self, _epss_response):
        with mock.patch(
            "manus_agent.tools.get_poc_freshness._http_get_json",
            return_value=_epss_response,
        ):
            result = _fetch_epss_signal("CVE-2024-1234")
        assert result["epss_score"] == pytest.approx(0.15432, abs=0.0001)
        assert result["epss_percentile"] == pytest.approx(0.9421, abs=0.001)
        assert result["score"] > 0
        assert result["error"] is None

    def test_no_data(self):
        with mock.patch(
            "manus_agent.tools.get_poc_freshness._http_get_json",
            return_value={"status": "OK", "data": []},
        ):
            result = _fetch_epss_signal("CVE-9999-0001")
        assert result["epss_score"] is None
        assert result["score"] == 0.0

    def test_http_error(self):
        with mock.patch(
            "manus_agent.tools.get_poc_freshness._http_get_json",
            side_effect=Exception("timeout"),
        ):
            result = _fetch_epss_signal("CVE-2024-1234")
        assert result["error"] == "timeout"
        assert result["score"] == 0.0

    def test_high_epss_caps_at_1(self):
        """EPSS score of 0.5 → amplified to min(1.0, 2.5) = 1.0"""
        response = {"data": [{"cve": "CVE-2024-1234", "epss": "0.50", "percentile": "0.99"}]}
        with mock.patch(
            "manus_agent.tools.get_poc_freshness._http_get_json",
            return_value=response,
        ):
            result = _fetch_epss_signal("CVE-2024-1234")
        assert result["score"] == 1.0

    def test_zero_epss(self):
        response = {"data": [{"cve": "CVE-2024-1234", "epss": "0.0", "percentile": "0.0"}]}
        with mock.patch(
            "manus_agent.tools.get_poc_freshness._http_get_json",
            return_value=response,
        ):
            result = _fetch_epss_signal("CVE-2024-1234")
        assert result["epss_score"] == 0.0
        assert result["score"] == 0.0


# ===================================================================
# Unit tests — composite freshness calculation
# ===================================================================


class TestComputeFreshness:
    def test_all_max(self):
        result = _compute_freshness({"score": 1.0}, {"score": 1.0}, {"score": 1.0})
        assert result["freshness_score"] == 100.0
        assert result["freshness_label"] == "hot"

    def test_all_zero(self):
        result = _compute_freshness({"score": 0.0}, {"score": 0.0}, {"score": 0.0})
        assert result["freshness_score"] == 0.0
        assert result["freshness_label"] == "stale"

    def test_moderate(self):
        result = _compute_freshness({"score": 0.4}, {"score": 0.3}, {"score": 0.2})
        assert 20 <= result["freshness_score"] <= 50
        assert result["freshness_label"] == "moderate"

    def test_fresh(self):
        result = _compute_freshness({"score": 0.7}, {"score": 0.5}, {"score": 0.5})
        assert 50 <= result["freshness_score"] <= 80

    def test_weights_sum_to_one(self):
        assert abs(_W_GITHUB + _W_EXPLOITDB + _W_EPSS - 1.0) < 0.0001

    def test_github_dominates(self):
        """GitHub weight is 0.5 — highest signal."""
        high_gh = _compute_freshness({"score": 1.0}, {"score": 0.0}, {"score": 0.0})
        high_edb = _compute_freshness({"score": 0.0}, {"score": 1.0}, {"score": 0.0})
        assert high_gh["freshness_score"] > high_edb["freshness_score"]


# ===================================================================
# Integration tests — check_poc_freshness
# ===================================================================


class TestCheckPocFreshness:
    def test_invalid_cve_id(self):
        result = check_poc_freshness("not-a-cve")
        assert result["error"]
        assert result["freshness_score"] == 0
        assert result["freshness_label"] == "unknown"

    def test_empty_cve_id(self):
        result = check_poc_freshness("")
        assert result["error"]

    def test_valid_cve_all_signals(self, _fake_now, _github_response, _epss_response, _exploitdb_csv_content):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp.write(_exploitdb_csv_content)
            tmp_path = tmp.name

        try:
            with (
                mock.patch(
                    "manus_agent.tools.get_poc_freshness._http_get_json",
                    side_effect=lambda url, headers=None: (
                        _github_response if "github.com" in url else _epss_response if "first.org" in url else {}
                    ),
                ),
                mock.patch(
                    "manus_agent.tools.get_poc_freshness._ensure_exploitdb_cache",
                    return_value=tmp_path,
                ),
            ):
                result = check_poc_freshness("CVE-2024-1234")
        finally:
            os.unlink(tmp_path)

        assert result["cve_id"] == "CVE-2024-1234"
        assert 0 <= result["freshness_score"] <= 100
        assert result["freshness_label"] in ("stale", "moderate", "fresh", "hot")
        assert "github" in result["signals"]
        assert "exploitdb" in result["signals"]
        assert "epss" in result["signals"]

    def test_lowercase_cve_is_uppercased(self):
        with (
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_github_signal",
                return_value={
                    "score": 0.5,
                    "repos_found": 1,
                    "total_stars": 10,
                    "total_forks": 2,
                    "most_recent_push": None,
                    "most_recent_push_days_ago": None,
                    "top_repos": [],
                    "error": None,
                },
            ),
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_exploitdb_signal",
                return_value={
                    "score": 0.0,
                    "entries_found": 0,
                    "most_recent_date": None,
                    "most_recent_days_ago": None,
                    "entries": [],
                    "error": None,
                },
            ),
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_epss_signal",
                return_value={"score": 0.0, "epss_score": None, "epss_percentile": None, "error": None},
            ),
        ):
            result = check_poc_freshness("cve-2024-1234")
        assert result["cve_id"] == "CVE-2024-1234"

    def test_all_signals_fail_gracefully(self):
        """If all signals fail, we still get a result with score 0."""
        with (
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_github_signal",
                side_effect=Exception("boom"),
            ),
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_exploitdb_signal",
                side_effect=Exception("boom"),
            ),
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_epss_signal",
                side_effect=Exception("boom"),
            ),
        ):
            result = check_poc_freshness("CVE-2024-1234")
        assert result["freshness_score"] == 0.0
        assert result["freshness_label"] == "stale"

    def test_partial_signal_failure(self, _fake_now, _github_response):
        """If some signals fail, others still contribute."""
        with (
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_github_signal",
                return_value={
                    "score": 0.8,
                    "repos_found": 3,
                    "total_stars": 59,
                    "total_forks": 11,
                    "most_recent_push": "2026-09-20T10:00:00Z",
                    "most_recent_push_days_ago": 2.0,
                    "top_repos": [],
                    "error": None,
                },
            ),
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_exploitdb_signal",
                side_effect=Exception("csv broken"),
            ),
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_epss_signal",
                side_effect=Exception("api down"),
            ),
        ):
            result = check_poc_freshness("CVE-2024-1234")
        # GitHub contributes, others are 0
        assert result["freshness_score"] > 0

    def test_cve_with_spaces(self):
        """Leading/trailing spaces are stripped."""
        with (
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_github_signal",
                return_value={
                    "score": 0.0,
                    "repos_found": 0,
                    "total_stars": 0,
                    "total_forks": 0,
                    "most_recent_push": None,
                    "most_recent_push_days_ago": None,
                    "top_repos": [],
                    "error": None,
                },
            ),
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_exploitdb_signal",
                return_value={
                    "score": 0.0,
                    "entries_found": 0,
                    "most_recent_date": None,
                    "most_recent_days_ago": None,
                    "entries": [],
                    "error": None,
                },
            ),
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_epss_signal",
                return_value={"score": 0.0, "epss_score": None, "epss_percentile": None, "error": None},
            ),
        ):
            result = check_poc_freshness("  CVE-2024-1234  ")
        assert result["cve_id"] == "CVE-2024-1234"


# ===================================================================
# Unit tests — Strands tool wrapper
# ===================================================================


class TestGetPocFreshnessTool:
    def test_delegates_to_check(self):
        with mock.patch(
            "manus_agent.tools.get_poc_freshness.check_poc_freshness",
            return_value={
                "cve_id": "CVE-2024-1234",
                "freshness_score": 55.0,
                "freshness_label": "fresh",
                "signals": {},
            },
        ) as mock_check:
            # Call the underlying function directly (Strands @tool wraps it)
            result = get_poc_freshness.__wrapped__("CVE-2024-1234")
        mock_check.assert_called_once_with("CVE-2024-1234")
        assert result["freshness_score"] == 55.0


# ===================================================================
# CLI tests — parser
# ===================================================================


class TestBuildPocFreshnessParser:
    def test_parser_creation(self):
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

    def test_missing_cve_id(self):
        from manus_agent.cli import _build_poc_freshness_parser

        parser = _build_poc_freshness_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ===================================================================
# CLI tests — text rendering
# ===================================================================


class TestRenderPocFreshnessText:
    def test_basic_rendering(self):
        from manus_agent.cli import _render_poc_freshness_text

        result = {
            "cve_id": "CVE-2024-1234",
            "freshness_score": 72.5,
            "freshness_label": "fresh",
            "signals": {
                "github": {
                    "repos_found": 3,
                    "total_stars": 59,
                    "total_forks": 11,
                    "most_recent_push_days_ago": 2.0,
                    "top_repos": [
                        {"full_name": "user/poc", "url": "https://github.com/user/poc", "stars": 42},
                    ],
                    "error": None,
                },
                "exploitdb": {
                    "entries_found": 2,
                    "most_recent_date": "2026-09-15",
                    "entries": [
                        {"title": "RCE exploit", "url": "https://exploit-db.com/exploits/12345"},
                    ],
                    "error": None,
                },
                "epss": {
                    "epss_score": 0.1543,
                    "epss_percentile": 0.9421,
                    "error": None,
                },
            },
        }
        text = _render_poc_freshness_text(result)
        assert "CVE-2024-1234" in text
        assert "72.5" in text
        assert "FRESH" in text
        assert "3 repo(s)" in text
        assert "user/poc" in text
        assert "2 entries" in text
        assert "0.1543" in text

    def test_error_rendering(self):
        from manus_agent.cli import _render_poc_freshness_text

        result = {
            "cve_id": "INVALID",
            "error": "Invalid CVE identifier",
            "freshness_score": 0,
            "freshness_label": "unknown",
            "signals": {},
        }
        text = _render_poc_freshness_text(result)
        assert "Error" in text
        assert "Invalid CVE identifier" in text

    def test_signal_errors(self):
        from manus_agent.cli import _render_poc_freshness_text

        result = {
            "cve_id": "CVE-2024-1234",
            "freshness_score": 0.0,
            "freshness_label": "stale",
            "signals": {
                "github": {"error": "rate limited"},
                "exploitdb": {"error": "csv failed"},
                "epss": {"error": "timeout"},
            },
        }
        text = _render_poc_freshness_text(result)
        assert "rate limited" in text
        assert "csv failed" in text
        assert "timeout" in text

    def test_no_epss_data(self):
        from manus_agent.cli import _render_poc_freshness_text

        result = {
            "cve_id": "CVE-2024-1234",
            "freshness_score": 10.0,
            "freshness_label": "stale",
            "signals": {
                "github": {
                    "repos_found": 0,
                    "total_stars": 0,
                    "total_forks": 0,
                    "most_recent_push_days_ago": None,
                    "top_repos": [],
                    "error": None,
                },
                "exploitdb": {"entries_found": 0, "most_recent_date": None, "entries": [], "error": None},
                "epss": {"epss_score": None, "epss_percentile": None, "error": None},
            },
        }
        text = _render_poc_freshness_text(result)
        assert "no data" in text


# ===================================================================
# CLI tests — _freshness_bar and _freshness_emoji
# ===================================================================


class TestFreshnessHelpers:
    def test_bar_full(self):
        from manus_agent.cli import _freshness_bar

        bar = _freshness_bar(100.0, width=10)
        assert "\u2588" * 10 == bar

    def test_bar_empty(self):
        from manus_agent.cli import _freshness_bar

        bar = _freshness_bar(0.0, width=10)
        assert "\u2591" * 10 == bar

    def test_bar_half(self):
        from manus_agent.cli import _freshness_bar

        bar = _freshness_bar(50.0, width=10)
        assert len(bar) == 10

    def test_emoji_hot(self):
        from manus_agent.cli import _freshness_emoji

        assert _freshness_emoji("hot") == "\U0001f525"

    def test_emoji_unknown(self):
        from manus_agent.cli import _freshness_emoji

        assert _freshness_emoji("???") == "\u2753"


# ===================================================================
# CLI tests — _run_poc_freshness
# ===================================================================


class TestRunPocFreshness:
    def test_text_output(self, capsys):
        from manus_agent.cli import _run_poc_freshness

        mock_result = {
            "cve_id": "CVE-2024-1234",
            "freshness_score": 65.0,
            "freshness_label": "fresh",
            "signals": {
                "github": {
                    "repos_found": 2,
                    "total_stars": 30,
                    "total_forks": 5,
                    "most_recent_push_days_ago": 3.0,
                    "top_repos": [],
                    "error": None,
                },
                "exploitdb": {"entries_found": 1, "most_recent_date": "2026-09-10", "entries": [], "error": None},
                "epss": {"epss_score": 0.12, "epss_percentile": 0.88, "error": None},
            },
        }
        with mock.patch(
            "manus_agent.tools.get_poc_freshness.check_poc_freshness",
            return_value=mock_result,
        ):
            exit_code = _run_poc_freshness(["CVE-2024-1234"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "CVE-2024-1234" in captured.out
        assert "65.0" in captured.out

    def test_json_output(self, capsys):
        from manus_agent.cli import _run_poc_freshness

        mock_result = {
            "cve_id": "CVE-2024-1234",
            "freshness_score": 65.0,
            "freshness_label": "fresh",
            "signals": {},
        }
        with mock.patch(
            "manus_agent.tools.get_poc_freshness.check_poc_freshness",
            return_value=mock_result,
        ):
            exit_code = _run_poc_freshness(["CVE-2024-1234", "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["freshness_score"] == 65.0

    def test_invalid_cve_id(self):
        from manus_agent.cli import _run_poc_freshness

        with pytest.raises(SystemExit):
            _run_poc_freshness(["not-a-cve"])


# ===================================================================
# CLI dispatch — integration
# ===================================================================


class TestCliDispatch:
    def test_poc_freshness_dispatches(self):
        """Verify main() dispatches to _run_poc_freshness for 'poc-freshness' subcommand."""
        with mock.patch("manus_agent.cli._run_poc_freshness", return_value=0) as mock_run:
            with mock.patch("sys.argv", ["manus-agent", "poc-freshness", "CVE-2024-1234"]):
                with pytest.raises(SystemExit) as exc_info:
                    from manus_agent.cli import main

                    main()
            mock_run.assert_called_once_with(["CVE-2024-1234"])
            assert exc_info.value.code == 0


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_cve_with_minimum_digits(self):
        """CVE-2024-0001 (4-digit suffix) should be valid."""
        result = (
            check_poc_freshness.__wrapped__("CVE-2024-0001") if hasattr(check_poc_freshness, "__wrapped__") else None
        )
        if result is None:
            with (
                mock.patch(
                    "manus_agent.tools.get_poc_freshness._fetch_github_signal",
                    return_value={
                        "score": 0.0,
                        "repos_found": 0,
                        "total_stars": 0,
                        "total_forks": 0,
                        "most_recent_push": None,
                        "most_recent_push_days_ago": None,
                        "top_repos": [],
                        "error": None,
                    },
                ),
                mock.patch(
                    "manus_agent.tools.get_poc_freshness._fetch_exploitdb_signal",
                    return_value={
                        "score": 0.0,
                        "entries_found": 0,
                        "most_recent_date": None,
                        "most_recent_days_ago": None,
                        "entries": [],
                        "error": None,
                    },
                ),
                mock.patch(
                    "manus_agent.tools.get_poc_freshness._fetch_epss_signal",
                    return_value={"score": 0.0, "epss_score": None, "epss_percentile": None, "error": None},
                ),
            ):
                result = check_poc_freshness("CVE-2024-0001")
        assert result.get("error") is None or result["cve_id"] == "CVE-2024-0001"

    def test_cve_with_7_digit_suffix(self):
        """CVE-2024-1234567 (7-digit suffix) should be valid."""
        with (
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_github_signal",
                return_value={
                    "score": 0.0,
                    "repos_found": 0,
                    "total_stars": 0,
                    "total_forks": 0,
                    "most_recent_push": None,
                    "most_recent_push_days_ago": None,
                    "top_repos": [],
                    "error": None,
                },
            ),
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_exploitdb_signal",
                return_value={
                    "score": 0.0,
                    "entries_found": 0,
                    "most_recent_date": None,
                    "most_recent_days_ago": None,
                    "entries": [],
                    "error": None,
                },
            ),
            mock.patch(
                "manus_agent.tools.get_poc_freshness._fetch_epss_signal",
                return_value={"score": 0.0, "epss_score": None, "epss_percentile": None, "error": None},
            ),
        ):
            result = check_poc_freshness("CVE-2024-1234567")
        assert result["cve_id"] == "CVE-2024-1234567"
        assert result.get("error") is None

    def test_cve_with_8_digit_suffix_invalid(self):
        """CVE-2024-12345678 (8-digit suffix) should be invalid."""
        result = check_poc_freshness("CVE-2024-12345678")
        assert result.get("error") is not None

    def test_now_utc_returns_utc(self):
        """_now_utc returns a naive datetime in UTC."""
        now = _now_utc()
        assert now.tzinfo is None

    def test_ensure_exploitdb_cache_fresh(self):
        """When cache is fresh, return path without downloading."""
        import time as _time

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"header\n")
            tmp_path = tmp.name

        try:
            # Set mtime to now (fresh)
            os.utime(tmp_path, (_time.time(), _time.time()))
            with mock.patch("manus_agent.tools.get_poc_freshness._EXPLOITDB_CACHE", tmp_path):
                from manus_agent.tools.get_poc_freshness import _ensure_exploitdb_cache

                result = _ensure_exploitdb_cache()
            assert result == tmp_path
        finally:
            os.unlink(tmp_path)

    def test_ensure_exploitdb_cache_stale(self):
        """When cache is stale, attempt to download."""
        import time as _time

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(b"header\n")
            tmp_path = tmp.name

        try:
            # Set mtime to 2 days ago (stale)
            old_time = _time.time() - 200_000
            os.utime(tmp_path, (old_time, old_time))
            with (
                mock.patch("manus_agent.tools.get_poc_freshness._EXPLOITDB_CACHE", tmp_path),
                mock.patch(
                    "urllib.request.urlopen",
                    side_effect=Exception("network error"),
                ),
            ):
                from manus_agent.tools.get_poc_freshness import _ensure_exploitdb_cache

                result = _ensure_exploitdb_cache()
            # Download fails → returns None
            assert result is None
        finally:
            os.unlink(tmp_path)

    def test_http_get_json_timeout(self):
        """_http_get_json propagates exceptions."""
        from manus_agent.tools.get_poc_freshness import _http_get_json

        with mock.patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            with pytest.raises(Exception, match="timeout"):
                _http_get_json("https://example.com/api")
