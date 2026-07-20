"""Comprehensive tests for poc-freshness CLI subcommand and poc_freshness tool."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from manus_agent.tools.poc_freshness import (
    _check_trickest,
    _compute_freshness,
    _count_nvd_exploit_refs,
    _days_ago,
    _decay_score,
    _search_exploitdb,
    _search_github_pocs,
)

# ---------------------------------------------------------------------------
# Unit tests for decay scoring
# ---------------------------------------------------------------------------


class TestDecayScore:
    """Tests for the exponential decay scoring function."""

    def test_today_scores_100(self):
        assert _decay_score(0) == pytest.approx(100.0)

    def test_half_life_scores_50(self):
        score = _decay_score(60)  # _HALF_LIFE_DAYS = 60
        assert score == pytest.approx(50.0, rel=0.01)

    def test_very_old_approaches_zero(self):
        score = _decay_score(365)
        assert score < 2.0

    def test_one_week_still_high(self):
        score = _decay_score(7)
        assert score > 90.0

    def test_one_month_moderate(self):
        score = _decay_score(30)
        assert 60.0 < score < 80.0


class TestDaysAgo:
    """Tests for date parsing and days-ago calculation."""

    def test_iso_with_z(self):
        # A date in the past should return positive days
        days = _days_ago("2020-01-01T00:00:00Z")
        assert days > 365

    def test_iso_with_offset(self):
        days = _days_ago("2020-06-15T12:00:00+00:00")
        assert days > 365

    def test_date_only(self):
        days = _days_ago("2020-01-01")
        assert days > 365

    def test_future_returns_zero(self):
        # If somehow a date is in the future, clamp to 0
        days = _days_ago("2099-01-01T00:00:00Z")
        assert days == 0


# ---------------------------------------------------------------------------
# Unit tests for individual signal gatherers (mocked HTTP)
# ---------------------------------------------------------------------------


class TestSearchGithubPocs:
    """Tests for GitHub PoC repo search."""

    @patch("manus_agent.tools.poc_freshness._http_get_json")
    def test_returns_repos(self, mock_get):
        mock_get.return_value = {
            "items": [
                {
                    "full_name": "attacker/CVE-2024-3094-poc",
                    "html_url": "https://github.com/attacker/CVE-2024-3094-poc",
                    "pushed_at": "2026-07-18T10:00:00Z",
                    "created_at": "2026-07-15T10:00:00Z",
                    "stargazers_count": 42,
                }
            ]
        }
        result = _search_github_pocs("CVE-2024-3094")
        assert len(result) == 1
        assert result[0]["name"] == "attacker/CVE-2024-3094-poc"
        assert result[0]["stars"] == 42
        assert result[0]["days_since_push"] >= 0

    @patch("manus_agent.tools.poc_freshness._http_get_json")
    def test_empty_results(self, mock_get):
        mock_get.return_value = {"items": []}
        result = _search_github_pocs("CVE-2099-9999")
        assert result == []

    @patch("manus_agent.tools.poc_freshness._http_get_json")
    def test_api_failure_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("rate limited")
        result = _search_github_pocs("CVE-2024-3094")
        assert result == []

    @patch("manus_agent.tools.poc_freshness._http_get_json")
    def test_missing_fields_handled(self, mock_get):
        mock_get.return_value = {
            "items": [
                {
                    "full_name": "test/repo",
                    "html_url": "https://github.com/test/repo",
                    "pushed_at": "",
                    "created_at": "",
                    "stargazers_count": 0,
                }
            ]
        }
        result = _search_github_pocs("CVE-2024-1234")
        assert len(result) == 1
        assert result[0]["days_since_push"] == 9999

    @patch("manus_agent.tools.poc_freshness._http_get_json")
    def test_respects_per_page_limit(self, mock_get):
        items = [
            {
                "full_name": f"user/repo-{i}",
                "html_url": f"https://github.com/user/repo-{i}",
                "pushed_at": "2026-07-01T00:00:00Z",
                "created_at": "2026-06-01T00:00:00Z",
                "stargazers_count": i,
            }
            for i in range(15)
        ]
        mock_get.return_value = {"items": items}
        result = _search_github_pocs("CVE-2024-3094")
        assert len(result) == 10  # capped at 10


class TestCheckTrickest:
    """Tests for trickest/cve index check."""

    @patch("manus_agent.tools.poc_freshness._http_get_text")
    def test_found_with_pocs(self, mock_get):
        mock_get.return_value = "### POC\n#### Github\n- https://github.com/user/poc1\n- https://github.com/user/poc2\n"
        result = _check_trickest("CVE-2024-3094")
        assert result["found"] is True
        assert result["poc_count"] == 2

    @patch("manus_agent.tools.poc_freshness._http_get_text")
    def test_not_found(self, mock_get):
        mock_get.side_effect = Exception("404")
        result = _check_trickest("CVE-2099-9999")
        assert result["found"] is False
        assert result["poc_count"] == 0

    def test_invalid_cve_format(self):
        result = _check_trickest("not-a-cve")
        assert result["found"] is False


class TestSearchExploitdb:
    """Tests for Exploit-DB CSV search."""

    @patch("manus_agent.tools.poc_freshness._ensure_exploitdb_csv")
    def test_csv_unavailable(self, mock_ensure):
        mock_ensure.return_value = None
        result = _search_exploitdb("CVE-2024-3094")
        assert result == []

    @patch("manus_agent.tools.poc_freshness._ensure_exploitdb_csv")
    def test_finds_matching_entries(self, mock_ensure, tmp_path):
        csv_file = tmp_path / "exploits.csv"
        csv_file.write_text(
            "id,description,date_published,codes\n"
            "51234,XSS in Foo,2026-07-10,CVE-2024-3094;OSVDB-12345\n"
            "51235,SQLi in Bar,2026-06-01,CVE-2024-9999\n"
        )
        mock_ensure.return_value = str(csv_file)
        result = _search_exploitdb("CVE-2024-3094")
        assert len(result) == 1
        assert result[0]["edb_id"] == "51234"

    @patch("manus_agent.tools.poc_freshness._ensure_exploitdb_csv")
    def test_no_match(self, mock_ensure, tmp_path):
        csv_file = tmp_path / "exploits.csv"
        csv_file.write_text("id,description,date_published,codes\n51235,SQLi in Bar,2026-06-01,CVE-2024-9999\n")
        mock_ensure.return_value = str(csv_file)
        result = _search_exploitdb("CVE-2024-3094")
        assert result == []


class TestCountNvdExploitRefs:
    """Tests for NVD exploit reference counting."""

    @patch("manus_agent.tools.poc_freshness._http_get_json")
    def test_counts_exploit_tags(self, mock_get):
        mock_get.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "references": [
                            {"url": "https://example.com/poc", "tags": ["Exploit"]},
                            {"url": "https://nvd.nist.gov", "tags": ["Vendor Advisory"]},
                            {"url": "https://exploit-db.com/1234", "tags": ["Third Party Advisory", "Exploit"]},
                        ]
                    }
                }
            ]
        }
        count = _count_nvd_exploit_refs("CVE-2024-3094")
        assert count == 2

    @patch("manus_agent.tools.poc_freshness._http_get_json")
    def test_no_exploit_refs(self, mock_get):
        mock_get.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "references": [
                            {"url": "https://nvd.nist.gov", "tags": ["Vendor Advisory"]},
                        ]
                    }
                }
            ]
        }
        count = _count_nvd_exploit_refs("CVE-2024-3094")
        assert count == 0

    @patch("manus_agent.tools.poc_freshness._http_get_json")
    def test_api_failure(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        count = _count_nvd_exploit_refs("CVE-2024-3094")
        assert count == 0


# ---------------------------------------------------------------------------
# Unit tests for composite scoring
# ---------------------------------------------------------------------------


class TestComputeFreshness:
    """Tests for the composite freshness scoring algorithm."""

    def test_no_signals_scores_zero(self):
        result = _compute_freshness([], {"found": False, "poc_count": 0}, [], 0)
        assert result["freshness_score"] == 0.0
        assert result["classification"] == "MINIMAL - No recent PoC activity detected"

    def test_recent_github_activity_scores_high(self):
        repos = [
            {
                "name": "test/poc",
                "url": "https://github.com/test/poc",
                "pushed_at": "2026-07-19T00:00:00Z",
                "created_at": "2026-07-19T00:00:00Z",
                "stars": 10,
                "days_since_push": 1,
                "days_since_create": 1,
            }
        ]
        result = _compute_freshness(repos, {"found": False, "poc_count": 0}, [], 0)
        assert result["freshness_score"] > 80

    def test_old_activity_scores_low(self):
        repos = [
            {
                "name": "test/poc",
                "url": "https://github.com/test/poc",
                "pushed_at": "2025-01-01T00:00:00Z",
                "created_at": "2024-06-01T00:00:00Z",
                "stars": 5,
                "days_since_push": 500,
                "days_since_create": 700,
            }
        ]
        result = _compute_freshness(repos, {"found": False, "poc_count": 0}, [], 0)
        assert result["freshness_score"] < 20

    def test_multiple_signals_amplify(self):
        repos = [
            {
                "name": "test/poc",
                "url": "https://github.com/test/poc",
                "pushed_at": "2026-07-15T00:00:00Z",
                "created_at": "2026-07-10T00:00:00Z",
                "stars": 100,
                "days_since_push": 5,
                "days_since_create": 10,
            }
        ]
        trickest = {"found": True, "poc_count": 8}
        edb = [{"edb_id": "123", "description": "test", "date_published": "2026-07-10", "days_since_publish": 10}]
        result = _compute_freshness(repos, trickest, edb, 5)
        assert result["freshness_score"] > 90

    def test_trickest_only(self):
        result = _compute_freshness([], {"found": True, "poc_count": 3}, [], 0)
        assert result["freshness_score"] > 0
        assert any(s["source"] == "trickest_cve_index" for s in result["signals"])

    def test_exploitdb_only(self):
        edb = [{"edb_id": "999", "description": "test", "date_published": "2026-07-01", "days_since_publish": 19}]
        result = _compute_freshness([], {"found": False, "poc_count": 0}, edb, 0)
        assert result["freshness_score"] > 0
        assert any(s["source"] == "exploit_db" for s in result["signals"])

    def test_nvd_only(self):
        result = _compute_freshness([], {"found": False, "poc_count": 0}, [], 5)
        assert result["freshness_score"] > 0
        assert any(s["source"] == "nvd_exploit_refs" for s in result["signals"])

    def test_score_capped_at_100(self):
        # Extreme signals should not exceed 100
        repos = [
            {
                "name": f"test/poc-{i}",
                "url": f"https://github.com/test/poc-{i}",
                "pushed_at": "2026-07-20T00:00:00Z",
                "created_at": "2026-07-20T00:00:00Z",
                "stars": 500,
                "days_since_push": 0,
                "days_since_create": 0,
            }
            for i in range(10)
        ]
        trickest = {"found": True, "poc_count": 20}
        edb = [{"edb_id": "1", "description": "x", "date_published": "2026-07-20", "days_since_publish": 0}]
        result = _compute_freshness(repos, trickest, edb, 10)
        assert result["freshness_score"] <= 100.0

    def test_classification_critical(self):
        repos = [
            {
                "name": "test/poc",
                "url": "u",
                "pushed_at": "now",
                "created_at": "now",
                "stars": 200,
                "days_since_push": 0,
                "days_since_create": 0,
            }
        ]
        result = _compute_freshness(repos, {"found": True, "poc_count": 10}, [], 5)
        assert "CRITICAL" in result["classification"]

    def test_classification_high(self):
        repos = [
            {
                "name": "test/poc",
                "url": "u",
                "pushed_at": "x",
                "created_at": "x",
                "stars": 10,
                "days_since_push": 14,
                "days_since_create": 60,
            }
        ]
        result = _compute_freshness(repos, {"found": True, "poc_count": 3}, [], 0)
        assert "HIGH" in result["classification"] or "CRITICAL" in result["classification"]

    def test_star_bonus_threshold(self):
        # Stars < 50 should not produce github_stars signal
        repos = [
            {
                "name": "test/poc",
                "url": "u",
                "pushed_at": "x",
                "created_at": "x",
                "stars": 49,
                "days_since_push": 5,
                "days_since_create": 100,
            }
        ]
        result = _compute_freshness(repos, {"found": False, "poc_count": 0}, [], 0)
        assert not any(s["source"] == "github_stars" for s in result["signals"])

    def test_star_bonus_above_threshold(self):
        repos = [
            {
                "name": "test/poc",
                "url": "u",
                "pushed_at": "x",
                "created_at": "x",
                "stars": 100,
                "days_since_push": 5,
                "days_since_create": 100,
            }
        ]
        result = _compute_freshness(repos, {"found": False, "poc_count": 0}, [], 0)
        assert any(s["source"] == "github_stars" for s in result["signals"])

    def test_new_repo_bonus(self):
        repos = [
            {
                "name": "test/poc",
                "url": "u",
                "pushed_at": "x",
                "created_at": "x",
                "stars": 5,
                "days_since_push": 2,
                "days_since_create": 2,  # < 30 days
            }
        ]
        result = _compute_freshness(repos, {"found": False, "poc_count": 0}, [], 0)
        assert any(s["source"] == "github_new_repo" for s in result["signals"])

    def test_no_new_repo_bonus_when_old(self):
        repos = [
            {
                "name": "test/poc",
                "url": "u",
                "pushed_at": "x",
                "created_at": "x",
                "stars": 5,
                "days_since_push": 5,
                "days_since_create": 60,  # > 30 days
            }
        ]
        result = _compute_freshness(repos, {"found": False, "poc_count": 0}, [], 0)
        assert not any(s["source"] == "github_new_repo" for s in result["signals"])


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestCliPocFreshness:
    """Tests for the manus-agent poc-freshness CLI subcommand."""

    @patch("manus_agent.tools.poc_freshness._count_nvd_exploit_refs")
    @patch("manus_agent.tools.poc_freshness._search_exploitdb")
    @patch("manus_agent.tools.poc_freshness._check_trickest")
    @patch("manus_agent.tools.poc_freshness._search_github_pocs")
    def test_text_output(self, mock_gh, mock_tr, mock_edb, mock_nvd, capsys):
        mock_gh.return_value = [
            {
                "name": "attacker/poc",
                "url": "https://github.com/attacker/poc",
                "pushed_at": "2026-07-18T00:00:00Z",
                "created_at": "2026-07-15T00:00:00Z",
                "stars": 25,
                "days_since_push": 2,
                "days_since_create": 5,
            }
        ]
        mock_tr.return_value = {"found": True, "poc_count": 4}
        mock_edb.return_value = []
        mock_nvd.return_value = 2

        from manus_agent.cli import _run_poc_freshness

        rc = _run_poc_freshness(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Freshness Score" in out
        assert "CVE-2024-3094" in out
        assert "Classification" in out

    @patch("manus_agent.tools.poc_freshness._count_nvd_exploit_refs")
    @patch("manus_agent.tools.poc_freshness._search_exploitdb")
    @patch("manus_agent.tools.poc_freshness._check_trickest")
    @patch("manus_agent.tools.poc_freshness._search_github_pocs")
    def test_json_output(self, mock_gh, mock_tr, mock_edb, mock_nvd, capsys):
        mock_gh.return_value = []
        mock_tr.return_value = {"found": True, "poc_count": 2}
        mock_edb.return_value = []
        mock_nvd.return_value = 0

        from manus_agent.cli import _run_poc_freshness

        rc = _run_poc_freshness(["CVE-2024-3094", "--output", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "freshness_score" in data
        assert "classification" in data
        assert "signals" in data
        assert data["cve_id"] == "CVE-2024-3094"

    @patch("manus_agent.tools.poc_freshness._count_nvd_exploit_refs")
    @patch("manus_agent.tools.poc_freshness._search_exploitdb")
    @patch("manus_agent.tools.poc_freshness._check_trickest")
    @patch("manus_agent.tools.poc_freshness._search_github_pocs")
    def test_invalid_cve_format(self, mock_gh, mock_tr, mock_edb, mock_nvd, capsys):
        from manus_agent.cli import _run_poc_freshness

        rc = _run_poc_freshness(["not-a-cve"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid CVE ID" in err

    @patch("manus_agent.tools.poc_freshness._count_nvd_exploit_refs")
    @patch("manus_agent.tools.poc_freshness._search_exploitdb")
    @patch("manus_agent.tools.poc_freshness._check_trickest")
    @patch("manus_agent.tools.poc_freshness._search_github_pocs")
    def test_all_signals_fail_gracefully(self, mock_gh, mock_tr, mock_edb, mock_nvd, capsys):
        mock_gh.side_effect = Exception("network error")
        mock_tr.side_effect = Exception("network error")
        mock_edb.side_effect = Exception("network error")
        mock_nvd.side_effect = Exception("network error")

        from manus_agent.cli import _run_poc_freshness

        rc = _run_poc_freshness(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "0.0/100" in out or "0/100" in out

    @patch("manus_agent.tools.poc_freshness._count_nvd_exploit_refs")
    @patch("manus_agent.tools.poc_freshness._search_exploitdb")
    @patch("manus_agent.tools.poc_freshness._check_trickest")
    @patch("manus_agent.tools.poc_freshness._search_github_pocs")
    def test_case_insensitive_cve(self, mock_gh, mock_tr, mock_edb, mock_nvd, capsys):
        mock_gh.return_value = []
        mock_tr.return_value = {"found": False, "poc_count": 0}
        mock_edb.return_value = []
        mock_nvd.return_value = 0

        from manus_agent.cli import _run_poc_freshness

        rc = _run_poc_freshness(["cve-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2024-3094" in out

    @patch("manus_agent.tools.poc_freshness._count_nvd_exploit_refs")
    @patch("manus_agent.tools.poc_freshness._search_exploitdb")
    @patch("manus_agent.tools.poc_freshness._check_trickest")
    @patch("manus_agent.tools.poc_freshness._search_github_pocs")
    def test_json_output_structure(self, mock_gh, mock_tr, mock_edb, mock_nvd, capsys):
        mock_gh.return_value = [
            {
                "name": "user/exploit",
                "url": "https://github.com/user/exploit",
                "pushed_at": "2026-07-10T00:00:00Z",
                "created_at": "2026-06-01T00:00:00Z",
                "stars": 75,
                "days_since_push": 10,
                "days_since_create": 49,
            }
        ]
        mock_tr.return_value = {"found": True, "poc_count": 5}
        mock_edb.return_value = [
            {"edb_id": "55555", "description": "PoC", "date_published": "2026-07-05", "days_since_publish": 15}
        ]
        mock_nvd.return_value = 3

        from manus_agent.cli import _run_poc_freshness

        rc = _run_poc_freshness(["CVE-2024-3094", "--output", "json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data["freshness_score"], (int, float))
        assert isinstance(data["signals"], list)
        assert data["github_repos_found"] == 1
        assert data["exploitdb_entries_found"] == 1
        assert data["trickest_indexed"] is True
        assert data["trickest_poc_count"] == 5
        assert data["nvd_exploit_refs"] == 3


# ---------------------------------------------------------------------------
# Tool function tests
# ---------------------------------------------------------------------------


class TestPocFreshnessTool:
    """Tests for the @tool decorated poc_freshness function."""

    @patch("manus_agent.tools.poc_freshness._count_nvd_exploit_refs")
    @patch("manus_agent.tools.poc_freshness._search_exploitdb")
    @patch("manus_agent.tools.poc_freshness._check_trickest")
    @patch("manus_agent.tools.poc_freshness._search_github_pocs")
    def test_invalid_cve(self, mock_gh, mock_tr, mock_edb, mock_nvd):
        from manus_agent.tools.poc_freshness import poc_freshness

        # Call the underlying function directly
        result = poc_freshness._tool_func(cve_id="invalid")
        assert "Invalid CVE ID" in result

    @patch("manus_agent.tools.poc_freshness._count_nvd_exploit_refs")
    @patch("manus_agent.tools.poc_freshness._search_exploitdb")
    @patch("manus_agent.tools.poc_freshness._check_trickest")
    @patch("manus_agent.tools.poc_freshness._search_github_pocs")
    def test_valid_cve_returns_report(self, mock_gh, mock_tr, mock_edb, mock_nvd):
        mock_gh.return_value = []
        mock_tr.return_value = {"found": False, "poc_count": 0}
        mock_edb.return_value = []
        mock_nvd.return_value = 0

        from manus_agent.tools.poc_freshness import poc_freshness

        # Call the underlying function directly
        result = poc_freshness._tool_func(cve_id="CVE-2024-3094")
        assert "Freshness Score" in result
        assert "CVE-2024-3094" in result
