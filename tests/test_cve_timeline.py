"""
Tests for get_cve_timeline tool and the manus-use timeline CLI subcommand.

All HTTP calls are mocked — no real network I/O.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _tool_use(cve_id: str, tool_use_id: str = "tu-1") -> dict:
    return {"toolUseId": tool_use_id, "input": {"cve_id": cve_id}}


def _nvd_resp(published: str = "2024-03-29", kev_date: str | None = None) -> dict:
    cve: dict = {
        "published": published,
        "lastModified": published,
        "references": [],
    }
    if kev_date:
        cve["cisaExploitAdd"] = kev_date
    return {"vulnerabilities": [{"cve": cve}]}


def _epss_resp(scores: list[float], base_date: str = "2024-01-01") -> dict:
    from datetime import datetime, timedelta

    base = datetime.strptime(base_date, "%Y-%m-%d")
    series = [
        {
            "date": (base + timedelta(days=i)).strftime("%Y-%m-%d"),
            "epss": str(score),
            "percentile": "0.9",
        }
        for i, score in enumerate(scores)
    ]
    latest = series[-1] if series else {"date": base_date, "epss": "0.01", "percentile": "0.5"}
    return {
        "data": [
            {
                "cve": "CVE-MOCK",
                "date": latest["date"],
                "epss": latest["epss"],
                "percentile": latest["percentile"],
                "time-series": series[:-1],
            }
        ]
    }


def _mock_resp(json_data: dict, status_code: int = 200) -> MagicMock:
    m = MagicMock()
    m.status_code = status_code
    m.ok = status_code < 400
    m.json.return_value = json_data
    m.raise_for_status.side_effect = None if status_code < 400 else Exception("HTTP error")
    m.text = json.dumps(json_data)
    return m


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestIsoDate:
    def test_iso_date_full_timestamp(self):
        from manus_use.tools.get_cve_timeline import _iso_date

        assert _iso_date("2024-03-29T14:00:00.000Z") == "2024-03-29"

    def test_iso_date_already_iso(self):
        from manus_use.tools.get_cve_timeline import _iso_date

        assert _iso_date("2024-03-29") == "2024-03-29"

    def test_iso_date_none(self):
        from manus_use.tools.get_cve_timeline import _iso_date

        assert _iso_date(None) is None

    def test_iso_date_empty(self):
        from manus_use.tools.get_cve_timeline import _iso_date

        assert _iso_date("") is None

    def test_iso_date_unrecognised(self):
        from manus_use.tools.get_cve_timeline import _iso_date

        assert _iso_date("not-a-date") is None


class TestDaysBetween:
    def test_positive_gap(self):
        from manus_use.tools.get_cve_timeline import _days_between

        assert _days_between("2024-01-01", "2024-03-31") == 90

    def test_same_day(self):
        from manus_use.tools.get_cve_timeline import _days_between

        assert _days_between("2024-06-01", "2024-06-01") == 0

    def test_negative_gap(self):
        from manus_use.tools.get_cve_timeline import _days_between

        # 2024 is a leap year: Jan(31)+Feb(29)+Mar(31)+Apr(30)+May(31) = 152 days before Jun
        assert _days_between("2024-06-01", "2024-01-01") == -152

    def test_missing_a(self):
        from manus_use.tools.get_cve_timeline import _days_between

        assert _days_between(None, "2024-06-01") is None

    def test_missing_b(self):
        from manus_use.tools.get_cve_timeline import _days_between

        assert _days_between("2024-06-01", None) is None


class TestGithubHeaders:
    def test_no_token(self):
        from manus_use.tools.get_cve_timeline import _github_headers

        with patch.dict("os.environ", {}, clear=True):
            headers = _github_headers()
        assert "Accept" in headers
        assert "Authorization" not in headers

    def test_with_token(self):
        from manus_use.tools.get_cve_timeline import _github_headers

        with patch.dict("os.environ", {"GITHUB_TOKEN": "tok_test"}):
            headers = _github_headers()
        assert headers["Authorization"] == "token tok_test"


# ---------------------------------------------------------------------------
# Tests for _fetch_nvd_info
# ---------------------------------------------------------------------------


class TestFetchNvdInfo:
    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_published_date(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_nvd_info

        mock_get.return_value = _mock_resp(_nvd_resp("2024-03-29"))
        result = _fetch_nvd_info("CVE-2024-3094")
        assert result["published"] == "2024-03-29"
        assert "nvd_url" in result

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_embedded_kev_date(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_nvd_info

        mock_get.return_value = _mock_resp(_nvd_resp("2024-03-29", kev_date="2024-04-01"))
        result = _fetch_nvd_info("CVE-2024-3094")
        assert result["kev_date"] == "2024-04-01"

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_empty_on_no_vulns(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_nvd_info

        mock_get.return_value = _mock_resp({"vulnerabilities": []})
        result = _fetch_nvd_info("CVE-2024-9999")
        assert result == {}

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_empty_on_network_error(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_nvd_info

        mock_get.side_effect = Exception("timeout")
        result = _fetch_nvd_info("CVE-2024-3094")
        assert result == {}


# ---------------------------------------------------------------------------
# Tests for _fetch_ghsa_patch_date
# ---------------------------------------------------------------------------


class TestFetchGhsaPatchDate:
    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_none_when_no_advisories(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_ghsa_patch_date

        mock_get.return_value = _mock_resp([])
        d, u = _fetch_ghsa_patch_date("CVE-2024-3094")
        assert d is None
        assert u is None

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_published_at_when_no_commit_url(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_ghsa_patch_date

        advisory = {
            "published_at": "2024-04-05T12:00:00Z",
            "updated_at": "2024-04-05T12:00:00Z",
            "references": [{"url": "https://example.com/advisory"}],
            "html_url": "https://github.com/advisories/GHSA-test",
        }
        mock_get.return_value = _mock_resp([advisory])
        d, u = _fetch_ghsa_patch_date("CVE-2024-3094")
        assert d == "2024-04-05"

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_commit_date_when_available(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_ghsa_patch_date

        commit_resp = {
            "commit": {
                "committer": {"date": "2024-03-25T10:00:00Z"},
                "author": {"date": "2024-03-25T09:00:00Z"},
            }
        }
        advisory = {
            "published_at": "2024-04-05T12:00:00Z",
            "references": [
                {"url": "https://github.com/org/repo/commit/abc1234567890abcdef"}
            ],
            "html_url": "https://github.com/advisories/GHSA-test",
        }

        def side_effect(url, *args, **kwargs):
            if "api.github.com/repos" in url and "/commits/" in url:
                return _mock_resp(commit_resp)
            return _mock_resp([advisory])

        mock_get.side_effect = side_effect
        d, u = _fetch_ghsa_patch_date("CVE-2024-3094")
        assert d == "2024-03-25"

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_none_on_exception(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_ghsa_patch_date

        mock_get.side_effect = Exception("network error")
        d, u = _fetch_ghsa_patch_date("CVE-2024-3094")
        assert d is None and u is None


# ---------------------------------------------------------------------------
# Tests for _fetch_trickest_poc
# ---------------------------------------------------------------------------


class TestFetchTrickestPoc:
    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_none_on_404(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_trickest_poc

        mock_get.return_value = _mock_resp({}, status_code=404)
        d, u = _fetch_trickest_poc("CVE-2024-3094")
        assert d is None and u is None

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_extracts_repo_creation_date(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_trickest_poc

        md_content = "# CVE-2024-3094\n\nhttps://github.com/hacker/exploit-3094\n"
        repo_resp = {"created_at": "2024-03-30T08:00:00Z", "name": "exploit-3094"}

        def side_effect(url, *args, **kwargs):
            if "trickest" in url:
                m = MagicMock()
                m.status_code = 200
                m.ok = True
                m.text = md_content
                m.raise_for_status.side_effect = None
                return m
            return _mock_resp(repo_resp)

        mock_get.side_effect = side_effect
        d, u = _fetch_trickest_poc("CVE-2024-3094")
        assert d == "2024-03-30"
        assert "hacker/exploit-3094" in u

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_falls_back_to_date_in_markdown(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_trickest_poc

        md_content = "# CVE-2024-3094\n\nDiscovered 2024-03-31.\n"

        def side_effect(url, *args, **kwargs):
            if "trickest" in url:
                m = MagicMock()
                m.status_code = 200
                m.ok = True
                m.text = md_content
                m.raise_for_status.side_effect = None
                return m
            # No repos found — return empty
            return _mock_resp({}, status_code=404)

        mock_get.side_effect = side_effect
        d, u = _fetch_trickest_poc("CVE-2024-3094")
        assert d == "2024-03-31"

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_none_on_exception(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_trickest_poc

        mock_get.side_effect = Exception("network error")
        d, u = _fetch_trickest_poc("CVE-2024-3094")
        assert d is None and u is None

    def test_invalid_cve_id(self):
        from manus_use.tools.get_cve_timeline import _fetch_trickest_poc

        d, u = _fetch_trickest_poc("NOT-A-CVE")
        assert d is None and u is None


# ---------------------------------------------------------------------------
# Tests for _fetch_epss_spike
# ---------------------------------------------------------------------------


class TestFetchEpssSpike:
    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_no_spike_when_jump_below_threshold(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_epss_spike

        scores = [0.01, 0.02, 0.03, 0.04, 0.05]  # max 7-day jump = 0.04
        mock_get.return_value = _mock_resp(_epss_resp(scores))
        spike_date, jump = _fetch_epss_spike("CVE-2024-3094")
        assert spike_date is None
        assert jump < 0.10

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_spike_detected_when_jump_at_threshold(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_epss_spike

        # Jump from day0=0.01 to day3=0.12 = 0.11 > 0.10
        scores = [0.01, 0.02, 0.05, 0.12, 0.13]
        mock_get.return_value = _mock_resp(_epss_resp(scores))
        spike_date, jump = _fetch_epss_spike("CVE-2024-3094")
        assert spike_date is not None
        assert jump >= 0.10

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_none_on_empty_data(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_epss_spike

        mock_get.return_value = _mock_resp({"data": []})
        spike_date, jump = _fetch_epss_spike("CVE-2024-3094")
        assert spike_date is None
        assert jump == 0.0

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_none_on_exception(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_epss_spike

        mock_get.side_effect = Exception("timeout")
        spike_date, jump = _fetch_epss_spike("CVE-2024-3094")
        assert spike_date is None


# ---------------------------------------------------------------------------
# Tests for _fetch_cisa_kev
# ---------------------------------------------------------------------------


class TestFetchCisaKev:
    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_found_in_kev(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_cisa_kev

        kev_data = {
            "vulnerabilities": [
                {
                    "cveID": "CVE-2024-3094",
                    "dateAdded": "2024-04-10",
                    "requiredAction": "Apply updates",
                }
            ]
        }
        mock_get.return_value = _mock_resp(kev_data)
        d, action = _fetch_cisa_kev("CVE-2024-3094")
        assert d == "2024-04-10"
        assert action == "Apply updates"

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_not_found_in_kev(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_cisa_kev

        mock_get.return_value = _mock_resp({"vulnerabilities": []})
        d, action = _fetch_cisa_kev("CVE-2024-9999")
        assert d is None and action is None

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_returns_none_on_exception(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_cisa_kev

        mock_get.side_effect = Exception("network error")
        d, action = _fetch_cisa_kev("CVE-2024-3094")
        assert d is None and action is None

    @patch("manus_use.tools.get_cve_timeline.requests.get")
    def test_case_insensitive_match(self, mock_get):
        from manus_use.tools.get_cve_timeline import _fetch_cisa_kev

        kev_data = {
            "vulnerabilities": [
                {"cveID": "cve-2024-3094", "dateAdded": "2024-04-10", "requiredAction": "Patch"}
            ]
        }
        mock_get.return_value = _mock_resp(kev_data)
        d, _ = _fetch_cisa_kev("CVE-2024-3094")
        assert d == "2024-04-10"


# ---------------------------------------------------------------------------
# Tests for build_cve_timeline
# ---------------------------------------------------------------------------


class TestBuildCveTimeline:
    def _mock_all_sources(
        self,
        disclosure="2024-03-29",
        patch_d=None,
        poc_d=None,
        spike_d=None,
        kev_d=None,
        max_jump=0.0,
    ):
        """Return a dict of patch targets for mocking all fetchers."""
        return {
            "_fetch_nvd_info": {"published": disclosure, "nvd_url": "https://nvd.nist.gov/..."},
            "_fetch_ghsa_patch_date": (patch_d, "https://github.com/org/repo/commit/abc"),
            "_fetch_nvd_patch_date": (None, None),
            "_fetch_trickest_poc": (poc_d, "https://github.com/poc/repo"),
            "_fetch_epss_spike": (spike_d, max_jump),
            "_fetch_cisa_kev": (kev_d, "Apply updates" if kev_d else None),
        }

    @patch("manus_use.tools.get_cve_timeline._fetch_cisa_kev")
    @patch("manus_use.tools.get_cve_timeline._fetch_epss_spike")
    @patch("manus_use.tools.get_cve_timeline._fetch_trickest_poc")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_ghsa_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_info")
    def test_all_events_present(
        self, m_nvd, m_ghsa, m_nvd_patch, m_trickest, m_epss, m_kev
    ):
        from manus_use.tools.get_cve_timeline import build_cve_timeline

        m_nvd.return_value = {"published": "2024-03-29", "nvd_url": "https://nvd.nist.gov/"}
        m_ghsa.return_value = ("2024-03-25", "https://github.com/org/repo/commit/abc")
        m_nvd_patch.return_value = (None, None)
        m_trickest.return_value = ("2024-03-30", "https://github.com/poc/repo")
        m_epss.return_value = ("2024-04-05", 0.15)
        m_kev.return_value = ("2024-04-10", "Apply updates")

        result = build_cve_timeline("CVE-2024-3094")

        assert result["cve_id"] == "CVE-2024-3094"
        event_names = [e["event"] for e in result["events"]]
        assert any("Disclosed" in n for n in event_names)
        assert any("Patch" in n for n in event_names)
        assert any("PoC" in n for n in event_names)
        assert any("EPSS" in n for n in event_names)
        assert any("KEV" in n for n in event_names)

    @patch("manus_use.tools.get_cve_timeline._fetch_cisa_kev")
    @patch("manus_use.tools.get_cve_timeline._fetch_epss_spike")
    @patch("manus_use.tools.get_cve_timeline._fetch_trickest_poc")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_ghsa_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_info")
    def test_events_sorted_chronologically(
        self, m_nvd, m_ghsa, m_nvd_patch, m_trickest, m_epss, m_kev
    ):
        from manus_use.tools.get_cve_timeline import build_cve_timeline

        m_nvd.return_value = {"published": "2024-03-29", "nvd_url": "https://nvd.nist.gov/"}
        m_ghsa.return_value = ("2024-03-25", "https://github.com/org/repo/commit/abc")
        m_nvd_patch.return_value = (None, None)
        m_trickest.return_value = ("2024-04-01", "https://github.com/poc/repo")
        m_epss.return_value = ("2024-04-10", 0.20)
        m_kev.return_value = ("2024-04-15", "Apply")

        result = build_cve_timeline("CVE-2024-3094")
        dates = [e["date"] for e in result["events"] if e["date"]]
        assert dates == sorted(dates)

    @patch("manus_use.tools.get_cve_timeline._fetch_cisa_kev")
    @patch("manus_use.tools.get_cve_timeline._fetch_epss_spike")
    @patch("manus_use.tools.get_cve_timeline._fetch_trickest_poc")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_ghsa_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_info")
    def test_velocity_disclosure_to_poc(
        self, m_nvd, m_ghsa, m_nvd_patch, m_trickest, m_epss, m_kev
    ):
        from manus_use.tools.get_cve_timeline import build_cve_timeline

        m_nvd.return_value = {"published": "2024-03-29", "nvd_url": "https://nvd.nist.gov/"}
        m_ghsa.return_value = (None, None)
        m_nvd_patch.return_value = (None, None)
        m_trickest.return_value = ("2024-04-05", "https://github.com/poc/repo")  # +7 days
        m_epss.return_value = (None, 0.0)
        m_kev.return_value = (None, None)

        result = build_cve_timeline("CVE-2024-3094")
        assert result["velocity"]["disclosure_to_poc_days"] == 7
        assert result["velocity"]["fast_weaponisation"] is True

    @patch("manus_use.tools.get_cve_timeline._fetch_cisa_kev")
    @patch("manus_use.tools.get_cve_timeline._fetch_epss_spike")
    @patch("manus_use.tools.get_cve_timeline._fetch_trickest_poc")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_ghsa_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_info")
    def test_fast_weaponisation_flag_off_when_poc_gt_7_days(
        self, m_nvd, m_ghsa, m_nvd_patch, m_trickest, m_epss, m_kev
    ):
        from manus_use.tools.get_cve_timeline import build_cve_timeline

        m_nvd.return_value = {"published": "2024-03-29", "nvd_url": "https://nvd.nist.gov/"}
        m_ghsa.return_value = (None, None)
        m_nvd_patch.return_value = (None, None)
        m_trickest.return_value = ("2024-04-10", "https://github.com/poc/repo")  # +12 days
        m_epss.return_value = (None, 0.0)
        m_kev.return_value = (None, None)

        result = build_cve_timeline("CVE-2024-3094")
        assert result["velocity"]["fast_weaponisation"] is False

    @patch("manus_use.tools.get_cve_timeline._fetch_cisa_kev")
    @patch("manus_use.tools.get_cve_timeline._fetch_epss_spike")
    @patch("manus_use.tools.get_cve_timeline._fetch_trickest_poc")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_ghsa_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_info")
    def test_weaponized_source_prefers_kev(
        self, m_nvd, m_ghsa, m_nvd_patch, m_trickest, m_epss, m_kev
    ):
        from manus_use.tools.get_cve_timeline import build_cve_timeline

        m_nvd.return_value = {"published": "2024-01-01", "nvd_url": "https://nvd.nist.gov/"}
        m_ghsa.return_value = (None, None)
        m_nvd_patch.return_value = (None, None)
        m_trickest.return_value = (None, None)
        m_epss.return_value = ("2024-02-01", 0.20)  # spike
        m_kev.return_value = ("2024-03-01", "Patch now")  # kev

        result = build_cve_timeline("CVE-2024-3094")
        assert result["velocity"]["weaponized_source"] == "CISA KEV"

    @patch("manus_use.tools.get_cve_timeline._fetch_cisa_kev")
    @patch("manus_use.tools.get_cve_timeline._fetch_epss_spike")
    @patch("manus_use.tools.get_cve_timeline._fetch_trickest_poc")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_ghsa_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_info")
    def test_weaponized_source_falls_back_to_epss(
        self, m_nvd, m_ghsa, m_nvd_patch, m_trickest, m_epss, m_kev
    ):
        from manus_use.tools.get_cve_timeline import build_cve_timeline

        m_nvd.return_value = {"published": "2024-01-01", "nvd_url": "https://nvd.nist.gov/"}
        m_ghsa.return_value = (None, None)
        m_nvd_patch.return_value = (None, None)
        m_trickest.return_value = (None, None)
        m_epss.return_value = ("2024-02-01", 0.20)
        m_kev.return_value = (None, None)

        result = build_cve_timeline("CVE-2024-3094")
        assert result["velocity"]["weaponized_source"] == "EPSS spike"

    @patch("manus_use.tools.get_cve_timeline._fetch_cisa_kev")
    @patch("manus_use.tools.get_cve_timeline._fetch_epss_spike")
    @patch("manus_use.tools.get_cve_timeline._fetch_trickest_poc")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_ghsa_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_info")
    def test_no_events_when_all_sources_fail(
        self, m_nvd, m_ghsa, m_nvd_patch, m_trickest, m_epss, m_kev
    ):
        from manus_use.tools.get_cve_timeline import build_cve_timeline

        m_nvd.return_value = {}
        m_ghsa.return_value = (None, None)
        m_nvd_patch.return_value = (None, None)
        m_trickest.return_value = (None, None)
        m_epss.return_value = (None, 0.0)
        m_kev.return_value = (None, None)

        result = build_cve_timeline("CVE-2024-9999")
        assert result["events"] == []
        assert result["velocity"] == {}

    @patch("manus_use.tools.get_cve_timeline._fetch_cisa_kev")
    @patch("manus_use.tools.get_cve_timeline._fetch_epss_spike")
    @patch("manus_use.tools.get_cve_timeline._fetch_trickest_poc")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_ghsa_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_info")
    def test_kev_date_from_nvd_when_cisa_missing(
        self, m_nvd, m_ghsa, m_nvd_patch, m_trickest, m_epss, m_kev
    ):
        from manus_use.tools.get_cve_timeline import build_cve_timeline

        m_nvd.return_value = {
            "published": "2024-01-01",
            "kev_date": "2024-02-01",
            "nvd_url": "https://nvd.nist.gov/",
        }
        m_ghsa.return_value = (None, None)
        m_nvd_patch.return_value = (None, None)
        m_trickest.return_value = (None, None)
        m_epss.return_value = (None, 0.0)
        m_kev.return_value = (None, None)  # CISA lookup fails

        result = build_cve_timeline("CVE-2024-3094")
        event_names = [e["event"] for e in result["events"]]
        assert any("KEV" in n for n in event_names)

    @patch("manus_use.tools.get_cve_timeline._fetch_cisa_kev")
    @patch("manus_use.tools.get_cve_timeline._fetch_epss_spike")
    @patch("manus_use.tools.get_cve_timeline._fetch_trickest_poc")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_ghsa_patch_date")
    @patch("manus_use.tools.get_cve_timeline._fetch_nvd_info")
    def test_cve_id_uppercased(self, m_nvd, m_ghsa, m_nvd_patch, m_trickest, m_epss, m_kev):
        from manus_use.tools.get_cve_timeline import build_cve_timeline

        m_nvd.return_value = {}
        m_ghsa.return_value = (None, None)
        m_nvd_patch.return_value = (None, None)
        m_trickest.return_value = (None, None)
        m_epss.return_value = (None, 0.0)
        m_kev.return_value = (None, None)

        result = build_cve_timeline("cve-2024-3094")
        assert result["cve_id"] == "CVE-2024-3094"


# ---------------------------------------------------------------------------
# Tests for render_timeline_text
# ---------------------------------------------------------------------------


class TestRenderTimelineText:
    def _timeline(self, events=None, velocity=None):
        return {
            "cve_id": "CVE-2024-3094",
            "events": events or [],
            "velocity": velocity or {},
            "current_date": "2026-06-27",
        }

    def test_renders_no_events_message(self):
        from manus_use.tools.get_cve_timeline import render_timeline_text

        text = render_timeline_text(self._timeline())
        assert "No timeline events found" in text

    def test_renders_events(self):
        from manus_use.tools.get_cve_timeline import render_timeline_text

        events = [
            {"event": "CVE Disclosed", "date": "2024-03-29", "source": "NVD", "url": "https://nvd.nist.gov/"},
            {"event": "First Public PoC", "date": "2024-03-31", "source": "Trickest", "url": None},
        ]
        text = render_timeline_text(self._timeline(events=events))
        assert "2024-03-29" in text
        assert "CVE Disclosed" in text
        assert "First Public PoC" in text

    def test_renders_velocity_metrics(self):
        from manus_use.tools.get_cve_timeline import render_timeline_text

        velocity = {
            "days_since_disclosure": 820,
            "disclosure_to_poc_days": 2,
            "fast_weaponisation": True,
            "disclosure_to_weaponized_days": 14,
            "weaponized_source": "CISA KEV",
        }
        text = render_timeline_text(self._timeline(velocity=velocity))
        assert "FAST" in text
        assert "820" in text

    def test_renders_insufficient_data_message_when_no_velocity(self):
        from manus_use.tools.get_cve_timeline import render_timeline_text

        text = render_timeline_text(self._timeline())
        assert "Insufficient data" in text


# ---------------------------------------------------------------------------
# Tests for get_cve_timeline (Strands entry point)
# ---------------------------------------------------------------------------


class TestGetCveTimelineEntryPoint:
    @patch("manus_use.tools.get_cve_timeline.build_cve_timeline")
    def test_success_text_output(self, mock_build):
        from manus_use.tools.get_cve_timeline import get_cve_timeline

        mock_build.return_value = {
            "cve_id": "CVE-2024-3094",
            "events": [],
            "velocity": {},
            "current_date": "2026-06-27",
        }

        result = get_cve_timeline(_tool_use("CVE-2024-3094"))
        assert result["status"] == "success"
        assert any("text" in c for c in result["content"])
        assert any("json" in c for c in result["content"])
        mock_build.assert_called_once_with("CVE-2024-3094")

    def test_invalid_cve_id_returns_error(self):
        from manus_use.tools.get_cve_timeline import get_cve_timeline

        result = get_cve_timeline(_tool_use("NOT-A-CVE"))
        assert result["status"] == "error"

    def test_empty_cve_id_returns_error(self):
        from manus_use.tools.get_cve_timeline import get_cve_timeline

        result = get_cve_timeline({"toolUseId": "tu-1", "input": {"cve_id": ""}})
        assert result["status"] == "error"

    def test_lowercase_cve_id_accepted(self):
        from manus_use.tools.get_cve_timeline import get_cve_timeline

        with patch("manus_use.tools.get_cve_timeline.build_cve_timeline") as mock_build:
            mock_build.return_value = {
                "cve_id": "CVE-2024-3094",
                "events": [],
                "velocity": {},
                "current_date": "2026-06-27",
            }
            result = get_cve_timeline(_tool_use("cve-2024-3094"))
        assert result["status"] == "success"

    @patch("manus_use.tools.get_cve_timeline.build_cve_timeline")
    def test_json_content_block_present(self, mock_build):
        from manus_use.tools.get_cve_timeline import get_cve_timeline

        timeline_data = {
            "cve_id": "CVE-2024-3094",
            "events": [],
            "velocity": {"days_since_disclosure": 100},
            "current_date": "2026-06-27",
        }
        mock_build.return_value = timeline_data

        result = get_cve_timeline(_tool_use("CVE-2024-3094"))
        json_blocks = [c for c in result["content"] if "json" in c]
        assert json_blocks
        assert json_blocks[0]["json"]["velocity"]["days_since_disclosure"] == 100


# ---------------------------------------------------------------------------
# Tests for CLI subcommand (manus-use timeline)
# ---------------------------------------------------------------------------


class TestCliTimeline:
    def test_timeline_registered_in_subcommands(self):
        from manus_use.cli import _SUBCOMMANDS

        assert "timeline" in _SUBCOMMANDS

    def test_timeline_help_exits_zero(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "manus_use.cli", "timeline", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "CVE-ID" in result.stdout or "cve" in result.stdout.lower()

    def test_timeline_missing_cve_is_error(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "manus_use.cli", "timeline"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_timeline_invalid_cve_is_error(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "manus_use.cli", "timeline", "NOT-A-CVE"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    @patch("manus_use.tools.get_cve_timeline.build_cve_timeline")
    def test_run_timeline_text_output(self, mock_build):
        from manus_use.cli import _run_timeline

        mock_build.return_value = {
            "cve_id": "CVE-2024-3094",
            "events": [],
            "velocity": {},
            "current_date": "2026-06-27",
        }

        with patch("builtins.print"):
            ret = _run_timeline(["CVE-2024-3094"])
        assert ret == 0

    @patch("manus_use.tools.get_cve_timeline.build_cve_timeline")
    def test_run_timeline_json_output(self, mock_build):
        import io
        import json as _json

        from manus_use.cli import _run_timeline

        mock_build.return_value = {
            "cve_id": "CVE-2024-3094",
            "events": [],
            "velocity": {"days_since_disclosure": 99},
            "current_date": "2026-06-27",
        }

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            ret = _run_timeline(["CVE-2024-3094", "--output", "json"])
        assert ret == 0
        data = _json.loads(captured.getvalue())
        assert data["cve_id"] == "CVE-2024-3094"

    def test_run_timeline_invalid_cve_returns_1(self):
        from manus_use.cli import _run_timeline

        ret = _run_timeline(["NOT-A-CVE"])
        assert ret == 1


# ---------------------------------------------------------------------------
# Tests for VI agent wiring
# ---------------------------------------------------------------------------


class TestViAgentWiring:
    def test_get_cve_timeline_imported_in_vi_agent(self):
        """Verify the vi_agent module references get_cve_timeline."""
        import inspect

        import manus_use.agents.vi_agent as vi_mod

        src = inspect.getsource(vi_mod)
        assert "get_cve_timeline" in src

    def test_get_cve_timeline_in_tool_list(self):
        """Verify get_cve_timeline appears in the agent tools list source."""
        import inspect

        import manus_use.agents.vi_agent as vi_mod

        src = inspect.getsource(vi_mod)
        assert src.count("get_cve_timeline") >= 2  # import + list

    def test_system_prompt_mentions_timeline(self):
        """Verify the SYSTEM_PROMPT constant mentions the new tool."""
        import manus_use.agents.vi_agent as vi_mod

        assert "get_cve_timeline" in vi_mod.SYSTEM_PROMPT

    def test_tool_spec_name_matches(self):
        from manus_use.tools.get_cve_timeline import TOOL_SPEC

        assert TOOL_SPEC["name"] == "get_cve_timeline"

    def test_tool_spec_has_cve_id_input(self):
        from manus_use.tools.get_cve_timeline import TOOL_SPEC

        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert "cve_id" in props
        assert "cve_id" in TOOL_SPEC["inputSchema"]["json"]["required"]
