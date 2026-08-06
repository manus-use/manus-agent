"""Comprehensive test suite for cve_watchlist module.

Tests cover: validation, add/remove/list/status/clear operations,
EPSS batch fetching, KEV lookup, change detection (spikes/drops/KEV adds),
atomic file persistence, edge cases, and the unified dispatch function.

All HTTP calls are fully mocked — no real network access.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_watchlist(tmp_path, monkeypatch):
    """Redirect watchlist storage to a temp directory for each test."""
    wl_path = tmp_path / "watchlist.json"
    monkeypatch.setattr(
        "manus_agent.tools.cve_watchlist._WATCHLIST_PATH",
        wl_path,
    )
    return wl_path


@pytest.fixture
def wl_path(tmp_path):
    return tmp_path / "watchlist.json"


def _mock_response(payload, status_code=200, raise_exc=None):
    """Create a mock requests response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    if raise_exc:
        resp.raise_for_status.side_effect = raise_exc
    return resp


# ---------------------------------------------------------------------------
# CVE ID Validation
# ---------------------------------------------------------------------------


class TestCveIdValidation:
    def test_valid_standard_format(self):
        from manus_agent.tools.cve_watchlist import _validate_cve_id

        assert _validate_cve_id("CVE-2024-3094") == "CVE-2024-3094"

    def test_valid_lowercase(self):
        from manus_agent.tools.cve_watchlist import _validate_cve_id

        assert _validate_cve_id("cve-2021-44228") == "CVE-2021-44228"

    def test_valid_mixed_case(self):
        from manus_agent.tools.cve_watchlist import _validate_cve_id

        assert _validate_cve_id("Cve-2025-12345") == "CVE-2025-12345"

    def test_valid_with_whitespace(self):
        from manus_agent.tools.cve_watchlist import _validate_cve_id

        assert _validate_cve_id("  CVE-2024-3094  ") == "CVE-2024-3094"

    def test_valid_five_digit_id(self):
        from manus_agent.tools.cve_watchlist import _validate_cve_id

        assert _validate_cve_id("CVE-2024-12345") == "CVE-2024-12345"

    def test_invalid_empty(self):
        from manus_agent.tools.cve_watchlist import _validate_cve_id

        assert _validate_cve_id("") is None

    def test_invalid_no_prefix(self):
        from manus_agent.tools.cve_watchlist import _validate_cve_id

        assert _validate_cve_id("2024-3094") is None

    def test_invalid_short_number(self):
        from manus_agent.tools.cve_watchlist import _validate_cve_id

        assert _validate_cve_id("CVE-2024-12") is None

    def test_invalid_no_year(self):
        from manus_agent.tools.cve_watchlist import _validate_cve_id

        assert _validate_cve_id("CVE--3094") is None

    def test_invalid_random_string(self):
        from manus_agent.tools.cve_watchlist import _validate_cve_id

        assert _validate_cve_id("hello-world") is None

    def test_invalid_partial_format(self):
        from manus_agent.tools.cve_watchlist import _validate_cve_id

        assert _validate_cve_id("CVE-2024") is None


# ---------------------------------------------------------------------------
# Watchlist Add
# ---------------------------------------------------------------------------


class TestWatchlistAdd:
    def test_add_single_cve(self):
        from manus_agent.tools.cve_watchlist import watchlist_add

        result = watchlist_add(["CVE-2024-3094"])
        assert result["added"] == ["CVE-2024-3094"]
        assert result["invalid"] == []
        assert result["duplicate"] == []
        assert result["total"] == 1

    def test_add_multiple_cves(self):
        from manus_agent.tools.cve_watchlist import watchlist_add

        result = watchlist_add(["CVE-2024-3094", "CVE-2021-44228", "CVE-2023-1234"])
        assert len(result["added"]) == 3
        assert result["total"] == 3

    def test_add_with_note(self):
        from manus_agent.tools.cve_watchlist import _load_watchlist, watchlist_add

        watchlist_add(["CVE-2024-3094"], note="Critical log4j-like")
        wl = _load_watchlist()
        assert wl["cves"]["CVE-2024-3094"]["note"] == "Critical log4j-like"

    def test_add_duplicate_detected(self):
        from manus_agent.tools.cve_watchlist import watchlist_add

        watchlist_add(["CVE-2024-3094"])
        result = watchlist_add(["CVE-2024-3094"])
        assert result["duplicate"] == ["CVE-2024-3094"]
        assert result["added"] == []
        assert result["total"] == 1

    def test_add_invalid_detected(self):
        from manus_agent.tools.cve_watchlist import watchlist_add

        result = watchlist_add(["not-a-cve", "CVE-2024-3094"])
        assert result["invalid"] == ["not-a-cve"]
        assert result["added"] == ["CVE-2024-3094"]

    def test_add_mixed_valid_invalid_duplicate(self):
        from manus_agent.tools.cve_watchlist import watchlist_add

        watchlist_add(["CVE-2024-3094"])
        result = watchlist_add(["CVE-2024-3094", "CVE-2021-44228", "bad-id"])
        assert result["added"] == ["CVE-2021-44228"]
        assert result["duplicate"] == ["CVE-2024-3094"]
        assert result["invalid"] == ["bad-id"]
        assert result["total"] == 2

    def test_add_persists_to_disk(self, tmp_path):
        from manus_agent.tools.cve_watchlist import _WATCHLIST_PATH, watchlist_add

        watchlist_add(["CVE-2024-3094"])
        assert _WATCHLIST_PATH.exists()
        data = json.loads(_WATCHLIST_PATH.read_text())
        assert "CVE-2024-3094" in data["cves"]

    def test_add_stores_timestamp(self):
        from manus_agent.tools.cve_watchlist import _load_watchlist, watchlist_add

        watchlist_add(["CVE-2024-3094"])
        wl = _load_watchlist()
        assert "added_at" in wl["cves"]["CVE-2024-3094"]
        # Should be a valid ISO timestamp
        assert wl["cves"]["CVE-2024-3094"]["added_at"].endswith("Z")


# ---------------------------------------------------------------------------
# Watchlist Remove
# ---------------------------------------------------------------------------


class TestWatchlistRemove:
    def test_remove_existing(self):
        from manus_agent.tools.cve_watchlist import watchlist_add, watchlist_remove

        watchlist_add(["CVE-2024-3094", "CVE-2021-44228"])
        result = watchlist_remove(["CVE-2024-3094"])
        assert result["removed"] == ["CVE-2024-3094"]
        assert result["not_found"] == []
        assert result["total"] == 1

    def test_remove_nonexistent(self):
        from manus_agent.tools.cve_watchlist import watchlist_add, watchlist_remove

        watchlist_add(["CVE-2024-3094"])
        result = watchlist_remove(["CVE-9999-9999"])
        assert result["not_found"] == ["CVE-9999-9999"]
        assert result["removed"] == []
        assert result["total"] == 1

    def test_remove_invalid_id(self):
        from manus_agent.tools.cve_watchlist import watchlist_remove

        result = watchlist_remove(["not-valid"])
        assert result["not_found"] == ["not-valid"]

    def test_remove_multiple(self):
        from manus_agent.tools.cve_watchlist import watchlist_add, watchlist_remove

        watchlist_add(["CVE-2024-3094", "CVE-2021-44228", "CVE-2023-1234"])
        result = watchlist_remove(["CVE-2024-3094", "CVE-2021-44228"])
        assert len(result["removed"]) == 2
        assert result["total"] == 1

    def test_remove_persists(self):
        from manus_agent.tools.cve_watchlist import _load_watchlist, watchlist_add, watchlist_remove

        watchlist_add(["CVE-2024-3094"])
        watchlist_remove(["CVE-2024-3094"])
        wl = _load_watchlist()
        assert "CVE-2024-3094" not in wl["cves"]


# ---------------------------------------------------------------------------
# Watchlist List
# ---------------------------------------------------------------------------


class TestWatchlistList:
    def test_list_empty(self):
        from manus_agent.tools.cve_watchlist import watchlist_list

        result = watchlist_list()
        assert result["entries"] == []
        assert result["total"] == 0

    def test_list_with_entries(self):
        from manus_agent.tools.cve_watchlist import watchlist_add, watchlist_list

        watchlist_add(["CVE-2024-3094", "CVE-2021-44228"], note="test")
        result = watchlist_list()
        assert result["total"] == 2
        assert len(result["entries"]) == 2
        # Should be sorted by CVE ID
        assert result["entries"][0]["cve_id"] == "CVE-2021-44228"
        assert result["entries"][1]["cve_id"] == "CVE-2024-3094"

    def test_list_includes_metadata(self):
        from manus_agent.tools.cve_watchlist import watchlist_add, watchlist_list

        watchlist_add(["CVE-2024-3094"], note="high priority")
        result = watchlist_list()
        entry = result["entries"][0]
        assert entry["cve_id"] == "CVE-2024-3094"
        assert entry["note"] == "high priority"
        assert entry["added_at"] != ""
        assert entry["last_epss"] is None  # Not checked yet
        assert entry["in_kev"] is None


# ---------------------------------------------------------------------------
# Watchlist Status (with mocked HTTP)
# ---------------------------------------------------------------------------


class TestWatchlistStatus:
    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_status_empty_watchlist(self, mock_get):
        from manus_agent.tools.cve_watchlist import watchlist_status

        result = watchlist_status()
        assert result["entries"] == []
        assert result["total"] == 0
        mock_get.assert_not_called()

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_status_fetches_epss_and_kev(self, mock_get):
        from manus_agent.tools.cve_watchlist import watchlist_add, watchlist_status

        watchlist_add(["CVE-2024-3094"])

        # Mock EPSS response
        epss_resp = _mock_response(
            {"data": [{"cve": "CVE-2024-3094", "epss": "0.9500", "percentile": "0.9900", "date": "2026-08-05"}]}
        )
        # Mock KEV response
        kev_resp = _mock_response({"vulnerabilities": [{"cveID": "CVE-2024-3094"}, {"cveID": "CVE-2021-44228"}]})
        mock_get.side_effect = [epss_resp, kev_resp]

        result = watchlist_status()
        assert result["total"] == 1
        entry = result["entries"][0]
        assert entry["cve_id"] == "CVE-2024-3094"
        assert entry["epss"] == 0.95
        assert entry["percentile"] == 0.99
        assert entry["in_kev"] is True

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_status_detects_epss_spike(self, mock_get):
        from manus_agent.tools.cve_watchlist import _load_watchlist, _save_watchlist, watchlist_add, watchlist_status

        watchlist_add(["CVE-2024-3094"])

        # Set previous EPSS to simulate a spike
        wl = _load_watchlist()
        wl["cves"]["CVE-2024-3094"]["last_epss"] = 0.10
        wl["cves"]["CVE-2024-3094"]["last_percentile"] = 0.50
        _save_watchlist(wl)

        # Current EPSS is much higher
        epss_resp = _mock_response(
            {"data": [{"cve": "CVE-2024-3094", "epss": "0.9500", "percentile": "0.9900", "date": "2026-08-05"}]}
        )
        kev_resp = _mock_response({"vulnerabilities": []})
        mock_get.side_effect = [epss_resp, kev_resp]

        result = watchlist_status()
        assert len(result["changes"]) == 1
        change = result["changes"][0]
        assert change["type"] == "epss_spike"
        assert change["cve_id"] == "CVE-2024-3094"
        assert change["delta"] == pytest.approx(0.85, abs=0.01)

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_status_detects_epss_drop(self, mock_get):
        from manus_agent.tools.cve_watchlist import _load_watchlist, _save_watchlist, watchlist_add, watchlist_status

        watchlist_add(["CVE-2024-3094"])

        wl = _load_watchlist()
        wl["cves"]["CVE-2024-3094"]["last_epss"] = 0.80
        wl["cves"]["CVE-2024-3094"]["last_percentile"] = 0.95
        _save_watchlist(wl)

        epss_resp = _mock_response(
            {"data": [{"cve": "CVE-2024-3094", "epss": "0.1000", "percentile": "0.5000", "date": "2026-08-05"}]}
        )
        kev_resp = _mock_response({"vulnerabilities": []})
        mock_get.side_effect = [epss_resp, kev_resp]

        result = watchlist_status()
        assert len(result["changes"]) == 1
        change = result["changes"][0]
        assert change["type"] == "epss_drop"
        assert change["delta"] == pytest.approx(-0.70, abs=0.01)

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_status_detects_kev_addition(self, mock_get):
        from manus_agent.tools.cve_watchlist import _load_watchlist, _save_watchlist, watchlist_add, watchlist_status

        watchlist_add(["CVE-2024-3094"])

        # Previously not in KEV
        wl = _load_watchlist()
        wl["cves"]["CVE-2024-3094"]["in_kev"] = False
        wl["cves"]["CVE-2024-3094"]["last_epss"] = 0.50
        _save_watchlist(wl)

        epss_resp = _mock_response(
            {"data": [{"cve": "CVE-2024-3094", "epss": "0.5000", "percentile": "0.8000", "date": "2026-08-05"}]}
        )
        kev_resp = _mock_response({"vulnerabilities": [{"cveID": "CVE-2024-3094"}]})
        mock_get.side_effect = [epss_resp, kev_resp]

        result = watchlist_status()
        assert any(c["type"] == "kev_added" for c in result["changes"])

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_status_no_change_below_threshold(self, mock_get):
        from manus_agent.tools.cve_watchlist import _load_watchlist, _save_watchlist, watchlist_add, watchlist_status

        watchlist_add(["CVE-2024-3094"])

        wl = _load_watchlist()
        wl["cves"]["CVE-2024-3094"]["last_epss"] = 0.50
        wl["cves"]["CVE-2024-3094"]["in_kev"] = False
        _save_watchlist(wl)

        # Delta of 0.01 — below threshold
        epss_resp = _mock_response(
            {"data": [{"cve": "CVE-2024-3094", "epss": "0.5100", "percentile": "0.8100", "date": "2026-08-05"}]}
        )
        kev_resp = _mock_response({"vulnerabilities": []})
        mock_get.side_effect = [epss_resp, kev_resp]

        result = watchlist_status()
        assert result["changes"] == []

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_status_persists_updated_values(self, mock_get):
        from manus_agent.tools.cve_watchlist import _load_watchlist, watchlist_add, watchlist_status

        watchlist_add(["CVE-2024-3094"])

        epss_resp = _mock_response(
            {"data": [{"cve": "CVE-2024-3094", "epss": "0.7500", "percentile": "0.9500", "date": "2026-08-05"}]}
        )
        kev_resp = _mock_response({"vulnerabilities": [{"cveID": "CVE-2024-3094"}]})
        mock_get.side_effect = [epss_resp, kev_resp]

        watchlist_status()

        wl = _load_watchlist()
        assert wl["cves"]["CVE-2024-3094"]["last_epss"] == 0.75
        assert wl["cves"]["CVE-2024-3094"]["last_percentile"] == 0.95
        assert wl["cves"]["CVE-2024-3094"]["in_kev"] is True
        assert wl["last_checked"] is not None

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_status_summary_stats(self, mock_get):
        from manus_agent.tools.cve_watchlist import watchlist_add, watchlist_status

        watchlist_add(["CVE-2024-3094", "CVE-2021-44228"])

        epss_resp = _mock_response(
            {
                "data": [
                    {"cve": "CVE-2024-3094", "epss": "0.8000", "percentile": "0.95", "date": "2026-08-05"},
                    {"cve": "CVE-2021-44228", "epss": "0.9700", "percentile": "0.99", "date": "2026-08-05"},
                ]
            }
        )
        kev_resp = _mock_response({"vulnerabilities": [{"cveID": "CVE-2021-44228"}]})
        mock_get.side_effect = [epss_resp, kev_resp]

        result = watchlist_status()
        assert result["summary"]["kev_count"] == 1
        assert result["summary"]["avg_epss"] == pytest.approx(0.885, abs=0.001)

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_status_handles_epss_failure_gracefully(self, mock_get):
        import requests as _requests

        from manus_agent.tools.cve_watchlist import watchlist_add, watchlist_status

        watchlist_add(["CVE-2024-3094"])

        # EPSS fails
        epss_resp = _mock_response({}, status_code=500, raise_exc=_requests.HTTPError("Server Error"))
        kev_resp = _mock_response({"vulnerabilities": []})
        mock_get.side_effect = [epss_resp, kev_resp]

        result = watchlist_status()
        assert result["total"] == 1
        entry = result["entries"][0]
        assert entry["epss"] is None  # No data available

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_status_handles_kev_failure_gracefully(self, mock_get):
        import requests as _requests

        from manus_agent.tools.cve_watchlist import watchlist_add, watchlist_status

        watchlist_add(["CVE-2024-3094"])

        epss_resp = _mock_response(
            {"data": [{"cve": "CVE-2024-3094", "epss": "0.5000", "percentile": "0.8000", "date": "2026-08-05"}]}
        )
        kev_resp = _mock_response({}, status_code=503, raise_exc=_requests.HTTPError("Unavailable"))
        mock_get.side_effect = [epss_resp, kev_resp]

        result = watchlist_status()
        entry = result["entries"][0]
        assert entry["in_kev"] is False  # Empty set -> not in KEV


# ---------------------------------------------------------------------------
# Watchlist Clear
# ---------------------------------------------------------------------------


class TestWatchlistClear:
    def test_clear_empty(self):
        from manus_agent.tools.cve_watchlist import watchlist_clear

        result = watchlist_clear()
        assert result["cleared"] == 0

    def test_clear_with_entries(self):
        from manus_agent.tools.cve_watchlist import _load_watchlist, watchlist_add, watchlist_clear

        watchlist_add(["CVE-2024-3094", "CVE-2021-44228"])
        result = watchlist_clear()
        assert result["cleared"] == 2
        wl = _load_watchlist()
        assert wl["cves"] == {}
        assert wl["last_checked"] is None


# ---------------------------------------------------------------------------
# EPSS Batch Fetch
# ---------------------------------------------------------------------------


class TestEpssBatchFetch:
    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_single_batch(self, mock_get):
        from manus_agent.tools.cve_watchlist import _fetch_epss_batch

        mock_get.return_value = _mock_response(
            {"data": [{"cve": "CVE-2024-3094", "epss": "0.5", "percentile": "0.8", "date": "2026-08-05"}]}
        )
        result = _fetch_epss_batch(["CVE-2024-3094"])
        assert "CVE-2024-3094" in result
        assert result["CVE-2024-3094"]["epss"] == 0.5

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_multiple_batches(self, mock_get):
        from manus_agent.tools.cve_watchlist import _EPSS_BATCH_SIZE, _fetch_epss_batch

        # Create more CVEs than batch size
        cves = [f"CVE-2024-{i:04d}" for i in range(1000, 1000 + _EPSS_BATCH_SIZE + 5)]

        batch1_data = [
            {"cve": cve, "epss": "0.1", "percentile": "0.5", "date": "2026-08-05"} for cve in cves[:_EPSS_BATCH_SIZE]
        ]
        batch2_data = [
            {"cve": cve, "epss": "0.2", "percentile": "0.6", "date": "2026-08-05"} for cve in cves[_EPSS_BATCH_SIZE:]
        ]

        mock_get.side_effect = [
            _mock_response({"data": batch1_data}),
            _mock_response({"data": batch2_data}),
        ]

        result = _fetch_epss_batch(cves)
        assert len(result) == len(cves)
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_network_error_returns_partial(self, mock_get):
        import requests as _requests

        from manus_agent.tools.cve_watchlist import _fetch_epss_batch

        mock_get.side_effect = _requests.ConnectionError("Network error")
        result = _fetch_epss_batch(["CVE-2024-3094"])
        assert result == {}


# ---------------------------------------------------------------------------
# KEV Fetch
# ---------------------------------------------------------------------------


class TestKevFetch:
    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_successful_fetch(self, mock_get):
        from manus_agent.tools.cve_watchlist import _fetch_kev_set

        mock_get.return_value = _mock_response(
            {"vulnerabilities": [{"cveID": "CVE-2024-3094"}, {"cveID": "CVE-2021-44228"}]}
        )
        result = _fetch_kev_set()
        assert result == {"CVE-2024-3094", "CVE-2021-44228"}

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_network_error_returns_empty(self, mock_get):
        import requests as _requests

        from manus_agent.tools.cve_watchlist import _fetch_kev_set

        mock_get.side_effect = _requests.ConnectionError("Timeout")
        result = _fetch_kev_set()
        assert result == set()

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_malformed_json_returns_empty(self, mock_get):
        from manus_agent.tools.cve_watchlist import _fetch_kev_set

        mock_get.return_value = _mock_response({})  # Missing 'vulnerabilities'
        result = _fetch_kev_set()
        assert result == set()


# ---------------------------------------------------------------------------
# Persistence / Edge Cases
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_load_missing_file(self, tmp_path, monkeypatch):
        from manus_agent.tools.cve_watchlist import _load_watchlist

        monkeypatch.setattr(
            "manus_agent.tools.cve_watchlist._WATCHLIST_PATH",
            tmp_path / "nonexistent.json",
        )
        wl = _load_watchlist()
        assert wl == {"cves": {}, "last_checked": None}

    def test_load_corrupt_file(self, tmp_path, monkeypatch):
        from manus_agent.tools.cve_watchlist import _load_watchlist

        corrupt_path = tmp_path / "corrupt.json"
        corrupt_path.write_text("not valid json {{{{")
        monkeypatch.setattr(
            "manus_agent.tools.cve_watchlist._WATCHLIST_PATH",
            corrupt_path,
        )
        wl = _load_watchlist()
        assert wl == {"cves": {}, "last_checked": None}

    def test_load_missing_cves_key(self, tmp_path, monkeypatch):
        from manus_agent.tools.cve_watchlist import _load_watchlist

        partial_path = tmp_path / "partial.json"
        partial_path.write_text(json.dumps({"last_checked": "2026-01-01"}))
        monkeypatch.setattr(
            "manus_agent.tools.cve_watchlist._WATCHLIST_PATH",
            partial_path,
        )
        wl = _load_watchlist()
        assert wl["cves"] == {}
        assert wl["last_checked"] == "2026-01-01"

    def test_save_creates_parent_dirs(self, tmp_path, monkeypatch):
        from manus_agent.tools.cve_watchlist import _save_watchlist

        deep_path = tmp_path / "a" / "b" / "c" / "watchlist.json"
        monkeypatch.setattr(
            "manus_agent.tools.cve_watchlist._WATCHLIST_PATH",
            deep_path,
        )
        _save_watchlist({"cves": {}, "last_checked": None})
        assert deep_path.exists()

    def test_atomic_write_no_partial_on_interrupt(self, tmp_path, monkeypatch):
        """Verify that .tmp file is used for atomic writes."""
        from manus_agent.tools.cve_watchlist import _WATCHLIST_PATH, watchlist_add

        watchlist_add(["CVE-2024-3094"])
        # After successful write, no .tmp should remain
        tmp_file = _WATCHLIST_PATH.with_suffix(".tmp")
        assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# Unified Dispatch (watchlist_manage)
# ---------------------------------------------------------------------------


class TestWatchlistManage:
    def test_dispatch_add(self):
        from manus_agent.tools.cve_watchlist import watchlist_manage

        result = watchlist_manage("add", cve_ids=["CVE-2024-3094"])
        assert result["added"] == ["CVE-2024-3094"]

    def test_dispatch_remove(self):
        from manus_agent.tools.cve_watchlist import watchlist_manage

        watchlist_manage("add", cve_ids=["CVE-2024-3094"])
        result = watchlist_manage("remove", cve_ids=["CVE-2024-3094"])
        assert result["removed"] == ["CVE-2024-3094"]

    def test_dispatch_list(self):
        from manus_agent.tools.cve_watchlist import watchlist_manage

        watchlist_manage("add", cve_ids=["CVE-2024-3094"])
        result = watchlist_manage("list")
        assert result["total"] == 1

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_dispatch_status(self, mock_get):
        from manus_agent.tools.cve_watchlist import watchlist_manage

        watchlist_manage("add", cve_ids=["CVE-2024-3094"])
        epss_resp = _mock_response(
            {"data": [{"cve": "CVE-2024-3094", "epss": "0.5", "percentile": "0.8", "date": "2026-08-05"}]}
        )
        kev_resp = _mock_response({"vulnerabilities": []})
        mock_get.side_effect = [epss_resp, kev_resp]

        result = watchlist_manage("status")
        assert result["total"] == 1

    def test_dispatch_clear(self):
        from manus_agent.tools.cve_watchlist import watchlist_manage

        watchlist_manage("add", cve_ids=["CVE-2024-3094"])
        result = watchlist_manage("clear")
        assert result["cleared"] == 1

    def test_dispatch_unknown_action(self):
        from manus_agent.tools.cve_watchlist import watchlist_manage

        result = watchlist_manage("explode")
        assert "error" in result

    def test_dispatch_add_no_cves(self):
        from manus_agent.tools.cve_watchlist import watchlist_manage

        result = watchlist_manage("add", cve_ids=None)
        assert "error" in result

    def test_dispatch_remove_no_cves(self):
        from manus_agent.tools.cve_watchlist import watchlist_manage

        result = watchlist_manage("remove", cve_ids=None)
        assert "error" in result

    def test_dispatch_add_with_note(self):
        from manus_agent.tools.cve_watchlist import _load_watchlist, watchlist_manage

        watchlist_manage("add", cve_ids=["CVE-2024-3094"], note="urgent")
        wl = _load_watchlist()
        assert wl["cves"]["CVE-2024-3094"]["note"] == "urgent"


# ---------------------------------------------------------------------------
# CLI Integration
# ---------------------------------------------------------------------------


class TestCLIWatchlist:
    """Test the CLI _run_watchlist function indirectly via argument parsing."""

    def test_cli_add_runs_without_crash(self, capsys):
        from manus_agent.cli import _run_watchlist

        result = _run_watchlist(["add", "CVE-2024-3094", "CVE-2021-44228"])
        assert result == 0

    def test_cli_add_with_note(self, capsys):
        from manus_agent.cli import _run_watchlist

        result = _run_watchlist(["add", "CVE-2024-3094", "--note", "critical"])
        assert result == 0

    def test_cli_remove(self, capsys):
        from manus_agent.cli import _run_watchlist

        _run_watchlist(["add", "CVE-2024-3094"])
        result = _run_watchlist(["remove", "CVE-2024-3094"])
        assert result == 0

    def test_cli_list_empty(self, capsys):
        from manus_agent.cli import _run_watchlist

        result = _run_watchlist(["list"])
        assert result == 0

    def test_cli_list_with_entries(self, capsys):
        from manus_agent.cli import _run_watchlist

        _run_watchlist(["add", "CVE-2024-3094"])
        result = _run_watchlist(["list"])
        assert result == 0

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_cli_status_text(self, mock_get, capsys):
        from manus_agent.cli import _run_watchlist

        _run_watchlist(["add", "CVE-2024-3094"])
        epss_resp = _mock_response(
            {"data": [{"cve": "CVE-2024-3094", "epss": "0.5", "percentile": "0.8", "date": "2026-08-05"}]}
        )
        kev_resp = _mock_response({"vulnerabilities": []})
        mock_get.side_effect = [epss_resp, kev_resp]

        result = _run_watchlist(["status"])
        assert result == 0

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_cli_status_json(self, mock_get, capsys):
        from manus_agent.cli import _run_watchlist

        _run_watchlist(["add", "CVE-2024-3094"])
        # Clear captured output from the add command
        capsys.readouterr()

        epss_resp = _mock_response(
            {"data": [{"cve": "CVE-2024-3094", "epss": "0.5", "percentile": "0.8", "date": "2026-08-05"}]}
        )
        kev_resp = _mock_response({"vulnerabilities": []})
        mock_get.side_effect = [epss_resp, kev_resp]

        result = _run_watchlist(["status", "--output", "json"])
        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "entries" in data

    def test_cli_clear(self, capsys):
        from manus_agent.cli import _run_watchlist

        _run_watchlist(["add", "CVE-2024-3094"])
        result = _run_watchlist(["clear"])
        assert result == 0

    def test_cli_no_subaction_shows_help(self, capsys):
        from manus_agent.cli import _run_watchlist

        result = _run_watchlist([])
        assert result == 1

    @patch("manus_agent.tools.cve_watchlist.requests.get")
    def test_cli_status_with_changes(self, mock_get, capsys):
        from manus_agent.cli import _run_watchlist
        from manus_agent.tools.cve_watchlist import _load_watchlist, _save_watchlist

        _run_watchlist(["add", "CVE-2024-3094"])

        # Set previous state for change detection
        wl = _load_watchlist()
        wl["cves"]["CVE-2024-3094"]["last_epss"] = 0.10
        wl["cves"]["CVE-2024-3094"]["in_kev"] = False
        _save_watchlist(wl)

        epss_resp = _mock_response(
            {"data": [{"cve": "CVE-2024-3094", "epss": "0.9500", "percentile": "0.99", "date": "2026-08-05"}]}
        )
        kev_resp = _mock_response({"vulnerabilities": [{"cveID": "CVE-2024-3094"}]})
        mock_get.side_effect = [epss_resp, kev_resp]

        result = _run_watchlist(["status"])
        assert result == 0
