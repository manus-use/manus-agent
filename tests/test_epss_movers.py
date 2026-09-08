"""Comprehensive test suite for get_epss_movers tool module.

All HTTP calls are mocked — no real network requests are made.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from manus_agent.tools.get_epss_movers import (
    TOOL_SPEC,
    _fetch_epss_for_cves,
    _fetch_epss_snapshot,
    compute_movers,
    get_epss_movers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use(
    days: int = 7,
    top: int = 20,
    tool_use_id: str = "test-id-001",
) -> dict[str, Any]:
    """Build a minimal Strands ToolUse dict."""
    return {
        "toolUseId": tool_use_id,
        "input": {"days": days, "top": top},
    }


def _make_tool_use_empty(tool_use_id: str = "test-id-001") -> dict[str, Any]:
    """Build a ToolUse with no input (uses defaults)."""
    return {"toolUseId": tool_use_id, "input": {}}


def _make_epss_response(
    data: list[dict[str, str]],
    status_code: int = 200,
) -> MagicMock:
    """Build a mock requests.Response for the EPSS API."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = {"data": data}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=resp
        )
    return resp


def _make_snapshot_data(
    cve_scores: dict[str, float],
    target_date: str = "2026-09-07",
) -> list[dict[str, str]]:
    """Build EPSS API data entries for a snapshot."""
    return [
        {
            "cve": cve,
            "epss": str(score),
            "percentile": str(min(score * 10, 1.0)),
            "date": target_date,
        }
        for cve, score in cve_scores.items()
    ]


# ---------------------------------------------------------------------------
# TOOL_SPEC tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Verify TOOL_SPEC structure matches Strands conventions."""

    def test_name(self):
        assert TOOL_SPEC["name"] == "get_epss_movers"

    def test_has_description(self):
        assert isinstance(TOOL_SPEC["description"], str)
        assert len(TOOL_SPEC["description"]) > 20

    def test_description_mentions_emerging_threats(self):
        desc = TOOL_SPEC["description"].lower()
        assert "emerging" in desc or "movers" in desc

    def test_input_schema_has_days(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert "days" in props
        assert props["days"]["type"] == "integer"

    def test_input_schema_has_top(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert "top" in props
        assert props["top"]["type"] == "integer"

    def test_no_required_fields(self):
        required = TOOL_SPEC["inputSchema"]["json"].get("required", [])
        assert required == []

    def test_days_default(self):
        assert TOOL_SPEC["inputSchema"]["json"]["properties"]["days"]["default"] == 7

    def test_top_default(self):
        assert TOOL_SPEC["inputSchema"]["json"]["properties"]["top"]["default"] == 20


# ---------------------------------------------------------------------------
# _fetch_epss_snapshot tests
# ---------------------------------------------------------------------------


class TestFetchEpssSnapshot:
    """Tests for _fetch_epss_snapshot helper."""

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_returns_data_list(self, mock_get):
        data = _make_snapshot_data({"CVE-2024-0001": 0.95, "CVE-2024-0002": 0.90})
        mock_get.return_value = _make_epss_response(data)

        result = _fetch_epss_snapshot(date(2026, 9, 7))
        assert len(result) == 2
        assert result[0]["cve"] == "CVE-2024-0001"

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_passes_date_param(self, mock_get):
        mock_get.return_value = _make_epss_response([])
        _fetch_epss_snapshot(date(2026, 9, 7))

        call_args = mock_get.call_args
        assert call_args[0][0]["date"] == "2026-09-07"

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_passes_descending_order(self, mock_get):
        mock_get.return_value = _make_epss_response([])
        _fetch_epss_snapshot(date(2026, 9, 7))

        call_args = mock_get.call_args
        assert call_args[0][0]["order"] == "!epss"

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_limit_capped_at_100(self, mock_get):
        mock_get.return_value = _make_epss_response([])
        _fetch_epss_snapshot(date(2026, 9, 7), limit=200)

        call_args = mock_get.call_args
        assert call_args[0][0]["limit"] == 100

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_empty_response(self, mock_get):
        mock_get.return_value = _make_epss_response([])
        result = _fetch_epss_snapshot(date(2026, 9, 7))
        assert result == []

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_offset_passed(self, mock_get):
        mock_get.return_value = _make_epss_response([])
        _fetch_epss_snapshot(date(2026, 9, 7), offset=50)

        call_args = mock_get.call_args
        assert call_args[0][0]["offset"] == 50


# ---------------------------------------------------------------------------
# _fetch_epss_for_cves tests
# ---------------------------------------------------------------------------


class TestFetchEpssForCves:
    """Tests for _fetch_epss_for_cves helper."""

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_returns_score_mapping(self, mock_get):
        data = [
            {"cve": "CVE-2024-0001", "epss": "0.50", "percentile": "0.80"},
            {"cve": "CVE-2024-0002", "epss": "0.30", "percentile": "0.60"},
        ]
        mock_get.return_value = _make_epss_response(data)

        result = _fetch_epss_for_cves(
            ["CVE-2024-0001", "CVE-2024-0002"], date(2026, 9, 1)
        )
        assert result["CVE-2024-0001"] == 0.50
        assert result["CVE-2024-0002"] == 0.30

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_missing_cves_omitted(self, mock_get):
        data = [{"cve": "CVE-2024-0001", "epss": "0.50", "percentile": "0.80"}]
        mock_get.return_value = _make_epss_response(data)

        result = _fetch_epss_for_cves(
            ["CVE-2024-0001", "CVE-2024-0099"], date(2026, 9, 1)
        )
        assert "CVE-2024-0001" in result
        assert "CVE-2024-0099" not in result

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_batches_large_lists(self, mock_get):
        """CVE lists > 30 should be split into batches."""
        cve_list = [f"CVE-2024-{i:04d}" for i in range(45)]
        data = [{"cve": c, "epss": "0.10", "percentile": "0.50"} for c in cve_list[:30]]
        mock_get.return_value = _make_epss_response(data)

        _fetch_epss_for_cves(cve_list, date(2026, 9, 1))
        assert mock_get.call_count == 2  # 30 + 15

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_passes_date_to_api(self, mock_get):
        mock_get.return_value = _make_epss_response([])
        _fetch_epss_for_cves(["CVE-2024-0001"], date(2026, 8, 15))

        call_args = mock_get.call_args
        assert call_args[0][0]["date"] == "2026-08-15"

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_uppercases_cve_ids(self, mock_get):
        data = [{"cve": "cve-2024-0001", "epss": "0.50", "percentile": "0.80"}]
        mock_get.return_value = _make_epss_response(data)

        result = _fetch_epss_for_cves(["cve-2024-0001"], date(2026, 9, 1))
        assert "CVE-2024-0001" in result

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_invalid_epss_defaults_to_zero(self, mock_get):
        data = [{"cve": "CVE-2024-0001", "epss": "not-a-number", "percentile": "0.80"}]
        mock_get.return_value = _make_epss_response(data)

        result = _fetch_epss_for_cves(["CVE-2024-0001"], date(2026, 9, 1))
        assert result["CVE-2024-0001"] == 0.0

    @patch("manus_agent.tools.get_epss_movers._epss_get_with_retry")
    def test_empty_cve_list(self, mock_get):
        result = _fetch_epss_for_cves([], date(2026, 9, 1))
        assert result == {}
        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# compute_movers tests
# ---------------------------------------------------------------------------


class TestComputeMovers:
    """Tests for the core compute_movers function."""

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_basic_movers(self, mock_snapshot, mock_for_cves):
        """Two CVEs both increase, should be returned sorted by delta."""
        mock_snapshot.return_value = _make_snapshot_data(
            {"CVE-2024-0001": 0.90, "CVE-2024-0002": 0.80},
            "2026-09-07",
        )
        mock_for_cves.return_value = {
            "CVE-2024-0001": 0.50,
            "CVE-2024-0002": 0.70,
        }

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        assert result["error"] is None
        movers = result["movers"]
        assert len(movers) == 2
        # CVE-0001 had bigger delta (0.40 vs 0.10)
        assert movers[0]["cve"] == "CVE-2024-0001"
        assert movers[0]["delta"] == pytest.approx(0.40, abs=0.001)
        assert movers[1]["cve"] == "CVE-2024-0002"
        assert movers[1]["delta"] == pytest.approx(0.10, abs=0.001)

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_filters_below_min_delta(self, mock_snapshot, mock_for_cves):
        """CVEs with delta below min_delta are excluded."""
        mock_snapshot.return_value = _make_snapshot_data(
            {"CVE-2024-0001": 0.50, "CVE-2024-0002": 0.50001},
            "2026-09-07",
        )
        mock_for_cves.return_value = {
            "CVE-2024-0001": 0.10,
            "CVE-2024-0002": 0.50,  # delta ~= 0.00001 — below threshold
        }

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
            min_delta=0.01,
        )

        movers = result["movers"]
        assert len(movers) == 1
        assert movers[0]["cve"] == "CVE-2024-0001"

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_top_limit(self, mock_snapshot, mock_for_cves):
        """Only returns top N movers."""
        cve_scores = {f"CVE-2024-{i:04d}": 0.90 - i * 0.01 for i in range(10)}
        mock_snapshot.return_value = _make_snapshot_data(cve_scores, "2026-09-07")
        mock_for_cves.return_value = {
            cve: 0.10 for cve in cve_scores
        }

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
            top=3,
        )

        assert len(result["movers"]) == 3
        # Highest deltas should come first
        assert result["movers"][0]["delta"] >= result["movers"][1]["delta"]

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_new_cve_flagged(self, mock_snapshot, mock_for_cves):
        """CVEs not in baseline should have is_new=True and pct_change=None."""
        mock_snapshot.return_value = _make_snapshot_data(
            {"CVE-2024-0001": 0.80},
            "2026-09-07",
        )
        mock_for_cves.return_value = {}  # not found in baseline

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        movers = result["movers"]
        assert len(movers) == 1
        assert movers[0]["is_new"] is True
        assert movers[0]["pct_change"] is None
        assert movers[0]["baseline_epss"] == 0.0

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_pct_change_calculated(self, mock_snapshot, mock_for_cves):
        """Percentage change should be (delta / baseline) * 100."""
        mock_snapshot.return_value = _make_snapshot_data(
            {"CVE-2024-0001": 0.60},
            "2026-09-07",
        )
        mock_for_cves.return_value = {"CVE-2024-0001": 0.20}

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        assert result["movers"][0]["pct_change"] == pytest.approx(200.0, abs=0.1)

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_empty_snapshot(self, mock_snapshot, mock_for_cves):
        """Empty recent snapshot returns empty movers."""
        mock_snapshot.return_value = []

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        assert result["movers"] == []
        assert result["total_compared"] == 0

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_decreasing_scores_excluded(self, mock_snapshot, mock_for_cves):
        """CVEs whose score decreased should not appear in movers."""
        mock_snapshot.return_value = _make_snapshot_data(
            {"CVE-2024-0001": 0.30},
            "2026-09-07",
        )
        mock_for_cves.return_value = {"CVE-2024-0001": 0.90}

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        assert result["movers"] == []

    def test_invalid_date_order_raises(self):
        """recent_date must be after baseline_date."""
        with pytest.raises(ValueError, match="must be after"):
            compute_movers(
                recent_date=date(2026, 9, 1),
                baseline_date=date(2026, 9, 7),
            )

    def test_same_date_raises(self):
        """Same dates should raise ValueError."""
        with pytest.raises(ValueError, match="must be after"):
            compute_movers(
                recent_date=date(2026, 9, 7),
                baseline_date=date(2026, 9, 7),
            )

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_dates_in_output(self, mock_snapshot, mock_for_cves):
        """Output includes recent_date and baseline_date as ISO strings."""
        mock_snapshot.return_value = []

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        assert result["recent_date"] == "2026-09-07"
        assert result["baseline_date"] == "2026-08-31"

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_total_with_increase_count(self, mock_snapshot, mock_for_cves):
        """total_with_increase counts all CVEs with positive delta."""
        cve_scores = {f"CVE-2024-{i:04d}": 0.50 + i * 0.01 for i in range(5)}
        mock_snapshot.return_value = _make_snapshot_data(cve_scores, "2026-09-07")
        mock_for_cves.return_value = {cve: 0.10 for cve in cve_scores}

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
            top=2,
        )

        assert result["total_with_increase"] == 5
        assert len(result["movers"]) == 2  # but only top 2 returned

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_percentile_included(self, mock_snapshot, mock_for_cves):
        """Movers include the percentile from the recent snapshot."""
        mock_snapshot.return_value = [
            {
                "cve": "CVE-2024-0001",
                "epss": "0.80",
                "percentile": "0.95123",
                "date": "2026-09-07",
            }
        ]
        mock_for_cves.return_value = {"CVE-2024-0001": 0.20}

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        assert result["movers"][0]["percentile"] == pytest.approx(0.95123, abs=0.00001)

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_scores_rounded(self, mock_snapshot, mock_for_cves):
        """Scores in output are rounded to 6 decimal places."""
        mock_snapshot.return_value = _make_snapshot_data(
            {"CVE-2024-0001": 0.123456789},
            "2026-09-07",
        )
        mock_for_cves.return_value = {"CVE-2024-0001": 0.0}

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        m = result["movers"][0]
        assert m["recent_epss"] == round(0.123456789, 6)

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_sort_by_delta_then_recent(self, mock_snapshot, mock_for_cves):
        """On equal delta, sort by recent_epss descending."""
        mock_snapshot.return_value = _make_snapshot_data(
            {"CVE-2024-0001": 0.50, "CVE-2024-0002": 0.60},
            "2026-09-07",
        )
        # Both have same delta of 0.40
        mock_for_cves.return_value = {
            "CVE-2024-0001": 0.10,
            "CVE-2024-0002": 0.20,
        }

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        movers = result["movers"]
        assert movers[0]["cve"] == "CVE-2024-0002"  # higher recent_epss


# ---------------------------------------------------------------------------
# Retry logic tests
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for _epss_get_with_retry."""

    @patch("manus_agent.tools.get_epss_movers._RETRY_BASE_DELAY", 0)
    @patch("manus_agent.tools.get_epss_movers._MAX_RETRIES", 3)
    @patch("manus_agent.tools.get_epss_movers.requests.get")
    def test_retries_on_429(self, mock_get):
        """Should retry on 429 rate-limit."""
        from manus_agent.tools.get_epss_movers import _epss_get_with_retry

        resp_429 = MagicMock(spec=requests.Response)
        resp_429.status_code = 429
        resp_429.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=resp_429
        )

        resp_ok = MagicMock(spec=requests.Response)
        resp_ok.status_code = 200
        resp_ok.raise_for_status = MagicMock()

        mock_get.side_effect = [resp_429, resp_ok]

        result = _epss_get_with_retry({"cve": "CVE-2024-0001"})
        assert result.status_code == 200
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.get_epss_movers._RETRY_BASE_DELAY", 0)
    @patch("manus_agent.tools.get_epss_movers._MAX_RETRIES", 2)
    @patch("manus_agent.tools.get_epss_movers.requests.get")
    def test_retries_on_500(self, mock_get):
        """Should retry on 500 server error."""
        from manus_agent.tools.get_epss_movers import _epss_get_with_retry

        resp_500 = MagicMock(spec=requests.Response)
        resp_500.status_code = 500
        resp_500.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=resp_500
        )

        mock_get.side_effect = [resp_500, resp_500]

        with pytest.raises(requests.exceptions.HTTPError):
            _epss_get_with_retry({"cve": "CVE-2024-0001"})
        assert mock_get.call_count == 2

    @patch("manus_agent.tools.get_epss_movers._RETRY_BASE_DELAY", 0)
    @patch("manus_agent.tools.get_epss_movers._MAX_RETRIES", 3)
    @patch("manus_agent.tools.get_epss_movers.requests.get")
    def test_no_retry_on_403(self, mock_get):
        """Should NOT retry on 403 forbidden."""
        from manus_agent.tools.get_epss_movers import _epss_get_with_retry

        resp_403 = MagicMock(spec=requests.Response)
        resp_403.status_code = 403
        resp_403.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=resp_403
        )

        mock_get.return_value = resp_403

        with pytest.raises(requests.exceptions.HTTPError):
            _epss_get_with_retry({"cve": "CVE-2024-0001"})
        assert mock_get.call_count == 1

    @patch("manus_agent.tools.get_epss_movers._RETRY_BASE_DELAY", 0)
    @patch("manus_agent.tools.get_epss_movers._MAX_RETRIES", 3)
    @patch("manus_agent.tools.get_epss_movers.requests.get")
    def test_retries_on_connection_error(self, mock_get):
        """Should retry on connection errors."""
        from manus_agent.tools.get_epss_movers import _epss_get_with_retry

        resp_ok = MagicMock(spec=requests.Response)
        resp_ok.status_code = 200
        resp_ok.raise_for_status = MagicMock()

        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Network unreachable"),
            resp_ok,
        ]

        result = _epss_get_with_retry({"cve": "CVE-2024-0001"})
        assert result.status_code == 200

    @patch("manus_agent.tools.get_epss_movers._RETRY_BASE_DELAY", 0)
    @patch("manus_agent.tools.get_epss_movers._MAX_RETRIES", 2)
    @patch("manus_agent.tools.get_epss_movers.requests.get")
    def test_exhausted_retries_raises(self, mock_get):
        """After all retries exhausted, should raise."""
        from manus_agent.tools.get_epss_movers import _epss_get_with_retry

        mock_get.side_effect = requests.exceptions.ConnectionError("down")

        with pytest.raises(requests.exceptions.ConnectionError):
            _epss_get_with_retry({"cve": "CVE-2024-0001"})
        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# get_epss_movers (Strands tool entry point) tests
# ---------------------------------------------------------------------------


class TestGetEpssMovers:
    """Tests for the Strands tool entry point."""

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_success_with_movers(self, mock_compute):
        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [
                {
                    "cve": "CVE-2024-0001",
                    "recent_epss": 0.90,
                    "baseline_epss": 0.50,
                    "delta": 0.40,
                    "pct_change": 80.0,
                    "percentile": 0.99,
                    "is_new": False,
                }
            ],
            "total_compared": 100,
            "total_with_increase": 25,
            "error": None,
        }

        result = get_epss_movers(_make_tool_use(), now=date(2026, 9, 8))

        assert result["status"] == "success"
        assert result["toolUseId"] == "test-id-001"
        # Should have text and json content
        assert len(result["content"]) == 2

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_success_empty_movers(self, mock_compute):
        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [],
            "total_compared": 100,
            "total_with_increase": 0,
            "error": None,
        }

        result = get_epss_movers(_make_tool_use(), now=date(2026, 9, 8))

        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "No significant" in text

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_summary_includes_cve_ids(self, mock_compute):
        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [
                {
                    "cve": "CVE-2024-9999",
                    "recent_epss": 0.80,
                    "baseline_epss": 0.30,
                    "delta": 0.50,
                    "pct_change": 166.67,
                    "percentile": 0.98,
                    "is_new": False,
                }
            ],
            "total_compared": 100,
            "total_with_increase": 1,
            "error": None,
        }

        result = get_epss_movers(_make_tool_use(), now=date(2026, 9, 8))
        text = result["content"][0]["text"]
        assert "CVE-2024-9999" in text

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_new_cve_marked_in_summary(self, mock_compute):
        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [
                {
                    "cve": "CVE-2024-0001",
                    "recent_epss": 0.80,
                    "baseline_epss": 0.0,
                    "delta": 0.80,
                    "pct_change": None,
                    "percentile": 0.98,
                    "is_new": True,
                }
            ],
            "total_compared": 100,
            "total_with_increase": 1,
            "error": None,
        }

        result = get_epss_movers(_make_tool_use(), now=date(2026, 9, 8))
        text = result["content"][0]["text"]
        assert "NEW" in text

    def test_invalid_days_too_high(self):
        result = get_epss_movers(_make_tool_use(days=500), now=date(2026, 9, 8))
        assert result["status"] == "error"
        assert "days" in result["content"][0]["text"].lower()

    def test_invalid_days_zero(self):
        result = get_epss_movers(_make_tool_use(days=0), now=date(2026, 9, 8))
        assert result["status"] == "error"

    def test_invalid_days_negative(self):
        result = get_epss_movers(_make_tool_use(days=-1), now=date(2026, 9, 8))
        assert result["status"] == "error"

    def test_invalid_top_too_high(self):
        result = get_epss_movers(_make_tool_use(top=200), now=date(2026, 9, 8))
        assert result["status"] == "error"
        assert "top" in result["content"][0]["text"].lower()

    def test_invalid_top_zero(self):
        result = get_epss_movers(_make_tool_use(top=0), now=date(2026, 9, 8))
        assert result["status"] == "error"

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_default_inputs(self, mock_compute):
        """When no inputs are provided, defaults are used."""
        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [],
            "total_compared": 0,
            "total_with_increase": 0,
            "error": None,
        }

        result = get_epss_movers(_make_tool_use_empty(), now=date(2026, 9, 8))
        assert result["status"] == "success"

        # compute_movers should be called with default 7-day window
        call_kwargs = mock_compute.call_args
        assert call_kwargs[1]["top"] == 20

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_api_failure_returns_error(self, mock_compute):
        mock_compute.side_effect = requests.exceptions.ConnectionError(
            "Connection refused"
        )

        result = get_epss_movers(_make_tool_use(), now=date(2026, 9, 8))
        assert result["status"] == "error"
        assert "EPSS API" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_value_error_returns_error(self, mock_compute):
        mock_compute.side_effect = ValueError("bad dates")

        result = get_epss_movers(_make_tool_use(), now=date(2026, 9, 8))
        assert result["status"] == "error"
        assert "bad dates" in result["content"][0]["text"]

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_json_content_included(self, mock_compute):
        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [],
            "total_compared": 50,
            "total_with_increase": 0,
            "error": None,
        }

        result = get_epss_movers(_make_tool_use(), now=date(2026, 9, 8))
        json_content = result["content"][1]["json"]
        assert json_content["total_compared"] == 50

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_tool_use_id_preserved(self, mock_compute):
        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [],
            "total_compared": 0,
            "total_with_increase": 0,
            "error": None,
        }

        result = get_epss_movers(
            _make_tool_use(tool_use_id="custom-id-123"),
            now=date(2026, 9, 8),
        )
        assert result["toolUseId"] == "custom-id-123"

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_now_parameter_used(self, mock_compute):
        """The 'now' parameter should control date calculations."""
        mock_compute.return_value = {
            "recent_date": "2026-06-14",
            "baseline_date": "2026-06-07",
            "movers": [],
            "total_compared": 0,
            "total_with_increase": 0,
            "error": None,
        }

        get_epss_movers(_make_tool_use(days=7), now=date(2026, 6, 15))

        call_kwargs = mock_compute.call_args
        # recent = 2026-06-14 (today - 1), baseline = 2026-06-07 (recent - 7)
        assert call_kwargs[1]["recent_date"] == date(2026, 6, 14)
        assert call_kwargs[1]["baseline_date"] == date(2026, 6, 7)


# ---------------------------------------------------------------------------
# CLI subcommand tests
# ---------------------------------------------------------------------------


class TestCliEpssMovers:
    """Tests for the CLI dispatch and output."""

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_text_output(self, mock_compute, capsys):
        from manus_agent.cli import _run_epss_movers

        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [
                {
                    "cve": "CVE-2024-0001",
                    "recent_epss": 0.90,
                    "baseline_epss": 0.50,
                    "delta": 0.40,
                    "pct_change": 80.0,
                    "percentile": 0.99,
                    "is_new": False,
                }
            ],
            "total_compared": 100,
            "total_with_increase": 1,
            "error": None,
        }

        exit_code = _run_epss_movers([])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "CVE-2024-0001" in captured.out
        assert "0.9000" in captured.out

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_json_output(self, mock_compute, capsys):
        from manus_agent.cli import _run_epss_movers

        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [],
            "total_compared": 100,
            "total_with_increase": 0,
            "error": None,
        }

        exit_code = _run_epss_movers(["--output", "json"])
        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "movers" in data
        assert data["total_compared"] == 100

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_custom_days_and_top(self, mock_compute, capsys):
        from manus_agent.cli import _run_epss_movers

        mock_compute.return_value = {
            "recent_date": "2026-08-08",
            "baseline_date": "2026-07-09",
            "movers": [],
            "total_compared": 50,
            "total_with_increase": 0,
            "error": None,
        }

        exit_code = _run_epss_movers(["--days", "30", "--top", "50"])
        assert exit_code == 0

        call_kwargs = mock_compute.call_args
        assert call_kwargs[1]["top"] == 50

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_api_error_returns_1(self, mock_compute, capsys):
        from manus_agent.cli import _run_epss_movers

        mock_compute.side_effect = requests.exceptions.ConnectionError("down")

        exit_code = _run_epss_movers([])
        assert exit_code == 1

        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_no_movers_text(self, mock_compute, capsys):
        from manus_agent.cli import _run_epss_movers

        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [],
            "total_compared": 100,
            "total_with_increase": 0,
            "error": None,
        }

        exit_code = _run_epss_movers([])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "No significant" in captured.out

    def test_subcommand_in_set(self):
        """epss-movers should be registered in _SUBCOMMANDS."""
        from manus_agent.cli import _SUBCOMMANDS

        assert "epss-movers" in _SUBCOMMANDS

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_new_cve_in_text_output(self, mock_compute, capsys):
        from manus_agent.cli import _run_epss_movers

        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [
                {
                    "cve": "CVE-2026-1234",
                    "recent_epss": 0.75,
                    "baseline_epss": 0.0,
                    "delta": 0.75,
                    "pct_change": None,
                    "percentile": 0.97,
                    "is_new": True,
                }
            ],
            "total_compared": 100,
            "total_with_increase": 1,
            "error": None,
        }

        exit_code = _run_epss_movers([])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "CVE-2026-1234" in captured.out
        assert "NEW" in captured.out


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_snapshot_with_invalid_scores(self, mock_snapshot, mock_for_cves):
        """Invalid EPSS values in snapshot should be handled gracefully."""
        mock_snapshot.return_value = [
            {
                "cve": "CVE-2024-0001",
                "epss": "not-a-number",
                "percentile": "also-bad",
                "date": "2026-09-07",
            }
        ]
        mock_for_cves.return_value = {}

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        # Should handle gracefully — score defaults to 0.0
        assert result["total_compared"] == 1

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_all_cves_decrease(self, mock_snapshot, mock_for_cves):
        """When all CVEs decrease, movers should be empty."""
        mock_snapshot.return_value = _make_snapshot_data(
            {"CVE-2024-0001": 0.10, "CVE-2024-0002": 0.20},
            "2026-09-07",
        )
        mock_for_cves.return_value = {
            "CVE-2024-0001": 0.90,
            "CVE-2024-0002": 0.80,
        }

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        assert result["movers"] == []
        assert result["total_with_increase"] == 0

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_single_cve_in_snapshot(self, mock_snapshot, mock_for_cves):
        """Should work with just one CVE in the snapshot."""
        mock_snapshot.return_value = _make_snapshot_data(
            {"CVE-2024-0001": 0.90},
            "2026-09-07",
        )
        mock_for_cves.return_value = {"CVE-2024-0001": 0.10}

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        assert len(result["movers"]) == 1
        assert result["total_compared"] == 1

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_mixed_increase_decrease(self, mock_snapshot, mock_for_cves):
        """Only CVEs with increases should appear in movers."""
        mock_snapshot.return_value = _make_snapshot_data(
            {
                "CVE-2024-0001": 0.80,  # up from 0.20
                "CVE-2024-0002": 0.10,  # down from 0.90
                "CVE-2024-0003": 0.50,  # up from 0.30
            },
            "2026-09-07",
        )
        mock_for_cves.return_value = {
            "CVE-2024-0001": 0.20,
            "CVE-2024-0002": 0.90,
            "CVE-2024-0003": 0.30,
        }

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        assert len(result["movers"]) == 2
        cve_ids = [m["cve"] for m in result["movers"]]
        assert "CVE-2024-0001" in cve_ids
        assert "CVE-2024-0003" in cve_ids
        assert "CVE-2024-0002" not in cve_ids

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_duplicate_cves_in_snapshot(self, mock_snapshot, mock_for_cves):
        """Duplicate CVEs in snapshot should be deduplicated (last wins)."""
        mock_snapshot.return_value = [
            {"cve": "CVE-2024-0001", "epss": "0.50", "percentile": "0.80", "date": "2026-09-07"},
            {"cve": "CVE-2024-0001", "epss": "0.60", "percentile": "0.85", "date": "2026-09-07"},
        ]
        mock_for_cves.return_value = {"CVE-2024-0001": 0.10}

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        # Should only have one mover for CVE-2024-0001
        assert result["total_compared"] == 1

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_empty_cve_id_skipped(self, mock_snapshot, mock_for_cves):
        """Entries with empty CVE ID should be skipped."""
        mock_snapshot.return_value = [
            {"cve": "", "epss": "0.50", "percentile": "0.80", "date": "2026-09-07"},
            {"cve": "CVE-2024-0001", "epss": "0.80", "percentile": "0.95", "date": "2026-09-07"},
        ]
        mock_for_cves.return_value = {"CVE-2024-0001": 0.30}

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 8, 31),
        )

        assert result["total_compared"] == 1
        assert result["movers"][0]["cve"] == "CVE-2024-0001"

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_multiple_movers_in_summary(self, mock_compute):
        """Summary should list all movers with ranking numbers."""
        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [
                {
                    "cve": f"CVE-2024-000{i}",
                    "recent_epss": 0.90 - i * 0.05,
                    "baseline_epss": 0.10,
                    "delta": 0.80 - i * 0.05,
                    "pct_change": (0.80 - i * 0.05) / 0.10 * 100,
                    "percentile": 0.99 - i * 0.01,
                    "is_new": False,
                }
                for i in range(3)
            ],
            "total_compared": 100,
            "total_with_increase": 3,
            "error": None,
        }

        result = get_epss_movers(_make_tool_use(), now=date(2026, 9, 8))
        text = result["content"][0]["text"]

        assert "CVE-2024-0000" in text
        assert "CVE-2024-0001" in text
        assert "CVE-2024-0002" in text

    @patch("manus_agent.tools.get_epss_movers.compute_movers")
    def test_json_serializable(self, mock_compute):
        """Output JSON content should be serializable."""
        mock_compute.return_value = {
            "recent_date": "2026-09-07",
            "baseline_date": "2026-08-31",
            "movers": [
                {
                    "cve": "CVE-2024-0001",
                    "recent_epss": 0.80,
                    "baseline_epss": 0.0,
                    "delta": 0.80,
                    "pct_change": None,
                    "percentile": 0.98,
                    "is_new": True,
                }
            ],
            "total_compared": 50,
            "total_with_increase": 1,
            "error": None,
        }

        result = get_epss_movers(_make_tool_use(), now=date(2026, 9, 8))
        json_data = result["content"][1]["json"]

        # Should not raise
        serialized = json.dumps(json_data)
        assert isinstance(serialized, str)

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_one_day_window(self, mock_snapshot, mock_for_cves):
        """Minimum 1-day window should work."""
        mock_snapshot.return_value = _make_snapshot_data(
            {"CVE-2024-0001": 0.80},
            "2026-09-07",
        )
        mock_for_cves.return_value = {"CVE-2024-0001": 0.70}

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2026, 9, 6),
        )

        assert len(result["movers"]) == 1

    @patch("manus_agent.tools.get_epss_movers._fetch_epss_for_cves")
    @patch("manus_agent.tools.get_epss_movers._fetch_epss_snapshot")
    def test_365_day_window(self, mock_snapshot, mock_for_cves):
        """Maximum 365-day window should work."""
        mock_snapshot.return_value = _make_snapshot_data(
            {"CVE-2024-0001": 0.80},
            "2026-09-07",
        )
        mock_for_cves.return_value = {"CVE-2024-0001": 0.01}

        result = compute_movers(
            recent_date=date(2026, 9, 7),
            baseline_date=date(2025, 9, 7),
        )

        assert len(result["movers"]) == 1
        assert result["movers"][0]["delta"] == pytest.approx(0.79, abs=0.01)
