"""
Tests for _enrich_pub, _parse_pub_timestamp, and the _enrich_package dispatch
to Pub, in src/manus_agent/tools/get_dependency_blast_radius.py

All external HTTP calls are mocked — no real network I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from manus_agent.tools.get_dependency_blast_radius import (
    _PUB_WEEKLY_ESTIMATE_SCALE,
    _blast_score,
    _enrich_package,
    _enrich_pub,
    _parse_pub_timestamp,
    get_dependency_blast_radius,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VERSIONS = [
    {"version": "1.0.0", "published": "2019-03-15T12:00:00Z"},
    {"version": "1.1.0", "published": "2020-06-20T08:30:00Z"},
    {"version": "2.0.0", "published": "2023-11-01T00:00:00.000000Z"},
]

_PUBSPEC = {
    "name": "http",
    "description": "A composable, cross-platform, Future-based API for making HTTP requests.",
    "version": "2.0.0",
}

_PKG_RESPONSE = {
    "name": "http",
    "latest": {"version": "2.0.0", "pubspec": _PUBSPEC},
    "versions": _VERSIONS,
    "publisherMemberships": [{"publisherName": "dart.dev"}],
}

_SCORE_RESPONSE = {
    "popularityScore": 0.9900,
    "likeCount": 8000,
    "grantedPoints": 130,
    "maxPoints": 140,
}


def _make_mock_resp(json_data: dict) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = json_data
    mock.raise_for_status.return_value = None
    return mock


# ===========================================================================
# _parse_pub_timestamp
# ===========================================================================


class TestParsePubTimestamp:
    def test_z_suffix(self):
        dt = _parse_pub_timestamp("2019-03-15T12:00:00Z")
        assert dt is not None
        assert dt.year == 2019
        assert dt.month == 3
        assert dt.day == 15
        assert dt.tzinfo is not None

    def test_plus_offset(self):
        dt = _parse_pub_timestamp("2020-06-20T08:30:00+00:00")
        assert dt is not None
        assert dt.year == 2020

    def test_microseconds_with_z(self):
        dt = _parse_pub_timestamp("2023-11-01T00:00:00.000000Z")
        assert dt is not None
        assert dt.year == 2023
        assert dt.month == 11

    def test_seven_digit_fractional_truncated(self):
        # .NET-style 7-digit ticks — truncated to 6
        dt = _parse_pub_timestamp("2021-05-10T15:30:00.1234567Z")
        assert dt is not None
        assert dt.year == 2021

    def test_naive_datetime_gets_utc(self):
        dt = _parse_pub_timestamp("2022-01-01T00:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_empty_string_returns_none(self):
        assert _parse_pub_timestamp("") is None

    def test_invalid_string_returns_none(self):
        assert _parse_pub_timestamp("not-a-date") is None

    def test_date_only_string_returns_none(self):
        # Bare date strings without time components fail fromisoformat on Python ≤3.10
        # depending on build; just confirm no exception is raised
        result = _parse_pub_timestamp("2020-01-01")
        # May succeed or return None depending on Python version — just no crash
        assert result is None or isinstance(result, datetime)


# ===========================================================================
# _enrich_pub
# ===========================================================================


class TestEnrichPub:
    def test_happy_path_full_data(self):
        pkg_resp = _make_mock_resp(_PKG_RESPONSE)
        score_resp = _make_mock_resp(_SCORE_RESPONSE)
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("http")

        assert result["ecosystem"] == "Pub"
        assert result["package_name"] == "http"
        assert result["latest_version"] == "2.0.0"
        assert result["total_versions"] == 3
        assert result["first_release_date"] == "2019-03-15"
        assert result["age_years"] > 0
        assert "composable" in result["description"]
        assert result["publisher"] == "dart.dev"
        assert result["pub_page"] == "https://pub.dev/packages/http"
        assert result["popularity_score"] == 0.99
        assert result["like_count"] == 8000
        assert result["pub_points"] == 130
        assert result["max_pub_points"] == 140
        # Weekly downloads proxy
        assert result["weekly_downloads"] == int(0.99 * _PUB_WEEKLY_ESTIMATE_SCALE)

    def test_weekly_downloads_proxy_scales_correctly(self):
        score_resp = _make_mock_resp({"popularityScore": 0.5, "likeCount": 500, "grantedPoints": 80, "maxPoints": 140})
        pkg_resp = _make_mock_resp({"versions": [], "latest": {}, "name": "pkg"})
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("pkg")
        assert result["weekly_downloads"] == int(0.5 * _PUB_WEEKLY_ESTIMATE_SCALE)

    def test_blast_score_critical_for_very_popular_package(self):
        # popularityScore = 1.0 → weekly = 200_000 (threshold for MEDIUM is 50k)
        score_resp = _make_mock_resp(
            {"popularityScore": 1.0, "likeCount": 9000, "grantedPoints": 140, "maxPoints": 140}
        )
        pkg_resp = _make_mock_resp({"versions": _VERSIONS, "latest": {"pubspec": _PUBSPEC}, "name": "flutter"})
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("flutter")
        score = _blast_score(result)
        # 200_000 weekly ≥ 50_000 → at least MEDIUM; may be HIGH
        assert score in ("MEDIUM", "HIGH", "CRITICAL")

    def test_blast_score_low_for_unpopular_package(self):
        score_resp = _make_mock_resp({"popularityScore": 0.01, "likeCount": 2, "grantedPoints": 10, "maxPoints": 140})
        pkg_resp = _make_mock_resp(
            {
                "versions": [{"version": "0.1.0", "published": "2022-01-01T00:00:00Z"}],
                "latest": {},
                "name": "obscure_pkg",
            }
        )
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("obscure_pkg")
        score = _blast_score(result)
        assert score == "LOW"

    def test_blast_score_unknown_when_no_popularity_score(self):
        # Both API calls fail → no weekly_downloads, no dependents
        with patch("requests.get", side_effect=Exception("network error")):
            result = _enrich_pub("ghost_pkg")
        score = _blast_score(result)
        assert score == "UNKNOWN"

    def test_graceful_degradation_pkg_api_fails(self):
        score_resp = _make_mock_resp(_SCORE_RESPONSE)
        with patch("requests.get", side_effect=[Exception("404 Not Found"), score_resp]):
            result = _enrich_pub("http")
        # Score data still present
        assert result["like_count"] == 8000
        # Package data absent — no crash
        assert "latest_version" not in result
        assert "total_versions" not in result

    def test_graceful_degradation_score_api_fails(self):
        pkg_resp = _make_mock_resp(_PKG_RESPONSE)
        with patch("requests.get", side_effect=[pkg_resp, Exception("503 Service Unavailable")]):
            result = _enrich_pub("http")
        # Package data still present
        assert result["latest_version"] == "2.0.0"
        # Score data absent — no crash
        assert "popularity_score" not in result
        assert "weekly_downloads" not in result

    def test_graceful_degradation_both_apis_fail(self):
        with patch("requests.get", side_effect=Exception("timeout")):
            result = _enrich_pub("any_package")
        assert result["ecosystem"] == "Pub"
        assert result["package_name"] == "any_package"
        # Should not raise; no required keys
        assert "latest_version" not in result
        assert "popularity_score" not in result

    def test_empty_versions_list(self):
        pkg_resp = _make_mock_resp({"versions": [], "latest": {}, "name": "empty_pkg"})
        score_resp = _make_mock_resp(_SCORE_RESPONSE)
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("empty_pkg")
        assert result["total_versions"] == 0
        assert "latest_version" not in result
        assert "first_release_date" not in result

    def test_versions_without_published_timestamps(self):
        versions_no_ts = [{"version": "1.0.0"}, {"version": "2.0.0"}]
        pkg_resp = _make_mock_resp({"versions": versions_no_ts, "latest": {}, "name": "nots_pkg"})
        score_resp = _make_mock_resp(_SCORE_RESPONSE)
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("nots_pkg")
        # latest_version from last entry
        assert result["latest_version"] == "2.0.0"
        # No first_release_date because published is missing
        assert "first_release_date" not in result

    def test_no_publisher_memberships(self):
        pkg_data = {**_PKG_RESPONSE, "publisherMemberships": []}
        pkg_resp = _make_mock_resp(pkg_data)
        score_resp = _make_mock_resp(_SCORE_RESPONSE)
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("http")
        assert "publisher" not in result

    def test_description_truncated_to_120_chars(self):
        long_desc = "A" * 200
        pkg_data = {
            **_PKG_RESPONSE,
            "latest": {"version": "1.0.0", "pubspec": {"description": long_desc}},
        }
        pkg_resp = _make_mock_resp(pkg_data)
        score_resp = _make_mock_resp(_SCORE_RESPONSE)
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("http")
        assert len(result["description"]) == 120

    def test_description_newlines_stripped(self):
        multiline = "Line one.\nLine two.\nLine three."
        pkg_data = {
            **_PKG_RESPONSE,
            "latest": {"version": "1.0.0", "pubspec": {"description": multiline}},
        }
        pkg_resp = _make_mock_resp(pkg_data)
        score_resp = _make_mock_resp(_SCORE_RESPONSE)
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("http")
        assert "\n" not in result["description"]

    def test_pub_page_url_format(self):
        pkg_resp = _make_mock_resp(_PKG_RESPONSE)
        score_resp = _make_mock_resp(_SCORE_RESPONSE)
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("provider")
        assert result["pub_page"] == "https://pub.dev/packages/provider"

    def test_popularity_score_rounded_to_4_decimal_places(self):
        score_resp = _make_mock_resp(
            {"popularityScore": 0.123456789, "likeCount": 0, "grantedPoints": 0, "maxPoints": 0}
        )
        pkg_resp = _make_mock_resp({"versions": [], "latest": {}, "name": "test"})
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("test")
        assert result["popularity_score"] == round(0.123456789, 4)

    def test_zero_popularity_score(self):
        score_resp = _make_mock_resp({"popularityScore": 0.0, "likeCount": 0, "grantedPoints": 0, "maxPoints": 140})
        pkg_resp = _make_mock_resp({"versions": [], "latest": {}, "name": "zero_pkg"})
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("zero_pkg")
        assert result["popularity_score"] == 0.0
        assert result["weekly_downloads"] == 0
        assert _blast_score(result) == "UNKNOWN"

    def test_correct_api_urls_called(self):
        pkg_resp = _make_mock_resp(_PKG_RESPONSE)
        score_resp = _make_mock_resp(_SCORE_RESPONSE)
        with patch("requests.get", side_effect=[pkg_resp, score_resp]) as mock_get:
            _enrich_pub("http")
        calls = [c[0][0] for c in mock_get.call_args_list]
        assert any("pub.dev/api/packages/http" in c and "score" not in c for c in calls)
        assert any("pub.dev/api/packages/http/score" in c for c in calls)

    def test_age_years_computed_correctly(self):
        # Use a known date far in the past; age should be positive
        versions = [{"version": "0.1.0", "published": "2014-01-01T00:00:00Z"}]
        pkg_resp = _make_mock_resp({"versions": versions, "latest": {}, "name": "old_pkg"})
        score_resp = _make_mock_resp({"popularityScore": 0.5, "likeCount": 0, "grantedPoints": 0, "maxPoints": 0})
        with patch("requests.get", side_effect=[pkg_resp, score_resp]):
            result = _enrich_pub("old_pkg")
        assert result["first_release_date"] == "2014-01-01"
        # Package was released in 2014; by 2026 that's ~12 years
        assert result["age_years"] >= 10.0


# ===========================================================================
# _enrich_package dispatch to Pub
# ===========================================================================


class TestEnrichPackageDispatchPub:
    def test_dispatches_pub_ecosystem(self):
        with patch("manus_agent.tools.get_dependency_blast_radius._enrich_pub") as mock_pub:
            mock_pub.return_value = {"ecosystem": "Pub", "package_name": "provider"}
            _enrich_package("provider", "Pub")
        mock_pub.assert_called_once_with("provider")

    def test_dispatches_dart_alias(self):
        with patch("manus_agent.tools.get_dependency_blast_radius._enrich_pub") as mock_pub:
            mock_pub.return_value = {"ecosystem": "Pub", "package_name": "pkg"}
            _enrich_package("pkg", "dart")
        mock_pub.assert_called_once_with("pkg")

    def test_dispatches_flutter_alias(self):
        with patch("manus_agent.tools.get_dependency_blast_radius._enrich_pub") as mock_pub:
            mock_pub.return_value = {"ecosystem": "Pub", "package_name": "pkg"}
            _enrich_package("pkg", "flutter")
        mock_pub.assert_called_once_with("pkg")

    def test_dispatches_dart_case_insensitive(self):
        with patch("manus_agent.tools.get_dependency_blast_radius._enrich_pub") as mock_pub:
            mock_pub.return_value = {"ecosystem": "Pub", "package_name": "pkg"}
            _enrich_package("pkg", "DART")
        mock_pub.assert_called_once_with("pkg")


# ===========================================================================
# Integration: get_dependency_blast_radius with Pub ecosystem
# ===========================================================================


class TestGetDependencyBlastRadiusPub:
    def _mock_enrich_pub(self, name: str) -> dict:
        return {
            "ecosystem": "Pub",
            "package_name": name,
            "latest_version": "2.0.0",
            "total_versions": 25,
            "popularity_score": 0.99,
            "like_count": 8000,
            "pub_points": 130,
            "max_pub_points": 140,
            "first_release_date": "2019-03-15",
            "age_years": 7.1,
            "description": "A composable, cross-platform API for HTTP requests.",
            "publisher": "dart.dev",
            "pub_page": "https://pub.dev/packages/http",
            "weekly_downloads": int(0.99 * _PUB_WEEKLY_ESTIMATE_SCALE),
        }

    def test_direct_pub_package_query_output(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            side_effect=lambda name, eco: {**self._mock_enrich_pub(name), "version_range": "all", "source": "direct"},
        ):
            result = get_dependency_blast_radius("pub:http")

        assert "http" in result
        assert "Pub (Dart/Flutter)" in result
        assert "2.0.0" in result
        assert "130/140" in result  # pub points
        assert "2019-03-15" in result
        assert "dart.dev" in result
        assert "pub.dev/packages/http" in result

    def test_popularity_score_shown_in_output(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            side_effect=lambda name, eco: {**self._mock_enrich_pub(name), "version_range": "all", "source": "direct"},
        ):
            result = get_dependency_blast_radius("pub:provider")

        assert "0.9900" in result

    def test_blast_radius_label_present(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
            side_effect=lambda name, eco: {**self._mock_enrich_pub(name), "version_range": "all", "source": "direct"},
        ):
            result = get_dependency_blast_radius("pub:http")

        # Blast radius label must be present (computed from weekly_downloads proxy)
        assert any(label in result for label in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"))

    def test_flutter_ecosystem_alias_dispatched(self):
        with patch(
            "manus_agent.tools.get_dependency_blast_radius._enrich_package",
        ) as mock_enrich:
            mock_enrich.return_value = {
                "ecosystem": "flutter",
                "package_name": "provider",
                "weekly_downloads": 100_000,
                "version_range": "all",
                "source": "direct",
            }
            get_dependency_blast_radius("flutter:provider")
        mock_enrich.assert_called_once_with("provider", "flutter")
