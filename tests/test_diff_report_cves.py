"""Tests for diff_report_cves tool and diff-report CLI subcommand.

All external HTTP calls are mocked; no real network I/O.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures – minimal mock payloads
# ──────────────────────────────────────────────────────────────────────────────

_NVD_PAYLOAD_A = {
    "id": "CVE-2021-44228",
    "published": "2021-12-10T10:15:00.000",
    "descriptions": [{"lang": "en", "value": "Apache Log4j2 RCE via JNDI lookup."}],
    "metrics": {
        "cvssMetricV31": [
            {
                "cvssData": {
                    "version": "3.1",
                    "baseScore": 10.0,
                    "baseSeverity": "CRITICAL",
                    "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                    "attackVector": "NETWORK",
                    "privilegesRequired": "NONE",
                    "userInteraction": "NONE",
                    "scope": "CHANGED",
                    "confidentialityImpact": "HIGH",
                    "integrityImpact": "HIGH",
                    "availabilityImpact": "HIGH",
                }
            }
        ]
    },
    "weaknesses": [{"description": [{"value": "CWE-917"}]}],
    "configurations": [{"nodes": [{"cpeMatch": [{"criteria": "cpe:2.3:a:apache:log4j:2.0:*:*:*:*:*:*:*"}]}]}],
    "references": [
        {"url": "https://logging.apache.org/log4j/2.x/security.html"},
        {"url": "https://github.com/apache/logging-log4j2"},
    ],
}

_NVD_PAYLOAD_B = {
    "id": "CVE-2021-45046",
    "published": "2021-12-14T19:15:00.000",
    "descriptions": [{"lang": "en", "value": "Apache Log4j2 incomplete fix for CVE-2021-44228."}],
    "metrics": {
        "cvssMetricV31": [
            {
                "cvssData": {
                    "version": "3.1",
                    "baseScore": 9.0,
                    "baseSeverity": "CRITICAL",
                    "vectorString": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:C/C:H/I:H/A:H",
                    "attackVector": "NETWORK",
                    "privilegesRequired": "NONE",
                    "userInteraction": "NONE",
                    "scope": "CHANGED",
                    "confidentialityImpact": "HIGH",
                    "integrityImpact": "HIGH",
                    "availabilityImpact": "HIGH",
                }
            }
        ]
    },
    "weaknesses": [{"description": [{"value": "CWE-917"}]}],
    "configurations": [{"nodes": [{"cpeMatch": [{"criteria": "cpe:2.3:a:apache:log4j:2.0:*:*:*:*:*:*:*"}]}]}],
    "references": [
        {"url": "https://logging.apache.org/log4j/2.x/security.html"},
    ],
}

_NVD_RESPONSE_A = {"vulnerabilities": [{"cve": _NVD_PAYLOAD_A}]}
_NVD_RESPONSE_B = {"vulnerabilities": [{"cve": _NVD_PAYLOAD_B}]}

_EPSS_RESPONSE_A = {"data": [{"cve": "CVE-2021-44228", "epss": "0.975", "percentile": "0.999"}]}
_EPSS_RESPONSE_B = {"data": [{"cve": "CVE-2021-45046", "epss": "0.820", "percentile": "0.997"}]}

_KEV_CATALOG = {
    "vulnerabilities": [
        {
            "cveID": "CVE-2021-44228",
            "vendorProject": "Apache",
            "product": "Log4j",
            "dateAdded": "2021-12-10",
            "dueDate": "2021-12-24",
            "requiredAction": "Apply updates per vendor instructions.",
        }
    ]
}

_KEV_EMPTY = {"vulnerabilities": []}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_data
    m.raise_for_status = MagicMock()
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests – internal helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractHelpers:
    def test_extract_cvss_v31(self):
        from manus_agent.tools.diff_report_cves import _extract_cvss

        result = _extract_cvss(_NVD_PAYLOAD_A)
        assert result["score"] == 10.0
        assert result["severity"] == "CRITICAL"
        assert result["attack_vector"] == "NETWORK"
        assert result["privileges_required"] == "NONE"
        assert result["user_interaction"] == "NONE"
        assert result["version"] == "3.1"

    def test_extract_cvss_v2_fallback(self):
        from manus_agent.tools.diff_report_cves import _extract_cvss

        nvd = {
            "metrics": {
                "cvssMetricV2": [
                    {
                        "baseSeverity": "HIGH",
                        "cvssData": {
                            "baseScore": 7.5,
                            "vectorString": "AV:N/AC:L/Au:N/C:P/I:P/A:P",
                            "accessVector": "NETWORK",
                            "confidentialityImpact": "PARTIAL",
                            "integrityImpact": "PARTIAL",
                            "availabilityImpact": "PARTIAL",
                        },
                    }
                ]
            }
        }
        result = _extract_cvss(nvd)
        assert result["version"] == "2.0"
        assert result["score"] == 7.5
        assert result["severity"] == "HIGH"

    def test_extract_cvss_empty(self):
        from manus_agent.tools.diff_report_cves import _empty_cvss, _extract_cvss

        result = _extract_cvss({})
        assert result == _empty_cvss()

    def test_extract_cwe(self):
        from manus_agent.tools.diff_report_cves import _extract_cwe

        result = _extract_cwe(_NVD_PAYLOAD_A)
        assert result == ["CWE-917"]

    def test_extract_cwe_filters_noinfo(self):
        from manus_agent.tools.diff_report_cves import _extract_cwe

        nvd = {
            "weaknesses": [
                {
                    "description": [
                        {"value": "NVD-CWE-noinfo"},
                        {"value": "NVD-CWE-Other"},
                        {"value": "CWE-79"},
                    ]
                }
            ]
        }
        result = _extract_cwe(nvd)
        assert result == ["CWE-79"]

    def test_extract_affected(self):
        from manus_agent.tools.diff_report_cves import _extract_affected

        result = _extract_affected(_NVD_PAYLOAD_A)
        assert result == "Apache / Log4J"

    def test_extract_affected_empty(self):
        from manus_agent.tools.diff_report_cves import _extract_affected

        result = _extract_affected({})
        assert result == "Unknown"

    def test_extract_published(self):
        from manus_agent.tools.diff_report_cves import _extract_published

        result = _extract_published(_NVD_PAYLOAD_A)
        assert result == "2021-12-10"

    def test_extract_published_empty(self):
        from manus_agent.tools.diff_report_cves import _extract_published

        result = _extract_published({})
        assert result == "Unknown"

    def test_extract_description_truncation(self):
        from manus_agent.tools.diff_report_cves import _extract_description

        long_val = "x" * 500
        nvd = {"descriptions": [{"lang": "en", "value": long_val}]}
        result = _extract_description(nvd, max_chars=100)
        assert len(result) == 103  # 100 chars + "..."
        assert result.endswith("...")

    def test_extract_references(self):
        from manus_agent.tools.diff_report_cves import _extract_references

        result = _extract_references(_NVD_PAYLOAD_A, limit=1)
        assert len(result) == 1
        assert "apache.org" in result[0]


class TestFetchHelpers:
    def test_fetch_nvd_success(self):
        from manus_agent.tools.diff_report_cves import _fetch_nvd

        with patch("requests.get", return_value=_mock_response(_NVD_RESPONSE_A)):
            result = _fetch_nvd("CVE-2021-44228")
        assert result["id"] == "CVE-2021-44228"
        assert "error" not in result

    def test_fetch_nvd_not_found(self):
        from manus_agent.tools.diff_report_cves import _fetch_nvd

        with patch("requests.get", return_value=_mock_response({"vulnerabilities": []})):
            result = _fetch_nvd("CVE-9999-99999")
        assert "error" in result

    def test_fetch_nvd_request_exception(self):
        import requests as _req

        from manus_agent.tools.diff_report_cves import _fetch_nvd

        with patch("requests.get", side_effect=_req.RequestException("timeout")):
            result = _fetch_nvd("CVE-2021-44228")
        assert "error" in result
        assert "timeout" in result["error"]

    def test_fetch_epss_success(self):
        from manus_agent.tools.diff_report_cves import _fetch_epss

        with patch("requests.get", return_value=_mock_response(_EPSS_RESPONSE_A)):
            result = _fetch_epss("CVE-2021-44228")
        assert result["epss"] == "0.975"
        assert "error" not in result

    def test_fetch_epss_not_found(self):
        from manus_agent.tools.diff_report_cves import _fetch_epss

        with patch("requests.get", return_value=_mock_response({"data": []})):
            result = _fetch_epss("CVE-9999-99999")
        assert "error" in result

    def test_fetch_epss_request_exception(self):
        import requests as _req

        from manus_agent.tools.diff_report_cves import _fetch_epss

        with patch("requests.get", side_effect=_req.RequestException("refused")):
            result = _fetch_epss("CVE-2021-44228")
        assert "error" in result

    def test_fetch_kev_in_catalog(self):
        from manus_agent.tools.diff_report_cves import _fetch_kev

        with patch("requests.get", return_value=_mock_response(_KEV_CATALOG)):
            result = _fetch_kev("CVE-2021-44228")
        assert result["in_kev"] is True
        assert result["date_added"] == "2021-12-10"
        assert result["required_action"] == "Apply updates per vendor instructions."

    def test_fetch_kev_not_in_catalog(self):
        from manus_agent.tools.diff_report_cves import _fetch_kev

        with patch("requests.get", return_value=_mock_response(_KEV_CATALOG)):
            result = _fetch_kev("CVE-2021-45046")
        assert result["in_kev"] is False

    def test_fetch_kev_request_exception(self):
        import requests as _req

        from manus_agent.tools.diff_report_cves import _fetch_kev

        with patch("requests.get", side_effect=_req.RequestException("timeout")):
            result = _fetch_kev("CVE-2021-44228")
        assert result["in_kev"] is False
        assert "error" in result


class TestBuildProfile:
    def test_build_profile_success(self):
        from manus_agent.tools.diff_report_cves import _build_profile

        def mock_get(url, **kwargs):
            if "nvd.nist.gov" in url:
                return _mock_response(_NVD_RESPONSE_A)
            return _mock_response(_EPSS_RESPONSE_A)

        with patch("requests.get", side_effect=mock_get):
            profile = _build_profile("CVE-2021-44228")

        assert profile["cve_id"] == "CVE-2021-44228"
        assert profile["cvss"]["score"] == 10.0
        assert profile["epss"]["score"] == pytest.approx(0.975, abs=0.001)
        assert profile["cwe"] == ["CWE-917"]
        assert profile["nvd_error"] is None

    def test_build_profile_nvd_error_degrades(self):
        import requests as _req

        from manus_agent.tools.diff_report_cves import _build_profile, _empty_cvss

        def mock_get(url, **kwargs):
            if "nvd.nist.gov" in url:
                raise _req.RequestException("timeout")
            return _mock_response(_EPSS_RESPONSE_A)

        with patch("requests.get", side_effect=mock_get):
            profile = _build_profile("CVE-2021-44228")

        assert profile["nvd_error"] is not None
        assert profile["cvss"] == _empty_cvss()
        assert profile["cwe"] == []
        # EPSS still populated
        assert profile["epss"]["score"] == pytest.approx(0.975, abs=0.001)

    def test_build_profile_epss_error_degrades(self):
        import requests as _req

        from manus_agent.tools.diff_report_cves import _build_profile

        def mock_get(url, **kwargs):
            if "first.org" in url:
                raise _req.RequestException("timeout")
            return _mock_response(_NVD_RESPONSE_A)

        with patch("requests.get", side_effect=mock_get):
            profile = _build_profile("CVE-2021-44228")

        assert profile["epss_error"] is not None
        assert profile["epss"]["score"] is None
        # NVD still populated
        assert profile["cvss"]["score"] == 10.0


class TestPriorityScore:
    def test_kev_adds_10_points(self):
        from manus_agent.tools.diff_report_cves import _priority_score

        profile = {
            "kev": {"in_kev": True},
            "cvss": {},
            "epss": {},
        }
        score, reasons = _priority_score(profile)
        assert score >= 10
        assert any("KEV" in r for r in reasons)

    def test_critical_cvss_adds_8_points(self):
        from manus_agent.tools.diff_report_cves import _priority_score

        profile = {
            "kev": {"in_kev": False},
            "cvss": {"score": 9.8, "attack_vector": None, "privileges_required": None, "user_interaction": None},
            "epss": {},
        }
        score, reasons = _priority_score(profile)
        assert score == 8
        assert any("9.8" in r for r in reasons)

    def test_high_cvss_adds_5_points(self):
        from manus_agent.tools.diff_report_cves import _priority_score

        profile = {
            "kev": {},
            "cvss": {"score": 8.1, "attack_vector": None, "privileges_required": None, "user_interaction": None},
            "epss": {},
        }
        score, _ = _priority_score(profile)
        assert score == 5

    def test_medium_cvss_adds_2_points(self):
        from manus_agent.tools.diff_report_cves import _priority_score

        profile = {
            "kev": {},
            "cvss": {"score": 5.0, "attack_vector": None, "privileges_required": None, "user_interaction": None},
            "epss": {},
        }
        score, _ = _priority_score(profile)
        assert score == 2

    def test_epss_very_high_adds_8(self):
        from manus_agent.tools.diff_report_cves import _priority_score

        profile = {
            "kev": {},
            "cvss": {},
            "epss": {"score": 0.85},
        }
        score, reasons = _priority_score(profile)
        assert score == 8
        assert any("very high" in r for r in reasons)

    def test_network_av_adds_3(self):
        from manus_agent.tools.diff_report_cves import _priority_score

        profile = {
            "kev": {},
            "cvss": {"score": None, "attack_vector": "NETWORK", "privileges_required": None, "user_interaction": None},
            "epss": {},
        }
        score, _ = _priority_score(profile)
        assert score == 3

    def test_no_privileges_adds_2(self):
        from manus_agent.tools.diff_report_cves import _priority_score

        profile = {
            "kev": {},
            "cvss": {
                "score": None,
                "attack_vector": None,
                "privileges_required": "NONE",
                "user_interaction": None,
            },
            "epss": {},
        }
        score, _ = _priority_score(profile)
        assert score == 2

    def test_no_user_interaction_adds_1(self):
        from manus_agent.tools.diff_report_cves import _priority_score

        profile = {
            "kev": {},
            "cvss": {
                "score": None,
                "attack_vector": None,
                "privileges_required": None,
                "user_interaction": "NONE",
            },
            "epss": {},
        }
        score, _ = _priority_score(profile)
        assert score == 1

    def test_composite_log4shell(self):
        from manus_agent.tools.diff_report_cves import _priority_score

        # KEV(10) + CVSS 10.0(8) + EPSS 0.975(8) + NETWORK(3) + NO_PR(2) + NO_UI(1) = 32
        profile = {
            "kev": {"in_kev": True},
            "cvss": {
                "score": 10.0,
                "attack_vector": "NETWORK",
                "privileges_required": "NONE",
                "user_interaction": "NONE",
            },
            "epss": {"score": 0.975},
        }
        score, _ = _priority_score(profile)
        assert score == 32.0


class TestBuildDiffReport:
    def _make_profiles(self):

        def mock_get(url, **kwargs):
            if "CVE-2021-44228" in url or "first.org" in url and "44228" in str(kwargs):
                pass
            return None

        # Build synthetic profiles directly
        profile_a = {
            "cve_id": "CVE-2021-44228",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {
                "version": "3.1",
                "score": 10.0,
                "severity": "CRITICAL",
                "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                "attack_vector": "NETWORK",
                "privileges_required": "NONE",
                "user_interaction": "NONE",
                "scope": "CHANGED",
                "confidentiality_impact": "HIGH",
                "integrity_impact": "HIGH",
                "availability_impact": "HIGH",
            },
            "epss": {"score": 0.975, "percentile": 0.999},
            "cwe": ["CWE-917"],
            "affected": "Apache / Log4J",
            "published": "2021-12-10",
            "description": "Apache Log4j2 RCE via JNDI lookup.",
            "references": ["https://logging.apache.org/log4j/2.x/security.html"],
        }
        profile_b = {
            "cve_id": "CVE-2021-45046",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {
                "version": "3.1",
                "score": 9.0,
                "severity": "CRITICAL",
                "vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:C/C:H/I:H/A:H",
                "attack_vector": "NETWORK",
                "privileges_required": "NONE",
                "user_interaction": "NONE",
                "scope": "CHANGED",
                "confidentiality_impact": "HIGH",
                "integrity_impact": "HIGH",
                "availability_impact": "HIGH",
            },
            "epss": {"score": 0.820, "percentile": 0.997},
            "cwe": ["CWE-917"],
            "affected": "Apache / Log4J",
            "published": "2021-12-14",
            "description": "Incomplete fix for CVE-2021-44228.",
            "references": [],
        }
        kev_a = {
            "in_kev": True,
            "date_added": "2021-12-10",
            "due_date": "2021-12-24",
            "required_action": "Apply updates.",
        }
        kev_b = {"in_kev": False}
        return profile_a, profile_b, kev_a, kev_b

    def test_winner_is_higher_scorer(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report

        pa, pb, kev_a, kev_b = self._make_profiles()
        report = _build_diff_report(pa, pb, kev_a, kev_b)
        # A has KEV (+10) so should win
        assert report["higher_priority"] == "CVE-2021-44228"
        assert report["lower_priority"] == "CVE-2021-45046"
        assert report["priority_score_a"] > report["priority_score_b"]

    def test_confidence_margin_strong(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report

        pa, pb, kev_a, kev_b = self._make_profiles()
        report = _build_diff_report(pa, pb, kev_a, kev_b)
        assert report["confidence"] == "strong"
        assert report["confidence_margin"] >= 10

    def test_tie_when_equal_scores(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report

        pa, pb, _, _ = self._make_profiles()
        # Make both identical — no KEV for either
        kev_empty = {"in_kev": False}
        # Same profile for both
        report = _build_diff_report(pa, pa, kev_empty, kev_empty)
        assert report["higher_priority"] == "tie"
        assert report["confidence"] == "tie"

    def test_cvss_delta_computed(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report

        pa, pb, kev_a, kev_b = self._make_profiles()
        report = _build_diff_report(pa, pb, kev_a, kev_b)
        # 10.0 - 9.0 = +1.0
        assert "+1.000" in report["cvss_delta"]

    def test_epss_delta_computed(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report

        pa, pb, kev_a, kev_b = self._make_profiles()
        report = _build_diff_report(pa, pb, kev_a, kev_b)
        # 0.975 - 0.820 = 0.155
        assert "0.155" in report["epss_delta"]

    def test_severity_comparison(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report

        pa, pb, kev_a, kev_b = self._make_profiles()
        report = _build_diff_report(pa, pb, kev_a, kev_b)
        # Both CRITICAL → equal
        assert report["severity_comparison"]["higher"] == "equal"

    def test_severity_comparison_a_higher(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report

        pa, pb, kev_a, kev_b = self._make_profiles()
        pb = dict(pb)
        pb["cvss"] = dict(pb["cvss"], severity="HIGH")
        report = _build_diff_report(pa, pb, kev_a, kev_b)
        assert report["severity_comparison"]["higher"] == "CVE-2021-44228"

    def test_generated_at_field_present(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report

        pa, pb, kev_a, kev_b = self._make_profiles()
        report = _build_diff_report(pa, pb, kev_a, kev_b)
        assert "generated_at" in report
        assert "UTC" in report["generated_at"]

    def test_delta_arrow_none_values(self):
        from manus_agent.tools.diff_report_cves import _delta_arrow

        assert _delta_arrow(None, 1.0) == "N/A"
        assert _delta_arrow(1.0, None) == "N/A"
        assert _delta_arrow(None, None) == "N/A"

    def test_delta_arrow_equal_values(self):
        from manus_agent.tools.diff_report_cves import _delta_arrow

        result = _delta_arrow(0.500, 0.500)
        assert result == "equal"

    def test_delta_arrow_positive(self):
        from manus_agent.tools.diff_report_cves import _delta_arrow

        result = _delta_arrow(9.8, 7.5)
        assert result.startswith("+2.300")

    def test_delta_arrow_negative(self):
        from manus_agent.tools.diff_report_cves import _delta_arrow

        result = _delta_arrow(5.0, 9.0)
        assert result.startswith("-4.000")


class TestRenderMarkdown:
    def _build_report(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report

        profile_a = {
            "cve_id": "CVE-2021-44228",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {
                "version": "3.1",
                "score": 10.0,
                "severity": "CRITICAL",
                "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                "attack_vector": "NETWORK",
                "privileges_required": "NONE",
                "user_interaction": "NONE",
                "scope": "CHANGED",
                "confidentiality_impact": "HIGH",
                "integrity_impact": "HIGH",
                "availability_impact": "HIGH",
            },
            "epss": {"score": 0.975, "percentile": 0.999},
            "cwe": ["CWE-917"],
            "affected": "Apache / Log4J",
            "published": "2021-12-10",
            "description": "Apache Log4j2 RCE via JNDI lookup.",
            "references": ["https://logging.apache.org/log4j/2.x/security.html"],
        }
        profile_b = {
            "cve_id": "CVE-2021-45046",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {
                "version": "3.1",
                "score": 9.0,
                "severity": "CRITICAL",
                "vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:C/C:H/I:H/A:H",
                "attack_vector": "NETWORK",
                "privileges_required": "NONE",
                "user_interaction": "NONE",
                "scope": "CHANGED",
                "confidentiality_impact": "HIGH",
                "integrity_impact": "HIGH",
                "availability_impact": "HIGH",
            },
            "epss": {"score": 0.820, "percentile": 0.997},
            "cwe": ["CWE-917"],
            "affected": "Apache / Log4J",
            "published": "2021-12-14",
            "description": "Incomplete fix.",
            "references": [],
        }
        kev_a = {"in_kev": True, "date_added": "2021-12-10", "due_date": "2021-12-24", "required_action": "Apply."}
        kev_b = {"in_kev": False}
        return _build_diff_report(profile_a, profile_b, kev_a, kev_b)

    def test_markdown_has_header(self):
        from manus_agent.tools.diff_report_cves import _render_markdown

        report = self._build_report()
        md = _render_markdown(report)
        assert "# CVE Diff Report:" in md
        assert "CVE-2021-44228" in md
        assert "CVE-2021-45046" in md

    def test_markdown_has_executive_summary(self):
        from manus_agent.tools.diff_report_cves import _render_markdown

        report = self._build_report()
        md = _render_markdown(report)
        assert "## Executive Summary" in md
        assert "Verdict" in md

    def test_markdown_has_comparison_table(self):
        from manus_agent.tools.diff_report_cves import _render_markdown

        report = self._build_report()
        md = _render_markdown(report)
        assert "## Side-by-Side Comparison" in md
        assert "| Dimension |" in md
        assert "CVSS Score" in md
        assert "EPSS Score" in md
        assert "CISA KEV" in md

    def test_markdown_has_cvss_delta(self):
        from manus_agent.tools.diff_report_cves import _render_markdown

        report = self._build_report()
        md = _render_markdown(report)
        assert "## CVSS Delta Analysis" in md
        assert "+1.0" in md

    def test_markdown_has_epss_divergence(self):
        from manus_agent.tools.diff_report_cves import _render_markdown

        report = self._build_report()
        md = _render_markdown(report)
        assert "## EPSS Exploitation Probability Divergence" in md

    def test_markdown_has_kev_section(self):
        from manus_agent.tools.diff_report_cves import _render_markdown

        report = self._build_report()
        md = _render_markdown(report)
        assert "## CISA KEV Exploitation Status" in md
        assert "CVE-2021-44228" in md

    def test_markdown_has_cwe_section(self):
        from manus_agent.tools.diff_report_cves import _render_markdown

        report = self._build_report()
        md = _render_markdown(report)
        assert "## CWE Weakness Class Comparison" in md
        assert "CWE-917" in md

    def test_markdown_shared_cwe_note(self):
        from manus_agent.tools.diff_report_cves import _render_markdown

        report = self._build_report()
        md = _render_markdown(report)
        # Both have CWE-917, should note shared classes
        assert "Shared weakness classes" in md

    def test_markdown_has_prioritisation(self):
        from manus_agent.tools.diff_report_cves import _render_markdown

        report = self._build_report()
        md = _render_markdown(report)
        assert "## Prioritisation Rationale" in md

    def test_markdown_has_references(self):
        from manus_agent.tools.diff_report_cves import _render_markdown

        report = self._build_report()
        md = _render_markdown(report)
        assert "## References" in md
        assert "nvd.nist.gov" in md

    def test_markdown_has_footer_disclaimer(self):
        from manus_agent.tools.diff_report_cves import _render_markdown

        report = self._build_report()
        md = _render_markdown(report)
        assert "manus-agent" in md
        assert "NVD" in md
        assert "FIRST EPSS" in md

    def test_markdown_tie_report(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report, _render_markdown

        profile = {
            "cve_id": "CVE-2021-44228",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {
                "version": "3.1",
                "score": 7.5,
                "severity": "HIGH",
                "vector": "X",
                "attack_vector": "NETWORK",
                "privileges_required": None,
                "user_interaction": None,
                "scope": None,
                "confidentiality_impact": None,
                "integrity_impact": None,
                "availability_impact": None,
            },
            "epss": {"score": 0.1, "percentile": 0.8},
            "cwe": ["CWE-79"],
            "affected": "Example Corp / Widget",
            "published": "2021-01-01",
            "description": "XSS",
            "references": [],
        }
        profile_b = dict(profile, cve_id="CVE-2021-99999")
        report = _build_diff_report(profile, profile_b, {"in_kev": False}, {"in_kev": False})
        md = _render_markdown(report)
        assert "Tie" in md or "tie" in md.lower()

    def test_markdown_both_kev(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report, _render_markdown

        profile_a = {
            "cve_id": "CVE-2021-44228",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {
                "version": "3.1",
                "score": 10.0,
                "severity": "CRITICAL",
                "vector": "X",
                "attack_vector": "NETWORK",
                "privileges_required": "NONE",
                "user_interaction": "NONE",
                "scope": None,
                "confidentiality_impact": None,
                "integrity_impact": None,
                "availability_impact": None,
            },
            "epss": {"score": 0.97, "percentile": 0.99},
            "cwe": ["CWE-917"],
            "affected": "Apache / Log4J",
            "published": "2021-12-10",
            "description": "Log4Shell",
            "references": [],
        }
        profile_b = dict(profile_a, cve_id="CVE-2021-45046")
        kev_both = {"in_kev": True, "date_added": "2021-12-10", "due_date": "2021-12-24", "required_action": "Apply."}
        report = _build_diff_report(profile_a, profile_b, kev_both, kev_both)
        md = _render_markdown(report)
        assert "Both" in md or "both" in md.lower()

    def test_markdown_neither_kev(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report, _render_markdown

        profile = {
            "cve_id": "CVE-2021-44228",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {
                "version": "3.1",
                "score": 7.5,
                "severity": "HIGH",
                "vector": "X",
                "attack_vector": "NETWORK",
                "privileges_required": None,
                "user_interaction": None,
                "scope": None,
                "confidentiality_impact": None,
                "integrity_impact": None,
                "availability_impact": None,
            },
            "epss": {"score": 0.1, "percentile": 0.5},
            "cwe": ["CWE-79"],
            "affected": "Foo / Bar",
            "published": "2021-01-01",
            "description": "XSS",
            "references": [],
        }
        profile_b = dict(profile, cve_id="CVE-2021-99999")
        report = _build_diff_report(profile, profile_b, {"in_kev": False}, {"in_kev": False})
        md = _render_markdown(report)
        assert "Neither" in md

    def test_markdown_nvd_error_note(self):
        from manus_agent.tools.diff_report_cves import _build_diff_report, _render_markdown

        profile_a = {
            "cve_id": "CVE-9999-0001",
            "nvd_error": "No NVD record found",
            "epss_error": None,
            "cvss": {
                "version": None,
                "score": None,
                "severity": None,
                "vector": None,
                "attack_vector": None,
                "privileges_required": None,
                "user_interaction": None,
                "scope": None,
                "confidentiality_impact": None,
                "integrity_impact": None,
                "availability_impact": None,
            },
            "epss": {"score": 0.05, "percentile": 0.3},
            "cwe": [],
            "affected": "Unknown",
            "published": "Unknown",
            "description": "",
            "references": [],
        }
        profile_b = {
            "cve_id": "CVE-9999-0002",
            "nvd_error": None,
            "epss_error": None,
            "cvss": {
                "version": "3.1",
                "score": 5.5,
                "severity": "MEDIUM",
                "vector": "X",
                "attack_vector": "NETWORK",
                "privileges_required": None,
                "user_interaction": None,
                "scope": None,
                "confidentiality_impact": None,
                "integrity_impact": None,
                "availability_impact": None,
            },
            "epss": {"score": 0.05, "percentile": 0.3},
            "cwe": ["CWE-79"],
            "affected": "Foo / Bar",
            "published": "2021-01-01",
            "description": "XSS",
            "references": [],
        }
        report = _build_diff_report(profile_a, profile_b, {"in_kev": False}, {"in_kev": False})
        md = _render_markdown(report)
        assert "Source unavailable" in md


class TestDiffReportCvesTool:
    """Tests for the Strands tool entry point."""

    def _make_tool_use(self, cve_id_a: str, cve_id_b: str, fmt: str = "markdown") -> dict:
        return {
            "toolUseId": "test-tool-use-id",
            "name": "diff_report_cves",
            "input": {"cve_id_a": cve_id_a, "cve_id_b": cve_id_b, "output_format": fmt},
        }

    def _make_mock_get(self):
        def mock_get(url, **kwargs):
            params = kwargs.get("params", {})
            if "nvd.nist.gov" in url:
                if "44228" in url:
                    return _mock_response(_NVD_RESPONSE_A)
                return _mock_response(_NVD_RESPONSE_B)
            if "first.org" in url:
                if "44228" in str(params):
                    return _mock_response(_EPSS_RESPONSE_A)
                return _mock_response(_EPSS_RESPONSE_B)
            if "cisa.gov" in url:
                return _mock_response(_KEV_CATALOG)
            return _mock_response({})

        return mock_get

    def test_tool_returns_markdown_success(self):
        from manus_agent.tools.diff_report_cves import diff_report_cves

        tool = self._make_tool_use("CVE-2021-44228", "CVE-2021-45046", "markdown")
        with patch("requests.get", side_effect=self._make_mock_get()):
            result = diff_report_cves(tool)

        assert result["status"] == "success"
        text_content = next(c["text"] for c in result["content"] if "text" in c)
        assert "# CVE Diff Report:" in text_content
        json_content = next(c["json"] for c in result["content"] if "json" in c)
        assert "higher_priority" in json_content

    def test_tool_returns_json_success(self):
        from manus_agent.tools.diff_report_cves import diff_report_cves

        tool = self._make_tool_use("CVE-2021-44228", "CVE-2021-45046", "json")
        with patch("requests.get", side_effect=self._make_mock_get()):
            result = diff_report_cves(tool)

        assert result["status"] == "success"
        assert len(result["content"]) == 1
        assert "json" in result["content"][0]

    def test_tool_invalid_cve_a(self):
        from manus_agent.tools.diff_report_cves import diff_report_cves

        tool = self._make_tool_use("NOT-A-CVE", "CVE-2021-45046")
        result = diff_report_cves(tool)
        assert result["status"] == "error"
        assert "NOT-A-CVE" in result["content"][0]["text"]

    def test_tool_invalid_cve_b(self):
        from manus_agent.tools.diff_report_cves import diff_report_cves

        tool = self._make_tool_use("CVE-2021-44228", "BADID")
        result = diff_report_cves(tool)
        assert result["status"] == "error"
        assert "BADID" in result["content"][0]["text"]

    def test_tool_degraded_nvd(self):
        """Tool still returns success even when NVD is unavailable."""
        import requests as _req

        from manus_agent.tools.diff_report_cves import diff_report_cves

        def mock_get(url, **kwargs):
            if "nvd.nist.gov" in url:
                raise _req.RequestException("timeout")
            if "first.org" in url:
                return _mock_response(_EPSS_RESPONSE_A)
            if "cisa.gov" in url:
                return _mock_response(_KEV_EMPTY)
            return _mock_response({})

        tool = self._make_tool_use("CVE-2021-44228", "CVE-2021-45046")
        with patch("requests.get", side_effect=mock_get):
            result = diff_report_cves(tool)

        assert result["status"] == "success"

    def test_tool_output_logged(self):
        from manus_agent.tools.diff_report_cves import diff_report_cves

        tool = self._make_tool_use("CVE-2021-44228", "CVE-2021-45046")
        with patch("requests.get", side_effect=self._make_mock_get()):
            with patch("manus_agent.tools.diff_report_cves.log_tool_output_size") as mock_log:
                diff_report_cves(tool)
                mock_log.assert_called_once()
                call_args = mock_log.call_args[0]
                assert call_args[0] == "diff_report_cves"


# ──────────────────────────────────────────────────────────────────────────────
# CLI tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDiffReportCLI:
    def _make_mock_get(self):
        def mock_get(url, **kwargs):
            if "nvd.nist.gov" in url:
                if "44228" in url:
                    return _mock_response(_NVD_RESPONSE_A)
                return _mock_response(_NVD_RESPONSE_B)
            if "first.org" in url:
                return _mock_response(_EPSS_RESPONSE_A)
            if "cisa.gov" in url:
                return _mock_response(_KEV_CATALOG)
            return _mock_response({})

        return mock_get

    def test_cli_markdown_output(self, capsys):
        from manus_agent.cli import _run_diff_report

        with patch("requests.get", side_effect=self._make_mock_get()):
            rc = _run_diff_report(["CVE-2021-44228", "CVE-2021-45046"])

        assert rc == 0

    def test_cli_json_output(self, capsys):
        from manus_agent.cli import _run_diff_report

        with patch("requests.get", side_effect=self._make_mock_get()):
            rc = _run_diff_report(["CVE-2021-44228", "CVE-2021-45046", "--output", "json"])

        captured = capsys.readouterr()
        assert rc == 0
        parsed = json.loads(captured.out)
        assert "higher_priority" in parsed

    def test_cli_save_to_file(self, tmp_path):
        from manus_agent.cli import _run_diff_report

        out_file = tmp_path / "report.md"
        with patch("requests.get", side_effect=self._make_mock_get()):
            rc = _run_diff_report(["CVE-2021-44228", "CVE-2021-45046", "--save", str(out_file)])

        assert rc == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "# CVE Diff Report:" in content

    def test_cli_invalid_cve_exits_with_error(self):
        from manus_agent.cli import _run_diff_report

        with pytest.raises(SystemExit) as exc_info:
            _run_diff_report(["NOTACVE", "CVE-2021-45046"])
        assert exc_info.value.code != 0

    def test_cli_build_diff_report_parser_help(self):
        from manus_agent.cli import _build_diff_report_parser

        p = _build_diff_report_parser()
        assert "diff-report" in p.prog
        assert p.description is not None

    def test_cli_in_subcommands_set(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "diff-report" in _SUBCOMMANDS


class TestEpssLabel:
    def test_epss_none(self):
        from manus_agent.tools.diff_report_cves import _epss_label

        assert _epss_label(None) == "N/A"

    def test_epss_very_high(self):
        from manus_agent.tools.diff_report_cves import _epss_label

        result = _epss_label(0.75)
        assert "very high" in result
        assert "0.750" in result

    def test_epss_high(self):
        from manus_agent.tools.diff_report_cves import _epss_label

        result = _epss_label(0.50)
        assert "high" in result

    def test_epss_elevated(self):
        from manus_agent.tools.diff_report_cves import _epss_label

        result = _epss_label(0.15)
        assert "elevated" in result

    def test_epss_low(self):
        from manus_agent.tools.diff_report_cves import _epss_label

        result = _epss_label(0.01)
        assert "low" in result


class TestNaHelper:
    def test_na_none(self):
        from manus_agent.tools.diff_report_cves import _na

        assert _na(None) == "N/A"

    def test_na_value(self):
        from manus_agent.tools.diff_report_cves import _na

        assert _na(42) == "42"
        assert _na("NETWORK") == "NETWORK"
        assert _na(0) == "0"
