"""Comprehensive test suite for cve_enrich tool and CLI subcommand.

All HTTP calls are fully mocked — no real network access.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.cve_enrich import (
    _compute_risk_level,
    _fetch_cisa_kev,
    _fetch_epss,
    _fetch_nvd,
    _fetch_osv,
    _fetch_vulncheck_kev,
    enrich_cve,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_NVD_RESPONSE = {
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2024-3094",
                "published": "2024-03-29T17:15:00.000",
                "lastModified": "2024-04-12T12:00:00.000",
                "vulnStatus": "Analyzed",
                "descriptions": [{"lang": "en", "value": "XZ Utils backdoor via build process manipulation."}],
                "metrics": {
                    "cvssMetricV31": [
                        {
                            "cvssData": {
                                "baseScore": 10.0,
                                "baseSeverity": "CRITICAL",
                                "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                            }
                        }
                    ]
                },
                "weaknesses": [{"description": [{"lang": "en", "value": "CWE-506"}]}],
                "references": [
                    {"url": "https://example.com/advisory", "source": "cve@mitre.org", "tags": ["Advisory"]},
                    {"url": "https://github.com/example/commit/abc123", "source": "cve@mitre.org", "tags": ["Patch"]},
                ],
            }
        }
    ]
}

_SAMPLE_EPSS_RESPONSE = {
    "data": [{"cve": "CVE-2024-3094", "epss": "0.9752", "percentile": "0.9998", "date": "2024-04-01"}]
}

_SAMPLE_KEV_RESPONSE = {
    "vulnerabilities": [
        {
            "cveID": "CVE-2024-3094",
            "vendorProject": "XZ Utils",
            "product": "xz",
            "dateAdded": "2024-03-30",
            "dueDate": "2024-04-15",
            "requiredAction": "Apply mitigations per vendor instructions.",
            "knownRansomwareCampaignUse": "Unknown",
        }
    ]
}

_SAMPLE_OSV_RESPONSE = {
    "aliases": ["CVE-2024-3094", "GHSA-jq36-prxc-m8vr"],
    "affected": [
        {
            "package": {"ecosystem": "Debian", "name": "xz-utils"},
            "ranges": [
                {
                    "type": "ECOSYSTEM",
                    "events": [{"introduced": "0"}, {"fixed": "5.6.1+really5.4.5-1"}],
                }
            ],
        }
    ],
}

_SAMPLE_VULNCHECK_KEV_RESPONSE = {
    "data": [
        {
            "date_added": "2024-03-30",
            "exploit_maturity": "active",
            "reporting_source": "multi-source",
        }
    ]
}


# ---------------------------------------------------------------------------
# _fetch_nvd tests
# ---------------------------------------------------------------------------


class TestFetchNvd:
    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _SAMPLE_NVD_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_nvd("CVE-2024-3094")
        assert result["source"] == "nvd"
        assert result["cvss_score"] == 10.0
        assert result["cvss_severity"] == "CRITICAL"
        assert result["cwe_ids"] == ["CWE-506"]
        assert "XZ Utils" in result["description"]
        assert result["published"] == "2024-03-29T17:15:00.000"
        assert result["reference_count"] == 2

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_no_vulnerabilities(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_nvd("CVE-9999-99999")
        assert "error" in result

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_network_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("timeout")
        result = _fetch_nvd("CVE-2024-3094")
        assert result["source"] == "nvd"
        assert "error" in result

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_cvss_v30_fallback(self, mock_get):
        """Uses CVSS v3.0 when v3.1 is absent."""
        response = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2020-1234",
                        "published": "2020-01-01",
                        "lastModified": "2020-02-01",
                        "vulnStatus": "Analyzed",
                        "descriptions": [{"lang": "en", "value": "Test vuln"}],
                        "metrics": {
                            "cvssMetricV30": [
                                {
                                    "cvssData": {
                                        "baseScore": 7.5,
                                        "baseSeverity": "HIGH",
                                        "vectorString": "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
                                    }
                                }
                            ]
                        },
                        "weaknesses": [],
                        "references": [],
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_nvd("CVE-2020-1234")
        assert result["cvss_score"] == 7.5
        assert result["cvss_severity"] == "HIGH"

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_no_cvss_data(self, mock_get):
        """Handles missing CVSS gracefully."""
        response = {
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2024-0001",
                        "published": "2024-01-01",
                        "lastModified": "2024-01-02",
                        "vulnStatus": "Awaiting Analysis",
                        "descriptions": [{"lang": "en", "value": "Pending"}],
                        "metrics": {},
                        "weaknesses": [],
                        "references": [],
                    }
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_nvd("CVE-2024-0001")
        assert result["cvss_score"] is None
        assert result["cvss_severity"] is None

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_nvd_api_key_header(self, mock_get, monkeypatch):
        """NVD_API_KEY is sent as a header."""
        monkeypatch.setenv("NVD_API_KEY", "test-key-123")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        _fetch_nvd("CVE-2024-3094")
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["headers"]["apiKey"] == "test-key-123"


# ---------------------------------------------------------------------------
# _fetch_epss tests
# ---------------------------------------------------------------------------


class TestFetchEpss:
    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _SAMPLE_EPSS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_epss("CVE-2024-3094")
        assert result["source"] == "epss"
        assert result["score"] == pytest.approx(0.9752)
        assert result["percentile"] == pytest.approx(0.9998)
        assert result["date"] == "2024-04-01"

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_no_data(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_epss("CVE-9999-99999")
        assert result["score"] is None
        assert result["percentile"] is None

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_network_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("timeout")
        result = _fetch_epss("CVE-2024-3094")
        assert "error" in result


# ---------------------------------------------------------------------------
# _fetch_cisa_kev tests
# ---------------------------------------------------------------------------


class TestFetchCisaKev:
    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_found_in_kev(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _SAMPLE_KEV_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_cisa_kev("CVE-2024-3094")
        assert result["in_kev"] is True
        assert result["vendor"] == "XZ Utils"
        assert result["date_added"] == "2024-03-30"
        assert result["due_date"] == "2024-04-15"

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_not_in_kev(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [{"cveID": "CVE-OTHER-1234"}]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_cisa_kev("CVE-2024-3094")
        assert result["in_kev"] is False

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_case_insensitive_match(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _SAMPLE_KEV_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_cisa_kev("cve-2024-3094")
        assert result["in_kev"] is True

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_network_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("failed")
        result = _fetch_cisa_kev("CVE-2024-3094")
        assert "error" in result


# ---------------------------------------------------------------------------
# _fetch_osv tests
# ---------------------------------------------------------------------------


class TestFetchOsv:
    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _SAMPLE_OSV_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_osv("CVE-2024-3094")
        assert result["found"] is True
        assert len(result["affected_packages"]) == 1
        assert result["affected_packages"][0]["ecosystem"] == "Debian"
        assert result["affected_packages"][0]["name"] == "xz-utils"
        assert "5.6.1+really5.4.5-1" in result["affected_packages"][0]["fixed_versions"]
        assert "GHSA-jq36-prxc-m8vr" in result["aliases"]

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_not_found_404(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = _fetch_osv("CVE-9999-99999")
        assert result["found"] is False
        assert result["affected_packages"] == []

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_no_affected_field(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"aliases": ["CVE-2024-0001"]}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_osv("CVE-2024-0001")
        assert result["found"] is True
        assert result["affected_packages"] == []

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_network_error(self, mock_get):
        import requests

        mock_get.side_effect = requests.exceptions.Timeout("slow")
        result = _fetch_osv("CVE-2024-3094")
        assert "error" in result


# ---------------------------------------------------------------------------
# _fetch_vulncheck_kev tests
# ---------------------------------------------------------------------------


class TestFetchVulncheckKev:
    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_found(self, mock_get, monkeypatch):
        monkeypatch.setenv("VULNCHECK_API_KEY", "vc-test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = _SAMPLE_VULNCHECK_KEV_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_vulncheck_kev("CVE-2024-3094")
        assert result["available"] is True
        assert result["in_kev"] is True
        assert result["exploit_maturity"] == "active"

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_not_found(self, mock_get, monkeypatch):
        monkeypatch.setenv("VULNCHECK_API_KEY", "vc-test-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _fetch_vulncheck_kev("CVE-2024-3094")
        assert result["available"] is True
        assert result["in_kev"] is False

    def test_no_api_key(self, monkeypatch):
        monkeypatch.delenv("VULNCHECK_API_KEY", raising=False)
        result = _fetch_vulncheck_kev("CVE-2024-3094")
        assert result["available"] is False
        assert result["reason"] == "no_api_key"

    @patch("manus_agent.tools.cve_enrich.requests.get")
    def test_auth_header_sent(self, mock_get, monkeypatch):
        monkeypatch.setenv("VULNCHECK_API_KEY", "my-secret-key")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        _fetch_vulncheck_kev("CVE-2024-3094")
        call_kwargs = mock_get.call_args
        assert "Bearer my-secret-key" in call_kwargs[1]["headers"]["Authorization"]


# ---------------------------------------------------------------------------
# _compute_risk_level tests
# ---------------------------------------------------------------------------


class TestComputeRiskLevel:
    def test_critical_all_signals(self):
        result = _compute_risk_level(10.0, 0.95, True, True)
        assert result["level"] == "critical"
        assert result["score"] >= 60
        assert len(result["signals"]) == 4

    def test_high_kev_plus_medium_cvss(self):
        result = _compute_risk_level(5.0, 0.05, True, False)
        assert result["level"] == "high"

    def test_medium_high_cvss_no_exploitation(self):
        result = _compute_risk_level(8.5, 0.02, False, False)
        assert result["level"] == "medium"

    def test_low_score(self):
        result = _compute_risk_level(3.0, 0.001, False, False)
        assert result["level"] == "low"

    def test_unknown_no_data(self):
        result = _compute_risk_level(None, None, False, False)
        assert result["level"] == "unknown"
        assert result["score"] == 0.0

    def test_epss_high_alone_is_high(self):
        result = _compute_risk_level(None, 0.8, False, False)
        assert result["level"] in ("high", "medium")
        assert any("EPSS" in s for s in result["signals"])

    def test_vulncheck_kev_adds_signal(self):
        result = _compute_risk_level(7.0, 0.05, False, True)
        assert any("VulnCheck" in s for s in result["signals"])


# ---------------------------------------------------------------------------
# enrich_cve integration tests (all sources mocked)
# ---------------------------------------------------------------------------


class TestEnrichCve:
    @patch("manus_agent.tools.cve_enrich._fetch_vulncheck_kev")
    @patch("manus_agent.tools.cve_enrich._fetch_osv")
    @patch("manus_agent.tools.cve_enrich._fetch_cisa_kev")
    @patch("manus_agent.tools.cve_enrich._fetch_epss")
    @patch("manus_agent.tools.cve_enrich._fetch_nvd")
    def test_full_enrichment(self, mock_nvd, mock_epss, mock_kev, mock_osv, mock_vc):
        mock_nvd.return_value = {
            "source": "nvd",
            "cvss_score": 10.0,
            "cvss_severity": "CRITICAL",
            "cwe_ids": ["CWE-506"],
            "description": "XZ backdoor",
            "published": "2024-03-29",
        }
        mock_epss.return_value = {"source": "epss", "score": 0.975, "percentile": 0.999}
        mock_kev.return_value = {"source": "cisa_kev", "in_kev": True, "vendor": "XZ"}
        mock_osv.return_value = {
            "source": "osv",
            "found": True,
            "affected_packages": [{"ecosystem": "Debian", "name": "xz-utils", "fixed_versions": ["5.4.5"]}],
        }
        mock_vc.return_value = {"source": "vulncheck_kev", "available": True, "in_kev": True}

        result = enrich_cve("CVE-2024-3094")
        assert result["cve_id"] == "CVE-2024-3094"
        assert result["risk_assessment"]["level"] == "critical"
        assert result["nvd"]["cvss_score"] == 10.0
        assert result["epss"]["score"] == 0.975
        assert result["cisa_kev"]["in_kev"] is True
        assert result["osv"]["found"] is True
        assert result["vulncheck_kev"]["in_kev"] is True

    def test_invalid_cve_format(self):
        result = enrich_cve("not-a-cve")
        assert "error" in result
        assert "Invalid CVE ID" in result["error"]

    def test_invalid_cve_short(self):
        result = enrich_cve("CVE-2024-12")
        assert "error" in result

    @patch("manus_agent.tools.cve_enrich._fetch_vulncheck_kev")
    @patch("manus_agent.tools.cve_enrich._fetch_osv")
    @patch("manus_agent.tools.cve_enrich._fetch_cisa_kev")
    @patch("manus_agent.tools.cve_enrich._fetch_epss")
    @patch("manus_agent.tools.cve_enrich._fetch_nvd")
    def test_without_vulncheck(self, mock_nvd, mock_epss, mock_kev, mock_osv, mock_vc):
        mock_nvd.return_value = {"source": "nvd", "cvss_score": 5.0}
        mock_epss.return_value = {"source": "epss", "score": 0.01}
        mock_kev.return_value = {"source": "cisa_kev", "in_kev": False}
        mock_osv.return_value = {"source": "osv", "found": False, "affected_packages": []}

        result = enrich_cve("CVE-2024-3094", include_vulncheck=False)
        mock_vc.assert_not_called()
        # vulncheck_kev key will be empty dict or missing
        assert result.get("vulncheck_kev") == {} or "vulncheck_kev" not in result

    @patch("manus_agent.tools.cve_enrich._fetch_vulncheck_kev")
    @patch("manus_agent.tools.cve_enrich._fetch_osv")
    @patch("manus_agent.tools.cve_enrich._fetch_cisa_kev")
    @patch("manus_agent.tools.cve_enrich._fetch_epss")
    @patch("manus_agent.tools.cve_enrich._fetch_nvd")
    def test_case_normalization(self, mock_nvd, mock_epss, mock_kev, mock_osv, mock_vc):
        mock_nvd.return_value = {"source": "nvd", "cvss_score": 5.0}
        mock_epss.return_value = {"source": "epss", "score": 0.01}
        mock_kev.return_value = {"source": "cisa_kev", "in_kev": False}
        mock_osv.return_value = {"source": "osv", "found": False, "affected_packages": []}
        mock_vc.return_value = {"source": "vulncheck_kev", "available": False}

        result = enrich_cve("cve-2024-3094")
        assert result["cve_id"] == "CVE-2024-3094"

    @patch("manus_agent.tools.cve_enrich._fetch_vulncheck_kev")
    @patch("manus_agent.tools.cve_enrich._fetch_osv")
    @patch("manus_agent.tools.cve_enrich._fetch_cisa_kev")
    @patch("manus_agent.tools.cve_enrich._fetch_epss")
    @patch("manus_agent.tools.cve_enrich._fetch_nvd")
    def test_partial_failure_graceful(self, mock_nvd, mock_epss, mock_kev, mock_osv, mock_vc):
        """If one source fails, others still return data."""
        mock_nvd.return_value = {"source": "nvd", "error": "timeout"}
        mock_epss.return_value = {"source": "epss", "score": 0.5, "percentile": 0.9}
        mock_kev.return_value = {"source": "cisa_kev", "in_kev": False}
        mock_osv.return_value = {"source": "osv", "found": True, "affected_packages": []}
        mock_vc.return_value = {"source": "vulncheck_kev", "available": False}

        result = enrich_cve("CVE-2024-3094")
        # Should still have all keys
        assert "nvd" in result
        assert "epss" in result
        assert result["nvd"]["error"] == "timeout"
        assert result["epss"]["score"] == 0.5

    @patch("manus_agent.tools.cve_enrich._fetch_vulncheck_kev")
    @patch("manus_agent.tools.cve_enrich._fetch_osv")
    @patch("manus_agent.tools.cve_enrich._fetch_cisa_kev")
    @patch("manus_agent.tools.cve_enrich._fetch_epss")
    @patch("manus_agent.tools.cve_enrich._fetch_nvd")
    def test_exception_in_fetcher_caught(self, mock_nvd, mock_epss, mock_kev, mock_osv, mock_vc):
        """An unhandled exception in a fetcher is caught and reported."""
        mock_nvd.side_effect = RuntimeError("unexpected crash")
        mock_epss.return_value = {"source": "epss", "score": 0.01}
        mock_kev.return_value = {"source": "cisa_kev", "in_kev": False}
        mock_osv.return_value = {"source": "osv", "found": False, "affected_packages": []}
        mock_vc.return_value = {"source": "vulncheck_kev", "available": False}

        result = enrich_cve("CVE-2024-3094")
        assert "error" in result["nvd"]
        assert "unexpected crash" in result["nvd"]["error"]

    @patch("manus_agent.tools.cve_enrich._fetch_vulncheck_kev")
    @patch("manus_agent.tools.cve_enrich._fetch_osv")
    @patch("manus_agent.tools.cve_enrich._fetch_cisa_kev")
    @patch("manus_agent.tools.cve_enrich._fetch_epss")
    @patch("manus_agent.tools.cve_enrich._fetch_nvd")
    def test_whitespace_in_cve_id(self, mock_nvd, mock_epss, mock_kev, mock_osv, mock_vc):
        mock_nvd.return_value = {"source": "nvd", "cvss_score": 7.0}
        mock_epss.return_value = {"source": "epss", "score": 0.1}
        mock_kev.return_value = {"source": "cisa_kev", "in_kev": False}
        mock_osv.return_value = {"source": "osv", "found": False, "affected_packages": []}
        mock_vc.return_value = {"source": "vulncheck_kev", "available": False}

        result = enrich_cve("  CVE-2024-3094  ")
        assert result["cve_id"] == "CVE-2024-3094"


# ---------------------------------------------------------------------------
# CLI _run_enrich tests
# ---------------------------------------------------------------------------


class TestRunEnrichCli:
    @patch("manus_agent.tools.cve_enrich.enrich_cve")
    def test_json_output(self, mock_enrich, capsys):
        mock_enrich.return_value = {
            "cve_id": "CVE-2024-3094",
            "risk_assessment": {"level": "critical", "score": 90, "signals": ["CISA KEV"]},
            "nvd": {"cvss_score": 10.0},
            "epss": {"score": 0.95},
            "cisa_kev": {"in_kev": True},
            "osv": {"found": True, "affected_packages": []},
            "vulncheck_kev": {"available": True, "in_kev": True},
        }

        from manus_agent.cli import _run_enrich

        exit_code = _run_enrich(["CVE-2024-3094", "--output", "json"])
        assert exit_code == 0

    @patch("manus_agent.tools.cve_enrich.enrich_cve")
    def test_text_output(self, mock_enrich, capsys):
        mock_enrich.return_value = {
            "cve_id": "CVE-2024-3094",
            "risk_assessment": {"level": "high", "score": 50, "signals": ["CVSS 8.0 (high)"]},
            "nvd": {
                "cvss_score": 8.0,
                "cvss_severity": "HIGH",
                "cwe_ids": ["CWE-79"],
                "published": "2024-01-01",
                "status": "Analyzed",
                "description": "Test vuln",
            },
            "epss": {"score": 0.15, "percentile": 0.85},
            "cisa_kev": {"in_kev": False},
            "osv": {
                "found": True,
                "affected_packages": [{"ecosystem": "PyPI", "name": "flask", "fixed_versions": ["2.3.3"]}],
            },
            "vulncheck_kev": {"available": True, "in_kev": False},
        }

        from manus_agent.cli import _run_enrich

        exit_code = _run_enrich(["CVE-2024-3094"])
        assert exit_code == 0

    def test_invalid_cve_id(self):
        from manus_agent.cli import _run_enrich

        exit_code = _run_enrich(["not-a-cve"])
        assert exit_code == 1

    @patch("manus_agent.tools.cve_enrich.enrich_cve")
    def test_enrich_error_result(self, mock_enrich):
        mock_enrich.return_value = {"error": "Invalid CVE ID format"}

        from manus_agent.cli import _run_enrich

        exit_code = _run_enrich(["CVE-2024-3094"])
        assert exit_code == 1

    @patch("manus_agent.tools.cve_enrich.enrich_cve")
    def test_no_vulncheck_flag(self, mock_enrich):
        mock_enrich.return_value = {
            "cve_id": "CVE-2024-3094",
            "risk_assessment": {"level": "low", "score": 5, "signals": []},
            "nvd": {"cvss_score": 3.0},
            "epss": {"score": 0.001},
            "cisa_kev": {"in_kev": False},
            "osv": {"found": False, "affected_packages": []},
            "vulncheck_kev": {},
        }

        from manus_agent.cli import _run_enrich

        exit_code = _run_enrich(["CVE-2024-3094", "--no-vulncheck"])
        assert exit_code == 0
        # Verify include_vulncheck=False was passed
        mock_enrich.assert_called_once_with("CVE-2024-3094", include_vulncheck=False)


# ---------------------------------------------------------------------------
# Strands @tool interface test
# ---------------------------------------------------------------------------


class TestCveEnrichTool:
    @patch("manus_agent.tools.cve_enrich.enrich_cve")
    def test_tool_interface(self, mock_enrich):
        mock_enrich.return_value = {"cve_id": "CVE-2024-3094", "risk_assessment": {"level": "high"}}

        from manus_agent.tools.cve_enrich import cve_enrich

        cve_enrich("CVE-2024-3094")
        mock_enrich.assert_called_once_with("CVE-2024-3094", include_vulncheck=True)

    @patch("manus_agent.tools.cve_enrich.enrich_cve")
    def test_tool_interface_no_vulncheck(self, mock_enrich):
        mock_enrich.return_value = {"cve_id": "CVE-2024-3094", "risk_assessment": {"level": "low"}}

        from manus_agent.tools.cve_enrich import cve_enrich

        cve_enrich("CVE-2024-3094", include_vulncheck="false")
        mock_enrich.assert_called_once_with("CVE-2024-3094", include_vulncheck=False)
