"""Tests for generate_cve_report tool and manus-agent report CLI subcommand.

All HTTP calls are mocked — zero real network traffic.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_CVE = "CVE-2021-44228"
_CVE_LOWER = "cve-2021-44228"

# Minimal NVD API response
_NVD_RESPONSE = {
    "vulnerabilities": [
        {
            "cve": {
                "id": _CVE,
                "published": "2021-12-10T10:15:00.000",
                "lastModified": "2023-09-14T11:15:00.000",
                "vulnStatus": "Analyzed",
                "descriptions": [
                    {"lang": "en", "value": "Apache Log4j2 JNDI remote code execution vulnerability."},
                    {"lang": "es", "value": "Spanish description."},
                ],
                "metrics": {
                    "cvssMetricV31": [
                        {
                            "cvssData": {
                                "version": "3.1",
                                "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                                "baseScore": 10.0,
                                "baseSeverity": "CRITICAL",
                            }
                        }
                    ]
                },
                "weaknesses": [{"description": [{"lang": "en", "value": "CWE-917"}]}],
                "configurations": [],
                "references": [
                    {"url": "https://logging.apache.org/log4j/2.x/security.html"},
                    {"url": "https://nvd.nist.gov/vuln/detail/CVE-2021-44228"},
                ],
            }
        }
    ]
}

# Minimal EPSS API response
_EPSS_RESPONSE = {
    "data": [
        {
            "cve": _CVE,
            "epss": "0.97565",
            "percentile": "0.99993",
            "date": "2024-01-01",
        }
    ]
}

# Minimal CISA KEV response — CVE IS in the catalog
_KEV_RESPONSE = {
    "vulnerabilities": [
        {
            "cveID": _CVE,
            "vendorProject": "Apache",
            "product": "Log4j",
            "vulnerabilityName": "Apache Log4j2 Remote Code Execution Vulnerability",
            "dateAdded": "2021-12-10",
            "shortDescription": "Apache Log4j2 allows remote code execution.",
            "requiredAction": "Apply updates per vendor instructions.",
            "dueDate": "2021-12-24",
            "notes": "",
        }
    ]
}

# Minimal OSV response
_OSV_RESPONSE = {
    "vulns": [
        {
            "id": "GHSA-jfh8-c2jp-hdp9",
            "affected": [
                {
                    "package": {"name": "log4j-core", "ecosystem": "Maven"},
                    "ranges": [
                        {
                            "type": "ECOSYSTEM",
                            "events": [
                                {"introduced": "2.0-beta9"},
                                {"fixed": "2.17.1"},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
}

# Minimal GitHub Advisory response
_GHSA_RESPONSE = [
    {
        "ghsa_id": "GHSA-jfh8-c2jp-hdp9",
        "summary": "Remote code execution in Log4j 2",
        "severity": "critical",
        "published_at": "2021-12-10T00:00:00Z",
        "updated_at": "2022-01-01T00:00:00Z",
        "html_url": "https://github.com/advisories/GHSA-jfh8-c2jp-hdp9",
        "cwes": [{"cwe_id": "CWE-917"}],
        "cvss": {"score": 10.0, "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H"},
        "vulnerabilities": [
            {
                "package": {"name": "log4j-core", "ecosystem": "Maven"},
                "vulnerable_version_range": ">= 2.0-beta9, < 2.17.1",
                "patched_versions": ">= 2.17.1",
                "first_patched_version": {"identifier": "2.17.1"},
            }
        ],
    }
]


def _make_response(data: Any, status: int = 200) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Unit tests — individual fetch helpers
# ---------------------------------------------------------------------------


class TestFetchNvd:
    def test_success(self):
        from manus_agent.tools.generate_cve_report import _fetch_nvd

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.return_value = _make_response(_NVD_RESPONSE)
            result = _fetch_nvd(_CVE)

        assert result["available"] is True
        assert "Log4j2" in result["description"]
        assert result["cvss_score"] == 10.0
        assert result["cvss_severity"] == "CRITICAL"
        assert result["cvss_version"] == "3.1"
        assert "CWE-917" in result["cwes"]
        assert result["published"].startswith("2021-12-10")

    def test_not_found(self):
        from manus_agent.tools.generate_cve_report import _fetch_nvd

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.return_value = _make_response({"vulnerabilities": []})
            result = _fetch_nvd(_CVE)

        assert result["available"] is False
        assert "not found" in result["reason"]

    def test_network_error(self):
        import requests as _requests

        from manus_agent.tools.generate_cve_report import _fetch_nvd

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.side_effect = _requests.ConnectionError("network down")
            result = _fetch_nvd(_CVE)

        assert result["available"] is False

    def test_picks_english_description(self):
        from manus_agent.tools.generate_cve_report import _fetch_nvd

        nvd_resp = json.loads(json.dumps(_NVD_RESPONSE))
        nvd_resp["vulnerabilities"][0]["cve"]["descriptions"] = [
            {"lang": "fr", "value": "Description française."},
            {"lang": "en", "value": "English description."},
        ]

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.return_value = _make_response(nvd_resp)
            result = _fetch_nvd(_CVE)

        assert result["description"] == "English description."

    def test_cvss_fallback_to_v30(self):
        from manus_agent.tools.generate_cve_report import _fetch_nvd

        nvd_resp = json.loads(json.dumps(_NVD_RESPONSE))
        # Remove v31, add v30
        nvd_resp["vulnerabilities"][0]["cve"]["metrics"].pop("cvssMetricV31")
        nvd_resp["vulnerabilities"][0]["cve"]["metrics"]["cvssMetricV30"] = [
            {
                "cvssData": {
                    "version": "3.0",
                    "vectorString": "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                    "baseScore": 10.0,
                    "baseSeverity": "CRITICAL",
                }
            }
        ]

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.return_value = _make_response(nvd_resp)
            result = _fetch_nvd(_CVE)

        assert result["cvss_score"] == 10.0
        assert result["cvss_version"] == "3.0"

    def test_multiple_cwes(self):
        from manus_agent.tools.generate_cve_report import _fetch_nvd

        nvd_resp = json.loads(json.dumps(_NVD_RESPONSE))
        nvd_resp["vulnerabilities"][0]["cve"]["weaknesses"] = [
            {"description": [{"lang": "en", "value": "CWE-917"}]},
            {"description": [{"lang": "en", "value": "CWE-20"}]},
        ]

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.return_value = _make_response(nvd_resp)
            result = _fetch_nvd(_CVE)

        assert "CWE-917" in result["cwes"]
        assert "CWE-20" in result["cwes"]

    def test_deduplicates_cwes(self):
        from manus_agent.tools.generate_cve_report import _fetch_nvd

        nvd_resp = json.loads(json.dumps(_NVD_RESPONSE))
        nvd_resp["vulnerabilities"][0]["cve"]["weaknesses"] = [
            {"description": [{"lang": "en", "value": "CWE-917"}]},
            {"description": [{"lang": "en", "value": "CWE-917"}]},
        ]

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.return_value = _make_response(nvd_resp)
            result = _fetch_nvd(_CVE)

        assert result["cwes"].count("CWE-917") == 1


class TestFetchEpss:
    def test_success(self):
        from manus_agent.tools.generate_cve_report import _fetch_epss

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.return_value = _make_response(_EPSS_RESPONSE)
            result = _fetch_epss(_CVE)

        assert result["available"] is True
        assert abs(result["epss"] - 0.97565) < 1e-4
        assert abs(result["percentile"] - 0.99993) < 1e-4
        assert result["date"] == "2024-01-01"

    def test_empty_data(self):
        from manus_agent.tools.generate_cve_report import _fetch_epss

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.return_value = _make_response({"data": []})
            result = _fetch_epss(_CVE)

        assert result["available"] is False

    def test_network_error(self):
        import requests as _requests

        from manus_agent.tools.generate_cve_report import _fetch_epss

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.side_effect = _requests.Timeout("timed out")
            result = _fetch_epss(_CVE)

        assert result["available"] is False

    def test_uppercase_cve(self):
        from manus_agent.tools.generate_cve_report import _fetch_epss

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.return_value = _make_response(_EPSS_RESPONSE)
            _fetch_epss(_CVE_LOWER)
            call_params = mock_get.call_args
            assert "CVE-2021-44228" in str(call_params)


class TestFetchCisaKev:
    def test_found_in_kev(self, tmp_path: Path):
        from manus_agent.tools import generate_cve_report as mod

        with (
            patch.object(mod, "_CISA_CACHE_FILE", tmp_path / "kev_cache.json"),
            patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get,
        ):
            mock_get.return_value = _make_response(_KEV_RESPONSE)
            result = mod._fetch_cisa_kev(_CVE)

        assert result["available"] is True
        assert result["exploited"] is True
        assert result["vendor_project"] == "Apache"
        assert result["date_added"] == "2021-12-10"

    def test_not_in_kev(self, tmp_path: Path):
        from manus_agent.tools import generate_cve_report as mod

        kev_without_cve = {"vulnerabilities": []}

        with (
            patch.object(mod, "_CISA_CACHE_FILE", tmp_path / "kev_cache.json"),
            patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get,
        ):
            mock_get.return_value = _make_response(kev_without_cve)
            result = mod._fetch_cisa_kev(_CVE)

        assert result["available"] is True
        assert result["exploited"] is False

    def test_uses_cache(self, tmp_path: Path):
        from manus_agent.tools import generate_cve_report as mod

        cache_path = tmp_path / "kev_cache.json"
        cache_path.write_text(json.dumps({"timestamp": time.time(), "data": _KEV_RESPONSE}))

        with (
            patch.object(mod, "_CISA_CACHE_FILE", cache_path),
            patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get,
        ):
            result = mod._fetch_cisa_kev(_CVE)
            # Should NOT call requests.get because cache is fresh
            mock_get.assert_not_called()

        assert result["exploited"] is True

    def test_expired_cache_refetches(self, tmp_path: Path):
        from manus_agent.tools import generate_cve_report as mod

        cache_path = tmp_path / "kev_cache.json"
        cache_path.write_text(
            json.dumps({"timestamp": time.time() - 7200, "data": _KEV_RESPONSE})  # 2 hours ago
        )

        with (
            patch.object(mod, "_CISA_CACHE_FILE", cache_path),
            patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get,
        ):
            mock_get.return_value = _make_response(_KEV_RESPONSE)
            result = mod._fetch_cisa_kev(_CVE)
            mock_get.assert_called_once()

        assert result["exploited"] is True

    def test_network_error_returns_unavailable(self, tmp_path: Path):
        import requests as _requests

        from manus_agent.tools import generate_cve_report as mod

        with (
            patch.object(mod, "_CISA_CACHE_FILE", tmp_path / "kev_cache.json"),
            patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get,
        ):
            mock_get.side_effect = _requests.ConnectionError("offline")
            result = mod._fetch_cisa_kev(_CVE)

        assert result["available"] is False

    def test_corrupted_cache_falls_back_to_network(self, tmp_path: Path):
        from manus_agent.tools import generate_cve_report as mod

        cache_path = tmp_path / "kev_cache.json"
        cache_path.write_text("not valid json {{{")

        with (
            patch.object(mod, "_CISA_CACHE_FILE", cache_path),
            patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get,
        ):
            mock_get.return_value = _make_response(_KEV_RESPONSE)
            result = mod._fetch_cisa_kev(_CVE)
            mock_get.assert_called_once()

        assert result["exploited"] is True


class TestFetchOsv:
    def test_success(self):
        from manus_agent.tools.generate_cve_report import _fetch_osv

        with patch("manus_agent.tools.generate_cve_report.requests.post") as mock_post:
            mock_post.return_value = _make_response(_OSV_RESPONSE)
            result = _fetch_osv(_CVE)

        assert result["available"] is True
        pkgs = result["packages"]
        assert len(pkgs) == 1
        assert pkgs[0]["name"] == "log4j-core"
        assert pkgs[0]["ecosystem"] == "Maven"
        assert ">=2.0-beta9" in pkgs[0]["version_ranges"][0]

    def test_empty_response(self):
        from manus_agent.tools.generate_cve_report import _fetch_osv

        with patch("manus_agent.tools.generate_cve_report.requests.post") as mock_post:
            mock_post.return_value = _make_response({"vulns": []})
            result = _fetch_osv(_CVE)

        assert result["available"] is True
        assert result["packages"] == []

    def test_network_error(self):
        import requests as _requests

        from manus_agent.tools.generate_cve_report import _fetch_osv

        with patch("manus_agent.tools.generate_cve_report.requests.post") as mock_post:
            mock_post.side_effect = _requests.Timeout("timed out")
            result = _fetch_osv(_CVE)

        assert result["available"] is False

    def test_deduplicates_packages(self):
        from manus_agent.tools.generate_cve_report import _fetch_osv

        osv_dup = {
            "vulns": [
                {
                    "id": "GHSA-aaa",
                    "affected": [
                        {
                            "package": {"name": "log4j-core", "ecosystem": "Maven"},
                            "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "2.0"}, {"fixed": "2.17.1"}]}],
                        }
                    ],
                },
                {
                    "id": "GHSA-bbb",
                    "affected": [
                        {
                            "package": {"name": "log4j-core", "ecosystem": "Maven"},
                            "ranges": [],
                        }
                    ],
                },
            ]
        }

        with patch("manus_agent.tools.generate_cve_report.requests.post") as mock_post:
            mock_post.return_value = _make_response(osv_dup)
            result = _fetch_osv(_CVE)

        pkg_names = [p["name"] for p in result["packages"]]
        assert pkg_names.count("log4j-core") == 1

    def test_version_range_unfixed(self):
        from manus_agent.tools.generate_cve_report import _fetch_osv

        osv_unfixed = {
            "vulns": [
                {
                    "id": "GHSA-zzz",
                    "affected": [
                        {
                            "package": {"name": "vuln-pkg", "ecosystem": "PyPI"},
                            "ranges": [
                                {
                                    "type": "ECOSYSTEM",
                                    "events": [{"introduced": "1.0"}],
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        with patch("manus_agent.tools.generate_cve_report.requests.post") as mock_post:
            mock_post.return_value = _make_response(osv_unfixed)
            result = _fetch_osv(_CVE)

        assert any("unfixed" in r for r in result["packages"][0]["version_ranges"])


class TestFetchGhsa:
    def test_success(self):
        from manus_agent.tools.generate_cve_report import _fetch_ghsa

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.return_value = _make_response(_GHSA_RESPONSE)
            result = _fetch_ghsa(_CVE)

        assert result["available"] is True
        advs = result["advisories"]
        assert len(advs) == 1
        assert advs[0]["ghsa_id"] == "GHSA-jfh8-c2jp-hdp9"
        assert advs[0]["severity"] == "critical"
        assert advs[0]["cvss_score"] == 10.0
        assert advs[0]["packages"][0]["first_patched"] == "2.17.1"

    def test_empty_advisories(self):
        from manus_agent.tools.generate_cve_report import _fetch_ghsa

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.return_value = _make_response([])
            result = _fetch_ghsa(_CVE)

        assert result["available"] is True
        assert result["advisories"] == []

    def test_network_error(self):
        import requests as _requests

        from manus_agent.tools.generate_cve_report import _fetch_ghsa

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.side_effect = _requests.ConnectionError("down")
            result = _fetch_ghsa(_CVE)

        assert result["available"] is False

    def test_uses_github_token(self, monkeypatch: pytest.MonkeyPatch):
        from manus_agent.tools.generate_cve_report import _fetch_ghsa

        monkeypatch.setenv("GITHUB_TOKEN", "ghp_testtoken123")

        with patch("manus_agent.tools.generate_cve_report.requests.get") as mock_get:
            mock_get.return_value = _make_response(_GHSA_RESPONSE)
            _fetch_ghsa(_CVE)
            call_kwargs = mock_get.call_args.kwargs
            headers = call_kwargs.get("headers", {})
            assert "Authorization" in headers
            assert "ghp_testtoken123" in headers["Authorization"]


# ---------------------------------------------------------------------------
# Unit tests — build_report and render_markdown
# ---------------------------------------------------------------------------


class TestBuildReport:
    def _all_sources_ok(self) -> dict:
        return {
            "nvd": {
                "available": True,
                "description": "Test vuln description.",
                "cvss_score": 10.0,
                "cvss_severity": "CRITICAL",
                "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                "cvss_version": "3.1",
                "cwes": ["CWE-917"],
                "references": ["https://nvd.nist.gov/vuln/detail/CVE-2021-44228"],
                "published": "2021-12-10T10:15:00.000",
                "last_modified": "2023-09-14T11:15:00.000",
                "cpe_products": [],
                "vuln_status": "Analyzed",
            },
            "epss": {"available": True, "epss": 0.97565, "percentile": 0.99993, "date": "2024-01-01"},
            "kev": {
                "available": True,
                "exploited": True,
                "vendor_project": "Apache",
                "product": "Log4j",
                "vulnerability_name": "Apache Log4j2 RCE",
                "date_added": "2021-12-10",
                "short_description": "RCE in log4j",
                "required_action": "Apply updates.",
                "due_date": "2021-12-24",
                "notes": "",
            },
            "osv": {
                "available": True,
                "packages": [
                    {
                        "name": "log4j-core",
                        "ecosystem": "Maven",
                        "version_ranges": [">=2.0-beta9, <2.17.1"],
                        "osv_id": "GHSA-jfh8-c2jp-hdp9",
                    }
                ],
            },
            "ghsa": {
                "available": True,
                "advisories": [
                    {
                        "ghsa_id": "GHSA-jfh8-c2jp-hdp9",
                        "summary": "RCE in Log4j 2",
                        "severity": "critical",
                        "published_at": "2021-12-10T00:00:00Z",
                        "updated_at": "2022-01-01T00:00:00Z",
                        "html_url": "https://github.com/advisories/GHSA-jfh8-c2jp-hdp9",
                        "packages": [
                            {
                                "name": "log4j-core",
                                "ecosystem": "Maven",
                                "vulnerable_version_range": ">= 2.0-beta9, < 2.17.1",
                                "patched_versions": ">= 2.17.1",
                                "first_patched": "2.17.1",
                            }
                        ],
                        "cwe_ids": ["CWE-917"],
                        "cvss_score": 10.0,
                        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                    }
                ],
            },
        }

    def test_basic_fields(self):
        from manus_agent.tools.generate_cve_report import _build_report

        s = self._all_sources_ok()
        report = _build_report(_CVE, **s)

        assert report["cve_id"] == _CVE
        assert report["cvss_score"] == 10.0
        assert report["cvss_severity"] == "CRITICAL"
        assert report["epss_score"] == pytest.approx(0.97565, abs=1e-4)
        assert report["exploited_in_wild"] is True
        assert report["kev_date_added"] == "2021-12-10"
        assert report["published"] == "2021-12-10"

    def test_packages_merged_from_osv_and_ghsa(self):
        from manus_agent.tools.generate_cve_report import _build_report

        s = self._all_sources_ok()
        report = _build_report(_CVE, **s)

        pkg_names = [p["name"] for p in report["affected_packages"]]
        # One from OSV, one from GHSA (same logical package but different source entries)
        assert "log4j-core" in pkg_names
        assert len(pkg_names) >= 1

    def test_sources_dict(self):
        from manus_agent.tools.generate_cve_report import _build_report

        s = self._all_sources_ok()
        report = _build_report(_CVE, **s)

        assert report["sources"]["nvd"] is True
        assert report["sources"]["epss"] is True
        assert report["sources"]["cisa_kev"] is True
        assert report["sources"]["osv"] is True
        assert report["sources"]["ghsa"] is True

    def test_unavailable_sources_handled(self):
        from manus_agent.tools.generate_cve_report import _build_report

        report = _build_report(
            _CVE,
            nvd={"available": False},
            epss={"available": False},
            kev={"available": False},
            osv={"available": False},
            ghsa={"available": False},
        )

        assert report["cve_id"] == _CVE
        assert report["cvss_score"] is None
        assert report["epss_score"] is None
        assert report["exploited_in_wild"] is False
        assert report["affected_packages"] == []
        assert report["sources"]["nvd"] is False

    def test_references_merged(self):
        from manus_agent.tools.generate_cve_report import _build_report

        s = self._all_sources_ok()
        report = _build_report(_CVE, **s)

        refs = report["references"]
        assert any("nvd.nist.gov" in r for r in refs)
        assert any("github.com/advisories" in r for r in refs)


class TestRenderMarkdown:
    def _make_report(self, **overrides) -> dict:
        base = {
            "cve_id": _CVE,
            "published": "2021-12-10",
            "cvss_severity": "CRITICAL",
            "cvss_score": 10.0,
            "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
            "cvss_version": "3.1",
            "description": "Apache Log4j2 JNDI remote code execution.",
            "cwes": ["CWE-917"],
            "epss_score": 0.97565,
            "epss_percentile": 0.99993,
            "exploited_in_wild": True,
            "kev_date_added": "2021-12-10",
            "kev_required_action": "Apply updates per vendor instructions.",
            "kev_due_date": "2021-12-24",
            "kev_vulnerability_name": "Apache Log4j2 RCE",
            "affected_packages": [
                {
                    "source": "OSV",
                    "name": "log4j-core",
                    "ecosystem": "Maven",
                    "version_ranges": [">=2.0-beta9, <2.17.1"],
                    "patched_versions": "",
                    "first_patched": "2.17.1",
                }
            ],
            "references": [
                "https://logging.apache.org/log4j/2.x/security.html",
                "https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
            ],
            "sources": {
                "nvd": True,
                "epss": True,
                "cisa_kev": True,
                "osv": True,
                "ghsa": True,
            },
        }
        base.update(overrides)
        return base

    def test_contains_cve_id(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report())
        assert _CVE in md

    def test_contains_all_sections(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report())
        for section in (
            "## Summary",
            "## Technical Details",
            "## Affected Packages",
            "## Exploitation Status",
            "## Recommendations",
            "## References",
        ):
            assert section in md, f"Missing section: {section!r}"

    def test_kev_warning_shown(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report(exploited_in_wild=True))
        assert "ACTIVELY EXPLOITED" in md or "KEV" in md

    def test_kev_not_in_catalog(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report(exploited_in_wild=False))
        assert "No Known Active Exploitation" in md

    def test_affected_packages_table(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report())
        assert "log4j-core" in md
        assert "Maven" in md
        assert "2.17.1" in md

    def test_no_packages_fallback(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report(affected_packages=[]))
        assert "No affected package records" in md

    def test_epss_section(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report())
        assert "EPSS" in md
        assert "97.6" in md or "0.9756" in md  # Score appears in some form

    def test_no_cvss_no_score_row(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report(cvss_score=None, cvss_severity=""))
        assert "N/A" in md  # Falls back gracefully

    def test_references_listed(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report())
        assert "logging.apache.org" in md

    def test_data_sources_section(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report())
        assert "Data Sources" in md
        assert "NVD" in md
        assert "EPSS" in md

    def test_p0_recommendation(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report(cvss_score=10.0))
        assert "P0" in md

    def test_p1_recommendation(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report(cvss_score=8.0, exploited_in_wild=False))
        assert "P1" in md

    def test_p2_recommendation(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report(cvss_score=5.0, exploited_in_wild=False))
        assert "P2" in md

    def test_p3_recommendation(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report(cvss_score=2.0, exploited_in_wild=False))
        assert "P3" in md

    def test_no_description_fallback(self):
        from manus_agent.tools.generate_cve_report import _render_markdown

        md = _render_markdown(self._make_report(description=""))
        assert "No description available" in md


# ---------------------------------------------------------------------------
# Unit tests — severity helpers
# ---------------------------------------------------------------------------


class TestSeverityHelpers:
    @pytest.mark.parametrize(
        "score,severity,expected_emoji",
        [
            (10.0, "CRITICAL", "🔴"),
            (8.5, "HIGH", "🟠"),
            (5.0, "MEDIUM", "🟡"),
            (2.0, "LOW", "🟢"),
        ],
    )
    def test_severity_badge(self, score, severity, expected_emoji):
        from manus_agent.tools.generate_cve_report import _severity_badge

        badge = _severity_badge(score, severity)
        assert expected_emoji in badge
        assert str(score) in badge

    def test_severity_badge_unknown(self):
        from manus_agent.tools.generate_cve_report import _severity_badge

        badge = _severity_badge(None, "")
        assert "Unknown" in badge

    @pytest.mark.parametrize(
        "epss,pct,expected_tier",
        [
            (0.95, 0.999, "Very High"),
            (0.60, 0.95, "High"),
            (0.30, 0.80, "Medium"),
            (0.10, 0.50, "Low"),
            (0.01, 0.20, "Very Low"),
        ],
    )
    def test_epss_label(self, epss, pct, expected_tier):
        from manus_agent.tools.generate_cve_report import _epss_label

        label = _epss_label(epss, pct)
        assert expected_tier in label


# ---------------------------------------------------------------------------
# Integration-level tests — generate_cve_report @tool (all mocked)
# ---------------------------------------------------------------------------


class TestGenerateCveReport:
    def _mock_all(self, tmp_path: Path):
        """Return a context-manager-compatible patch that mocks all HTTP."""
        import manus_agent.tools.generate_cve_report as mod

        def side_effect_get(url, *args, **kwargs):
            if "nvd.nist.gov" in url:
                return _make_response(_NVD_RESPONSE)
            if "first.org" in url:
                return _make_response(_EPSS_RESPONSE)
            if "cisa.gov" in url:
                return _make_response(_KEV_RESPONSE)
            if "api.github.com" in url:
                return _make_response(_GHSA_RESPONSE)
            return _make_response({})

        def side_effect_post(url, *args, **kwargs):
            if "osv.dev" in url:
                return _make_response(_OSV_RESPONSE)
            return _make_response({})

        get_patch = patch.object(mod.requests, "get", side_effect=side_effect_get)
        post_patch = patch.object(mod.requests, "post", side_effect=side_effect_post)
        cache_patch = patch.object(mod, "_CISA_CACHE_FILE", tmp_path / "kev_cache.json")
        return get_patch, post_patch, cache_patch

    def test_full_success(self, tmp_path: Path):
        from manus_agent.tools.generate_cve_report import generate_cve_report

        get_p, post_p, cache_p = self._mock_all(tmp_path)
        with get_p, post_p, cache_p:
            result = generate_cve_report(_CVE)

        assert "error" not in result
        assert result["cve_id"] == _CVE
        assert "markdown" in result
        assert "report" in result
        md = result["markdown"]
        assert "# CVE Report: CVE-2021-44228" in md

    def test_invalid_cve_id_returns_error(self):
        from manus_agent.tools.generate_cve_report import generate_cve_report

        result = generate_cve_report("NOTACVE")
        assert "error" in result
        assert "NOTACVE" in result["error"]

    def test_cve_normalised_to_uppercase(self, tmp_path: Path):
        from manus_agent.tools.generate_cve_report import generate_cve_report

        get_p, post_p, cache_p = self._mock_all(tmp_path)
        with get_p, post_p, cache_p:
            result = generate_cve_report(_CVE_LOWER)

        assert result["cve_id"] == _CVE

    def test_source_unavailability_graceful(self, tmp_path: Path):
        """All sources returning errors → report still generated, no crash."""
        import requests as _requests

        from manus_agent.tools import generate_cve_report as mod

        with (
            patch.object(mod, "_CISA_CACHE_FILE", tmp_path / "kev_cache.json"),
            patch.object(mod.requests, "get", side_effect=_requests.ConnectionError("offline")),
            patch.object(mod.requests, "post", side_effect=_requests.ConnectionError("offline")),
        ):
            result = mod.generate_cve_report(_CVE)

        assert "error" not in result
        assert result["cve_id"] == _CVE
        assert "markdown" in result
        # All sources failed → sources dict should show False
        assert result["report"]["sources"]["nvd"] is False
        assert result["report"]["sources"]["epss"] is False

    def test_partial_failure_still_renders(self, tmp_path: Path):
        """NVD fails but EPSS succeeds → report renders from available data."""
        import requests as _requests

        from manus_agent.tools import generate_cve_report as mod

        def side_effect_get(url, *args, **kwargs):
            if "nvd.nist.gov" in url:
                raise _requests.Timeout("NVD timeout")
            if "first.org" in url:
                return _make_response(_EPSS_RESPONSE)
            if "cisa.gov" in url:
                return _make_response({"vulnerabilities": []})
            if "api.github.com" in url:
                return _make_response([])
            return _make_response({})

        with (
            patch.object(mod, "_CISA_CACHE_FILE", tmp_path / "kev_cache.json"),
            patch.object(mod.requests, "get", side_effect=side_effect_get),
            patch.object(mod.requests, "post", return_value=_make_response({"vulns": []})),
        ):
            result = mod.generate_cve_report(_CVE)

        assert "error" not in result
        assert result["report"]["sources"]["nvd"] is False
        assert result["report"]["sources"]["epss"] is True
        assert result["report"]["epss_score"] == pytest.approx(0.97565, abs=1e-4)

    def test_markdown_has_required_sections(self, tmp_path: Path):
        from manus_agent.tools.generate_cve_report import generate_cve_report

        get_p, post_p, cache_p = self._mock_all(tmp_path)
        with get_p, post_p, cache_p:
            result = generate_cve_report(_CVE)

        md = result["markdown"]
        required = [
            "## Summary",
            "## Technical Details",
            "## Affected Packages",
            "## Exploitation Status",
            "## Recommendations",
            "## References",
            "## Data Sources",
        ]
        for section in required:
            assert section in md, f"Missing section: {section!r}"

    def test_report_dict_has_required_keys(self, tmp_path: Path):
        from manus_agent.tools.generate_cve_report import generate_cve_report

        get_p, post_p, cache_p = self._mock_all(tmp_path)
        with get_p, post_p, cache_p:
            result = generate_cve_report(_CVE)

        rpt = result["report"]
        for key in (
            "cve_id",
            "cvss_score",
            "epss_score",
            "exploited_in_wild",
            "affected_packages",
            "references",
            "sources",
        ):
            assert key in rpt, f"Missing key: {key!r}"


# ---------------------------------------------------------------------------
# CLI tests — manus-agent report subcommand
# ---------------------------------------------------------------------------


class TestReportCli:
    """Tests for the `manus-agent report` CLI subcommand."""

    def _invoke(self, argv: list[str], monkeypatch: pytest.MonkeyPatch, mock_result: dict | None = None):
        """Run _run_report(argv) with mocked generate_cve_report and capture stdout."""
        import io

        from manus_agent.cli import _run_report

        if mock_result is None:
            mock_result = {
                "cve_id": _CVE,
                "markdown": "# CVE Report: CVE-2021-44228\n\n## Summary\nTest description.\n",
                "report": {
                    "cve_id": _CVE,
                    "cvss_score": 10.0,
                    "exploited_in_wild": True,
                    "sources": {"nvd": True, "epss": True, "cisa_kev": True, "osv": True, "ghsa": True},
                },
            }

        captured = io.StringIO()
        with (
            patch("manus_agent.cli._run_report.__module__", "manus_agent.cli"),
            patch("manus_agent.tools.generate_cve_report.generate_cve_report", return_value=mock_result),
            patch("sys.stdout", captured),
        ):
            exit_code = _run_report(argv)

        return exit_code, captured.getvalue()

    def test_invalid_cve_exits_1(self):
        from manus_agent.cli import _run_report

        with pytest.raises(SystemExit) as exc_info:
            _run_report(["NOTACVE"])
        assert exc_info.value.code != 0

    def test_markdown_output(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        import io

        from manus_agent.cli import _run_report

        mock_result = {
            "cve_id": _CVE,
            "markdown": "# CVE Report: CVE-2021-44228\nContent here.\n",
            "report": {},
        }

        with (
            patch("manus_agent.tools.generate_cve_report.generate_cve_report", return_value=mock_result),
        ):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                code = _run_report([_CVE])

        assert code == 0
        assert "CVE-2021-44228" in captured.getvalue()

    def test_json_output(self, tmp_path: Path):
        import io
        import json as _json

        from manus_agent.cli import _run_report

        mock_result = {
            "cve_id": _CVE,
            "markdown": "# Report",
            "report": {"cvss_score": 10.0},
        }

        with (
            patch("manus_agent.tools.generate_cve_report.generate_cve_report", return_value=mock_result),
        ):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                code = _run_report([_CVE, "--output", "json"])

        assert code == 0
        parsed = _json.loads(captured.getvalue())
        assert parsed["cve_id"] == _CVE

    def test_save_to_file(self, tmp_path: Path):
        import io

        from manus_agent.cli import _run_report

        out_file = tmp_path / "report.md"
        mock_result = {
            "cve_id": _CVE,
            "markdown": "# CVE Report",
            "report": {},
        }

        with (
            patch("manus_agent.tools.generate_cve_report.generate_cve_report", return_value=mock_result),
        ):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                code = _run_report([_CVE, "--save", str(out_file)])

        assert code == 0
        assert out_file.exists()
        assert "CVE Report" in out_file.read_text()

    def test_error_from_tool_exits_1(self, tmp_path: Path):

        from manus_agent.cli import _run_report

        error_result = {"error": "Invalid CVE ID", "cve_id": "BAD"}

        with (
            patch("manus_agent.tools.generate_cve_report.generate_cve_report", return_value=error_result),
        ):
            code = _run_report([_CVE])

        assert code == 1

    def test_subcommand_in_subcommands_set(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "report" in _SUBCOMMANDS

    def test_help_exits_cleanly(self):
        from manus_agent.cli import _build_report_parser

        parser = _build_report_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
