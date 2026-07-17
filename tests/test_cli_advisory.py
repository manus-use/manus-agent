"""Tests for the `manus-agent advisory` CLI subcommand."""

import json
from unittest.mock import patch

from manus_agent.cli import _run_advisory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_ADVISORY = {
    "found": True,
    "ghsa_id": "GHSA-abcd-1234-efgh",
    "cve_id": "CVE-2024-3094",
    "severity": "critical",
    "summary": "XZ Utils backdoor allowing remote code execution",
    "description": "A malicious backdoor was discovered in XZ Utils versions 5.6.0 and 5.6.1.",
    "published_at": "2024-03-29T00:00:00Z",
    "updated_at": "2024-04-01T12:00:00Z",
    "withdrawn_at": None,
    "html_url": "https://github.com/advisories/GHSA-abcd-1234-efgh",
    "cvss": {
        "score": 10.0,
        "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
    },
    "cwes": [
        {"cwe_id": "CWE-506", "name": "Embedded Malicious Code"},
    ],
    "vulnerabilities": [
        {
            "package": {"ecosystem": "deb", "name": "xz-utils"},
            "vulnerable_version_range": ">= 5.6.0, <= 5.6.1",
            "patched_versions": None,
            "first_patched_version": {"identifier": "5.6.1+really5.4.5-1"},
        },
    ],
    "references": [
        {"url": "https://nvd.nist.gov/vuln/detail/CVE-2024-3094"},
        {"url": "https://www.openwall.com/lists/oss-security/2024/03/29/4"},
    ],
    "credits": [
        {"login": "AndresFreundTec", "type": "reporter"},
    ],
}

NOT_FOUND_RESULT = {
    "found": False,
    "message": "No advisory found on GitHub for CVE-2099-99999.",
}

ERROR_RESULT = {
    "error": "HTTP error occurred while querying GitHub Advisory API: 500 Server Error",
}


# ---------------------------------------------------------------------------
# Text output tests
# ---------------------------------------------------------------------------


class TestAdvisoryTextOutput:
    """Test advisory subcommand text output."""

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_found_prints_header(self, mock_fetch, capsys):
        mock_fetch.return_value = SAMPLE_ADVISORY
        rc = _run_advisory(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "GHSA-abcd-1234-efgh" in out
        assert "CVE-2024-3094" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_found_prints_severity(self, mock_fetch, capsys):
        mock_fetch.return_value = SAMPLE_ADVISORY
        _run_advisory(["CVE-2024-3094"])
        out = capsys.readouterr().out
        assert "CRITICAL" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_found_prints_cvss(self, mock_fetch, capsys):
        mock_fetch.return_value = SAMPLE_ADVISORY
        _run_advisory(["CVE-2024-3094"])
        out = capsys.readouterr().out
        assert "10.0" in out
        assert "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_found_prints_cwes(self, mock_fetch, capsys):
        mock_fetch.return_value = SAMPLE_ADVISORY
        _run_advisory(["CVE-2024-3094"])
        out = capsys.readouterr().out
        assert "CWE-506" in out
        assert "Embedded Malicious Code" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_found_prints_affected_packages(self, mock_fetch, capsys):
        mock_fetch.return_value = SAMPLE_ADVISORY
        _run_advisory(["CVE-2024-3094"])
        out = capsys.readouterr().out
        assert "deb/xz-utils" in out
        assert ">= 5.6.0, <= 5.6.1" in out
        assert "5.6.1+really5.4.5-1" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_found_prints_references(self, mock_fetch, capsys):
        mock_fetch.return_value = SAMPLE_ADVISORY
        _run_advisory(["CVE-2024-3094"])
        out = capsys.readouterr().out
        assert "nvd.nist.gov" in out
        assert "openwall.com" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_found_prints_credits(self, mock_fetch, capsys):
        mock_fetch.return_value = SAMPLE_ADVISORY
        _run_advisory(["CVE-2024-3094"])
        out = capsys.readouterr().out
        assert "AndresFreundTec" in out
        assert "reporter" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_found_prints_dates(self, mock_fetch, capsys):
        mock_fetch.return_value = SAMPLE_ADVISORY
        _run_advisory(["CVE-2024-3094"])
        out = capsys.readouterr().out
        assert "2024-03-29" in out
        assert "2024-04-01" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_found_prints_url(self, mock_fetch, capsys):
        mock_fetch.return_value = SAMPLE_ADVISORY
        _run_advisory(["CVE-2024-3094"])
        out = capsys.readouterr().out
        assert "https://github.com/advisories/GHSA-abcd-1234-efgh" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_not_found(self, mock_fetch, capsys):
        mock_fetch.return_value = NOT_FOUND_RESULT
        rc = _run_advisory(["CVE-2099-99999"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No advisory found" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_error(self, mock_fetch, capsys):
        mock_fetch.return_value = ERROR_RESULT
        rc = _run_advisory(["CVE-2024-3094"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "HTTP error" in err


# ---------------------------------------------------------------------------
# JSON output tests
# ---------------------------------------------------------------------------


class TestAdvisoryJsonOutput:
    """Test advisory subcommand JSON output."""

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_json_output_valid(self, mock_fetch, capsys):
        mock_fetch.return_value = SAMPLE_ADVISORY
        rc = _run_advisory(["CVE-2024-3094", "--output", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["ghsa_id"] == "GHSA-abcd-1234-efgh"
        assert data["found"] is True

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_json_output_not_found(self, mock_fetch, capsys):
        mock_fetch.return_value = NOT_FOUND_RESULT
        rc = _run_advisory(["CVE-2099-99999", "--output", "json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["found"] is False

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_json_output_error(self, mock_fetch, capsys):
        mock_fetch.return_value = ERROR_RESULT
        rc = _run_advisory(["CVE-2024-3094", "--output", "json"])
        # JSON output always returns 0 for structured output
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert "error" in data


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAdvisoryEdgeCases:
    """Test edge cases and unusual payloads."""

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_no_cvss(self, mock_fetch, capsys):
        data = {**SAMPLE_ADVISORY, "cvss": None}
        mock_fetch.return_value = data
        rc = _run_advisory(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "GHSA-abcd-1234-efgh" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_empty_vulnerabilities(self, mock_fetch, capsys):
        data = {**SAMPLE_ADVISORY, "vulnerabilities": []}
        mock_fetch.return_value = data
        rc = _run_advisory(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "GHSA-abcd-1234-efgh" in out
        assert "Affected Packages" not in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_no_cwes(self, mock_fetch, capsys):
        data = {**SAMPLE_ADVISORY, "cwes": []}
        mock_fetch.return_value = data
        rc = _run_advisory(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CWEs" not in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_withdrawn(self, mock_fetch, capsys):
        data = {**SAMPLE_ADVISORY, "withdrawn_at": "2024-05-01T00:00:00Z"}
        mock_fetch.return_value = data
        rc = _run_advisory(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Withdrawn" in out
        assert "2024-05-01" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_patched_versions_string(self, mock_fetch, capsys):
        """Test when patched_versions is a string instead of a dict."""
        data = {
            **SAMPLE_ADVISORY,
            "vulnerabilities": [
                {
                    "package": {"ecosystem": "npm", "name": "express"},
                    "vulnerable_version_range": "< 4.19.2",
                    "patched_versions": "4.19.2",
                    "first_patched_version": None,
                },
            ],
        }
        mock_fetch.return_value = data
        rc = _run_advisory(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "npm/express" in out
        assert "4.19.2" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_no_patched_version(self, mock_fetch, capsys):
        """Test when no patched version is available."""
        data = {
            **SAMPLE_ADVISORY,
            "vulnerabilities": [
                {
                    "package": {"ecosystem": "pip", "name": "some-pkg"},
                    "vulnerable_version_range": "< 2.0",
                    "patched_versions": None,
                    "first_patched_version": None,
                },
            ],
        }
        mock_fetch.return_value = data
        rc = _run_advisory(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "no patch available" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_long_description_truncated(self, mock_fetch, capsys):
        """Test that very long descriptions are truncated in text mode."""
        long_desc = "\n".join([f"Line {i}: detail" for i in range(50)])
        data = {**SAMPLE_ADVISORY, "description": long_desc}
        mock_fetch.return_value = data
        rc = _run_advisory(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "truncated" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_many_references_truncated(self, mock_fetch, capsys):
        """References beyond 10 show a count."""
        refs = [{"url": f"https://example.com/{i}"} for i in range(15)]
        data = {**SAMPLE_ADVISORY, "references": refs}
        mock_fetch.return_value = data
        rc = _run_advisory(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "and 5 more" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_references_as_strings(self, mock_fetch, capsys):
        """Some API responses return references as plain strings."""
        data = {**SAMPLE_ADVISORY, "references": ["https://example.com/ref1"]}
        mock_fetch.return_value = data
        rc = _run_advisory(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "https://example.com/ref1" in out

    @patch("manus_agent.tools.get_github_advisory.fetch_github_advisory")
    def test_advisory_credits_as_strings(self, mock_fetch, capsys):
        """Credits may be plain strings."""
        data = {**SAMPLE_ADVISORY, "credits": ["researcher1", "researcher2"]}
        mock_fetch.return_value = data
        rc = _run_advisory(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "researcher1" in out


# ---------------------------------------------------------------------------
# Library function tests (fetch_github_advisory)
# ---------------------------------------------------------------------------


class TestFetchGithubAdvisory:
    """Test the reusable fetch_github_advisory library function."""

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_success_returns_advisory(self, mock_config, mock_get):
        from manus_agent.tools.get_github_advisory import fetch_github_advisory

        mock_config.side_effect = Exception("no config")
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{"ghsa_id": "GHSA-test", "cve_id": "CVE-2024-0001"}]

        result = fetch_github_advisory("CVE-2024-0001")
        assert result["found"] is True
        assert result["ghsa_id"] == "GHSA-test"

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_empty_list_returns_not_found(self, mock_config, mock_get):
        from manus_agent.tools.get_github_advisory import fetch_github_advisory

        mock_config.side_effect = Exception("no config")
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = []

        result = fetch_github_advisory("CVE-2024-0001")
        assert result["found"] is False

    def test_invalid_cve_id(self):
        from manus_agent.tools.get_github_advisory import fetch_github_advisory

        result = fetch_github_advisory("not-a-cve")
        assert "error" in result
        assert "Invalid CVE ID" in result["error"]

    def test_empty_cve_id(self):
        from manus_agent.tools.get_github_advisory import fetch_github_advisory

        result = fetch_github_advisory("")
        assert "error" in result

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_uses_provided_token(self, mock_config, mock_get):
        from manus_agent.tools.get_github_advisory import fetch_github_advisory

        mock_config.side_effect = Exception("no config")
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{"ghsa_id": "GHSA-test"}]

        fetch_github_advisory("CVE-2024-0001", github_token="ghp_test123")
        call_kwargs = mock_get.call_args
        assert "token ghp_test123" in call_kwargs.kwargs.get("headers", {}).get("Authorization", "")

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_http_error_404(self, mock_config, mock_get):
        import requests as req

        from manus_agent.tools.get_github_advisory import fetch_github_advisory

        mock_config.side_effect = Exception("no config")
        mock_response = mock_get.return_value
        mock_response.status_code = 404
        http_err = req.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_err

        result = fetch_github_advisory("CVE-2024-0001")
        assert result["found"] is False

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_http_error_500(self, mock_config, mock_get):
        import requests as req

        from manus_agent.tools.get_github_advisory import fetch_github_advisory

        mock_config.side_effect = Exception("no config")
        mock_response = mock_get.return_value
        mock_response.status_code = 500
        http_err = req.exceptions.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_err

        result = fetch_github_advisory("CVE-2024-0001")
        assert "error" in result

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_connection_error(self, mock_config, mock_get):
        import requests as req

        from manus_agent.tools.get_github_advisory import fetch_github_advisory

        mock_config.side_effect = Exception("no config")
        mock_get.side_effect = req.exceptions.ConnectionError("Network unreachable")

        result = fetch_github_advisory("CVE-2024-0001")
        assert "error" in result
        assert "Network unreachable" in result["error"]

    @patch("manus_agent.tools.get_github_advisory.requests.get")
    @patch("manus_agent.tools.get_github_advisory.Config.from_file")
    def test_malformed_json(self, mock_config, mock_get):
        from manus_agent.tools.get_github_advisory import fetch_github_advisory

        mock_config.side_effect = Exception("no config")
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = ValueError("bad json")

        result = fetch_github_advisory("CVE-2024-0001")
        assert "error" in result
        assert "unexpected response format" in result["error"]
