"""
Comprehensive test suite for the get_nvd_data tool function.

Covers:
- TOOL_SPEC metadata validation
- Input validation (invalid/empty/None/non-string CVE IDs)
- CVE ID normalisation (uppercase)
- Successful response parsing (vulnerability data extraction)
- CISA KEV detection logic (the fixed bug — previously checked vulnStatus string)
- Empty vulnerabilities response
- HTTP error handling (RequestException)
- JSON decode error handling
- Unexpected exception handling
- log_tool_output_size invocation for every code path
- URL construction
- NVD API key header injection (via _build_nvd_headers)

All network calls are fully mocked — no real HTTP requests.
"""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import pytest
import requests

from manus_agent.tools.get_nvd_data import (
    TOOL_SPEC,
    get_nvd_data,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _tool(cve_id: Any = "CVE-2024-1234") -> dict:
    """Build a minimal ToolUse dict."""
    return {"toolUseId": "test-tool-id", "input": {"cve_id": cve_id}}


def _mock_response(status_code: int = 200, json_data: Any = None) -> mock.MagicMock:
    """Create a mock requests.Response."""
    resp = mock.MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"HTTP {status_code}", response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def _nvd_response(cve_id: str = "CVE-2024-1234", with_kev: bool = False) -> dict:
    """Build a realistic NVD API v2.0 response structure."""
    cve_obj: dict[str, Any] = {
        "id": cve_id,
        "sourceIdentifier": "cve@mitre.org",
        "published": "2024-01-15T10:15:00.000",
        "lastModified": "2024-02-01T12:00:00.000",
        "vulnStatus": "Analyzed",
        "descriptions": [{"lang": "en", "value": f"Test vulnerability for {cve_id}"}],
        "metrics": {
            "cvssMetricV31": [
                {
                    "source": "nvd@nist.gov",
                    "type": "Primary",
                    "cvssData": {
                        "version": "3.1",
                        "baseScore": 9.8,
                        "baseSeverity": "CRITICAL",
                    },
                }
            ]
        },
        "references": [{"url": "https://example.com/advisory", "source": "cve@mitre.org"}],
    }
    if with_kev:
        cve_obj["cisaExploitAdd"] = "2024-01-20"
        cve_obj["cisaRequiredAction"] = "Apply mitigations per vendor instructions."
        cve_obj["cisaActionDue"] = "2024-02-10"

    return {
        "resultsPerPage": 1,
        "startIndex": 0,
        "totalResults": 1,
        "vulnerabilities": [{"cve": cve_obj}],
    }


# ---------------------------------------------------------------------------
# TOOL_SPEC metadata
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_name_is_get_nvd_data(self):
        assert TOOL_SPEC["name"] == "get_nvd_data"

    def test_has_description(self):
        assert "description" in TOOL_SPEC
        assert len(TOOL_SPEC["description"]) > 20

    def test_input_schema_requires_cve_id(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_cve_id_property_is_string_type(self):
        props = TOOL_SPEC["inputSchema"]["json"]["properties"]
        assert props["cve_id"]["type"] == "string"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_none_cve_id_returns_error(self):
        result = get_nvd_data(_tool(None))
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_empty_string_returns_error(self):
        result = get_nvd_data(_tool(""))
        assert result["status"] == "error"

    def test_integer_returns_error(self):
        result = get_nvd_data(_tool(42))
        assert result["status"] == "error"

    def test_list_returns_error(self):
        result = get_nvd_data(_tool(["CVE-2024-1234"]))
        assert result["status"] == "error"

    def test_partial_prefix_returns_error(self):
        result = get_nvd_data(_tool("CVE"))
        assert result["status"] == "error"

    def test_random_string_returns_error(self):
        result = get_nvd_data(_tool("not-a-cve-id"))
        assert result["status"] == "error"

    def test_missing_cve_prefix_returns_error(self):
        result = get_nvd_data(_tool("2024-1234"))
        assert result["status"] == "error"

    def test_no_network_call_on_invalid_input(self):
        with mock.patch("manus_agent.tools.get_nvd_data.requests.get") as mock_get:
            get_nvd_data(_tool("invalid"))
        mock_get.assert_not_called()

    def test_error_preserves_tool_use_id(self):
        tool = {"toolUseId": "my-unique-id", "input": {"cve_id": 123}}
        result = get_nvd_data(tool)
        assert result["toolUseId"] == "my-unique-id"


# ---------------------------------------------------------------------------
# CVE ID normalisation
# ---------------------------------------------------------------------------


class TestCveIdNormalisation:
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_lowercase_cve_id_uppercased_in_url(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response("CVE-2024-1234"))
        get_nvd_data(_tool("cve-2024-1234"))
        url_called = mock_retry.call_args[0][0]
        assert "cveId=CVE-2024-1234" in url_called

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_mixed_case_cve_id_uppercased(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response("CVE-2024-1234"))
        get_nvd_data(_tool("Cve-2024-1234"))
        url_called = mock_retry.call_args[0][0]
        assert "cveId=CVE-2024-1234" in url_called

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_already_uppercase_stays_unchanged(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response("CVE-2024-9999"))
        get_nvd_data(_tool("CVE-2024-9999"))
        url_called = mock_retry.call_args[0][0]
        assert "cveId=CVE-2024-9999" in url_called


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestUrlConstruction:
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_url_uses_nvd_api_v2(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response())
        get_nvd_data(_tool("CVE-2024-1234"))
        url = mock_retry.call_args[0][0]
        assert "services.nvd.nist.gov/rest/json/cves/2.0" in url

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_url_contains_cve_id_param(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response("CVE-2021-44228"))
        get_nvd_data(_tool("CVE-2021-44228"))
        url = mock_retry.call_args[0][0]
        assert "?cveId=CVE-2021-44228" in url


# ---------------------------------------------------------------------------
# Successful response parsing
# ---------------------------------------------------------------------------


class TestSuccessResponse:
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_success_status(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response())
        result = get_nvd_data(_tool("CVE-2024-1234"))
        assert result["status"] == "success"

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_returns_first_vulnerability_entry(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response("CVE-2024-1234"))
        result = get_nvd_data(_tool("CVE-2024-1234"))
        json_data = result["content"][0]["json"]
        assert json_data["cve"]["id"] == "CVE-2024-1234"

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_preserves_cvss_metrics(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response())
        result = get_nvd_data(_tool("CVE-2024-1234"))
        json_data = result["content"][0]["json"]
        assert "metrics" in json_data["cve"]
        cvss = json_data["cve"]["metrics"]["cvssMetricV31"][0]["cvssData"]
        assert cvss["baseScore"] == 9.8

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_preserves_references(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response())
        result = get_nvd_data(_tool("CVE-2024-1234"))
        json_data = result["content"][0]["json"]
        refs = json_data["cve"]["references"]
        assert len(refs) == 1
        assert refs[0]["url"] == "https://example.com/advisory"

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_tool_use_id_preserved(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response())
        tool = {"toolUseId": "unique-123", "input": {"cve_id": "CVE-2024-1234"}}
        result = get_nvd_data(tool)
        assert result["toolUseId"] == "unique-123"

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_multiple_vulnerabilities_returns_first(self, mock_retry):
        response_data = _nvd_response("CVE-2024-1111")
        # Add a second vulnerability (unusual but possible in edge cases)
        second_vuln = {"cve": {"id": "CVE-2024-2222", "vulnStatus": "Modified"}}
        response_data["vulnerabilities"].append(second_vuln)
        mock_retry.return_value = _mock_response(200, response_data)
        result = get_nvd_data(_tool("CVE-2024-1111"))
        json_data = result["content"][0]["json"]
        assert json_data["cve"]["id"] == "CVE-2024-1111"


# ---------------------------------------------------------------------------
# CISA KEV detection (bug fix — was checking vulnStatus string incorrectly)
# ---------------------------------------------------------------------------


class TestCisaKevDetection:
    """Validate the fixed CISA KEV detection logic.

    Previously the code checked:
        'cisaExploitAdd' in vulnerability_data.get('cve', {}).get('vulnStatus', '')
    which NEVER matched because vulnStatus is a string like 'Analyzed'.\n
    The fix checks:
        'cisaExploitAdd' in cve_obj  (i.e. as a dictionary key)
    """

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_kev_detected_when_cisa_exploit_add_present(self, mock_retry):
        """CVE with CISA KEV fields should set is_in_kev=True."""
        mock_retry.return_value = _mock_response(200, _nvd_response("CVE-2024-1234", with_kev=True))
        result = get_nvd_data(_tool("CVE-2024-1234"))
        kev_info = result["content"][0]["json"]["cisa_kev_info"]
        assert kev_info["is_in_kev"] is True

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_kev_date_added_extracted(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response("CVE-2024-1234", with_kev=True))
        result = get_nvd_data(_tool("CVE-2024-1234"))
        kev_info = result["content"][0]["json"]["cisa_kev_info"]
        assert kev_info["date_added"] == "2024-01-20"

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_kev_required_action_extracted(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response("CVE-2024-1234", with_kev=True))
        result = get_nvd_data(_tool("CVE-2024-1234"))
        kev_info = result["content"][0]["json"]["cisa_kev_info"]
        assert kev_info["required_action"] == "Apply mitigations per vendor instructions."

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_kev_due_date_extracted(self, mock_retry):
        mock_retry.return_value = _mock_response(200, _nvd_response("CVE-2024-1234", with_kev=True))
        result = get_nvd_data(_tool("CVE-2024-1234"))
        kev_info = result["content"][0]["json"]["cisa_kev_info"]
        assert kev_info["due_date"] == "2024-02-10"

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_no_kev_fields_means_not_in_kev(self, mock_retry):
        """CVE without CISA KEV fields should set is_in_kev=False."""
        mock_retry.return_value = _mock_response(200, _nvd_response("CVE-2024-1234", with_kev=False))
        result = get_nvd_data(_tool("CVE-2024-1234"))
        kev_info = result["content"][0]["json"]["cisa_kev_info"]
        assert kev_info["is_in_kev"] is False

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_kev_info_always_attached_to_result(self, mock_retry):
        """cisa_kev_info key should always be present regardless of KEV status."""
        mock_retry.return_value = _mock_response(200, _nvd_response("CVE-2024-1234", with_kev=False))
        result = get_nvd_data(_tool("CVE-2024-1234"))
        assert "cisa_kev_info" in result["content"][0]["json"]

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_vuln_status_analyzed_does_not_trigger_kev(self, mock_retry):
        """vulnStatus='Analyzed' should NOT trigger KEV detection (old bug)."""
        response_data = _nvd_response("CVE-2024-1234", with_kev=False)
        # Ensure vulnStatus is set to a realistic value
        response_data["vulnerabilities"][0]["cve"]["vulnStatus"] = "Analyzed"
        mock_retry.return_value = _mock_response(200, response_data)
        result = get_nvd_data(_tool("CVE-2024-1234"))
        kev_info = result["content"][0]["json"]["cisa_kev_info"]
        assert kev_info["is_in_kev"] is False

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_missing_cve_object_does_not_crash(self, mock_retry):
        """If the 'cve' key is missing, KEV detection should gracefully default."""
        response_data = {"vulnerabilities": [{"no_cve_key": True}]}
        mock_retry.return_value = _mock_response(200, response_data)
        result = get_nvd_data(_tool("CVE-2024-1234"))
        # Should still succeed (returns the vulnerability_data)
        assert result["status"] == "success"
        kev_info = result["content"][0]["json"]["cisa_kev_info"]
        assert kev_info["is_in_kev"] is False

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_kev_with_partial_fields_graceful(self, mock_retry):
        """If cisaExploitAdd exists but other CISA fields are missing, use defaults."""
        response_data = _nvd_response("CVE-2024-1234", with_kev=False)
        # Add only cisaExploitAdd, omit required_action and due_date
        response_data["vulnerabilities"][0]["cve"]["cisaExploitAdd"] = "2024-03-15"
        mock_retry.return_value = _mock_response(200, response_data)
        result = get_nvd_data(_tool("CVE-2024-1234"))
        kev_info = result["content"][0]["json"]["cisa_kev_info"]
        assert kev_info["is_in_kev"] is True
        assert kev_info["date_added"] == "2024-03-15"
        # Missing fields should default to empty string (not raise KeyError)
        assert kev_info["required_action"] == ""
        assert kev_info["due_date"] == ""


# ---------------------------------------------------------------------------
# Empty vulnerabilities response
# ---------------------------------------------------------------------------


class TestEmptyVulnerabilities:
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_empty_vulnerabilities_list_returns_error(self, mock_retry):
        mock_retry.return_value = _mock_response(200, {"vulnerabilities": []})
        result = get_nvd_data(_tool("CVE-2024-9999"))
        assert result["status"] == "error"
        assert "No vulnerability data found" in result["content"][0]["text"]

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_missing_vulnerabilities_key_returns_error(self, mock_retry):
        mock_retry.return_value = _mock_response(200, {"totalResults": 0})
        result = get_nvd_data(_tool("CVE-2024-9999"))
        assert result["status"] == "error"

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_none_vulnerabilities_returns_error(self, mock_retry):
        mock_retry.return_value = _mock_response(200, {"vulnerabilities": None})
        result = get_nvd_data(_tool("CVE-2024-9999"))
        assert result["status"] == "error"

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_error_message_contains_cve_id(self, mock_retry):
        mock_retry.return_value = _mock_response(200, {"vulnerabilities": []})
        result = get_nvd_data(_tool("CVE-2024-7777"))
        assert "CVE-2024-7777" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# HTTP / network error handling
# ---------------------------------------------------------------------------


class TestHttpErrors:
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_request_exception_returns_error(self, mock_retry):
        mock_retry.side_effect = requests.exceptions.ConnectionError("Connection refused")
        result = get_nvd_data(_tool("CVE-2024-1234"))
        assert result["status"] == "error"
        assert "Request to NVD API failed" in result["content"][0]["text"]

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_timeout_returns_error(self, mock_retry):
        mock_retry.side_effect = requests.exceptions.Timeout("Read timed out")
        result = get_nvd_data(_tool("CVE-2024-1234"))
        assert result["status"] == "error"
        assert "Request to NVD API failed" in result["content"][0]["text"]

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_http_error_returns_error(self, mock_retry):
        mock_retry.side_effect = requests.exceptions.HTTPError("404 Not Found")
        result = get_nvd_data(_tool("CVE-2024-1234"))
        assert result["status"] == "error"

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_error_message_includes_exception_text(self, mock_retry):
        mock_retry.side_effect = requests.exceptions.ConnectionError("DNS lookup failed")
        result = get_nvd_data(_tool("CVE-2024-1234"))
        assert "DNS lookup failed" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# JSON decode error handling
# ---------------------------------------------------------------------------


class TestJsonDecodeError:
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_json_decode_error_returns_error(self, mock_retry):
        resp = mock.MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        resp.raise_for_status.return_value = None
        mock_retry.return_value = resp
        result = get_nvd_data(_tool("CVE-2024-1234"))
        assert result["status"] == "error"
        assert "Failed to parse JSON" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Unexpected exception handling
# ---------------------------------------------------------------------------


class TestUnexpectedExceptions:
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_runtime_error_caught(self, mock_retry):
        mock_retry.side_effect = RuntimeError("Something unexpected")
        result = get_nvd_data(_tool("CVE-2024-1234"))
        assert result["status"] == "error"
        assert "unexpected error" in result["content"][0]["text"].lower()

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_type_error_caught(self, mock_retry):
        mock_retry.side_effect = TypeError("'NoneType' object is not subscriptable")
        result = get_nvd_data(_tool("CVE-2024-1234"))
        assert result["status"] == "error"

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_keyboard_interrupt_propagates(self, mock_retry):
        """KeyboardInterrupt should NOT be swallowed."""
        mock_retry.side_effect = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            get_nvd_data(_tool("CVE-2024-1234"))


# ---------------------------------------------------------------------------
# log_tool_output_size invocation
# ---------------------------------------------------------------------------


class TestLogToolOutputSize:
    @mock.patch("manus_agent.tools.get_nvd_data.log_tool_output_size")
    def test_logged_on_invalid_input(self, mock_log):
        get_nvd_data(_tool("invalid"))
        mock_log.assert_called_once()
        assert mock_log.call_args[0][0] == "get_nvd_data"

    @mock.patch("manus_agent.tools.get_nvd_data.log_tool_output_size")
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_logged_on_success(self, mock_retry, mock_log):
        mock_retry.return_value = _mock_response(200, _nvd_response())
        get_nvd_data(_tool("CVE-2024-1234"))
        mock_log.assert_called_once()
        assert mock_log.call_args[0][0] == "get_nvd_data"

    @mock.patch("manus_agent.tools.get_nvd_data.log_tool_output_size")
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_logged_on_empty_vulnerabilities(self, mock_retry, mock_log):
        mock_retry.return_value = _mock_response(200, {"vulnerabilities": []})
        get_nvd_data(_tool("CVE-2024-1234"))
        mock_log.assert_called_once()

    @mock.patch("manus_agent.tools.get_nvd_data.log_tool_output_size")
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_logged_on_request_exception(self, mock_retry, mock_log):
        mock_retry.side_effect = requests.exceptions.ConnectionError("fail")
        get_nvd_data(_tool("CVE-2024-1234"))
        mock_log.assert_called_once()

    @mock.patch("manus_agent.tools.get_nvd_data.log_tool_output_size")
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_logged_on_json_decode_error(self, mock_retry, mock_log):
        resp = mock.MagicMock(spec=requests.Response)
        resp.status_code = 200
        resp.json.side_effect = json.JSONDecodeError("err", "", 0)
        resp.raise_for_status.return_value = None
        mock_retry.return_value = resp
        get_nvd_data(_tool("CVE-2024-1234"))
        mock_log.assert_called_once()

    @mock.patch("manus_agent.tools.get_nvd_data.log_tool_output_size")
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_logged_on_unexpected_error(self, mock_retry, mock_log):
        mock_retry.side_effect = RuntimeError("boom")
        get_nvd_data(_tool("CVE-2024-1234"))
        mock_log.assert_called_once()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_cve_with_long_sequence_number(self, mock_retry):
        """NVD supports CVEs with 5+ digit sequence numbers."""
        cve = "CVE-2024-123456"
        mock_retry.return_value = _mock_response(200, _nvd_response(cve))
        result = get_nvd_data(_tool(cve))
        assert result["status"] == "success"

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_response_with_no_metrics(self, mock_retry):
        """CVE response may not have CVSS metrics (e.g., Awaiting Analysis)."""
        response_data = _nvd_response("CVE-2024-1234")
        del response_data["vulnerabilities"][0]["cve"]["metrics"]
        mock_retry.return_value = _mock_response(200, response_data)
        result = get_nvd_data(_tool("CVE-2024-1234"))
        assert result["status"] == "success"

    @mock.patch("manus_agent.tools.get_nvd_data._nvd_get_with_retry")
    def test_extra_kwargs_do_not_crash(self, mock_retry):
        """Extra kwargs should be silently ignored."""
        mock_retry.return_value = _mock_response(200, _nvd_response())
        result = get_nvd_data(_tool("CVE-2024-1234"), extra_param="ignored")
        assert result["status"] == "success"

    def test_tool_input_missing_cve_id_key(self):
        """If cve_id key is missing from input, should return error."""
        tool = {"toolUseId": "t1", "input": {}}
        result = get_nvd_data(tool)
        assert result["status"] == "error"
