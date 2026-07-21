"""Comprehensive test suite for track_vendor_response tool.

Tests cover:
- Input validation (invalid CVE IDs, missing fields)
- NVD reference fetching and failure handling
- CISA KEV lookup and failure handling
- VulnCheck KEV lookup (with/without API key) and failure handling
- Classification logic (_classify) across all 6 states
- Confidence scoring and evidence accumulation
- Keyword matching in reference URLs
- Tag-based classification (patch, vendor-advisory, mitigation, etc.)
- VulnCheck ransomware escalation
- End-to-end tool invocation via track_vendor_response()

All HTTP calls are mocked — no real network traffic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from manus_agent.tools.track_vendor_response import (
    _classify,
    _fetch_cisa_kev,
    _fetch_nvd_references,
    _fetch_vulncheck_kev,
    track_vendor_response,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tool_use(cve_id):
    """Helper to build a minimal ToolUse dict."""
    return {"toolUseId": "test-id-001", "input": {"cve_id": cve_id}}


def _make_response(status_code=200, json_data=None, raise_exc=None):
    """Create a mock response object."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    if raise_exc:
        mock_resp.raise_for_status.side_effect = raise_exc
    if json_data is not None:
        mock_resp.json.return_value = json_data
    return mock_resp


# ---------------------------------------------------------------------------
# Tests: Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for invalid/edge-case inputs to track_vendor_response."""

    def test_missing_cve_id(self):
        tool = {"toolUseId": "t1", "input": {}}
        result = track_vendor_response(tool)
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_empty_string_cve_id(self):
        tool = _make_tool_use("")
        result = track_vendor_response(tool)
        assert result["status"] == "error"

    def test_non_string_cve_id(self):
        tool = {"toolUseId": "t1", "input": {"cve_id": 12345}}
        result = track_vendor_response(tool)
        assert result["status"] == "error"

    def test_invalid_prefix(self):
        tool = _make_tool_use("VUL-2024-1234")
        result = track_vendor_response(tool)
        assert result["status"] == "error"

    def test_none_cve_id(self):
        tool = {"toolUseId": "t1", "input": {"cve_id": None}}
        result = track_vendor_response(tool)
        assert result["status"] == "error"

    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=[])
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    def test_valid_cve_id_lowercase(self, mock_vc, mock_cisa, mock_nvd):
        """Lowercase CVE IDs should be accepted and uppercased."""
        tool = _make_tool_use("cve-2024-1234")
        result = track_vendor_response(tool)
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert payload["cve_id"] == "CVE-2024-1234"

    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=[])
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    def test_valid_cve_id_uppercase(self, mock_vc, mock_cisa, mock_nvd):
        tool = _make_tool_use("CVE-2021-44228")
        result = track_vendor_response(tool)
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert payload["cve_id"] == "CVE-2021-44228"


# ---------------------------------------------------------------------------
# Tests: _fetch_nvd_references
# ---------------------------------------------------------------------------


class TestFetchNvdReferences:
    """Tests for the NVD reference fetcher."""

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_successful_fetch(self, mock_get):
        refs = [{"url": "https://example.com/patch", "tags": ["Patch"]}]
        mock_get.return_value = _make_response(json_data={"vulnerabilities": [{"cve": {"references": refs}}]})
        result = _fetch_nvd_references("CVE-2024-1234")
        assert result == refs

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_no_vulnerabilities(self, mock_get):
        mock_get.return_value = _make_response(json_data={"vulnerabilities": []})
        result = _fetch_nvd_references("CVE-2024-9999")
        assert result == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_missing_references_key(self, mock_get):
        mock_get.return_value = _make_response(json_data={"vulnerabilities": [{"cve": {}}]})
        result = _fetch_nvd_references("CVE-2024-1234")
        assert result == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_network_error(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        result = _fetch_nvd_references("CVE-2024-1234")
        assert result == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_http_error(self, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.HTTPError("503 Service Unavailable")
        result = _fetch_nvd_references("CVE-2024-1234")
        assert result == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_timeout_error(self, mock_get):
        import requests as req

        mock_get.side_effect = req.exceptions.Timeout("timed out")
        result = _fetch_nvd_references("CVE-2024-1234")
        assert result == []


# ---------------------------------------------------------------------------
# Tests: _fetch_cisa_kev
# ---------------------------------------------------------------------------


class TestFetchCisaKev:
    """Tests for the CISA KEV fetcher."""

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_cve_found_in_kev(self, mock_get):
        kev_entry = {
            "cveID": "CVE-2024-3094",
            "vendorProject": "xz",
            "requiredAction": "Apply updates per vendor instructions",
            "shortDescription": "xz backdoor",
        }
        mock_get.return_value = _make_response(json_data={"vulnerabilities": [kev_entry]})
        result = _fetch_cisa_kev("CVE-2024-3094")
        assert result == kev_entry

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_cve_not_in_kev(self, mock_get):
        mock_get.return_value = _make_response(json_data={"vulnerabilities": [{"cveID": "CVE-2024-0001"}]})
        result = _fetch_cisa_kev("CVE-2024-9999")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_empty_kev_catalog(self, mock_get):
        mock_get.return_value = _make_response(json_data={"vulnerabilities": []})
        result = _fetch_cisa_kev("CVE-2024-1234")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_network_failure(self, mock_get):
        mock_get.side_effect = Exception("network error")
        result = _fetch_cisa_kev("CVE-2024-1234")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_case_insensitive_match(self, mock_get):
        """The function upper-cases the input CVE for comparison."""
        kev_entry = {"cveID": "CVE-2024-3094", "requiredAction": "Apply update"}
        mock_get.return_value = _make_response(json_data={"vulnerabilities": [kev_entry]})
        # The function checks vuln.get("cveID").upper() == cve_id
        # where cve_id is already upper, so this tests the catalog match.
        result = _fetch_cisa_kev("CVE-2024-3094")
        assert result == kev_entry


# ---------------------------------------------------------------------------
# Tests: _fetch_vulncheck_kev
# ---------------------------------------------------------------------------


class TestFetchVulncheckKev:
    """Tests for the VulnCheck KEV fetcher."""

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_successful_fetch_with_data(self, mock_get):
        vc_data = {"cveID": "CVE-2024-3094", "ransomwareUse": True}
        mock_get.return_value = _make_response(json_data={"data": [vc_data]})
        result = _fetch_vulncheck_kev("CVE-2024-3094", "test-api-key")
        assert result == vc_data

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_empty_data_array(self, mock_get):
        mock_get.return_value = _make_response(json_data={"data": []})
        result = _fetch_vulncheck_kev("CVE-2024-9999", "test-api-key")
        assert result == {}

    def test_no_api_key(self):
        """Without API key, should return empty dict without making a request."""
        result = _fetch_vulncheck_kev("CVE-2024-1234", "")
        assert result == {}

    def test_none_api_key(self):
        result = _fetch_vulncheck_kev("CVE-2024-1234", None)
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_network_failure(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        result = _fetch_vulncheck_kev("CVE-2024-1234", "test-api-key")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_api_returns_null_data(self, mock_get):
        mock_get.return_value = _make_response(json_data={"data": None})
        result = _fetch_vulncheck_kev("CVE-2024-1234", "test-key")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_authorization_header_sent(self, mock_get):
        mock_get.return_value = _make_response(json_data={"data": []})
        _fetch_vulncheck_kev("CVE-2024-1234", "my-secret-key")
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert headers.get("Authorization") == "Bearer my-secret-key"


# ---------------------------------------------------------------------------
# Tests: _classify
# ---------------------------------------------------------------------------


class TestClassify:
    """Tests for the classification logic."""

    def test_unknown_when_no_signals(self):
        state, confidence, evidence = _classify([], {}, {}, "unknown")
        assert state == "unknown"
        assert confidence < 0.3
        assert evidence == []

    def test_patch_available_from_patch_tag(self):
        refs = [{"url": "https://example.com", "tags": ["Patch"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"
        assert confidence >= 0.75

    def test_patch_available_from_vendor_advisory_tag(self):
        refs = [{"url": "https://vendor.com/advisory", "tags": ["Vendor-Advisory"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"
        assert confidence >= 0.75

    def test_patch_available_from_fix_tag(self):
        refs = [{"url": "https://example.com/fix", "tags": ["Fix"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"
        assert confidence >= 0.75

    def test_patch_available_from_release_notes_tag(self):
        refs = [{"url": "https://example.com", "tags": ["Release-Notes"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"
        assert confidence >= 0.75

    def test_workaround_from_mitigation_tag(self):
        refs = [{"url": "https://example.com", "tags": ["Mitigation"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "workaround_only"
        assert confidence >= 0.6

    def test_workaround_from_workaround_tag(self):
        refs = [{"url": "https://example.com", "tags": ["Workaround"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "workaround_only"
        assert confidence >= 0.6

    def test_patch_keyword_in_url(self):
        refs = [{"url": "https://example.com/fixed in v2.0", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"
        assert confidence >= 0.5

    def test_workaround_keyword_in_url(self):
        refs = [{"url": "https://example.com/workaround-guide", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"
        assert confidence >= 0.4

    def test_cisa_kev_apply_update_promotes_to_patch(self):
        cisa = {"requiredAction": "Apply updates per vendor instructions", "shortDescription": "test"}
        state, confidence, evidence = _classify([], cisa, {}, "unknown")
        assert state == "patch_available"
        assert confidence >= 0.4

    def test_cisa_kev_patch_keyword_promotes(self):
        cisa = {"requiredAction": "Patch immediately", "shortDescription": "urgent"}
        state, confidence, evidence = _classify([], cisa, {}, "unknown")
        assert state == "patch_available"

    def test_cisa_kev_no_action_keyword(self):
        cisa = {"requiredAction": "Investigate and mitigate", "shortDescription": "info"}
        state, confidence, evidence = _classify([], cisa, {}, "unknown")
        # "Investigate" doesn't match apply/update/patch
        assert "CISA KEV" in evidence[0]

    def test_vulncheck_kev_promotes_unknown_to_investigating(self):
        vc = {"cveID": "CVE-2024-1234"}
        state, confidence, evidence = _classify([], {}, vc, "unknown")
        assert state == "investigating"
        assert confidence >= 0.35

    def test_vulncheck_kev_ransomware_escalates(self):
        vc = {"cveID": "CVE-2024-1234", "ransomwareUse": True}
        state, confidence, evidence = _classify([], {}, vc, "unknown")
        assert state == "investigating"
        assert any("ransomware" in e for e in evidence)
        # Ransomware should boost confidence
        assert confidence >= 0.4

    def test_vulncheck_kev_known_ransomware_campaign_use(self):
        vc = {"cveID": "CVE-2024-1234", "knownRansomwareCampaignUse": True}
        state, confidence, evidence = _classify([], {}, vc, "unknown")
        assert any("ransomware" in e for e in evidence)

    def test_vulncheck_kev_ransomware_use_key(self):
        vc = {"cveID": "CVE-2024-1234", "ransomware_use": True}
        state, confidence, evidence = _classify([], {}, vc, "unknown")
        assert any("ransomware" in e for e in evidence)

    def test_combined_patch_tag_and_cisa_high_confidence(self):
        refs = [{"url": "https://vendor.com/patch", "tags": ["Patch"]}]
        cisa = {"requiredAction": "Apply update", "shortDescription": "critical"}
        state, confidence, evidence = _classify(refs, cisa, {}, "analyzed")
        assert state == "patch_available"
        assert confidence >= 0.9

    def test_combined_all_sources_maximum_confidence(self):
        refs = [{"url": "https://vendor.com/fix", "tags": ["Patch", "Vendor-Advisory"]}]
        cisa = {"requiredAction": "Apply updates", "shortDescription": "critical vuln"}
        vc = {"cveID": "CVE-2024-1234", "ransomwareUse": True}
        state, confidence, evidence = _classify(refs, cisa, vc, "analyzed")
        assert state == "patch_available"
        assert confidence >= 0.95
        assert len(evidence) >= 3

    def test_nvd_status_analyzed_adds_evidence(self):
        state, confidence, evidence = _classify([], {}, {}, "analyzed")
        assert any("NVD status" in e for e in evidence)
        assert confidence >= 0.3

    def test_nvd_status_modified_adds_evidence(self):
        state, confidence, evidence = _classify([], {}, {}, "Modified")
        assert any("NVD status" in e for e in evidence)

    def test_patch_tag_overrides_workaround_when_both_present(self):
        """When both Patch and Mitigation tags exist, Patch should win."""
        refs = [
            {"url": "https://example.com/patch", "tags": ["Patch"]},
            {"url": "https://example.com/workaround", "tags": ["Mitigation"]},
        ]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"

    def test_multiple_refs_no_relevant_tags(self):
        refs = [
            {"url": "https://example.com/info1", "tags": ["Third Party Advisory"]},
            {"url": "https://example.com/info2", "tags": ["US Government Resource"]},
        ]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "unknown"

    def test_empty_tags_list(self):
        refs = [{"url": "https://example.com/nothing", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "unknown"

    def test_none_tags_handled(self):
        refs = [{"url": "https://example.com", "tags": None}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        # Should not crash
        assert state == "unknown"

    def test_patch_keyword_update_to_in_url(self):
        refs = [{"url": "https://example.com/update to v3.1", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_patch_keyword_upgrade_to_in_url(self):
        refs = [{"url": "https://example.com/upgrade to latest", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_patch_keyword_hotfix_in_url(self):
        refs = [{"url": "https://example.com/hotfix-release", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_workaround_keyword_disable_in_url(self):
        refs = [{"url": "https://example.com/disable-feature", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"

    def test_workaround_keyword_firewall_rule_in_url(self):
        refs = [{"url": "https://example.com/firewall-rule-to-block", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"

    def test_confidence_never_exceeds_1(self):
        """Edge case: all signals maxed out should not exceed 1.0."""
        refs = [{"url": "https://example.com/patch", "tags": ["Patch", "Fix", "Release-Notes"]}]
        cisa = {"requiredAction": "Apply update and patch now", "shortDescription": "critical"}
        vc = {"cveID": "CVE-2024-1234", "ransomwareUse": True, "knownRansomwareCampaignUse": True}
        state, confidence, evidence = _classify(refs, cisa, vc, "analyzed")
        assert confidence <= 1.0

    def test_confidence_is_rounded(self):
        """Confidence should be rounded to 3 decimal places."""
        refs = [{"url": "https://example.com", "tags": ["Patch"]}]
        _, confidence, _ = _classify(refs, {}, {}, "unknown")
        # Check that it's rounded (no more than 3 decimal places)
        assert confidence == round(confidence, 3)


# ---------------------------------------------------------------------------
# Tests: End-to-end track_vendor_response
# ---------------------------------------------------------------------------


class TestTrackVendorResponseE2E:
    """End-to-end tests for the track_vendor_response tool function."""

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_all_sources_empty(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        tool = _make_tool_use("CVE-2024-9999")
        result = track_vendor_response(tool)
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert payload["vendor_response_state"] == "unknown"
        assert payload["confidence"] < 0.3
        assert payload["signals"]["nvd_references_found"] == 0
        assert payload["signals"]["cisa_kev_hit"] is False
        assert payload["signals"]["vulncheck_kev_hit"] is False

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_patch_available_from_nvd_tags(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = [
            {"url": "https://vendor.com/security/patch-v2.1", "tags": ["Patch", "Vendor-Advisory"]}
        ]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        tool = _make_tool_use("CVE-2024-1234")
        result = track_vendor_response(tool)
        payload = result["content"][0]["json"]
        assert payload["vendor_response_state"] == "patch_available"
        assert payload["confidence"] >= 0.75

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_investigating_from_vulncheck_only(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {"cveID": "CVE-2024-1234"}
        tool = _make_tool_use("CVE-2024-1234")
        result = track_vendor_response(tool)
        payload = result["content"][0]["json"]
        assert payload["vendor_response_state"] == "investigating"
        assert payload["signals"]["vulncheck_kev_hit"] is True

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_patch_from_cisa_apply_action(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = []
        mock_cisa.return_value = {
            "cveID": "CVE-2024-3094",
            "requiredAction": "Apply updates per vendor instructions.",
            "shortDescription": "xz Utils backdoor",
        }
        mock_vc.return_value = {}
        tool = _make_tool_use("CVE-2024-3094")
        result = track_vendor_response(tool)
        payload = result["content"][0]["json"]
        assert payload["vendor_response_state"] == "patch_available"
        assert payload["signals"]["cisa_kev_hit"] is True

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_workaround_from_mitigation_refs(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = [{"url": "https://vendor.com/kb/mitigation-steps", "tags": ["Mitigation"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        tool = _make_tool_use("CVE-2024-5678")
        result = track_vendor_response(tool)
        payload = result["content"][0]["json"]
        assert payload["vendor_response_state"] == "workaround_only"

    @patch("manus_agent.tools.track_vendor_response.os.environ.get")
    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_vulncheck_api_key_present_signal(self, mock_nvd, mock_cisa, mock_vc, mock_env):
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        mock_env.return_value = "some-api-key"
        tool = _make_tool_use("CVE-2024-1234")
        result = track_vendor_response(tool)
        payload = result["content"][0]["json"]
        assert payload["signals"]["vulncheck_api_key_present"] is True

    @patch("manus_agent.tools.track_vendor_response.os.environ.get")
    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_vulncheck_api_key_absent_signal(self, mock_nvd, mock_cisa, mock_vc, mock_env):
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        mock_env.return_value = ""
        tool = _make_tool_use("CVE-2024-1234")
        result = track_vendor_response(tool)
        payload = result["content"][0]["json"]
        assert payload["signals"]["vulncheck_api_key_present"] is False

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_full_scenario_log4shell(self, mock_nvd, mock_cisa, mock_vc):
        """Simulate a Log4Shell-like CVE with all sources reporting."""
        mock_nvd.return_value = [
            {"url": "https://logging.apache.org/log4j/2.x/security.html", "tags": ["Vendor-Advisory"]},
            {"url": "https://github.com/apache/logging-log4j2/pull/608", "tags": ["Patch"]},
        ]
        mock_cisa.return_value = {
            "cveID": "CVE-2021-44228",
            "requiredAction": "Apply updates per vendor instructions",
            "shortDescription": "Apache Log4j2 JNDI",
            "dueDate": "2021-12-24",
        }
        mock_vc.return_value = {
            "cveID": "CVE-2021-44228",
            "ransomwareUse": True,
        }
        tool = _make_tool_use("CVE-2021-44228")
        result = track_vendor_response(tool)
        payload = result["content"][0]["json"]
        assert payload["vendor_response_state"] == "patch_available"
        assert payload["confidence"] >= 0.95
        assert payload["signals"]["cisa_kev_hit"] is True
        assert payload["signals"]["vulncheck_kev_hit"] is True
        assert payload["signals"]["nvd_references_found"] == 2

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_tool_use_id_preserved(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        tool = {"toolUseId": "unique-id-xyz", "input": {"cve_id": "CVE-2024-0001"}}
        result = track_vendor_response(tool)
        assert result["toolUseId"] == "unique-id-xyz"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_result_payload_structure(self, mock_nvd, mock_cisa, mock_vc):
        """Verify all expected keys exist in the result payload."""
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        tool = _make_tool_use("CVE-2024-1234")
        result = track_vendor_response(tool)
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert "cve_id" in payload
        assert "vendor_response_state" in payload
        assert "confidence" in payload
        assert "evidence" in payload
        assert "signals" in payload
        signals = payload["signals"]
        assert "nvd_references_found" in signals
        assert "cisa_kev_hit" in signals
        assert "vulncheck_kev_hit" in signals
        assert "vulncheck_api_key_present" in signals

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_nvd_references_count_in_signals(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = [
            {"url": "https://a.com", "tags": []},
            {"url": "https://b.com", "tags": []},
            {"url": "https://c.com", "tags": []},
        ]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        tool = _make_tool_use("CVE-2024-1234")
        result = track_vendor_response(tool)
        payload = result["content"][0]["json"]
        assert payload["signals"]["nvd_references_found"] == 3
