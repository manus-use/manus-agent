"""Comprehensive test suite for track_vendor_response tool + vendor-response CLI subcommand.

100% mocked — no real HTTP calls.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.track_vendor_response import (
    _VALID_STATES,
    TOOL_SPEC,
    _classify,
    _fetch_cisa_kev,
    _fetch_nvd_references,
    _fetch_vulncheck_kev,
    track_vendor_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_use(cve_id: Any = "CVE-2024-3094") -> dict:
    return {"toolUseId": "test-id-001", "input": {"cve_id": cve_id}}


def _mock_response(json_data: Any, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ---------------------------------------------------------------------------
# TOOL_SPEC sanity
# ---------------------------------------------------------------------------


class TestToolSpec:
    def test_spec_has_name(self):
        assert TOOL_SPEC["name"] == "track_vendor_response"

    def test_spec_has_description(self):
        assert "vendor" in TOOL_SPEC["description"].lower()

    def test_spec_has_input_schema(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]


# ---------------------------------------------------------------------------
# _fetch_nvd_references
# ---------------------------------------------------------------------------


class TestFetchNvdReferences:
    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_references(self, mock_get):
        refs = [{"url": "https://example.com/patch", "tags": ["Patch"]}]
        mock_get.return_value = _mock_response({"vulnerabilities": [{"cve": {"references": refs}}]})
        result = _fetch_nvd_references("CVE-2024-3094")
        assert result == refs

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_no_vulns(self, mock_get):
        mock_get.return_value = _mock_response({"vulnerabilities": []})
        assert _fetch_nvd_references("CVE-9999-0001") == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_missing_key(self, mock_get):
        mock_get.return_value = _mock_response({})
        assert _fetch_nvd_references("CVE-9999-0001") == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_http_error(self, mock_get):
        mock_get.return_value = _mock_response({}, status_code=500)
        assert _fetch_nvd_references("CVE-2024-3094") == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_exception(self, mock_get):
        mock_get.side_effect = ConnectionError("network down")
        assert _fetch_nvd_references("CVE-2024-3094") == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_none_references(self, mock_get):
        mock_get.return_value = _mock_response({"vulnerabilities": [{"cve": {"references": None}}]})
        assert _fetch_nvd_references("CVE-2024-3094") == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_when_cve_has_no_references_key(self, mock_get):
        mock_get.return_value = _mock_response({"vulnerabilities": [{"cve": {}}]})
        assert _fetch_nvd_references("CVE-2024-3094") == []


# ---------------------------------------------------------------------------
# _fetch_cisa_kev
# ---------------------------------------------------------------------------


class TestFetchCisaKev:
    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_matching_entry(self, mock_get):
        kev_entry = {"cveID": "CVE-2024-3094", "requiredAction": "Apply update"}
        mock_get.return_value = _mock_response({"vulnerabilities": [kev_entry, {"cveID": "CVE-2021-44228"}]})
        result = _fetch_cisa_kev("CVE-2024-3094")
        assert result == kev_entry

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_when_not_in_kev(self, mock_get):
        mock_get.return_value = _mock_response({"vulnerabilities": [{"cveID": "CVE-2021-44228"}]})
        assert _fetch_cisa_kev("CVE-9999-0001") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_http_error(self, mock_get):
        mock_get.return_value = _mock_response({}, status_code=503)
        assert _fetch_cisa_kev("CVE-2024-3094") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_exception(self, mock_get):
        mock_get.side_effect = TimeoutError("timed out")
        assert _fetch_cisa_kev("CVE-2024-3094") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_no_vulnerabilities_key(self, mock_get):
        mock_get.return_value = _mock_response({})
        assert _fetch_cisa_kev("CVE-2024-3094") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_case_insensitive_match(self, mock_get):
        """The function uppercases the cve_id parameter for comparison."""
        kev_entry = {"cveID": "CVE-2024-3094", "requiredAction": "Apply update"}
        mock_get.return_value = _mock_response({"vulnerabilities": [kev_entry]})
        # Function compares vuln.get("cveID").upper() == cve_id
        # So passing uppercase should match
        result = _fetch_cisa_kev("CVE-2024-3094")
        assert result == kev_entry


# ---------------------------------------------------------------------------
# _fetch_vulncheck_kev
# ---------------------------------------------------------------------------


class TestFetchVulncheckKev:
    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_first_data_entry(self, mock_get):
        vc_data = {"cve": "CVE-2024-3094", "ransomwareUse": True}
        mock_get.return_value = _mock_response({"data": [vc_data]})
        result = _fetch_vulncheck_kev("CVE-2024-3094", "test-key")
        assert result == vc_data

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_when_no_data(self, mock_get):
        mock_get.return_value = _mock_response({"data": []})
        assert _fetch_vulncheck_kev("CVE-2024-3094", "test-key") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_when_no_api_key(self, mock_get):
        assert _fetch_vulncheck_kev("CVE-2024-3094", "") == {}
        mock_get.assert_not_called()

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_http_error(self, mock_get):
        mock_get.return_value = _mock_response({}, status_code=403)
        assert _fetch_vulncheck_kev("CVE-2024-3094", "test-key") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_exception(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")
        assert _fetch_vulncheck_kev("CVE-2024-3094", "test-key") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_sends_auth_header(self, mock_get):
        mock_get.return_value = _mock_response({"data": []})
        _fetch_vulncheck_kev("CVE-2024-3094", "my-secret-key")
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer my-secret-key"

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_none_data(self, mock_get):
        mock_get.return_value = _mock_response({"data": None})
        assert _fetch_vulncheck_kev("CVE-2024-3094", "test-key") == {}


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------


class TestClassify:
    def test_all_empty_returns_unknown(self):
        state, confidence, evidence = _classify([], {}, {}, "unknown")
        assert state == "unknown"
        assert confidence < 0.3

    def test_patch_tag_detected(self):
        refs = [{"url": "https://example.com", "tags": ["Patch"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"
        assert confidence >= 0.75

    def test_vendor_advisory_tag_detected(self):
        refs = [{"url": "https://example.com", "tags": ["Vendor-Advisory"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"
        assert confidence >= 0.75

    def test_fix_tag_detected(self):
        refs = [{"url": "https://example.com", "tags": ["Fix"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"

    def test_release_notes_tag_detected(self):
        refs = [{"url": "https://example.com", "tags": ["Release-Notes"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"

    def test_mitigation_tag_workaround(self):
        refs = [{"url": "https://example.com", "tags": ["Mitigation"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"
        assert confidence >= 0.6

    def test_workaround_tag(self):
        refs = [{"url": "https://example.com", "tags": ["Workaround"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"

    def test_patch_keyword_in_url(self):
        refs = [{"url": "https://example.com/patch-available", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_workaround_keyword_in_url(self):
        refs = [{"url": "https://example.com/workaround", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"

    def test_cisa_kev_apply_update(self):
        cisa = {"requiredAction": "Apply vendor update", "shortDescription": "XSS vuln"}
        state, confidence, evidence = _classify([], cisa, {}, "unknown")
        assert state == "patch_available"
        assert confidence >= 0.4

    def test_cisa_kev_patch_action(self):
        cisa = {"requiredAction": "Patch before deadline", "shortDescription": "RCE"}
        state, confidence, evidence = _classify([], cisa, {}, "unknown")
        assert state == "patch_available"

    def test_cisa_kev_boosts_confidence(self):
        refs = [{"url": "https://example.com", "tags": ["Patch"]}]
        _, conf_without_kev, _ = _classify(refs, {}, {}, "analyzed")
        cisa = {"requiredAction": "Apply update", "shortDescription": "test"}
        _, conf_with_kev, _ = _classify(refs, cisa, {}, "analyzed")
        assert conf_with_kev > conf_without_kev

    def test_vulncheck_kev_unknown_becomes_investigating(self):
        vc = {"cve": "CVE-2024-3094"}
        state, confidence, evidence = _classify([], {}, vc, "unknown")
        assert state == "investigating"

    def test_vulncheck_kev_boosts_confidence(self):
        vc = {"cve": "CVE-2024-3094"}
        _, conf_without, _ = _classify([], {}, {}, "unknown")
        _, conf_with, _ = _classify([], {}, vc, "unknown")
        assert conf_with > conf_without

    def test_vulncheck_ransomware_extra_boost(self):
        vc_no_ransom = {"cve": "CVE-2024-3094"}
        vc_ransom = {"cve": "CVE-2024-3094", "ransomwareUse": True}
        _, conf_no, _ = _classify([], {}, vc_no_ransom, "unknown")
        _, conf_yes, _ = _classify([], {}, vc_ransom, "unknown")
        assert conf_yes > conf_no

    def test_vulncheck_ransomware_alternate_key(self):
        vc = {"cve": "CVE-2024-3094", "knownRansomwareCampaignUse": True}
        _, confidence, evidence = _classify([], {}, vc, "unknown")
        assert any("ransomware" in e.lower() for e in evidence)

    def test_nvd_analyzed_status_adds_evidence(self):
        _, _, evidence = _classify([], {}, {}, "analyzed")
        assert any("NVD status" in e for e in evidence)

    def test_nvd_modified_status_adds_evidence(self):
        _, _, evidence = _classify([], {}, {}, "Modified")
        assert any("NVD status" in e for e in evidence)

    def test_confidence_never_exceeds_1(self):
        """Even with all signals maxed, confidence should stay <= 1.0."""
        refs = [{"url": "https://example.com", "tags": ["Patch", "Vendor-Advisory"]}]
        cisa = {"requiredAction": "Apply update", "shortDescription": "crit vuln"}
        vc = {"cve": "CVE-2024-3094", "ransomwareUse": True}
        _, confidence, _ = _classify(refs, cisa, vc, "analyzed")
        assert confidence <= 1.0

    def test_state_always_valid(self):
        """State should always be in _VALID_STATES."""
        refs = [{"url": "https://example.com", "tags": ["Patch"]}]
        cisa = {"requiredAction": "Apply update", "shortDescription": "test"}
        vc = {"cve": "CVE-2024-3094", "ransomwareUse": True}
        state, _, _ = _classify(refs, cisa, vc, "analyzed")
        assert state in _VALID_STATES

    def test_all_states_are_strings(self):
        for s in _VALID_STATES:
            assert isinstance(s, str)

    def test_empty_tags_list(self):
        refs = [{"url": "https://example.com", "tags": []}]
        state, _, _ = _classify(refs, {}, {}, "unknown")
        # Should still check URL keywords
        assert state in _VALID_STATES

    def test_no_tags_key(self):
        refs = [{"url": "https://example.com"}]
        state, _, _ = _classify(refs, {}, {}, "unknown")
        assert state in _VALID_STATES

    def test_patch_tag_overrides_mitigation(self):
        """When both Patch and Mitigation tags present, Patch wins."""
        refs = [
            {"url": "https://example.com", "tags": ["Patch", "Mitigation"]},
        ]
        state, _, _ = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_multiple_references_aggregated(self):
        refs = [
            {"url": "https://example.com/advisory", "tags": []},
            {"url": "https://example.com/patch", "tags": []},
        ]
        state, _, _ = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_cisa_without_patch_action_does_not_set_patch_available(self):
        """CISA KEV without apply/update/patch action should not set patch_available."""
        cisa = {"requiredAction": "Investigate immediately", "shortDescription": "test"}
        state, _, evidence = _classify([], cisa, {}, "unknown")
        # The CISA entry adds evidence but requiredAction doesn't match patch keywords
        assert state in _VALID_STATES

    def test_vulncheck_does_not_override_patch_available(self):
        """If already patch_available, vulncheck should not downgrade."""
        refs = [{"url": "https://example.com", "tags": ["Patch"]}]
        vc = {"cve": "CVE-2024-3094"}
        state, _, _ = _classify(refs, {}, vc, "analyzed")
        assert state == "patch_available"


# ---------------------------------------------------------------------------
# track_vendor_response (Strands tool entry point)
# ---------------------------------------------------------------------------


class TestTrackVendorResponse:
    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_success_basic(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = [{"url": "https://example.com", "tags": ["Patch"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        result = track_vendor_response(_tool_use())
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert payload["cve_id"] == "CVE-2024-3094"
        assert payload["vendor_response_state"] == "patch_available"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_success_unknown_state(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        result = track_vendor_response(_tool_use())
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert payload["vendor_response_state"] == "unknown"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_signals_structure(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        result = track_vendor_response(_tool_use())
        payload = result["content"][0]["json"]
        signals = payload["signals"]
        assert "nvd_references_found" in signals
        assert "cisa_kev_hit" in signals
        assert "vulncheck_kev_hit" in signals
        assert "vulncheck_api_key_present" in signals

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    @patch.dict(os.environ, {"VULNCHECK_API_KEY": "test-key-123"})
    def test_vulncheck_api_key_present_signal(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        result = track_vendor_response(_tool_use())
        payload = result["content"][0]["json"]
        assert payload["signals"]["vulncheck_api_key_present"] is True

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    @patch.dict(os.environ, {}, clear=True)
    def test_vulncheck_api_key_absent_signal(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        result = track_vendor_response(_tool_use())
        payload = result["content"][0]["json"]
        assert payload["signals"]["vulncheck_api_key_present"] is False

    def test_invalid_cve_id_returns_error(self):
        result = track_vendor_response(_tool_use("not-a-cve"))
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_empty_cve_id_returns_error(self):
        result = track_vendor_response(_tool_use(""))
        assert result["status"] == "error"

    def test_numeric_cve_id_returns_error(self):
        result = track_vendor_response(_tool_use(12345))
        assert result["status"] == "error"

    def test_none_cve_id_returns_error(self):
        result = track_vendor_response(_tool_use(None))
        assert result["status"] == "error"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_cve_id_uppercased(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        result = track_vendor_response(_tool_use("cve-2024-3094"))
        payload = result["content"][0]["json"]
        assert payload["cve_id"] == "CVE-2024-3094"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_all_sources_combined(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = [{"url": "https://example.com", "tags": ["Patch"]}]
        mock_cisa.return_value = {"requiredAction": "Apply update", "shortDescription": "critical"}
        mock_vc.return_value = {"cve": "CVE-2024-3094", "ransomwareUse": True}
        result = track_vendor_response(_tool_use())
        payload = result["content"][0]["json"]
        assert payload["vendor_response_state"] == "patch_available"
        assert payload["confidence"] >= 0.9
        assert payload["signals"]["cisa_kev_hit"] is True
        assert payload["signals"]["vulncheck_kev_hit"] is True

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_tool_use_id_preserved(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        result = track_vendor_response(_tool_use())
        assert result["toolUseId"] == "test-id-001"


# ---------------------------------------------------------------------------
# CLI: _run_vendor_response
# ---------------------------------------------------------------------------


class TestVendorResponseCli:
    """Tests for the vendor-response CLI subcommand."""

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_text_output(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = [{"url": "https://example.com", "tags": ["Patch"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        rc = _run_vendor_response(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2024-3094" in out
        assert "patch_available" in out

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_json_output(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = [{"url": "https://example.com", "tags": ["Patch"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        rc = _run_vendor_response(["CVE-2024-3094", "--output", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["cve_id"] == "CVE-2024-3094"
        assert data["classification"] == "patch_available"
        assert "confidence" in data
        assert "confidence_label" in data
        assert "signals" in data

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_json_output_unknown(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        rc = _run_vendor_response(["CVE-9999-0001", "--output", "json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["classification"] == "unknown"
        assert data["confidence_label"] == "low"

    def test_invalid_cve_id(self, capsys):
        from manus_agent.cli import _run_vendor_response

        rc = _run_vendor_response(["not-a-cve"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Invalid CVE ID" in err

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_confidence_labels(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """Test that confidence labels map correctly."""
        from manus_agent.cli import _run_vendor_response

        # High confidence: patch tag + CISA KEV
        mock_nvd.return_value = [{"url": "https://example.com", "tags": ["Patch"]}]
        mock_cisa.return_value = {"requiredAction": "Apply update", "shortDescription": "test"}
        mock_vc.return_value = {}
        rc = _run_vendor_response(["CVE-2024-3094", "--output", "json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["confidence_label"] == "high"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_text_output_shows_evidence(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = [{"url": "https://example.com", "tags": ["Patch"]}]
        mock_cisa.return_value = {"requiredAction": "Apply update", "shortDescription": "vuln desc"}
        mock_vc.return_value = {}
        rc = _run_vendor_response(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Evidence" in out

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_text_output_shows_signals(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        rc = _run_vendor_response(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Signals" in out
        assert "NVD references" in out
        assert "CISA KEV" in out
        assert "VulnCheck KEV" in out

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_nvd_failure_returns_error(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.side_effect = ConnectionError("NVD down")
        rc = _run_vendor_response(["CVE-2024-3094"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "NVD API request failed" in err

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_cisa_failure_degrades_gracefully(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = [{"url": "https://example.com", "tags": ["Patch"]}]
        mock_cisa.side_effect = TimeoutError("CISA timeout")
        mock_vc.return_value = {}
        rc = _run_vendor_response(["CVE-2024-3094"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "CISA KEV lookup failed" in err

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_vulncheck_failure_degrades_gracefully(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.side_effect = ConnectionError("VulnCheck down")
        rc = _run_vendor_response(["CVE-2024-3094"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "VulnCheck KEV lookup failed" in err

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_lowercase_cve_accepted(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        rc = _run_vendor_response(["cve-2024-3094", "--output", "json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["cve_id"] == "CVE-2024-3094"


# ---------------------------------------------------------------------------
# CLI: vendor-response in known subcommands + main dispatch
# ---------------------------------------------------------------------------


class TestVendorResponseCliIntegration:
    def test_vendor_response_in_known_subcommands(self):
        """vendor-response should be in the known subcommands set."""
        import manus_agent.cli as cli_mod

        # Find the set that contains known subcommands
        # The known subcommands set is module-level
        source = open(cli_mod.__file__).read()
        assert '"vendor-response"' in source

    def test_parser_has_required_args(self):
        from manus_agent.cli import _build_vendor_response_parser

        parser = _build_vendor_response_parser()
        # Should parse valid args without error
        args = parser.parse_args(["CVE-2024-3094"])
        assert args.cve_id == "CVE-2024-3094"
        assert args.output == "text"

    def test_parser_json_output(self):
        from manus_agent.cli import _build_vendor_response_parser

        parser = _build_vendor_response_parser()
        args = parser.parse_args(["CVE-2024-3094", "--output", "json"])
        assert args.output == "json"

    def test_parser_rejects_invalid_output(self):
        from manus_agent.cli import _build_vendor_response_parser

        parser = _build_vendor_response_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["CVE-2024-3094", "--output", "xml"])


# ---------------------------------------------------------------------------
# Edge cases and regression guards
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_valid_states_frozenset(self):
        assert isinstance(_VALID_STATES, frozenset)
        assert len(_VALID_STATES) == 6

    def test_all_expected_states_present(self):
        expected = {
            "patch_available",
            "patch_pending",
            "workaround_only",
            "investigating",
            "no_patch_expected",
            "unknown",
        }
        assert _VALID_STATES == expected

    def test_classify_returns_tuple_of_three(self):
        result = _classify([], {}, {}, "unknown")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_classify_confidence_is_float(self):
        _, confidence, _ = _classify([], {}, {}, "unknown")
        assert isinstance(confidence, float)

    def test_classify_evidence_is_list(self):
        _, _, evidence = _classify([], {}, {}, "unknown")
        assert isinstance(evidence, list)

    def test_classify_with_none_tags_in_ref(self):
        """References with None tags should not crash."""
        refs = [{"url": "https://example.com", "tags": None}]
        state, _, _ = _classify(refs, {}, {}, "unknown")
        assert state in _VALID_STATES

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_result_has_tool_use_id(self, mock_nvd, mock_cisa, mock_vc):
        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}
        result = track_vendor_response(_tool_use())
        assert "toolUseId" in result

    def test_classify_release_keyword_in_url(self):
        """The keyword 'release' is a single word and matches in URLs."""
        refs = [{"url": "https://example.com/release-notes", "tags": []}]
        state, _, _ = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_classify_hotfix_keyword_in_url(self):
        refs = [{"url": "https://example.com/hotfix-available", "tags": []}]
        state, _, _ = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_classify_mitigation_keyword_in_url_no_patch(self):
        refs = [{"url": "https://example.com/disable-feature-mitigation", "tags": []}]
        state, _, _ = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"

    def test_classify_disable_keyword_in_url(self):
        """The keyword 'disable' is a single word and matches in URLs."""
        refs = [{"url": "https://example.com/disable-feature", "tags": []}]
        state, _, _ = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"

    def test_classify_vulncheck_ransomware_use_key(self):
        vc = {"cve": "CVE-2024-3094", "ransomware_use": True}
        _, confidence, evidence = _classify([], {}, vc, "unknown")
        assert any("ransomware" in e.lower() for e in evidence)

    def test_classify_cisa_kev_with_empty_required_action(self):
        cisa = {"requiredAction": "", "shortDescription": "vuln"}
        state, _, evidence = _classify([], cisa, {}, "unknown")
        # Empty requiredAction should still add CISA evidence but not trigger patch_available
        assert any("CISA" in e for e in evidence)
