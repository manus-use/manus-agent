"""Comprehensive test suite for track_vendor_response tool + vendor-response CLI.

100% mocked — no real HTTP calls.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Tool-level tests
# ---------------------------------------------------------------------------


class TestTrackVendorResponseTool:
    """Tests for the track_vendor_response() tool function."""

    def _make_tool_use(self, cve_id: str) -> dict:
        return {
            "toolUseId": "test-tool-use-id",
            "name": "track_vendor_response",
            "input": {"cve_id": cve_id},
        }

    def test_invalid_cve_id_returns_error(self):
        from manus_agent.tools.track_vendor_response import track_vendor_response

        result = track_vendor_response(self._make_tool_use("not-a-cve"))
        assert result["status"] == "error"
        assert "Invalid CVE ID" in result["content"][0]["text"]

    def test_empty_cve_id_returns_error(self):
        from manus_agent.tools.track_vendor_response import track_vendor_response

        result = track_vendor_response(self._make_tool_use(""))
        assert result["status"] == "error"

    def test_numeric_cve_id_returns_error(self):
        from manus_agent.tools.track_vendor_response import track_vendor_response

        tool = {
            "toolUseId": "test-id",
            "name": "track_vendor_response",
            "input": {"cve_id": 12345},
        }
        result = track_vendor_response(tool)
        assert result["status"] == "error"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_valid_cve_returns_success(self, mock_nvd, mock_cisa, mock_vc):
        from manus_agent.tools.track_vendor_response import track_vendor_response

        mock_nvd.return_value = [{"url": "https://example.com/fix", "tags": ["Patch"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        result = track_vendor_response(self._make_tool_use("CVE-2024-3094"))
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert payload["cve_id"] == "CVE-2024-3094"
        assert "vendor_response_state" in payload
        assert "confidence" in payload

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_lowercase_cve_id_is_uppercased(self, mock_nvd, mock_cisa, mock_vc):
        from manus_agent.tools.track_vendor_response import track_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        result = track_vendor_response(self._make_tool_use("cve-2024-3094"))
        payload = result["content"][0]["json"]
        assert payload["cve_id"] == "CVE-2024-3094"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_all_sources_empty_returns_unknown(self, mock_nvd, mock_cisa, mock_vc):
        from manus_agent.tools.track_vendor_response import track_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        result = track_vendor_response(self._make_tool_use("CVE-2099-9999"))
        payload = result["content"][0]["json"]
        assert payload["vendor_response_state"] == "unknown"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_signals_include_api_key_present(self, mock_nvd, mock_cisa, mock_vc):
        from manus_agent.tools.track_vendor_response import track_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict(os.environ, {"VULNCHECK_API_KEY": "test-key"}):
            result = track_vendor_response(self._make_tool_use("CVE-2024-1234"))
        payload = result["content"][0]["json"]
        assert payload["signals"]["vulncheck_api_key_present"] is True

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_no_api_key_signals_false(self, mock_nvd, mock_cisa, mock_vc):
        from manus_agent.tools.track_vendor_response import track_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict(os.environ, {}, clear=True):
            result = track_vendor_response(self._make_tool_use("CVE-2024-1234"))
        payload = result["content"][0]["json"]
        assert payload["signals"]["vulncheck_api_key_present"] is False

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_tool_use_id_propagated(self, mock_nvd, mock_cisa, mock_vc):
        from manus_agent.tools.track_vendor_response import track_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        result = track_vendor_response(self._make_tool_use("CVE-2024-1234"))
        assert result["toolUseId"] == "test-tool-use-id"


# ---------------------------------------------------------------------------
# _fetch_nvd_references tests
# ---------------------------------------------------------------------------


class TestFetchNvdReferences:
    """Tests for the NVD reference fetcher."""

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_successful_fetch(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_nvd_references

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "vulnerabilities": [
                    {
                        "cve": {
                            "references": [
                                {"url": "https://example.com/patch", "tags": ["Patch"]},
                                {"url": "https://example.com/advisory", "tags": ["Vendor Advisory"]},
                            ]
                        }
                    }
                ]
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        refs = _fetch_nvd_references("CVE-2024-3094")
        assert len(refs) == 2
        assert refs[0]["tags"] == ["Patch"]

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_empty_vulnerabilities(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_nvd_references

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"vulnerabilities": []},
        )
        mock_get.return_value.raise_for_status = MagicMock()

        refs = _fetch_nvd_references("CVE-2099-0001")
        assert refs == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_no_references_key(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_nvd_references

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"vulnerabilities": [{"cve": {}}]},
        )
        mock_get.return_value.raise_for_status = MagicMock()

        refs = _fetch_nvd_references("CVE-2024-0001")
        assert refs == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_network_error_returns_empty(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_nvd_references

        mock_get.side_effect = Exception("Connection timed out")

        refs = _fetch_nvd_references("CVE-2024-3094")
        assert refs == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_http_error_returns_empty(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_nvd_references

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("403 Forbidden")
        mock_get.return_value = mock_resp

        refs = _fetch_nvd_references("CVE-2024-3094")
        assert refs == []


# ---------------------------------------------------------------------------
# _fetch_cisa_kev tests
# ---------------------------------------------------------------------------


class TestFetchCisaKev:
    """Tests for the CISA KEV fetcher."""

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_cve_found_in_kev(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_cisa_kev

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "vulnerabilities": [
                    {
                        "cveID": "CVE-2024-3094",
                        "shortDescription": "xz utils backdoor",
                        "requiredAction": "Apply updates per vendor instructions.",
                    },
                    {
                        "cveID": "CVE-2021-44228",
                        "shortDescription": "Log4j",
                        "requiredAction": "Apply updates.",
                    },
                ]
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = _fetch_cisa_kev("CVE-2024-3094")
        assert result["cveID"] == "CVE-2024-3094"
        assert "xz utils" in result["shortDescription"]

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_cve_not_in_kev(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_cisa_kev

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "vulnerabilities": [
                    {"cveID": "CVE-2021-44228", "shortDescription": "Log4j"},
                ]
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = _fetch_cisa_kev("CVE-2099-0001")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_network_error_returns_empty(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_cisa_kev

        mock_get.side_effect = Exception("Network error")

        result = _fetch_cisa_kev("CVE-2024-3094")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_empty_catalog(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_cisa_kev

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"vulnerabilities": []},
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = _fetch_cisa_kev("CVE-2024-3094")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_case_insensitive_match(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_cisa_kev

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "vulnerabilities": [
                    {"cveID": "cve-2024-3094", "shortDescription": "test"},
                ]
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        # The function compares cveID.upper() with the passed-in cve_id,
        # so "CVE-2024-3094" should match "cve-2024-3094".
        result = _fetch_cisa_kev("CVE-2024-3094")
        assert result["cveID"] == "cve-2024-3094"


# ---------------------------------------------------------------------------
# _fetch_vulncheck_kev tests
# ---------------------------------------------------------------------------


class TestFetchVulncheckKev:
    """Tests for the VulnCheck KEV fetcher."""

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_successful_fetch(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_vulncheck_kev

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [
                    {
                        "cve": "CVE-2024-3094",
                        "ransomwareUse": True,
                    }
                ]
            },
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = _fetch_vulncheck_kev("CVE-2024-3094", "test-api-key")
        assert result["cve"] == "CVE-2024-3094"

    def test_no_api_key_returns_empty(self):
        from manus_agent.tools.track_vendor_response import _fetch_vulncheck_kev

        result = _fetch_vulncheck_kev("CVE-2024-3094", "")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_empty_data_returns_empty(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_vulncheck_kev

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": []},
        )
        mock_get.return_value.raise_for_status = MagicMock()

        result = _fetch_vulncheck_kev("CVE-2099-0001", "test-key")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_network_error_returns_empty(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_vulncheck_kev

        mock_get.side_effect = Exception("Timeout")

        result = _fetch_vulncheck_kev("CVE-2024-3094", "test-key")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_auth_header_sent(self, mock_get):
        from manus_agent.tools.track_vendor_response import _fetch_vulncheck_kev

        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": []},
        )
        mock_get.return_value.raise_for_status = MagicMock()

        _fetch_vulncheck_kev("CVE-2024-3094", "my-secret-key")
        call_kwargs = mock_get.call_args
        assert "Authorization" in call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))


# ---------------------------------------------------------------------------
# _classify tests
# ---------------------------------------------------------------------------


class TestClassify:
    """Tests for the classification logic."""

    def test_patch_tag_produces_patch_available(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com/fix", "tags": ["Patch"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"
        assert confidence >= 0.7

    def test_vendor_advisory_tag_produces_patch_available(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com/advisory", "tags": ["Vendor-Advisory"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"

    def test_fix_tag_produces_patch_available(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com", "tags": ["Fix"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"

    def test_release_notes_tag_produces_patch_available(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com", "tags": ["Release-Notes"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"

    def test_mitigation_tag_produces_workaround(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com/mitigate", "tags": ["Mitigation"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"

    def test_workaround_tag_produces_workaround(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com", "tags": ["Workaround"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"

    def test_patch_keyword_in_url(self):
        from manus_agent.tools.track_vendor_response import _classify

        # "patch" keyword matches in URL path
        refs = [{"url": "https://example.com/patch-v2.1", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"
        assert any("patch" in e.lower() for e in evidence)

    def test_workaround_keyword_in_url(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com/workaround-guide", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"

    def test_cisa_kev_apply_update_elevates(self):
        from manus_agent.tools.track_vendor_response import _classify

        cisa = {
            "cveID": "CVE-2024-3094",
            "shortDescription": "xz utils",
            "requiredAction": "Apply updates per vendor instructions.",
        }
        state, confidence, evidence = _classify([], cisa, {}, "unknown")
        assert state == "patch_available"
        assert confidence >= 0.4

    def test_cisa_kev_patch_action(self):
        from manus_agent.tools.track_vendor_response import _classify

        cisa = {
            "cveID": "CVE-2024-1234",
            "requiredAction": "Patch the affected software.",
        }
        state, confidence, evidence = _classify([], cisa, {}, "unknown")
        assert state == "patch_available"

    def test_vulncheck_kev_investigating(self):
        from manus_agent.tools.track_vendor_response import _classify

        vc = {"cve": "CVE-2024-3094"}
        state, confidence, evidence = _classify([], {}, vc, "unknown")
        assert state == "investigating"
        assert any("vulncheck" in e.lower() for e in evidence)

    def test_vulncheck_kev_ransomware_boosts_confidence(self):
        from manus_agent.tools.track_vendor_response import _classify

        vc = {"cve": "CVE-2024-3094", "ransomwareUse": True}
        state1, conf1, _ = _classify([], {}, {"cve": "CVE-2024-3094"}, "unknown")
        state2, conf2, _ = _classify([], {}, vc, "unknown")
        assert conf2 >= conf1
        # Ransomware evidence should be mentioned
        _, _, evidence = _classify([], {}, vc, "unknown")
        assert any("ransomware" in e.lower() for e in evidence)

    def test_vulncheck_known_ransomware_campaign_use(self):
        from manus_agent.tools.track_vendor_response import _classify

        vc = {"cve": "CVE-2024-3094", "knownRansomwareCampaignUse": True}
        _, _, evidence = _classify([], {}, vc, "unknown")
        assert any("ransomware" in e.lower() for e in evidence)

    def test_vulncheck_ransomware_use_field(self):
        from manus_agent.tools.track_vendor_response import _classify

        vc = {"cve": "CVE-2024-3094", "ransomware_use": True}
        _, _, evidence = _classify([], {}, vc, "unknown")
        assert any("ransomware" in e.lower() for e in evidence)

    def test_empty_inputs_return_unknown(self):
        from manus_agent.tools.track_vendor_response import _classify

        state, confidence, evidence = _classify([], {}, {}, "unknown")
        assert state == "unknown"
        assert confidence <= 0.3

    def test_all_signals_combined_high_confidence(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com/patch", "tags": ["Patch"]}]
        cisa = {
            "cveID": "CVE-2024-3094",
            "shortDescription": "test",
            "requiredAction": "Apply update.",
        }
        vc = {"cve": "CVE-2024-3094", "ransomwareUse": True}
        state, confidence, evidence = _classify(refs, cisa, vc, "analyzed")
        assert state == "patch_available"
        assert confidence >= 0.9

    def test_nvd_analyzed_status_adds_evidence(self):
        from manus_agent.tools.track_vendor_response import _classify

        _, _, evidence = _classify([], {}, {}, "analyzed")
        assert any("NVD status" in e for e in evidence)

    def test_nvd_modified_status_adds_evidence(self):
        from manus_agent.tools.track_vendor_response import _classify

        _, _, evidence = _classify([], {}, {}, "Modified")
        assert any("NVD status" in e for e in evidence)

    def test_confidence_capped_at_one_or_less(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com/patch", "tags": ["Patch", "Fix"]}]
        cisa = {
            "cveID": "CVE-2024-3094",
            "shortDescription": "test",
            "requiredAction": "Apply update now.",
        }
        vc = {"cve": "CVE-2024-3094", "ransomwareUse": True}
        _, confidence, _ = _classify(refs, cisa, vc, "analyzed")
        assert confidence <= 1.0

    def test_ref_without_tags_is_safe(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com/something"}]
        state, _, _ = _classify(refs, {}, {}, "unknown")
        # Should not crash — tags key may be missing
        assert state in ("unknown", "patch_available", "workaround_only", "investigating")

    def test_ref_without_url_is_safe(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"tags": ["Patch"]}]
        state, _, _ = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_multiple_refs_with_mixed_tags(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [
            {"url": "https://example.com/info", "tags": ["Third Party Advisory"]},
            {"url": "https://example.com/fix", "tags": ["Patch"]},
        ]
        state, _, _ = _classify(refs, {}, {}, "analyzed")
        assert state == "patch_available"

    def test_patch_tag_takes_precedence_over_mitigation(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [
            {"url": "https://example.com", "tags": ["Patch", "Mitigation"]},
        ]
        state, _, _ = _classify(refs, {}, {}, "analyzed")
        # Patch should win over mitigation
        assert state == "patch_available"


# ---------------------------------------------------------------------------
# TOOL_SPEC tests
# ---------------------------------------------------------------------------


class TestToolSpec:
    """Tests for the tool specification."""

    def test_tool_spec_name(self):
        from manus_agent.tools.track_vendor_response import TOOL_SPEC

        assert TOOL_SPEC["name"] == "track_vendor_response"

    def test_tool_spec_has_input_schema(self):
        from manus_agent.tools.track_vendor_response import TOOL_SPEC

        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]

    def test_tool_spec_description_non_empty(self):
        from manus_agent.tools.track_vendor_response import TOOL_SPEC

        assert len(TOOL_SPEC["description"]) > 50

    def test_valid_states_set(self):
        from manus_agent.tools.track_vendor_response import _VALID_STATES

        assert "patch_available" in _VALID_STATES
        assert "unknown" in _VALID_STATES
        assert len(_VALID_STATES) == 6


# ---------------------------------------------------------------------------
# CLI: _build_vendor_response_parser tests
# ---------------------------------------------------------------------------


class TestBuildVendorResponseParser:
    """Tests for the vendor-response CLI argument parser."""

    def test_parser_accepts_cve_id(self):
        from manus_agent.cli import _build_vendor_response_parser

        parser = _build_vendor_response_parser()
        args = parser.parse_args(["CVE-2024-3094"])
        assert args.cve_id == "CVE-2024-3094"

    def test_parser_default_output_is_text(self):
        from manus_agent.cli import _build_vendor_response_parser

        parser = _build_vendor_response_parser()
        args = parser.parse_args(["CVE-2024-3094"])
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

    def test_parser_requires_cve_id(self):
        from manus_agent.cli import _build_vendor_response_parser

        parser = _build_vendor_response_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_prog_name(self):
        from manus_agent.cli import _build_vendor_response_parser

        parser = _build_vendor_response_parser()
        assert parser.prog == "manus-agent vendor-response"


# ---------------------------------------------------------------------------
# CLI: _run_vendor_response tests
# ---------------------------------------------------------------------------


class TestRunVendorResponse:
    """Tests for the _run_vendor_response CLI runner."""

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_text_output(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = [{"url": "https://example.com/fix", "tags": ["Patch"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        exit_code = _run_vendor_response(["CVE-2024-3094"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "CVE-2024-3094" in captured.out
        assert "Classification" in captured.out
        assert "Confidence" in captured.out

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_json_output(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = [{"url": "https://example.com/fix", "tags": ["Patch"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        exit_code = _run_vendor_response(["CVE-2024-3094", "--output", "json"])
        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["cve_id"] == "CVE-2024-3094"
        assert "classification" in data
        assert "confidence" in data
        assert "confidence_label" in data
        assert "signals" in data

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_json_output_has_evidence(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = [{"url": "https://example.com/fix", "tags": ["Patch"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        _run_vendor_response(["CVE-2024-3094", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data["evidence"], list)

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_confidence_label_high(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        # Patch tag + CISA KEV should give high confidence
        mock_nvd.return_value = [{"url": "https://example.com/fix", "tags": ["Patch"]}]
        mock_cisa.return_value = {
            "cveID": "CVE-2024-3094",
            "requiredAction": "Apply update.",
        }
        mock_vc.return_value = {}

        _run_vendor_response(["CVE-2024-3094", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["confidence_label"] == "high"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_confidence_label_low(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        _run_vendor_response(["CVE-2099-9999", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["confidence_label"] == "low"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_confidence_label_moderate(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        # Workaround keyword gives moderate confidence (0.5-0.74)
        mock_nvd.return_value = [{"url": "https://example.com/workaround-guide", "tags": ["Mitigation"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        _run_vendor_response(["CVE-2024-1234", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["confidence_label"] == "moderate"

    def test_invalid_cve_id_exits_nonzero(self):
        from manus_agent.cli import _run_vendor_response

        with pytest.raises(SystemExit) as exc_info:
            _run_vendor_response(["not-a-cve"])
        assert exc_info.value.code != 0

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_text_output_shows_signals(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = [{"url": "https://example.com", "tags": ["Patch"]}]
        mock_cisa.return_value = {"cveID": "CVE-2024-3094", "requiredAction": "Apply update."}
        mock_vc.return_value = {"cve": "CVE-2024-3094"}

        _run_vendor_response(["CVE-2024-3094"])
        out = capsys.readouterr().out
        assert "Signals:" in out
        assert "NVD references found" in out
        assert "CISA KEV hit" in out
        assert "VulnCheck KEV hit" in out

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_text_output_shows_evidence(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = [{"url": "https://example.com", "tags": ["Patch"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        _run_vendor_response(["CVE-2024-3094"])
        out = capsys.readouterr().out
        assert "Evidence:" in out

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_json_signals_structure(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict(os.environ, {"VULNCHECK_API_KEY": "test-key-123"}):
            _run_vendor_response(["CVE-2024-1234", "--output", "json"])

        data = json.loads(capsys.readouterr().out)
        signals = data["signals"]
        assert isinstance(signals["nvd_references_found"], int)
        assert isinstance(signals["cisa_kev_hit"], bool)
        assert isinstance(signals["vulncheck_kev_hit"], bool)
        assert signals["vulncheck_api_key_present"] is True

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_cve_id_uppercased_in_output(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        _run_vendor_response(["cve-2024-3094", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["cve_id"] == "CVE-2024-3094"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_no_evidence_text_still_works(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """When no signals produce evidence, text output should still render cleanly."""
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        exit_code = _run_vendor_response(["CVE-2099-9999"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "CVE-2099-9999" in out

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_vulncheck_kev_plus_patch_is_high_confidence(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        mock_nvd.return_value = [{"url": "https://example.com/fix", "tags": ["Patch"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {"cve": "CVE-2024-3094", "ransomwareUse": True}

        _run_vendor_response(["CVE-2024-3094", "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["confidence"] >= 0.85
        assert data["confidence_label"] == "high"


# ---------------------------------------------------------------------------
# CLI dispatch integration test
# ---------------------------------------------------------------------------


class TestCliDispatch:
    """Tests for the CLI dispatch to vendor-response."""

    def test_vendor_response_in_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "vendor-response" in _SUBCOMMANDS

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_main_dispatches_vendor_response(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import main

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.object(sys, "argv", ["manus-agent", "vendor-response", "CVE-2024-3094", "--output", "json"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

        data = json.loads(capsys.readouterr().out)
        assert data["cve_id"] == "CVE-2024-3094"
