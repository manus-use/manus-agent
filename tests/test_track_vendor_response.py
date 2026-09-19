"""Comprehensive test suite for track_vendor_response tool + CLI subcommand.

100% mocked — no real HTTP calls.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from manus_agent.tools.track_vendor_response import (
    _PATCH_KEYWORDS,
    _STATE_EMOJI,
    _STATE_LABELS,
    _VALID_STATES,
    _WORKAROUND_KEYWORDS,
    TOOL_SPEC,
    _classify,
    _confidence_bar,
    _confidence_label,
    _fetch_cisa_kev,
    _fetch_nvd_references,
    _fetch_vulncheck_kev,
    _render_text,
    _run_tracking,
    track_vendor_response,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _tool_use(cve_id: Any = "CVE-2024-3094") -> dict:
    return {"toolUseId": "test-123", "input": {"cve_id": cve_id}}


def _make_ref(url: str = "", tags: list[str] | None = None) -> dict:
    r: dict[str, Any] = {}
    if url:
        r["url"] = url
    if tags is not None:
        r["tags"] = tags
    return r


# ═══════════════════════════════════════════════════════════════════════════
# 1. Module-level constants
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_valid_states_are_frozenset(self):
        assert isinstance(_VALID_STATES, frozenset)
        assert len(_VALID_STATES) == 6

    def test_all_states_present(self):
        expected = {
            "patch_available",
            "patch_pending",
            "workaround_only",
            "investigating",
            "no_patch_expected",
            "unknown",
        }
        assert _VALID_STATES == expected

    def test_patch_keywords_are_frozenset(self):
        assert isinstance(_PATCH_KEYWORDS, frozenset)
        assert "patch" in _PATCH_KEYWORDS
        assert "hotfix" in _PATCH_KEYWORDS

    def test_workaround_keywords_are_frozenset(self):
        assert isinstance(_WORKAROUND_KEYWORDS, frozenset)
        assert "workaround" in _WORKAROUND_KEYWORDS
        assert "mitigation" in _WORKAROUND_KEYWORDS

    def test_state_emoji_covers_all_states(self):
        assert set(_STATE_EMOJI.keys()) == _VALID_STATES

    def test_state_labels_covers_all_states(self):
        assert set(_STATE_LABELS.keys()) == _VALID_STATES


# ═══════════════════════════════════════════════════════════════════════════
# 2. TOOL_SPEC
# ═══════════════════════════════════════════════════════════════════════════


class TestToolSpec:
    def test_spec_name(self):
        assert TOOL_SPEC["name"] == "track_vendor_response"

    def test_spec_has_description(self):
        assert "vendor" in TOOL_SPEC["description"].lower()

    def test_spec_requires_cve_id(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]


# ═══════════════════════════════════════════════════════════════════════════
# 3. _fetch_nvd_references
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchNvdReferences:
    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_references_on_success(self, mock_get):
        refs = [{"url": "https://example.com", "tags": ["Patch"]}]
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"vulnerabilities": [{"cve": {"references": refs}}]},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        result = _fetch_nvd_references("CVE-2024-1234")
        assert result == refs

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_no_vulns(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"vulnerabilities": []},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        assert _fetch_nvd_references("CVE-2024-9999") == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_missing_refs_key(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"vulnerabilities": [{"cve": {}}]},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        assert _fetch_nvd_references("CVE-2024-1234") == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_http_error(self, mock_get):
        mock_get.side_effect = Exception("connection timeout")
        assert _fetch_nvd_references("CVE-2024-1234") == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_raise_for_status(self, mock_get):
        mock_get.return_value.raise_for_status.side_effect = Exception("404")
        assert _fetch_nvd_references("CVE-2024-1234") == []

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_null_vulnerabilities(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"vulnerabilities": None},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        assert _fetch_nvd_references("CVE-2024-0001") == []


# ═══════════════════════════════════════════════════════════════════════════
# 4. _fetch_cisa_kev
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchCisaKev:
    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_matching_entry(self, mock_get):
        entry = {"cveID": "CVE-2024-3094", "requiredAction": "Apply update"}
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"vulnerabilities": [entry]},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        assert _fetch_cisa_kev("CVE-2024-3094") == entry

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_no_match(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"vulnerabilities": [{"cveID": "CVE-9999-0001"}]},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        assert _fetch_cisa_kev("CVE-2024-3094") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_exception(self, mock_get):
        mock_get.side_effect = Exception("network error")
        assert _fetch_cisa_kev("CVE-2024-3094") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_case_insensitive_match(self, mock_get):
        entry = {"cveID": "cve-2024-3094", "requiredAction": "Patch"}
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"vulnerabilities": [entry]},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        assert _fetch_cisa_kev("CVE-2024-3094") == entry

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_null_vulnerabilities(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"vulnerabilities": None},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        assert _fetch_cisa_kev("CVE-2024-3094") == {}


# ═══════════════════════════════════════════════════════════════════════════
# 5. _fetch_vulncheck_kev
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchVulncheckKev:
    def test_returns_empty_when_no_api_key(self):
        assert _fetch_vulncheck_kev("CVE-2024-3094", "") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_first_data_entry(self, mock_get):
        data_entry = {"cve": "CVE-2024-3094", "ransomwareUse": True}
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [data_entry]},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        result = _fetch_vulncheck_kev("CVE-2024-3094", "test-key")
        assert result == data_entry

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_empty_data(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": []},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        assert _fetch_vulncheck_kev("CVE-2024-3094", "test-key") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_null_data(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": None},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        assert _fetch_vulncheck_kev("CVE-2024-3094", "test-key") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_returns_empty_on_exception(self, mock_get):
        mock_get.side_effect = Exception("403 Forbidden")
        assert _fetch_vulncheck_kev("CVE-2024-3094", "bad-key") == {}

    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_sends_auth_header(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": []},
        )
        mock_get.return_value.raise_for_status = MagicMock()
        _fetch_vulncheck_kev("CVE-2024-3094", "my-secret-key")
        call_kwargs = mock_get.call_args
        assert "Authorization" in call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))


# ═══════════════════════════════════════════════════════════════════════════
# 6. _classify
# ═══════════════════════════════════════════════════════════════════════════


class TestClassify:
    def test_unknown_with_no_signals(self):
        state, conf, evidence = _classify([], {}, {}, "unknown")
        assert state == "unknown"
        assert conf == 0.2
        assert evidence == []

    def test_patch_available_from_patch_tag(self):
        refs = [_make_ref(tags=["Patch"])]
        state, conf, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"
        assert conf >= 0.75

    def test_patch_available_from_vendor_advisory_tag(self):
        refs = [_make_ref(tags=["Vendor-Advisory"])]
        state, conf, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_patch_available_from_fix_tag(self):
        refs = [_make_ref(tags=["Fix"])]
        state, conf, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_patch_available_from_release_notes_tag(self):
        refs = [_make_ref(tags=["Release-Notes"])]
        state, conf, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_workaround_only_from_mitigation_tag(self):
        refs = [_make_ref(tags=["Mitigation"])]
        state, conf, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"
        assert conf >= 0.6

    def test_workaround_only_from_workaround_tag(self):
        refs = [_make_ref(tags=["Workaround"])]
        state, conf, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"

    def test_patch_keyword_in_url(self):
        refs = [_make_ref(url="https://example.com/patch/fix")]
        state, conf, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"
        assert conf >= 0.5

    def test_workaround_keyword_in_url(self):
        refs = [_make_ref(url="https://example.com/workaround")]
        state, conf, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"
        assert conf >= 0.4

    def test_nvd_analyzed_status_boosts_confidence(self):
        state, conf, evidence = _classify([], {}, {}, "analyzed")
        assert conf >= 0.3
        assert any("NVD status" in e for e in evidence)

    def test_nvd_modified_status_boosts_confidence(self):
        state, conf, evidence = _classify([], {}, {}, "Modified")
        assert conf >= 0.3

    def test_cisa_kev_apply_update(self):
        kev = {"requiredAction": "Apply update per vendor instructions", "shortDescription": "test"}
        state, conf, evidence = _classify([], kev, {}, "unknown")
        assert state == "patch_available"
        assert any("CISA KEV" in e for e in evidence)

    def test_cisa_kev_patch_action(self):
        kev = {"requiredAction": "Patch immediately", "shortDescription": "test"}
        state, conf, evidence = _classify([], kev, {}, "unknown")
        assert state == "patch_available"

    def test_cisa_kev_no_action_keyword(self):
        kev = {"requiredAction": "Investigate", "shortDescription": "test"}
        state, conf, evidence = _classify([], kev, {}, "unknown")
        # KEV present but no patch-related action — still unknown
        assert any("CISA KEV" in e for e in evidence)

    def test_vulncheck_kev_bumps_to_investigating(self):
        vc = {"cve": "CVE-2024-3094"}
        state, conf, evidence = _classify([], {}, vc, "unknown")
        assert state == "investigating"
        assert any("VulnCheck KEV" in e for e in evidence)

    def test_vulncheck_kev_ransomware_escalation(self):
        vc = {"cve": "CVE-2024-3094", "ransomwareUse": True}
        state, conf, evidence = _classify([], {}, vc, "unknown")
        assert any("ransomware" in e.lower() for e in evidence)

    def test_vulncheck_kev_ransomware_use_alt_key(self):
        vc = {"cve": "CVE-2024-3094", "ransomware_use": True}
        state, conf, evidence = _classify([], {}, vc, "unknown")
        assert any("ransomware" in e.lower() for e in evidence)

    def test_vulncheck_kev_known_ransomware_campaign(self):
        vc = {"cve": "CVE-2024-3094", "knownRansomwareCampaignUse": "Yes"}
        state, conf, evidence = _classify([], {}, vc, "unknown")
        assert any("ransomware" in e.lower() for e in evidence)

    def test_combined_patch_tag_and_cisa(self):
        refs = [_make_ref(tags=["Patch"])]
        kev = {"requiredAction": "Apply update", "shortDescription": "test"}
        state, conf, evidence = _classify(refs, kev, {}, "analyzed")
        assert state == "patch_available"
        assert conf >= 0.9

    def test_combined_all_signals(self):
        refs = [_make_ref(tags=["Patch"])]
        kev = {"requiredAction": "Apply update", "shortDescription": "test"}
        vc = {"cve": "CVE-2024-3094", "ransomwareUse": True}
        state, conf, evidence = _classify(refs, kev, vc, "analyzed")
        assert state == "patch_available"
        assert conf >= 0.9
        assert len(evidence) >= 3

    def test_confidence_capped_at_098(self):
        refs = [_make_ref(tags=["Patch"])]
        kev = {"requiredAction": "Apply update", "shortDescription": "test"}
        vc = {"cve": "CVE-2024-3094", "ransomwareUse": True}
        state, conf, evidence = _classify(refs, kev, vc, "analyzed")
        assert conf <= 0.98

    def test_empty_tags_list(self):
        refs = [_make_ref(tags=[])]
        state, conf, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "unknown"

    def test_none_tags(self):
        refs = [{"url": "https://example.com"}]
        state, conf, evidence = _classify(refs, {}, {}, "unknown")
        # No tags at all — should not crash
        assert state in _VALID_STATES

    def test_patch_tag_overrides_workaround_in_url(self):
        # If we have both a Patch tag and a workaround URL, patch wins
        refs = [_make_ref(url="https://example.com/workaround", tags=["Patch"])]
        state, conf, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_cisa_kev_upgrades_investigating_to_patch(self):
        vc = {"cve": "CVE-2024-3094"}
        kev = {"requiredAction": "Apply update", "shortDescription": "test"}
        # VulnCheck makes it investigating, then CISA upgrades to patch_available
        state, conf, evidence = _classify([], kev, vc, "unknown")
        assert state == "patch_available"


# ═══════════════════════════════════════════════════════════════════════════
# 7. _confidence_bar and _confidence_label
# ═══════════════════════════════════════════════════════════════════════════


class TestConfidenceHelpers:
    def test_bar_full(self):
        bar = _confidence_bar(1.0, width=10)
        assert len(bar) == 10
        assert "\u2588" * 10 == bar

    def test_bar_empty(self):
        bar = _confidence_bar(0.0, width=10)
        assert len(bar) == 10
        assert "\u2591" * 10 == bar

    def test_bar_half(self):
        bar = _confidence_bar(0.5, width=10)
        assert len(bar) == 10
        assert bar.count("\u2588") == 5

    def test_bar_default_width(self):
        bar = _confidence_bar(0.75)
        assert len(bar) == 20

    def test_label_high(self):
        assert _confidence_label(0.95) == "High"
        assert _confidence_label(0.9) == "High"

    def test_label_medium(self):
        assert _confidence_label(0.75) == "Medium"
        assert _confidence_label(0.6) == "Medium"

    def test_label_low(self):
        assert _confidence_label(0.45) == "Low"
        assert _confidence_label(0.3) == "Low"

    def test_label_very_low(self):
        assert _confidence_label(0.1) == "Very Low"
        assert _confidence_label(0.0) == "Very Low"


# ═══════════════════════════════════════════════════════════════════════════
# 8. _render_text
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderText:
    def _sample_result(self, **overrides) -> dict:
        base = {
            "cve_id": "CVE-2024-3094",
            "vendor_response_state": "patch_available",
            "confidence": 0.85,
            "evidence": ["NVD reference tags include: ['patch']"],
            "signals": {
                "nvd_references_found": 3,
                "cisa_kev_hit": True,
                "vulncheck_kev_hit": False,
                "vulncheck_api_key_present": True,
            },
        }
        base.update(overrides)
        return base

    def test_contains_cve_id(self):
        text = _render_text(self._sample_result())
        assert "CVE-2024-3094" in text

    def test_contains_status_label(self):
        text = _render_text(self._sample_result())
        assert "Patch Available" in text

    def test_contains_confidence_percentage(self):
        text = _render_text(self._sample_result())
        assert "85.0%" in text

    def test_contains_confidence_label(self):
        text = _render_text(self._sample_result())
        assert "Medium" in text

    def test_contains_signals_section(self):
        text = _render_text(self._sample_result())
        assert "Signals" in text
        assert "NVD references" in text

    def test_cisa_kev_yes(self):
        text = _render_text(self._sample_result())
        assert "CISA KEV hit" in text
        assert "Yes" in text

    def test_vulncheck_no_api_key(self):
        text = _render_text(
            self._sample_result(
                signals={
                    "nvd_references_found": 0,
                    "cisa_kev_hit": False,
                    "vulncheck_kev_hit": False,
                    "vulncheck_api_key_present": False,
                }
            )
        )
        assert "N/A (no API key)" in text

    def test_vulncheck_with_key_and_hit(self):
        text = _render_text(
            self._sample_result(
                signals={
                    "nvd_references_found": 1,
                    "cisa_kev_hit": False,
                    "vulncheck_kev_hit": True,
                    "vulncheck_api_key_present": True,
                }
            )
        )
        # Should show "Yes" for VulnCheck
        lines = text.split("\n")
        vc_lines = [line for line in lines if "VulnCheck" in line]
        assert any("Yes" in line for line in vc_lines)

    def test_evidence_listed(self):
        text = _render_text(self._sample_result())
        assert "Evidence" in text
        assert "\u2022" in text  # bullet point

    def test_no_evidence_section_when_empty(self):
        text = _render_text(self._sample_result(evidence=[]))
        assert "Evidence" not in text

    def test_unknown_state_renders(self):
        text = _render_text(self._sample_result(vendor_response_state="unknown", confidence=0.2))
        assert "Unknown" in text
        assert "\u2753" in text

    def test_all_states_have_emoji_in_render(self):
        for state in _VALID_STATES:
            text = _render_text(self._sample_result(vendor_response_state=state))
            assert _STATE_EMOJI[state] in text


# ═══════════════════════════════════════════════════════════════════════════
# 9. _run_tracking
# ═══════════════════════════════════════════════════════════════════════════


class TestRunTracking:
    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=[])
    def test_returns_dict_with_expected_keys(self, mock_nvd, mock_cisa, mock_vc):
        result = _run_tracking("CVE-2024-1234")
        assert "cve_id" in result
        assert "vendor_response_state" in result
        assert "confidence" in result
        assert "evidence" in result
        assert "signals" in result

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=[])
    def test_uppercases_cve_id(self, mock_nvd, mock_cisa, mock_vc):
        result = _run_tracking("cve-2024-1234")
        assert result["cve_id"] == "CVE-2024-1234"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=[{"url": "https://example.com", "tags": ["Patch"]}],
    )
    def test_patch_available_when_nvd_has_patch_tag(self, mock_nvd, mock_cisa, mock_vc):
        result = _run_tracking("CVE-2024-3094")
        assert result["vendor_response_state"] == "patch_available"

    @patch.dict(os.environ, {"VULNCHECK_API_KEY": "test-key"})
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_vulncheck_kev",
        return_value={"cve": "CVE-2024-3094"},
    )
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=[])
    def test_signals_vulncheck_key_present(self, mock_nvd, mock_cisa, mock_vc):
        result = _run_tracking("CVE-2024-3094")
        assert result["signals"]["vulncheck_api_key_present"] is True
        assert result["signals"]["vulncheck_kev_hit"] is True

    @patch.dict(os.environ, {}, clear=True)
    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=[])
    def test_signals_no_vulncheck_key(self, mock_nvd, mock_cisa, mock_vc):
        result = _run_tracking("CVE-2024-1234")
        assert result["signals"]["vulncheck_api_key_present"] is False


# ═══════════════════════════════════════════════════════════════════════════
# 10. track_vendor_response (Strands tool entry point)
# ═══════════════════════════════════════════════════════════════════════════


class TestTrackVendorResponseTool:
    @patch("manus_agent.tools.track_vendor_response._run_tracking")
    def test_success_result(self, mock_run):
        mock_run.return_value = {
            "cve_id": "CVE-2024-3094",
            "vendor_response_state": "patch_available",
            "confidence": 0.85,
            "evidence": [],
            "signals": {},
        }
        result = track_vendor_response(_tool_use("CVE-2024-3094"))
        assert result["status"] == "success"
        assert result["content"][0]["json"]["cve_id"] == "CVE-2024-3094"

    def test_invalid_cve_id_returns_error(self):
        result = track_vendor_response(_tool_use("not-a-cve"))
        assert result["status"] == "error"
        assert "Invalid" in result["content"][0]["text"]

    def test_empty_cve_id_returns_error(self):
        result = track_vendor_response(_tool_use(""))
        assert result["status"] == "error"

    def test_non_string_cve_id_returns_error(self):
        result = track_vendor_response(_tool_use(12345))
        assert result["status"] == "error"

    def test_none_cve_id_returns_error(self):
        result = track_vendor_response(_tool_use(None))
        assert result["status"] == "error"

    @patch("manus_agent.tools.track_vendor_response._run_tracking")
    def test_tool_use_id_propagated(self, mock_run):
        mock_run.return_value = {
            "cve_id": "CVE-2024-3094",
            "vendor_response_state": "unknown",
            "confidence": 0.2,
            "evidence": [],
            "signals": {},
        }
        tool = {"toolUseId": "my-special-id", "input": {"cve_id": "CVE-2024-3094"}}
        result = track_vendor_response(tool)
        assert result["toolUseId"] == "my-special-id"


# ═══════════════════════════════════════════════════════════════════════════
# 11. CLI: _build_vendor_response_parser
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildVendorResponseParser:
    def test_parser_prog(self):
        from manus_agent.cli import _build_vendor_response_parser

        p = _build_vendor_response_parser()
        assert p.prog == "manus-agent vendor-response"

    def test_parser_accepts_cve_id(self):
        from manus_agent.cli import _build_vendor_response_parser

        p = _build_vendor_response_parser()
        args = p.parse_args(["CVE-2024-3094"])
        assert args.cve_id == "CVE-2024-3094"

    def test_parser_default_output_text(self):
        from manus_agent.cli import _build_vendor_response_parser

        p = _build_vendor_response_parser()
        args = p.parse_args(["CVE-2024-3094"])
        assert args.output == "text"

    def test_parser_accepts_json_output(self):
        from manus_agent.cli import _build_vendor_response_parser

        p = _build_vendor_response_parser()
        args = p.parse_args(["CVE-2024-3094", "--output", "json"])
        assert args.output == "json"

    def test_parser_rejects_invalid_output(self):
        from manus_agent.cli import _build_vendor_response_parser

        p = _build_vendor_response_parser()
        with pytest.raises(SystemExit):
            p.parse_args(["CVE-2024-3094", "--output", "xml"])


# ═══════════════════════════════════════════════════════════════════════════
# 12. CLI: _run_vendor_response
# ═══════════════════════════════════════════════════════════════════════════


class TestRunVendorResponseCli:
    @patch(
        "manus_agent.tools.track_vendor_response._run_tracking",
        return_value={
            "cve_id": "CVE-2024-3094",
            "vendor_response_state": "patch_available",
            "confidence": 0.85,
            "evidence": ["Test evidence"],
            "signals": {
                "nvd_references_found": 2,
                "cisa_kev_hit": True,
                "vulncheck_kev_hit": False,
                "vulncheck_api_key_present": True,
            },
        },
    )
    def test_text_output(self, mock_run, capsys):
        from manus_agent.cli import _run_vendor_response

        rc = _run_vendor_response(["CVE-2024-3094"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "CVE-2024-3094" in out
        assert "Patch Available" in out

    @patch(
        "manus_agent.tools.track_vendor_response._run_tracking",
        return_value={
            "cve_id": "CVE-2024-3094",
            "vendor_response_state": "unknown",
            "confidence": 0.2,
            "evidence": [],
            "signals": {
                "nvd_references_found": 0,
                "cisa_kev_hit": False,
                "vulncheck_kev_hit": False,
                "vulncheck_api_key_present": False,
            },
        },
    )
    def test_json_output(self, mock_run, capsys):
        from manus_agent.cli import _run_vendor_response

        rc = _run_vendor_response(["CVE-2024-3094", "--output", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["cve_id"] == "CVE-2024-3094"
        assert data["vendor_response_state"] == "unknown"

    def test_invalid_cve_format_exits(self):
        from manus_agent.cli import _run_vendor_response

        with pytest.raises(SystemExit):
            _run_vendor_response(["NOTACVE"])

    def test_missing_cve_exits(self):
        from manus_agent.cli import _run_vendor_response

        with pytest.raises(SystemExit):
            _run_vendor_response([])


# ═══════════════════════════════════════════════════════════════════════════
# 13. CLI dispatch: vendor-response in _SUBCOMMANDS
# ═══════════════════════════════════════════════════════════════════════════


class TestCliDispatch:
    def test_vendor_response_in_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "vendor-response" in _SUBCOMMANDS

    @patch(
        "manus_agent.tools.track_vendor_response._run_tracking",
        return_value={
            "cve_id": "CVE-2024-3094",
            "vendor_response_state": "patch_available",
            "confidence": 0.85,
            "evidence": [],
            "signals": {
                "nvd_references_found": 1,
                "cisa_kev_hit": False,
                "vulncheck_kev_hit": False,
                "vulncheck_api_key_present": False,
            },
        },
    )
    def test_main_dispatches_vendor_response_text(self, mock_run, capsys):
        import sys

        from manus_agent.cli import main

        with patch.object(sys, "argv", ["manus-agent", "vendor-response", "CVE-2024-3094"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "CVE-2024-3094" in out

    @patch(
        "manus_agent.tools.track_vendor_response._run_tracking",
        return_value={
            "cve_id": "CVE-2024-3094",
            "vendor_response_state": "unknown",
            "confidence": 0.2,
            "evidence": [],
            "signals": {
                "nvd_references_found": 0,
                "cisa_kev_hit": False,
                "vulncheck_kev_hit": False,
                "vulncheck_api_key_present": False,
            },
        },
    )
    def test_main_dispatches_vendor_response_json(self, mock_run, capsys):
        import sys

        from manus_agent.cli import main

        with patch.object(sys, "argv", ["manus-agent", "vendor-response", "CVE-2024-3094", "--output", "json"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["cve_id"] == "CVE-2024-3094"


# ═══════════════════════════════════════════════════════════════════════════
# 14. Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_classify_returns_valid_state(self):
        """All code paths produce a state in _VALID_STATES."""
        for refs in [[], [_make_ref(tags=["Patch"])], [_make_ref(url="https://x.com/workaround")]]:
            for kev in [{}, {"requiredAction": "Apply update", "shortDescription": "x"}]:
                for vc in [{}, {"cve": "CVE-2024-0001"}]:
                    for nvd in ["unknown", "analyzed", "Modified"]:
                        state, conf, _ = _classify(refs, kev, vc, nvd)
                        assert state in _VALID_STATES
                        assert 0 <= conf <= 1.0

    def test_classify_confidence_is_rounded(self):
        state, conf, _ = _classify(
            [_make_ref(tags=["Patch"])],
            {"requiredAction": "Apply update", "shortDescription": "x"},
            {"cve": "CVE-2024-3094", "ransomwareUse": True},
            "analyzed",
        )
        # confidence should have at most 3 decimal places
        assert conf == round(conf, 3)

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=[])
    def test_run_tracking_with_whitespace_cve(self, mock_nvd, mock_cisa, mock_vc):
        result = _run_tracking("  cve-2024-1234  ")
        # Should still work (uppercased, though whitespace isn't stripped by _run_tracking itself)
        assert "CVE" in result["cve_id"]

    def test_render_text_missing_signals_key(self):
        """_render_text should handle missing 'signals' gracefully."""
        result = {
            "cve_id": "CVE-2024-0001",
            "vendor_response_state": "unknown",
            "confidence": 0.2,
            "evidence": [],
        }
        text = _render_text(result)
        assert "CVE-2024-0001" in text

    def test_render_text_missing_evidence_key(self):
        """_render_text should handle missing 'evidence' gracefully."""
        result = {
            "cve_id": "CVE-2024-0001",
            "vendor_response_state": "unknown",
            "confidence": 0.2,
            "signals": {},
        }
        text = _render_text(result)
        assert "CVE-2024-0001" in text
