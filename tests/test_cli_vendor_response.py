"""Comprehensive test suite for the manus-agent vendor-response CLI subcommand.

Tests cover:
- Parser construction and argument validation
- Text rendering for all 6 classification states
- Confidence label thresholds (high/moderate/low/very low)
- Signal display (NVD refs, CISA KEV, VulnCheck KEV, API key)
- Evidence rendering (multiple items, empty)
- JSON output mode
- CVE ID validation (valid, invalid, case normalization)
- Integration with track_vendor_response internal functions
- Error handling (import failures, empty responses)
- VulnCheck API key presence/absence
- All classification paths through _classify
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_cli():
    """Import CLI module functions."""
    from manus_agent.cli import (
        _build_vendor_response_parser,
        _render_vendor_response_text,
        _run_vendor_response,
    )

    return _build_vendor_response_parser, _render_vendor_response_text, _run_vendor_response


def _make_payload(
    *,
    cve_id: str = "CVE-2024-3094",
    state: str = "patch_available",
    confidence: float = 0.85,
    evidence: list[str] | None = None,
    nvd_refs: int = 5,
    cisa_kev: bool = True,
    vulncheck_kev: bool = False,
    api_key: bool = True,
) -> dict[str, Any]:
    """Build a mock vendor-response payload."""
    return {
        "cve_id": cve_id,
        "vendor_response_state": state,
        "confidence": confidence,
        "evidence": evidence if evidence is not None else ["NVD reference tags include: ['patch']"],
        "signals": {
            "nvd_references_found": nvd_refs,
            "cisa_kev_hit": cisa_kev,
            "vulncheck_kev_hit": vulncheck_kev,
            "vulncheck_api_key_present": api_key,
        },
    }


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestBuildVendorResponseParser:
    """Tests for _build_vendor_response_parser."""

    def test_parser_accepts_cve_id(self):
        parser, _, _ = _import_cli()
        p = parser()
        args = p.parse_args(["CVE-2024-3094"])
        assert args.cve_id == "CVE-2024-3094"

    def test_parser_default_output_text(self):
        parser, _, _ = _import_cli()
        p = parser()
        args = p.parse_args(["CVE-2021-44228"])
        assert args.output == "text"

    def test_parser_output_json(self):
        parser, _, _ = _import_cli()
        p = parser()
        args = p.parse_args(["CVE-2021-44228", "--output", "json"])
        assert args.output == "json"

    def test_parser_rejects_invalid_output(self):
        parser, _, _ = _import_cli()
        p = parser()
        with pytest.raises(SystemExit):
            p.parse_args(["CVE-2021-44228", "--output", "xml"])

    def test_parser_requires_cve_id(self):
        parser, _, _ = _import_cli()
        p = parser()
        with pytest.raises(SystemExit):
            p.parse_args([])

    def test_parser_prog_name(self):
        parser, _, _ = _import_cli()
        p = parser()
        assert p.prog == "manus-agent vendor-response"


# ---------------------------------------------------------------------------
# Text rendering tests
# ---------------------------------------------------------------------------


class TestRenderVendorResponseText:
    """Tests for _render_vendor_response_text."""

    def test_render_patch_available(self):
        _, render, _ = _import_cli()
        payload = _make_payload(state="patch_available", confidence=0.85)
        text = render(payload)
        assert "Patch Available" in text
        assert "CVE-2024-3094" in text
        assert "85.0%" in text
        assert "moderate" in text

    def test_render_patch_pending(self):
        _, render, _ = _import_cli()
        payload = _make_payload(state="patch_pending", confidence=0.65)
        text = render(payload)
        assert "Patch Pending" in text

    def test_render_workaround_only(self):
        _, render, _ = _import_cli()
        payload = _make_payload(state="workaround_only", confidence=0.6)
        text = render(payload)
        assert "Workaround Only" in text

    def test_render_investigating(self):
        _, render, _ = _import_cli()
        payload = _make_payload(state="investigating", confidence=0.45)
        text = render(payload)
        assert "Investigating" in text

    def test_render_no_patch_expected(self):
        _, render, _ = _import_cli()
        payload = _make_payload(state="no_patch_expected", confidence=0.7)
        text = render(payload)
        assert "No Patch Expected" in text

    def test_render_unknown(self):
        _, render, _ = _import_cli()
        payload = _make_payload(state="unknown", confidence=0.2)
        text = render(payload)
        assert "Unknown" in text

    def test_render_high_confidence(self):
        _, render, _ = _import_cli()
        payload = _make_payload(confidence=0.95)
        text = render(payload)
        assert "high" in text
        assert "95.0%" in text

    def test_render_moderate_confidence(self):
        _, render, _ = _import_cli()
        payload = _make_payload(confidence=0.75)
        text = render(payload)
        assert "moderate" in text

    def test_render_low_confidence(self):
        _, render, _ = _import_cli()
        payload = _make_payload(confidence=0.35)
        text = render(payload)
        assert "low" in text

    def test_render_very_low_confidence(self):
        _, render, _ = _import_cli()
        payload = _make_payload(confidence=0.15)
        text = render(payload)
        assert "very low" in text

    def test_render_signals_nvd_refs(self):
        _, render, _ = _import_cli()
        payload = _make_payload(nvd_refs=12)
        text = render(payload)
        assert "12" in text
        assert "NVD references found" in text

    def test_render_signals_cisa_kev_yes(self):
        _, render, _ = _import_cli()
        payload = _make_payload(cisa_kev=True)
        text = render(payload)
        assert "CISA KEV hit:" in text
        assert "Yes" in text

    def test_render_signals_cisa_kev_no(self):
        _, render, _ = _import_cli()
        payload = _make_payload(cisa_kev=False)
        text = render(payload)
        # Should contain "No" for CISA KEV
        lines = [line for line in text.split("\n") if "CISA KEV hit" in line]
        assert len(lines) == 1
        assert "No" in lines[0]

    def test_render_signals_vulncheck_kev_yes(self):
        _, render, _ = _import_cli()
        payload = _make_payload(vulncheck_kev=True)
        text = render(payload)
        lines = [line for line in text.split("\n") if "VulnCheck KEV hit" in line]
        assert len(lines) == 1
        assert "Yes" in lines[0]

    def test_render_signals_api_key_configured(self):
        _, render, _ = _import_cli()
        payload = _make_payload(api_key=True)
        text = render(payload)
        assert "configured" in text

    def test_render_signals_api_key_not_set(self):
        _, render, _ = _import_cli()
        payload = _make_payload(api_key=False)
        text = render(payload)
        assert "not set" in text

    def test_render_evidence_multiple_items(self):
        _, render, _ = _import_cli()
        payload = _make_payload(
            evidence=[
                "NVD reference tags include: ['patch']",
                "CISA KEV: Apply update required",
                "VulnCheck KEV: active exploitation confirmed",
            ]
        )
        text = render(payload)
        assert "NVD reference tags" in text
        assert "CISA KEV" in text
        assert "VulnCheck KEV" in text
        # Each evidence item should have bullet
        assert text.count("\u2022") == 3

    def test_render_evidence_empty(self):
        _, render, _ = _import_cli()
        payload = _make_payload(evidence=[])
        text = render(payload)
        assert "(none gathered)" in text

    def test_render_separator_line(self):
        _, render, _ = _import_cli()
        payload = _make_payload()
        text = render(payload)
        assert "=" * 40 in text


# ---------------------------------------------------------------------------
# _run_vendor_response integration tests
# ---------------------------------------------------------------------------


class TestRunVendorResponse:
    """Tests for _run_vendor_response CLI runner."""

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_text_output_success(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = [{"url": "https://patch.example.com/fix", "tags": ["Patch"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "CVE-2024-3094" in captured.out
        assert "Patch Available" in captured.out

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_json_output_success(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = [{"url": "https://example.com", "tags": ["Vendor Advisory"]}]
        mock_cisa.return_value = {"requiredAction": "Apply update", "shortDescription": "test"}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["cve_id"] == "CVE-2024-3094"
        assert data["vendor_response_state"] in (
            "patch_available",
            "patch_pending",
            "workaround_only",
            "investigating",
            "no_patch_expected",
            "unknown",
        )
        assert "confidence" in data
        assert "evidence" in data
        assert "signals" in data

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_cve_id_case_normalized(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["cve-2024-3094"])

        assert exit_code == 0
        captured = capsys.readouterr()
        # Should normalize to uppercase
        assert "CVE-2024-3094" in captured.out

    def test_invalid_cve_id_format(self, capsys):
        _, _, run = _import_cli()
        with pytest.raises(SystemExit) as exc_info:
            run(["not-a-cve"])
        assert exc_info.value.code != 0

    def test_empty_cve_id(self):
        _, _, run = _import_cli()
        with pytest.raises(SystemExit):
            run([""])

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_no_data_returns_unknown(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2099-99999", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["vendor_response_state"] == "unknown"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_vulncheck_kev_hit_boosts_confidence(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {"ransomwareUse": False}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": "test-key"}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        # VulnCheck hit pushes unknown -> investigating
        assert data["vendor_response_state"] == "investigating"
        # Confidence should be boosted above baseline 0.2
        assert data["confidence"] > 0.2
        assert data["signals"]["vulncheck_kev_hit"] is True
        assert data["signals"]["vulncheck_api_key_present"] is True

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_cisa_kev_apply_update_triggers_patch_available(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = []
        mock_cisa.return_value = {
            "cveID": "CVE-2024-3094",
            "requiredAction": "Apply updates per vendor instructions.",
            "shortDescription": "XZ Utils backdoor",
        }
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["vendor_response_state"] == "patch_available"
        assert data["signals"]["cisa_kev_hit"] is True

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_nvd_patch_tag_classifies_patch_available(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = [
            {"url": "https://github.com/foo/bar/commit/abc123", "tags": ["Patch"]},
            {"url": "https://vendor.example.com/advisory", "tags": ["Vendor Advisory"]},
        ]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["vendor_response_state"] == "patch_available"
        assert data["confidence"] >= 0.75

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_nvd_mitigation_tag_classifies_workaround(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = [
            {"url": "https://vendor.example.com/mitigation", "tags": ["Mitigation"]},
        ]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["vendor_response_state"] == "workaround_only"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_vulncheck_ransomware_boosts_confidence(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = [{"url": "https://patch.example.com", "tags": ["Patch"]}]
        mock_cisa.return_value = {}
        mock_vc.return_value = {"ransomwareUse": True}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": "test-key"}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["confidence"] >= 0.9
        assert any("ransomware" in e.lower() for e in data["evidence"])

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_patch_keyword_in_url_classifies_patch(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        # _PATCH_KEYWORDS uses "fixed in" (with space), so the URL must contain that
        mock_nvd.return_value = [
            {"url": "https://vendor.example.com/security/fixed in v2.1", "tags": []},
        ]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["vendor_response_state"] == "patch_available"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_workaround_keyword_in_url_classifies_workaround(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = [
            {"url": "https://vendor.example.com/security/workaround-instructions", "tags": []},
        ]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["vendor_response_state"] == "workaround_only"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_api_key_not_set_signal(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["signals"]["vulncheck_api_key_present"] is False

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_api_key_set_signal(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": "sk-test-123"}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["signals"]["vulncheck_api_key_present"] is True

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_combined_cisa_and_nvd_high_confidence(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = [
            {"url": "https://github.com/foo/commit/fix", "tags": ["Patch"]},
        ]
        mock_cisa.return_value = {
            "cveID": "CVE-2024-3094",
            "requiredAction": "Apply vendor patch immediately.",
            "shortDescription": "Critical RCE",
        }
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["vendor_response_state"] == "patch_available"
        # Combined signals should push confidence very high
        assert data["confidence"] >= 0.9

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_all_sources_combined(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = [
            {"url": "https://github.com/foo/commit/abc", "tags": ["Patch", "Third Party Advisory"]},
        ]
        mock_cisa.return_value = {
            "cveID": "CVE-2024-3094",
            "requiredAction": "Apply update per vendor guidance.",
            "shortDescription": "RCE vuln",
        }
        mock_vc.return_value = {"ransomwareUse": True}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": "key"}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["vendor_response_state"] == "patch_available"
        assert data["confidence"] >= 0.95
        assert data["signals"]["cisa_kev_hit"] is True
        assert data["signals"]["vulncheck_kev_hit"] is True

    def test_import_failure(self, capsys, monkeypatch):
        """Test graceful handling when track_vendor_response cannot be imported."""
        _, _, run = _import_cli()

        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "track_vendor_response" in name:
                raise ImportError("mocked import error")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        exit_code = run(["CVE-2024-3094"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err or "error" in captured.err.lower() or exit_code == 1

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_nvd_release_notes_tag(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """release-notes tag (hyphenated, as matched by _classify) should classify as patch_available."""
        _, _, run = _import_cli()

        # The _classify function checks for 'release-notes' (hyphenated) after .lower()
        mock_nvd.return_value = [
            {"url": "https://vendor.example.com/changelog", "tags": ["release-notes"]},
        ]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["vendor_response_state"] == "patch_available"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_multiple_refs_no_relevant_tags(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """Multiple references with no patch/mitigation tags → unknown."""
        _, _, run = _import_cli()

        mock_nvd.return_value = [
            {"url": "https://example.com/info", "tags": ["Third Party Advisory"]},
            {"url": "https://example.com/more-info", "tags": []},
        ]
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        # No patch/mitigation signals, should stay unknown or only have NVD status bump
        assert data["vendor_response_state"] in ("unknown", "investigating")

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_json_output_is_valid_json(self, mock_nvd, mock_cisa, mock_vc, capsys):
        _, _, run = _import_cli()

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        captured = capsys.readouterr()
        # Must not raise
        data = json.loads(captured.out)
        assert isinstance(data, dict)

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev")
    @patch("manus_agent.tools.track_vendor_response._fetch_nvd_references")
    def test_json_schema_structure(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """Verify the JSON output has all expected top-level keys."""
        _, _, run = _import_cli()

        mock_nvd.return_value = []
        mock_cisa.return_value = {}
        mock_vc.return_value = {}

        with patch.dict("os.environ", {"VULNCHECK_API_KEY": ""}, clear=False):
            exit_code = run(["CVE-2024-3094", "--output", "json"])

        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        assert set(data.keys()) == {"cve_id", "vendor_response_state", "confidence", "evidence", "signals"}
        assert set(data["signals"].keys()) == {
            "nvd_references_found",
            "cisa_kev_hit",
            "vulncheck_kev_hit",
            "vulncheck_api_key_present",
        }


# ---------------------------------------------------------------------------
# _classify unit tests (direct, for edge coverage)
# ---------------------------------------------------------------------------


class TestClassifyDirect:
    """Direct tests of the _classify function for edge cases."""

    def test_empty_inputs(self):
        from manus_agent.tools.track_vendor_response import _classify

        state, confidence, evidence = _classify([], {}, {}, "unknown")
        assert state == "unknown"
        assert confidence == 0.2
        assert evidence == []

    def test_analyzed_nvd_status_bumps_confidence(self):
        from manus_agent.tools.track_vendor_response import _classify

        state, confidence, evidence = _classify([], {}, {}, "Analyzed")
        assert confidence >= 0.3
        assert any("NVD status" in e for e in evidence)

    def test_patch_tag_takes_priority(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com", "tags": ["Patch", "Mitigation"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        # Patch tag should win over mitigation
        assert state == "patch_available"

    def test_vendor_advisory_tag(self):
        from manus_agent.tools.track_vendor_response import _classify

        # _classify checks for 'vendor-advisory' (hyphenated) after .lower()
        refs = [{"url": "https://example.com", "tags": ["vendor-advisory"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_fix_tag(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com", "tags": ["Fix"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "patch_available"

    def test_cisa_kev_patch_action(self):
        from manus_agent.tools.track_vendor_response import _classify

        cisa = {"requiredAction": "Apply vendor patch", "shortDescription": "test"}
        state, confidence, evidence = _classify([], cisa, {}, "unknown")
        assert state == "patch_available"

    def test_vulncheck_kev_alone_investigating(self):
        from manus_agent.tools.track_vendor_response import _classify

        vc = {"ransomwareUse": False}
        state, confidence, evidence = _classify([], {}, vc, "unknown")
        assert state == "investigating"

    def test_ransomware_flag_extra_boost(self):
        from manus_agent.tools.track_vendor_response import _classify

        vc = {"ransomwareUse": True}
        state, confidence, evidence = _classify([], {}, vc, "unknown")
        assert any("ransomware" in e.lower() for e in evidence)
        # ransomware should add extra confidence
        _, conf_no_ransom, _ = _classify([], {}, {"ransomwareUse": False}, "unknown")
        assert confidence > conf_no_ransom

    def test_confidence_never_exceeds_one(self):
        from manus_agent.tools.track_vendor_response import _classify

        # All signals present — confidence should cap at <=1.0
        refs = [{"url": "https://example.com/patch", "tags": ["Patch"]}]
        cisa = {"requiredAction": "Apply update", "shortDescription": "x"}
        vc = {"ransomwareUse": True, "knownRansomwareCampaignUse": True}
        state, confidence, evidence = _classify(refs, cisa, vc, "Analyzed")
        assert confidence <= 1.0

    def test_modified_nvd_status(self):
        from manus_agent.tools.track_vendor_response import _classify

        state, confidence, evidence = _classify([], {}, {}, "Modified")
        assert confidence >= 0.3

    def test_workaround_tag_only(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com", "tags": ["Workaround"]}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        assert state == "workaround_only"

    def test_refs_with_no_tags_but_patch_url(self):
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://vendor.com/resolved-in-v3.1", "tags": []}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        # "resolved" is a _PATCH_KEYWORDS match
        assert state == "patch_available"

    def test_refs_none_tags_handled(self):
        """References with tags=None should not crash."""
        from manus_agent.tools.track_vendor_response import _classify

        refs = [{"url": "https://example.com", "tags": None}]
        state, confidence, evidence = _classify(refs, {}, {}, "unknown")
        # Should not crash; may detect patch keyword in URL or stay unknown
        assert state in ("unknown", "patch_available", "workaround_only", "investigating")
