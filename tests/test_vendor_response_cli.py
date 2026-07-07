"""Tests for the manus-agent vendor-response CLI subcommand.

All HTTP calls are mocked — no real network traffic.
"""

from __future__ import annotations

import json
import sys
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CVE = "CVE-2024-3094"
_CVE_UPPER = "CVE-2024-3094"

_MOCK_REFS = [
    {"url": "https://example.com/advisory", "tags": ["Patch", "Vendor Advisory"]},
    {"url": "https://github.com/example/repo/releases/tag/1.2.3", "tags": ["Release Notes"]},
]

_MOCK_CISA_KEV = {
    "cveID": _CVE_UPPER,
    "shortDescription": "XZ Utils backdoor",
    "requiredAction": "Apply update per vendor instructions",
}

_EMPTY_VC_KEV: dict = {}


def _make_classify_result(
    state: str = "patch_available",
    confidence: float = 0.9,
    evidence: list[str] | None = None,
) -> tuple[str, float, list[str]]:
    return state, confidence, evidence or ["NVD reference tags include: ['patch', 'vendor-advisory']"]


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


class TestVendorResponseParser:
    def test_parser_builds(self):
        from manus_agent import cli

        p = cli._build_vendor_response_parser()
        assert p is not None

    def test_parser_requires_cve_id(self):
        from manus_agent import cli

        with pytest.raises(SystemExit) as exc_info:
            cli._build_vendor_response_parser().parse_args([])
        assert exc_info.value.code == 2

    def test_parser_accepts_cve_id(self):
        from manus_agent import cli

        args = cli._build_vendor_response_parser().parse_args([_CVE])
        assert args.cve_id == _CVE

    def test_output_default_is_text(self):
        from manus_agent import cli

        args = cli._build_vendor_response_parser().parse_args([_CVE])
        assert args.output == "text"

    def test_output_json_accepted(self):
        from manus_agent import cli

        args = cli._build_vendor_response_parser().parse_args([_CVE, "--output", "json"])
        assert args.output == "json"

    def test_output_text_accepted(self):
        from manus_agent import cli

        args = cli._build_vendor_response_parser().parse_args([_CVE, "--output", "text"])
        assert args.output == "text"

    def test_invalid_output_rejected(self):
        from manus_agent import cli

        with pytest.raises(SystemExit) as exc_info:
            cli._build_vendor_response_parser().parse_args([_CVE, "--output", "yaml"])
        assert exc_info.value.code == 2

    def test_parser_prog_name(self):
        from manus_agent import cli

        p = cli._build_vendor_response_parser()
        assert "vendor-response" in p.prog


# ---------------------------------------------------------------------------
# _run_vendor_response: text output
# ---------------------------------------------------------------------------


class TestVendorResponseTextOutput:
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_text_output_exits_zero(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        rc = cli._run_vendor_response([_CVE])
        assert rc == 0

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_text_output_contains_cve_id(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        cli._run_vendor_response([_CVE])
        out = capsys.readouterr().out
        assert _CVE_UPPER in out

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_text_output_contains_status(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        cli._run_vendor_response([_CVE])
        out = capsys.readouterr().out
        assert "Status" in out

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_text_output_patch_available(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """Patch tags in NVD references → PATCH AVAILABLE state."""
        from manus_agent import cli

        cli._run_vendor_response([_CVE])
        out = capsys.readouterr().out
        assert "PATCH AVAILABLE" in out

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=[])
    def test_text_output_unknown_when_no_data(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """No signals → UNKNOWN state."""
        from manus_agent import cli

        cli._run_vendor_response([_CVE])
        out = capsys.readouterr().out
        assert "UNKNOWN" in out

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @mock.patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=[{"url": "https://example.com/workaround", "tags": ["Mitigation"]}],
    )
    def test_text_output_workaround_only(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """Mitigation tag → WORKAROUND ONLY state."""
        from manus_agent import cli

        cli._run_vendor_response([_CVE])
        out = capsys.readouterr().out
        assert "WORKAROUND" in out

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_text_output_shows_confidence(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        cli._run_vendor_response([_CVE])
        out = capsys.readouterr().out
        assert "Confidence" in out

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_text_output_shows_signals(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        cli._run_vendor_response([_CVE])
        out = capsys.readouterr().out
        assert "Signals" in out

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_text_output_cisa_kev_yes(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        cli._run_vendor_response([_CVE])
        out = capsys.readouterr().out
        assert "CISA KEV" in out and "yes" in out

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_text_output_cisa_kev_no(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        cli._run_vendor_response([_CVE])
        out = capsys.readouterr().out
        assert "CISA KEV" in out
        # When no CISA KEV, "no" appears in that line
        cisa_line = [ln for ln in out.splitlines() if "CISA KEV" in ln][0]
        assert "no" in cisa_line


# ---------------------------------------------------------------------------
# _run_vendor_response: JSON output
# ---------------------------------------------------------------------------


class TestVendorResponseJsonOutput:
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_json_output_valid_json(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        rc = cli._run_vendor_response([_CVE, "--output", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, dict)

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_json_output_has_cve_id_field(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        cli._run_vendor_response([_CVE, "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["cve_id"] == _CVE_UPPER

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_json_output_has_vendor_response_state(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        cli._run_vendor_response([_CVE, "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert "vendor_response_state" in data
        assert data["vendor_response_state"] in {
            "patch_available",
            "patch_pending",
            "workaround_only",
            "investigating",
            "no_patch_expected",
            "unknown",
        }

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_json_output_has_confidence(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        cli._run_vendor_response([_CVE, "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_json_output_has_evidence_list(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        cli._run_vendor_response([_CVE, "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert isinstance(data["evidence"], list)

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_json_output_has_signals_dict(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        cli._run_vendor_response([_CVE, "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        sigs = data["signals"]
        assert "nvd_references_found" in sigs
        assert "cisa_kev_hit" in sigs
        assert "vulncheck_kev_hit" in sigs
        assert "vulncheck_api_key_present" in sigs

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_json_state_is_patch_available_with_patch_tags(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """NVD patch tag + CISA KEV → patch_available."""
        from manus_agent import cli

        cli._run_vendor_response([_CVE, "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["vendor_response_state"] == "patch_available"

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=[])
    def test_json_state_is_unknown_with_no_data(self, mock_nvd, mock_cisa, mock_vc, capsys):
        from manus_agent import cli

        cli._run_vendor_response([_CVE, "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["vendor_response_state"] == "unknown"

    @mock.patch(
        "manus_agent.tools.track_vendor_response._fetch_vulncheck_kev",
        return_value={
            "cveID": _CVE_UPPER,
            "ransomwareUse": True,
        },
    )
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=[])
    def test_json_vulncheck_kev_hit_recorded(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """VulnCheck KEV hit is recorded in signals."""
        from manus_agent import cli

        cli._run_vendor_response([_CVE, "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["signals"]["vulncheck_kev_hit"] is True

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_json_nvd_references_count(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """nvd_references_found reflects the actual count."""
        from manus_agent import cli

        cli._run_vendor_response([_CVE, "--output", "json"])
        data = json.loads(capsys.readouterr().out)
        assert data["signals"]["nvd_references_found"] == len(_MOCK_REFS)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestVendorResponseInputValidation:
    def test_invalid_cve_id_exits_nonzero(self, capsys):
        from manus_agent import cli

        with pytest.raises(SystemExit) as exc_info:
            cli._run_vendor_response(["not-a-cve"])
        assert exc_info.value.code != 0

    def test_cve_id_case_insensitive(self, capsys):
        """Lowercase CVE id is accepted and normalised to uppercase."""
        from manus_agent import cli

        with (
            mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=[]),
            mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={}),
            mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={}),
        ):
            rc = cli._run_vendor_response(["cve-2024-3094", "--output", "json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data["cve_id"] == "CVE-2024-3094"


# ---------------------------------------------------------------------------
# CLI dispatch routing
# ---------------------------------------------------------------------------


class TestVendorResponseDispatch:
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_main_routes_to_vendor_response(self, mock_nvd, mock_cisa, mock_vc):
        from manus_agent import cli

        captured: dict = {}

        def fake_run(argv: list) -> int:
            captured["called"] = True
            captured["argv"] = argv
            return 0

        with mock.patch.object(cli, "_run_vendor_response", side_effect=fake_run):
            with mock.patch.object(sys, "argv", ["manus-agent", "vendor-response", _CVE]):
                with pytest.raises(SystemExit) as exc_info:
                    cli.main()
        assert exc_info.value.code == 0
        assert captured.get("called") is True

    def test_vendor_response_in_subcommands_set(self):
        from manus_agent import cli

        assert "vendor-response" in cli._SUBCOMMANDS

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=[])
    def test_main_vendor_response_does_not_route_to_single_shot(self, mock_nvd, mock_cisa, mock_vc):
        from manus_agent import cli

        with mock.patch.object(cli, "_run_single_shot") as m_shot:
            with mock.patch.object(sys, "argv", ["manus-agent", "vendor-response", _CVE]):
                with pytest.raises(SystemExit):
                    cli.main()
        m_shot.assert_not_called()

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_full_e2e_text_output(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """Full end-to-end: sys.argv → main() → _run_vendor_response → stdout."""
        from manus_agent import cli

        with mock.patch.object(sys, "argv", ["manus-agent", "vendor-response", _CVE]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert _CVE_UPPER in out
        assert "Status" in out

    @mock.patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value=_EMPTY_VC_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value=_MOCK_CISA_KEV)
    @mock.patch("manus_agent.tools.track_vendor_response._fetch_nvd_references", return_value=_MOCK_REFS)
    def test_full_e2e_json_output(self, mock_nvd, mock_cisa, mock_vc, capsys):
        """Full end-to-end: sys.argv → main() → stdout (json)."""
        from manus_agent import cli

        with mock.patch.object(sys, "argv", ["manus-agent", "vendor-response", _CVE, "--output", "json"]):
            with pytest.raises(SystemExit) as exc_info:
                cli.main()
        assert exc_info.value.code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["cve_id"] == _CVE_UPPER
        assert "vendor_response_state" in data
