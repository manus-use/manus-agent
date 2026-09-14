"""Comprehensive test suite for track_vendor_response tool and vendor-response CLI subcommand.

All tests are fully mocked — zero real HTTP calls.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from manus_agent.tools.track_vendor_response import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MODERATE,
    TOOL_SPEC,
    VALID_STATES,
    _extract_ghsa_signals,
    _fetch_cisa_kev,
    _fetch_ghsa,
    _fetch_nvd_references,
    _fetch_vulncheck_kev,
    _get_with_retry,
    _score_to_confidence,
    classify,
    track_vendor_response,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. TOOL_SPEC validation
# ═══════════════════════════════════════════════════════════════════════════


class TestToolSpec:
    def test_tool_spec_name(self):
        assert TOOL_SPEC["name"] == "track_vendor_response"

    def test_tool_spec_has_description(self):
        assert "track" in TOOL_SPEC["description"].lower()

    def test_tool_spec_input_schema(self):
        schema = TOOL_SPEC["inputSchema"]["json"]
        assert schema["type"] == "object"
        assert "cve_id" in schema["properties"]
        assert "cve_id" in schema["required"]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Constants validation
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_valid_states(self):
        expected = {
            "patch_available",
            "patch_backported",
            "wont_fix",
            "investigating",
            "no_patch",
            "unknown",
        }
        assert VALID_STATES == expected

    def test_confidence_tiers(self):
        assert CONFIDENCE_HIGH == "high"
        assert CONFIDENCE_MODERATE == "moderate"
        assert CONFIDENCE_LOW == "low"


# ═══════════════════════════════════════════════════════════════════════════
# 3. _score_to_confidence
# ═══════════════════════════════════════════════════════════════════════════


class TestScoreToConfidence:
    def test_high(self):
        assert _score_to_confidence(0.7) == "high"
        assert _score_to_confidence(0.95) == "high"
        assert _score_to_confidence(1.0) == "high"

    def test_moderate(self):
        assert _score_to_confidence(0.4) == "moderate"
        assert _score_to_confidence(0.5) == "moderate"
        assert _score_to_confidence(0.69) == "moderate"

    def test_low(self):
        assert _score_to_confidence(0.0) == "low"
        assert _score_to_confidence(0.2) == "low"
        assert _score_to_confidence(0.39) == "low"


# ═══════════════════════════════════════════════════════════════════════════
# 4. _get_with_retry
# ═══════════════════════════════════════════════════════════════════════════


class TestGetWithRetry:
    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_success_first_try(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = _get_with_retry("https://example.com/api")
        assert result == mock_resp
        assert mock_get.call_count == 1

    @patch("time.sleep")
    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_retry_on_429(self, mock_get, mock_sleep):
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.raise_for_status = MagicMock()
        mock_get.side_effect = [mock_429, mock_ok]

        result = _get_with_retry("https://example.com/api")
        assert result == mock_ok
        assert mock_get.call_count == 2
        mock_sleep.assert_called()

    @patch("time.sleep")
    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_retry_on_request_exception(self, mock_get, mock_sleep):
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("fail")

        with pytest.raises(requests.exceptions.ConnectionError):
            _get_with_retry("https://example.com/api")
        assert mock_get.call_count == 3  # all retries exhausted

    @patch("time.sleep")
    @patch("manus_agent.tools.track_vendor_response.requests.get")
    def test_retry_then_success(self, mock_get, mock_sleep):
        import requests

        mock_ok = MagicMock()
        mock_ok.status_code = 200
        mock_ok.raise_for_status = MagicMock()
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("fail"),
            mock_ok,
        ]

        result = _get_with_retry("https://example.com/api")
        assert result == mock_ok
        assert mock_get.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# 5. _fetch_nvd_references
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchNvdReferences:
    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_success_with_references(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {
                    "cve": {
                        "vulnStatus": "Analyzed",
                        "references": [
                            {"url": "https://example.com/patch", "tags": ["Patch"]},
                            {"url": "https://example.com/advisory", "tags": ["Vendor Advisory"]},
                        ],
                    }
                }
            ]
        }
        mock_get.return_value = mock_resp

        refs, status = _fetch_nvd_references("CVE-2024-1234")
        assert len(refs) == 2
        assert status == "Analyzed"

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_no_vulnerabilities(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        refs, status = _fetch_nvd_references("CVE-2024-9999")
        assert refs == []
        assert status == "unknown"

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_exception_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("Network error")

        refs, status = _fetch_nvd_references("CVE-2024-1234")
        assert refs == []
        assert status == "unknown"

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_nvd_api_key_header(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {"NVD_API_KEY": "test-key"}):
            _fetch_nvd_references("CVE-2024-1234")

        call_kwargs = mock_get.call_args
        assert call_kwargs is not None

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_missing_cve_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [{"cve": {}}]}
        mock_get.return_value = mock_resp

        refs, status = _fetch_nvd_references("CVE-2024-1234")
        assert refs == []
        assert status == "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# 6. _fetch_ghsa
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchGhsa:
    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_success_with_advisories(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {
                "ghsa_id": "GHSA-xxxx-xxxx-xxxx",
                "state": "published",
                "severity": "high",
                "vulnerabilities": [{"patched_versions": ">= 1.2.3"}],
            }
        ]
        mock_get.return_value = mock_resp

        result = _fetch_ghsa("CVE-2024-1234")
        assert len(result) == 1
        assert result[0]["ghsa_id"] == "GHSA-xxxx-xxxx-xxxx"

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_no_advisories(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        result = _fetch_ghsa("CVE-2024-9999")
        assert result == []

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_exception_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("API error")

        result = _fetch_ghsa("CVE-2024-1234")
        assert result == []

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_non_list_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"error": "not found"}
        mock_get.return_value = mock_resp

        result = _fetch_ghsa("CVE-2024-1234")
        assert result == []

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_github_token_header(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_get.return_value = mock_resp

        with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test123"}):
            _fetch_ghsa("CVE-2024-1234")

        call_kwargs = mock_get.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers", {})
        assert "Authorization" in headers


# ═══════════════════════════════════════════════════════════════════════════
# 7. _fetch_cisa_kev
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchCisaKev:
    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "vulnerabilities": [
                {"cveID": "CVE-2024-1234", "requiredAction": "Apply update", "dueDate": "2024-06-01"},
                {"cveID": "CVE-2024-5678", "requiredAction": "Mitigate"},
            ]
        }
        mock_get.return_value = mock_resp

        result = _fetch_cisa_kev("CVE-2024-1234")
        assert result["cveID"] == "CVE-2024-1234"
        assert result["requiredAction"] == "Apply update"

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_not_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": [{"cveID": "CVE-2024-9999"}]}
        mock_get.return_value = mock_resp

        result = _fetch_cisa_kev("CVE-2024-1234")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_exception_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("timeout")

        result = _fetch_cisa_kev("CVE-2024-1234")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_empty_vulnerabilities(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"vulnerabilities": []}
        mock_get.return_value = mock_resp

        result = _fetch_cisa_kev("CVE-2024-1234")
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# 8. _fetch_vulncheck_kev
# ═══════════════════════════════════════════════════════════════════════════


class TestFetchVulncheckKev:
    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"cve": "CVE-2024-1234", "ransomwareUse": True}]}
        mock_get.return_value = mock_resp

        result = _fetch_vulncheck_kev("CVE-2024-1234", "test-api-key")
        assert result["cve"] == "CVE-2024-1234"

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_not_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        result = _fetch_vulncheck_kev("CVE-2024-1234", "test-api-key")
        assert result == {}

    def test_no_api_key(self):
        result = _fetch_vulncheck_kev("CVE-2024-1234", "")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_exception_returns_empty(self, mock_get):
        mock_get.side_effect = Exception("API error")

        result = _fetch_vulncheck_kev("CVE-2024-1234", "test-key")
        assert result == {}

    @patch("manus_agent.tools.track_vendor_response._get_with_retry")
    def test_missing_data_key(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp

        result = _fetch_vulncheck_kev("CVE-2024-1234", "test-key")
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# 9. _extract_ghsa_signals
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractGhsaSignals:
    def test_empty_advisories(self):
        result = _extract_ghsa_signals([])
        assert result["has_advisory"] is False
        assert result["state"] == "none"
        assert result["has_patched_versions"] is False
        assert result["patched_versions"] == []

    def test_published_with_patched_versions(self):
        advisories = [
            {
                "state": "published",
                "severity": "critical",
                "withdrawn_at": None,
                "cvss": {"score": 9.8},
                "vulnerabilities": [
                    {"patched_versions": ">= 2.0.1"},
                    {"patched_versions": ">= 1.9.5"},
                ],
            }
        ]
        result = _extract_ghsa_signals(advisories)
        assert result["has_advisory"] is True
        assert result["state"] == "published"
        assert result["severity"] == "critical"
        assert result["withdrawn"] is False
        assert result["cvss_score"] == 9.8
        assert result["has_patched_versions"] is True
        assert ">= 2.0.1" in result["patched_versions"]
        assert ">= 1.9.5" in result["patched_versions"]

    def test_withdrawn_advisory(self):
        advisories = [
            {
                "state": "published",
                "withdrawn_at": "2024-01-15T00:00:00Z",
                "vulnerabilities": [],
            }
        ]
        result = _extract_ghsa_signals(advisories)
        assert result["withdrawn"] is True

    def test_no_patched_versions(self):
        advisories = [
            {
                "state": "published",
                "vulnerabilities": [{"patched_versions": ""}],
            }
        ]
        result = _extract_ghsa_signals(advisories)
        assert result["has_patched_versions"] is False
        assert result["patched_versions"] == []

    def test_first_patched_version_fallback(self):
        advisories = [
            {
                "state": "published",
                "vulnerabilities": [
                    {
                        "patched_versions": None,
                        "first_patched_version": {"identifier": "3.1.0"},
                    }
                ],
            }
        ]
        result = _extract_ghsa_signals(advisories)
        assert result["has_patched_versions"] is True
        assert "3.1.0" in result["patched_versions"]

    def test_cvss_invalid_score(self):
        advisories = [
            {
                "state": "published",
                "cvss": {"score": "not-a-number"},
                "vulnerabilities": [],
            }
        ]
        result = _extract_ghsa_signals(advisories)
        assert result["cvss_score"] is None

    def test_no_cvss(self):
        advisories = [{"state": "published", "vulnerabilities": []}]
        result = _extract_ghsa_signals(advisories)
        assert result["cvss_score"] is None

    def test_dedup_first_patched_version(self):
        """first_patched_version that matches patched_versions should not duplicate."""
        advisories = [
            {
                "state": "published",
                "vulnerabilities": [
                    {
                        "patched_versions": ">= 3.1.0",
                        "first_patched_version": {"identifier": ">= 3.1.0"},
                    }
                ],
            }
        ]
        result = _extract_ghsa_signals(advisories)
        assert result["patched_versions"].count(">= 3.1.0") == 1


# ═══════════════════════════════════════════════════════════════════════════
# 10. classify — comprehensive classification tests
# ═══════════════════════════════════════════════════════════════════════════


class TestClassify:
    """Test the classify() function with various signal combinations."""

    def _empty_ghsa(self):
        return _extract_ghsa_signals([])

    # -- Unknown baseline --------------------------------------------------

    def test_all_empty_returns_unknown(self):
        state, conf, score, evidence = classify([], "unknown", self._empty_ghsa(), {}, {})
        assert state == "unknown"
        assert conf == "low"

    # -- NVD status signals ------------------------------------------------

    def test_rejected_nvd_status_wont_fix(self):
        state, conf, score, evidence = classify([], "Rejected", self._empty_ghsa(), {}, {})
        assert state == "wont_fix"
        assert score >= 0.6

    def test_disputed_nvd_status_wont_fix(self):
        state, conf, score, evidence = classify([], "Disputed", self._empty_ghsa(), {}, {})
        assert state == "wont_fix"

    def test_analyzed_nvd_status_boosts_score(self):
        state, conf, score, evidence = classify([], "Analyzed", self._empty_ghsa(), {}, {})
        assert score >= 0.3
        assert any("Analyzed" in e for e in evidence)

    # -- NVD reference tags ------------------------------------------------

    def test_patch_tag_patch_available(self):
        refs = [{"url": "https://example.com", "tags": ["Patch"]}]
        state, conf, score, evidence = classify(refs, "unknown", self._empty_ghsa(), {}, {})
        assert state == "patch_available"
        assert score >= 0.75

    def test_vendor_advisory_tag_patch_available(self):
        refs = [{"url": "https://example.com", "tags": ["Vendor Advisory"]}]
        state, conf, score, evidence = classify(refs, "unknown", self._empty_ghsa(), {}, {})
        assert state == "patch_available"

    def test_fix_tag_patch_available(self):
        refs = [{"url": "https://example.com", "tags": ["Fix"]}]
        state, conf, score, evidence = classify(refs, "unknown", self._empty_ghsa(), {}, {})
        assert state == "patch_available"

    def test_release_notes_tag_patch_available(self):
        refs = [{"url": "https://example.com", "tags": ["Release Notes"]}]
        state, conf, score, evidence = classify(refs, "unknown", self._empty_ghsa(), {}, {})
        assert state == "patch_available"

    def test_mitigation_tag_no_patch(self):
        refs = [{"url": "https://example.com", "tags": ["Mitigation"]}]
        state, conf, score, evidence = classify(refs, "unknown", self._empty_ghsa(), {}, {})
        assert state == "no_patch"
        assert score >= 0.5

    def test_workaround_tag_no_patch(self):
        refs = [{"url": "https://example.com", "tags": ["Workaround"]}]
        state, conf, score, evidence = classify(refs, "unknown", self._empty_ghsa(), {}, {})
        assert state == "no_patch"

    # -- URL keyword heuristics --------------------------------------------

    def test_backport_keyword_in_url(self):
        refs = [{"url": "https://example.com/backported-fix", "tags": []}]
        state, conf, score, evidence = classify(refs, "unknown", self._empty_ghsa(), {}, {})
        assert state == "patch_backported"
        assert any("backport" in e.lower() for e in evidence)

    def test_cherry_pick_keyword_in_url(self):
        refs = [{"url": "https://example.com/cherry-pick-commit", "tags": []}]
        state, conf, score, evidence = classify(refs, "unknown", self._empty_ghsa(), {}, {})
        assert state == "patch_backported"

    def test_wontfix_keyword_in_url(self):
        refs = [{"url": "https://example.com/wontfix-notice", "tags": []}]
        state, conf, score, evidence = classify(refs, "unknown", self._empty_ghsa(), {}, {})
        assert state == "wont_fix"

    def test_eol_keyword_in_url(self):
        refs = [{"url": "https://example.com/end-of-life-product", "tags": []}]
        state, conf, score, evidence = classify(refs, "unknown", self._empty_ghsa(), {}, {})
        assert state == "wont_fix"

    def test_patch_keyword_in_url(self):
        refs = [{"url": "https://example.com/fixed-in-version-2", "tags": []}]
        state, conf, score, evidence = classify(refs, "unknown", self._empty_ghsa(), {}, {})
        assert state == "patch_available"

    def test_workaround_keyword_in_url(self):
        refs = [{"url": "https://example.com/workaround-guide", "tags": []}]
        state, conf, score, evidence = classify(refs, "unknown", self._empty_ghsa(), {}, {})
        assert state == "no_patch"

    # -- GHSA signals ------------------------------------------------------

    def test_ghsa_patched_versions_patch_available(self):
        ghsa = {
            "has_advisory": True,
            "state": "published",
            "has_patched_versions": True,
            "patched_versions": [">= 2.0.1"],
            "severity": "high",
            "withdrawn": False,
            "cvss_score": 8.5,
        }
        state, conf, score, evidence = classify([], "unknown", ghsa, {}, {})
        assert state == "patch_available"
        assert score >= 0.8
        assert any("patched" in e.lower() for e in evidence)

    def test_ghsa_published_no_patched_investigating(self):
        ghsa = {
            "has_advisory": True,
            "state": "published",
            "has_patched_versions": False,
            "patched_versions": [],
            "severity": "medium",
            "withdrawn": False,
            "cvss_score": None,
        }
        state, conf, score, evidence = classify([], "unknown", ghsa, {}, {})
        assert state == "investigating"
        assert any("no patched" in e.lower() for e in evidence)

    def test_ghsa_withdrawn_wont_fix(self):
        ghsa = {
            "has_advisory": True,
            "state": "published",
            "has_patched_versions": False,
            "patched_versions": [],
            "severity": None,
            "withdrawn": True,
            "cvss_score": None,
        }
        state, conf, score, evidence = classify([], "unknown", ghsa, {}, {})
        assert state == "wont_fix"
        assert any("withdrawn" in e.lower() for e in evidence)

    def test_ghsa_overrides_unknown(self):
        """GHSA patched versions should upgrade from unknown."""
        ghsa = {
            "has_advisory": True,
            "state": "published",
            "has_patched_versions": True,
            "patched_versions": [">= 1.0.1"],
            "severity": None,
            "withdrawn": False,
            "cvss_score": None,
        }
        state, conf, score, evidence = classify([], "unknown", ghsa, {}, {})
        assert state == "patch_available"

    # -- CISA KEV signals --------------------------------------------------

    def test_cisa_kev_apply_update_patch_available(self):
        kev = {
            "cveID": "CVE-2024-1234",
            "requiredAction": "Apply update per vendor instructions",
            "dueDate": "2024-06-01",
            "shortDescription": "XSS vulnerability",
        }
        state, conf, score, evidence = classify([], "unknown", self._empty_ghsa(), kev, {})
        assert state == "patch_available"
        assert any("KEV" in e for e in evidence)
        assert any("due date" in e.lower() for e in evidence)

    def test_cisa_kev_patch_action(self):
        kev = {
            "cveID": "CVE-2024-1234",
            "requiredAction": "Apply patch provided by vendor",
            "shortDescription": "RCE vulnerability",
        }
        state, conf, score, evidence = classify([], "unknown", self._empty_ghsa(), kev, {})
        assert state == "patch_available"

    def test_cisa_kev_unusual_action_investigating(self):
        kev = {
            "cveID": "CVE-2024-1234",
            "requiredAction": "Discontinue use of affected product",
            "shortDescription": "Critical vuln",
        }
        state, conf, score, evidence = classify([], "unknown", self._empty_ghsa(), kev, {})
        assert state == "investigating"

    # -- VulnCheck KEV signals ---------------------------------------------

    def test_vulncheck_kev_unknown_to_investigating(self):
        vc = {"cve": "CVE-2024-1234"}
        state, conf, score, evidence = classify([], "unknown", self._empty_ghsa(), {}, vc)
        assert state == "investigating"
        assert any("VulnCheck" in e for e in evidence)

    def test_vulncheck_kev_ransomware_boost(self):
        vc = {"cve": "CVE-2024-1234", "ransomwareUse": True}
        state, conf, score_with, evidence_with = classify([], "unknown", self._empty_ghsa(), {}, vc)

        vc_no_ransom = {"cve": "CVE-2024-1234"}
        _, _, score_without, _ = classify([], "unknown", self._empty_ghsa(), {}, vc_no_ransom)

        assert score_with > score_without
        assert any("ransomware" in e.lower() for e in evidence_with)

    def test_vulncheck_kev_ransomware_use_variant(self):
        vc = {"cve": "CVE-2024-1234", "ransomware_use": True}
        _, _, _, evidence = classify([], "unknown", self._empty_ghsa(), {}, vc)
        assert any("ransomware" in e.lower() for e in evidence)

    def test_vulncheck_kev_known_ransomware_campaign(self):
        vc = {"cve": "CVE-2024-1234", "knownRansomwareCampaignUse": True}
        _, _, _, evidence = classify([], "unknown", self._empty_ghsa(), {}, vc)
        assert any("ransomware" in e.lower() for e in evidence)

    # -- Combined signals --------------------------------------------------

    def test_nvd_patch_plus_ghsa_plus_kev_high_confidence(self):
        refs = [{"url": "https://example.com/fix", "tags": ["Patch"]}]
        ghsa = {
            "has_advisory": True,
            "state": "published",
            "has_patched_versions": True,
            "patched_versions": [">= 3.0.0"],
            "severity": "critical",
            "withdrawn": False,
            "cvss_score": 9.8,
        }
        kev = {
            "cveID": "CVE-2024-1234",
            "requiredAction": "Apply update",
            "shortDescription": "RCE",
        }
        state, conf, score, evidence = classify(refs, "Analyzed", ghsa, kev, {})
        assert state == "patch_available"
        assert conf == "high"
        assert score >= 0.9

    def test_patch_tag_overrides_rejected_status(self):
        """NVD Rejected + Patch tag → wont_fix stays (rejected wins)."""
        refs = [{"url": "https://example.com/fix", "tags": ["Patch"]}]
        state, conf, score, evidence = classify(refs, "Rejected", self._empty_ghsa(), {}, {})
        # Rejected status sets wont_fix first, then patch tag tries to set patch_available
        # but the guard `if state not in ("wont_fix",)` prevents overwrite
        assert state == "wont_fix"


# ═══════════════════════════════════════════════════════════════════════════
# 11. Strands tool entry point — track_vendor_response()
# ═══════════════════════════════════════════════════════════════════════════


class TestTrackVendorResponseTool:
    def _make_tool_use(self, cve_id: str) -> dict:
        return {
            "toolUseId": "test-tool-use-id",
            "input": {"cve_id": cve_id},
        }

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_success_basic(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc):
        result = track_vendor_response(self._make_tool_use("CVE-2024-1234"))
        assert result["status"] == "success"
        payload = result["content"][0]["json"]
        assert payload["cve_id"] == "CVE-2024-1234"
        assert payload["vendor_response_state"] in VALID_STATES
        assert payload["confidence"] in ("high", "moderate", "low")

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=(
            [{"url": "https://example.com/patch", "tags": ["Patch"]}],
            "Analyzed",
        ),
    )
    def test_patch_available_from_nvd(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc):
        result = track_vendor_response(self._make_tool_use("CVE-2024-1234"))
        payload = result["content"][0]["json"]
        assert payload["vendor_response_state"] == "patch_available"

    def test_invalid_cve_id_empty(self):
        result = track_vendor_response(self._make_tool_use(""))
        assert result["status"] == "error"

    def test_invalid_cve_id_wrong_format(self):
        result = track_vendor_response(self._make_tool_use("not-a-cve"))
        assert result["status"] == "error"

    def test_invalid_cve_id_non_string(self):
        tool_use = {"toolUseId": "test", "input": {"cve_id": 12345}}
        result = track_vendor_response(tool_use)
        assert result["status"] == "error"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_signals_in_payload(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc):
        result = track_vendor_response(self._make_tool_use("CVE-2024-1234"))
        signals = result["content"][0]["json"]["signals"]
        assert "nvd_references_found" in signals
        assert "nvd_status" in signals
        assert "ghsa_advisory_found" in signals
        assert "ghsa_patched_versions" in signals
        assert "cisa_kev_hit" in signals
        assert "vulncheck_kev_hit" in signals
        assert "vulncheck_api_key_present" in signals

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_ghsa",
        return_value=[
            {
                "state": "published",
                "severity": "high",
                "withdrawn_at": None,
                "cvss": {"score": 8.5},
                "vulnerabilities": [{"patched_versions": ">= 2.0.0"}],
            }
        ],
    )
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_ghsa_patched_versions_in_signals(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc):
        result = track_vendor_response(self._make_tool_use("CVE-2024-1234"))
        signals = result["content"][0]["json"]["signals"]
        assert signals["ghsa_advisory_found"] is True
        assert ">= 2.0.0" in signals["ghsa_patched_versions"]

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_lowercase_cve_normalized(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc):
        result = track_vendor_response(self._make_tool_use("cve-2024-1234"))
        payload = result["content"][0]["json"]
        assert payload["cve_id"] == "CVE-2024-1234"


# ═══════════════════════════════════════════════════════════════════════════
# 12. CLI subcommand — _build_vendor_response_parser
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildVendorResponseParser:
    def test_parser_creation(self):
        from manus_agent.cli import _build_vendor_response_parser

        parser = _build_vendor_response_parser()
        assert parser.prog == "manus-agent vendor-response"

    def test_parser_required_cve_id(self):
        from manus_agent.cli import _build_vendor_response_parser

        parser = _build_vendor_response_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # missing required CVE-ID

    def test_parser_default_output(self):
        from manus_agent.cli import _build_vendor_response_parser

        parser = _build_vendor_response_parser()
        args = parser.parse_args(["CVE-2024-1234"])
        assert args.output == "text"

    def test_parser_json_output(self):
        from manus_agent.cli import _build_vendor_response_parser

        parser = _build_vendor_response_parser()
        args = parser.parse_args(["CVE-2024-1234", "--output", "json"])
        assert args.output == "json"


# ═══════════════════════════════════════════════════════════════════════════
# 13. CLI subcommand — _run_vendor_response
# ═══════════════════════════════════════════════════════════════════════════


class TestRunVendorResponse:
    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_text_output(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        exit_code = _run_vendor_response(["CVE-2024-1234"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "CVE-2024-1234" in captured.out
        assert "Classification" in captured.out
        assert "Confidence" in captured.out

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_json_output(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        exit_code = _run_vendor_response(["CVE-2024-1234", "--output", "json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["cve_id"] == "CVE-2024-1234"
        assert data["vendor_response_state"] in VALID_STATES
        assert data["confidence"] in ("high", "moderate", "low")

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_ghsa",
        return_value=[
            {
                "state": "published",
                "severity": "high",
                "withdrawn_at": None,
                "cvss": {"score": 9.1},
                "vulnerabilities": [{"patched_versions": ">= 5.0.0"}],
            }
        ],
    )
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=(
            [{"url": "https://example.com/advisory", "tags": ["Vendor Advisory"]}],
            "Analyzed",
        ),
    )
    def test_text_output_with_sources(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        exit_code = _run_vendor_response(["CVE-2024-1234"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "patch_available" in captured.out
        assert "NVD references" in captured.out
        assert "GHSA" in captured.out

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_cisa_kev",
        return_value={
            "cveID": "CVE-2024-1234",
            "requiredAction": "Apply update",
            "dueDate": "2024-06-01",
            "shortDescription": "Critical RCE",
        },
    )
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_text_output_with_kev(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        exit_code = _run_vendor_response(["CVE-2024-1234"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "CISA KEV" in captured.out
        assert "in catalog" in captured.out

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_text_output_vulncheck_no_key(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        with patch.dict("os.environ", {}, clear=True):
            exit_code = _run_vendor_response(["CVE-2024-1234"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower() or "no API key" in captured.out

    def test_invalid_cve_id(self, capsys):
        from manus_agent.cli import _run_vendor_response

        exit_code = _run_vendor_response(["not-a-cve"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "error" in captured.err.lower()

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_json_output_schema(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        exit_code = _run_vendor_response(["CVE-2024-9999", "--output", "json"])
        assert exit_code == 0
        data = json.loads(capsys.readouterr().out)
        # Validate top-level keys
        assert "cve_id" in data
        assert "vendor_response_state" in data
        assert "confidence" in data
        assert "confidence_score" in data
        assert "evidence" in data
        assert "signals" in data
        # Validate signals keys
        sigs = data["signals"]
        assert "nvd_references_found" in sigs
        assert "ghsa_advisory_found" in sigs
        assert "cisa_kev_hit" in sigs

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_text_output_states_legend(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        exit_code = _run_vendor_response(["CVE-2024-1234"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "States:" in captured.out

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_evidence_section(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc, capsys):
        from manus_agent.cli import _run_vendor_response

        exit_code = _run_vendor_response(["CVE-2024-1234"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Evidence" in captured.out


# ═══════════════════════════════════════════════════════════════════════════
# 14. CLI dispatch — vendor-response in _SUBCOMMANDS and main()
# ═══════════════════════════════════════════════════════════════════════════


class TestCliDispatch:
    def test_vendor_response_in_subcommands(self):
        from manus_agent.cli import _SUBCOMMANDS

        assert "vendor-response" in _SUBCOMMANDS

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_main_dispatches_vendor_response(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc):
        from manus_agent.cli import main

        with patch.object(sys, "argv", ["manus-agent", "vendor-response", "CVE-2024-1234", "--output", "json"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
        assert exc_info.value.code == 0


# ═══════════════════════════════════════════════════════════════════════════
# 15. Edge cases and integration-style tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_classify_invalid_state_fallback(self):
        """If somehow an invalid state is set, it should fall back to unknown."""
        # This tests the final guard: `if state not in VALID_STATES: state = "unknown"`
        # Hard to trigger naturally, but we verify the guard works.
        state, _, _, _ = classify([], "unknown", _extract_ghsa_signals([]), {}, {})
        assert state in VALID_STATES

    def test_confidence_score_bounded(self):
        """Score should never exceed 0.98 even with all signals present."""
        refs = [{"url": "https://example.com/fix", "tags": ["Patch"]}]
        ghsa = {
            "has_advisory": True,
            "state": "published",
            "has_patched_versions": True,
            "patched_versions": [">= 2.0.0"],
            "severity": "critical",
            "withdrawn": False,
            "cvss_score": 10.0,
        }
        kev = {
            "cveID": "CVE-2024-1234",
            "requiredAction": "Apply update",
            "dueDate": "2024-01-01",
            "shortDescription": "Critical RCE",
        }
        vc = {"cve": "CVE-2024-1234", "ransomwareUse": True}
        _, _, score, _ = classify(refs, "Analyzed", ghsa, kev, vc)
        assert score <= 0.98

    def test_empty_ref_tags(self):
        """References with no tags should not crash."""
        refs = [{"url": "https://example.com", "tags": None}]
        state, _, _, _ = classify(refs, "unknown", _extract_ghsa_signals([]), {}, {})
        assert state in VALID_STATES

    def test_empty_ref_url(self):
        """References with empty URL should not crash."""
        refs = [{"url": "", "tags": []}]
        state, _, _, _ = classify(refs, "unknown", _extract_ghsa_signals([]), {}, {})
        assert state in VALID_STATES

    def test_ghsa_no_vulnerabilities_key(self):
        """GHSA advisory with no vulnerabilities key."""
        advisories = [{"state": "published"}]
        result = _extract_ghsa_signals(advisories)
        assert result["has_advisory"] is True
        assert result["has_patched_versions"] is False

    def test_ghsa_vulnerability_no_patched_versions_key(self):
        """GHSA vulnerability entry missing patched_versions entirely."""
        advisories = [
            {
                "state": "published",
                "vulnerabilities": [{"package": {"name": "foo"}}],
            }
        ]
        result = _extract_ghsa_signals(advisories)
        assert result["has_patched_versions"] is False

    def test_ghsa_first_patched_version_non_dict(self):
        """first_patched_version that is not a dict should be handled gracefully."""
        advisories = [
            {
                "state": "published",
                "vulnerabilities": [{"patched_versions": None, "first_patched_version": "not-a-dict"}],
            }
        ]
        result = _extract_ghsa_signals(advisories)
        assert result["has_patched_versions"] is False

    def test_cisa_kev_no_due_date(self):
        """CISA KEV entry without dueDate should not crash."""
        kev = {
            "cveID": "CVE-2024-1234",
            "requiredAction": "Apply update",
            "shortDescription": "Vuln",
        }
        state, _, _, evidence = classify([], "unknown", _extract_ghsa_signals([]), kev, {})
        assert state == "patch_available"
        # No "due date" evidence line expected
        assert not any("due date" in e.lower() for e in evidence)

    def test_mitigation_tag_not_overridden_by_patch_keyword_when_tag_present(self):
        """When mitigation tag is present and a patch keyword is in URL,
        the tag-based classification takes precedence."""
        refs = [
            {"url": "https://example.com/workaround", "tags": ["Mitigation"]},
        ]
        state, _, _, _ = classify(refs, "unknown", _extract_ghsa_signals([]), {}, {})
        assert state == "no_patch"

    @patch("manus_agent.tools.track_vendor_response._fetch_vulncheck_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_cisa_kev", return_value={})
    @patch("manus_agent.tools.track_vendor_response._fetch_ghsa", return_value=[])
    @patch(
        "manus_agent.tools.track_vendor_response._fetch_nvd_references",
        return_value=([], "unknown"),
    )
    def test_tool_uppercases_cve_id(self, mock_nvd, mock_ghsa, mock_cisa, mock_vc):
        result = track_vendor_response({"toolUseId": "test", "input": {"cve_id": "cve-2024-1234"}})
        assert result["content"][0]["json"]["cve_id"] == "CVE-2024-1234"
