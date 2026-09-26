#!/usr/bin/env python3
"""
Tool for tracking and classifying vendor patch/response status for a CVE.

Queries multiple public sources (NVD, CISA KEV, VulnCheck KEV) to produce a
6-state vendor response classification:

  patch_available     — A confirmed fix/patch has been released.
  patch_pending       — Vendor acknowledged; fix in progress or announced.
  workaround_only     — Vendor published a mitigation but no patch yet.
  investigating       — Vendor acknowledged but response status is unclear.
  no_patch_expected   — Vendor will not fix (EoL, won't-fix, disputed).
  unknown             — Insufficient data to classify.

VulnCheck KEV integration: a KEV hit elevates classification confidence and
can push the state toward ``patch_available`` or ``investigating`` depending on
what the NVD/vendor data shows — because actively exploited CVEs almost always
trigger a vendor response.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size
from manus_agent.utils.cve_utils import cve_validation_error

# 6 valid classification states.
_VALID_STATES = frozenset(
    {
        "patch_available",
        "patch_pending",
        "workaround_only",
        "investigating",
        "no_patch_expected",
        "unknown",
    }
)

# Keywords that strongly suggest a patch exists.
_PATCH_KEYWORDS = frozenset(
    {
        "fixed in",
        "patch",
        "update to",
        "upgrade to",
        "version",
        "release",
        "resolved",
        "remediated",
        "hotfix",
    }
)

# Keywords that suggest only a workaround.
_WORKAROUND_KEYWORDS = frozenset(
    {
        "workaround",
        "mitigation",
        "disable",
        "restrict",
        "block",
        "firewall rule",
        "configuration change",
    }
)

TOOL_SPEC = {
    "name": "track_vendor_response",
    "description": (
        "Tracks and classifies the vendor patch/response status for a given CVE ID. "
        "Queries NVD, CISA KEV, and VulnCheck KEV to produce a 6-state classification: "
        "patch_available, patch_pending, workaround_only, investigating, "
        "no_patch_expected, or unknown. "
        "VulnCheck KEV hits elevate confidence: actively exploited CVEs almost "
        "always have a vendor response. Use after get_nvd_data and get_vulncheck_data "
        "to build a complete remediation picture."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to track (e.g., 'CVE-2024-3094').",
                }
            },
            "required": ["cve_id"],
        }
    },
}


def _fetch_nvd_references(cve_id: str) -> list[dict[str, Any]]:
    """Return the NVD reference list for *cve_id*, or [] on failure."""
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities") or []
        if not vulns:
            return []
        return vulns[0].get("cve", {}).get("references") or []
    except Exception:  # noqa: BLE001
        return []


def _fetch_cisa_kev(cve_id: str) -> dict[str, Any]:
    """Return CISA KEV entry for *cve_id*, or {} if not found."""
    try:
        resp = requests.get(
            "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        for vuln in data.get("vulnerabilities") or []:
            if vuln.get("cveID", "").upper() == cve_id:
                return vuln
    except Exception:  # noqa: BLE001
        pass
    return {}


def _fetch_vulncheck_kev(cve_id: str, api_key: str) -> dict[str, Any]:
    """Return VulnCheck KEV data for *cve_id*, or {} if unavailable/no key."""
    if not api_key:
        return {}
    try:
        url = "https://api.vulncheck.com/v3/index/vulncheck-kev"
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        resp = requests.get(url, headers=headers, params={"cve": cve_id}, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or []
        return data[0] if data else {}
    except Exception:  # noqa: BLE001
        return {}


def _classify(
    references: list[dict[str, Any]],
    cisa_kev: dict[str, Any],
    vulncheck_kev: dict[str, Any],
    nvd_status: str,
) -> tuple[str, float, list[str]]:
    """Derive (state, confidence, evidence_list) from gathered signals.

    Confidence is a float in [0, 1]:
      0.9+ = high  (multiple strong signals agree)
      0.6–0.9 = medium  (at least one strong signal)
      0.3–0.6 = low  (weak / indirect signals only)
      < 0.3 = very low  (essentially guessing)
    """
    evidence: list[str] = []
    state = "unknown"
    confidence = 0.2

    # ── NVD vuln status ──────────────────────────────────────────────────────
    nvd_status_lower = nvd_status.lower()
    if "modified" in nvd_status_lower or "analyzed" in nvd_status_lower:
        evidence.append(f"NVD status: {nvd_status}")
        confidence = max(confidence, 0.3)

    # ── Reference-tag analysis ───────────────────────────────────────────────
    ref_tags_flat: set[str] = set()
    ref_urls: list[str] = []
    for ref in references:
        for tag in ref.get("tags") or []:
            ref_tags_flat.add(tag.lower())
        url = ref.get("url", "")
        if url:
            ref_urls.append(url.lower())

    has_patch_tag = "patch" in ref_tags_flat or "vendor-advisory" in ref_tags_flat
    has_fix_tag = "fix" in ref_tags_flat or "release-notes" in ref_tags_flat
    has_mitigation_tag = "mitigation" in ref_tags_flat or "workaround" in ref_tags_flat

    if has_patch_tag or has_fix_tag:
        state = "patch_available"
        confidence = max(confidence, 0.75)
        evidence.append(f"NVD reference tags include: {sorted(ref_tags_flat)}")

    elif has_mitigation_tag and state == "unknown":
        state = "workaround_only"
        confidence = max(confidence, 0.6)
        evidence.append(f"NVD reference tags include mitigation/workaround: {sorted(ref_tags_flat)}")

    # Heuristic: scan ref URLs for patch/fix keywords.
    url_text = " ".join(ref_urls)
    for kw in _PATCH_KEYWORDS:
        if kw in url_text:
            if state == "unknown":
                state = "patch_available"
                confidence = max(confidence, 0.5)
            evidence.append(f"Patch keyword '{kw}' found in reference URLs.")
            break

    if state == "unknown":
        for kw in _WORKAROUND_KEYWORDS:
            if kw in url_text:
                state = "workaround_only"
                confidence = max(confidence, 0.4)
                evidence.append(f"Workaround keyword '{kw}' found in reference URLs.")
                break

    # ── CISA KEV signal ──────────────────────────────────────────────────────
    if cisa_kev:
        required_action = (cisa_kev.get("requiredAction") or "").lower()
        evidence.append(f"CISA KEV: {cisa_kev.get('shortDescription', 'in KEV catalog')}")
        # CISA often lists "Apply update" — implies patch_available.
        if "apply" in required_action or "update" in required_action or "patch" in required_action:
            if state in ("unknown", "investigating"):
                state = "patch_available"
            confidence = min(confidence + 0.2, 0.95)

    # ── VulnCheck KEV signal ─────────────────────────────────────────────────
    if vulncheck_kev:
        evidence.append("VulnCheck KEV: active exploitation confirmed via multi-source aggregation")
        # Active exploitation + unknown status → bump to investigating at minimum.
        if state == "unknown":
            state = "investigating"
        # Boost confidence: confirmed exploitation means there's strong vendor pressure.
        confidence = min(confidence + 0.15, 0.95)

        ransomware = bool(
            vulncheck_kev.get("ransomwareUse")
            or vulncheck_kev.get("ransomware_use")
            or vulncheck_kev.get("knownRansomwareCampaignUse")
        )
        if ransomware:
            evidence.append("VulnCheck KEV: ransomware association — escalated priority")
            confidence = min(confidence + 0.05, 0.98)

    # ── Final consistency check ───────────────────────────────────────────────
    if state not in _VALID_STATES:
        state = "unknown"

    return state, round(confidence, 3), evidence


def track_vendor_response(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Classify vendor patch/response status for a CVE using NVD + KEV sources."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id: str = tool_input.get("cve_id", "")

    if (err := cve_validation_error(cve_id, tool_use_id)) is not None:
        log_tool_output_size("track_vendor_response", err)
        return err

    cve_id = cve_id.upper()
    api_key = os.environ.get("VULNCHECK_API_KEY", "").strip()

    # Gather data from each source independently (failures are non-fatal).
    references = _fetch_nvd_references(cve_id)
    cisa_kev = _fetch_cisa_kev(cve_id)
    vulncheck_kev = _fetch_vulncheck_kev(cve_id, api_key)

    # Derive NVD vuln status from reference tags if available.
    nvd_status = "unknown"
    if references:
        nvd_status = "analyzed"

    state, confidence, evidence = _classify(references, cisa_kev, vulncheck_kev, nvd_status)

    vulncheck_kev_hit = bool(vulncheck_kev)
    cisa_kev_hit = bool(cisa_kev)

    payload: dict[str, Any] = {
        "cve_id": cve_id,
        "vendor_response_state": state,
        "confidence": confidence,
        "evidence": evidence,
        "signals": {
            "nvd_references_found": len(references),
            "cisa_kev_hit": cisa_kev_hit,
            "vulncheck_kev_hit": vulncheck_kev_hit,
            "vulncheck_api_key_present": bool(api_key),
        },
    }

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"json": payload}],
    }
    log_tool_output_size("track_vendor_response", result)
    return result
