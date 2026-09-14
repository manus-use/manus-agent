#!/usr/bin/env python3
"""
Tool for tracking and classifying vendor patch/response status for a CVE.

Queries multiple public sources (NVD references, GitHub Security Advisories,
CISA KEV, VulnCheck KEV) to produce a 6-state vendor response classification:

  patch_available     — A confirmed fix/patch has been released.
  patch_backported    — A backport fix exists (older/LTS branch patched).
  wont_fix            — Vendor explicitly will not fix (EoL, disputed, won't-fix).
  investigating       — Vendor acknowledged; response status unclear or in progress.
  no_patch            — No patch exists yet; vendor has not committed to a fix.
  unknown             — Insufficient data to classify.

Sources queried (in order):
  1. NVD reference URL tags and patterns
  2. GitHub Security Advisories (GHSA) — published state + patched_versions
  3. CISA KEV — required-action + due-date
  4. VulnCheck KEV (optional, requires VULNCHECK_API_KEY)

Confidence is rated high / moderate / low.
A VulnCheck KEV hit upgrades confidence when VULNCHECK_API_KEY is set.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 6 valid classification states (aligned with README spec).
VALID_STATES = frozenset(
    {
        "patch_available",
        "patch_backported",
        "wont_fix",
        "investigating",
        "no_patch",
        "unknown",
    }
)

# Confidence tiers.
CONFIDENCE_HIGH = "high"
CONFIDENCE_MODERATE = "moderate"
CONFIDENCE_LOW = "low"

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

# Keywords suggesting a backport fix.
_BACKPORT_KEYWORDS = frozenset(
    {
        "backport",
        "backported",
        "cherry-pick",
        "cherry-picked",
        "lts",
        "stable branch",
    }
)

# Keywords that suggest only a workaround / no full patch.
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

# Keywords that suggest vendor won't fix.
_WONTFIX_KEYWORDS = frozenset(
    {
        "won't fix",
        "wontfix",
        "will not fix",
        "end of life",
        "end-of-life",
        "eol",
        "disputed",
        "not a vulnerability",
        "by design",
    }
)

TOOL_SPEC = {
    "name": "track_vendor_response",
    "description": (
        "Tracks and classifies the vendor patch/response status for a given CVE ID. "
        "Queries NVD references, GitHub Security Advisories (GHSA), CISA KEV, and "
        "optionally VulnCheck KEV to produce a 6-state classification: "
        "patch_available, patch_backported, wont_fix, investigating, no_patch, or unknown. "
        "Confidence is rated high/moderate/low. "
        "A VulnCheck KEV hit upgrades confidence when VULNCHECK_API_KEY is set."
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

# ---------------------------------------------------------------------------
# HTTP helpers (retry/back-off)
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BACKOFF = [1.0, 2.0, 4.0]


def _get_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    timeout: int = 15,
) -> requests.Response:
    """GET with simple retry/back-off on transient failures."""
    import time

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)])
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Source fetchers
# ---------------------------------------------------------------------------


def _fetch_nvd_references(cve_id: str) -> tuple[list[dict[str, Any]], str]:
    """Return (NVD reference list, vulnStatus) for *cve_id*, or ([], 'unknown') on failure."""
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
    headers: dict[str, str] = {}
    nvd_api_key = os.environ.get("NVD_API_KEY", "").strip()
    if nvd_api_key:
        headers["apiKey"] = nvd_api_key
    try:
        resp = _get_with_retry(url, headers=headers or None, timeout=15)
        data = resp.json()
        vulns = data.get("vulnerabilities") or []
        if not vulns:
            return [], "unknown"
        cve_data = vulns[0].get("cve", {})
        refs = cve_data.get("references") or []
        vuln_status = cve_data.get("vulnStatus", "unknown")
        return refs, vuln_status
    except Exception:  # noqa: BLE001
        return [], "unknown"


def _fetch_ghsa(cve_id: str) -> list[dict[str, Any]]:
    """Return GitHub Security Advisories for *cve_id*, or [] on failure.

    Uses the public GitHub REST API: GET /advisories?cve_id=CVE-XXXX-NNNN
    """
    url = "https://api.github.com/advisories"
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    try:
        resp = _get_with_retry(url, headers=headers, params={"cve_id": cve_id}, timeout=15)
        data = resp.json()
        if isinstance(data, list):
            return data
        return []
    except Exception:  # noqa: BLE001
        return []


def _fetch_cisa_kev(cve_id: str) -> dict[str, Any]:
    """Return CISA KEV entry for *cve_id*, or {} if not found."""
    try:
        resp = _get_with_retry(
            "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
            timeout=15,
        )
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
        resp = _get_with_retry(url, headers=headers, params={"cve": cve_id}, timeout=20)
        payload = resp.json()
        data = payload.get("data") or []
        return data[0] if data else {}
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# GHSA analysis helpers
# ---------------------------------------------------------------------------


def _extract_ghsa_signals(advisories: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract vendor-response signals from GHSA advisories.

    Returns dict with keys:
      - has_advisory: bool
      - state: str (e.g., 'published', 'withdrawn')
      - has_patched_versions: bool
      - patched_versions: list[str]
      - severity: str | None
      - withdrawn: bool
      - cvss_score: float | None
    """
    result: dict[str, Any] = {
        "has_advisory": False,
        "state": "none",
        "has_patched_versions": False,
        "patched_versions": [],
        "severity": None,
        "withdrawn": False,
        "cvss_score": None,
    }
    if not advisories:
        return result

    result["has_advisory"] = True
    # Use the first (most relevant) advisory.
    adv = advisories[0]
    result["state"] = adv.get("state", "unknown")
    result["severity"] = adv.get("severity")
    result["withdrawn"] = adv.get("withdrawn_at") is not None

    # Extract CVSS score
    cvss = adv.get("cvss", {})
    if isinstance(cvss, dict) and cvss.get("score") is not None:
        try:
            result["cvss_score"] = float(cvss["score"])
        except (TypeError, ValueError):
            pass

    # Extract patched versions from vulnerabilities array.
    patched: list[str] = []
    for vuln in adv.get("vulnerabilities") or []:
        pv = vuln.get("patched_versions")
        if pv and isinstance(pv, str) and pv.strip():
            patched.append(pv.strip())
        first_patched = vuln.get("first_patched_version", {})
        if isinstance(first_patched, dict):
            fpv = first_patched.get("identifier", "")
            if fpv and fpv not in patched:
                patched.append(fpv)

    result["patched_versions"] = patched
    result["has_patched_versions"] = bool(patched)
    return result


# ---------------------------------------------------------------------------
# Classification engine
# ---------------------------------------------------------------------------


def _score_to_confidence(score: float) -> str:
    """Map a numeric confidence score (0-1) to a tier label."""
    if score >= 0.7:
        return CONFIDENCE_HIGH
    if score >= 0.4:
        return CONFIDENCE_MODERATE
    return CONFIDENCE_LOW


def classify(
    references: list[dict[str, Any]],
    nvd_status: str,
    ghsa_signals: dict[str, Any],
    cisa_kev: dict[str, Any],
    vulncheck_kev: dict[str, Any],
) -> tuple[str, str, float, list[str]]:
    """Derive (state, confidence_label, raw_score, evidence_list) from gathered signals.

    Returns:
        state: one of VALID_STATES
        confidence_label: 'high', 'moderate', or 'low'
        raw_score: float in [0, 1]
        evidence: list of human-readable evidence strings
    """
    evidence: list[str] = []
    state = "unknown"
    score = 0.15

    # ── NVD vuln status ──────────────────────────────────────────────────────
    nvd_status_lower = nvd_status.lower()
    if nvd_status_lower in ("rejected", "disputed"):
        state = "wont_fix"
        score = max(score, 0.6)
        evidence.append(f"NVD status is '{nvd_status}' — vendor disputed or rejected")

    elif nvd_status_lower in ("modified", "analyzed"):
        evidence.append(f"NVD status: {nvd_status}")
        score = max(score, 0.3)

    # ── NVD reference tag analysis ───────────────────────────────────────────
    ref_tags_flat: set[str] = set()
    ref_urls: list[str] = []
    for ref in references:
        for tag in ref.get("tags") or []:
            ref_tags_flat.add(tag.lower())
        url = ref.get("url", "")
        if url:
            ref_urls.append(url.lower())

    has_patch_tag = "patch" in ref_tags_flat or "vendor advisory" in ref_tags_flat or "vendor-advisory" in ref_tags_flat
    has_fix_tag = "fix" in ref_tags_flat or "release notes" in ref_tags_flat or "release-notes" in ref_tags_flat
    has_mitigation_tag = "mitigation" in ref_tags_flat or "workaround" in ref_tags_flat

    if has_patch_tag or has_fix_tag:
        if state not in ("wont_fix",):
            state = "patch_available"
        score = max(score, 0.75)
        evidence.append(f"NVD reference tags include: {sorted(ref_tags_flat)}")

    elif has_mitigation_tag and state == "unknown":
        state = "no_patch"
        score = max(score, 0.5)
        evidence.append(f"NVD reference tags include mitigation/workaround: {sorted(ref_tags_flat)}")

    # Heuristic: scan ref URLs for patch/fix/backport/wontfix keywords.
    url_text = " ".join(ref_urls)

    for kw in _BACKPORT_KEYWORDS:
        if kw in url_text:
            if state in ("unknown", "no_patch"):
                state = "patch_backported"
                score = max(score, 0.55)
            evidence.append(f"Backport keyword '{kw}' found in reference URLs")
            break

    for kw in _WONTFIX_KEYWORDS:
        if kw in url_text:
            if state == "unknown":
                state = "wont_fix"
                score = max(score, 0.5)
            evidence.append(f"Won't-fix keyword '{kw}' found in reference URLs")
            break

    if state == "unknown":
        for kw in _PATCH_KEYWORDS:
            if kw in url_text:
                state = "patch_available"
                score = max(score, 0.5)
                evidence.append(f"Patch keyword '{kw}' found in reference URLs")
                break

    if state == "unknown":
        for kw in _WORKAROUND_KEYWORDS:
            if kw in url_text:
                state = "no_patch"
                score = max(score, 0.4)
                evidence.append(f"Workaround keyword '{kw}' found in reference URLs")
                break

    # ── GHSA signals ─────────────────────────────────────────────────────────
    if ghsa_signals.get("has_advisory"):
        ghsa_state = ghsa_signals.get("state", "unknown")
        evidence.append(f"GitHub Advisory found (state: {ghsa_state})")

        if ghsa_signals.get("withdrawn"):
            evidence.append("GitHub Advisory was withdrawn — may indicate dispute")
            if state == "unknown":
                state = "wont_fix"
                score = max(score, 0.45)

        if ghsa_signals.get("has_patched_versions"):
            patched = ghsa_signals["patched_versions"]
            evidence.append(f"GHSA patched versions: {', '.join(patched)}")
            if state in ("unknown", "investigating", "no_patch"):
                state = "patch_available"
            score = max(score, 0.8)

        elif ghsa_state == "published" and state == "unknown":
            # Published GHSA but no patched versions → investigating
            state = "investigating"
            score = max(score, 0.5)
            evidence.append("GHSA published but no patched versions listed")

    # ── CISA KEV signal ──────────────────────────────────────────────────────
    if cisa_kev:
        required_action = (cisa_kev.get("requiredAction") or "").lower()
        due_date = cisa_kev.get("dueDate", "")
        short_desc = cisa_kev.get("shortDescription", "in KEV catalog")
        evidence.append(f"CISA KEV: {short_desc}")
        if due_date:
            evidence.append(f"CISA KEV due date: {due_date}")

        # CISA often lists "Apply update" — implies patch_available.
        if "apply" in required_action or "update" in required_action or "patch" in required_action:
            if state in ("unknown", "investigating", "no_patch"):
                state = "patch_available"
            score = min(score + 0.2, 0.95)
        else:
            # KEV entry exists but action is unusual — at least investigating
            if state == "unknown":
                state = "investigating"
            score = min(score + 0.1, 0.95)

    # ── VulnCheck KEV signal ─────────────────────────────────────────────────
    if vulncheck_kev:
        evidence.append("VulnCheck KEV: active exploitation confirmed via multi-source aggregation")
        # Active exploitation + unknown status → bump to investigating at minimum.
        if state == "unknown":
            state = "investigating"
        # Boost confidence: confirmed exploitation means strong vendor pressure.
        score = min(score + 0.15, 0.95)

        ransomware = bool(
            vulncheck_kev.get("ransomwareUse")
            or vulncheck_kev.get("ransomware_use")
            or vulncheck_kev.get("knownRansomwareCampaignUse")
        )
        if ransomware:
            evidence.append("VulnCheck KEV: ransomware association — escalated priority")
            score = min(score + 0.05, 0.98)

    # ── Final consistency check ──────────────────────────────────────────────
    if state not in VALID_STATES:
        state = "unknown"

    confidence_label = _score_to_confidence(score)
    return state, confidence_label, round(score, 3), evidence


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------


def track_vendor_response(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Classify vendor patch/response status for a CVE using NVD + GHSA + KEV sources."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id: str = tool_input.get("cve_id", "")

    if not isinstance(cve_id, str) or not cve_id.upper().startswith("CVE-"):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID format. Must be a string like 'CVE-YYYY-NNNN'."}],
        }
        log_tool_output_size("track_vendor_response", result)
        return result

    cve_id = cve_id.upper()
    api_key = os.environ.get("VULNCHECK_API_KEY", "").strip()

    # Gather data from each source independently (failures are non-fatal).
    references, nvd_status = _fetch_nvd_references(cve_id)
    ghsa_advisories = _fetch_ghsa(cve_id)
    ghsa_signals = _extract_ghsa_signals(ghsa_advisories)
    cisa_kev = _fetch_cisa_kev(cve_id)
    vulncheck_kev = _fetch_vulncheck_kev(cve_id, api_key)

    state, confidence_label, raw_score, evidence = classify(
        references, nvd_status, ghsa_signals, cisa_kev, vulncheck_kev
    )

    payload: dict[str, Any] = {
        "cve_id": cve_id,
        "vendor_response_state": state,
        "confidence": confidence_label,
        "confidence_score": raw_score,
        "evidence": evidence,
        "signals": {
            "nvd_references_found": len(references),
            "nvd_status": nvd_status,
            "ghsa_advisory_found": ghsa_signals.get("has_advisory", False),
            "ghsa_patched_versions": ghsa_signals.get("patched_versions", []),
            "cisa_kev_hit": bool(cisa_kev),
            "vulncheck_kev_hit": bool(vulncheck_kev),
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
