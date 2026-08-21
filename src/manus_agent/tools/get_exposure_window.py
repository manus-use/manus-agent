"""
Tool: get_exposure_window

Computes the vulnerability exposure window for a CVE — the elapsed time between
CVE publication (disclosure) and patch availability. Provides concrete metrics
that security teams need for SLA compliance, risk reporting, and remediation
prioritisation.

Data sources:
1. **NVD** — CVE publish date, modification date, reference URLs (Patch tags)
2. **CISA KEV** — date added to Known Exploited Vulnerabilities catalog
3. **EPSS** — current exploitation probability (contextualises exposure urgency)
4. **GitHub Advisory Database (GHSA)** — patch/advisory publication timestamps

Output includes:
- disclosure_date: when the CVE was first published
- patch_date: earliest detected patch availability (from NVD refs + GHSA)
- exposure_days: days between disclosure and patch (None if no patch yet)
- status: patched | unpatched | unknown
- kev_date: when CISA added to KEV (if applicable)
- kev_exposure_days: days from KEV addition until patch (if both exist)
- current_epss: current EPSS score
- risk_label: contextual exposure risk (critical/high/moderate/low)
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

__all__ = ["get_exposure_window", "compute_exposure_window", "TOOL_SPEC"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
_EPSS_URL = "https://api.first.org/data/v1/epss?cve={cve_id}"
_CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
_GHSA_API_URL = "https://api.github.com/advisories?cve_id={cve_id}"

_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Risk label thresholds (exposure_days, epss pairs)
_RISK_THRESHOLDS = {
    "critical": {"min_days": 90, "min_epss": 0.5},
    "high": {"min_days": 30, "min_epss": 0.3},
    "moderate": {"min_days": 7, "min_epss": 0.1},
}

# ---------------------------------------------------------------------------
# Strands TOOL_SPEC
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "get_exposure_window",
    "description": (
        "Computes the vulnerability exposure window for a CVE — the elapsed time "
        "between CVE disclosure and patch availability. Returns disclosure date, "
        "patch date, exposure duration in days, CISA KEV timing, current EPSS score, "
        "and a contextual risk label. Use this to answer: 'How long were systems "
        "vulnerable?' and 'Is the exposure window still open?'"
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier (e.g., 'CVE-2021-44228').",
                }
            },
            "required": ["cve_id"],
        }
    },
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _build_nvd_headers() -> dict[str, str]:
    """Return request headers, injecting NVD_API_KEY when available."""
    headers: dict[str, str] = {"Accept": "application/json"}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key
    return headers


def _build_github_headers() -> dict[str, str]:
    """Return GitHub API headers with token if available."""
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_with_retry(url: str, *, headers: dict[str, str] | None = None, timeout: int = 15) -> requests.Response:
    """GET with exponential back-off on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUS:
                last_exc = requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))
                continue
            resp.raise_for_status()
            return resp
        except requests.ConnectionError as exc:
            last_exc = exc
            time.sleep(_RETRY_BASE_DELAY * (2**attempt))
        except requests.Timeout as exc:
            last_exc = exc
            time.sleep(_RETRY_BASE_DELAY * (2**attempt))
    raise last_exc or RuntimeError("All retries exhausted")


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------


def _fetch_nvd_data(cve_id: str) -> dict[str, Any] | None:
    """Fetch CVE record from NVD. Returns the vulnerability dict or None."""
    url = _NVD_CVE_URL.format(cve_id=cve_id.upper())
    try:
        resp = _get_with_retry(url, headers=_build_nvd_headers())
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if vulns:
            return vulns[0].get("cve", {})
    except Exception:
        pass
    return None


def _fetch_epss(cve_id: str) -> float | None:
    """Fetch current EPSS score. Returns float or None."""
    url = _EPSS_URL.format(cve_id=cve_id.upper())
    try:
        resp = _get_with_retry(url, timeout=10)
        data = resp.json()
        entries = data.get("data", [])
        if entries:
            return float(entries[0].get("epss", 0))
    except Exception:
        pass
    return None


def _fetch_kev_date(cve_id: str) -> str | None:
    """Check CISA KEV for the CVE and return dateAdded or None."""
    try:
        resp = _get_with_retry(_CISA_KEV_URL, timeout=20)
        data = resp.json()
        for vuln in data.get("vulnerabilities", []):
            if vuln.get("cveID", "").upper() == cve_id.upper():
                return vuln.get("dateAdded")
    except Exception:
        pass
    return None


def _fetch_ghsa_patch_date(cve_id: str) -> str | None:
    """Query GitHub Advisory Database for earliest patch/advisory date."""
    url = _GHSA_API_URL.format(cve_id=cve_id.upper())
    try:
        resp = _get_with_retry(url, headers=_build_github_headers(), timeout=15)
        advisories = resp.json()
        if not isinstance(advisories, list) or not advisories:
            return None

        earliest: str | None = None
        for adv in advisories:
            # published_at is when the advisory was published
            pub = adv.get("published_at") or adv.get("updated_at")
            if pub and (earliest is None or pub < earliest):
                earliest = pub
        return earliest
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Date parsing helpers
# ---------------------------------------------------------------------------


def _parse_date(date_str: str | None) -> date | None:
    """Parse ISO date string to date object. Handles multiple formats."""
    if not date_str:
        return None
    # Try ISO formats
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            # Strip timezone suffix for parsing
            clean = re.sub(r"[Zz]$", "", date_str)
            clean = re.sub(r"[+-]\d{2}:\d{2}$", "", clean)
            return datetime.strptime(clean, fmt).date()
        except ValueError:
            continue
    return None


def _days_between(d1: date | None, d2: date | None) -> int | None:
    """Compute days between two dates. Returns None if either is None."""
    if d1 is None or d2 is None:
        return None
    return (d2 - d1).days


# ---------------------------------------------------------------------------
# Patch detection from NVD references
# ---------------------------------------------------------------------------


def _extract_patch_date_from_refs(nvd_record: dict[str, Any]) -> str | None:
    """Look for Patch-tagged references in NVD and extract earliest date signal.

    NVD references with tag 'Patch' indicate fix availability. The reference URL
    itself may contain date info (GitHub commits, release URLs), or we fall back
    to the CVE's lastModified date as a proxy when patches exist.
    """
    references = nvd_record.get("references", [])
    patch_refs = [r for r in references if "Patch" in (r.get("tags") or [])]

    if not patch_refs:
        return None

    # If we have patch references, use the CVE's lastModified as a conservative
    # estimate for when patch info became available in NVD
    return nvd_record.get("lastModified")


# ---------------------------------------------------------------------------
# Risk label computation
# ---------------------------------------------------------------------------


def _compute_risk_label(
    exposure_days: int | None,
    epss: float | None,
    is_in_kev: bool,
    is_patched: bool,
) -> str:
    """Compute contextual exposure risk label.

    Factors:
    - Longer exposure = higher risk
    - Higher EPSS = higher risk
    - In CISA KEV = automatic upgrade
    - Unpatched = automatic upgrade
    """
    if is_in_kev and not is_patched:
        return "critical"

    eff_epss = epss if epss is not None else 0.0
    eff_days = exposure_days if exposure_days is not None else 0

    # Unpatched CVEs with any EPSS signal
    if not is_patched:
        if eff_epss >= _RISK_THRESHOLDS["critical"]["min_epss"]:
            return "critical"
        if eff_epss >= _RISK_THRESHOLDS["high"]["min_epss"]:
            return "high"
        if eff_days >= _RISK_THRESHOLDS["moderate"]["min_days"]:
            return "high"
        return "moderate"

    # Patched — risk reflects how long the window was open
    if eff_days >= _RISK_THRESHOLDS["critical"]["min_days"] and eff_epss >= _RISK_THRESHOLDS["high"]["min_epss"]:
        return "critical"
    if eff_days >= _RISK_THRESHOLDS["high"]["min_days"] or eff_epss >= _RISK_THRESHOLDS["high"]["min_epss"]:
        return "high"
    if eff_days >= _RISK_THRESHOLDS["moderate"]["min_days"] or eff_epss >= _RISK_THRESHOLDS["moderate"]["min_epss"]:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_exposure_window(cve_id: str) -> dict[str, Any]:
    """Compute the exposure window for a CVE.

    Returns a structured dict with all timing metrics and risk assessment.
    This is the pure-logic entry point (no Strands wrapper) for testability.
    """
    cve_upper = cve_id.strip().upper()

    if not _CVE_RE.match(cve_upper):
        return {"error": f"Invalid CVE ID format: {cve_id}"}

    # Fetch data from all sources
    nvd_record = _fetch_nvd_data(cve_upper)
    if not nvd_record:
        return {"error": f"CVE not found in NVD: {cve_upper}"}

    epss = _fetch_epss(cve_upper)
    kev_date_str = _fetch_kev_date(cve_upper)
    ghsa_patch_date_str = _fetch_ghsa_patch_date(cve_upper)

    # Parse disclosure date (NVD published)
    disclosure_str = nvd_record.get("published")
    disclosure_date = _parse_date(disclosure_str)

    # Parse patch dates from multiple sources
    nvd_patch_str = _extract_patch_date_from_refs(nvd_record)
    nvd_patch_date = _parse_date(nvd_patch_str)
    ghsa_patch_date = _parse_date(ghsa_patch_date_str)

    # Use earliest patch signal
    patch_date: date | None = None
    patch_source: str = "unknown"
    if nvd_patch_date and ghsa_patch_date:
        if nvd_patch_date <= ghsa_patch_date:
            patch_date = nvd_patch_date
            patch_source = "nvd_reference"
        else:
            patch_date = ghsa_patch_date
            patch_source = "ghsa"
    elif nvd_patch_date:
        patch_date = nvd_patch_date
        patch_source = "nvd_reference"
    elif ghsa_patch_date:
        patch_date = ghsa_patch_date
        patch_source = "ghsa"

    # Compute exposure metrics
    kev_date = _parse_date(kev_date_str)
    exposure_days = _days_between(disclosure_date, patch_date)
    kev_exposure_days = _days_between(kev_date, patch_date) if kev_date else None

    # If no patch detected and CVE is old, compute days since disclosure
    today = date.today()
    if patch_date is None and disclosure_date:
        exposure_days = (today - disclosure_date).days

    is_patched = patch_date is not None
    status = "patched" if is_patched else "unpatched"

    # Compute risk label
    risk_label = _compute_risk_label(
        exposure_days=exposure_days,
        epss=epss,
        is_in_kev=kev_date is not None,
        is_patched=is_patched,
    )

    return {
        "cve_id": cve_upper,
        "status": status,
        "disclosure_date": str(disclosure_date) if disclosure_date else None,
        "patch_date": str(patch_date) if patch_date else None,
        "patch_source": patch_source if is_patched else None,
        "exposure_days": exposure_days,
        "kev_date": str(kev_date) if kev_date else None,
        "kev_exposure_days": kev_exposure_days,
        "current_epss": epss,
        "risk_label": risk_label,
        "description": nvd_record.get("descriptions", [{}])[0].get("value", "")
        if nvd_record.get("descriptions")
        else "",
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_text(result: dict[str, Any]) -> str:
    """Format exposure window result as human-readable text."""
    if "error" in result:
        return f"Error: {result['error']}"

    lines = [
        f"CVE Exposure Window: {result['cve_id']}",
        f"{'=' * 50}",
        "",
        f"Status:          {result['status'].upper()}",
        f"Risk Label:      {result['risk_label'].upper()}",
        "",
        f"Disclosure Date: {result['disclosure_date'] or 'Unknown'}",
        f"Patch Date:      {result['patch_date'] or 'No patch detected'}",
    ]

    if result.get("patch_source"):
        lines.append(f"Patch Source:    {result['patch_source']}")

    lines.append("")

    if result["exposure_days"] is not None:
        if result["status"] == "patched":
            lines.append(f"Exposure Window: {result['exposure_days']} days (closed)")
        else:
            lines.append(f"Exposure Window: {result['exposure_days']} days (STILL OPEN)")
    else:
        lines.append("Exposure Window: Unknown")

    if result.get("kev_date"):
        lines.append(f"CISA KEV Date:   {result['kev_date']}")
        if result.get("kev_exposure_days") is not None:
            lines.append(f"KEV → Patch:     {result['kev_exposure_days']} days")
        else:
            lines.append("KEV → Patch:     No patch yet")

    if result.get("current_epss") is not None:
        lines.append(f"Current EPSS:    {result['current_epss']:.4f}")

    if result.get("description"):
        desc = result["description"]
        if len(desc) > 200:
            desc = desc[:197] + "..."
        lines.extend(["", f"Description: {desc}"])

    return "\n".join(lines)


def _format_json(result: dict[str, Any]) -> str:
    """Format exposure window result as JSON."""
    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Strands tool handler
# ---------------------------------------------------------------------------


def get_exposure_window(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands tool entry point for exposure window computation."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool.get("input", {})
    cve_id = tool_input.get("cve_id", "")

    if not cve_id or not isinstance(cve_id, str):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Missing or invalid 'cve_id' parameter."}],
        }
        log_tool_output_size("get_exposure_window", result)
        return result

    exposure = compute_exposure_window(cve_id)

    if "error" in exposure:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": exposure["error"]}],
        }
        log_tool_output_size("get_exposure_window", result)
        return result

    text_output = _format_text(exposure)
    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": text_output}],
    }
    log_tool_output_size("get_exposure_window", result)
    return result
