"""
Tool for computing a temporal priority score (0–100) for a CVE.

Combines six signals into a single urgency score to answer:
"Given everything I know today, how urgent is this CVE?"

Signals and default weights:
  1. CVSS base score (0–10 → 0–25 pts)        weight=25
  2. Current EPSS probability (0–1 → 0–25 pts)  weight=25
  3. EPSS spike recency (recent jump → bonus)    weight=15
  4. CISA KEV membership (boolean → 0 or max)    weight=20
  5. Patch availability (patched → reduce urgency) weight=10
  6. CVE age decay (older → lower urgency)       weight=5

All HTTP calls use retry/back-off.  Graceful degradation: if any data source
is unavailable, remaining signals still produce a partial score.
"""

from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# TOOL_SPEC — Strands SDK module-based tool specification
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "get_temporal_priority",
    "description": (
        "Computes a 0–100 temporal priority (urgency) score for a given CVE by combining "
        "CVSS base score, current EPSS probability, EPSS spike recency, CISA KEV membership, "
        "patch availability, and CVE age. Higher scores mean more urgent. "
        "Designed to answer: 'given everything I know today, how urgent is this CVE?'"
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to score (e.g., 'CVE-2024-3094').",
                },
            },
            "required": ["cve_id"],
        }
    },
}

# ---------------------------------------------------------------------------
# Configuration (overridable via env vars)
# ---------------------------------------------------------------------------

_MAX_RETRIES = int(os.environ.get("TEMPORAL_PRIORITY_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.environ.get("TEMPORAL_PRIORITY_RETRY_DELAY", "2.0"))
_REQUEST_TIMEOUT = int(os.environ.get("TEMPORAL_PRIORITY_TIMEOUT", "20"))

# Weight configuration
_W_CVSS = int(os.environ.get("TEMPORAL_PRIORITY_W_CVSS", "25"))
_W_EPSS = int(os.environ.get("TEMPORAL_PRIORITY_W_EPSS", "25"))
_W_SPIKE = int(os.environ.get("TEMPORAL_PRIORITY_W_SPIKE", "15"))
_W_KEV = int(os.environ.get("TEMPORAL_PRIORITY_W_KEV", "20"))
_W_PATCH = int(os.environ.get("TEMPORAL_PRIORITY_W_PATCH", "10"))
_W_AGE = int(os.environ.get("TEMPORAL_PRIORITY_W_AGE", "5"))

# EPSS spike threshold (absolute jump in 30 days considered a spike)
_SPIKE_THRESHOLD = float(os.environ.get("TEMPORAL_PRIORITY_SPIKE_THRESHOLD", "0.05"))

# Age half-life in days: after this many days, the age component halves
_AGE_HALF_LIFE_DAYS = float(os.environ.get("TEMPORAL_PRIORITY_AGE_HALF_LIFE", "180"))


# ---------------------------------------------------------------------------
# HTTP helper with retry/back-off
# ---------------------------------------------------------------------------


def _http_get(url: str, params: dict | None = None, headers: dict | None = None) -> requests.Response:
    """GET with retry + exponential back-off."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 429:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                time.sleep(delay)
                last_exc = requests.exceptions.HTTPError("429 Too Many Requests")
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Signal fetchers
# ---------------------------------------------------------------------------


def fetch_nvd_data(cve_id: str) -> dict[str, Any]:
    """Fetch NVD vulnerability data. Returns dict with cvss_score, published_date, has_patch."""
    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "")
    if api_key:
        headers["apiKey"] = api_key

    try:
        resp = _http_get(url, params={"cveId": cve_id.upper()}, headers=headers or None)
        data = resp.json()
    except Exception:
        return {"cvss_score": None, "published_date": None, "has_patch": None}

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {"cvss_score": None, "published_date": None, "has_patch": None}

    cve_item = vulns[0].get("cve", {})

    # Extract CVSS score (prefer v3.1, fallback to v3.0, then v2.0)
    cvss_score = _extract_cvss(cve_item)

    # Published date
    published_date = cve_item.get("published")

    # Patch availability: look for references with tag "Patch"
    has_patch = _check_patch_refs(cve_item)

    return {
        "cvss_score": cvss_score,
        "published_date": published_date,
        "has_patch": has_patch,
    }


def _extract_cvss(cve_item: dict) -> float | None:
    """Extract highest CVSS base score from NVD metrics."""
    metrics = cve_item.get("metrics", {})

    # Try CVSS 3.1
    for m in metrics.get("cvssMetricV31", []):
        score = m.get("cvssData", {}).get("baseScore")
        if score is not None:
            return float(score)

    # Try CVSS 3.0
    for m in metrics.get("cvssMetricV30", []):
        score = m.get("cvssData", {}).get("baseScore")
        if score is not None:
            return float(score)

    # Try CVSS 2.0
    for m in metrics.get("cvssMetricV2", []):
        score = m.get("cvssData", {}).get("baseScore")
        if score is not None:
            return float(score)

    return None


def _check_patch_refs(cve_item: dict) -> bool:
    """Check if any NVD references have the 'Patch' tag."""
    refs = cve_item.get("references", [])
    for ref in refs:
        tags = ref.get("tags", [])
        if "Patch" in tags:
            return True
    return False


def fetch_epss_data(cve_id: str) -> dict[str, Any]:
    """Fetch current EPSS score and recent time-series for spike detection."""
    url = "https://api.first.org/data/v1/epss"
    try:
        resp = _http_get(url, params={"cve": cve_id.upper(), "scope": "time-series", "limit": "30"})
        data = resp.json()
    except Exception:
        return {"current_epss": None, "spike_detected": None, "max_jump": None}

    entries = data.get("data", [])
    if not entries:
        return {"current_epss": None, "spike_detected": None, "max_jump": None}

    entry = entries[0]
    current_epss = _safe_float(entry.get("epss"))

    # Analyse time-series for spike
    series = entry.get("time-series", [])
    spike_info = _analyse_spike(series)

    return {
        "current_epss": current_epss,
        "spike_detected": spike_info["spike_detected"],
        "max_jump": spike_info["max_jump"],
        "days_since_spike": spike_info["days_since_spike"],
    }


def _analyse_spike(series: list[dict]) -> dict[str, Any]:
    """Detect the largest recent spike in EPSS scores."""
    if not series or len(series) < 2:
        return {"spike_detected": False, "max_jump": 0.0, "days_since_spike": None}

    # Sort by date oldest-first
    sorted_pts = sorted(series, key=lambda p: p.get("date", ""))
    scores = [_safe_float(p.get("epss", "0")) or 0.0 for p in sorted_pts]
    dates = [p.get("date", "") for p in sorted_pts]

    max_jump = 0.0
    max_jump_idx = 0
    for i in range(1, len(scores)):
        jump = scores[i] - scores[i - 1]
        if jump > max_jump:
            max_jump = jump
            max_jump_idx = i

    spike_detected = max_jump >= _SPIKE_THRESHOLD

    # Days since the spike
    days_since_spike: int | None = None
    if spike_detected and max_jump_idx < len(dates) and dates[max_jump_idx]:
        try:
            spike_date = datetime.strptime(dates[max_jump_idx], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_since_spike = (now - spike_date).days
        except (ValueError, TypeError):
            pass

    return {
        "spike_detected": spike_detected,
        "max_jump": max_jump,
        "days_since_spike": days_since_spike,
    }


def fetch_kev_status(cve_id: str) -> dict[str, Any]:
    """Check CISA KEV catalog for the CVE."""
    url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    try:
        resp = _http_get(url)
        data = resp.json()
    except Exception:
        return {"in_kev": None, "date_added": None}

    vulnerabilities = data.get("vulnerabilities", [])
    cve_upper = cve_id.upper()
    for vuln in vulnerabilities:
        if vuln.get("cveID", "").upper() == cve_upper:
            return {"in_kev": True, "date_added": vuln.get("dateAdded")}

    return {"in_kev": False, "date_added": None}


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_score(
    cvss_score: float | None,
    current_epss: float | None,
    spike_detected: bool | None,
    days_since_spike: int | None,
    in_kev: bool | None,
    has_patch: bool | None,
    published_date: str | None,
) -> dict[str, Any]:
    """
    Compute the 0–100 temporal priority score from individual signals.

    Returns a dict with the overall score, label, and per-signal breakdown.
    """
    breakdown: dict[str, Any] = {}
    total = 0.0
    max_possible = 0.0

    # 1. CVSS component (0–25)
    if cvss_score is not None:
        # Linear scale: 0 maps to 0, 10 maps to _W_CVSS
        cvss_pts = (cvss_score / 10.0) * _W_CVSS
        breakdown["cvss"] = {"raw": cvss_score, "points": round(cvss_pts, 2), "max": _W_CVSS}
        total += cvss_pts
        max_possible += _W_CVSS
    else:
        breakdown["cvss"] = {"raw": None, "points": 0, "max": _W_CVSS, "note": "unavailable"}
        max_possible += _W_CVSS

    # 2. EPSS component (0–25)
    if current_epss is not None:
        # Non-linear: use sqrt to amplify mid-range values
        # EPSS of 0.5 → sqrt(0.5) ≈ 0.71 → 17.7 pts
        epss_pts = math.sqrt(min(current_epss, 1.0)) * _W_EPSS
        breakdown["epss"] = {"raw": current_epss, "points": round(epss_pts, 2), "max": _W_EPSS}
        total += epss_pts
        max_possible += _W_EPSS
    else:
        breakdown["epss"] = {"raw": None, "points": 0, "max": _W_EPSS, "note": "unavailable"}
        max_possible += _W_EPSS

    # 3. EPSS spike recency (0–15)
    if spike_detected is not None:
        if spike_detected:
            # Decay based on how long ago the spike was
            if days_since_spike is not None and days_since_spike >= 0:
                # Exponential decay: spike from today = full points, decays with half-life of 14 days
                spike_decay = math.exp(-0.693 * days_since_spike / 14.0)
                spike_pts = spike_decay * _W_SPIKE
            else:
                # Spike detected but unknown timing → give 75%
                spike_pts = 0.75 * _W_SPIKE
        else:
            spike_pts = 0.0
        breakdown["spike"] = {
            "detected": spike_detected,
            "days_since": days_since_spike,
            "points": round(spike_pts, 2),
            "max": _W_SPIKE,
        }
        total += spike_pts
        max_possible += _W_SPIKE
    else:
        breakdown["spike"] = {"detected": None, "points": 0, "max": _W_SPIKE, "note": "unavailable"}
        max_possible += _W_SPIKE

    # 4. CISA KEV (0 or 20)
    if in_kev is not None:
        kev_pts = _W_KEV if in_kev else 0.0
        breakdown["kev"] = {"in_kev": in_kev, "points": round(kev_pts, 2), "max": _W_KEV}
        total += kev_pts
        max_possible += _W_KEV
    else:
        breakdown["kev"] = {"in_kev": None, "points": 0, "max": _W_KEV, "note": "unavailable"}
        max_possible += _W_KEV

    # 5. Patch availability (reduces urgency when patched)
    if has_patch is not None:
        if has_patch:
            # Patched → subtract up to _W_PATCH points (less urgent)
            patch_pts = -_W_PATCH
        else:
            # No patch → full urgency bonus
            patch_pts = _W_PATCH
        breakdown["patch"] = {"has_patch": has_patch, "points": round(patch_pts, 2), "max": _W_PATCH}
        total += patch_pts
        max_possible += _W_PATCH
    else:
        breakdown["patch"] = {"has_patch": None, "points": 0, "max": _W_PATCH, "note": "unavailable"}
        max_possible += _W_PATCH

    # 6. CVE age (newer = more urgent, exponential decay)
    age_days = _compute_age_days(published_date)
    if age_days is not None:
        # Exponential decay with configurable half-life
        age_factor = math.exp(-0.693 * age_days / _AGE_HALF_LIFE_DAYS)
        age_pts = age_factor * _W_AGE
        breakdown["age"] = {
            "days": age_days,
            "points": round(age_pts, 2),
            "max": _W_AGE,
        }
        total += age_pts
        max_possible += _W_AGE
    else:
        breakdown["age"] = {"days": None, "points": 0, "max": _W_AGE, "note": "unavailable"}
        max_possible += _W_AGE

    # Normalize to 0–100
    # max_possible includes patch as positive contribution
    # Minimum possible contribution from patch is -_W_PATCH
    # Theoretical range: [-_W_PATCH, sum_of_all_weights]
    total_weights = _W_CVSS + _W_EPSS + _W_SPIKE + _W_KEV + _W_PATCH + _W_AGE
    # Shift total so minimum is 0: add _W_PATCH
    adjusted_total = total + _W_PATCH
    adjusted_max = total_weights + _W_PATCH  # max range
    score = (adjusted_total / adjusted_max) * 100.0 if adjusted_max > 0 else 0.0
    score = max(0.0, min(100.0, score))

    label = _urgency_label(score)

    return {
        "score": round(score, 1),
        "label": label,
        "breakdown": breakdown,
        "signals_available": sum(
            1 for v in breakdown.values() if isinstance(v, dict) and v.get("note") != "unavailable"
        ),
        "signals_total": len(breakdown),
    }


def _compute_age_days(published_date: str | None) -> int | None:
    """Compute days since publication from an ISO date string."""
    if not published_date:
        return None
    try:
        # NVD uses ISO format: 2024-01-15T00:00:00.000
        pub = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0, (now - pub).days)
    except (ValueError, TypeError):
        return None


def _urgency_label(score: float) -> str:
    """Map a 0–100 score to a human-readable urgency label."""
    if score >= 85:
        return "CRITICAL"
    elif score >= 70:
        return "HIGH"
    elif score >= 50:
        return "MEDIUM"
    elif score >= 30:
        return "LOW"
    else:
        return "INFORMATIONAL"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def compute_temporal_priority(cve_id: str) -> dict[str, Any]:
    """Fetch all signals and compute the temporal priority score."""
    cve_id = cve_id.strip().upper()

    # Fetch all signals (graceful degradation on each)
    nvd = fetch_nvd_data(cve_id)
    epss = fetch_epss_data(cve_id)
    kev = fetch_kev_status(cve_id)

    result = compute_score(
        cvss_score=nvd["cvss_score"],
        current_epss=epss["current_epss"],
        spike_detected=epss["spike_detected"],
        days_since_spike=epss.get("days_since_spike"),
        in_kev=kev["in_kev"],
        has_patch=nvd["has_patch"],
        published_date=nvd["published_date"],
    )

    result["cve_id"] = cve_id
    return result


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def render_text(result: dict[str, Any]) -> str:
    """Render the temporal priority result as human-readable text."""
    lines: list[str] = []
    score = result["score"]
    label = result["label"]
    cve_id = result.get("cve_id", "UNKNOWN")

    # Header
    lines.append(f"Temporal Priority Score for {cve_id}")
    lines.append("=" * 50)
    lines.append(f"  Score : {score:.1f} / 100  [{label}]")
    lines.append(f"  Signals available: {result['signals_available']}/{result['signals_total']}")
    lines.append("")

    # Breakdown
    lines.append("Signal Breakdown:")
    lines.append("-" * 50)

    breakdown = result.get("breakdown", {})

    # CVSS
    cvss = breakdown.get("cvss", {})
    if cvss.get("raw") is not None:
        lines.append(f"  CVSS base score   : {cvss['raw']:.1f}/10.0  → {cvss['points']:.1f}/{cvss['max']} pts")
    else:
        lines.append(f"  CVSS base score   : unavailable  → 0/{cvss.get('max', _W_CVSS)} pts")

    # EPSS
    epss_info = breakdown.get("epss", {})
    if epss_info.get("raw") is not None:
        lines.append(
            f"  EPSS probability  : {epss_info['raw']:.4f}  → {epss_info['points']:.1f}/{epss_info['max']} pts"
        )
    else:
        lines.append(f"  EPSS probability  : unavailable  → 0/{epss_info.get('max', _W_EPSS)} pts")

    # Spike
    spike = breakdown.get("spike", {})
    if spike.get("detected") is not None:
        if spike["detected"]:
            days_str = f" ({spike['days_since']}d ago)" if spike.get("days_since") is not None else ""
            lines.append(f"  EPSS spike        : ⚠️  YES{days_str}  → {spike['points']:.1f}/{spike['max']} pts")
        else:
            lines.append(f"  EPSS spike        : none  → 0/{spike['max']} pts")
    else:
        lines.append(f"  EPSS spike        : unavailable  → 0/{spike.get('max', _W_SPIKE)} pts")

    # KEV
    kev_info = breakdown.get("kev", {})
    if kev_info.get("in_kev") is not None:
        if kev_info["in_kev"]:
            lines.append(f"  CISA KEV          : 🚨 IN CATALOG  → {kev_info['points']:.1f}/{kev_info['max']} pts")
        else:
            lines.append(f"  CISA KEV          : not listed  → 0/{kev_info['max']} pts")
    else:
        lines.append(f"  CISA KEV          : unavailable  → 0/{kev_info.get('max', _W_KEV)} pts")

    # Patch
    patch = breakdown.get("patch", {})
    if patch.get("has_patch") is not None:
        if patch["has_patch"]:
            lines.append(f"  Patch available   : ✅ yes  → {patch['points']:.1f}/{patch['max']} pts (reduces urgency)")
        else:
            lines.append(f"  Patch available   : ❌ no   → +{patch['points']:.1f}/{patch['max']} pts (unpatched bonus)")
    else:
        lines.append(f"  Patch available   : unavailable  → 0/{patch.get('max', _W_PATCH)} pts")

    # Age
    age = breakdown.get("age", {})
    if age.get("days") is not None:
        lines.append(f"  CVE age           : {age['days']}d  → {age['points']:.1f}/{age['max']} pts")
    else:
        lines.append(f"  CVE age           : unavailable  → 0/{age.get('max', _W_AGE)} pts")

    lines.append("")
    lines.append("-" * 50)

    # Urgency bar
    bar_len = 20
    filled = int(round(score / 100.0 * bar_len))
    bar = "█" * filled + "░" * (bar_len - filled)
    lines.append(f"  [{bar}] {score:.1f}%")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------


def get_temporal_priority(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands SDK handler for the temporal-priority tool."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool.get("input", {}) or {}
    cve_id = tool_input.get("cve_id")

    if not isinstance(cve_id, str) or not cve_id.strip():
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID. Must be a non-empty string like 'CVE-2024-3094'."}],
        }
        log_tool_output_size("get_temporal_priority", result)
        return result

    if not re.match(r"^CVE-\d{4}-\d{4,}$", cve_id.strip(), re.IGNORECASE):
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Invalid CVE format: '{cve_id}'. Expected format: CVE-YYYY-NNNNN."}],
        }
        log_tool_output_size("get_temporal_priority", result)
        return result

    import json

    payload = compute_temporal_priority(cve_id)
    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": json.dumps(payload, indent=2)}],
    }
    log_tool_output_size("get_temporal_priority", result)
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
