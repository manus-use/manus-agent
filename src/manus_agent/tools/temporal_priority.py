"""
Tool for computing a temporal priority score (0–100) for a CVE.

Combines multiple signals to answer: "given everything I know today, how urgent
is this vulnerability?"

Signals
-------
1. CVSS base score (from NVD)
2. Current EPSS score (from FIRST.org)
3. EPSS spike recency (recent jump → higher urgency)
4. CISA KEV membership (active exploitation)
5. Patch availability (NVD references tagged "Patch")
6. CVE age (newer CVEs get a recency boost)

Each signal is normalised to 0–1 and combined via configurable weights to produce
a final 0–100 urgency score with a human-readable urgency label.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# TOOL_SPEC — Strands SDK interface
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "temporal_priority",
    "description": (
        "Computes a temporal priority score (0–100) for a CVE combining CVSS base score, "
        "current EPSS, EPSS spike recency, CISA KEV membership, patch availability, and "
        "CVE age. Answers: 'given everything I know today, how urgent is this?' "
        "Use after get_nvd_data and get_epss_trend for a single actionable urgency number."
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
# Weights (sum to 1.0) — override via environment for tuning/testing
# ---------------------------------------------------------------------------

_W_CVSS = float(os.environ.get("TP_W_CVSS", "0.25"))
_W_EPSS = float(os.environ.get("TP_W_EPSS", "0.25"))
_W_SPIKE = float(os.environ.get("TP_W_SPIKE", "0.15"))
_W_KEV = float(os.environ.get("TP_W_KEV", "0.20"))
_W_PATCH = float(os.environ.get("TP_W_PATCH", "0.05"))
_W_AGE = float(os.environ.get("TP_W_AGE", "0.10"))

# ---------------------------------------------------------------------------
# Retry / back-off constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = int(os.environ.get("TP_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.environ.get("TP_RETRY_BASE_DELAY", "1.5"))
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# EPSS spike thresholds
_SPIKE_THRESHOLD = 0.05  # absolute jump considered a spike
_SPIKE_RECENCY_HALFLIFE_DAYS = 14  # exponential decay half-life


# ---------------------------------------------------------------------------
# HTTP helper with retry/back-off
# ---------------------------------------------------------------------------


def _get_with_retry(url: str, params: dict | None = None, headers: dict | None = None, timeout: int = 20) -> dict:
    """HTTP GET with exponential back-off on retryable failures."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Signal collectors
# ---------------------------------------------------------------------------


def _fetch_nvd(cve_id: str) -> dict[str, Any]:
    """Fetch NVD CVE data. Returns parsed JSON or empty dict on failure."""
    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "")
    if api_key:
        headers["apiKey"] = api_key
    params = {"cveId": cve_id.upper()}
    try:
        return _get_with_retry(url, params=params, headers=headers)
    except Exception:
        return {}


def _fetch_epss(cve_id: str) -> dict[str, Any]:
    """Fetch current EPSS score and time-series. Returns parsed JSON or empty dict."""
    url = "https://api.first.org/data/v1/epss"
    params = {"cve": cve_id.upper(), "scope": "time-series", "limit": "30"}
    try:
        return _get_with_retry(url, params=params)
    except Exception:
        return {}


def _fetch_kev() -> set[str]:
    """Fetch the CISA KEV catalog and return a set of CVE IDs."""
    url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    try:
        data = _get_with_retry(url, timeout=15)
        vulns = data.get("vulnerabilities", [])
        return {v.get("cveID", "").upper() for v in vulns}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Signal scoring functions (each returns 0.0–1.0)
# ---------------------------------------------------------------------------


def score_cvss(nvd_data: dict) -> tuple[float, float | None]:
    """Extract best CVSS score and normalise to 0–1. Returns (normalised, raw_score)."""
    vulns = nvd_data.get("vulnerabilities", [])
    if not vulns:
        return 0.0, None
    cve_item = vulns[0].get("cve", {})
    metrics = cve_item.get("metrics", {})

    # Try CVSS 3.1 first, then 3.0, then 2.0
    best_score: float | None = None
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key, [])
        if entries:
            cvss_data = entries[0].get("cvssData", {})
            score = cvss_data.get("baseScore")
            if score is not None:
                best_score = float(score)
                break

    if best_score is None:
        return 0.0, None

    # CVSS 2.0 is out of 10, same as 3.x
    return best_score / 10.0, best_score


def score_epss_current(epss_data: dict) -> tuple[float, float | None]:
    """Extract latest EPSS score (already 0–1). Returns (score, raw_epss)."""
    data_entries = epss_data.get("data", [])
    if not data_entries:
        return 0.0, None
    entry = data_entries[0]
    # Current score is top-level epss field
    epss_val = entry.get("epss")
    if epss_val is not None:
        val = float(epss_val)
        return val, val
    # Fallback: first time-series point
    ts = entry.get("time-series", [])
    if ts:
        val = float(ts[0].get("epss", 0))
        return val, val
    return 0.0, None


def score_epss_spike(epss_data: dict) -> tuple[float, dict[str, Any]]:
    """
    Detect the largest recent spike in EPSS and score its urgency via exponential decay.
    Returns (spike_score_0_1, details_dict).
    """
    data_entries = epss_data.get("data", [])
    if not data_entries:
        return 0.0, {"max_jump": 0, "spike_detected": False}

    entry = data_entries[0]
    ts = list(entry.get("time-series", []))
    if len(ts) < 2:
        return 0.0, {"max_jump": 0, "spike_detected": False}

    # Sort oldest-first
    points = sorted(ts, key=lambda x: x.get("date", ""))
    max_jump = 0.0
    max_jump_date: str | None = None
    for i in range(1, len(points)):
        prev = float(points[i - 1].get("epss", 0))
        curr = float(points[i].get("epss", 0))
        jump = curr - prev
        if jump > max_jump:
            max_jump = jump
            max_jump_date = points[i].get("date")

    spike_detected = max_jump >= _SPIKE_THRESHOLD
    if not spike_detected or not max_jump_date:
        return 0.0, {"max_jump": round(max_jump, 4), "spike_detected": False}

    # Exponential decay based on how recent the spike was
    try:
        spike_dt = datetime.strptime(max_jump_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_ago = max((now - spike_dt).days, 0)
    except (ValueError, TypeError):
        days_ago = 30  # fallback

    decay = math.exp(-0.693 * days_ago / _SPIKE_RECENCY_HALFLIFE_DAYS)  # ln(2) ≈ 0.693
    # Scale: a spike of 0.10 at day 0 → 1.0; smaller spikes scale linearly
    magnitude_factor = min(max_jump / 0.10, 1.0)
    score = decay * magnitude_factor

    return min(score, 1.0), {
        "max_jump": round(max_jump, 4),
        "spike_date": max_jump_date,
        "days_ago": days_ago,
        "spike_detected": True,
    }


def score_kev(cve_id: str, kev_set: set[str]) -> tuple[float, bool]:
    """Return 1.0 if CVE is in KEV, 0.0 otherwise."""
    in_kev = cve_id.upper() in kev_set
    return (1.0 if in_kev else 0.0), in_kev


def score_patch_availability(nvd_data: dict) -> tuple[float, bool]:
    """
    Check NVD references for patch tags. Patch available → lower urgency (inverted).
    Returns (score_0_1, has_patch). Higher score = MORE urgent (no patch).
    """
    vulns = nvd_data.get("vulnerabilities", [])
    if not vulns:
        return 0.5, False  # unknown → middle ground

    cve_item = vulns[0].get("cve", {})
    references = cve_item.get("references", [])
    has_patch = any("Patch" in ref.get("tags", []) for ref in references)

    # No patch = higher urgency (1.0), patch available = lower urgency (0.0)
    return (0.0 if has_patch else 1.0), has_patch


def score_age(nvd_data: dict) -> tuple[float, int | None]:
    """
    Score based on CVE age. Newer CVEs are more urgent (less time for defenders).
    Returns (score_0_1, age_days).
    Uses exponential decay: 0 days → 1.0, ~90 days → 0.5, ~365 days → ~0.06.
    """
    vulns = nvd_data.get("vulnerabilities", [])
    if not vulns:
        return 0.5, None  # unknown → middle ground

    cve_item = vulns[0].get("cve", {})
    published = cve_item.get("published")
    if not published:
        return 0.5, None

    try:
        # NVD format: "2024-03-29T13:15:00.000"
        pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age_days = max((now - pub_dt).days, 0)
    except (ValueError, TypeError):
        return 0.5, None

    # Exponential decay with 90-day half-life
    decay = math.exp(-0.693 * age_days / 90)
    return decay, age_days


# ---------------------------------------------------------------------------
# Composite scorer
# ---------------------------------------------------------------------------


def compute_temporal_priority(
    cve_id: str,
    nvd_data: dict | None = None,
    epss_data: dict | None = None,
    kev_set: set[str] | None = None,
) -> dict[str, Any]:
    """
    Compute the temporal priority score for a CVE.

    Parameters
    ----------
    cve_id : str
        CVE identifier.
    nvd_data : dict, optional
        Pre-fetched NVD API response. If None, will be fetched.
    epss_data : dict, optional
        Pre-fetched EPSS API response. If None, will be fetched.
    kev_set : set, optional
        Pre-fetched KEV CVE ID set. If None, will be fetched.

    Returns
    -------
    dict with keys: score, label, signals, weights, cve_id
    """
    cve_id = cve_id.upper().strip()

    # Fetch data if not provided
    if nvd_data is None:
        nvd_data = _fetch_nvd(cve_id)
    if epss_data is None:
        epss_data = _fetch_epss(cve_id)
    if kev_set is None:
        kev_set = _fetch_kev()

    # Compute individual signals
    cvss_score, cvss_raw = score_cvss(nvd_data)
    epss_score, epss_raw = score_epss_current(epss_data)
    spike_score, spike_details = score_epss_spike(epss_data)
    kev_score, in_kev = score_kev(cve_id, kev_set)
    patch_score, has_patch = score_patch_availability(nvd_data)
    age_score, age_days = score_age(nvd_data)

    # Weighted composite
    raw_composite = (
        _W_CVSS * cvss_score
        + _W_EPSS * epss_score
        + _W_SPIKE * spike_score
        + _W_KEV * kev_score
        + _W_PATCH * patch_score
        + _W_AGE * age_score
    )

    # Scale to 0–100
    final_score = round(min(max(raw_composite * 100, 0), 100), 1)

    # Urgency label
    if final_score >= 80:
        label = "CRITICAL"
    elif final_score >= 60:
        label = "HIGH"
    elif final_score >= 40:
        label = "MEDIUM"
    elif final_score >= 20:
        label = "LOW"
    else:
        label = "INFORMATIONAL"

    return {
        "cve_id": cve_id,
        "score": final_score,
        "label": label,
        "signals": {
            "cvss": {
                "normalised": round(cvss_score, 4),
                "raw_score": cvss_raw,
                "weight": _W_CVSS,
            },
            "epss_current": {
                "normalised": round(epss_score, 4),
                "raw_epss": epss_raw,
                "weight": _W_EPSS,
            },
            "epss_spike": {
                "normalised": round(spike_score, 4),
                "weight": _W_SPIKE,
                **spike_details,
            },
            "cisa_kev": {
                "normalised": round(kev_score, 4),
                "in_kev": in_kev,
                "weight": _W_KEV,
            },
            "patch_availability": {
                "normalised": round(patch_score, 4),
                "has_patch": has_patch,
                "weight": _W_PATCH,
                "note": "higher = no patch (more urgent)",
            },
            "age": {
                "normalised": round(age_score, 4),
                "age_days": age_days,
                "weight": _W_AGE,
                "note": "higher = newer CVE (more urgent)",
            },
        },
        "interpretation": (
            f"{cve_id} has a temporal priority of {final_score}/100 ({label}). "
            + (f"CVSS {cvss_raw}/10. " if cvss_raw else "")
            + (f"EPSS {epss_raw:.4f}. " if epss_raw else "")
            + ("In CISA KEV (actively exploited). " if in_kev else "")
            + ("Patch available. " if has_patch else "No patch found. ")
            + (f"Published {age_days} days ago." if age_days is not None else "")
        ),
    }


# ---------------------------------------------------------------------------
# Strands tool entry point
# ---------------------------------------------------------------------------


def temporal_priority(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands SDK tool entry point."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")

    if not isinstance(cve_id, str) or not cve_id.strip():
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID. Must be a non-empty string."}],
        }
        log_tool_output_size("temporal_priority", result)
        return result

    try:
        data = compute_temporal_priority(cve_id)
    except Exception as exc:
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Failed to compute temporal priority: {exc}"}],
        }
        log_tool_output_size("temporal_priority", result)
        return result

    import json

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": json.dumps(data, indent=2)}],
    }
    log_tool_output_size("temporal_priority", result)
    return result
