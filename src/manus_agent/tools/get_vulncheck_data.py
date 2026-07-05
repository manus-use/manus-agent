#!/usr/bin/env python3
"""
Tool for fetching VulnCheck vulnerability intelligence for a CVE.

Queries two VulnCheck API endpoints:
- VulnCheck KEV: exploitation status aggregated from 100+ sources (FBI Flash,
  CERT advisories, threat intel feeds) — broader than CISA KEV.
- VulnCheck NVD2: enriched NVD data with improved CPE matching for more
  accurate version-range analysis.

Requires ``VULNCHECK_API_KEY`` environment variable. When absent the tool
returns ``available: False`` instead of raising so callers can degrade
gracefully.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Retry / back-off configuration
# ---------------------------------------------------------------------------
# Maximum number of HTTP attempts per endpoint (1 = no retry).
# Override with VULNCHECK_MAX_RETRIES env var (useful in tests: set to "1").
_VC_MAX_RETRIES: int = int(os.environ.get("VULNCHECK_MAX_RETRIES", "3"))

# Base delay between retries in seconds (doubles each attempt: 1s, 2s, 4s…).
# Override with VULNCHECK_RETRY_BASE_DELAY env var (set to "0" in tests).
_VC_RETRY_BASE_DELAY: float = float(os.environ.get("VULNCHECK_RETRY_BASE_DELAY", "1.0"))

# HTTP status codes that are retryable (rate-limit or transient server error).
_VC_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

TOOL_SPEC = {
    "name": "get_vulncheck_data",
    "description": (
        "Fetches VulnCheck vulnerability intelligence for a given CVE ID from two endpoints: "
        "(1) VulnCheck KEV — exploitation status aggregated from 100+ sources including FBI Flash, "
        "CERT advisories, and threat-intel feeds, providing far broader active-exploitation coverage "
        "than CISA KEV alone; "
        "(2) VulnCheck NVD2 — enriched NVD data with improved CPE matching for more accurate "
        "affected-version analysis. "
        "Requires VULNCHECK_API_KEY environment variable; returns available=False when absent. "
        "A kev.in_kev=True result is a strong signal of confirmed active exploitation. "
        "kev.ransomware_use=True indicates ransomware groups have weaponised this CVE."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to look up (e.g., 'CVE-2024-3094').",
                }
            },
            "required": ["cve_id"],
        }
    },
}

_BASE_URL = "https://api.vulncheck.com/v3/index"
_TIMEOUT = 20


def _make_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


def _vc_get_with_retry(
    url: str,
    headers: dict[str, str],
    params: dict[str, str],
    timeout: int,
) -> requests.Response:
    """GET *url* with exponential back-off retry on retryable status codes.

    Retries on HTTP 429 (rate-limit) and 5xx transient errors up to
    ``_VC_MAX_RETRIES`` total attempts.  Non-retryable 4xx responses
    (e.g. 401, 403, 404) are raised immediately so callers can surface
    authentication errors without wasting quota on pointless retries.

    Back-off schedule (base delay ``_VC_RETRY_BASE_DELAY`` = 1 s by default):
      attempt 2 →  1 s
      attempt 3 →  2 s
      attempt 4 →  4 s

    Returns the final :class:`requests.Response` on success.
    Raises :class:`requests.exceptions.RequestException` on permanent failure.
    """
    last_exc: requests.exceptions.RequestException | None = None
    for attempt in range(1, _VC_MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            if response.status_code in _VC_RETRYABLE_STATUSES:
                if attempt < _VC_MAX_RETRIES:
                    sleep_secs = _VC_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    time.sleep(sleep_secs)
                    continue
                # Final attempt — raise so the caller sees the error.
                response.raise_for_status()
            # Non-retryable 4xx (401, 403, 404, …) → raise immediately.
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            # Only retry on network-level errors or retryable HTTP errors.
            # For HTTP errors, check whether the status is retryable; for
            # connection/timeout errors (no response), always retry.
            http_resp = getattr(exc, "response", None)
            if http_resp is not None and http_resp.status_code not in _VC_RETRYABLE_STATUSES:
                raise  # Non-retryable HTTP error — propagate immediately.
            last_exc = exc
            if attempt < _VC_MAX_RETRIES:
                sleep_secs = _VC_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                time.sleep(sleep_secs)
    # Exhausted all retries.
    if last_exc is not None:
        raise last_exc
    # Should never reach here, but appease mypy.
    raise requests.exceptions.RequestException("VulnCheck request failed after all retries")


def _fetch_kev(cve_id: str, api_key: str) -> dict[str, Any]:
    """Query VulnCheck KEV for a specific CVE.

    Tries ``?cve=CVE-XXXX`` first (direct lookup).  If the response contains
    no matching entry, returns an empty KEV record.
    """
    url = f"{_BASE_URL}/vulncheck-kev"
    params: dict[str, str] = {"cve": cve_id.upper()}
    response = _vc_get_with_retry(url, headers=_make_headers(api_key), params=params, timeout=_TIMEOUT)
    payload = response.json()

    data = payload.get("data") or []
    if not data:
        return {
            "in_kev": False,
            "date_added": None,
            "sources": [],
            "ransomware_use": False,
            "notes": None,
        }

    # The first matching entry is authoritative.
    entry = data[0]

    # Sources: VulnCheck KEV records may carry a "sources" list or "reportedBy".
    sources: list[str] = []
    for src_field in ("sources", "reportedBy", "reported_by"):
        val = entry.get(src_field)
        if isinstance(val, list):
            sources = [str(s) for s in val if s]
            break
        elif isinstance(val, str) and val:
            sources = [val]
            break

    # Date added: try several common field names.
    date_added: str | None = None
    for date_field in ("dateAdded", "date_added", "dateKnownRansomware", "published"):
        val = entry.get(date_field)
        if val:
            date_added = str(val)
            break

    # Ransomware association flag.
    ransomware_use: bool = bool(
        entry.get("ransomwareUse")
        or entry.get("ransomware_use")
        or entry.get("knownRansomwareCampaignUse")
        or entry.get("known_ransomware_campaign_use")
    )

    notes_val = entry.get("notes") or entry.get("note") or entry.get("shortDescription")
    notes: str | None = str(notes_val) if notes_val else None

    return {
        "in_kev": True,
        "date_added": date_added,
        "sources": sources,
        "ransomware_use": ransomware_use,
        "notes": notes,
    }


def _fetch_nvd2(cve_id: str, api_key: str) -> dict[str, Any]:
    """Query VulnCheck NVD2 for enriched NVD data."""
    url = f"{_BASE_URL}/nist-nvd2"
    params: dict[str, str] = {"cve": cve_id.upper()}
    response = _vc_get_with_retry(url, headers=_make_headers(api_key), params=params, timeout=_TIMEOUT)
    payload = response.json()

    data = payload.get("data") or []
    if not data:
        return {
            "cvss_v3_score": None,
            "cvss_v3_vector": None,
            "cvss_v3_severity": None,
            "cpe_matches": [],
            "description": None,
            "published": None,
            "last_modified": None,
        }

    entry = data[0]

    # ── CVSS v3 ──────────────────────────────────────────────────────────────
    cvss_v3_score: float | None = None
    cvss_v3_vector: str | None = None
    cvss_v3_severity: str | None = None

    # VulnCheck NVD2 may wrap metrics under various keys.
    metrics = entry.get("metrics") or {}
    for metric_key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV3"):
        metric_list = metrics.get(metric_key) or []
        if not isinstance(metric_list, list):
            metric_list = [metric_list]
        for m in metric_list:
            cvss_data = m.get("cvssData") or m
            score = cvss_data.get("baseScore")
            if score is not None:
                cvss_v3_score = float(score)
                cvss_v3_vector = cvss_data.get("vectorString")
                cvss_v3_severity = cvss_data.get("baseSeverity")
                break
        if cvss_v3_score is not None:
            break

    # ── CPE matches ──────────────────────────────────────────────────────────
    cpe_matches: list[str] = []
    configurations = entry.get("configurations") or []
    for config in configurations:
        nodes = config.get("nodes") or []
        for node in nodes:
            for cpe_match in node.get("cpeMatch") or []:
                criteria = cpe_match.get("criteria")
                if criteria and criteria not in cpe_matches:
                    cpe_matches.append(criteria)

    # ── Description ──────────────────────────────────────────────────────────
    description: str | None = None
    for desc_entry in entry.get("descriptions") or []:
        if desc_entry.get("lang") == "en":
            description = desc_entry.get("value")
            break

    return {
        "cvss_v3_score": cvss_v3_score,
        "cvss_v3_vector": cvss_v3_vector,
        "cvss_v3_severity": cvss_v3_severity,
        "cpe_matches": cpe_matches,
        "description": description,
        "published": entry.get("published"),
        "last_modified": entry.get("lastModified"),
    }


def get_vulncheck_data(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Fetch VulnCheck KEV and NVD2 intelligence for a CVE."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id: str = tool_input.get("cve_id", "")

    if not isinstance(cve_id, str) or not cve_id.upper().startswith("CVE-"):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID format. Must be a string like 'CVE-YYYY-NNNN'."}],
        }
        log_tool_output_size("get_vulncheck_data", result)
        return result

    cve_id = cve_id.upper()
    api_key = os.environ.get("VULNCHECK_API_KEY", "").strip()

    if not api_key:
        payload: dict[str, Any] = {
            "cve_id": cve_id,
            "available": False,
            "kev": {
                "in_kev": False,
                "date_added": None,
                "sources": [],
                "ransomware_use": False,
                "notes": None,
            },
            "nvd2": {
                "cvss_v3_score": None,
                "cvss_v3_vector": None,
                "cvss_v3_severity": None,
                "cpe_matches": [],
                "description": None,
                "published": None,
                "last_modified": None,
            },
            "error": "VULNCHECK_API_KEY environment variable is not set. VulnCheck enrichment is unavailable.",
        }
        result = {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"json": payload}],
        }
        log_tool_output_size("get_vulncheck_data", result)
        return result

    kev_data: dict[str, Any] = {
        "in_kev": False,
        "date_added": None,
        "sources": [],
        "ransomware_use": False,
        "notes": None,
    }
    nvd2_data: dict[str, Any] = {
        "cvss_v3_score": None,
        "cvss_v3_vector": None,
        "cvss_v3_severity": None,
        "cpe_matches": [],
        "description": None,
        "published": None,
        "last_modified": None,
    }
    error_msg: str | None = None

    try:
        kev_data = _fetch_kev(cve_id, api_key)
    except requests.exceptions.RequestException as exc:
        error_msg = f"VulnCheck KEV request failed: {exc}"
    except Exception as exc:  # noqa: BLE001
        error_msg = f"Unexpected error fetching VulnCheck KEV: {exc}"

    try:
        nvd2_data = _fetch_nvd2(cve_id, api_key)
    except requests.exceptions.RequestException as exc:
        msg = f"VulnCheck NVD2 request failed: {exc}"
        error_msg = f"{error_msg}; {msg}" if error_msg else msg
    except Exception as exc:  # noqa: BLE001
        msg = f"Unexpected error fetching VulnCheck NVD2: {exc}"
        error_msg = f"{error_msg}; {msg}" if error_msg else msg

    payload = {
        "cve_id": cve_id,
        "available": True,
        "kev": kev_data,
        "nvd2": nvd2_data,
        "error": error_msg,
    }

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"json": payload}],
    }
    log_tool_output_size("get_vulncheck_data", result)
    return result
