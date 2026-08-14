"""
Tool for reconstructing the full event timeline of a CVE.

Given a CVE identifier, this tool gathers chronological events from multiple
sources:
1. NVD — publish date, last modified date, CVSS score assignment
2. EPSS — first appearance, significant spikes
3. CISA KEV — date added to the Known Exploited Vulnerabilities catalog
4. Patches — commit/release dates from NVD references and GitHub advisories

The result is a sorted timeline of events showing how quickly a vulnerability
was discovered, scored, weaponised, and patched.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_NVD_MAX_RETRIES = int(os.environ.get("NVD_MAX_RETRIES", "3"))
_NVD_BACKOFF_BASE = float(os.environ.get("NVD_BACKOFF_BASE", "2.0"))
_REQUEST_TIMEOUT = int(os.environ.get("CVE_TIMELINE_TIMEOUT", "20"))

# ---------------------------------------------------------------------------
# TOOL_SPEC (Strands SDK)
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "get_cve_timeline",
    "description": (
        "Reconstructs the full event timeline for a CVE: NVD publication date, "
        "CVSS score assignment, EPSS first appearance and spikes, CISA KEV addition "
        "date, and patch/commit dates from NVD references and GitHub advisories. "
        "Returns a chronologically sorted list of events showing how quickly the "
        "vulnerability was discovered, scored, weaponised, and patched."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to look up (e.g., 'CVE-2021-44228').",
                },
            },
            "required": ["cve_id"],
        }
    },
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _get_with_retry(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = _NVD_MAX_RETRIES,
    backoff_base: float = _NVD_BACKOFF_BASE,
    timeout: int = _REQUEST_TIMEOUT,
) -> requests.Response:
    """GET with exponential back-off on 429/5xx."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < max_retries:
                    time.sleep(backoff_base ** (attempt + 1))
                    continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff_base ** (attempt + 1))
                continue
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------


def _fetch_nvd_events(cve_id: str) -> list[dict[str, Any]]:
    """Fetch NVD data and extract timeline events."""
    url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "")
    if api_key:
        headers["apiKey"] = api_key

    try:
        resp = _get_with_retry(url, params={"cveId": cve_id.upper()}, headers=headers)
        data = resp.json()
    except Exception:
        return []

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return []

    cve_data = vulns[0].get("cve", {})
    events: list[dict[str, Any]] = []

    # Published date
    published = cve_data.get("published", "")
    if published:
        events.append(
            {
                "date": _parse_iso(published),
                "source": "NVD",
                "event": "CVE published",
                "detail": f"{cve_id} published in National Vulnerability Database",
            }
        )

    # Last modified
    last_modified = cve_data.get("lastModified", "")
    if last_modified and last_modified != published:
        events.append(
            {
                "date": _parse_iso(last_modified),
                "source": "NVD",
                "event": "NVD record updated",
                "detail": "NVD entry last modified (score/description/reference change)",
            }
        )

    # CVSS score assignment — use the earliest metric we find
    cvss_date = published  # CVSS is assigned at publication in most cases
    cvss_score, cvss_version = _extract_cvss(cve_data)
    if cvss_score is not None:
        events.append(
            {
                "date": _parse_iso(cvss_date),
                "source": "NVD",
                "event": f"CVSS {cvss_version} score assigned",
                "detail": f"Base score: {cvss_score}",
            }
        )

    # CISA KEV from NVD cisaExploitAdd field (present in NVD 2.0 for KEV entries)
    cisa_add = cve_data.get("cisaExploitAdd", "")
    if cisa_add:
        events.append(
            {
                "date": _parse_iso(cisa_add),
                "source": "CISA KEV",
                "event": "Added to CISA KEV",
                "detail": "Confirmed active exploitation — added to Known Exploited Vulnerabilities catalog",
            }
        )

    # Patch references from NVD references
    patch_events = _extract_patch_references(cve_data)
    events.extend(patch_events)

    return events


def _extract_cvss(cve_data: dict[str, Any]) -> tuple[float | None, str]:
    """Extract the highest-priority CVSS score (3.1 > 3.0 > 2.0)."""
    metrics = cve_data.get("metrics", {})

    for key, version in [
        ("cvssMetricV31", "3.1"),
        ("cvssMetricV30", "3.0"),
        ("cvssMetricV2", "2.0"),
    ]:
        metric_list = metrics.get(key, [])
        if metric_list:
            cvss_data = metric_list[0].get("cvssData", {})
            score = cvss_data.get("baseScore")
            if score is not None:
                return float(score), version

    return None, ""


def _extract_patch_references(cve_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract patch/commit events from NVD references."""
    events: list[dict[str, Any]] = []
    references = cve_data.get("references", [])
    seen_urls: set[str] = set()

    for ref in references:
        url = ref.get("url", "")
        tags = ref.get("tags", [])

        if url in seen_urls:
            continue

        is_patch = "Patch" in tags or "patch" in tags
        is_commit = bool(re.search(r"github\.com/.+/commit/[0-9a-f]+", url))
        is_release = bool(re.search(r"github\.com/.+/releases/tag/", url))

        if is_patch or is_commit or is_release:
            seen_urls.add(url)
            event_type = "Patch commit" if is_commit else "Release" if is_release else "Patch"
            events.append(
                {
                    "date": None,  # NVD references don't have dates; will be enriched
                    "source": "NVD reference",
                    "event": f"{event_type} published",
                    "detail": url,
                }
            )

    return events


def _fetch_kev_date(cve_id: str) -> list[dict[str, Any]]:
    """Fetch CISA KEV catalog and extract the addition date for this CVE."""
    try:
        resp = _get_with_retry("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json")
        data = resp.json()
    except Exception:
        return []

    events: list[dict[str, Any]] = []
    for vuln in data.get("vulnerabilities", []):
        if vuln.get("cveID", "").upper() == cve_id.upper():
            date_added = vuln.get("dateAdded", "")
            if date_added:
                events.append(
                    {
                        "date": _parse_iso(date_added),
                        "source": "CISA KEV",
                        "event": "Added to CISA KEV",
                        "detail": (
                            f"Vulnerability: {vuln.get('vulnerabilityName', 'N/A')}. "
                            f"Required action: {vuln.get('requiredAction', 'N/A')}. "
                            f"Due date: {vuln.get('dueDate', 'N/A')}."
                        ),
                    }
                )
            break

    return events


def _fetch_epss_events(cve_id: str) -> list[dict[str, Any]]:
    """Fetch EPSS time series and extract first appearance + spike events."""
    try:
        resp = _get_with_retry(
            "https://api.first.org/data/v1/epss",
            params={"cve": cve_id.upper(), "scope": "time-series", "limit": "365"},
        )
        data = resp.json()
    except Exception:
        return []

    series = data.get("data", [])
    if not series:
        return []

    # The API may return data under a nested structure
    if isinstance(series, list) and len(series) > 0 and "epss" in series[0]:
        points = series
    elif isinstance(series, list) and len(series) > 0 and "time-series" in series[0]:
        points = series[0].get("time-series", [])
    else:
        # Try alternative response format
        points = series[0].get("time-series", []) if isinstance(series, list) and series else []

    if not points:
        return []

    # Sort oldest first
    points_sorted = sorted(points, key=lambda p: p.get("date", ""))
    events: list[dict[str, Any]] = []

    # First EPSS entry
    first = points_sorted[0]
    first_score = float(first.get("epss", 0))
    events.append(
        {
            "date": _parse_iso(first["date"]),
            "source": "EPSS",
            "event": "First EPSS score",
            "detail": f"Initial score: {first_score:.4f} ({first_score * 100:.2f}%)",
        }
    )

    # Detect spikes (jump > 10% absolute in a single day)
    spike_threshold = 0.10
    prev_score = first_score
    for pt in points_sorted[1:]:
        score = float(pt.get("epss", 0))
        jump = score - prev_score
        if jump >= spike_threshold:
            events.append(
                {
                    "date": _parse_iso(pt["date"]),
                    "source": "EPSS",
                    "event": "EPSS spike detected",
                    "detail": f"Score jumped {prev_score:.4f} → {score:.4f} (+{jump:.4f})",
                }
            )
        prev_score = score

    # Current/latest score
    latest = points_sorted[-1]
    latest_score = float(latest.get("epss", 0))
    if latest["date"] != first["date"]:
        events.append(
            {
                "date": _parse_iso(latest["date"]),
                "source": "EPSS",
                "event": "Latest EPSS score",
                "detail": f"Current score: {latest_score:.4f} ({latest_score * 100:.2f}%)",
            }
        )

    return events


def _fetch_github_advisory_dates(cve_id: str) -> list[dict[str, Any]]:
    """Query GitHub Advisory Database for publish/update dates."""
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = _get_with_retry(
            "https://api.github.com/advisories",
            params={"cve_id": cve_id.upper()},
            headers=headers,
        )
        advisories = resp.json()
    except Exception:
        return []

    if not isinstance(advisories, list) or not advisories:
        return []

    events: list[dict[str, Any]] = []
    adv = advisories[0]

    published_at = adv.get("published_at", "")
    if published_at:
        events.append(
            {
                "date": _parse_iso(published_at),
                "source": "GitHub Advisory",
                "event": "GitHub advisory published",
                "detail": f"GHSA: {adv.get('ghsa_id', 'N/A')} — {adv.get('summary', '')[:100]}",
            }
        )

    updated_at = adv.get("updated_at", "")
    if updated_at and updated_at != published_at:
        events.append(
            {
                "date": _parse_iso(updated_at),
                "source": "GitHub Advisory",
                "event": "GitHub advisory updated",
                "detail": f"GHSA {adv.get('ghsa_id', 'N/A')} last updated",
            }
        )

    return events


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def _parse_iso(date_str: str) -> str:
    """Parse various date formats to ISO 8601 date string (YYYY-MM-DD)."""
    if not date_str:
        return ""

    # Already a simple date
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    # ISO 8601 with time component
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Last resort: take first 10 chars if they look like a date
    if len(date_str) >= 10 and re.match(r"^\d{4}-\d{2}-\d{2}", date_str):
        return date_str[:10]

    return date_str


# ---------------------------------------------------------------------------
# Timeline assembly
# ---------------------------------------------------------------------------


def build_timeline(cve_id: str) -> dict[str, Any]:
    """Assemble the full CVE timeline from all sources."""
    cve_id = cve_id.strip().upper()

    if not re.match(r"^CVE-\d{4}-\d{4,}$", cve_id):
        return {"error": f"Invalid CVE ID format: {cve_id}", "events": []}

    all_events: list[dict[str, Any]] = []

    # Gather events from all sources
    all_events.extend(_fetch_nvd_events(cve_id))
    all_events.extend(_fetch_epss_events(cve_id))

    # Only fetch KEV if NVD didn't already give us the date
    has_kev = any(e.get("source") == "CISA KEV" for e in all_events)
    if not has_kev:
        all_events.extend(_fetch_kev_date(cve_id))

    all_events.extend(_fetch_github_advisory_dates(cve_id))

    # Deduplicate KEV events (NVD may give us one, and direct KEV query too)
    kev_events = [e for e in all_events if e.get("event") == "Added to CISA KEV"]
    if len(kev_events) > 1:
        # Keep the one with more detail
        best = max(kev_events, key=lambda e: len(e.get("detail", "")))
        all_events = [e for e in all_events if e.get("event") != "Added to CISA KEV"]
        all_events.append(best)

    # Filter out events without dates and sort chronologically
    dated_events = [e for e in all_events if e.get("date")]
    undated_events = [e for e in all_events if not e.get("date")]

    dated_events.sort(key=lambda e: e["date"])

    # Compute time deltas between key events
    summary = _compute_summary(dated_events, cve_id)

    return {
        "cve_id": cve_id,
        "events": dated_events,
        "undated_references": undated_events,
        "summary": summary,
    }


def _compute_summary(events: list[dict[str, Any]], cve_id: str) -> dict[str, Any]:
    """Compute key time deltas between milestone events."""
    summary: dict[str, Any] = {"cve_id": cve_id, "total_events": len(events)}

    # Find key dates
    nvd_publish = next((e["date"] for e in events if e["event"] == "CVE published"), None)
    kev_date = next((e["date"] for e in events if "KEV" in e.get("event", "")), None)
    ghsa_date = next((e["date"] for e in events if "advisory published" in e.get("event", "")), None)
    first_epss = next((e["date"] for e in events if e["event"] == "First EPSS score"), None)

    summary["nvd_published"] = nvd_publish
    summary["kev_added"] = kev_date
    summary["ghsa_published"] = ghsa_date
    summary["first_epss"] = first_epss

    # Time from publish to KEV (days to confirmed exploitation)
    if nvd_publish and kev_date:
        delta = _days_between(nvd_publish, kev_date)
        summary["days_publish_to_kev"] = delta

    # Time from publish to advisory
    if nvd_publish and ghsa_date:
        delta = _days_between(nvd_publish, ghsa_date)
        summary["days_publish_to_ghsa"] = delta

    # Has active exploitation
    summary["in_kev"] = kev_date is not None

    # Has EPSS spikes
    summary["epss_spikes"] = sum(1 for e in events if "spike" in e.get("event", "").lower())

    return summary


def _days_between(date1: str, date2: str) -> int:
    """Calculate days between two YYYY-MM-DD date strings."""
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        d2 = datetime.strptime(date2, "%Y-%m-%d")
        return abs((d2 - d1).days)
    except (ValueError, TypeError):
        return -1


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_timeline_text(result: dict[str, Any]) -> str:
    """Format the timeline result as human-readable text."""
    if "error" in result and not result.get("events"):
        return f"Error: {result['error']}"

    lines: list[str] = []
    cve_id = result.get("cve_id", "")
    lines.append(f"╔══ CVE Timeline: {cve_id} ══╗")
    lines.append("")

    events = result.get("events", [])
    if not events:
        lines.append("  No timeline events found.")
        return "\n".join(lines)

    # Group by date for cleaner display
    current_date = ""
    for event in events:
        date = event.get("date", "unknown")
        if date != current_date:
            current_date = date
            lines.append(f"  ┌─ {date} ─────────────────────")

        source = event.get("source", "")
        evt = event.get("event", "")
        detail = event.get("detail", "")
        lines.append(f"  │ [{source}] {evt}")
        if detail:
            lines.append(f"  │   └─ {detail}")

    lines.append("  └───────────────────────────────────")
    lines.append("")

    # Summary
    summary = result.get("summary", {})
    lines.append("  ── Summary ──")
    if summary.get("days_publish_to_kev") is not None:
        days = summary["days_publish_to_kev"]
        lines.append(f"  • CVE publish → KEV: {days} days")
    if summary.get("days_publish_to_ghsa") is not None:
        days = summary["days_publish_to_ghsa"]
        lines.append(f"  • CVE publish → GitHub advisory: {days} days")
    lines.append(f"  • In CISA KEV: {'Yes' if summary.get('in_kev') else 'No'}")
    lines.append(f"  • EPSS spikes: {summary.get('epss_spikes', 0)}")
    lines.append(f"  • Total events: {summary.get('total_events', 0)}")

    # Undated references
    undated = result.get("undated_references", [])
    if undated:
        lines.append("")
        lines.append("  ── Patch References (undated) ──")
        for ref in undated:
            lines.append(f"  • {ref.get('event', '')}: {ref.get('detail', '')}")

    return "\n".join(lines)


def format_timeline_json(result: dict[str, Any]) -> str:
    """Format the timeline result as JSON."""
    import json

    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Strands tool handler
# ---------------------------------------------------------------------------


def handler(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands SDK tool handler entry point."""
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")

    result = build_timeline(cve_id)
    output_text = format_timeline_text(result)
    log_tool_output_size("get_cve_timeline", output_text)

    return {
        "toolUseId": tool["toolUseId"],
        "status": "error" if "error" in result and not result.get("events") else "success",
        "content": [{"text": output_text}],
    }
