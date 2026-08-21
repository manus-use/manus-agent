"""
Tool for reconstructing the full event timeline of a CVE.

Gathers dates from multiple sources — NVD (publish/modify), EPSS (score history),
CISA KEV (exploitation confirmation), and NVD references (patch/advisory dates) —
to produce a chronological narrative of the vulnerability lifecycle.
"""

from __future__ import annotations

import json as _json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

# ---------------------------------------------------------------------------
# Retry / back-off constants
# ---------------------------------------------------------------------------
_MAX_RETRIES = int(os.environ.get("CVE_TIMELINE_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.environ.get("CVE_TIMELINE_RETRY_BASE_DELAY", "2"))
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# CVE ID pattern
_CVE_PATTERN = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)

TOOL_SPEC = {
    "name": "cve_timeline",
    "description": (
        "Reconstructs the full event timeline for a CVE: NVD publish date, "
        "modification date, EPSS score history with spike detection, CISA KEV "
        "addition date, and patch/advisory dates extracted from NVD references. "
        "Produces a chronological list of events showing how quickly a vulnerability "
        "was weaponised and fixed. Use after get_nvd_data for deeper temporal context."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier (e.g., 'CVE-2021-44228').",
                },
            },
            "required": ["cve_id"],
        }
    },
}


# ---------------------------------------------------------------------------
# HTTP helper with retry
# ---------------------------------------------------------------------------


def _get_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = 20,
) -> requests.Response:
    """GET with exponential back-off retry on transient errors."""
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        if attempt > 0:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            time.sleep(delay)
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUS:
                last_exc = requests.exceptions.HTTPError(f"HTTP {resp.status_code}", response=resp)
                if attempt < _MAX_RETRIES:
                    continue
                raise last_exc
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError:
            raise
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                continue

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------


def _fetch_nvd_dates(cve_id: str) -> dict[str, Any]:
    """Fetch publish/modify dates and references from NVD."""
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id.upper()}"
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key

    try:
        resp = _get_with_retry(url, headers=headers)
        data = resp.json()
    except Exception as exc:
        return {"error": str(exc)}

    vulns = data.get("vulnerabilities", [])
    if not vulns:
        return {"error": "CVE not found in NVD"}

    cve_data = vulns[0].get("cve", {})
    result: dict[str, Any] = {}

    # Publication and modification dates
    published = cve_data.get("published")
    if published:
        result["published"] = published[:10]  # YYYY-MM-DD

    last_modified = cve_data.get("lastModified")
    if last_modified:
        result["last_modified"] = last_modified[:10]

    # CVSS score for context
    metrics = cve_data.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        metric_list = metrics.get(key, [])
        if metric_list:
            cvss_data = metric_list[0].get("cvssData", {})
            result["cvss_score"] = cvss_data.get("baseScore")
            result["cvss_severity"] = cvss_data.get("baseSeverity")
            break

    # Extract reference dates (patches, advisories)
    references = cve_data.get("references", [])
    ref_events: list[dict[str, str]] = []
    for ref in references:
        tags = ref.get("tags", [])
        url_str = ref.get("url", "")
        # Look for patch and advisory references
        if "Patch" in tags or "Vendor Advisory" in tags or "Third Party Advisory" in tags:
            ref_type = "patch" if "Patch" in tags else "advisory"
            ref_events.append({"type": ref_type, "url": url_str, "tags": tags})

    result["references"] = ref_events
    return result


def _fetch_epss_history(cve_id: str, days: int = 90) -> dict[str, Any]:
    """Fetch EPSS score history from FIRST.org."""
    url = "https://api.first.org/data/v1/epss"
    params: dict[str, Any] = {
        "cve": cve_id.upper(),
        "scope": "time-series",
        "limit": min(days, 365),
    }

    try:
        resp = _get_with_retry(url, params=params)
        data = resp.json()
    except Exception as exc:
        return {"error": str(exc)}

    epss_data = data.get("data", [])
    if not epss_data:
        return {"error": "No EPSS data available"}

    cve_entry = epss_data[0] if epss_data else {}
    time_series = cve_entry.get("time-series", cve_entry.get("timeSeries", []))

    if not time_series:
        # Might be a single-point response
        epss_val = cve_entry.get("epss")
        date_val = cve_entry.get("date")
        if epss_val and date_val:
            return {
                "current_score": float(epss_val),
                "first_seen": date_val,
                "history": [{"date": date_val, "epss": float(epss_val)}],
            }
        return {"error": "No EPSS time-series data"}

    # Sort oldest first
    points = sorted(time_series, key=lambda p: p.get("date", ""))
    history = [{"date": p["date"], "epss": float(p["epss"])} for p in points]

    # Detect spikes (>= 0.10 jump between consecutive days)
    spikes: list[dict[str, Any]] = []
    for i in range(1, len(history)):
        jump = history[i]["epss"] - history[i - 1]["epss"]
        if jump >= 0.10:
            spikes.append(
                {
                    "date": history[i]["date"],
                    "from_score": history[i - 1]["epss"],
                    "to_score": history[i]["epss"],
                    "jump": round(jump, 4),
                }
            )

    result: dict[str, Any] = {
        "current_score": history[-1]["epss"] if history else None,
        "first_seen": history[0]["date"] if history else None,
        "last_seen": history[-1]["date"] if history else None,
        "history_points": len(history),
        "spikes": spikes,
    }
    return result


def _fetch_kev_date(cve_id: str) -> dict[str, Any]:
    """Check CISA KEV catalog for the CVE and return the date added."""
    url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

    try:
        resp = _get_with_retry(url)
        data = resp.json()
    except Exception as exc:
        return {"error": str(exc)}

    cve_upper = cve_id.upper()
    for vuln in data.get("vulnerabilities", []):
        if vuln.get("cveID", "").upper() == cve_upper:
            return {
                "in_kev": True,
                "date_added": vuln.get("dateAdded"),
                "due_date": vuln.get("dueDate"),
                "known_ransomware": vuln.get("knownRansomwareCampaignUse", "Unknown"),
                "vendor": vuln.get("vendorProject"),
                "product": vuln.get("product"),
                "short_description": vuln.get("shortDescription"),
            }

    return {"in_kev": False}


# ---------------------------------------------------------------------------
# Timeline assembly
# ---------------------------------------------------------------------------


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse a YYYY-MM-DD date string to datetime."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _days_between(date1: str | None, date2: str | None) -> int | None:
    """Calculate days between two date strings."""
    d1 = _parse_date(date1)
    d2 = _parse_date(date2)
    if d1 and d2:
        return abs((d2 - d1).days)
    return None


def build_timeline(cve_id: str) -> dict[str, Any]:
    """
    Assemble the full CVE timeline from multiple sources.

    Returns a dict with:
      - events: list of {date, event_type, description} sorted chronologically
      - summary: high-level stats (time-to-exploit, time-to-patch, etc.)
      - sources: which data sources were successfully queried
    """
    cve_id = cve_id.upper().strip()

    if not _CVE_PATTERN.match(cve_id):
        return {"error": f"Invalid CVE ID format: {cve_id}"}

    events: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    summary: dict[str, Any] = {"cve_id": cve_id}

    # --- NVD data ---
    nvd = _fetch_nvd_dates(cve_id)
    if "error" in nvd:
        sources["nvd"] = f"error: {nvd['error']}"
    else:
        sources["nvd"] = "ok"
        if nvd.get("published"):
            events.append(
                {
                    "date": nvd["published"],
                    "event_type": "nvd_published",
                    "description": "CVE published in NVD",
                }
            )
            summary["nvd_published"] = nvd["published"]

        if nvd.get("last_modified") and nvd.get("last_modified") != nvd.get("published"):
            events.append(
                {
                    "date": nvd["last_modified"],
                    "event_type": "nvd_modified",
                    "description": "CVE record last modified in NVD",
                }
            )

        if nvd.get("cvss_score"):
            summary["cvss_score"] = nvd["cvss_score"]
            summary["cvss_severity"] = nvd.get("cvss_severity")

        # Reference-based events (patches/advisories)
        for ref in nvd.get("references", []):
            ref_type = ref.get("type", "reference")
            url_str = ref.get("url", "")
            # Try to extract date from GitHub commit URLs or common patterns
            date_from_url = _extract_date_from_url(url_str)
            if date_from_url:
                label = "Patch published" if ref_type == "patch" else "Advisory published"
                events.append(
                    {
                        "date": date_from_url,
                        "event_type": f"{ref_type}_released",
                        "description": f"{label}: {url_str[:80]}",
                    }
                )

    # --- EPSS history ---
    epss = _fetch_epss_history(cve_id)
    if "error" in epss:
        sources["epss"] = f"error: {epss['error']}"
    else:
        sources["epss"] = "ok"
        if epss.get("first_seen"):
            events.append(
                {
                    "date": epss["first_seen"],
                    "event_type": "epss_first_score",
                    "description": f"First EPSS score recorded ({epss.get('current_score', 'N/A')} current)",
                }
            )
            summary["epss_current"] = epss.get("current_score")

        # EPSS spikes are significant timeline events
        for spike in epss.get("spikes", []):
            events.append(
                {
                    "date": spike["date"],
                    "event_type": "epss_spike",
                    "description": (
                        f"EPSS spike: {spike['from_score']:.4f} → {spike['to_score']:.4f} (+{spike['jump']:.4f})"
                    ),
                }
            )

    # --- CISA KEV ---
    kev = _fetch_kev_date(cve_id)
    if "error" in kev:
        sources["kev"] = f"error: {kev['error']}"
    else:
        sources["kev"] = "ok"
        summary["in_kev"] = kev.get("in_kev", False)
        if kev.get("in_kev") and kev.get("date_added"):
            events.append(
                {
                    "date": kev["date_added"],
                    "event_type": "kev_added",
                    "description": "Added to CISA KEV (confirmed exploitation in the wild)",
                }
            )
            summary["kev_date_added"] = kev["date_added"]
            summary["kev_due_date"] = kev.get("due_date")
            summary["kev_ransomware"] = kev.get("known_ransomware")

    # --- Sort events chronologically ---
    events.sort(key=lambda e: e.get("date", "9999-99-99"))

    # --- Compute time deltas ---
    nvd_pub = summary.get("nvd_published")
    kev_added = summary.get("kev_date_added")
    if nvd_pub and kev_added:
        days = _days_between(nvd_pub, kev_added)
        if days is not None:
            summary["days_publish_to_exploit"] = days

    # Time from publish to first EPSS spike
    if nvd_pub and epss.get("spikes"):
        first_spike_date = epss["spikes"][0]["date"]
        days = _days_between(nvd_pub, first_spike_date)
        if days is not None:
            summary["days_publish_to_epss_spike"] = days

    return {
        "cve_id": cve_id,
        "events": events,
        "summary": summary,
        "sources": sources,
        "event_count": len(events),
    }


def _extract_date_from_url(url: str) -> str | None:
    """
    Attempt to extract a date from common URL patterns.
    Supports GitHub commit/release URLs and advisory publication URLs.
    """
    # GitHub release/tag URLs often contain dates or version timestamps
    # e.g., https://github.com/org/repo/commit/abc123
    # We can't reliably get dates from these without an API call,
    # so we only extract from URLs with explicit date patterns.

    # Pattern: YYYY-MM-DD in URL path
    match = re.search(r"(\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))", url)
    if match:
        return match.group(1)

    # Pattern: YYYY/MM/DD in URL path
    match = re.search(r"/(\d{4})/(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/", url)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    return None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_timeline_text(result: dict[str, Any]) -> str:
    """Format timeline as human-readable text."""
    if "error" in result:
        return f"Error: {result['error']}"

    lines: list[str] = []
    cve_id = result["cve_id"]
    summary = result.get("summary", {})
    events = result.get("events", [])

    lines.append(f"CVE Timeline: {cve_id}")
    lines.append("=" * (len(f"CVE Timeline: {cve_id}")))
    lines.append("")

    # Summary section
    if summary.get("cvss_score"):
        lines.append(f"CVSS: {summary['cvss_score']} ({summary.get('cvss_severity', 'N/A')})")
    if summary.get("epss_current") is not None:
        lines.append(f"EPSS (current): {summary['epss_current']:.4f}")
    if summary.get("in_kev"):
        lines.append("KEV Status: ⚠️  IN CISA KEV (actively exploited)")
    else:
        lines.append("KEV Status: Not in CISA KEV")
    lines.append("")

    # Events
    if events:
        lines.append("Timeline Events:")
        lines.append("-" * 50)
        for event in events:
            date = event.get("date", "Unknown")
            etype = event.get("event_type", "")
            desc = event.get("description", "")
            icon = _event_icon(etype)
            lines.append(f"  {date}  {icon} {desc}")
        lines.append("")
    else:
        lines.append("No timeline events found.")
        lines.append("")

    # Time deltas
    deltas: list[str] = []
    if summary.get("days_publish_to_exploit") is not None:
        deltas.append(f"  Publish → KEV exploit: {summary['days_publish_to_exploit']} days")
    if summary.get("days_publish_to_epss_spike") is not None:
        deltas.append(f"  Publish → EPSS spike: {summary['days_publish_to_epss_spike']} days")
    if deltas:
        lines.append("Key Intervals:")
        lines.extend(deltas)
        lines.append("")

    # Sources
    sources = result.get("sources", {})
    lines.append("Data Sources:")
    for src, status in sources.items():
        icon = "✓" if status == "ok" else "✗"
        lines.append(f"  {icon} {src}: {status}")

    return "\n".join(lines)


def format_timeline_json(result: dict[str, Any]) -> str:
    """Format timeline as JSON string."""
    return _json.dumps(result, indent=2, default=str)


def _event_icon(event_type: str) -> str:
    """Return an icon for the event type."""
    icons = {
        "nvd_published": "📋",
        "nvd_modified": "✏️",
        "epss_first_score": "📊",
        "epss_spike": "📈",
        "kev_added": "🚨",
        "patch_released": "🩹",
        "advisory_released": "📢",
    }
    return icons.get(event_type, "•")


# ---------------------------------------------------------------------------
# Strands tool handler
# ---------------------------------------------------------------------------


def handler(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands-compatible tool handler."""
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")

    if not cve_id:
        content = "Error: cve_id is required"
        log_tool_output_size("cve_timeline", content)
        return {"status": "error", "content": [{"text": content}]}

    result = build_timeline(cve_id)
    output = format_timeline_json(result)
    log_tool_output_size("cve_timeline", output)

    status = "error" if "error" in result else "success"
    return {"status": status, "content": [{"text": output}]}
