"""
Tool for reconstructing the full event timeline of a CVE.

Assembles chronological lifecycle events from multiple sources:
- NVD (published, last modified dates, CVSS assignment)
- EPSS (first score appearance, current score)
- CISA KEV (date added to Known Exploited Vulnerabilities catalog)
- GitHub Advisory Database (advisory publish/update dates)
- VulnCheck KEV (additional exploitation context)

Useful for understanding how quickly a vulnerability was weaponised and fixed,
and for communicating risk timelines to stakeholders.
"""

from __future__ import annotations

import os
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

TOOL_SPEC = {
    "name": "get_cve_timeline",
    "description": (
        "Reconstructs the full event timeline for a CVE by querying multiple "
        "sources: NVD publish/modify dates, EPSS first-seen score, CISA KEV "
        "addition date, and GitHub advisory dates. Returns a chronological list "
        "of events that shows how quickly a vulnerability was disclosed, scored, "
        "weaponised, and (if applicable) added to mandatory-patch catalogs. "
        "Use after get_nvd_data for a temporal view of vulnerability lifecycle."
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
    for attempt in range(_MAX_RETRIES):
        if attempt > 0:
            time.sleep(_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                last_exc = requests.exceptions.HTTPError(response=resp)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt >= _MAX_RETRIES - 1:
                break
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Source fetchers — each returns a list of event dicts
# ---------------------------------------------------------------------------


def _fetch_nvd_events(cve_id: str) -> list[dict[str, Any]]:
    """Fetch NVD published/modified dates and CVSS info."""
    events: list[dict[str, Any]] = []
    headers: dict[str, str] = {}
    api_key = os.environ.get("NVD_API_KEY", "").strip()
    if api_key:
        headers["apiKey"] = api_key

    try:
        resp = _get_with_retry(
            "https://services.nvd.nist.gov/rest/json/cves/2.0",
            headers=headers,
            params={"cveId": cve_id.upper()},
            timeout=20,
        )
        data = resp.json()
    except Exception:
        return events

    vulnerabilities = data.get("vulnerabilities", [])
    if not vulnerabilities:
        return events

    cve_data = vulnerabilities[0].get("cve", {})

    # Published date
    published = cve_data.get("published", "")
    if published:
        events.append(
            {
                "date": published[:10],
                "timestamp": published,
                "source": "NVD",
                "event": "CVE published",
                "detail": f"{cve_id.upper()} record created in NVD",
            }
        )

    # Last modified date (only if different from published)
    last_modified = cve_data.get("lastModified", "")
    if last_modified and last_modified[:10] != published[:10]:
        events.append(
            {
                "date": last_modified[:10],
                "timestamp": last_modified,
                "source": "NVD",
                "event": "NVD record updated",
                "detail": "NVD entry last modified (re-analysis, score change, or reference update)",
            }
        )

    # CVSS score assignment
    metrics = cve_data.get("metrics", {})
    cvss_score = None
    cvss_version = None
    # Try CVSS 3.1, then 3.0, then 2.0
    for version_key, ver_label in [
        ("cvssMetricV31", "3.1"),
        ("cvssMetricV30", "3.0"),
        ("cvssMetricV2", "2.0"),
    ]:
        metric_list = metrics.get(version_key, [])
        if metric_list:
            cvss_data = metric_list[0].get("cvssData", {})
            cvss_score = cvss_data.get("baseScore")
            cvss_version = ver_label
            severity = cvss_data.get("baseSeverity", "")
            if cvss_score and published:
                events.append(
                    {
                        "date": published[:10],
                        "timestamp": published,
                        "source": "NVD",
                        "event": f"CVSS {cvss_version} assigned",
                        "detail": f"Base score: {cvss_score} ({severity})",
                    }
                )
            break

    return events


def _fetch_epss_events(cve_id: str) -> list[dict[str, Any]]:
    """Fetch EPSS first-seen date and current score."""
    events: list[dict[str, Any]] = []
    try:
        resp = _get_with_retry(
            "https://api.first.org/data/v1/epss",
            params={"cve": cve_id.upper(), "scope": "time-series"},
            timeout=20,
        )
        data = resp.json()
    except Exception:
        return events

    entries = data.get("data", [])
    if not entries:
        return events

    entry = entries[0]
    time_series = entry.get("time-series", [])

    if time_series:
        # Sort by date to find earliest
        sorted_series = sorted(time_series, key=lambda x: x.get("date", ""))
        earliest = sorted_series[0]
        latest = sorted_series[-1]

        events.append(
            {
                "date": earliest["date"],
                "timestamp": earliest["date"] + "T00:00:00Z",
                "source": "EPSS",
                "event": "EPSS tracking started",
                "detail": f"First EPSS score: {float(earliest.get('epss', 0)):.4f} "
                f"(percentile: {float(earliest.get('percentile', 0)):.2%})",
            }
        )

        # Current/latest score
        current_epss = float(entry.get("epss", latest.get("epss", 0)))
        current_percentile = float(entry.get("percentile", latest.get("percentile", 0)))
        current_date = entry.get("date", latest.get("date", ""))
        if current_date and current_date != earliest["date"]:
            events.append(
                {
                    "date": current_date,
                    "timestamp": current_date + "T00:00:00Z",
                    "source": "EPSS",
                    "event": "EPSS current score",
                    "detail": f"Score: {current_epss:.4f} (percentile: {current_percentile:.2%})",
                }
            )

    return events


def _fetch_kev_events(cve_id: str) -> list[dict[str, Any]]:
    """Check CISA KEV catalog for the CVE's addition date."""
    events: list[dict[str, Any]] = []
    try:
        resp = _get_with_retry(
            "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
            timeout=20,
        )
        data = resp.json()
    except Exception:
        return events

    cve_upper = cve_id.upper()
    for vuln in data.get("vulnerabilities", []):
        if vuln.get("cveID", "").upper() == cve_upper:
            date_added = vuln.get("dateAdded", "")
            due_date = vuln.get("dueDate", "")
            short_desc = vuln.get("shortDescription", "")
            if date_added:
                detail = "Added to CISA Known Exploited Vulnerabilities catalog"
                if short_desc:
                    detail += f" — {short_desc[:120]}"
                events.append(
                    {
                        "date": date_added,
                        "timestamp": date_added + "T00:00:00Z",
                        "source": "CISA KEV",
                        "event": "Added to CISA KEV",
                        "detail": detail,
                    }
                )
            if due_date:
                events.append(
                    {
                        "date": due_date,
                        "timestamp": due_date + "T00:00:00Z",
                        "source": "CISA KEV",
                        "event": "KEV remediation deadline",
                        "detail": f"Federal agencies must remediate by {due_date}",
                    }
                )
            break

    return events


def _fetch_github_advisory_events(cve_id: str) -> list[dict[str, Any]]:
    """Query GitHub Advisory Database for advisory publish/update dates."""
    events: list[dict[str, Any]] = []
    github_token = os.environ.get("GITHUB_TOKEN", "").strip()

    # Use the REST API search endpoint (no auth required, but token helps rate limits)
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    try:
        resp = _get_with_retry(
            "https://api.github.com/advisories",
            headers=headers,
            params={"cve_id": cve_id.upper(), "per_page": "5"},
            timeout=20,
        )
        advisories = resp.json()
    except Exception:
        return events

    if not isinstance(advisories, list) or not advisories:
        return events

    advisory = advisories[0]
    ghsa_id = advisory.get("ghsa_id", "")
    published_at = advisory.get("published_at", "")
    updated_at = advisory.get("updated_at", "")
    severity = advisory.get("severity", "")

    if published_at:
        detail = f"GitHub Security Advisory {ghsa_id} published"
        if severity:
            detail += f" (severity: {severity})"
        events.append(
            {
                "date": published_at[:10],
                "timestamp": published_at,
                "source": "GitHub Advisory",
                "event": "GHSA published",
                "detail": detail,
            }
        )

    if updated_at and updated_at[:10] != published_at[:10]:
        events.append(
            {
                "date": updated_at[:10],
                "timestamp": updated_at,
                "source": "GitHub Advisory",
                "event": "GHSA updated",
                "detail": f"Advisory {ghsa_id} last updated",
            }
        )

    return events


def _fetch_vulncheck_kev_events(cve_id: str) -> list[dict[str, Any]]:
    """Fetch VulnCheck KEV data for additional exploitation context."""
    events: list[dict[str, Any]] = []
    api_key = os.environ.get("VULNCHECK_API_KEY", "").strip()
    if not api_key:
        return events

    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    try:
        resp = _get_with_retry(
            "https://api.vulncheck.com/v3/index/vulncheck-kev",
            headers=headers,
            params={"cve": cve_id.upper()},
            timeout=20,
        )
        data = resp.json()
    except Exception:
        return events

    entries = data.get("data", [])
    if not entries:
        return events

    entry = entries[0]
    date_added = entry.get("date_added", "")
    if date_added:
        events.append(
            {
                "date": date_added[:10],
                "timestamp": date_added if "T" in date_added else date_added + "T00:00:00Z",
                "source": "VulnCheck KEV",
                "event": "VulnCheck KEV entry",
                "detail": "Confirmed exploitation tracked by VulnCheck",
            }
        )

    return events


# ---------------------------------------------------------------------------
# Main assembly logic
# ---------------------------------------------------------------------------


def build_timeline(cve_id: str) -> dict[str, Any]:
    """
    Assemble a chronological timeline for the given CVE from all available sources.

    Returns a dict with:
      - cve_id: normalised CVE ID
      - events: sorted list of event dicts (date, source, event, detail)
      - sources_queried: list of source names attempted
      - sources_with_data: list of sources that returned events
      - span_days: days between earliest and latest event (or None)
    """
    cve_id = cve_id.strip().upper()
    all_events: list[dict[str, Any]] = []
    sources_with_data: list[str] = []

    # Fetch from each source
    source_fetchers = [
        ("NVD", _fetch_nvd_events),
        ("EPSS", _fetch_epss_events),
        ("CISA KEV", _fetch_kev_events),
        ("GitHub Advisory", _fetch_github_advisory_events),
        ("VulnCheck KEV", _fetch_vulncheck_kev_events),
    ]

    sources_queried = [name for name, _ in source_fetchers]

    for source_name, fetcher in source_fetchers:
        events = fetcher(cve_id)
        if events:
            sources_with_data.append(source_name)
            all_events.extend(events)

    # Sort events chronologically by date (then by source for ties)
    source_priority = {
        "NVD": 0,
        "EPSS": 1,
        "CISA KEV": 2,
        "GitHub Advisory": 3,
        "VulnCheck KEV": 4,
    }
    all_events.sort(key=lambda e: (e.get("date", "9999-99-99"), source_priority.get(e.get("source", ""), 99)))

    # Calculate span
    span_days: int | None = None
    if len(all_events) >= 2:
        try:
            earliest = datetime.strptime(all_events[0]["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            latest = datetime.strptime(all_events[-1]["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            span_days = (latest - earliest).days
        except (ValueError, KeyError):
            pass

    return {
        "cve_id": cve_id,
        "events": all_events,
        "sources_queried": sources_queried,
        "sources_with_data": sources_with_data,
        "span_days": span_days,
        "event_count": len(all_events),
    }


# ---------------------------------------------------------------------------
# Strands tool handler
# ---------------------------------------------------------------------------


def handler(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands SDK tool handler for get_cve_timeline."""
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "").strip()

    if not cve_id:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [{"text": "cve_id is required"}],
        }

    try:
        result = build_timeline(cve_id)
    except Exception as exc:
        return {
            "toolUseId": tool["toolUseId"],
            "status": "error",
            "content": [{"text": f"Timeline assembly failed: {exc}"}],
        }

    import json

    output_text = json.dumps(result, indent=2)
    log_tool_output_size("get_cve_timeline", output_text)

    return {
        "toolUseId": tool["toolUseId"],
        "status": "success",
        "content": [{"text": output_text}],
    }
