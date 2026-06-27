"""
Tool for building a structured timeline for a CVE.

Aggregates key dates from multiple sources and produces a chronological view:

  disclosure_date  -> nvd_published
  patch_date       -> earliest fixing commit found in GHSA / NVD references
  poc_date         -> earliest public PoC repo (Trickest CVE index / GitHub)
  epss_spike_date  -> date of the largest 7-day EPSS jump (if >= threshold)
  kev_date         -> date added to the CISA KEV catalogue

Each event record contains:
  - event  : human-readable label
  - date   : ISO-8601 date string (YYYY-MM-DD) or None
  - source : where the date came from
  - url    : reference URL (optional)

Velocity metrics (in days):
  - disclosure_to_patch_days        : disclosure to earliest patch commit
  - disclosure_to_poc_days          : disclosure to first public PoC
  - poc_to_kev_days                 : PoC publication to CISA KEV listing
  - disclosure_to_weaponized_days   : disclosure to first confirmed exploitation

disclosure_to_poc <= 7 days is flagged as "fast weaponisation".
"""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from typing import Any

import requests
from strands.types.tools import ToolResult, ToolUse

from manus_use.tools.tool_output_logger import log_tool_output_size

_NVD_CVE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
_GHSA_API_URL = "https://api.github.com/advisories?cve_id={cve_id}"
_EPSS_API_URL = "https://api.first.org/data/v1/epss"
_CISA_KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
_TRICKEST_URL = "https://raw.githubusercontent.com/trickest/cve/main/{year}/{cve_id}.md"

_EPSS_SPIKE_THRESHOLD = 0.10

TOOL_SPEC = {
    "name": "get_cve_timeline",
    "description": (
        "Builds a structured timeline for a CVE by aggregating key dates from NVD, GHSA, "
        "CISA KEV, FIRST.org EPSS, and the Trickest public PoC index. "
        "Returns a chronological list of events (disclosure, patch, first PoC, EPSS "
        "exploitation spike, CISA KEV listing) and velocity metrics showing how quickly "
        "the vulnerability moved from disclosure to weaponisation. "
        "Use this to understand the full lifecycle of a CVE at a glance."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier to look up (e.g., CVE-2024-3094).",
                },
            },
            "required": ["cve_id"],
        }
    },
}


def _iso_date(s: str | None) -> str | None:
    """Normalise various timestamp formats to YYYY-MM-DD, or return None."""
    if not s:
        return None
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", str(s))
    if m:
        return m.group(1)
    return None


def _days_between(a: str | None, b: str | None) -> int | None:
    """Return difference in days (b - a), or None if either date is missing."""
    if not a or not b:
        return None
    try:
        da = datetime.strptime(a, "%Y-%m-%d").date()
        db = datetime.strptime(b, "%Y-%m-%d").date()
        return (db - da).days
    except ValueError:
        return None


def _github_headers() -> dict[str, str]:
    """Build GitHub API headers, including auth token if GITHUB_TOKEN is set."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _commit_date(api_url: str) -> str | None:
    """Call a GitHub commit API URL and return its committer/author date."""
    try:
        resp = requests.get(api_url, headers=_github_headers(), timeout=10)
        if not resp.ok:
            return None
        data = resp.json()
        raw = data.get("commit", {}).get("committer", {}).get("date") or data.get("commit", {}).get("author", {}).get(
            "date"
        )
        return _iso_date(raw)
    except Exception:
        return None


def _fetch_nvd_info(cve_id: str) -> dict[str, Any]:
    """Return NVD metadata: published, lastModified, and embedded KEV date."""
    try:
        resp = requests.get(_NVD_CVE_URL.format(cve_id=cve_id), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return {}
        cve = vulns[0].get("cve", {})
        return {
            "published": _iso_date(cve.get("published")),
            "last_modified": _iso_date(cve.get("lastModified")),
            "kev_date": _iso_date(cve.get("cisaExploitAdd")),
            "nvd_url": f"https://nvd.nist.gov/vuln/detail/{cve_id}",
        }
    except Exception:
        return {}


def _fetch_ghsa_patch_date(cve_id: str) -> tuple[str | None, str | None]:
    """Query GHSA for a fixing-commit date. Returns (date_str, url)."""
    try:
        resp = requests.get(
            _GHSA_API_URL.format(cve_id=cve_id),
            headers=_github_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        advisories = resp.json()
        if not isinstance(advisories, list) or not advisories:
            return None, None

        advisory = advisories[0]
        for ref in advisory.get("references", []):
            url = ref if isinstance(ref, str) else ref.get("url", "")
            m = re.search(r"github\.com/([^/]+/[^/]+)/commit/([0-9a-f]{7,40})", url)
            if m:
                api_url = f"https://api.github.com/repos/{m.group(1)}/commits/{m.group(2)}"
                cd = _commit_date(api_url)
                if cd:
                    return cd, url
                published = _iso_date(advisory.get("published_at") or advisory.get("updated_at"))
                return published, url

        published = _iso_date(advisory.get("published_at") or advisory.get("updated_at"))
        return published, advisory.get("html_url")
    except Exception:
        return None, None


def _fetch_nvd_patch_date(cve_id: str) -> tuple[str | None, str | None]:
    """Scan NVD refs for GitHub commit URLs. Returns (date_str, commit_url)."""
    try:
        resp = requests.get(_NVD_CVE_URL.format(cve_id=cve_id), timeout=15)
        resp.raise_for_status()
        data = resp.json()
        vulns = data.get("vulnerabilities", [])
        if not vulns:
            return None, None

        hits: list[tuple[str, str]] = []
        for ref in vulns[0].get("cve", {}).get("references", []):
            url = ref.get("url", "")
            m = re.search(r"github\.com/([^/]+/[^/]+)/commit/([0-9a-f]{7,40})", url)
            if m:
                api_url = f"https://api.github.com/repos/{m.group(1)}/commits/{m.group(2)}"
                cd = _commit_date(api_url)
                if cd:
                    hits.append((cd, url))

        if hits:
            hits.sort(key=lambda x: x[0])
            return hits[0]
        return None, None
    except Exception:
        return None, None


def _fetch_trickest_poc(cve_id: str) -> tuple[str | None, str | None]:
    """Check Trickest CVE index for a PoC. Returns (date_str, url)."""
    m = re.match(r"CVE-(\d{4})-\d+", cve_id)
    if not m:
        return None, None
    year = m.group(1)
    trickest_url = _TRICKEST_URL.format(year=year, cve_id=cve_id)

    try:
        resp = requests.get(trickest_url, timeout=10)
        if resp.status_code == 404:
            return None, None
        resp.raise_for_status()
        text = resp.text

        repo_paths = re.findall(r"https://github\.com/([^/\s)>\"]+/[^/\s)>\"]+)", text)
        for repo_path in repo_paths:
            if "trickest" in repo_path.lower():
                continue
            try:
                api_url = f"https://api.github.com/repos/{repo_path}"
                cr = requests.get(api_url, headers=_github_headers(), timeout=10)
                if cr.ok:
                    repo_data = cr.json()
                    created_at = _iso_date(repo_data.get("created_at"))
                    if created_at:
                        return created_at, f"https://github.com/{repo_path}"
            except Exception:
                continue

        dates = sorted(set(re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", text)))
        if dates:
            return dates[0], trickest_url
        return None, trickest_url
    except Exception:
        return None, None


def _fetch_epss_spike(cve_id: str) -> tuple[str | None, float]:
    """Fetch EPSS history and return (spike_date, max_7d_jump)."""
    try:
        resp = requests.get(
            _EPSS_API_URL,
            params={"cve": cve_id, "scope": "time-series", "limit": 365},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        entries = data.get("data", [])
        if not entries:
            return None, 0.0

        entry = entries[0]
        series = list(entry.get("time-series", []))
        current = {"date": entry.get("date", ""), "epss": entry.get("epss", "0")}
        if current["date"] and not any(p["date"] == current["date"] for p in series):
            series = [current] + series

        points = sorted(
            [{"date": p["date"], "epss": float(p["epss"])} for p in series],
            key=lambda x: x["date"],
        )
        scores = [p["epss"] for p in points]

        max_jump = 0.0
        spike_date: str | None = None
        for i in range(len(scores) - 1):
            for j in range(i + 1, min(i + 8, len(scores))):
                delta = scores[j] - scores[i]
                if delta > max_jump:
                    max_jump = delta
                    spike_date = points[j]["date"]

        if max_jump >= _EPSS_SPIKE_THRESHOLD:
            return spike_date, round(max_jump, 6)
        return None, round(max_jump, 6)
    except Exception:
        return None, 0.0


def _fetch_cisa_kev(cve_id: str) -> tuple[str | None, str | None]:
    """Scan CISA KEV catalogue. Returns (date_added, required_action)."""
    try:
        resp = requests.get(_CISA_KEV_URL, timeout=20)
        resp.raise_for_status()
        for vuln in resp.json().get("vulnerabilities", []):
            if vuln.get("cveID", "").upper() == cve_id:
                return _iso_date(vuln.get("dateAdded")), vuln.get("requiredAction")
        return None, None
    except Exception:
        return None, None


def build_cve_timeline(cve_id: str) -> dict[str, Any]:
    """Collect dates from all sources and return a structured timeline dict."""
    cve_id = cve_id.upper()

    nvd = _fetch_nvd_info(cve_id)
    disclosure_date = nvd.get("published")
    nvd_url = nvd.get("nvd_url", f"https://nvd.nist.gov/vuln/detail/{cve_id}")

    patch_date, patch_url = _fetch_ghsa_patch_date(cve_id)
    if not patch_date:
        patch_date, patch_url = _fetch_nvd_patch_date(cve_id)

    poc_date, poc_url = _fetch_trickest_poc(cve_id)
    epss_spike_date, max_7d_jump = _fetch_epss_spike(cve_id)
    kev_date, _kev_action = _fetch_cisa_kev(cve_id)
    if not kev_date:
        kev_date = nvd.get("kev_date")

    events: list[dict[str, Any]] = []

    if disclosure_date:
        events.append(
            {
                "event": "CVE Disclosed (NVD Published)",
                "date": disclosure_date,
                "source": "NVD",
                "url": nvd_url,
            }
        )

    if patch_date:
        events.append(
            {
                "event": "Patch Committed",
                "date": patch_date,
                "source": "GHSA / GitHub",
                "url": patch_url,
            }
        )

    if poc_date:
        events.append(
            {
                "event": "First Public PoC",
                "date": poc_date,
                "source": "Trickest CVE Index / GitHub",
                "url": poc_url,
            }
        )

    if epss_spike_date:
        events.append(
            {
                "event": f"EPSS Exploitation Spike (delta {max_7d_jump:.3f} in 7 days)",
                "date": epss_spike_date,
                "source": "FIRST.org EPSS",
                "url": "https://www.first.org/epss/",
            }
        )

    if kev_date:
        events.append(
            {
                "event": "Added to CISA KEV (Confirmed Exploitation in Wild)",
                "date": kev_date,
                "source": "CISA KEV",
                "url": "https://www.cisa.gov/known-exploited-vulnerabilities-catalog",
            }
        )

    events.sort(key=lambda e: (e["date"] is None, e["date"] or ""))

    velocity: dict[str, Any] = {}

    d2patch = _days_between(disclosure_date, patch_date)
    if d2patch is not None:
        velocity["disclosure_to_patch_days"] = d2patch

    d2poc = _days_between(disclosure_date, poc_date)
    if d2poc is not None:
        velocity["disclosure_to_poc_days"] = d2poc
        velocity["fast_weaponisation"] = d2poc <= 7

    poc2kev = _days_between(poc_date, kev_date)
    if poc2kev is not None:
        velocity["poc_to_kev_days"] = poc2kev

    weaponized_date = kev_date or epss_spike_date
    weaponized_source = "CISA KEV" if kev_date else ("EPSS spike" if epss_spike_date else None)
    d2w = _days_between(disclosure_date, weaponized_date)
    if d2w is not None and weaponized_source:
        velocity["disclosure_to_weaponized_days"] = d2w
        velocity["weaponized_source"] = weaponized_source

    today_str = date.today().isoformat()
    dsd = _days_between(disclosure_date, today_str)
    if dsd is not None:
        velocity["days_since_disclosure"] = dsd

    return {
        "cve_id": cve_id,
        "events": events,
        "velocity": velocity,
        "current_date": today_str,
    }


def render_timeline_text(timeline: dict[str, Any]) -> str:
    """Render a timeline dict as a human-readable text report."""
    cve_id = timeline["cve_id"]
    events = timeline["events"]
    velocity = timeline["velocity"]
    sep = "=" * 56

    lines: list[str] = [f"CVE Timeline: {cve_id}", sep]

    if not events:
        lines.append("  No timeline events found.")
    else:
        for ev in events:
            d = ev["date"] or "unknown date"
            lines.append(f"  {d}  {ev['event']}")
            lines.append(f"               Source: {ev['source']}")
            if ev.get("url"):
                lines.append(f"               URL:    {ev['url']}")

    lines += ["", "Velocity Metrics", "-" * 40]

    if not velocity:
        lines.append("  Insufficient data to compute velocity.")
    else:
        if "days_since_disclosure" in velocity:
            lines.append(f"  Days since disclosure       : {velocity['days_since_disclosure']}")
        if "disclosure_to_patch_days" in velocity:
            d = velocity["disclosure_to_patch_days"]
            flag = "  (patch preceded disclosure?)" if d < 0 else ""
            lines.append(f"  Disclosure -> Patch          : {d} days{flag}")
        if "disclosure_to_poc_days" in velocity:
            d = velocity["disclosure_to_poc_days"]
            fast = "  [FAST <=7 days]" if velocity.get("fast_weaponisation") else ""
            lines.append(f"  Disclosure -> First PoC      : {d} days{fast}")
        if "poc_to_kev_days" in velocity:
            lines.append(f"  PoC -> CISA KEV listing      : {velocity['poc_to_kev_days']} days")
        if "disclosure_to_weaponized_days" in velocity:
            src = velocity.get("weaponized_source", "")
            lines.append(
                f"  Disclosure -> Weaponized     : {velocity['disclosure_to_weaponized_days']} days  (via {src})"
            )

    return "\n".join(lines)


def get_cve_timeline(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Build a structured CVE lifecycle timeline from multiple intelligence sources."""
    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")

    if not isinstance(cve_id, str) or not cve_id.upper().startswith("CVE-"):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID format. Must be a string like CVE-YYYY-NNNN."}],
        }
        log_tool_output_size("get_cve_timeline", result)
        return result

    timeline = build_cve_timeline(cve_id)
    text_report = render_timeline_text(timeline)

    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [
            {"text": text_report},
            {"json": timeline},
        ],
    }
    log_tool_output_size("get_cve_timeline", result)
    return result
