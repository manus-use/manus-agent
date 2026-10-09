"""
Tool: get_poc_freshness

Measures how recently PoC (Proof-of-Concept) activity occurred for a CVE.

Queries multiple public sources to build a freshness profile:

  1. **GitHub Search** — finds repos mentioning the CVE, checks last-push
     dates, star counts, and fork counts to gauge ongoing interest.
  2. **Exploit-DB** — checks if entries exist and their publication dates.
  3. **trickest/cve** — counts known PoC links in the curated index.

Produces a 0–100 *freshness score* where:

  - **80–100** → Active/hot — recent commits, stars still climbing
  - **50–79**  → Warm — activity within the last few months
  - **20–49**  → Cooling — PoCs exist but no recent updates
  - **0–19**   → Stale — old or no public PoCs found

CLI: ``manus-agent poc-freshness CVE-XXXX-YYYY``
"""

from __future__ import annotations

import logging
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

from strands.types.tools import ToolResult, ToolUse

from manus_agent.tools.tool_output_logger import log_tool_output_size

__all__ = ["get_poc_freshness", "TOOL_SPEC", "compute_freshness"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CVE_RE = re.compile(r"^CVE-\d{4}-\d+$", re.IGNORECASE)
_CVE_YEAR_RE = re.compile(r"CVE-(\d{4})-\d+", re.IGNORECASE)
_GITHUB_API = "https://api.github.com"
_TRICKEST_RAW = "https://raw.githubusercontent.com/trickest/cve/main/{year}/{cve_id}.md"
_URL_RE = re.compile(r"https?://\S+")
_REQUEST_TIMEOUT = 20  # seconds

# Freshness decay: days since last activity → score contribution
# After 365 days of inactivity the recency component approaches 0.
_DECAY_HALF_LIFE_DAYS = 30  # half-life for exponential decay

# Weight allocation (must sum to 1.0)
_W_RECENCY = 0.40  # how recent is the latest activity
_W_VOLUME = 0.25  # how many PoC repos/entries exist
_W_STARS = 0.20  # community interest (stars + forks)
_W_EXPLOITDB = 0.15  # presence on Exploit-DB (curated = higher signal)

# ---------------------------------------------------------------------------
# TOOL_SPEC (Strands module-based pattern)
# ---------------------------------------------------------------------------

TOOL_SPEC = {
    "name": "get_poc_freshness",
    "description": (
        "Measures how recently PoC (Proof-of-Concept) activity occurred for a CVE. "
        "Checks GitHub repos (last-push recency, stars, forks), Exploit-DB entries, "
        "and trickest/cve PoC links. Returns a 0-100 freshness_score: "
        "80-100 = active/hot, 50-79 = warm, 20-49 = cooling, 0-19 = stale. "
        "A high score means attacker interest is ongoing and the CVE warrants "
        "immediate attention."
    ),
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "cve_id": {
                    "type": "string",
                    "description": "The CVE identifier (e.g. 'CVE-2024-3094').",
                },
            },
            "required": ["cve_id"],
        }
    },
}

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    """Minimal HTTP GET returning parsed JSON."""
    import json

    hdrs = {"User-Agent": "manus-agent/poc-freshness"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str) -> str:
    """Minimal HTTP GET returning response body as text."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "manus-agent/poc-freshness"},
    )
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
        return resp.read().decode("utf-8")


# ---------------------------------------------------------------------------
# Source: GitHub search — repos mentioning the CVE
# ---------------------------------------------------------------------------


def _fetch_github_repos(cve_id: str) -> dict[str, Any]:
    """Search GitHub for repositories mentioning the CVE.

    Returns a dict with:
      - repos: list of {name, url, pushed_at, stars, forks}
      - total_count: number of matching repos
      - newest_push: ISO timestamp of the most recent push
      - error: str or None
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"token {token}"

    query = urllib.parse.quote(cve_id.upper())
    url = f"{_GITHUB_API}/search/repositories?q={query}&sort=updated&order=desc&per_page=30"

    try:
        data = _get_json(url, headers)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GitHub search failed for %s: %s", cve_id, exc)
        return {"repos": [], "total_count": 0, "newest_push": None, "error": str(exc)}

    items = data.get("items", [])
    total = data.get("total_count", len(items))

    repos = []
    newest_push: str | None = None
    for item in items:
        pushed = item.get("pushed_at", "")
        repos.append(
            {
                "name": item.get("full_name", ""),
                "url": item.get("html_url", ""),
                "pushed_at": pushed,
                "stars": item.get("stargazers_count", 0),
                "forks": item.get("forks_count", 0),
            }
        )
        if pushed and (newest_push is None or pushed > newest_push):
            newest_push = pushed

    return {
        "repos": repos,
        "total_count": total,
        "newest_push": newest_push,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Source: Exploit-DB — check for entries via GitLab CSV cache
# ---------------------------------------------------------------------------


def _fetch_exploitdb_entries(cve_id: str) -> dict[str, Any]:
    """Search the Exploit-DB GitLab CSV for entries matching the CVE.

    Returns a dict with:
      - entries: list of {id, description, date_published, url}
      - count: number of matches
      - newest_date: ISO date of most recent entry
      - error: str or None
    """
    import csv
    import io
    import time

    cache_path = "/tmp/exploitdb_cache.csv"
    cache_ttl = 86_400  # 24h

    csv_url = "https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv"

    # Use cached CSV if fresh enough
    csv_text: str | None = None
    try:
        stat = os.stat(cache_path)
        if time.time() - stat.st_mtime < cache_ttl:
            with open(cache_path, encoding="utf-8", errors="replace") as fh:
                csv_text = fh.read()
    except OSError:
        pass

    if csv_text is None:
        try:
            csv_text = _get_text(csv_url)
            with open(cache_path, "w", encoding="utf-8") as fh:
                fh.write(csv_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Exploit-DB CSV fetch failed: %s", exc)
            return {"entries": [], "count": 0, "newest_date": None, "error": str(exc)}

    # Parse CSV — columns: id, file, description, date_published, ...
    # The CVE ID typically appears in the 'codes' column (index 11 in the CSV).
    entries: list[dict[str, Any]] = []
    cve_upper = cve_id.upper()
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader, None)
    if not header:
        return {"entries": [], "count": 0, "newest_date": None, "error": "Empty CSV"}

    # Build column index map
    col_map = {col.strip().lower(): i for i, col in enumerate(header)}
    id_col = col_map.get("id", 0)
    desc_col = col_map.get("description", 2)
    date_col = col_map.get("date_published", 3)
    codes_col = col_map.get("codes", None)

    for row in reader:
        # Check if CVE appears in the codes column or the description
        match = False
        if codes_col is not None and codes_col < len(row):
            if cve_upper in row[codes_col].upper():
                match = True
        if not match and desc_col < len(row):
            if cve_upper in row[desc_col].upper():
                match = True
        if not match:
            # Also search the entire row as a fallback
            row_text = ";".join(row).upper()
            if cve_upper in row_text:
                match = True

        if match:
            eid = row[id_col] if id_col < len(row) else ""
            date_pub = row[date_col] if date_col < len(row) else ""
            desc = row[desc_col] if desc_col < len(row) else ""
            entries.append(
                {
                    "id": eid,
                    "description": desc[:200],
                    "date_published": date_pub,
                    "url": f"https://www.exploit-db.com/exploits/{eid}" if eid else "",
                }
            )

    newest_date: str | None = None
    for e in entries:
        dp = e.get("date_published", "")
        if dp and (newest_date is None or dp > newest_date):
            newest_date = dp

    return {
        "entries": entries,
        "count": len(entries),
        "newest_date": newest_date,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Source: trickest/cve — curated PoC link index
# ---------------------------------------------------------------------------


def _fetch_trickest_pocs(cve_id: str) -> dict[str, Any]:
    """Fetch PoC links from the trickest/cve GitHub index.

    Returns a dict with:
      - poc_count: number of PoC links
      - github_pocs: list of GitHub PoC URLs
      - reference_pocs: list of reference PoC URLs
      - error: str or None
    """
    cve_upper = cve_id.upper()
    year_match = _CVE_YEAR_RE.match(cve_upper)
    if not year_match:
        return {
            "poc_count": 0,
            "github_pocs": [],
            "reference_pocs": [],
            "error": "Invalid CVE ID",
        }

    year = year_match.group(1)
    url = _TRICKEST_RAW.format(year=year, cve_id=cve_upper)

    try:
        markdown = _get_text(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {
                "poc_count": 0,
                "github_pocs": [],
                "reference_pocs": [],
                "error": None,
            }
        return {
            "poc_count": 0,
            "github_pocs": [],
            "reference_pocs": [],
            "error": f"HTTP {exc.code}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "poc_count": 0,
            "github_pocs": [],
            "reference_pocs": [],
            "error": str(exc),
        }

    # Parse sections
    github_pocs: list[str] = []
    reference_pocs: list[str] = []
    section = None

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#### Github"):
            section = "github"
            continue
        elif stripped.startswith("#### Reference"):
            section = "reference"
            continue
        elif stripped.startswith("###"):
            section = None
            continue

        urls = _URL_RE.findall(stripped)
        for u in urls:
            u = u.rstrip(")")
            if section == "github":
                github_pocs.append(u)
            elif section == "reference":
                reference_pocs.append(u)

    all_urls = list(dict.fromkeys(github_pocs + reference_pocs))
    return {
        "poc_count": len(all_urls),
        "github_pocs": github_pocs,
        "reference_pocs": reference_pocs,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Freshness scoring engine
# ---------------------------------------------------------------------------


def _days_since(iso_str: str | None, now: datetime | None = None) -> float | None:
    """Return the number of days between *iso_str* and *now*.

    Returns ``None`` when *iso_str* is falsy or unparseable.
    """
    if not iso_str:
        return None
    now = now or datetime.now(timezone.utc)
    try:
        # Handle both date-only and datetime formats
        if "T" in iso_str:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(iso_str + "T00:00:00+00:00")
        delta = (now - dt).total_seconds() / 86_400
        return max(delta, 0.0)
    except (ValueError, TypeError):
        return None


def _decay(days: float | None) -> float:
    """Exponential decay: 1.0 at day 0, 0.5 at half-life, → 0."""
    if days is None:
        return 0.0
    return math.exp(-0.693 * days / _DECAY_HALF_LIFE_DAYS)


def _volume_score(count: int) -> float:
    """Logarithmic volume score: 0 repos → 0, 1 → 0.5, 5 → 0.82, 30 → 1.0."""
    if count <= 0:
        return 0.0
    return min(math.log1p(count) / math.log1p(30), 1.0)


def _star_score(total_stars: int, total_forks: int) -> float:
    """Combined interest score from stars + forks (forks weighted 2x)."""
    combined = total_stars + 2 * total_forks
    if combined <= 0:
        return 0.0
    # Log scale: 1 → 0.14, 10 → 0.48, 100 → 0.82, 1000 → 1.0
    return min(math.log1p(combined) / math.log1p(1000), 1.0)


def compute_freshness(
    github_data: dict[str, Any],
    exploitdb_data: dict[str, Any],
    trickest_data: dict[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Compute the 0-100 freshness score from source data.

    This function is the pure scoring engine — no I/O. Exposed for
    direct use by the CLI and for testing.

    Returns a dict with:
      - freshness_score: int (0-100)
      - label: str (hot/warm/cooling/stale)
      - components: dict of sub-scores
      - newest_activity: ISO str or None
      - days_since_activity: float or None
      - summary: human-readable summary string
    """
    now = now or datetime.now(timezone.utc)

    # --- Recency component (most recent activity across all sources) ---
    newest_dates: list[str] = []
    gh_push = github_data.get("newest_push")
    if gh_push:
        newest_dates.append(gh_push)
    edb_date = exploitdb_data.get("newest_date")
    if edb_date:
        newest_dates.append(edb_date)

    newest_activity = max(newest_dates) if newest_dates else None
    days = _days_since(newest_activity, now)
    recency_raw = _decay(days)

    # --- Volume component ---
    gh_count = github_data.get("total_count", 0)
    trickest_count = trickest_data.get("poc_count", 0)
    edb_count = exploitdb_data.get("count", 0)
    total_pocs = gh_count + trickest_count + edb_count
    volume_raw = _volume_score(total_pocs)

    # --- Stars/interest component ---
    total_stars = sum(r.get("stars", 0) for r in github_data.get("repos", []))
    total_forks = sum(r.get("forks", 0) for r in github_data.get("repos", []))
    stars_raw = _star_score(total_stars, total_forks)

    # --- Exploit-DB component (curated = higher signal) ---
    edb_raw = min(edb_count / 3.0, 1.0) if edb_count > 0 else 0.0

    # --- Weighted score ---
    weighted = _W_RECENCY * recency_raw + _W_VOLUME * volume_raw + _W_STARS * stars_raw + _W_EXPLOITDB * edb_raw
    score = int(round(weighted * 100))
    score = max(0, min(100, score))

    # --- Label ---
    if score >= 80:
        label = "hot"
    elif score >= 50:
        label = "warm"
    elif score >= 20:
        label = "cooling"
    else:
        label = "stale"

    # --- Build summary ---
    parts = []
    if days is not None:
        if days < 1:
            parts.append("activity within the last 24 hours")
        elif days < 7:
            parts.append(f"activity {int(days)} day(s) ago")
        elif days < 30:
            parts.append(f"activity {int(days)} days ago (~{int(days / 7)} weeks)")
        elif days < 365:
            parts.append(f"activity {int(days)} days ago (~{int(days / 30)} months)")
        else:
            parts.append(f"last activity {int(days)} days ago (>{int(days / 365)} year(s))")
    else:
        parts.append("no datable PoC activity found")

    if total_pocs > 0:
        parts.append(
            f"{total_pocs} PoC source(s) across {_count_active_sources(github_data, exploitdb_data, trickest_data)} source(s)"
        )
    if total_stars > 0:
        parts.append(f"{total_stars} star(s), {total_forks} fork(s) on GitHub")
    if edb_count > 0:
        parts.append(f"{edb_count} Exploit-DB entry/entries")

    summary = "; ".join(parts) if parts else "No PoC data found."

    return {
        "freshness_score": score,
        "label": label,
        "components": {
            "recency": round(recency_raw, 4),
            "volume": round(volume_raw, 4),
            "stars": round(stars_raw, 4),
            "exploitdb": round(edb_raw, 4),
        },
        "newest_activity": newest_activity,
        "days_since_activity": round(days, 1) if days is not None else None,
        "total_pocs": total_pocs,
        "github_repos": github_data.get("total_count", 0),
        "github_stars": total_stars,
        "github_forks": total_forks,
        "exploitdb_entries": edb_count,
        "trickest_pocs": trickest_count,
        "sources_checked": _list_sources_checked(github_data, exploitdb_data, trickest_data),
        "sources_failed": _list_sources_failed(github_data, exploitdb_data, trickest_data),
        "summary": summary,
    }


def _count_active_sources(
    github_data: dict[str, Any],
    exploitdb_data: dict[str, Any],
    trickest_data: dict[str, Any],
) -> int:
    count = 0
    if github_data.get("total_count", 0) > 0:
        count += 1
    if exploitdb_data.get("count", 0) > 0:
        count += 1
    if trickest_data.get("poc_count", 0) > 0:
        count += 1
    return count


def _list_sources_checked(
    github_data: dict[str, Any],
    exploitdb_data: dict[str, Any],
    trickest_data: dict[str, Any],
) -> list[str]:
    sources = []
    if github_data.get("error") is None:
        sources.append("github")
    if exploitdb_data.get("error") is None:
        sources.append("exploitdb")
    if trickest_data.get("error") is None:
        sources.append("trickest")
    return sources


def _list_sources_failed(
    github_data: dict[str, Any],
    exploitdb_data: dict[str, Any],
    trickest_data: dict[str, Any],
) -> list[str]:
    sources = []
    if github_data.get("error") is not None:
        sources.append("github")
    if exploitdb_data.get("error") is not None:
        sources.append("exploitdb")
    if trickest_data.get("error") is not None:
        sources.append("trickest")
    return sources


# ---------------------------------------------------------------------------
# Public data-fetching entry point (used by CLI)
# ---------------------------------------------------------------------------


def fetch_poc_freshness(cve_id: str) -> dict[str, Any]:
    """Fetch all source data and compute freshness for a CVE.

    This is the main entry point for the CLI. It performs real HTTP calls.
    For testing, call ``compute_freshness`` directly with mocked data.
    """
    cve_upper = cve_id.strip().upper()
    github_data = _fetch_github_repos(cve_upper)
    exploitdb_data = _fetch_exploitdb_entries(cve_upper)
    trickest_data = _fetch_trickest_pocs(cve_upper)

    result = compute_freshness(github_data, exploitdb_data, trickest_data)
    result["cve_id"] = cve_upper
    return result


# ---------------------------------------------------------------------------
# Strands tool handler
# ---------------------------------------------------------------------------


def get_poc_freshness(tool: ToolUse, **kwargs: Any) -> ToolResult:
    """Strands SDK tool handler for poc-freshness scoring."""
    import json

    tool_use_id = tool["toolUseId"]
    tool_input = tool["input"]
    cve_id = tool_input.get("cve_id", "")

    if not isinstance(cve_id, str) or not _CVE_RE.match(cve_id.strip()):
        result: ToolResult = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Invalid CVE ID format. Must be a string like 'CVE-YYYY-NNNN'."}],
        }
        log_tool_output_size("get_poc_freshness", result)
        return result

    try:
        data = fetch_poc_freshness(cve_id)
    except Exception as exc:  # noqa: BLE001
        result = {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"PoC freshness check failed: {exc}"}],
        }
        log_tool_output_size("get_poc_freshness", result)
        return result

    text = json.dumps(data, indent=2)
    result = {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": text}],
    }
    log_tool_output_size("get_poc_freshness", result)
    return result
